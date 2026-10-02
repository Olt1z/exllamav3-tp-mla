"""
Indexador DSA dividido entre os ranks do tensor parallel, no prefill.

Sem isto, cada rank do TP recalcula o indexador INTEIRO: pontua cada linha do chunk contra todos os
pools visíveis do contexto e faz o top-k, embora a seleção seja a mesma para todas as cabeças e
portanto para todos os ranks. A 900k tokens são ~225k pools, ~7,5 TFLOP por chunk de 4096 linhas em
cada camada "full" -- o custo que mais cresce com o contexto, e hoje tp vezes redundante.

Com `EXL3_INDEXADOR_DIVIDIDO=1`, no prefill cada rank pontua e seleciona só a sua faixa de linhas do
chunk, e um all-gather de int32 junta as faixas: todo rank sai com a MESMA matriz de índices, byte a
byte. A atenção esparsa, as camadas "shared" e o resto do caminho não mudam.

Regras que este arquivo tranca, e por quê:

  - **Faixas em múltiplos de 256 linhas** (o `slab` de `_indexer_topk*`). A seleção é por linha, mas
    no caminho em tiles o `k_sel` de cada slab sai do FIM do slab (`min(topk, t_slab)`); cortar um
    slab ao meio mudaria o `k_sel` das linhas dele no começo do regime esparso. Com as fronteiras nos
    mesmos lugares, o rank faz exatamente os slabs que o caminho replicado faria para aquelas linhas.
  - **Faixas contíguas, as primeiras com um slab a mais** quando o número de slabs não divide pelo
    mundo. O último slab pode ser parcial (R não múltiplo de 256), e fica no rank que já tem menos.
    Rank sem slab nenhum recebe faixa vazia e ainda assim entra no all-gather -- coletivo é de todos.
  - **A decisão de dividir só depende do que é igual em todo rank** (flag, mundo, CP, tamanho do
    chunk, regime esparso). Se um rank dividisse e outro não, um esperaria num all-gather que o outro
    nunca chama.

Determinismo: no caminho replicado cada rank escolhia os seus próprios índices, e em quase-empate no
k-ésimo score (fp16) ranks diferentes podiam escolher tokens diferentes -- as cabeças de um rank
atendiam um conjunto e as de outro, outro. Dividido, cada linha é selecionada por UM rank e copiada
para os demais, então o conjunto é único por linha. Não fica determinístico entre execuções (os
kernels de pontuação são os mesmos), só consistente entre ranks.

As funções puras (sem torch) ficam no topo para os testes rodarem sem placa.
"""
from __future__ import annotations

# Linhas por slab em _indexer_topk / _indexer_topk_kpool. Se mudar lá, muda aqui.
SLAB = 256


def faixas_por_rank(linhas: int, mundo: int, slab: int = SLAB) -> list[tuple[int, int]]:
    """Partição contígua de [0, linhas) em `mundo` faixas [a, b).

    Fronteiras internas em múltiplos de `slab`; a última faixa não vazia termina em `linhas`. Os
    `n_slabs % mundo` primeiros ranks levam um slab a mais. Faixas vazias saem como (x, x)."""
    assert mundo >= 1 and linhas >= 0 and slab >= 1
    n_slabs = -(-linhas // slab)
    base, resto = divmod(n_slabs, mundo)
    faixas = []
    s0 = 0
    for r in range(mundo):
        n = base + (1 if r < resto else 0)
        a = min(s0 * slab, linhas)
        b = min((s0 + n) * slab, linhas)
        faixas.append((a, b))
        s0 += n
    return faixas


def linhas_max(faixas: list[tuple[int, int]]) -> int:
    """Altura de cada fatia do all-gather: a maior faixa (as menores vão completadas com -1)."""
    return max((b - a for a, b in faixas), default = 0)


def deve_dividir(
    ligado: bool,
    mundo: int,
    coletivo_ok: bool,
    cp_world: int,
    seqlen: int,
    min_linhas: int,
    com_cache: bool,
    aquecimento: bool,
) -> bool:
    """Divide o indexador deste chunk entre os ranks?

    Só no prefill de chunk grande (`seqlen >= min_linhas`; o decode tem poucas linhas e o all-gather
    custaria mais que a pontuação), com TP de verdade, sem context parallel (lá o plano do indexador
    e o grupo do coletivo são outros -- fica o caminho de sempre), no caminho com cache (o sem cache é
    o da calibração, sem TP) e fora do aquecimento (o TPBackendNull não faz coletivo, e índice de lixo
    no gather da atenção é acesso fora da memória). O chamador já sabe que a camada é "full" e que o
    regime é esparso."""
    return (
        ligado
        and mundo > 1
        and coletivo_ok
        and cp_world == 1
        and com_cache
        and not aquecimento
        and seqlen >= max(min_linhas, 1)
    )


# ---------------------------------------------------------------------------------------------------
# Lado torch: empacotar a faixa local, e montar a matriz inteira a partir do all-gather.

def empacotar(indices, bsz: int, seqlen: int, faixa: tuple[int, int], altura: int):
    """A faixa local de `indices` (bsz * seqlen, k_pad) como (bsz, altura, k_pad) contígua, com as
    linhas que sobram em -1 (nenhum token) -- é isso que um rank de faixa curta ou vazia manda."""
    import torch
    a, b = faixa
    k_pad = indices.shape[-1]
    v = indices.view(bsz, seqlen, k_pad)
    if b - a == altura:
        return v[:, a:b].contiguous()
    out = torch.full((bsz, altura, k_pad), -1, dtype = indices.dtype, device = indices.device)
    if b > a:
        out[:, : b - a] = v[:, a:b]
    return out


def desempacotar(reunido, faixas: list[tuple[int, int]], bsz: int, seqlen: int):
    """De (mundo, bsz, altura, k_pad), saído do all-gather, para (bsz * seqlen, k_pad).

    Toda linha, inclusive as do próprio rank, sai da cópia reunida: é o que garante que todo rank
    termina com exatamente os mesmos bytes."""
    import torch
    mundo, _, altura, k_pad = reunido.shape
    assert mundo == len(faixas)
    if bsz == 1 and all(a == r * altura for r, (a, b) in enumerate(faixas) if b > a):
        # Faixas encostadas na grade da altura (o caso comum: 4096 linhas em 4 ranks): é uma view
        return reunido.view(mundo * altura, k_pad)[:seqlen]
    out = torch.empty((bsz, seqlen, k_pad), dtype = reunido.dtype, device = reunido.device)
    for r, (a, b) in enumerate(faixas):
        if b > a:
            out[:, a:b] = reunido[r, :, : b - a]
    return out.view(bsz * seqlen, k_pad)
