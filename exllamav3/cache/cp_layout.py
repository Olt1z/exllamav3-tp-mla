"""
Context parallel: como a sequência se reparte entre os ranks.

A regra é uma linha: **o token de posição global `p` pertence ao rank `p % world`, e mora na
posição local `p // world`.** Tudo o mais decorre disso.

Por que intercalar por TOKEN e não por página:

- **Equilíbrio exato.** Por página, os 256 tokens de uma página caem todos no mesmo rank, e durante
  a geração um único rank recebe todos os tokens novos por 256 passos seguidos. Por token, a
  diferença entre o rank mais e o menos carregado nunca passa de 1.
- **A página física continua existindo em TODOS os ranks**, cada um guardando a sua fração dela.
  É o que salva o `copy_page`: uma cópia de página é a mesma operação em todo rank, cada um
  copiando as próprias linhas. Repartir por página faria a cópia atravessar placas.
- **O `block_table` fica idêntico entre os ranks**, porque o índice de página vira
  `p // (world · PAGE_SIZE)`, que não depende do rank. O despacho do TP segue mandando um pickle
  único para todos (`model_tp.py`), que era a otimização que a primeira versão do plano teria
  desfeito ao recortar o `block_table` por rank.

E o que torna isto barato de verdade: **do ponto de vista do kernel de atenção, o rank enxerga uma
sequência normal, só que mais curta**. Ele não precisa saber que faltam tokens no meio — a chave
rope já foi gravada rotacionada com a posição original, então o produto q·k continua carregando a
posição relativa certa. Os kernels de atenção não mudam.

**Onde isto NÃO vale, e é preciso saber antes de tropeçar:**

- **Prefill.** A máscara causal precisa da posição global: o token local `j` do rank `r` está na
  posição global `j·world + r`, e uma consulta na posição `p` só pode ver `j ≤ (p − r)/world`. É
  por isso que o prefill tem mecanismo próprio no plano (all-gather do KV latente com sharding
  zigzag), e não é este.
- **O plano agrupado do indexador da DSA (`k_pool`).** Uma entrada agrupa `kpool` tokens globais
  CONSECUTIVOS, e sob intercalamento por token nenhum rank tem esses tokens juntos. O agrupamento
  deixa de ser calculável localmente. Obstáculo real da etapa de top-k distribuído; não há conserto
  neste arquivo.
"""
from __future__ import annotations

import torch


def comprimento_local(total: int, world: int, rank: int) -> int:
    """Quantos dos `total` tokens globais [0, total) pertencem a este rank."""
    return total // world + (1 if rank < total % world else 0)


def comprimentos_locais(cache_seqlens: torch.Tensor, world: int, rank: int) -> torch.Tensor:
    """A versão por lote, no device: um comprimento local por linha.

    Sem sincronizar com o host — `cache_seqlens` já vive na GPU e o resultado vai direto para o
    kernel de atenção no lugar do comprimento global."""
    if world == 1:
        return cache_seqlens
    return cache_seqlens // world + (rank < cache_seqlens % world).to(cache_seqlens.dtype)


def tokens_por_pagina(page_size: int, world: int) -> int:
    """Quantos tokens GLOBAIS uma página cobre. É o passo do hash do cache de prefixo: sob CP uma
    página física guarda `page_size` tokens locais, que são `page_size · world` globais."""
    return page_size * world


def paginas_para(max_num_tokens: int, page_size: int, world: int) -> int:
    """Quantas páginas físicas cada rank aloca para uma capacidade global de `max_num_tokens`.

    É aqui que o cache encolhe: a capacidade lógica não muda e a alocação por placa cai `world`
    vezes."""
    assert max_num_tokens % (page_size * world) == 0, (
        f"max_num_tokens ({max_num_tokens}) tem de ser multiplo de page_size x world "
        f"({page_size} x {world}); senao a ultima pagina virtual fica incompleta e os ranks "
        f"discordam de quantas paginas existem"
    )
    return max_num_tokens // (page_size * world)


if __name__ == "__main__":
    # A partição tem de ser exata: todo token global em exatamente um rank, sem buraco e sem
    # colisão, e as posições locais de cada rank contíguas a partir de zero. É o que quebra se a
    # aritmética estiver errada, e não precisa de GPU para verificar.
    for world in (1, 2, 3, 4, 8):
        for total in range(0, 200):
            vistos = {}
            for r in range(world):
                locais = [p for p in range(total) if p % world == r]
                # comprimento_local bate com a contagem
                n = comprimento_local(total, world, r)
                assert n == len(locais), \
                    f"world {world} rank {r} total {total}: {n} != {len(locais)}"
                # posicao local e contigua a partir de zero, na ordem global
                for esperado, p in enumerate(locais):
                    assert p // world == esperado, \
                        f"world {world} rank {r}: posicao global {p} -> local {p // world}, " \
                        f"esperado {esperado}"
                    assert p not in vistos, f"token {p} em dois ranks: {vistos[p]} e {r}"
                    vistos[p] = r
            assert len(vistos) == total, f"world {world} total {total}: {len(vistos)} tokens vistos"
    print("particao exata: sem buraco, sem colisao, posicoes locais contiguas · "
          "worlds 1,2,3,4,8 x totais 0..199")

    # A diferença de carga entre o rank mais e o menos ocupado nunca passa de 1
    for world in (2, 3, 4, 8):
        for total in range(0, 5000, 7):
            cargas = [comprimento_local(total, world, r) for r in range(world)]
            assert max(cargas) - min(cargas) <= 1, \
                f"world {world} total {total}: desequilibrio {max(cargas) - min(cargas)}"
    print("desequilibrio nunca passa de 1 token · worlds 2,3,4,8")

    # O indice de pagina virtual nao depende do rank: e o que permite mandar um block_table so
    for world in (2, 4):
        page_size = 256
        for p in range(0, 100000, 13):
            paginas = {(p // world) // page_size for r in range(world) if p % world == r}
            # a pagina local do dono tem de ser a pagina virtual global
            assert paginas == {p // (page_size * world)}, \
                f"world {world} pos {p}: pagina local {paginas} != virtual {p // (page_size * world)}"
    print("indice de pagina identico entre ranks · o block_table nao precisa ser recortado")

    # paginas_para encolhe por world e recusa capacidade que nao fecha
    assert paginas_para(1024 * 1024, 256, 4) == 1024
    assert paginas_para(1024 * 1024, 256, 1) == 4096
    try:
        paginas_para(1000, 256, 4)
        raise AssertionError("deveria ter recusado capacidade nao multipla")
    except AssertionError as e:
        assert "multiplo" in str(e), e
    print("paginas_para encolhe por world e recusa capacidade que nao fecha")
