"""
Context parallel: a peça compartilhada.

Quando o cache não reparte por cabeça (`num_kv_heads_local < 1` — MLA, QSA, planos de indexador
da DSA, e GQA com menos cabeças de KV que ranks), o TP replica o cache inteiro em cada rank. O CP
reparte a SEQUÊNCIA: cada rank atende todas as `H_g` cabeças sobre a sua fatia de tokens, e as
saídas parciais se combinam exatamente por log-sum-exp.

**Isto não vive dentro de uma família de atenção.** Recebe a saída local já normalizada e o `lse`
local de qualquer caminho, e devolve a saída global. Serve MLA densa, DSA esparsa e a atenção
paginada genérica sem saber qual é qual.

A identidade que torna isto barato, e que dispensa mexer nos kernels de combine que já existem:

    acc_r = o_local_r · l_r          (o combine local já dividiu por l_r)
    lse_r = m_r + log(l_r)

    o = Σ_r exp(m_r − m_g)·acc_r / Σ_r exp(m_r − m_g)·l_r
      = Σ_r exp(lse_r − lse_g)·o_local_r          com  lse_g = logsumexp_r(lse_r)

Os pesos somam 1. Ou seja: com a saída local normalizada e o `lse` local em mãos, o combine entre
ranks é uma soma ponderada — e uma soma é exatamente o que um reduce-scatter faz. O reduce-scatter
entrega de quebra o head-scatter que o `o_proj` queria, então combine e scatter são a MESMA
operação.

Os quatro kernels de combine do fork (`_dsa_attn_combine_kernel`, `_mla_decode_combine_kernel`,
`_paged_attn_decode_combine_kernel`, `_paged_attn_prefill_combine_kernel`) ficam intocados: eles
já produzem `o_local`, e o `lse_r` sai de `ws_ml`, que eles leem mas não consomem inteiro.

Medido em 08/09/2026 (prova 19): o par all_gather(lse) + reduce_scatter custa, POR CAMADA COM
CACHE, menos que um all-reduce dos que ja existem -- 71 us contra 81 us de um all_reduce de 8 KB,
porque all_reduce e internamente reduce-scatter + all-gather e paga duas fases onde este par paga
uma. As outras duas formas (all_reduce de slot, all_gather da parcial inteira) reprovaram o portao
com rascunho de decode. O custo por token e esse valor vezes o numero de camadas com cache DO
MODELO, que se le da contagem de modulos -- nao e constante (no caso de prova sao 11 de 45, e da
0,78 ms em TP4 com dcp = 2).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize = ["n_splits", "n_rows"])
def _cp_lse_kernel(
    ws_ml,               # (n_pid * n_splits * BLOCK_H * 2) fp32, m e l intercalados
    out_lse,             # (n_pid * BLOCK_H) fp32
    n_splits,
    n_rows,              # n_pid * BLOCK_H, para mascarar a cauda
    BLOCK_H: tl.constexpr,
):
    """O log-sum-exp local, a partir das parciais que o kernel de split já deixou em ws_ml.

    Mesma aritmética do laço dos combines existentes, sem tocar no acumulador: só o par (m, l).
    É um kernel de leitura minúscula — `n_splits × BLOCK_H × 2` floats por programa."""
    pid = tl.program_id(0)
    hloc = tl.arange(0, BLOCK_H)

    m_run = tl.full((BLOCK_H,), -float("inf"), tl.float32)
    l_run = tl.zeros((BLOCK_H,), tl.float32)

    for s in range(n_splits):
        base = ((pid * n_splits + s) * BLOCK_H + hloc)
        m_s = tl.load(ws_ml + base * 2)
        l_s = tl.load(ws_ml + base * 2 + 1)
        m_new = tl.maximum(m_run, m_s)
        m_exp = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.where(m_run == -float("inf"), 0.0, tl.exp(m_run - m_exp))
        beta = tl.where(m_s == -float("inf"), 0.0, tl.exp(m_s - m_exp))
        l_run = l_run * alpha + l_s * beta
        m_run = m_new

    # Fatia vazia (rank sem token nenhum, que acontece com sequência curta e N grande) tem de sair
    # -inf explicitamente: l = 0 daria log(0) = -inf pelo caminho certo, mas m = -inf com l != 0 por
    # lixo de buffer envenenaria o logsumexp global de todos os ranks
    vazio = (l_run <= 0.0) | (m_run == -float("inf"))
    lse = tl.where(vazio, -float("inf"), m_run + tl.log(tl.where(vazio, 1.0, l_run)))

    offs = pid * BLOCK_H + hloc
    tl.store(out_lse + offs, lse, mask = offs < n_rows)


@triton.jit(do_not_specialize = ["N", "R", "H", "rank", "stride_row", "stride_head"])
def _cp_correct_kernel(
    o_local,             # saída local já normalizada, indexada por row*stride_row + head*stride_head
    lse_all,             # (N, R * H) fp32, contígua: rank*(R*H) + row*H + head
    out,                 # (H, R, D) fp32 contígua — o layout que o reduce-scatter quer
    lse_out,             # (R * H) fp32, o lse global (para CP encadeado); ignorado se SALVAR_LSE = 0
    N, R, H, rank,
    stride_row, stride_head,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SALVAR_LSE: tl.constexpr,
):
    """Pesa a saída local por exp(lse_local − lse_global) e escreve head-major em fp32.

    Head-major porque o reduce-scatter fatia a dimensão externa: escrever já transposto aqui evita
    uma cópia do tensor inteiro por camada (13–14 µs medidos na prova 19 quando feita à parte).
    A soma sobre os ranks fica com o reduce-scatter, em fp32.

    Os pesos somam 1 por construção, então isto NÃO é uma média que precisa de normalização
    depois: o reduce-scatter entrega o resultado final."""
    idx = tl.program_id(0)
    dtile = tl.program_id(1)
    row = idx // H
    head = idx % H

    offs_d = dtile * BLOCK_D + tl.arange(0, BLOCK_D)
    valid_d = offs_d < D

    # logsumexp sobre os ranks. NaN/inf de buffer não inicializado viram -inf: um único rank com
    # lixo aqui contaminaria o máximo e zeraria o peso de todos os outros
    m_g = -float("inf")
    for r in range(N):
        x = tl.load(lse_all + r * (R * H) + idx)
        x = tl.where((x != x) | (x == float("inf")), -float("inf"), x)
        m_g = tl.maximum(m_g, x)

    m_safe = tl.where(m_g == -float("inf"), 0.0, m_g)
    soma = 0.0
    for r in range(N):
        x = tl.load(lse_all + r * (R * H) + idx)
        x = tl.where((x != x) | (x == float("inf")), -float("inf"), x)
        soma += tl.where(x == -float("inf"), 0.0, tl.exp(x - m_safe))

    lse_g = tl.where(soma <= 0.0, -float("inf"), m_safe + tl.log(tl.where(soma <= 0.0, 1.0, soma)))

    x_r = tl.load(lse_all + rank * (R * H) + idx)
    x_r = tl.where((x_r != x_r) | (x_r == float("inf")), -float("inf"), x_r)
    w = tl.where((x_r == -float("inf")) | (lse_g == -float("inf")), 0.0, tl.exp(x_r - lse_g))

    o = tl.load(o_local + row * stride_row + head * stride_head + offs_d,
                mask = valid_d, other = 0.0).to(tl.float32)

    tl.store(out + (head * R + row) * D + offs_d, o * w, mask = valid_d)

    if SALVAR_LSE:
        if dtile == 0:
            tl.store(lse_out + idx, lse_g)


# Indices inversos das duas reordenacoes, por geometria (ver reordenar_lse_mla_denso)
_ORIGEM: dict = {}


def cp_lse_local(ws_ml: torch.Tensor, n_pid: int, n_splits: int, block_h: int) -> torch.Tensor:
    """O log-sum-exp local por (linha, cabeça), lido das parciais do kernel de split."""
    n_rows = n_pid * block_h
    lse = torch.empty((n_rows,), dtype = torch.float32, device = ws_ml.device)
    with torch.cuda.device(ws_ml.device):
        _cp_lse_kernel[(n_pid,)](ws_ml, lse, n_splits, n_rows, BLOCK_H = block_h,
                                 num_warps = 1, num_stages = 2)
    return lse


def reordenar_lse_mla_denso(bruto: torch.Tensor, bsz: int, q_len: int, n_q_heads: int,
                            block_m: int, block_h: int, block_rows: int) -> torch.Tensor:
    """Do índice plano do workspace para a ordem (R, H), que é a que o combine entre ranks usa.

    **Esta é a única função do arquivo que conhece um caminho de atenção**, e é de propósito: os
    dois kernels acima são genéricos, e o que varia entre as famílias é só como o programa `pid` e
    a linha `hloc` se decompõem em (consulta, cabeça). Aqui está a decomposição do
    `_mla_decode_combine_kernel`:

        h_blocks = cdiv(n_q_heads, BLOCK_H);  h_block = pid % h_blocks;  batch = pid // h_blocks
        row_q = rows % BLOCK_M;  row_h = h_block * BLOCK_H + rows // BLOCK_M

    As linhas com `row_q >= q_len` ou `row_h >= n_q_heads` são preenchimento do bloco e não
    correspondem a nada — ficam de fora. Cada novo caminho que ganhar CP escreve a sua função
    irmã; nenhum deles mexe nos kernels.
    """
    dev = bruto.device
    R = bsz * q_len
    chave = ("denso", str(dev), bsz, q_len, n_q_heads, block_m, block_h, block_rows)
    origem = _ORIGEM.get(chave)
    if origem is None:
        # A permutacao inversa, calculada UMA vez por geometria. Indexar por mascara booleana
        # a cada chamada (`bruto[valido]`) forca um nonzero, que sincroniza com o host: na
        # prova 26 isso custava ~5 ms por token numa camada MLA so, mais que os coletivos.
        h_blocks = -(-n_q_heads // block_h)
        programs = bsz * h_blocks
        idx = torch.arange(programs * block_rows, device = dev)
        pid, linhas = idx // block_rows, idx % block_rows
        h_block, batch = pid % h_blocks, pid // h_blocks
        row_q = linhas % block_m
        row_h = h_block * block_h + linhas // block_m
        valido = (row_q < q_len) & (row_h < n_q_heads)
        destino = (batch * q_len + row_q) * n_q_heads + row_h
        origem = torch.empty((R * n_q_heads,), dtype = torch.long, device = dev)
        origem[destino[valido]] = idx[valido]
        _ORIGEM[chave] = origem
    # um gather so, sem sincronizar: toda (linha, cabeca) existe exatamente uma vez no workspace
    return bruto[origem].view(R, n_q_heads)


def reordenar_lse_dsa(bruto: torch.Tensor, R: int, H: int, block_h: int) -> torch.Tensor:
    """A irmã da anterior, para o caminho ESPARSO (`_dsa_attn_combine_kernel`).

    A decomposição é mais simples que a do denso, porque aqui não há bloco de consultas:

        h_blocks = cdiv(H, BLOCK_H);  row = pid // h_blocks
        head = (pid % h_blocks) * BLOCK_H + hloc

    Duas funções curtas em vez de um kernel parametrizado: a decomposição é a única coisa que
    varia entre as famílias, e escrevê-la explícita por caminho é mais legível do que passar
    strides que ninguém consegue conferir de cabeça.
    """
    dev = bruto.device
    chave = ("dsa", str(dev), R, H, block_h)
    origem = _ORIGEM.get(chave)
    if origem is None:
        h_blocks = -(-H // block_h)
        idx = torch.arange(R * h_blocks * block_h, device = dev)
        pid, hloc = idx // block_h, idx % block_h
        row = pid // h_blocks
        head = (pid % h_blocks) * block_h + hloc
        valido = head < H
        destino = row * H + head
        origem = torch.empty((R * H,), dtype = torch.long, device = dev)
        origem[destino[valido]] = idx[valido]
        _ORIGEM[chave] = origem
    return bruto[origem].view(R, H)


def cp_combinar(
    backend,
    o_local: torch.Tensor,
    lse_local: torch.Tensor,
    R: int,
    H: int,
    D: int,
    rank: int,
    world: int,
    stride_row: int,
    stride_head: int,
    salvar_lse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Combina a saída de atenção entre os ranks do grupo de CP.

    Entra: a saída local JÁ normalizada, para todas as `H` cabeças do grupo, sobre a fatia de
    sequência deste rank. Sai: a fatia de `H/world` cabeças deste rank, já combinada sobre a
    sequência inteira, em `(H/world, R, D)` fp32 — head-major, que é o que o `o_proj` quer.

    Dois coletivos: all_gather do lse (minúsculo) e reduce_scatter da saída pesada. A prova 19
    mediu que esse par custa MENOS que um all_reduce de 8 KB, porque um all_reduce já é
    internamente reduce-scatter + all-gather e paga duas fases onde este par paga uma.
    """
    assert H % world == 0, f"H={H} tem de dividir por world={world} (restrição dura do CP)"
    dev = o_local.device

    lse_all = torch.empty((world, R * H), dtype = torch.float32, device = dev)
    backend.all_gather(lse_all, lse_local.reshape(R * H))

    pesada = torch.empty((H, R, D), dtype = torch.float32, device = dev)
    lse_g = torch.empty((R * H,), dtype = torch.float32, device = dev) if salvar_lse else pesada
    grid = (R * H, triton.cdiv(D, 128))
    with torch.cuda.device(dev):
        _cp_correct_kernel[grid](
            o_local, lse_all, pesada, lse_g,
            world, R, H, rank, stride_row, stride_head,
            D = D, BLOCK_D = 128, SALVAR_LSE = 1 if salvar_lse else 0,
            num_warps = 4, num_stages = 2,
        )

    saida = torch.empty((H // world, R, D), dtype = torch.float32, device = dev)
    backend.reduce_scatter(saida, pesada)
    return saida, (lse_g if salvar_lse else None)


def referencia(o_locais: list[torch.Tensor], lses: list[torch.Tensor]) -> torch.Tensor:
    """A mesma conta em torch puro, para o autoteste e para os portões de KL da bancada.

    o_locais[r]: (R, H, D) já normalizado; lses[r]: (R, H)."""
    pilha_lse = torch.stack([x.float() for x in lses])              # (N, R, H)
    lse_g = torch.logsumexp(pilha_lse, dim = 0)                     # (R, H)
    pesos = torch.exp(pilha_lse - lse_g.unsqueeze(0))               # (N, R, H)
    pilha_o = torch.stack([x.float() for x in o_locais])            # (N, R, H, D)
    return (pilha_o * pesos.unsqueeze(-1)).sum(dim = 0)             # (R, H, D)


if __name__ == "__main__":
    # Autoteste: a combinação por log-sum-exp sobre fatias disjuntas tem de dar exatamente a
    # atenção sobre a união das fatias. Roda em uma placa só, simulando os ranks em sequência.
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    R, H, D, S, N = 3, 8, 64, 40, 4

    q = torch.randn(R, H, D, device = dev, dtype = torch.float32)
    k = torch.randn(S, D, device = dev, dtype = torch.float32)
    v = torch.randn(S, D, device = dev, dtype = torch.float32)

    escores = torch.einsum("rhd,sd->rhs", q, k) / D ** 0.5
    esperado = torch.einsum("rhs,sd->rhd", torch.softmax(escores, dim = -1), v)

    # fatias intercaladas, como o CP reparte por token: o rank r fica com s % N == r
    o_locais, lses = [], []
    for r in range(N):
        sel = torch.arange(r, S, N, device = dev)
        e = escores[:, :, sel]
        p = torch.softmax(e, dim = -1)
        o_locais.append(torch.einsum("rhs,sd->rhd", p, v[sel]))
        lses.append(torch.logsumexp(e, dim = -1))

    obtido = referencia(o_locais, lses)
    erro = (obtido - esperado).abs().max().item()
    assert erro < 1e-4, f"combine por log-sum-exp divergiu: erro maximo {erro}"
    print(f"combine por log-sum-exp exato sobre {N} fatias intercaladas · erro maximo {erro:.2e}")

    # Fatia vazia: um rank sem token nenhum nao pode contaminar o resultado
    o_locais.append(torch.zeros_like(o_locais[0]))
    lses.append(torch.full_like(lses[0], -float("inf")))
    obtido_v = referencia(o_locais, lses)
    erro_v = (obtido_v - esperado).abs().max().item()
    assert erro_v < 1e-4, f"fatia vazia contaminou o combine: erro maximo {erro_v}"
    print(f"fatia vazia ignorada corretamente · erro maximo {erro_v:.2e}")

    if dev != "cuda":
        print("sem GPU: os dois kernels Triton nao foram exercitados")
        raise SystemExit(0)

    # --- os kernels, com os ranks emulados numa placa so ------------------------------------
    o_locais, lses = o_locais[:N], lses[:N]

    # 1. _cp_lse_kernel contra o logsumexp do torch, a partir de um ws_ml sintetico no layout
    #    que os kernels de split produzem: base = ((pid * n_splits + s) * BLOCK_H + hloc) * 2
    BLOCK_H, n_splits = 8, 16
    n_pid = (R * H) // BLOCK_H
    ws = torch.randn(n_pid, n_splits, BLOCK_H, 2, device = dev, dtype = torch.float32)
    ws[..., 1] = ws[..., 1].abs() + 0.1          # l tem de ser positivo
    lse_ker = cp_lse_local(ws.reshape(-1).contiguous(), n_pid, n_splits, BLOCK_H)
    m, l = ws[..., 0], ws[..., 1]                # (n_pid, n_splits, BLOCK_H)
    lse_ref = torch.logsumexp(m + torch.log(l), dim = 1).reshape(-1)
    erro_l = (lse_ker - lse_ref).abs().max().item()
    assert erro_l < 1e-4, f"_cp_lse_kernel divergiu: erro maximo {erro_l}"
    print(f"_cp_lse_kernel bate com logsumexp do torch · erro maximo {erro_l:.2e}")

    # 2. _cp_correct_kernel: rodado uma vez por rank e somado, emula o reduce_scatter. A soma
    #    tem de dar a mesma coisa que a referencia.
    lse_all = torch.stack([x.reshape(R * H) for x in lses]).contiguous()
    acc = torch.zeros(H, R, D, device = dev, dtype = torch.float32)
    for r in range(N):
        pesada = torch.empty(H, R, D, device = dev, dtype = torch.float32)
        lse_g = torch.empty(R * H, device = dev, dtype = torch.float32)
        o_r = o_locais[r].contiguous()           # (R, H, D), strides (H*D, D, 1)
        _cp_correct_kernel[(R * H, triton.cdiv(D, 128))](
            o_r, lse_all, pesada, lse_g,
            N, R, H, r, H * D, D,
            D = D, BLOCK_D = 128, SALVAR_LSE = 1,
            num_warps = 4, num_stages = 2,
        )
        acc += pesada
    obtido_k = acc.permute(1, 0, 2)              # (H, R, D) -> (R, H, D)
    erro_k = (obtido_k - esperado).abs().max().item()
    assert erro_k < 1e-4, f"_cp_correct_kernel divergiu: erro maximo {erro_k}"
    print(f"_cp_correct_kernel + soma dos ranks = atencao completa · erro maximo {erro_k:.2e}")

    lse_g_ref = torch.logsumexp(lse_all, dim = 0)
    erro_g = (lse_g - lse_g_ref).abs().max().item()
    assert erro_g < 1e-4, f"lse global divergiu: erro maximo {erro_g}"
    print(f"lse global do kernel bate com o do torch · erro maximo {erro_g:.2e}")
