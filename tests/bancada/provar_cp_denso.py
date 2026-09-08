"""
O context parallel no caminho DENSO da MLA, provado numa placa só.

    python3 tests/bancada/provar_cp_denso.py
    python3 tests/bancada/provar_cp_denso.py --tokens 1500 --world 4 --cabecas 16

É o portão da etapa 5 do plano, sem precisar de TP. Emula os ranks em sequência no mesmo device:

  1. referência: um cache inteiro, a atenção sobre a sequência toda;
  2. `world` caches com `cp_rank` diferente, cada um atendendo só a sua fatia de tokens e
     devolvendo `(saída local normalizada, lse local)`;
  3. o combine por log-sum-exp, que tem de reproduzir a referência.

O que isto prova de verdade, e que nenhum teste anterior provava: que a atenção sobre uma fatia
intercalada da sequência é **exatamente** um termo da combinação. Ou seja, que os kernels de
atenção não precisam saber que faltam tokens no meio — a chave rope carrega a posição, e o
`q·k` sai certo.

Fica abaixo de `index_topk` de propósito: acima disso a produção usa o caminho ESPARSO, que é
etapa própria e depende do top-k distribuído.
"""
import argparse

import torch

from exllamav3.cache.mla import CacheLayer_MLA_fp16, CacheLayer_MLA_quant
from exllamav3.cache.cp_layout import comprimento_local
from exllamav3.constants import PAGE_SIZE
from exllamav3.modules.attention_fn.mla_triton import mla_attn_triton_decode
from exllamav3.modules.attention_fn.cp import referencia


class AtencaoFalsa:
    def __init__(self, d_c, d_r):
        self.kv_lora_rank = d_c
        self.qk_rope_head_dim = d_r
        self.idx_plane_dim = None
        self.index_kpool = 0


def montar_cache(cls, atencao, capacidade, world, rank, dev, **extra):
    c = cls(None, atencao, cache_id = 0, max_num_tokens = capacidade,
            cp_world = world, cp_rank = rank, **extra)
    c.alloc(torch.device(dev))
    return c


def preencher(cache, ckv, kpe, block_table, dev, passo = 64):
    total = ckv.shape[1]
    pos = 0
    while pos < total:
        n = min(passo, total - pos)
        seqlens = torch.full((1,), pos, dtype = torch.int32, device = dev)
        cache.update_kv_direct(seqlens, block_table, ckv[:, pos:pos + n], kpe[:, pos:pos + n], n)
        pos += n


def atender(cache, q_lat, q_pe, block_table, k_len, bsz, q_len, dev, quant):
    seqlens = torch.full((bsz,), k_len, dtype = torch.int32, device = dev)
    qc = (cache.sk, cache.bits) if quant else None
    ckv = cache.qk if quant else cache.k
    return mla_attn_triton_decode(
        q_lat, q_pe, ckv, cache.v, block_table, seqlens,
        bsz = bsz, q_len = q_len, causal = False,
        max_kv_len = k_len, qc = qc, devolver_lse = True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type = int, default = 1500, help = "abaixo de index_topk 2048")
    p.add_argument("--world", type = int, default = 4)
    p.add_argument("--cabecas", type = int, default = 16)
    p.add_argument("--tolerancia", type = float, default = 2e-3)
    a = p.parse_args()

    dev = "cuda"
    D_c, D_r, H = 512, 64, a.cabecas
    bsz, q_len = 1, 1
    R = bsz * q_len
    atencao = AtencaoFalsa(D_c, D_r)
    capacidade = -(-a.tokens // (PAGE_SIZE * a.world)) * PAGE_SIZE * a.world

    torch.manual_seed(0)
    ckv = torch.randn(1, a.tokens, D_c, device = dev, dtype = torch.half) * 0.1
    kpe = torch.randn(1, a.tokens, D_r, device = dev, dtype = torch.half) * 0.1
    q_lat = torch.randn(H, R, D_c, device = dev, dtype = torch.half) * 0.1
    q_pe = torch.randn(H, R, D_r, device = dev, dtype = torch.half) * 0.1

    falhas = []
    for nome, cls, quant, extra in [("fp16", CacheLayer_MLA_fp16, False, {}),
                                    ("quant Q8", CacheLayer_MLA_quant, True, dict(k_bits = 8))]:
        # 1. referencia sem CP
        ref = montar_cache(cls, atencao, capacidade, 1, 0, dev, **extra)
        bt_ref = torch.arange(capacidade // PAGE_SIZE, dtype = torch.int32, device = dev).view(1, -1)
        preencher(ref, ckv, kpe, bt_ref, dev)
        o_ref, lse_ref = atender(ref, q_lat, q_pe, bt_ref, a.tokens, bsz, q_len, dev, quant)
        o_ref = o_ref.float().clone()

        # 2. os ranks, cada um so com a sua fatia
        n_pag = capacidade // (PAGE_SIZE * a.world)
        bt = torch.arange(n_pag, dtype = torch.int32, device = dev).view(1, -1)
        o_locais, lses = [], []
        for r in range(a.world):
            c = montar_cache(cls, atencao, capacidade, a.world, r, dev, **extra)
            preencher(c, ckv, kpe, bt, dev)
            k_local = comprimento_local(a.tokens, a.world, r)
            o_r, lse_r = atender(c, q_lat, q_pe, bt, k_local, bsz, q_len, dev, quant)
            # a saida vem head-major (H, R, D_c); o combine trabalha em (R, H, D_c)
            o_locais.append(o_r.float().permute(1, 0, 2).contiguous())
            lses.append(lse_r)
            c.free()
        ref.free()

        # 3. o combine tem de reproduzir a referencia
        obtido = referencia(o_locais, lses).permute(1, 0, 2)     # -> (H, R, D_c)
        erro = (obtido - o_ref).abs().max().item()
        rel = erro / max(o_ref.abs().max().item(), 1e-9)
        ok = rel <= a.tolerancia
        print(f"{nome:9} · world {a.world} · {a.tokens} tokens · erro relativo {rel:.3e} "
              f"{'OK' if ok else 'FALHOU'}")
        if not ok:
            falhas.append(nome)

        # o lse global do combine tem de bater com o da referencia: e o mesmo softmax
        lse_g = torch.logsumexp(torch.stack([x.float() for x in lses]), dim = 0)
        d_lse = (lse_g - lse_ref.float()).abs().max().item()
        ok_lse = d_lse <= 5e-3
        print(f"{nome:9} · lse global == lse da referencia: {d_lse:.3e} "
              f"{'OK' if ok_lse else 'FALHOU'}")
        if not ok_lse:
            falhas.append(f"{nome} lse")

    # Um rank sem token nenhum nao pode contaminar: acontece com sequencia curta e world grande
    if a.world > 1:
        curto = a.world - 1
        print(f"\nsequencia de {curto} tokens em world {a.world}: "
              f"{sum(1 for r in range(a.world) if comprimento_local(curto, a.world, r) == 0)} "
              f"rank(s) sem token")

    print()
    if falhas:
        print(f"FALHOU: {', '.join(falhas)}")
        raise SystemExit(1)
    print("CP DENSO OK: a atencao sobre a fatia intercalada e um termo exato da combinacao")


if __name__ == "__main__":
    main()
