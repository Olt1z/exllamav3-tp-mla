"""
A fatia por rank do context parallel, provada numa placa só.

    python3 tests/bancada/provar_fatia_do_cache.py
    python3 tests/bancada/provar_fatia_do_cache.py --tokens 3000 --world 4

Não precisa de TP nem de várias placas: constrói `world` caches com `cp_rank` diferente no mesmo
device, escreve a MESMA sequência em todos, e exige que a **união** dos caches seja exatamente o
que um cache sem CP guardaria.

É o que testa o código que a etapa 4 realmente acrescentou — a escrita mascarada nos kernels de
append. O portão de `world = 1` do plano é identidade por construção e não prova nada; este prova.

Cobre os dois caminhos de escrita:
  fp16   `_mla_kv_update_kernel`
  quant  `_mla_kv_quant_scatter_kernel` (o pacote int32 + escalas, que tem máscara própria)

E o plano de chaves do indexador, que também é escrita por token.
"""
import argparse

import torch

from exllamav3.cache.mla import CacheLayer_MLA_fp16, CacheLayer_MLA_quant
from exllamav3.cache.cp_layout import comprimento_local
from exllamav3.constants import PAGE_SIZE


class AtencaoFalsa:
    """O mínimo que CacheLayer_MLA_* lê do módulo de atenção."""
    def __init__(self, d_c, d_r, idx):
        self.kv_lora_rank = d_c
        self.qk_rope_head_dim = d_r
        self.idx_plane_dim = idx
        self.index_head_dim = idx
        self.index_kpool = 0          # o plano agrupado nao tem versao sob CP, e falha alto


def montar(cls, atencao, capacidade, world, rank, dev, **extra):
    c = cls(None, atencao, cache_id = 0, max_num_tokens = capacidade,
            cp_world = world, cp_rank = rank, **extra)
    c.alloc(torch.device(dev))
    return c


def escrever(cache, ckv, kpe, kidx, block_table, dev, passo):
    """Escreve a sequência em pedaços de `passo`, como o gerador faz."""
    total = ckv.shape[1]
    pos = 0
    while pos < total:
        n = min(passo, total - pos)
        seqlens = torch.full((1,), pos, dtype = torch.int32, device = dev)
        cache.update_kv_direct(seqlens, block_table, ckv[:, pos:pos + n], kpe[:, pos:pos + n], n)
        if kidx is not None:
            cache.update_idx_direct(seqlens, block_table, kidx[:, pos:pos + n], n)
        pos += n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type = int, default = 2000)
    p.add_argument("--world", type = int, default = 4)
    p.add_argument("--passo", type = int, default = 7, help = "chunk de escrita; primo de proposito")
    a = p.parse_args()

    dev = "cuda"
    D_c, D_r, D_i = 512, 64, 128
    atencao = AtencaoFalsa(D_c, D_r, D_i)
    # capacidade multipla de PAGE_SIZE * world para toda pagina virtual fechar
    capacidade = -(-a.tokens // (PAGE_SIZE * a.world)) * PAGE_SIZE * a.world

    torch.manual_seed(0)
    ckv = torch.randn(1, a.tokens, D_c, device = dev, dtype = torch.half)
    kpe = torch.randn(1, a.tokens, D_r, device = dev, dtype = torch.half)
    kidx = torch.randn(1, a.tokens, D_i, device = dev, dtype = torch.half)

    falhas = []
    for nome, cls, extra in [("fp16", CacheLayer_MLA_fp16, {}),
                             ("quant Q8", CacheLayer_MLA_quant, dict(k_bits = 8))]:
        # referencia: um cache so, sem CP
        ref = montar(cls, atencao, capacidade, 1, 0, dev, **extra)
        bt_ref = torch.arange(capacidade // PAGE_SIZE, dtype = torch.int32,
                              device = dev).view(1, -1)
        escrever(ref, ckv, kpe, kidx, bt_ref, dev, a.passo)

        # os `world` ranks, cada um com o seu pedaco
        n_pag = capacidade // (PAGE_SIZE * a.world)
        bt = torch.arange(n_pag, dtype = torch.int32, device = dev).view(1, -1)
        ranks = [montar(cls, atencao, capacidade, a.world, r, dev, **extra)
                 for r in range(a.world)]
        for r, c in enumerate(ranks):
            escrever(c, ckv, kpe, kidx, bt, dev, a.passo)

        # a alocacao por placa caiu `world` vezes?
        enc = ref.storage_size() / ranks[0].storage_size()
        ok_enc = abs(enc - a.world) < 1e-6
        print(f"{nome:9} · alocacao por placa caiu {enc:.2f}x (esperado {a.world}x) "
              f"{'OK' if ok_enc else 'FALHOU'}")
        if not ok_enc:
            falhas.append(f"{nome}: encolhimento {enc}")

        # a uniao dos ranks tem de reproduzir a referencia, token a token
        piores = {}
        for p_glob in range(a.tokens):
            r = p_glob % a.world
            p_loc = p_glob // a.world
            for campo, largura in (("v", D_r), ("k_idx", D_i)):
                t_ref = getattr(ref, campo).reshape(-1, largura)[p_glob]
                t_cp = getattr(ranks[r], campo).reshape(-1, largura)[p_loc]
                d = (t_ref.float() - t_cp.float()).abs().max().item()
                piores[campo] = max(piores.get(campo, 0.0), d)
            # o latente: fp16 cru numa classe, pacote int32 + escalas na outra
            if cls is CacheLayer_MLA_fp16:
                t_ref = ref.k.reshape(-1, D_c)[p_glob]
                t_cp = ranks[r].k.reshape(-1, D_c)[p_loc]
                d = (t_ref.float() - t_cp.float()).abs().max().item()
            else:
                a_ref = ref.qk.reshape(-1, ref.qshape[-1])[p_glob]
                a_cp = ranks[r].qk.reshape(-1, ref.qshape[-1])[p_loc]
                s_ref = ref.sk.reshape(-1, ref.sshape[-1])[p_glob]
                s_cp = ranks[r].sk.reshape(-1, ref.sshape[-1])[p_loc]
                d = 0.0 if (torch.equal(a_ref, a_cp) and torch.equal(s_ref, s_cp)) else 1.0
            piores["latente"] = max(piores.get("latente", 0.0), d)

        for campo, d in sorted(piores.items()):
            bom = d == 0.0
            print(f"{nome:9} · {campo:8} uniao dos ranks == referencia: "
                  f"{'bit a bit' if bom else f'DIVERGIU (max {d:.3e})'}")
            if not bom:
                falhas.append(f"{nome}/{campo}")

        # e o que NAO e do rank tem de continuar zerado: escrita fora da fatia seria corrupcao
        # silenciosa do token de outro rank
        for r, c in enumerate(ranks):
            n_local = comprimento_local(a.tokens, a.world, r)
            sobra = c.v.reshape(-1, D_r)[n_local:]
            if sobra.numel() and sobra.abs().max().item() != 0.0:
                print(f"{nome:9} · rank {r} escreveu FORA da propria fatia")
                falhas.append(f"{nome}: rank {r} escreveu fora")
        for c in ranks + [ref]:
            c.free()

    # o plano agrupado tem de recusar CP em vez de calcular errado
    atencao_pool = AtencaoFalsa(D_c, D_r, D_i)
    atencao_pool.index_kpool = 8
    c = montar(CacheLayer_MLA_fp16, atencao_pool, capacidade, a.world, 0, dev)
    try:
        c.update_pool_direct(torch.zeros((1,), dtype = torch.int32, device = dev),
                             torch.zeros((1, 1), dtype = torch.int32, device = dev),
                             torch.zeros((1, 1, D_i), device = dev, dtype = torch.half))
        print("plano agrupado ACEITOU CP — devia ter recusado")
        falhas.append("k_pool aceitou CP")
    except AssertionError:
        print("plano agrupado recusa CP, como tem de ser")
    c.free()

    print()
    if falhas:
        print(f"FALHOU em {len(falhas)}: {', '.join(falhas)}")
        raise SystemExit(1)
    print(f"FATIA OK · world {a.world} · {a.tokens} tokens · chunk {a.passo}")


if __name__ == "__main__":
    main()
