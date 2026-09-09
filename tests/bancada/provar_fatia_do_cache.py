"""
A fatia por rank do context parallel, provada numa placa só.

    python3 tests/bancada/provar_fatia_do_cache.py
    python3 tests/bancada/provar_fatia_do_cache.py --tokens 3000 --world 4

Não precisa de TP nem de várias placas: constrói `world` caches com `cp_rank` diferente no mesmo
device, escreve a MESMA sequência em todos, e confere os DOIS regimes que o CP usa:

  latente e rope       **REPARTIDOS** — o token `p` mora no rank `p % world`, na linha `p // world`;
                       a união dos ranks tem de reproduzir a referência sem CP.
  plano do indexador   **REPLICADO** — o token `p` está em TODOS os ranks, na linha `p`; cada rank
                       tem de bater com a referência inteira, não com uma fatia dela.

Testar os dois com a mesma regra esconderia exatamente o que separa as duas etapas. E o
encolhimento esperado NÃO é `world`: sai das formas declaradas (`shape_c`, `shape_r`, `shape_i`,
`shape_p`), porque a fração do ideal que se captura depende da razão entre indexador e principal —
que muda por modelo e é justamente o que não pode virar constante.

Cobre os dois caminhos de escrita:
  fp16   `_mla_kv_update_kernel`
  quant  `_mla_kv_quant_scatter_kernel` (o pacote int32 + escalas, que tem máscara própria)

E o plano AGRUPADO (`k_pool`), que era o obstáculo da etapa 4 e agora tem de funcionar.
"""
import argparse

import torch

from exllamav3.cache.mla import CacheLayer_MLA_fp16, CacheLayer_MLA_quant
from exllamav3.cache.cp_layout import comprimento_local
from exllamav3.constants import PAGE_SIZE


class AtencaoFalsa:
    """O mínimo que CacheLayer_MLA_* lê do módulo de atenção."""
    def __init__(self, d_c, d_r, idx, kpool = 0):
        self.kv_lora_rank = d_c
        self.qk_rope_head_dim = d_r
        self.idx_plane_dim = idx
        self.index_head_dim = idx
        self.index_kpool = kpool


def _bytes_indexador(c):
    """Quantos bytes do cache sao plano de indexador, lido das formas declaradas."""
    import numpy as np
    t = 0
    if c.shape_i: t += int(np.prod(c.shape_i)) * torch.half.itemsize
    if c.shape_p: t += int(np.prod(c.shape_p)) * torch.half.itemsize
    return t


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

        # A alocacao por placa cai, mas NAO por `world`: o latente e repartido e o indexador e
        # replicado. O esperado sai das formas declaradas, nunca de constante -- e a fracao do
        # ideal capturada e exatamente o que muda entre modelos.
        principal = ref.storage_size() - _bytes_indexador(ref)
        idx = _bytes_indexador(ref)
        esperado = (principal + idx) / (principal / a.world + idx)
        enc = ref.storage_size() / ranks[0].storage_size()
        ok_enc = abs(enc - esperado) < 1e-3
        print(f"{nome:9} · alocacao por placa caiu {enc:.2f}x "
              f"(esperado {esperado:.2f}x pelas formas; ideal seria {a.world}x, e a diferenca "
              f"e o indexador replicado) {'OK' if ok_enc else 'FALHOU'}")
        if not ok_enc:
            falhas.append(f"{nome}: encolhimento {enc} != {esperado}")

        # a uniao dos ranks tem de reproduzir a referencia, token a token. Os dois planos vivem
        # em regimes diferentes: o latente/rope e REPARTIDO (o token p esta no rank p % world, na
        # linha p // world) e o do indexador e REPLICADO (o token p esta em TODOS os ranks, na
        # linha p). Testar os dois com a mesma regra esconderia justamente o que a etapa mudou.
        piores = {}
        for p_glob in range(a.tokens):
            r = p_glob % a.world
            p_loc = p_glob // a.world
            t_ref = ref.v.reshape(-1, D_r)[p_glob]
            t_cp = ranks[r].v.reshape(-1, D_r)[p_loc]
            piores["v"] = max(piores.get("v", 0.0),
                              (t_ref.float() - t_cp.float()).abs().max().item())
            # replicado: confere em TODO rank, na posicao global
            t_ref_i = ref.k_idx.reshape(-1, D_i)[p_glob]
            for c in ranks:
                t_cp_i = c.k_idx.reshape(-1, D_i)[p_glob]
                piores["k_idx (replicado)"] = max(
                    piores.get("k_idx (replicado)", 0.0),
                    (t_ref_i.float() - t_cp_i.float()).abs().max().item())
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

    # O plano AGRUPADO era o obstaculo da etapa 4 e agora tem de FUNCIONAR: com o indexador
    # replicado, cada rank tem os `kpool` tokens globais consecutivos que a entrada agrupa.
    atencao_pool = AtencaoFalsa(D_c, D_r, D_i, kpool = 8)
    ref_p = montar(CacheLayer_MLA_fp16, atencao_pool, capacidade, 1, 0, dev)
    ranks_p = [montar(CacheLayer_MLA_fp16, atencao_pool, capacidade, a.world, r, dev)
               for r in range(a.world)]
    n_pools = 24
    chaves = torch.randn(1, n_pools, D_i, device = dev, dtype = torch.half)
    bt_r = torch.arange(capacidade // PAGE_SIZE, dtype = torch.int32, device = dev).view(1, -1)
    bt_c = torch.arange(capacidade // (PAGE_SIZE * a.world), dtype = torch.int32,
                        device = dev).view(1, -1)
    zero = torch.zeros((1,), dtype = torch.int32, device = dev)
    ref_p.update_pool_direct(zero, bt_r, chaves)
    for c in ranks_p:
        c.update_pool_direct(zero, bt_c, chaves)
    ok_pool = all(torch.equal(ref_p.k_pool.reshape(-1, D_i)[:n_pools],
                              c.k_pool.reshape(-1, D_i)[:n_pools]) for c in ranks_p)
    print(f"plano agrupado sob CP: {'bit a bit em todo rank' if ok_pool else 'DIVERGIU'} "
          f"(era o obstaculo da etapa 4)")
    if not ok_pool:
        falhas.append("k_pool sob CP")
    for c in ranks_p + [ref_p]:
        c.free()

    print()
    if falhas:
        print(f"FALHOU em {len(falhas)}: {', '.join(falhas)}")
        raise SystemExit(1)
    print(f"FATIA OK · world {a.world} · {a.tokens} tokens · chunk {a.passo}")


if __name__ == "__main__":
    main()
