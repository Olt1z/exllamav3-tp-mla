"""
O context parallel no caminho ESPARSO (DSA), provado numa placa só.

    python3 tests/bancada/provar_cp_esparso.py
    python3 tests/bancada/provar_cp_esparso.py --world 4 --tokens 30000 --topk 2048

Portão da etapa 7. É o caminho que a produção usa acima de `index_topk`, e o que a etapa 6
destravou: com o plano do indexador **replicado**, o top-k é global e idêntico em todo rank, e
cada rank atende só à interseção entre esse top-k e os tokens que possui.

A propriedade sob teste é essa interseção: **repartir os índices selecionados entre os ranks e
combinar por log-sum-exp tem de dar a mesma coisa que atender a todos de uma vez.** É o análogo
esparso do que `provar_cp_denso.py` provou para o denso.

O pool fica inteiro em todos os "ranks" de propósito. Repartir o cache já está provado em
`provar_fatia_do_cache.py`, e misturar as duas coisas num teste só faria uma esconder a outra.
"""
import argparse

import torch

from exllamav3.modules.attention_fn.dsa_triton import dsa_attn
from exllamav3.modules.attention_fn.cp import referencia


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type = int, default = 30000, help = "tamanho do pool")
    p.add_argument("--topk", type = int, default = 2048, help = "quantos o indexador seleciona")
    p.add_argument("--world", type = int, default = 4)
    p.add_argument("--cabecas", type = int, default = 16)
    p.add_argument("--tolerancia", type = float, default = 3e-3)
    a = p.parse_args()

    dev = "cuda"
    D_c, D_r, H, R = 512, 64, a.cabecas, 1

    torch.manual_seed(0)
    # pool contiguo com tabela identidade, que e o modo que a propria dsa_attn documenta
    pool_c = torch.randn(a.tokens, D_c, device = dev, dtype = torch.half) * 0.1
    pool_r = torch.randn(a.tokens, D_r, device = dev, dtype = torch.half) * 0.1
    q = torch.randn(R, H, D_c + D_r, device = dev, dtype = torch.half) * 0.1
    q_pe = q[:, :, D_c:].contiguous()
    bt = torch.arange(-(-a.tokens // 256), dtype = torch.int32, device = dev).view(1, -1)

    # a selecao GLOBAL do indexador: com o plano replicado ela e identica em todo rank
    sel = torch.randperm(a.tokens, device = dev)[:a.topk].sort().values.to(torch.int32)

    def atender(indices):
        if indices.numel() == 0:
            return None, None
        return dsa_attn(
            q, pool_c, pool_r, bt,
            indices = indices.view(1, -1), k_len = indices.numel(),
            scale = (D_c + D_r) ** -0.5, page_size = 256,
            q_pe = q_pe, out_latent = True, devolver_lse = True,
        )

    o_ref, lse_ref = atender(sel)
    o_ref = o_ref.float().clone()

    # cada rank fica com os selecionados que POSSUI: idx % world == r
    o_locais, lses, contagem = [], [], []
    for r in range(a.world):
        meus = sel[(sel % a.world) == r]
        contagem.append(meus.numel())
        o_r, lse_r = atender(meus)
        if o_r is None:
            # rank sem token selecionado: contribui peso zero, e o combine tem de aguentar
            o_locais.append(torch.zeros(R, H, D_c, device = dev, dtype = torch.float32))
            lses.append(torch.full((R, H), -float("inf"), device = dev, dtype = torch.float32))
        else:
            o_locais.append(o_r.float().permute(1, 0, 2).contiguous())
            lses.append(lse_r)

    obtido = referencia(o_locais, lses).permute(1, 0, 2)
    erro = (obtido - o_ref).abs().max().item()
    rel = erro / max(o_ref.abs().max().item(), 1e-9)

    lse_g = torch.logsumexp(torch.stack([x.float() for x in lses]), dim = 0)
    d_lse = (lse_g - lse_ref.float()).abs().max().item()

    print(f"pool {a.tokens} · top-k {a.topk} · world {a.world} · H {H}")
    print(f"  selecionados por rank: {contagem}  (soma {sum(contagem)}, esperado {a.topk})")
    print(f"  erro relativo da saida : {rel:.3e} {'OK' if rel <= a.tolerancia else 'FALHOU'}")
    print(f"  lse global vs referencia: {d_lse:.3e} {'OK' if d_lse <= 5e-3 else 'FALHOU'}")

    if sum(contagem) != a.topk:
        print("  PARTICAO DOS INDICES ERRADA")
        raise SystemExit(1)
    if rel > a.tolerancia or d_lse > 5e-3:
        raise SystemExit(1)
    print("\nCP ESPARSO OK: repartir os selecionados e combinar == atender a todos de uma vez")


if __name__ == "__main__":
    main()
