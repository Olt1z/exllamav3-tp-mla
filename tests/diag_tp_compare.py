"""Compara dois rastreios do diag_tp_trace.py e aponta o primeiro módulo que diverge."""
import sys, torch

a, b = torch.load(sys.argv[1]), torch.load(sys.argv[2])
def dif(x, y):
    if x is None or y is None:
        return "—"
    if x.shape != y.shape:
        return f"formas {tuple(x.shape)} × {tuple(y.shape)}"
    return f"|dif| máx {(x - y).abs().max():.5f} rel {(x - y).norm() / (x.norm() + 1e-9):.2e}"

for i in sorted(set(a["rec"]) | set(b["rec"])):
    ra, rb = a["rec"].get(i, []), b["rec"].get(i, [])
    for k in range(max(len(ra), len(rb))):
        if k >= len(ra) or k >= len(rb):
            print(f"módulo {i} chamada {k}: só em {'A' if k < len(ra) else 'B'}")
            continue
        ca, cb = ra[k], rb[k]
        marca = "" if ca["params"] == cb["params"] else f"  PARAMS DIFEREM: {ca['params']} × {cb['params']}"
        print(f"módulo {i:2d} chamada {k} {ca['nome'][:48]:<48} in: {dif(ca['in'], cb['in']):<40} out: {dif(ca['out'], cb['out'])}{marca}")
print("logits:", dif(a["logits"], b["logits"]))
