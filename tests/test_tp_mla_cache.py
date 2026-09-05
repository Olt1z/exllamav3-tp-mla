"""
Isola o caminho COM CACHE da MLAttention importada por TP, no mesmo processo e numa placa só.

Para cada camada MLA: prefill de S tokens e depois passos de decode de 1 token, no módulo
original (cache layer própria) e no importado com todas as cabeças (cache layer criada pelo
tp_import, achada por tp_cache_lookup), e na soma de dois importados com metade das cabeças.
O dict de params é compartilhado entre as camadas de uma mesma passada, então a camada de
indexador "full" publica dsa_topk_indices e a "shared" lê, como no modelo.

    python3 tests/test_tp_mla_cache.py /workspace/corte [S_prefill=34]
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from exllamav3 import Config, Model, Cache
from exllamav3.modules import MLAttention
from exllamav3.cache import CacheLayer_MLA_fp16
from exllamav3.model.model_tp_shared import SMProducer, SMConsumer

torch.manual_seed(0)
S = int(sys.argv[2]) if len(sys.argv) > 2 else 34
N_DEC = 3
MAX_TOKENS = 4096
config = Config.from_directory(sys.argv[1])
model = Model.from_config(config)
cache = Cache(model, max_num_tokens = MAX_TOKENS)
model.load(device = "cuda:0")
dev = 0
_im = torch.inference_mode(); _im.__enter__()

def dif(a, b):
    a, b = a.float(), b.float()
    return f"|dif| máx {(a - b).abs().max():.5f}  rel {(a - b).norm() / (a.norm() + 1e-9):.2e}"

producer = SMProducer()
consumer = SMConsumer(producer, device = dev)
local = {"device": dev, "consumer": consumer, "output_device": dev, "recurrent_modules": []}

mlas = [blk.attn for blk in model.modules if isinstance(getattr(blk, "attn", None), MLAttention)]
H = mlas[0].num_q_heads
variantes = {"original": [], "importado": [], "metade_a": [], "metade_b": []}
for m in mlas:
    exported = m.tp_export({m.key: (0, H, "heads")}, producer)
    variantes["original"].append(m)
    variantes["importado"].append(MLAttention.tp_import(local, exported, {m.key: (0, H, "heads")}, skip_reduction = True))
    variantes["metade_a"].append(MLAttention.tp_import(local, exported, {m.key: (0, H // 2, "heads")}, skip_reduction = True))
    variantes["metade_b"].append(MLAttention.tp_import(local, exported, {m.key: (H // 2, H, "heads")}, skip_reduction = True))
    producer.clear()
cache_id = id(cache)
caches_orig = []
for m in mlas:
    l = CacheLayer_MLA_fp16(None, m, cache_id, MAX_TOKENS); l.alloc(torch.device(dev)); caches_orig.append(l)

bt = torch.arange(MAX_TOKENS // 256, dtype = torch.int32, device = dev).view(1, -1)
xs = [torch.randn(1, S + N_DEC, config.hidden_size, dtype = torch.half, device = dev) for _ in mlas]

def passada(nome, mods, a, b):
    """Um chunk [a, b) por todas as camadas, params compartilhado entre camadas."""
    seqlens = torch.tensor([a], dtype = torch.int32, device = dev)
    params = {"attn_mode": "flash_attn", "block_table": bt, "cache_seqlens": seqlens,
              "positions": seqlens.clone(), "causal": True}
    outs = []
    for i, m in enumerate(mods):
        p = dict(params)
        p["cache"] = caches_orig[i] if nome == "original" else cache_id
        outs.append(m.forward(xs[i][:, a:b].contiguous(), p))
        if "dsa_topk_indices" in p:
            params["dsa_topk_indices"] = p["dsa_topk_indices"]
    return outs

passos = [(0, S)] + [(S + k, S + k + 1) for k in range(N_DEC)]
for a, b in passos:
    r = {nome: passada(nome, mods, a, b) for nome, mods in variantes.items()}
    print(f"\n== tokens [{a}, {b}) {'prefill' if b - a > 1 else 'decode'}  (esparso: {b > mlas[0].index_topk})")
    for i, m in enumerate(mlas):
        print(f"  camada {m.layer_idx} ({m.indexer_mode}): importado {dif(r['importado'][i], r['original'][i])}"
              f"   metades {dif(r['metade_a'][i] + r['metade_b'][i], r['original'][i])}")
