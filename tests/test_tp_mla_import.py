"""
Isola a exportação/importação TP da MLAttention, no mesmo processo e numa placa só.

Para cada camada MLA do modelo: (1) forward do módulo original; (2) forward do módulo
reconstruído por tp_export -> tp_import com todas as cabeças; (3) soma dos forwards de dois
módulos importados com metade das cabeças cada (o que o all-reduce faria). Compara os três
e, se divergirem, compara filho a filho (projeções, W_UK/W_UV, norms) para apontar o culpado.

    python3 tests/test_tp_mla_import.py /workspace/corte
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from exllamav3 import Config, Model, Cache
from exllamav3.modules import MLAttention
from exllamav3.model.model_tp_shared import SMProducer, SMConsumer

torch.manual_seed(0)
model_dir = sys.argv[1]
config = Config.from_directory(model_dir)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens = 4096)
model.load(device = "cuda:0")
dev = 0

def dif(a, b):
    a, b = a.float(), b.float()
    return f"|dif| máx {(a - b).abs().max():.5f}  rel {(a - b).norm() / (a.norm() + 1e-9):.2e}"

producer = SMProducer()
consumer = SMConsumer(producer, device = dev)
local = {"device": dev, "consumer": consumer}

mlas = [(i, blk.attn) for i, blk in enumerate(model.modules) if isinstance(getattr(blk, "attn", None), MLAttention)]
print(f"{len(mlas)} camadas MLA")
for i, m in mlas:
    H = m.num_q_heads
    x = torch.randn(1, 8, m.hidden_size, dtype = torch.half, device = dev)
    params = lambda: {"attn_mode": "flash_attn_nc", "causal": True, "position": 0}
    ref = m.forward(x.clone(), params())

    plan_full = {m.key: (0, H, "heads")}
    exported = m.tp_export(plan_full, producer)
    m_full = MLAttention.tp_import(local, exported, plan_full, skip_reduction = True)
    out_full = m_full.forward(x.clone(), params())

    m_a = MLAttention.tp_import(local, exported, {m.key: (0, H // 2, "heads")}, skip_reduction = True)
    m_b = MLAttention.tp_import(local, exported, {m.key: (H // 2, H, "heads")}, skip_reduction = True)
    out_split = m_a.forward(x.clone(), params()) + m_b.forward(x.clone(), params())

    print(f"\n== módulo {i} ({m.key}) H={H} indexer={m.indexer_mode}")
    print("  importado cheio  vs original:", dif(out_full, ref))
    print("  soma das metades vs original:", dif(out_split, ref))

    # Filho a filho, com o importado cheio
    for nome in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj", "idx_wq_b", "idx_wk", "idx_weights"):
        a, b = getattr(m, nome), getattr(m_full, nome)
        if a is None:
            continue
        xi = torch.randn(1, 8, a.in_features, dtype = torch.half, device = dev)
        print(f"  {nome:<20}", dif(a.forward(xi, {}), b.forward(xi, {})),
              f"in {a.in_features}/{b.in_features} out {a.out_features}/{b.out_features} tipo {a.quant_type}/{b.quant_type}")
    for nome in ("q_a_layernorm", "kv_a_layernorm", "idx_k_norm"):
        a, b = getattr(m, nome), getattr(m_full, nome)
        if a is None:
            continue
        xi = torch.randn(1, 8, a.weight.shape[-1], dtype = torch.half, device = dev)
        print(f"  {nome:<20}", dif(a.forward(xi, {}), b.forward(xi, {})),
              "bias" if getattr(b, "bias", None) is not None else "")
    print("  w_uk_flat            ", dif(m.w_uk_flat, m_full.w_uk_flat))
    print("  w_uv_flat            ", dif(m.w_uv_flat, m_full.w_uv_flat))
    print(f"  rope {m.rope is not None}/{m_full.rope is not None}  sm_scale {m.sm_scale:.5f}/{m_full.sm_scale:.5f}  "
          f"l4 {m.l4_beta}/{m_full.l4_beta}  out_dtype {m.out_dtype}/{m_full.out_dtype}")
    producer.clear()
