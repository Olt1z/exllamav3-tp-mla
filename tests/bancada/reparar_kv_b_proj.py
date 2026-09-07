"""
Repara um artefato EXL3 saído da esteira do GPTQModel em que o `kv_b_proj` foi quantizado.

O ExLlamaV3 nunca trata o `kv_b_proj` como Linear: lê `kv_b_proj.weight` cru (fp16/bf16/fp8) e
o dobra em W_UK e W_UV na carga (mla_attn.py load_local). Um artefato com `kv_b_proj.trellis`
não carrega. Este script desquantiza a treliça com o próprio reconstrutor do EXL3, grava o
peso fp16 num shard novo e aponta o índice para ele. Perda: a do EXL3 daquele tensor, igual
nos dois lados da comparação (uma placa × TP), então serve para a bancada.

    python3 reparar_kv_b_proj.py /workspace/corte
"""
import json
import os, sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from exllamav3.modules.quant.exl3 import LinearEXL3

d = sys.argv[1]
idx_path = f"{d}/model.safetensors.index.json"
if not os.path.exists(idx_path):
    print("sem índice de shards: artefato de um arquivo só (conversor nativo), nada a reparar")
    raise SystemExit(0)
idx = json.load(open(idx_path))
wm = idx["weight_map"]
cfg = json.load(open(f"{d}/config.json"))
tc = cfg.get("text_config", cfg)
esperado = (tc["num_attention_heads"] * (tc["qk_nope_head_dim"] + tc["v_head_dim"]), tc["kv_lora_rank"])

grupos = sorted({k.rsplit(".", 1)[0] for k in wm if k.endswith(".trellis") and k.endswith("kv_b_proj.trellis")})
if not grupos:
    print("nenhum kv_b_proj quantizado; nada a fazer")
    sys.exit(0)

novo = {}
for g in grupos:
    t = {}
    for sub in ("trellis", "suh", "svh", "su", "sv", "mcg", "mul1"):
        k = f"{g}.{sub}"
        if k in wm:
            with safe_open(f"{d}/{wm[k]}", "pt", device="cpu") as f:
                t[sub] = f.get_tensor(k)
    tr = t["trellis"].cuda()
    in_f, out_f = tr.shape[0] * 16, tr.shape[1] * 16
    lin = LinearEXL3(
        None, in_f, out_f, None, t.get("su"), t.get("sv"),
        t.get("suh").cuda() if t.get("suh") is not None else None,
        t.get("svh").cuda() if t.get("svh") is not None else None,
        tr, t.get("mcg"), t.get("mul1"), None, torch.half,
    )
    w = lin.get_weight_tensor().T.contiguous().half().cpu()   # HF: (out, in)
    assert tuple(w.shape) == esperado, f"{g}: {tuple(w.shape)} != {esperado}"
    novo[f"{g}.weight"] = w
    print(g, tuple(w.shape), f"|w| médio {w.float().abs().mean():.4f}")

shard = "model-kv_b_proj-fp16.safetensors"
save_file(novo, f"{d}/{shard}", metadata={"format": "pt"})
for k in novo:
    wm[k] = shard
json.dump(idx, open(idx_path, "w"), indent=2)
print(f"{len(novo)} tensores gravados em {shard}; índice atualizado")
