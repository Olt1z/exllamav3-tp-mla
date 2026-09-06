"""Restaura q/k/v do KDA em BF16 no artefato EXL3 do Flash (a esteira quantizou os três em separado, com suh
distintos, e o motor os funde num qkv_proj que só carrega em fp16). Baixa do repositório BF16 só os shards
necessários, grava um shard novo e tira os .trellis/.suh/.svh/.mul1 desses módulos do índice."""
import json, os, re, sys, subprocess
from safetensors import safe_open
from safetensors.torch import save_file
d, repo_bf16 = sys.argv[1], sys.argv[2]
idx_p = f"{d}/model.safetensors.index.json"; idx = json.load(open(idx_p)); w = idx["weight_map"]
re_qkv = re.compile(r"^(model\.language_model\.layers\.\d+\.self_attn\.[qkv]_proj)\.(trellis|suh|svh|mul1|mcg)$")
mods = sorted({re_qkv.match(k).group(1) for k in w if re_qkv.match(k)})
print("módulos KDA q/k/v em EXL3:", len(mods))
if not mods: sys.exit(0)
# índice do BF16: quais shards têm esses pesos
subprocess.run(["hf", "download", repo_bf16, "model.safetensors.index.json", "--local-dir", "/workspace/bf16"], check=True, stdout=subprocess.DEVNULL)
wb = json.load(open("/workspace/bf16/model.safetensors.index.json"))["weight_map"]
precisa = {m + ".weight": wb[m + ".weight"] for m in mods}
shards = sorted(set(precisa.values())); print("shards BF16 a baixar:", shards)
subprocess.run(["hf", "download", repo_bf16, *shards, "--local-dir", "/workspace/bf16"], check=True, stdout=subprocess.DEVNULL)
novos = {}
for k, sh in precisa.items():
    with safe_open(f"/workspace/bf16/{sh}", "pt") as f:
        novos[k] = f.get_tensor(k)
save_file(novos, f"{d}/model-kda-qkv-bf16.safetensors", metadata={"format": "pt"})
for k in list(w):
    if re_qkv.match(k): del w[k]
for k in novos: w[k] = "model-kda-qkv-bf16.safetensors"
json.dump(idx, open(idx_p, "w"), indent=2)
print(f"reparado: {len(novos)} pesos q/k/v em BF16, {len(mods)*4} entradas EXL3 removidas do índice")
