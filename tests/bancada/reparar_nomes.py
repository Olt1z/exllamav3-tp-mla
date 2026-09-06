"""Renomeia, num artefato EXL3 do GLM-5.3-Flash salvo pela esteira (transformers 5.x), os tensores cujo nome de
módulo mudou em relação ao checkpoint que o exllamav3 lê: attn_hc/ffn_hc.* -> hc_attn_*/hc_ffn_*, e
self_attn.forget_gate.{f_a_proj,f_b_proj,A_log,dt_bias} -> self_attn.{...}. Copia os tensores para um shard
novo com os nomes esperados e atualiza o índice (os antigos ficam, inofensivos)."""
import json, re, sys
from safetensors import safe_open
from safetensors.torch import save_file
d = sys.argv[1]
idx_p = f"{d}/model.safetensors.index.json"; idx = json.load(open(idx_p)); w = idx["weight_map"]
regras = [
    (re.compile(r"^(.*layers\.\d+\.)(attn|ffn)_hc\.(fn|base|scale)$"), lambda m: f"{m.group(1)}hc_{m.group(2)}_{m.group(3)}"),
    (re.compile(r"^(.*layers\.\d+\.self_attn\.)forget_gate\.(f_a_proj\.weight|f_b_proj\.weight|A_log|dt_bias)$"), lambda m: f"{m.group(1)}{m.group(2)}"),
]
novos = {}
for k in list(w):
    for re_, f in regras:
        m = re_.match(k)
        if m:
            alvo = f(m)
            if alvo not in w:
                with safe_open(f"{d}/{w[k]}", "pt") as fh: novos[alvo] = fh.get_tensor(k)
            break
if novos:
    save_file(novos, f"{d}/model-nomes-do-checkpoint-2.safetensors", metadata={"format": "pt"})
    for k in novos: w[k] = "model-nomes-do-checkpoint-2.safetensors"
    json.dump(idx, open(idx_p, "w"), indent=2)
print(f"renomeados: {len(novos)}", sorted(novos)[:4])
