#!/bin/bash
# Bancada do tensor parallel na MLAttention: sobe numa máquina da Vast com 2+ placas, instala o
# fork, baixa o corte de 4 camadas do GLM-5.3 em EXL3 e roda o teste de fumaça nas quatro formas:
#   1. uma placa, grava a base;          2. uma placa de novo, mede o ruído do motor;
#   3. TP, teacher forcing contra a base;  4. TP com contexto acima de index_topk (DSA esparso).
# Tudo vai para /workspace/bancada.log e, no fim, para $REPO_SAIDAS/saidas/tp-mla/$PROVA_ID/.
# Imagem: pytorch/pytorch:*-cuda12.8-cudnn9-devel (torch + nvcc). Variáveis do env da instância:
# HF_TOKEN, FORK (url), BRANCH (tp-mla), CORTE (repo HF do modelo), PROVA_ID, REPO_SAIDAS.
set -uo pipefail
mkdir -p /workspace && exec > >(tee -a /workspace/bancada.log) 2>&1
marco() { echo "=== $1 · $(date -u +%FT%TZ)"; }
marco "BANCADA tp-mla"

: "${HF_TOKEN:?HF_TOKEN ausente}"
export HF_TOKEN
FORK="${FORK:-https://github.com/Olt1z/exllamav3-tp-mla}"
BRANCH="${BRANCH:-tp-mla}"
CORTE="${CORTE:-Olt1z/GLM-5.3-podado-4L-EXL3-balanced-bl4ck0ut}"
REPO_SAIDAS="${REPO_SAIDAS:-Olt1z/quantizacao-bl4ck0ut}"
PROVA_ID="${PROVA_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count())"

publicar() {
  marco "publicando em $REPO_SAIDAS/saidas/tp-mla/$PROVA_ID"
  python3 - <<PY || true
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for f in ("bancada.log", "resumo.txt"):
    p = f"/workspace/{f}"
    if os.path.exists(p):
        api.upload_file(path_or_fileobj=p, path_in_repo=f"saidas/tp-mla/$PROVA_ID/{f}", repo_id="$REPO_SAIDAS")
PY
}
trap 'echo "=== FALHOU · linha $LINENO"; publicar' ERR

marco "1. dependências"
apt-get update -qq && apt-get install -y -qq git > /dev/null
pip install -q huggingface_hub

marco "2. fork $FORK @ $BRANCH"
cd /workspace && rm -rf exllamav3-tp-mla
git clone -q -b "$BRANCH" "$FORK" exllamav3-tp-mla && cd exllamav3-tp-mla
git log --oneline -1
pip install -q -r requirements.txt
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
NUCLEOS=$(nproc); export MAX_JOBS=$(( NUCLEOS > 8 ? 8 : (NUCLEOS < 2 ? 2 : NUCLEOS) ))
export TORCH_CUDA_ARCH_LIST="$CAP"
marco "compilando a extensão para sm $CAP com MAX_JOBS=$MAX_JOBS"
pip install -q --no-build-isolation -e . 2>&1 | tail -3
python3 -c "import exllamav3, exllamav3_ext; print('exllamav3', exllamav3.__version__)"

marco "3. corte $CORTE"
python3 - <<PY
import os
from huggingface_hub import snapshot_download
snapshot_download("$CORTE", local_dir="/workspace/corte", token=os.environ["HF_TOKEN"])
PY
du -sh /workspace/corte

set -e
marco "4a. base, uma placa"
CUDA_VISIBLE_DEVICES=0 python3 tests/tp_mla_smoke.py -m /workspace/corte --save /workspace/base.pt | tee /workspace/4a.txt
marco "4b. ruído, uma placa"
CUDA_VISIBLE_DEVICES=0 python3 tests/tp_mla_smoke.py -m /workspace/corte --compare /workspace/base.pt | tee /workspace/4b.txt
set +e
marco "4c. TP, todas as placas"
python3 tests/tp_mla_smoke.py -m /workspace/corte --tp --compare /workspace/base.pt | tee /workspace/4c.txt
echo "4c saiu com $?"
marco "4d. TP com contexto longo (DSA esparso)"
CUDA_VISIBLE_DEVICES=0 python3 tests/tp_mla_smoke.py -m /workspace/corte --prefill-tokens 3000 --save /workspace/base-longo.pt | tee /workspace/4d-base.txt
python3 tests/tp_mla_smoke.py -m /workspace/corte --tp --compare /workspace/base-longo.pt --prefill-tokens 3000 | tee /workspace/4d.txt
echo "4d saiu com $?"

{
  echo "bancada $PROVA_ID · $(nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c | tr '\n' ' ')"
  for f in 4a 4b 4c 4d-base 4d; do echo "--- $f"; grep -E "decode:|uso médio|KL média|OK|FALHOU|Error|error" /workspace/$f.txt | head -8; done
} | tee /workspace/resumo.txt
publicar
marco "FIM"
sleep infinity
