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
# BASE_DEVICES: placas da linha de base sem TP. "0" para corte que cabe numa placa; "all" para
# modelo grande, que a base carrega em autosplit (camadas repartidas, uma placa por vez)
BASE_DEVICES="${BASE_DEVICES:-0}"
TOKENS="${TOKENS:-64}"
if [ "$BASE_DEVICES" = "all" ]; then BASE=""; else BASE="CUDA_VISIBLE_DEVICES=$BASE_DEVICES"; fi
REPO_SAIDAS="${REPO_SAIDAS:-Olt1z/quantizacao-bl4ck0ut}"
# O primeiro forward compila Triton em cada rank; num host lento um rank passa dos 90 s padrão
# do coletivo nativo e o grupo aborta ("Synchronization timeout", visto em 06/09 no 4c)
export EXLLAMA_TP_SYNC_TIMEOUT="${EXLLAMA_TP_SYNC_TIMEOUT:-600}"
PROVA_ID="${PROVA_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count())"

publicar() {
  marco "publicando em $REPO_SAIDAS/saidas/tp-mla/$PROVA_ID"
  python3 - <<PY || true
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for f in ("bancada.log", "resumo.txt", "build.log"):
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
pip install -q --no-build-isolation -e . > /workspace/build.log 2>&1; grep -E -i "error|warning: unused|FAILED" /workspace/build.log | grep -v -i "warning" | head -20; tail -2 /workspace/build.log
# o torch vem antes: a extensão liga em libc10.so, que só entra no processo com ele importado
python3 -c "import torch, exllamav3_ext; from exllamav3.version import __version__ as v; print('exllamav3', v, 'ext ok')"

marco "3. corte $CORTE"
python3 - <<PY
import os
from huggingface_hub import snapshot_download
snapshot_download("$CORTE", local_dir="/workspace/corte", token=os.environ["HF_TOKEN"])
PY
du -sh /workspace/corte
# Artefato da esteira antiga com kv_b_proj em treliça: repara antes de carregar
python3 tests/bancada/reparar_kv_b_proj.py /workspace/corte

if [ -z "${SO_7:-}" ] && [ -z "${SO_8:-}" ]; then
marco "3b. teste por bloco: original × importado por TP, bloco a bloco e filho a filho"
env $BASE python3 tests/test_tp_block_import.py /workspace/corte 2>&1 | grep -v -E "it/s\]|━━" | tee /workspace/3b.txt
echo "3b saiu com ${PIPESTATUS[0]}"

set -e
marco "4a. base, uma placa"
env $BASE python3 tests/tp_mla_smoke.py -m /workspace/corte --tokens $TOKENS --save /workspace/base.pt | tee /workspace/4a.txt
marco "4b. base com o fio bf16 simulado (régua justa para saída fp32)"
env $BASE python3 tests/tp_mla_smoke.py -m /workspace/corte --simular-fio-bf16 --compare /workspace/base.pt --save /workspace/sim.pt | tee /workspace/4b.txt
set +e
marco "4c. TP, todas as placas"
# O primeiro processo TP da máquina trava de vez em quando num coletivo do backend nativo (visto
# em 06/09 duas vezes, em hosts e commits diferentes; o segundo processo passa sempre). Prazo
# curto e até três tentativas: o cache do Triton no disco sobrevive entre elas.
for tentativa in 1 2 3; do
  EXLLAMA_TP_SYNC_TIMEOUT=180 python3 tests/tp_mla_smoke.py -m /workspace/corte --tp --compare /workspace/sim.pt --save /workspace/tp.pt | tee /workspace/4c.txt
  RC=${PIPESTATUS[0]}; echo "4c tentativa $tentativa saiu com $RC"
  [ "$RC" -eq 0 ] && break
done
marco "4d. TP com contexto longo (DSA esparso)"
env $BASE python3 tests/tp_mla_smoke.py -m /workspace/corte --simular-fio-bf16 --prefill-tokens 3000 --save /workspace/base-longo.pt | tee /workspace/4d-base.txt
python3 tests/tp_mla_smoke.py -m /workspace/corte --tp --compare /workspace/base-longo.pt --prefill-tokens 3000 | tee /workspace/4d.txt
echo "4d saiu com $?"
marco "5. perfil do prefill por módulo (uma placa) e total em TP"
env $BASE python3 tests/bancada/perfil_prefill.py -m /workspace/corte --prefill-tokens ${PERFIL_TOKENS:-4096} 2>&1 | grep -v -E "it/s\]|━━" | tee /workspace/5a.txt
python3 tests/bancada/perfil_prefill.py -m /workspace/corte --tp --prefill-tokens ${PERFIL_TOKENS:-4096} 2>&1 | grep -v -E "it/s\]|━━" | tee /workspace/5b.txt
echo "5 saiu com $?"

# 6. DFlash 2: leitor do fork contra a referência (z-lab/dflash) e mecânica no gerador. DFLASH2 é o
# repositório do rascunho do MESMO alvo do corte; vazio pula a etapa.
DFLASH2="${DFLASH2:-incoai/GLM-5.3-Flash-DFlash2}"
if [ -n "$DFLASH2" ]; then
  marco "6. DFlash 2 ($DFLASH2) contra a referência"
  pip install -q -U "transformers>=4.51" 2>&1 | tail -1
  python3 - <<PY
import os
from huggingface_hub import snapshot_download
snapshot_download("$DFLASH2", local_dir="/workspace/dflash2", token=os.environ["HF_TOKEN"])
PY
  mkdir -p /workspace/dflash-ref/dflash && curl -sL https://raw.githubusercontent.com/z-lab/dflash/main/dflash/model.py -o /workspace/dflash-ref/dflash/model.py
  python3 tests/test_dflash2_referencia.py --alvo /workspace/corte --dflash2 /workspace/dflash2 --tp \
    --referencia /workspace/dflash-ref/dflash/model.py 2>&1 | grep -v -E "it/s\]|━━" | tee /workspace/6.txt
  echo "6 saiu com ${PIPESTATUS[0]}"
fi

fi

# 7. Etapa 2 do plano de desempenho: prefill pelo GERADOR (como o TabbyAPI), chunk 4096 × 8192,
# uma placa × TP2 (NCCL e nativo), com e sem o rascunho DFlash 2, em 4k/16k/30k. ETAPA2=1 liga;
# SO_7=1 pula 3b–6 e roda só isto.
if [ -n "${ETAPA2:-}" ] || [ -n "${SO_7:-}" ]; then
  marco "7. etapa 2: prefill pelo gerador"
  DFLASH2="${DFLASH2:-incoai/GLM-5.3-Flash-DFlash2}"
  if [ -n "$DFLASH2" ] && [ ! -d /workspace/dflash2 ]; then
    python3 - <<PY7
import os
from huggingface_hub import snapshot_download
snapshot_download("$DFLASH2", local_dir="/workspace/dflash2", token=os.environ["HF_TOKEN"])
PY7
  fi
  TAM="${TAM:-4096,16384,30000}"
  PG="python3 tests/bancada/perfil_gerador.py -m /workspace/corte --tokens $TAM --cache 32768"
  FILTRO="prefill|aceita|aquecimento|Error|error|Traceback|FIM_PERFIL"
  : > /workspace/7.txt
  CUDA_VISIBLE_DEVICES=0 $PG --chunk 4096 2>&1 | grep -E "$FILTRO" | tee -a /workspace/7.txt
  CUDA_VISIBLE_DEVICES=0 $PG --chunk 8192 2>&1 | grep -E "$FILTRO" | tee -a /workspace/7.txt
  $PG --tp --backend nccl --chunk 4096 2>&1 | grep -E "$FILTRO" | tee -a /workspace/7.txt
  $PG --tp --backend nccl --chunk 8192 2>&1 | grep -E "$FILTRO" | tee -a /workspace/7.txt
  $PG --tp --backend native --chunk 4096 2>&1 | grep -E "$FILTRO" | tee -a /workspace/7.txt
  if [ -d /workspace/dflash2 ]; then
    CUDA_VISIBLE_DEVICES=0 $PG --chunk 4096 --dflash2 /workspace/dflash2 --taps "${TAPS:-0,1,2,3,3}" --draft-stats 2>&1 | grep -E "$FILTRO" | tee -a /workspace/7.txt
    $PG --tp --backend nccl --chunk 4096 --dflash2 /workspace/dflash2 --taps "${TAPS:-0,1,2,3,3}" --draft-stats 2>&1 | grep -E "$FILTRO" | tee -a /workspace/7.txt
  fi
  echo "7 terminou"
fi

# 8. Etapa 3 do plano de desempenho: decode em grafo CUDA com projeções fp16 (BC-MLA e BC-KDA).
# A/B no corte, numa placa e em TP2: base eager (grafo desligado) gravada, depois o mesmo decode com o
# grafo ligado e o rastro de montagem, comparado por teacher forcing (KL) e por tok/s. SO_8=1 roda só isto.
if [ -n "${ETAPA3:-}" ] || [ -n "${SO_8:-}" ]; then
  marco "8. etapa 3: grafo CUDA com projeções fp16"
  T8="${TOKENS8:-128}"
  F8="decode:|KL média|OK|FALHOU|Error|error|Traceback|bc_mla|bc_gdn|BC_|build|declin|graph"
  : > /workspace/8.txt
  echo "--- 8a. uma placa, eager (base)" | tee -a /workspace/8.txt
  CUDA_VISIBLE_DEVICES=0 EXL3_BC_ATTN=0 EXL3_BC_GDN=0 python3 tests/tp_mla_smoke.py -m /workspace/corte --tokens $T8 --cache 8192 --save /workspace/eager.pt 2>&1 | grep -E "$F8" | tee -a /workspace/8.txt
  echo "--- 8b. uma placa, grafo ligado, rastro" | tee -a /workspace/8.txt
  CUDA_VISIBLE_DEVICES=0 EXL3_BC_ATTN_TRACE=1 EXL3_BC_GDN_TRACE=1 python3 tests/tp_mla_smoke.py -m /workspace/corte --tokens $T8 --cache 8192 --compare /workspace/eager.pt 2>&1 | grep -E "$F8" | tee -a /workspace/8.txt
  echo "--- 8c. uma placa, grafo, contexto longo (DSA esparso)" | tee -a /workspace/8.txt
  CUDA_VISIBLE_DEVICES=0 EXL3_BC_ATTN=0 EXL3_BC_GDN=0 python3 tests/tp_mla_smoke.py -m /workspace/corte --tokens $T8 --cache 8192 --prefill-tokens 3000 --save /workspace/eager-longo.pt 2>&1 | grep -E "$F8" | tee -a /workspace/8.txt
  CUDA_VISIBLE_DEVICES=0 EXL3_BC_ATTN_TRACE=1 python3 tests/tp_mla_smoke.py -m /workspace/corte --tokens $T8 --cache 8192 --prefill-tokens 3000 --compare /workspace/eager-longo.pt 2>&1 | grep -E "$F8" | tee -a /workspace/8.txt
  echo "--- 8d. TP2 NCCL, eager e grafo" | tee -a /workspace/8.txt
  EXL3_BC_ATTN=0 EXL3_BC_GDN=0 python3 tests/tp_mla_smoke.py -m /workspace/corte --tp --tokens $T8 --cache 8192 --save /workspace/eager-tp.pt 2>&1 | grep -E "$F8" | tee -a /workspace/8.txt
  EXL3_BC_ATTN_TRACE=1 EXL3_BC_GDN_TRACE=1 python3 tests/tp_mla_smoke.py -m /workspace/corte --tp --tokens $T8 --cache 8192 --compare /workspace/eager-tp.pt 2>&1 | grep -E "$F8" | tee -a /workspace/8.txt
  echo "8 terminou"
fi

{
  echo "bancada $PROVA_ID · $(nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c | tr '\n' ' ')"
  echo "--- 3b"; grep -E "== bloco|vs original|Error|error" /workspace/3b.txt | head -40
  for f in 4a 4b 4c 4d-base 4d; do echo "--- $f"; grep -E "decode:|uso médio|KL média|OK|FALHOU|Error|error" /workspace/$f.txt | head -8; done
  for f in 5a 5b; do echo "--- $f"; grep -E "prefill:|decode:|Error|error" /workspace/$f.txt | head -4; done
  echo "--- 6"; grep -E "^A\.|^B\.|rascunho:|referência:|^OK|FALHOU|Error|error" /workspace/6.txt 2>/dev/null | head -12
  echo "--- 7"; cat /workspace/7.txt 2>/dev/null
  echo "--- 8"; cat /workspace/8.txt 2>/dev/null
  echo "--- 5a tabelas"; sed -n '/PREFILL por módulo/,/FIM_PERFIL/p' /workspace/5a.txt | head -60
} | tee /workspace/resumo.txt
publicar
marco "FIM"
sleep infinity
