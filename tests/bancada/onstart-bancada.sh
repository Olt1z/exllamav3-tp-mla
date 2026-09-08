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
for f in ("bancada.log", "resumo.txt", "build.log", "10.txt", "10-cobertura.txt", "10-convert.txt", "11.txt", "11-solo.txt", "11-p0.txt", "11-p1.txt", "11-duas.txt", "12.txt", "13.txt", "14.txt", "15.txt"):
    p = f"/workspace/{f}"
    if os.path.exists(p):
        api.upload_file(path_or_fileobj=p, path_in_repo=f"saidas/tp-mla/$PROVA_ID/{f}", repo_id="$REPO_SAIDAS")
PY
}
trap 'echo "=== FALHOU · linha $LINENO"; publicar' ERR

marco "1. dependências"
apt-get update -qq && apt-get install -y -qq git > /dev/null
pip install -q huggingface_hub

# ---------------------------------------------------------------------------------------------
# 15. Quanto custa um all-reduce do TP, isolado do modelo. Só precisa do torch da imagem: não
# compila a extensão nem baixa o corte, então a máquina vive ~5 min em vez de 40. SO_15=1 roda só
# isto; duas placas ou mais. HIDDEN/CAMADAS descrevem o modelo que se quer extrapolar (o
# GLM-5.3-Flash tem hidden 4096 e 45 camadas, ou seja 90 all-reduces por token).
if [ -n "${SO_15:-}" ]; then
  marco "15. custo do all-reduce (NCCL), sem modelo"
  cd /workspace && rm -rf fork-ar && git clone -q --depth 1 -b "$BRANCH" "$FORK" fork-ar && cd fork-ar
  git log --oneline -1
  N=$(nvidia-smi --list-gpus | wc -l)
  # a topologia decide se a coletiva anda entre placas ou desce ao host
  { echo "=== topologia"; nvidia-smi topo -m 2>&1 | head -20; } | tee /workspace/15.txt
  for placas in $(seq 2 "$N"); do
    { echo; echo "=== $placas placas"; } | tee -a /workspace/15.txt
    torchrun --nproc_per_node="$placas" tests/bancada/medir_allreduce.py \
      --hidden "${HIDDEN:-4096}" --camadas "${CAMADAS:-90}" 2>&1 | tee -a /workspace/15.txt
  done
  publicar
  marco "FIM"
  exit 0
fi


marco "2. fork $FORK @ $BRANCH"
cd /workspace && rm -rf exllamav3-tp-mla
git clone -q -b "$BRANCH" "$FORK" exllamav3-tp-mla && cd exllamav3-tp-mla
git log --oneline -1
pip install -q -r requirements.txt
# A 1.4.8 exige setuptools >= 77 no pyproject; com --no-build-isolation vale o da imagem, que é mais velho
pip install -q -U "setuptools>=77" wheel
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
NUCLEOS=$(nproc); export MAX_JOBS=$(( NUCLEOS > 8 ? 8 : (NUCLEOS < 2 ? 2 : NUCLEOS) ))
export TORCH_CUDA_ARCH_LIST="$CAP"
marco "compilando a extensão para sm $CAP com MAX_JOBS=$MAX_JOBS"
pip install -q --no-build-isolation -e . > /workspace/build.log 2>&1; { grep -E -i "error|FAILED" /workspace/build.log | grep -v -i "warning" | head -20; tail -2 /workspace/build.log; } || true
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

if [ -z "${SO_7:-}" ] && [ -z "${SO_8:-}" ] && [ -z "${SO_9:-}" ] && [ -z "${SO_10:-}" ] && [ -z "${SO_11:-}" ] && [ -z "${SO_12:-}" ] && [ -z "${SO_13:-}" ] && [ -z "${SO_14:-}" ]; then
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
  F8="decode:|KL média|OK|FALHOU|Error|error|Traceback|BC-|Graph update"
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

# 9. Etapa 8 do plano: experts na RAM (offload nativo do ExLlamaV3), no corte em mul1, uma placa,
# modo layer-split (as flags recusam TP). Base tudo na placa; EXL3_MOE_CPU_OFFLOAD=1 (a camada MoE
# inteira na CPU); EXL3_MOE_CPU_SPLIT=N (os N experts de cauda de cada camada na CPU). Mede prefill
# 4k/16k/30k e decode pelo gerador. SO_9=1 roda só isto.
if [ -n "${ETAPA8:-}" ] || [ -n "${SO_9:-}" ]; then
  marco "9. etapa 8: experts na RAM"
  lscpu | grep -E "Model name|^CPU\(s\)|Thread|avx512" | head -4; free -g | head -2
  TAM9="${TAM9:-4096,16384,30000}"
  PG9="python3 tests/bancada/perfil_gerador.py -m /workspace/corte --tokens $TAM9 --cache 32768 --novos 128"
  F9="prefill|decode|Error|error|Traceback|FIM_PERFIL|mul1|offload|split|CPU|tok/s"
  : > /workspace/9.txt
  # V9: variantes separadas por ':' (a Vast rejeitou em silêncio o env inteiro, e o onstart junto, com vírgulas no valor em 07/09) ou vírgula; "placa" = tudo na GPU (não cabe no Flash inteiro numa placa)
  IFS=',:' read -ra VARIANTES9 <<< "${V9:-placa,EXL3_MOE_CPU_OFFLOAD=1,EXL3_MOE_CPU_SPLIT=64,EXL3_MOE_CPU_SPLIT=128,EXL3_MOE_CPU_SPLIT=192,EXL3_MOE_CPU_SPLIT=256}"
  for V in "${VARIANTES9[@]}"; do
    [ "$V" = "placa" ] && V=""
    echo "--- ${V:-tudo na placa}" | tee -a /workspace/9.txt
    env CUDA_VISIBLE_DEVICES=0 $V $PG9 --rotulo "${V:-placa}" 2>&1 | grep -E "$F9" | tee -a /workspace/9.txt
    nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1 | tee -a /workspace/9.txt; free -g | sed -n 2p | tee -a /workspace/9.txt
  done
  echo "9 terminou"
fi

# 10. Etapa 9 do plano: conversor nativo (convert.py --recipe) no corte, contra o artefato da esteira
# GPTQModel ($CORTE) e contra o BF16 de origem ($ORIGEM), mesma receita e mesmo corpus do repositório
# privado da quantização. Mede tempo por camada, manifest e KL. SO_10=1 roda só isto; uma placa.
if [ -n "${ETAPA9:-}" ] || [ -n "${SO_10:-}" ]; then
  marco "10. etapa 9: conversor nativo no corte"
  ORIGEM="${ORIGEM:-Olt1z/GLM-5.3-Flash-podado-4L-BF16}"
  REPO_QUANT="${REPO_QUANT:-Olt1z/quantizacao-bl4ck0ut}"
  RECEITA_ID="${RECEITA_ID:-cmtqbp54o006c1jgr2mi3ux0p}"
  SAIDA_NATIVO="${SAIDA_NATIVO:-Olt1z/GLM-5.3-Flash-podado-4L-EXL3-nativo-4bpw-bl4ck0ut}"
  pip install -q pyyaml
  free -g | head -2; df -h /workspace | tail -1
  python3 - <<PY
import os
from huggingface_hub import snapshot_download, hf_hub_download
snapshot_download("$ORIGEM", local_dir="/workspace/origem", token=os.environ["HF_TOKEN"])
d = "/workspace/quantizacao"
for f in ["scripts/receita_para_nativo.py", "scripts/manifest_exl3.py", "cal_bl4ck0ut.safetensors", "saidas/$RECEITA_ID/receita.json"]:
    hf_hub_download("$REPO_QUANT", f, local_dir=d, token=os.environ["HF_TOKEN"])
PY
  Q=/workspace/quantizacao; mkdir -p /workspace/work
  python3 $Q/scripts/receita_para_nativo.py --autoteste
  python3 $Q/scripts/receita_para_nativo.py --receita $Q/saidas/$RECEITA_ID/receita.json --modelo /workspace/origem --saida /workspace/work/recipe.yaml 2>&1 | tee /workspace/10-cobertura.txt | tail -12
  source /workspace/work/motor.env
  marco "10b. convert.py"
  T0=$(date +%s)
  # DEVICES9=0,1 quantiza com mais de uma placa (mede a escala do paralelismo por grupo e da calibração)
  DEVICES9="${DEVICES9:-0}"
  CUDA_VISIBLE_DEVICES=$DEVICES9 TERM=dumb COLUMNS=200 EXL3_CONVERT_TIMING=1 python3 convert.py -i /workspace/origem -w /workspace/work -o /workspace/exl3 \
    --recipe /workspace/work/recipe.yaml --codebook mul1 --devices "$DEVICES9" --cal_data $Q/cal_bl4ck0ut.safetensors \
    -cr 503 -cc 2048 -hb "$HEAD_BITS" -mb "$MTP_BITS" -vb "$VISION_BITS" -cpi 900 2>&1 | tee /workspace/10-convert.txt | grep -E "layers\.[0-9]+ +bpw|Unquantized|Estimated|!!|##|Error|error|All done"
  echo "convert.py: $(( $(date +%s) - T0 )) s no total" | tee -a /workspace/10.txt
  du -sh /workspace/exl3 | tee -a /workspace/10.txt
  grep -a -E "Timing model|Timing group|Timing batch|layers\.[0-9]+ +bpw" /workspace/10-convert.txt | sort | uniq -c | sort -rn | head -40 | cut -c1-200 | tee -a /workspace/10.txt
  # PROVA9_SO_CONVERT=1: tempo por fase, manifest e KL contra o BF16, sem comparar com a esteira antiga nem publicar
  if [ -n "${PROVA9_SO_CONVERT:-}" ]; then
    python3 $Q/scripts/manifest_exl3.py --artefato /workspace/exl3 --origem /workspace/origem --receita $Q/saidas/$RECEITA_ID/receita.json --saida /workspace/exl3/manifest.json 2>&1 | grep -E "aprovado|violac|g_sc" | tee -a /workspace/10.txt
    CUDA_VISIBLE_DEVICES=0 python3 tests/tp_mla_smoke.py -m /workspace/origem --tokens 128 --cache 8192 --save /workspace/bf16.pt 2>&1 | grep -E "KL|Error|error" | tee -a /workspace/10.txt
    echo "--- nativo vs BF16" | tee -a /workspace/10.txt
    CUDA_VISIBLE_DEVICES=0 python3 tests/tp_mla_smoke.py -m /workspace/exl3 --tokens 128 --cache 8192 --compare /workspace/bf16.pt 2>&1 | grep -E "KL|OK|FALHOU|Error|error" | tee -a /workspace/10.txt
    echo "10 terminou"
  fi
  if [ -z "${PROVA9_SO_CONVERT:-}" ]; then
  marco "10c. manifest"
  python3 $Q/scripts/manifest_exl3.py --artefato /workspace/exl3 --origem /workspace/origem --receita $Q/saidas/$RECEITA_ID/receita.json --saida /workspace/exl3/manifest.json 2>&1 | tail -15 | tee -a /workspace/10.txt
  marco "10d. KL: BF16 → nativo e → esteira GPTQModel"
  python3 tests/bancada/reparar_nomes.py /workspace/corte || true
  python3 tests/bancada/reparar_qkv_kda.py /workspace/corte "$ORIGEM" || true
  T10="${T10:-128}"
  CUDA_VISIBLE_DEVICES=0 python3 tests/tp_mla_smoke.py -m /workspace/origem --tokens $T10 --cache 8192 --save /workspace/bf16.pt 2>&1 | grep -E "decode:|KL|OK|FALHOU|Error|error" | tee -a /workspace/10.txt
  echo "--- nativo vs BF16" | tee -a /workspace/10.txt
  CUDA_VISIBLE_DEVICES=0 python3 tests/tp_mla_smoke.py -m /workspace/exl3 --tokens $T10 --cache 8192 --compare /workspace/bf16.pt --save /workspace/nativo.pt 2>&1 | grep -E "decode:|KL|OK|FALHOU|Error|error" | tee -a /workspace/10.txt
  echo "--- esteira GPTQModel vs BF16" | tee -a /workspace/10.txt
  CUDA_VISIBLE_DEVICES=0 python3 tests/tp_mla_smoke.py -m /workspace/corte --tokens $T10 --cache 8192 --compare /workspace/bf16.pt 2>&1 | grep -E "decode:|KL|OK|FALHOU|Error|error" | tee -a /workspace/10.txt
  echo "--- esteira GPTQModel vs nativo" | tee -a /workspace/10.txt
  CUDA_VISIBLE_DEVICES=0 python3 tests/tp_mla_smoke.py -m /workspace/corte --tokens $T10 --cache 8192 --compare /workspace/nativo.pt 2>&1 | grep -E "decode:|KL|OK|FALHOU|Error|error" | tee -a /workspace/10.txt
  marco "10e. publicar $SAIDA_NATIVO"
  python3 - <<PY
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo("$SAIDA_NATIVO", exist_ok=True)
api.upload_folder(folder_path="/workspace/exl3", repo_id="$SAIDA_NATIVO", commit_message="corte 4L do Flash em EXL3 mul1 pelo convert.py nativo, receita do hub, prova da etapa 9 ($PROVA_ID)")
PY
  echo "10 terminou"
  fi
fi

# 11. Escala em várias placas: dois conversores INDEPENDENTES ao mesmo tempo, um por placa, no corte.
# Compara o tempo por grupo com o de uma placa sozinha (5,4 s) e com o do conversor único em 2 placas
# (7–8,5 s): se ficar em 5,4 s a lentidão é disputa de host entre threads (processo por placa resolve);
# se subir junto, é hardware. SO_11=1 roda só isto; precisa de 2 placas e RAM para dois estados (2 × 67 GB
# a 503 linhas, então usa CR11 linhas, padrão 250).
if [ -n "${SO_11:-}" ]; then
  marco "11. dois conversores concorrentes, um por placa"
  ORIGEM="${ORIGEM:-Olt1z/GLM-5.3-Flash-podado-4L-BF16}"
  REPO_QUANT="${REPO_QUANT:-Olt1z/quantizacao-bl4ck0ut}"
  RECEITA_ID="${RECEITA_ID:-cmtqbp54o006c1jgr2mi3ux0p}"
  CR11="${CR11:-250}"
  pip install -q pyyaml
  free -g | head -2; nvidia-smi --query-gpu=name,memory.total,power.limit --format=csv,noheader
  python3 - <<PY
import os
from huggingface_hub import snapshot_download, hf_hub_download
snapshot_download("$ORIGEM", local_dir="/workspace/origem", token=os.environ["HF_TOKEN"])
for f in ["scripts/receita_para_nativo.py", "cal_bl4ck0ut.safetensors", "saidas/$RECEITA_ID/receita.json"]:
    hf_hub_download("$REPO_QUANT", f, local_dir="/workspace/quantizacao", token=os.environ["HF_TOKEN"])
PY
  Q=/workspace/quantizacao; mkdir -p /workspace/work
  python3 $Q/scripts/receita_para_nativo.py --receita $Q/saidas/$RECEITA_ID/receita.json --modelo /workspace/origem --saida /workspace/work/recipe.yaml > /workspace/11-cobertura.txt 2>&1
  source /workspace/work/motor.env
  conv() { # $1 placa  $2 rótulo  $3 devices locais
    CUDA_VISIBLE_DEVICES=$1 TERM=dumb COLUMNS=200 EXL3_CONVERT_TIMING=1 python3 convert.py -i /workspace/origem -w /workspace/work-$2 -o /workspace/exl3-$2 \
      --recipe /workspace/work/recipe.yaml --codebook mul1 --devices $3 --cal_data $Q/cal_bl4ck0ut.safetensors \
      -cr $CR11 -cc 2048 -hb "$HEAD_BITS" -mb "$MTP_BITS" -vb "$VISION_BITS" -cpi 3600 > /workspace/11-$2.txt 2>&1
  }
  marco "11a. uma placa sozinha (referência)"
  T0=$(date +%s); conv 0 solo 0; echo "solo: $(( $(date +%s) - T0 )) s" | tee -a /workspace/11.txt
  marco "11b. dois processos ao mesmo tempo, placas 0 e 1"
  T0=$(date +%s); conv 0 p0 0 & P0=$!; conv 1 p1 0 & P1=$!; wait $P0; wait $P1; echo "dois processos: $(( $(date +%s) - T0 )) s" | tee -a /workspace/11.txt
  marco "11c. um processo com as duas placas"
  T0=$(date +%s); conv 0,1 duas 0,1; echo "um processo, duas placas: $(( $(date +%s) - T0 )) s" | tee -a /workspace/11.txt
  for r in solo p0 p1 duas; do
    echo "--- $r" | tee -a /workspace/11.txt
    grep -a -E "Timing model.language_model.layers" /workspace/11-$r.txt | cut -c1-200 | tee -a /workspace/11.txt
    grep -a "Timing group" /workspace/11-$r.txt | grep "layers.3" | awk "{print \$NF}" | sort | uniq -c | sort -rn | head -4 | tee -a /workspace/11.txt
  done
  echo "11 terminou"
fi

# 12. Etapa 1 do plano "experts na RAM com tensor parallel": o teto do worker de CPU. Uma placa,
# corte em mul1, modo layer-split. (a) o host: CPU, NUMA, RAM; (b) banda crua de leitura da RAM
# por número de threads (torch, soma de 4 GiB); (c) perfil por fase do kernel de CPU
# (EXL3_MOE_CPU_PROF=1 imprime a média a cada 512 jobs: prep_gu, gemv_gu, act+prep_d, gemv_d,
# tf_d, accum) por fração de experts na CPU, com colocação ESTÁTICA (EXL3_MOE_CPU_SWAP=0) para a
# conta de bytes por token fechar: 8 ativos × N/288 na CPU; (d) varredura de threads numa fração
# fixa; (e) a mesma fração com a colocação dinâmica ligada, para medir o que o swap rende.
# SO_12=1 roda só isto. V12 separa variantes por ':' (vírgula derruba o env na Vast).
if [ -n "${SO_12:-}" ]; then
  marco "12. teto do worker de CPU"
  : > /workspace/12.txt
  { lscpu | grep -E "Model name|^CPU\(s\)|Thread\(s\) per core|Socket\(s\)|NUMA node\(s\)|L3 cache"; free -g | head -2; grep -E "Hugepagesize" /proc/meminfo; } | tee -a /workspace/12.txt
  NUC=$(nproc)
  echo "--- banda crua da RAM: torch, soma de 4 GiB float32, melhor de 5, GB/s por threads" | tee -a /workspace/12.txt
  python3 - <<PY 2>&1 | tee -a /workspace/12.txt
import os, time, torch
n = 4 * 1024**3 // 4
x = torch.ones(n, dtype=torch.float32); x += 1
nuc = os.cpu_count() or 1
for t in sorted({max(1, nuc // 8), max(1, nuc // 4), max(1, nuc // 2), max(1, nuc * 3 // 4), nuc}):
    torch.set_num_threads(t); x.sum(); best = 1e9
    for _ in range(5):
        t0 = time.perf_counter(); x.sum(); best = min(best, time.perf_counter() - t0)
    print(f"threads {t:>3}: {4 / best:6.1f} GB/s")
PY
  TAM12="${TAM12:-4096}"; NOVOS12="${NOVOS12:-1100}"
  PG12="python3 tests/bancada/perfil_gerador.py -m /workspace/corte --tokens $TAM12 --cache 8192 --novos $NOVOS12"
  F12="decode|moe_cpu prof|worker started|arena: new|swap sweep|Error|error|Traceback|FIM_PERFIL"
  IFS=',:' read -ra V12 <<< "${V12:-placa:EXL3_MOE_CPU_SPLIT=32:EXL3_MOE_CPU_SPLIT=64:EXL3_MOE_CPU_SPLIT=128:EXL3_MOE_CPU_SPLIT=192:EXL3_MOE_CPU_SPLIT=256:EXL3_MOE_CPU_OFFLOAD=1}"
  for V in "${V12[@]}"; do
    [ "$V" = "placa" ] && V=""
    echo "--- ${V:-tudo na placa} (colocação estática)" | tee -a /workspace/12.txt
    env CUDA_VISIBLE_DEVICES=0 EXL3_MOE_CPU_PROF=1 EXL3_MOE_CPU_SWAP=0 $V $PG12 --rotulo "${V:-placa}" 2>&1 | grep -E "$F12" | tee -a /workspace/12.txt
  done
  for T in $(( NUC / 4 )) $(( NUC / 2 )) $(( NUC * 3 / 4 )) $NUC; do
    [ "$T" -lt 1 ] && continue
    echo "--- EXL3_MOE_CPU_SPLIT=144 threads=$T (colocação estática)" | tee -a /workspace/12.txt
    env CUDA_VISIBLE_DEVICES=0 EXL3_MOE_CPU_PROF=1 EXL3_MOE_CPU_SWAP=0 EXL3_MOE_CPU_SPLIT=144 EXL3_MOE_CPU_THREADS=$T $PG12 --rotulo "split144-t$T" 2>&1 | grep -E "$F12" | tee -a /workspace/12.txt
  done
  echo "--- EXL3_MOE_CPU_SPLIT=144 (colocação dinâmica, padrão)" | tee -a /workspace/12.txt
  env CUDA_VISIBLE_DEVICES=0 EXL3_MOE_CPU_PROF=1 EXL3_MOE_CPU_SWAP_DEBUG=1 EXL3_MOE_CPU_SPLIT=144 $PG12 --rotulo "split144-swap" 2>&1 | grep -E "$F12" | tee -a /workspace/12.txt
  echo "12 terminou"
fi

# 13. Etapa 2 do plano "experts na RAM com tensor parallel": o remendo, no modo de canais
# (moe_tensor_split). Duas placas, corte em mul1. 13a base numa placa; 13b split numa placa
# (caminho que já existia) contra 13a; 13c TP2 canais sem split contra 13a pelo fio bf16;
# 13d TP2 canais COM split (o remendo) contra 13a — o worker tem de subir UMA vez, no rank de
# saída; 13e TP2 expert-parallel com split: tem de AVISAR e carregar inteiro (a armadilha que
# antes era silenciosa); 13f decode pelo gerador nos três arranjos. SO_13=1 roda só isto.
if [ -n "${SO_13:-}" ]; then
  marco "13. TP + experts na RAM (modo de canais)"
  : > /workspace/13.txt
  K13="${K13:-144}"; TOK13="${TOK13:-128}"
  SM="python3 tests/tp_mla_smoke.py -m /workspace/corte --tokens $TOK13 --cache 4096"
  F13="decode:|uso médio|KL média|^OK|FALHOU|Error|error|Traceback|CPU split experts|CPU MoE worker|ignored|skipped"
  echo "--- 13a. base, uma placa, tudo na placa" | tee -a /workspace/13.txt
  env CUDA_VISIBLE_DEVICES=0 $SM --save /workspace/13-base.pt 2>&1 | grep -E "$F13" | tee -a /workspace/13.txt
  echo "--- 13b. uma placa, EXL3_MOE_CPU_SPLIT=$K13 (caminho antigo), KL contra 13a" | tee -a /workspace/13.txt
  env CUDA_VISIBLE_DEVICES=0 EXL3_MOE_CPU_SWAP=0 EXL3_MOE_CPU_SPLIT=$K13 $SM --compare /workspace/13-base.pt 2>&1 | grep -E "$F13" | tee -a /workspace/13.txt
  echo "--- 13c. TP2 canais, tudo na placa, KL contra 13a (fio bf16)" | tee -a /workspace/13.txt
  $SM --tp --tp-moe-ts --compare /workspace/13-base.pt --simular-fio-bf16 2>&1 | grep -E "$F13" | tee -a /workspace/13.txt
  echo "--- 13d. TP2 canais, EXL3_MOE_CPU_SPLIT=$K13 (o remendo), KL contra 13a (fio bf16); worker UMA vez" | tee -a /workspace/13.txt
  EXL3_MOE_CPU_SWAP=0 EXL3_MOE_CPU_SPLIT=$K13 $SM --tp --tp-moe-ts --compare /workspace/13-base.pt --simular-fio-bf16 2>&1 | grep -E "$F13" | tee -a /workspace/13.txt
  echo "--- 13e. TP2 expert-parallel, EXL3_MOE_CPU_SPLIT=$K13: deve avisar e carregar inteiro" | tee -a /workspace/13.txt
  EXL3_MOE_CPU_SPLIT=$K13 $SM --tp --compare /workspace/13-base.pt --simular-fio-bf16 2>&1 | grep -E "$F13" | tee -a /workspace/13.txt
  echo "--- 13f. decode pelo gerador" | tee -a /workspace/13.txt
  PG13="python3 tests/bancada/perfil_gerador.py -m /workspace/corte --tokens 4096 --cache 8192 --novos 256"
  G13="decode|worker started|CPU split|Error|Traceback"
  env CUDA_VISIBLE_DEVICES=0 EXL3_MOE_CPU_SWAP=0 EXL3_MOE_CPU_SPLIT=$K13 $PG13 --rotulo "1placa-split$K13" 2>&1 | grep -E "$G13" | tee -a /workspace/13.txt
  $PG13 --tp --tp-moe-ts --rotulo "tp2-canais" 2>&1 | grep -E "$G13" | tee -a /workspace/13.txt
  EXL3_MOE_CPU_SWAP=0 EXL3_MOE_CPU_SPLIT=$K13 $PG13 --tp --tp-moe-ts --rotulo "tp2-canais-split$K13" 2>&1 | grep -E "$G13" | tee -a /workspace/13.txt
  echo "13 terminou"
fi

# 14. Etapa 3 do plano: cache quantizado no latente da MLA (Q8/Q6/Q4), qualidade contra fp16 com
# prefill longo (o erro do cache cresce com o contexto), numa placa e em TP2 canais; e o arranjo
# alvo, TP2 canais + Q8 + split. O corte tem UMA camada MLA, então VRAM do cache não é mensurável
# aqui — só a KL e o tamanho que a camada declara. SO_14=1 roda só isto; duas placas.
if [ -n "${SO_14:-}" ]; then
  marco "14. cache quantizado na MLA"
  : > /workspace/14.txt
  K14="${K14:-144}"; PF14="${PF14:-3000}"
  SM="python3 tests/tp_mla_smoke.py -m /workspace/corte --tokens 128 --cache 8192 --prefill-tokens $PF14"
  F14="^cache:|decode:|uso médio|KL média|^OK|FALHOU|Error|error|Traceback|CPU split experts|ignored|skipped"
  echo "--- 14a. base fp16, uma placa, prefill $PF14" | tee -a /workspace/14.txt
  env CUDA_VISIBLE_DEVICES=0 $SM --save /workspace/14-base.pt 2>&1 | grep -E "$F14" | tee -a /workspace/14.txt
  for B in 8 6 4; do
    echo "--- 14b. uma placa, cache Q$B, KL contra 14a" | tee -a /workspace/14.txt
    env CUDA_VISIBLE_DEVICES=0 $SM --cache-bits $B --compare /workspace/14-base.pt 2>&1 | grep -E "$F14" | tee -a /workspace/14.txt
  done
  echo "--- 14c. TP2 canais, cache fp16, KL contra 14a (fio bf16)" | tee -a /workspace/14.txt
  $SM --tp --tp-moe-ts --compare /workspace/14-base.pt --simular-fio-bf16 2>&1 | grep -E "$F14" | tee -a /workspace/14.txt
  echo "--- 14d. TP2 canais, cache Q8, KL contra 14a (fio bf16)" | tee -a /workspace/14.txt
  $SM --tp --tp-moe-ts --cache-bits 8 --compare /workspace/14-base.pt --simular-fio-bf16 2>&1 | grep -E "$F14" | tee -a /workspace/14.txt
  echo "--- 14e. TP2 canais, cache Q8, EXL3_MOE_CPU_SPLIT=$K14: o arranjo alvo" | tee -a /workspace/14.txt
  EXL3_MOE_CPU_SWAP=0 EXL3_MOE_CPU_SPLIT=$K14 $SM --tp --tp-moe-ts --cache-bits 8 --compare /workspace/14-base.pt --simular-fio-bf16 2>&1 | grep -E "$F14" | tee -a /workspace/14.txt
  echo "--- 14f. modo de canais pelo AMBIENTE (EXL3_TP_MOE_TENSOR_SPLIT=1, sem --tp-moe-ts), Q8 + split: o caminho do TabbyAPI" | tee -a /workspace/14.txt
  EXL3_TP_MOE_TENSOR_SPLIT=1 EXL3_MOE_CPU_SWAP=0 EXL3_MOE_CPU_SPLIT=$K14 $SM --tp --cache-bits 8 --compare /workspace/14-base.pt --simular-fio-bf16 2>&1 | grep -E "$F14" | tee -a /workspace/14.txt
  echo "--- 14g. gerador: TP2 canais Q8 + split, prompts 4k e 16k" | tee -a /workspace/14.txt
  EXL3_MOE_CPU_SWAP=0 EXL3_MOE_CPU_SPLIT=$K14 python3 tests/bancada/perfil_gerador.py -m /workspace/corte --tp --tp-moe-ts --cache-bits 8 --tokens 4096,16384 --cache 32768 --novos 256 --rotulo "tp2-q8-split$K14" 2>&1 | grep -E "prefill|decode|worker started|Error|Traceback" | tee -a /workspace/14.txt
  echo "14 terminou"
fi

{
  echo "bancada $PROVA_ID · $(nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c | tr '\n' ' ')"
  echo "--- 3b"; grep -E "== bloco|vs original|Error|error" /workspace/3b.txt | head -40
  for f in 4a 4b 4c 4d-base 4d; do echo "--- $f"; grep -E "decode:|uso médio|KL média|OK|FALHOU|Error|error" /workspace/$f.txt | head -8; done
  for f in 5a 5b; do echo "--- $f"; grep -E "prefill:|decode:|Error|error" /workspace/$f.txt | head -4; done
  echo "--- 6"; grep -E "^A\.|^B\.|rascunho:|referência:|^OK|FALHOU|Error|error" /workspace/6.txt 2>/dev/null | head -12
  echo "--- 7"; cat /workspace/7.txt 2>/dev/null
  echo "--- 8"; cat /workspace/8.txt 2>/dev/null
  echo "--- 9"; cat /workspace/9.txt 2>/dev/null
  echo "--- 10"; cat /workspace/10.txt 2>/dev/null
  echo "--- 11"; cat /workspace/11.txt 2>/dev/null
  echo "--- 12"; cat /workspace/12.txt 2>/dev/null
  echo "--- 13"; cat /workspace/13.txt 2>/dev/null
  echo "--- 14"; cat /workspace/14.txt 2>/dev/null
  echo "--- 5a tabelas"; sed -n '/PREFILL por módulo/,/FIM_PERFIL/p' /workspace/5a.txt | head -60
} | tee /workspace/resumo.txt
publicar
marco "FIM"
sleep infinity
