#!/bin/bash
cd /workspace
M=/workspace/glm-5.3-flash-abliterated-exl3-4.0bpw
echo "### E1: dcp 2, chunk 1024, denso (<2048) e esparso, varios chunks"
/opt/tabby/bin/python -u cercar_cp.py $M --dcp 2 --backend nccl --chunk 1024 --tamanhos 1000,1500,2000,2500,3000,4200 2>&1 | grep -E "^OK|^FIM|carga:|illegal|Error|Traceback" | grep -v "^frame\|TCPStore\|sendBytes" &
P=$!; sleep 400; kill -9 $P 2>/dev/null; for p in $(ps -eo pid,cmd | grep "[c]ercar_cp.py" | awk '{print $1}'); do kill -9 $p; done; sleep 5
echo "### E2: dcp 1 (TP2 puro), chunk 4096"
/opt/tabby/bin/python -u cercar_cp.py $M --dcp 1 --backend nccl --chunk 4096 --tamanhos 4060,4200,8000 2>&1 | grep -E "^OK|^FIM|carga:|illegal|Error|Traceback" | grep -v "^frame\|TCPStore\|sendBytes" &
P=$!; sleep 400; kill -9 $P 2>/dev/null; for p in $(ps -eo pid,cmd | grep "[c]ercar_cp.py" | awk '{print $1}'); do kill -9 $p; done
echo "### fim da serie2"
