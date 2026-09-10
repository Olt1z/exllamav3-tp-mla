#!/bin/bash
# supervisor: relanca o TabbyAPI sempre que ele cair (para a bancada, nao para producao)
L=$(grep "^/opt/tabby/bin/python main.py" /root/onstart.sh | head -1)
cd /opt/tabbyapi
export EXL3_DCP=2
while true; do
  echo "=== supervisor: subindo o TabbyAPI $(date -u +%H:%M:%S) ==="
  eval "$L"
  echo "=== supervisor: TabbyAPI saiu com $? ; 10 s e relanca ==="
  sleep 10
done
