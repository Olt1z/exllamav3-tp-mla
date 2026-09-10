#!/bin/bash
# espera o servidor, roda a serie crescente; a primeira queda derruba o servidor e encerra
K=$(grep "^api_key:" /opt/tabbyapi/api_tokens.yml | awk '{print $2}')
cd /workspace
for i in $(seq 1 60); do curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:5000/v1/models -H "Authorization: Bearer $K" | grep -q 200 && break; sleep 5; done
echo "serie inicio $(date -u +%H:%M:%S)"
for n in 8000 32000 64000 100000 131000 150000 200000; do
  echo "=== alvo $n"
  python3 -u recuperacao_longa.py http://127.0.0.1:5000 "$K" --tokens $n --max-tokens 512 --saida /workspace/serie-$n.json 2>&1 | grep -vE "^\s*$|^\s*\^|^  File|^    "
  code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:5000/v1/models -H "Authorization: Bearer $K")
  echo "servidor depois de $n: http=$code"
  [ "$code" = "200" ] || { echo "SERVIDOR CAIU em $n"; break; }
done
echo "serie fim $(date -u +%H:%M:%S)"
