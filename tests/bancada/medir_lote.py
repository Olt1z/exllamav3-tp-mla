"""
Quanto o servidor rende com VÁRIAS conversas ao mesmo tempo.

    python3 medir_lote.py URL TOKEN --paralelas 1,2,4 --prompt 200000 --novos 120

A pergunta que ele responde é a da etapa 15.1 do plano de context parallel: o
decode de um MoE a lote 1 é limitado por LANÇAMENTO, não por banda — 8 de 288
experts por token leem ~3 % dos pesos, e a placa passa o tempo esperando a CPU
enfileirar kernels. Se for isso mesmo, atender quatro conversas juntas custa
quase o mesmo que atender uma, e o agregado cresce quase linear.

Nada disso se mede com uma conversa só, que é tudo o que a bancada tinha até
agora. E não se mede com prompts diferentes: cada rodada usa o MESMO prefixo
para todas as conversas, mudando só um nonce no fim, porque prompts diferentes
misturariam custo de prefill com custo de concorrência.

Sai uma linha por grau de paralelismo:

    paralelas=4  ttft 61,3 s (min 59,9 max 63,1)  decode 41,2 tok/s por conversa
                 agregado 164,8 tok/s  ·  1,00 de eficiência contra o lote 1
"""

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

UA = "bl4ck0ut-bancada/1.0"

FRASE = (
    "A luz do sol atravessa a atmosfera e as moleculas de ar espalham mais os comprimentos de onda "
    "curtos do que os longos, e por isso o ceu parece azul ao meio-dia. "
)


def prompt_de(tokens_alvo: int, nonce: str) -> str:
    """Texto de aproximadamente `tokens_alvo` tokens, com um nonce no FIM.

    O nonce vai no fim de propósito: no começo ele quebraria o cache de prefixo
    de todas as conversas, e aí a medida seria de prefill, não de concorrência.
    Vai depois do texto comum, então as quatro compartilham o prefixo e o
    servidor exercita exatamente o que a produção faz.
    """
    # ~4 caracteres por token no português deste texto; a medida real vem do
    # `prompt_tokens` que o servidor devolve, então a aproximação só dimensiona.
    repeticoes = max(1, (tokens_alvo * 4) // len(FRASE))
    return FRASE * repeticoes + f"\n\nMarcador desta conversa: {nonce}.\n"


def uma_conversa(url: str, token: str, corpo: dict) -> dict:
    pedido = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(corpo).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}", "User-Agent": UA},
    )
    t0 = time.time()
    primeiro = None
    tokens = 0
    with urllib.request.urlopen(pedido, timeout=3600) as r:
        for linha in r:
            linha = linha.decode().strip()
            if not linha.startswith("data: "):
                continue
            dado = linha[6:]
            if dado == "[DONE]":
                break
            try:
                pedaco = json.loads(dado)
            except json.JSONDecodeError:
                continue
            delta = (pedaco.get("choices") or [{}])[0].get("delta", {})
            if delta.get("content") or delta.get("reasoning_content"):
                if primeiro is None:
                    primeiro = time.time()
                tokens += 1
            uso = pedaco.get("usage")
            if uso:
                tokens = uso.get("completion_tokens", tokens)
    fim = time.time()
    return {
        "ttft": (primeiro or fim) - t0,
        "decode_s": fim - (primeiro or fim),
        "tokens": tokens,
        "total_s": fim - t0,
    }


def rodada(url: str, token: str, modelo: str, paralelas: int, prompt_tokens: int, novos: int) -> dict:
    corpos = [
        {
            "model": modelo,
            "messages": [
                {"role": "user", "content": prompt_de(prompt_tokens, f"conversa-{i}-{int(time.time())}")}
            ],
            "max_tokens": novos,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        for i in range(paralelas)
    ]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=paralelas) as pool:
        resultados = list(pool.map(lambda c: uma_conversa(url, token, c), corpos))
    parede = time.time() - t0

    por_conversa = [r["tokens"] / r["decode_s"] if r["decode_s"] > 0 else 0 for r in resultados]
    # O agregado é a soma do que sai na FASE DE DECODE, não tokens/parede: o
    # relógio de parede inclui o prefill, e com uma conversa só ele domina —
    # o que fazia o lote 1 parecer render 9 tok/s onde ele rende 55.
    agregado = sum(por_conversa)
    vazao_parede = sum(r["tokens"] for r in resultados) / parede
    return {
        "paralelas": paralelas,
        "ttft_media": statistics.mean(r["ttft"] for r in resultados),
        "ttft_min": min(r["ttft"] for r in resultados),
        "ttft_max": max(r["ttft"] for r in resultados),
        "decode_por_conversa": statistics.mean(por_conversa),
        "agregado": agregado,
        "vazao_parede": vazao_parede,
        "parede_s": parede,
        "tokens": sum(r["tokens"] for r in resultados),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("token")
    p.add_argument("--modelo", default=None, help="padrão: o primeiro que o servidor listar")
    p.add_argument("--paralelas", default="1,2,4")
    p.add_argument("--prompt", type=int, default=8000, help="tokens de prompt por conversa")
    p.add_argument("--novos", type=int, default=120)
    a = p.parse_args()

    modelo = a.modelo
    if not modelo:
        pedido = urllib.request.Request(
            f"{a.url.rstrip('/')}/v1/models",
            headers={"Authorization": f"Bearer {a.token}", "User-Agent": UA},
        )
        with urllib.request.urlopen(pedido, timeout=60) as r:
            # A primeira entrada pode ser uma pasta do disco (o `lost+found` do
            # volume aparece como modelo): vale a que o servidor tem CARREGADA.
            dados = json.load(r)["data"]
            modelo = next((m["id"] for m in dados if m.get("id") != "lost+found"), dados[0]["id"])
    print(f"modelo: {modelo} · prompt ~{a.prompt} tok · {a.novos} tokens novos por conversa\n", flush=True)

    base = None
    saida = []
    for n in [int(x) for x in a.paralelas.split(",")]:
        try:
            r = rodada(a.url, a.token, modelo, n, a.prompt, a.novos)
        except urllib.error.HTTPError as e:
            print(f"paralelas={n}: HTTP {e.code} — {e.read()[:200]!r}", flush=True)
            continue
        if base is None:
            base = r["decode_por_conversa"]
        # Eficiência contra o IDEAL: n conversas rendendo o que uma rende sozinha.
        eficiencia = r["agregado"] / (base * n) if base else 0
        print(
            f"paralelas={r['paralelas']}  ttft {r['ttft_media']:.1f} s "
            f"(min {r['ttft_min']:.1f} max {r['ttft_max']:.1f})  "
            f"decode {r['decode_por_conversa']:.1f} tok/s por conversa\n"
            f"             agregado {r['agregado']:.1f} tok/s  ·  {eficiencia:.2f} do ideal linear "
            f"·  parede {r['vazao_parede']:.1f} tok/s",
            flush=True,
        )
        saida.append(r)

    print("\n" + json.dumps(saida, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
