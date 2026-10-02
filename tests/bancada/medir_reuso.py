"""
Régua do reaproveitamento de prompt (EXL3_REPLAY, EXL3_PONTOS_DE_GUARDA; ver exllamav3/generator/reuso.py).

    python tests/bancada/medir_reuso.py http://127.0.0.1:5000 $TABBY_API_TOKEN [--log /caminho/tabby.log]

Rode duas vezes, com o TabbyAPI subido sem e com os interruptores (eles são lidos uma vez, na carga):

    EXL3_REPLAY=0 EXL3_PONTOS_DE_GUARDA=0 ...   # antes
    EXL3_REPLAY=1 EXL3_PONTOS_DE_GUARDA=1 ...   # depois

Mede o que o README do TensorFold mede (prompt idêntico de 64k: 34 s -> 0,07 s; conversa nova com o
mesmo prompt de sistema de 7,9k: 4,24 s -> 0,13 s):

  idêntico   o mesmo prompt de ~64k duas vezes; a segunda é a nova tentativa do proxy. Confere que a
             saída em greedy é a mesma (o replay não pode mudar a resposta)
  sistema    três conversas novas com o mesmo prompt de sistema de ~7,9k. A primeira traz uma mensagem
             longa (~30k, como um agente que já leu arquivos): os checkpoints dela caem a cada 32.768
             tokens, nenhum dentro do sistema. As outras, perguntas curtas, deveriam retomar do fim do
             sistema (o primeiro <|user|>) em vez de refazer os ~7,9k

Tempo até o 1º token pelo relógio do cliente (localhost). Com --log, também o "cached" que o servidor
escreve na linha Metrics. Cada rodada leva um nonce no começo: nada de rodadas anteriores serve.
"""
import argparse, json, sys, time, urllib.request, uuid

sys.path.insert(0, __import__("os").path.dirname(__file__))
from medir_tabby import registros, ultima_metrics  # noqa: E402

TOK_POR_REGISTRO = 31           # ~31 tokens por registro no tokenizador do GLM-5.3-Flash


def chat(url, chave, mensagens, max_tokens):
    corpo = json.dumps({
        "model": "x", "messages": mensagens, "max_tokens": max_tokens, "temperature": 0, "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions", data = corpo,
        headers = {"Authorization": f"Bearer {chave}", "Content-Type": "application/json"},
    )
    t0 = time.time()
    t_primeiro, saida, uso = None, [], {}
    with urllib.request.urlopen(req, timeout = 3600) as r:
        for linha in r:
            linha = linha.decode().strip()
            if not linha.startswith("data:") or linha == "data: [DONE]":
                continue
            ev = json.loads(linha[5:])
            uso = ev.get("usage") or uso
            for c in ev.get("choices", []):
                d = c.get("delta", {})
                pedaco = (d.get("reasoning_content") or "") + (d.get("content") or "")
                if pedaco:
                    t_primeiro = t_primeiro or time.time()
                    saida.append(pedaco)
    return "".join(saida), uso, (t_primeiro or time.time()) - t0


def medir(nome, args, mensagens, max_tokens):
    pos = 0
    if args.log:
        with open(args.log, "rb") as f:
            f.seek(0, 2)
            pos = f.tell()
    texto, uso, ttft = chat(args.url, args.chave, mensagens, max_tokens)
    cache = None
    if args.log:
        time.sleep(0.5)
        m = ultima_metrics(args.log, pos)
        cache = m and m["cache_tok"]
    print(f"{nome:<22} entrada {uso.get('prompt_tokens')!s:>7} tok · 1º token {ttft:7.3f} s"
          f"{f' · cached {cache:.0f}' if cache is not None else ''}", flush = True)
    return texto, ttft


def main():
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("chave")
    p.add_argument("--log", help = "stdout do TabbyAPI; lê o 'cached' da linha Metrics")
    p.add_argument("--tokens-identico", type = int, default = 64000)
    p.add_argument("--tokens-sistema", type = int, default = 7900)
    p.add_argument("--tokens-primeira-conversa", type = int, default = 30000)
    p.add_argument("--saida", type = int, default = 64)
    args = p.parse_args()

    print("aquecimento (compila Triton, não conta)…", flush = True)
    for n in (200, args.tokens_sistema, args.tokens_identico):
        texto, _, _ = registros(max(1, n // TOK_POR_REGISTRO), semente = 11)
        chat(args.url, args.chave, [{"role": "user", "content": f"[{uuid.uuid4().hex}]\n{texto}\nResuma."}], 8)

    nonce = uuid.uuid4().hex[:10]
    texto, _, _ = registros(args.tokens_identico // TOK_POR_REGISTRO, semente = 5)
    msgs = [{"role": "user", "content": f"[sessão {nonce}]\n{texto}\n\nQual cidade aparece mais? Responda curto."}]
    a, t1 = medir("idêntico, 1ª vez", args, msgs, args.saida)
    b, t2 = medir("idêntico, de novo", args, msgs, args.saida)
    print(f"  saída igual: {'sim' if a == b else 'NÃO'} · {t1 / max(t2, 1e-6):.0f}x", flush = True)

    sistema, _, _ = registros(args.tokens_sistema // TOK_POR_REGISTRO, semente = 9)
    sistema = f"[sessão {uuid.uuid4().hex[:10]}] Você é um auditor. Registros de referência:\n{sistema}"
    anexo, _, _ = registros(args.tokens_primeira_conversa // TOK_POR_REGISTRO, semente = 13)
    perguntas = [f"Anexo:\n{anexo}\n\nQuantos registros do anexo estão pendentes?",
                 "Qual o maior valor em Recife?", "Liste três clientes de Manaus."]
    tempos = []
    for i, q in enumerate(perguntas):
        _, t = medir(f"sistema, conversa {i + 1}", args,
                     [{"role": "system", "content": sistema}, {"role": "user", "content": q}], args.saida)
        tempos.append(t)
    print(f"  conversas novas com o mesmo sistema: {tempos[1]:.3f} s e {tempos[2]:.3f} s até o 1º token "
          f"(compare com a rodada sem os interruptores)", flush = True)


if __name__ == "__main__":
    main()
