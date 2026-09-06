"""
Régua de desempenho do TabbyAPI: três prompts fixos, sempre os mesmos, e uma linha de números
por prompt. É o que toda etapa do plano de desempenho usa para dizer "antes" e "depois".

    python tests/bancada/medir_tabby.py http://127.0.0.1:5000 $TABBY_API_TOKEN [--log /caminho/tabby.log]

Roda de DENTRO da máquina: o proxy do Cloudflare corta pedidos acima de 100 s e o relógio do
cliente por localhost não mente. Sem --log, prefill e decode saem do relógio do cliente em
streaming (tempo até o primeiro token = prefill; o resto = decode). Com --log (arquivo onde o
stdout do TabbyAPI está indo), lê a linha `Metrics (ID: ...)` que o servidor escreve depois de
cada pedido: Process/Generate T/s e a aceitação do draft, que só existe lá.

Prompts (tokens aproximados no tokenizador do GLM-5.3-Flash):
  curto    ~50 de entrada, 600 de saída, duas vezes (a segunda é a que vale: decode puro e draft)
  longo    ~5,3k de entrada, 200 de saída (prefill; DSA passa de denso a esparso)
  dificil  ~31k de entrada (1.000 registros), soma de 11 valores e uma agulha; confere a resposta
"""
import argparse, json, random, re, sys, time, urllib.request

CURTO = "Explique em três frases por que o céu é azul e por que o pôr do sol é avermelhado. Depois escreva um poema de 40 versos sobre isso."

CIDADES = ["Ouro Preto", "Belém", "Curitiba", "Recife", "Manaus", "Goiânia", "Porto Alegre", "Salvador", "Fortaleza", "Cuiabá"]


def registros(n, semente = 7):
    """n registros determinísticos; devolve o texto, a soma dos valores de Ouro Preto e a agulha."""
    rnd = random.Random(semente)
    linhas, soma, agulha = [], 0, None
    alvo = "Ouro Preto"
    for i in range(1, n + 1):
        cidade = rnd.choice(CIDADES) if rnd.random() > 0.011 or i < 100 else alvo
        valor = rnd.randint(100, 99_999)
        codigo = f"{rnd.randint(0, 0xFFFFFF):06X}"
        if cidade == alvo:
            soma += valor
        if i == n // 2:
            agulha = f"CHAVE-{codigo}"
            linhas.append(f"registro {i:04d}: cliente {codigo} em {cidade}, valor R$ {valor}, observação: chave de auditoria {agulha}")
        else:
            linhas.append(f"registro {i:04d}: cliente {codigo} em {cidade}, valor R$ {valor}, observação: {rnd.choice(['pago', 'pendente', 'em análise', 'cancelado'])}")
    return "\n".join(linhas), soma, agulha


def prompt_longo():
    texto, _, _ = registros(180, semente = 3)
    return texto + "\n\nResuma em dez linhas o que estes registros mostram e aponte a cidade com mais registros."


def prompt_dificil():
    texto, soma, agulha = registros(1000)
    pergunta = (
        "\n\nDuas tarefas, e mostre o raciocínio curto antes de cada resposta final:\n"
        "1. Qual é a soma exata dos valores (R$) de todos os registros em Ouro Preto? Responda com o número inteiro.\n"
        "2. Qual é a chave de auditoria mencionada em um dos registros (formato CHAVE-XXXXXX)?"
    )
    return texto + pergunta, soma, agulha


def chat(url, chave, prompt, max_tokens):
    """Pedido em streaming; devolve texto, tokens de uso, tempo até o 1º token e tempo total."""
    corpo = json.dumps({
        "model": "x", "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0, "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions", data = corpo,
        headers = {"Authorization": f"Bearer {chave}", "Content-Type": "application/json"},
    )
    t0 = time.time()
    t_primeiro, texto, uso = None, [], None
    with urllib.request.urlopen(req, timeout = 3600) as r:
        for linha in r:
            linha = linha.decode().strip()
            if not linha.startswith("data:") or linha == "data: [DONE]":
                continue
            ev = json.loads(linha[5:])
            if ev.get("usage"):
                uso = ev["usage"]
            for c in ev.get("choices", []):
                d = c.get("delta", {})
                # o GLM começa em raciocínio: os primeiros deltas vêm em reasoning_content
                if (d.get("content") or d.get("reasoning_content")) and t_primeiro is None:
                    t_primeiro = time.time()
                if d.get("content"):
                    texto.append(d["content"])
    return "".join(texto), uso or {}, (t_primeiro or time.time()) - t0, time.time() - t0


RE_METRICS = re.compile(r"Metrics \(ID: (\S+)\): (\d+) tokens generated in ([\d.]+) seconds \((.*)\)")


def parse_metrics(linha):
    """Lê a linha `Metrics` do TabbyAPI (common/gen_logging.py). Campos por nome, em qualquer ordem."""
    m = RE_METRICS.search(linha)
    if not m:
        return None
    itens = m.group(4)
    def num(padrao):
        k = re.search(padrao, itens)
        return float(k.group(1)) if k else None
    return {
        "id": m.group(1),
        "gerados": int(m.group(2)),
        "segundos": float(m.group(3)),
        "cache_tok": num(r"Process: (\d+) cached"),
        "novos_tok": num(r"and (\d+) new tokens"),
        "prefill_tps": num(r"new tokens at ([\d.]+) T/s"),
        "decode_tps": num(r"Generate: ([\d.]+) T/s"),
        "contexto": num(r"Context: (\d+) tokens"),
        "draft_pct": num(r"accepted \(([\d.]+)%\)"),
    }


def ultima_metrics(caminho, depois_de):
    """Última linha Metrics do log, contando só o que foi escrito depois da posição dada."""
    with open(caminho, errors = "replace") as f:
        f.seek(depois_de)
        ultima = None
        for linha in f:
            if "Metrics (ID:" in linha:
                ultima = linha
    return parse_metrics(ultima) if ultima else None


def medir(nome, url, chave, prompt, max_tokens, log):
    pos = 0
    if log:
        with open(log, "rb") as f:
            f.seek(0, 2)
            pos = f.tell()
    texto, uso, ttft, total = chat(url, chave, prompt, max_tokens)
    entrada = uso.get("prompt_tokens")
    saida = uso.get("completion_tokens") or 0
    # relógio do cliente: o 1º token sai depois do prefill + 1 passo de decode; o erro é um passo
    r = {
        "prompt": nome, "entrada_tok": entrada, "saida_tok": saida,
        "prefill_tps": round(entrada / ttft, 1) if entrada and ttft > 0 else None,
        "decode_tps": round((saida - 1) / (total - ttft), 1) if saida > 1 and total > ttft else None,
        "draft_pct": None, "fonte": "cliente", "segundos": round(total, 1),
    }
    if log:
        time.sleep(0.5)
        m = ultima_metrics(log, pos)
        if m:
            r.update(prefill_tps = m["prefill_tps"], decode_tps = m["decode_tps"], draft_pct = m["draft_pct"],
                     cache_tok = m["cache_tok"], fonte = "servidor")
    return r, texto


def linha(r):
    d = f"{r['draft_pct']:.0f} %" if r.get("draft_pct") is not None else "-"
    return (f"{r['prompt']:<9} entrada {r['entrada_tok']!s:>6} tok · saída {r['saida_tok']:>5} tok · "
            f"prefill {r['prefill_tps'] or 0:>7.1f} tok/s · decode {r['decode_tps'] or 0:>6.1f} tok/s · draft {d:>5} · "
            f"{r['segundos']:.1f} s ({r['fonte']})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("chave")
    p.add_argument("--log", help = "arquivo com o stdout do TabbyAPI; lê a linha Metrics de cada pedido")
    p.add_argument("--prompts", default = "curto,longo,dificil")
    p.add_argument("--saida-json", help = "grava a lista de resultados neste arquivo")
    p.add_argument("--saida-dificil", type = int, default = 2000, help = "max_tokens do prompt difícil")
    args = p.parse_args()
    quais = args.prompts.split(",")

    print("aquecimento (compila Triton, não conta)…", flush = True)
    chat(args.url, args.chave, "Diga apenas: pronto.", 8)

    resultados = []
    if "curto" in quais:
        for i in (1, 2):
            r, _ = medir(f"curto/{i}", args.url, args.chave, CURTO, 600, args.log)
            resultados.append(r); print(linha(r), flush = True)
    if "longo" in quais:
        r, _ = medir("longo", args.url, args.chave, prompt_longo(), 200, args.log)
        resultados.append(r); print(linha(r), flush = True)
    if "dificil" in quais:
        prompt, soma, agulha = prompt_dificil()
        r, texto = medir("dificil", args.url, args.chave, prompt, args.saida_dificil, args.log)
        soma_ok = str(soma) in texto.replace(".", "").replace(",", "")
        agulha_ok = agulha in texto
        r["exato"] = {"soma": soma_ok, "agulha": agulha_ok}
        resultados.append(r); print(linha(r), flush = True)
        print(f"          exatidão: soma {'certa' if soma_ok else f'ERRADA (esperado {soma})'} · agulha {'certa' if agulha_ok else f'ERRADA (esperado {agulha})'}")

    if args.saida_json:
        with open(args.saida_json, "w") as f:
            json.dump(resultados, f, ensure_ascii = False, indent = 1)
        print("gravado:", args.saida_json)


if __name__ == "__main__":
    main()
