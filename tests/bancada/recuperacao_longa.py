"""
O modelo ACHA o que está no meio do contexto? A pergunta que KL e tok/s não respondem.

    python tests/bancada/recuperacao_longa.py http://127.0.0.1:5000 $TABBY_API_TOKEN
    python tests/bancada/recuperacao_longa.py URL TOKEN --tokens 200000,1000000

Anunciamos 1M de contexto e validamos com KL (fidelidade ao modelo base) e tok/s
(velocidade). Nenhum dos dois pega o modo de falha que importa aqui: um artefato pode ter
KL excelente e mesmo assim não encontrar um dado a 60 % do prompt. Isso é qualidade do
CACHE e da atenção em contexto longo, não do peso quantizado.

Método: quatro marcadores de valor aleatório são plantados em posições percentuais de um
prompt do tamanho pedido, e o modelo tem de devolver os quatro EXATOS num JSON. Um erro
em qualquer um reprova a rodada.

Quatro travas, porque um teste de recuperação é fácil de passar por acidente:

  - **nonce único na PRIMEIRA linha**, senão o cache de prefixo responde pelo modelo;
  - **`prompt_tokens` do servidor** tem de bater o alvo — prova que o contexto entrou
    mesmo, em vez de ter sido truncado em silêncio;
  - **tokens em cache = 0**, lido do próprio servidor: com cache quente o teste mede o
    cache, não o modelo;
  - **tipo estrito** na comparação, senão `True` casaria com `1`.

Os valores são hex de 16 dígitos, gerados na hora: nada memorizável, nada adivinhável.
"""
import argparse, json, random, sys, time, urllib.request, uuid

FRACOES_PADRAO = (0.05, 0.35, 0.65, 0.95)
NOMES = ("ambar", "bruma", "cobalto", "dalia")
CIDADES = ["Ouro Preto", "Belém", "Curitiba", "Recife", "Manaus", "Goiânia", "Porto Alegre", "Salvador"]


def encher(n_linhas, semente):
    """Registros determinísticos e variados. Enchimento repetitivo demais é comprimido pela
    atenção de um jeito que não representa contexto real."""
    rnd = random.Random(semente)
    return "\n".join(
        f"registro {i:06d}: cliente {rnd.randint(0, 0xFFFFFF):06X} em {rnd.choice(CIDADES)}, "
        f"valor R$ {rnd.randint(100, 99_999)}, situação {rnd.choice(['pago', 'pendente', 'em análise'])}"
        for i in range(1, n_linhas + 1)
    )


def tokens_de(url, chave, texto):
    """Comprimento em tokens pelo tokenizador do PRÓPRIO servidor. `None` se ele não expõe
    o endpoint — aí o alvo sai por estimativa e o número real vem do `usage` no fim."""
    corpo = json.dumps({"text": texto, "add_bos_token": False}).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/token/encode", data = corpo,
        headers = {"Authorization": f"Bearer {chave}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout = 120) as r:
            d = json.load(r)
        return d.get("length") or len(d.get("tokens") or [])
    except Exception:
        return None


def montar(url, chave, alvo, fracoes, margem = 0.04):
    """Prompt do tamanho pedido, com os marcadores nas frações. Devolve (prompt, esperado, posições).

    O alvo é um TETO, não uma mira: passar dele derruba o pedido inteiro com HTTP 400
    ("Prompt length N exceeds"), e num teste de 1M isso custa o prefill inteiro para
    descobrir. Medido em 08/09/2026: a calibração linhas->tokens de uma amostra errou ~5 %
    para cima e estourou o contexto. Daí as duas defesas: mirar `margem` abaixo do alvo, e
    MEDIR o prompt pronto, encolhendo até caber de fato.
    """
    esperado = {nome: f"{random.getrandbits(64):016x}" for nome in NOMES}
    amostra = encher(200, 1)
    por_linha = (tokens_de(url, chave, amostra) or len(amostra) // 4) / 200
    linhas_totais = max(len(NOMES) * 4, int(alvo * (1 - margem) / por_linha))

    for _ in range(4):
        prompt, posicoes = _com_marcadores(linhas_totais, esperado, fracoes)
        real = tokens_de(url, chave, prompt)
        if real is None or real <= alvo:
            return prompt, esperado, posicoes, real
        # encolhe pela razão medida, com um empurrão extra para não repetir a tentativa
        linhas_totais = int(linhas_totais * alvo * (1 - margem) / real)
    return prompt, esperado, posicoes, real


def _com_marcadores(linhas_totais, esperado, fracoes):
    corpo, posicoes = [], {}
    todas = encher(linhas_totais, 99).split("\n")
    cortes = {int(len(todas) * f): nome for f, nome in zip(fracoes, NOMES)}
    for i, linha in enumerate(todas):
        if i in cortes:
            nome = cortes[i]
            posicoes[nome] = round(i / len(todas), 3)
            corpo.append(f"### registro de auditoria: {nome} tem o valor de recuperação {esperado[nome]}")
        corpo.append(linha)

    pergunta = (
        "\n\nDevolva SOMENTE um objeto JSON, sem texto antes ou depois, com as chaves "
        + ", ".join(NOMES)
        + " e os valores de recuperação EXATOS que aparecem nos registros de auditoria acima.\n"
    )
    # o nonce vai na PRIMEIRA linha: cache de prefixo casa do token 0 para frente, então
    # qualquer coisa depois dele não invalidaria nada
    prompt = f"Consulta independente {uuid.uuid4().hex}. O arquivo abaixo é contexto.\n" + "\n".join(corpo) + pergunta
    return prompt, posicoes


def avaliar(texto, esperado):
    """Só passa com os quatro exatos, e com TIPO estrito: `True` não pode casar com `1`."""
    t = texto.strip()
    if "```" in t:
        t = t.split("```")[1].removeprefix("json").strip()
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j < i:
        return False, None
    try:
        obtido = json.loads(t[i : j + 1])
    except json.JSONDecodeError:
        return False, None
    if not isinstance(obtido, dict) or obtido.keys() != esperado.keys():
        return False, obtido
    return all(type(obtido[k]) is type(v) and obtido[k] == v for k, v in esperado.items()), obtido


def perguntar(url, chave, prompt, max_tokens):
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
    t_primeiro, texto, uso, motivo = None, [], None, None
    with urllib.request.urlopen(req, timeout = 7200) as r:
        for linha in r:
            linha = linha.decode().strip()
            if not linha.startswith("data:") or linha == "data: [DONE]":
                continue
            ev = json.loads(linha[5:])
            if ev.get("usage"):
                uso = ev["usage"]
            for c in ev.get("choices", []):
                if c.get("finish_reason"):
                    motivo = c["finish_reason"]
                d = c.get("delta", {})
                if (d.get("content") or d.get("reasoning_content")) and t_primeiro is None:
                    t_primeiro = time.time()
                if d.get("content"):
                    texto.append(d["content"])
    return "".join(texto), uso or {}, motivo, (t_primeiro or time.time()) - t0, time.time() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("chave")
    p.add_argument("--tokens", default = "200000", help = "tamanhos de prompt, separados por vírgula")
    p.add_argument("--max-tokens", type = int, default = 2048, help = "teto da resposta (o JSON é curto)")
    p.add_argument("--tolerancia", type = float, default = 0.15, help = "quanto o prompt pode ficar abaixo do alvo")
    p.add_argument("--saida", help = "grava o resultado em JSON")
    a = p.parse_args()

    resultados, todas_passaram = [], True
    for alvo in [int(t) for t in a.tokens.split(",")]:
        prompt, esperado, posicoes, medido = montar(a.url, a.chave, alvo, FRACOES_PADRAO)
        if medido is not None:
            print(f"prompt montado: {medido:,} tokens (teto {alvo:,})")
        texto, uso, motivo, ttft, total = perguntar(a.url, a.chave, prompt, a.max_tokens)

        entrada = uso.get("prompt_tokens")
        cache = (uso.get("prompt_tokens_details") or {}).get("cached_tokens")
        exato, obtido = avaliar(texto, esperado)
        contexto_ok = isinstance(entrada, int) and entrada >= alvo * (1 - a.tolerancia)
        cache_ok = cache in (None, 0)
        passou = exato and contexto_ok and cache_ok

        todas_passaram &= passou
        print(f"\n{'PASSOU' if passou else 'FALHOU'}  alvo {alvo:,} tok · servidor informou {entrada:,} tok"
              if isinstance(entrada, int) else f"\n{'PASSOU' if passou else 'FALHOU'}  alvo {alvo:,} tok · servidor não informou tokens")
        print(f"  marcadores em {', '.join(f'{n} {100*f:.0f}%' for n, f in posicoes.items())}")
        print(f"  recuperação exata: {'sim' if exato else 'NÃO'}   contexto entrou: {'sim' if contexto_ok else 'NÃO'}"
              f"   cache: {'limpo' if cache_ok else f'{cache} tokens QUENTES'}")
        print(f"  prefill {ttft:.1f} s · total {total:.1f} s · parada {motivo}")
        if not exato:
            print(f"  esperado: {esperado}")
            print(f"  obtido:   {obtido if obtido is not None else repr(texto[:200])}")

        resultados.append({
            "alvo_tokens": alvo, "prompt_tokens": entrada, "cached_tokens": cache,
            "posicoes": posicoes, "esperado": esperado, "obtido": obtido,
            "recuperacao_exata": exato, "contexto_entrou": contexto_ok, "cache_limpo": cache_ok,
            "passou": passou, "finish_reason": motivo,
            "ttft_s": round(ttft, 2), "total_s": round(total, 2),
            "escopo": "Quatro marcadores sintéticos em posições separadas; não é avaliação de "
                      "raciocínio em contexto longo nem benchmark de qualidade.",
        })

    if a.saida:
        with open(a.saida, "w") as f:
            json.dump({"resultados": resultados, "todas_passaram": todas_passaram}, f, indent = 2, ensure_ascii = False)
        print(f"\nresultado em {a.saida}")
    print(f"\n{'TODAS PASSARAM' if todas_passaram else 'HOUVE FALHA'}")
    return 0 if todas_passaram else 1


def autoteste():
    """A lógica de avaliação, sem servidor. `--autoteste` roda isto e sai."""
    falhas = []
    def ok(nome, cond, detalhe = ""):
        print(f"  {nome:<52} {'ok' if cond else 'FALHOU'}  {detalhe}")
        if not cond:
            falhas.append(nome)

    esperado = {"ambar": "0123456789abcdef", "bruma": "fedcba9876543210",
                "cobalto": "00000000deadbeef", "dalia": "1111111122222222"}
    bom = json.dumps(esperado)
    ok("os quatro exatos passam", avaliar(bom, esperado)[0])
    ok("dentro de bloco ``` passa", avaliar(f"```json\n{bom}\n```", esperado)[0])
    ok("com texto em volta passa", avaliar(f"Aqui está:\n{bom}\nEspero ter ajudado.", esperado)[0])

    um_errado = dict(esperado, dalia = "1111111122222223")
    ok("um dígito errado reprova", not avaliar(json.dumps(um_errado), esperado)[0])
    ok("chave faltando reprova", not avaliar(json.dumps({k: v for k, v in list(esperado.items())[:3]}), esperado)[0])
    ok("chave a mais reprova", not avaliar(json.dumps(dict(esperado, extra = "x")), esperado)[0])
    ok("sem JSON reprova", not avaliar("não encontrei os valores", esperado)[0])
    ok("JSON quebrado reprova", not avaliar('{"ambar": ', esperado)[0])

    ### O motivo do `type(...) is type(...)`: sem ele, `True == 1` e um modelo que
    ### devolvesse booleanos passaria num teste de valores hexadecimais.
    numerico = {"a": 1, "b": 2}
    ok("tipo estrito: True não casa com 1", not avaliar(json.dumps({"a": True, "b": 2}), numerico)[0])
    ok("mesmo tipo e valor casa", avaliar(json.dumps(numerico), numerico)[0])

    ### `montar` sem servidor: `tokens_de` falha na conexão e cai na estimativa por caracteres.
    prompt, esp, pos, medido = montar("http://127.0.0.1:1", "x", 4000, FRACOES_PADRAO)
    ok("sem servidor, a medição volta None e não trava", medido is None)
    ok("montar produz os quatro marcadores", len(esp) == 4 and len(pos) == 4, str(pos))
    ok("cada valor aparece uma vez no prompt", all(prompt.count(v) == 1 for v in esp.values()))
    ok("o nonce está na primeira linha", prompt.split("\n")[0].startswith("Consulta independente"))
    ok("as posições respeitam as frações pedidas",
       all(abs(pos[n] - f) < 0.02 for n, f in zip(NOMES, FRACOES_PADRAO)), str(pos))
    ok("a pergunta fica no fim", prompt.rstrip().endswith("acima."))
    ok("dois prompts não repetem valores",
       montar("http://127.0.0.1:1", "x", 4000, FRACOES_PADRAO)[1] != esp)

    ### O teto: um prompt maior que o contexto derruba o pedido com HTTP 400 depois de o
    ### servidor já ter recebido tudo. `montar` mede e encolhe; aqui o tokenizador é falso,
    ### para exercitar o laço sem servidor.
    import types
    real = globals()["tokens_de"]
    chamadas = []
    def falso(url, chave, texto):
        n = len(texto) // 4          # ~4 caracteres por token
        chamadas.append(n)
        return n
    globals()["tokens_de"] = falso
    try:
        _, _, _, medido2 = montar("x", "x", 20_000, FRACOES_PADRAO)
        ok("o prompt medido respeita o teto", medido2 is not None and medido2 <= 20_000,
           f"{medido2} tokens")
        ok("a medição do prompt pronto acontece", len(chamadas) >= 2, f"{len(chamadas)} chamadas")
    finally:
        globals()["tokens_de"] = real

    print("\nautoteste passou." if not falhas else f"\n{len(falhas)} FALHARAM: {falhas}")
    return 1 if falhas else 0


if __name__ == "__main__":
    if "--autoteste" in sys.argv:
        sys.exit(autoteste())
    sys.exit(main())
