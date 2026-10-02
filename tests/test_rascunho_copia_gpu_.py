"""
Rascunho por cópia no gerador, na GPU: a saída com EXL3_RASCUNHO_COPIA=1 é a mesma de sem a
cópia (mesma semente, mesmo fluxo de rng por job), e quanto ela rende em resposta de edição de
código e em prosa.

    python3 tests/test_rascunho_copia_gpu_.py MODELO [--paralelas 1,4] [--novos 400] -- -mtp -tp ...

Depois de ``--`` vão os argumentos do ``model_init`` (``-mtp`` para a cabeça MTP do checkpoint,
``-dm DIR`` para DFlash2, ``-tp``, ``-cs`` ...). O processo carrega o modelo uma vez, com o
cache já reservando o histórico recorrente da cópia (cache.py lê EXL3_COPIA_MAX), e roda cada
prompt duas vezes: um gerador sem a cópia e outro com ela, sobre o mesmo cache.

Sai uma linha por (prompt, paralelas, modo) com tok/s por conversa e agregado, e as contas da
cópia (rodadas, propostos, aceitos). Termina em erro se algum job divergiu.

Sobre "a mesma saída": cada posição verificada é amostrada dos logits do alvo com o próximo
número do rng do job, como no caminho serial; o rascunho só decide quantas posições a rodada
verifica. Janelas de larguras diferentes passam por kernels diferentes, então os logits podem
diferir no último bit; com amostragem gulosa e o prompt curto daqui isso não muda o token. Se
divergir, a linha mostra a primeira posição diferente para conferir se é empate numérico.
"""
import argparse
import inspect
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("EXL3_RASCUNHO_COPIA", "1")      # antes do Cache: reserva o histórico da cópia

from exllamav3 import model_init, Generator, Job, ArgmaxSampler                    # noqa: E402
from exllamav3.generator.sampler import ComboSampler                               # noqa: E402

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def prompts():
    codigo = open(os.path.join(RAIZ, "exllamav3", "generator", "loop_detect.py")).read()
    return {
        "edicao": (
            "Reescreva o arquivo Python abaixo inteiro, sem omitir nada, trocando o nome da classe "
            "`LoopDetector` por `DetectorDeLaco` em todas as ocorrências. Responda só com o código.\n\n"
            f"```python\n{codigo}\n```"
        ),
        "prosa": (
            "Escreva um ensaio de umas 600 palavras sobre a história da navegação a vela no "
            "Atlântico, do século XV ao XIX, com exemplos concretos."
        ),
    }


def codificar(tok, texto):
    try:
        return tok.hf_chat_template([{"role": "user", "content": texto}], add_generation_prompt = True,
                                    enable_thinking = False)
    except Exception:
        return tok.encode(texto, add_bos = True)


def rodar(r, draft_kw, ids, paralelas, novos, amostrado):
    model, config, cache, tok = r[:4]
    gen = Generator(model = model, cache = cache, tokenizer = tok, max_batch_size = max(paralelas, 1),
                    **draft_kw)
    jobs = []
    for i in range(paralelas):
        sampler = ComboSampler(temperature = 0.7, top_p = 0.9) if amostrado else ArgmaxSampler()
        j = Job(input_ids = ids, max_new_tokens = novos, sampler = sampler, seed = 1234 + i,
                identifier = i)
        jobs.append(j)
        gen.enqueue(j)
    saida = {i: [] for i in range(paralelas)}
    contas = {}
    t_primeiro = None
    while gen.num_remaining_jobs():
        for res in gen.iterate():
            ti = res.get("token_ids")
            if ti is not None and ti.numel():
                if t_primeiro is None:
                    t_primeiro = time.perf_counter()
                saida[res["identifier"]] += ti.view(-1).tolist()
            if res.get("eos"):
                contas[res["identifier"]] = (res.get("copy_rounds"), res.get("copy_drafted"),
                                             res.get("copy_accepted"), res.get("accepted_draft_tokens"),
                                             res.get("rejected_draft_tokens"))
    dt = time.perf_counter() - (t_primeiro or time.perf_counter())
    copia = gen.copia is not None
    del gen
    total = sum(len(v) for v in saida.values())
    return saida, contas, total / max(dt, 1e-9), copia


def main():
    p = argparse.ArgumentParser()
    p.add_argument("modelo")
    p.add_argument("--paralelas", default = "1,4")
    p.add_argument("--novos", type = int, default = 400)
    p.add_argument("--amostrado", action = "store_true", help = "temperatura 0,7 e top-p 0,9 em vez de gulosa")
    args, resto = p.parse_known_args()
    if resto and resto[0] == "--":
        resto = resto[1:]

    mp = argparse.ArgumentParser()
    model_init.add_args(mp, **{n: True for n in inspect.signature(model_init.add_args).parameters if "draft" in n})
    r = model_init.init(mp.parse_args(["-m", args.modelo] + resto))
    if len(r) < 7 or r[4] is None:
        sys.exit("passe um rascunhador depois de --: -mtp (cabeça MTP) ou -dm DIR (DFlash2)")
    draft_kw = {"draft_model": r[4], "draft_cache": r[6]}
    tok = r[3]

    divergiu = False
    for nome, texto in prompts().items():
        ids = codificar(tok, texto)
        for paralelas in [int(x) for x in args.paralelas.split(",")]:
            ref = None
            for modo in ("0", "1"):
                os.environ["EXL3_RASCUNHO_COPIA"] = modo
                saida, contas, tps, ligada = rodar(r, draft_kw, ids, paralelas, args.novos, args.amostrado)
                assert ligada == (modo == "1"), "o gerador não leu EXL3_RASCUNHO_COPIA"
                linha = (f"{nome:7s} paralelas={paralelas:2d} copia={modo}  "
                         f"{tps / paralelas:7.1f} tok/s por conversa  {tps:7.1f} agregado")
                if modo == "1":
                    rod = sum(c[0] or 0 for c in contas.values())
                    prop = sum(c[1] or 0 for c in contas.values())
                    ac = sum(c[2] or 0 for c in contas.values())
                    linha += f"  cópia: {rod} rodadas, {ac}/{prop} aceitos"
                print(linha, flush = True)
                if ref is None:
                    ref = saida
                    continue
                for i in ref:
                    if ref[i] != saida[i]:
                        k = next((x for x, (a, b) in enumerate(zip(ref[i], saida[i])) if a != b),
                                 min(len(ref[i]), len(saida[i])))
                        print(f"  !! job {i}: divergiu na posição {k} "
                              f"({ref[i][k:k + 4]} sem cópia, {saida[i][k:k + 4]} com)", flush = True)
                        divergiu = True
    if divergiu:
        sys.exit(1)
    print("PASS: a cópia não mudou a saída")


if __name__ == "__main__":
    main()
