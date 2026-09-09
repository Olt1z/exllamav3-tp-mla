"""
O GERADOR paginado sob context parallel: prefixo compartilhado, páginas cheias e lote.

    python3 tests/bancada/provar_gerador_cp.py -m /workspace/corte --tp --save base.pt
    python3 tests/bancada/provar_gerador_cp.py -m /workspace/corte --tp --dcp 2 --compare base.pt

O smoke (tp_mla_smoke.py) prova o forward com `batch_shape`, onde a block table é um `arange`.
O hub serve pelo GERADOR, com pool de páginas, hash de prefixo e lote — e é aí que a prova 24
achou o risco: sob CP cada rank tem `max_num_tokens // (PAGE_SIZE · dcp)` páginas físicas, e um
id de página acima disso lê fora do tensor em silêncio. A página LÓGICA do gerador passou a ser
`PAGE_SIZE · dcp` tokens (`Generator.page_tokens`), e esta prova exercita o que isso toca:

  A  um prompt longo, sozinho: cruza várias páginas lógicas (prefill em chunk + decode);
  B  o MESMO prefixo de A com cauda diferente, junto com C num lote de 2: B tem de reaproveitar
     as páginas de A pelo hash (cached_pages > 0) e C é fresco.

O que se compara entre `dcp 1` e `dcp > 1` é o logit do PRIMEIRO passo de cada job (só depende
do prompt, sem a deriva do greedy num corte de texto-lixo) e a concordância do top-1 nos passos
seguintes. E o número de páginas reaproveitadas por B, que tem de ser o mesmo em unidades de
tokens: `cached_pages × page_tokens`.
"""
import sys, os, argparse, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler.presets import ArgmaxSampler

FRASE = ("A luz do sol atravessa a atmosfera e as moléculas de ar espalham mais os comprimentos de "
         "onda curtos do que os longos. ")


def rodar(gen, tok, jobs_spec, novos):
    """jobs_spec: lista de (nome, ids). Enfileira todos, itera até acabar, devolve por nome:
    tokens, logits do passo 0, cached_pages, cached_tokens."""
    saida = {}
    for nome, ids in jobs_spec:
        job = Job(input_ids = ids, max_new_tokens = novos, sampler = ArgmaxSampler(),
                  return_logits = True, identifier = nome)
        gen.enqueue(job)
        saida[nome] = dict(tokens = [], logit0 = None, cached_pages = None, cached_tokens = None)
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            d = saida[r["identifier"]]
            if r.get("stage") == "prefill":
                continue
            if "token_ids" in r and r["token_ids"] is not None:
                d["tokens"].extend(r["token_ids"].flatten().tolist())
            if "logits" in r and d["logit0"] is None:
                # o primeiro resultado pode trazer varios passos: a primeira linha e o passo 0
                L = r["logits"].float()
                d["logit0"] = L.reshape(-1, L.shape[-1])[0].cpu()
            if r.get("cached_pages") is not None:
                d["cached_pages"] = r["cached_pages"]
                d["cached_tokens"] = r.get("cached_tokens")
    return saida


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--model_dir", required = True)
    p.add_argument("--tp", action = "store_true")
    p.add_argument("--dcp", type = int, default = 1)
    p.add_argument("--backend", default = "nccl")
    p.add_argument("--cache", type = int, default = 16384)
    p.add_argument("--prefixo", type = int, default = 1500, help = "tokens do prompt A (cruza páginas)")
    p.add_argument("--tokens", type = int, default = 24)
    p.add_argument("--save")
    p.add_argument("--compare")
    p.add_argument("--max-kl", type = float)
    a = p.parse_args()

    config = Config.from_directory(a.model_dir)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = a.cache)
    t0 = time.time()
    model.load(tensor_p = a.tp, tp_backend = a.backend, progressbar = True,
               tp_options = {"dcp": a.dcp} if a.dcp > 1 else None)
    print(f"carga: {time.time() - t0:.0f} s, dispositivos {model.active_devices}, cp_world {getattr(model, 'cp_world', 1)}")
    tok = Tokenizer.from_config(config)
    gen = Generator(model = model, cache = cache, tokenizer = tok, max_batch_size = 4)
    print(f"pagina logica: {gen.page_tokens} tokens · {gen.pagetable.max_pages} paginas · "
          f"{gen.max_total_tokens} tokens no pool")

    n_frase = len(tok.encode(FRASE)[0])
    base = FRASE * (a.prefixo // n_frase + 1)
    ids_a = tok.encode(model.default_chat_prompt(base + "Explique em uma frase por que o céu é azul."), encode_special_tokens = True)
    ids_b = tok.encode(model.default_chat_prompt(base + "Explique em uma frase por que o pôr do sol é vermelho."), encode_special_tokens = True)
    ids_c = tok.encode(model.default_chat_prompt("Diga bom dia."), encode_special_tokens = True)
    print(f"prompts: A {ids_a.shape[-1]} · B {ids_b.shape[-1]} · C {ids_c.shape[-1]} tokens")

    r1 = rodar(gen, tok, [("A", ids_a)], a.tokens)
    r2 = rodar(gen, tok, [("B", ids_b), ("C", ids_c)], a.tokens)
    res = {**r1, **r2}
    for nome in ("A", "B", "C"):
        d = res[nome]
        print(f"{nome}: {len(d['tokens'])} tokens · reaproveitou {d['cached_pages']} paginas "
              f"({(d['cached_pages'] or 0) * gen.page_tokens} tokens) · texto: "
              f"{tok.decode(torch.tensor([d['tokens']]))[0][:60]!r}")
    if res["B"]["cached_pages"] is None or (res["B"]["cached_pages"] or 0) * gen.page_tokens < 512:
        print("FALHOU: B nao reaproveitou o prefixo de A pelo hash de pagina")
        raise SystemExit(1)

    if a.save:
        torch.save({n: dict(tokens = torch.tensor(d["tokens"]), logit0 = d["logit0"],
                            cached_tokens = (d["cached_pages"] or 0) * gen.page_tokens) for n, d in res.items()},
                   a.save)
        print("gravado:", a.save)
    if a.compare:
        ref = torch.load(a.compare)
        pior = 0.0
        for nome in ("A", "B", "C"):
            la, lb = ref[nome]["logit0"].log_softmax(-1), res[nome]["logit0"].log_softmax(-1)
            kl = (la.exp() * (la - lb)).sum().item()
            ta, tb = ref[nome]["tokens"], torch.tensor(res[nome]["tokens"])
            n = min(len(ta), len(tb))
            top1 = (ta[:n] == tb[:n]).float().mean().item() if n else 0.0
            ct_ok = int(ref[nome]["cached_tokens"]) == (res[nome]["cached_pages"] or 0) * gen.page_tokens
            print(f"{nome}: KL do passo 0 {kl:.5f} · top-1 igual {top1:.0%} em {n} passos · "
                  f"prefixo reaproveitado igual em tokens: {'sim' if ct_ok else 'NAO'}")
            pior = max(pior, kl)
            if not ct_ok:
                raise SystemExit(1)
        if a.max_kl is not None and pior > a.max_kl:
            print(f"FALHOU: KL do passo 0 {pior:.5f} > {a.max_kl}")
            raise SystemExit(1)
    print("OK")


if __name__ == "__main__":
    main()
