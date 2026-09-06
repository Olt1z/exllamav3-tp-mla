"""
Prefill e decode pelo GERADOR, do jeito que o TabbyAPI usa: chunks de `--chunk`, rascunho
opcional (DFlash 2) com exportação de estados por chunk, contexto de vários tamanhos.

    python tests/bancada/perfil_gerador.py -m /workspace/corte --tokens 4096,16384,30000
    python tests/bancada/perfil_gerador.py -m /workspace/corte --tp --backend nccl --chunk 8192
    python tests/bancada/perfil_gerador.py -m /workspace/corte --dflash2 /workspace/dflash2 --draft-stats

Uma linha por tamanho: prefill tok/s (tempo até o primeiro token, do próprio job), decode tok/s
e, com rascunho, tokens aceitos por rodada e a aceitação por posição do bloco (draft_stats do
gerador). Cada prompt começa com um nonce, então nada vem do cache de páginas. O primeiro
tamanho roda duas vezes e a primeira leitura (compilação do Triton) é descartada.
"""
import sys, os, argparse, time, random, collections
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler.presets import ArgmaxSampler

ENCHIMENTO = "A luz do sol atravessa a atmosfera e as moléculas de ar espalham mais os comprimentos de onda curtos do que os longos. "


def prompt_de(tokenizer, n):
    nonce = f"[{random.randrange(10**9):09d}] "
    texto = nonce + ENCHIMENTO * (n // 20 + 2)
    ids = tokenizer.encode(texto, encode_special_tokens = True)[:, :n]
    return ids


def rodar(gen, ids, novos):
    job = Job(input_ids = ids, max_new_tokens = novos, sampler = ArgmaxSampler(), decode_special_tokens = True)
    gen.enqueue(job)
    fim = None
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("eos"):
                fim = r
    if fim is None or fim.get("stage") == "error":
        raise RuntimeError(f"job falhou: {fim.get('error') if fim else 'sem resultado'}")
    return fim, job


def resumo(rotulo, r, job, com_rascunho):
    p, n = r["prompt_tokens"], r["new_tokens"]
    tp, tg = r["time_prefill"], r["time_generate"]
    linha = f"{rotulo:<28} prefill {p:>6} tok em {tp * 1000:>7.0f} ms = {p / tp:>6.0f} tok/s"
    if n > 1 and tg > tp:
        linha += f" · decode {n} tok = {(n - 1) / (tg - tp):>5.1f} tok/s"
    if com_rascunho:
        ac, rj = r.get("accepted_draft_tokens", 0), r.get("rejected_draft_tokens", 0)
        rodadas = len(job.draft_stats) or max(1, (ac + rj) // max(1, gen_draft_len(job)))
        linha += f" · aceito {ac}/{ac + rj} ({100 * ac / max(1, ac + rj):.0f} %) · {n / rodadas:.2f} tok/rodada"
    print(linha)
    if com_rascunho and job.draft_stats:
        # aceitação por posição do bloco: quantas rodadas chegaram (aceitaram) até a posição i
        por_pos = collections.Counter()
        for _, janela, aceitos in job.draft_stats:
            for i in range(janela):
                por_pos[i] += 1 if i < aceitos else 0
        total = len(job.draft_stats)
        print("   aceita a posição i em % das rodadas: " + " ".join(f"{i + 1}:{100 * por_pos[i] / total:.0f}" for i in sorted(por_pos)))


def gen_draft_len(job):
    return job.draft_stats[0][1] if job.draft_stats else 7


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--model_dir", required = True)
    p.add_argument("--tp", action = "store_true")
    p.add_argument("--backend", default = "nccl", help = "backend do TP (o hub usa nccl)")
    p.add_argument("--chunk", type = int, default = 4096, help = "max_chunk_size do gerador (o hub passa --chunk-size)")
    p.add_argument("--tokens", default = "4096,16384,30000", help = "tamanhos de prompt, separados por vírgula")
    p.add_argument("--novos", type = int, default = 64, help = "tokens gerados por prompt")
    p.add_argument("--cache", type = int, default = 32768)
    p.add_argument("--dflash2", help = "pasta do rascunho DFlash 2; vazio = sem rascunho")
    p.add_argument("--draft-stats", action = "store_true", help = "aceitação por posição do bloco")
    p.add_argument("--taps", help = "sobrescreve target_layer_ids do rascunho (ex.: 0,1,2,3,3 no corte de 4 camadas, "
                                    "onde os taps reais 5..42 não existem; a numérica vira lixo, o custo do encanamento não)")
    p.add_argument("--reserva-gb", type = float, default = 0)
    p.add_argument("--rotulo", default = "")
    args = p.parse_args()
    tamanhos = [int(t) for t in args.tokens.split(",")]
    assert max(tamanhos) + args.novos + 64 <= args.cache, "cache menor que o maior prompt"

    cfg = Config.from_directory(args.model_dir)
    model = Model.from_config(cfg)
    draft = dcache = None
    if args.dflash2:
        cfg_d = Config.from_directory(args.dflash2)
        if args.taps:
            cfg_d.target_layer_ids = [int(t) for t in args.taps.split(",")]
            print(f"taps sobrescritos: {cfg_d.target_layer_ids}")
        draft = Model.from_config(cfg_d)
        dcache = Cache(draft, max_num_tokens = args.cache)
        draft.load(device = torch.device("cuda:0"), progressbar = True)
    cache = Cache(model, max_num_tokens = args.cache,
                  max_history = draft.caps.get("default_draft_size", 4) if draft else 0)
    if args.tp:
        model.load(tensor_p = True, tp_backend = args.backend, progressbar = True,
                   reserve_per_device = args.reserva_gb or None)
    else:
        model.load(device = torch.device("cuda:0"), progressbar = True)
    if draft:
        draft.attach_to(model)
    tokenizer = Tokenizer.from_config(cfg)
    gen = Generator(model = model, cache = cache, tokenizer = tokenizer, max_batch_size = 1,
                    max_chunk_size = args.chunk, draft_model = draft, draft_cache = dcache,
                    record_draft_stats = args.draft_stats and draft is not None)
    rot = args.rotulo or (f"{'TP ' + args.backend if args.tp else '1 placa'} chunk {args.chunk}"
                          + (" DFlash2" if draft else ""))
    print(f"{rot}: dispositivos {model.active_devices}, cache {args.cache}, rascunho {'sim' if draft else 'não'}")

    torch.manual_seed(0); random.seed(0)
    # aquecimento no MAIOR tamanho (Triton compila por forma; o KDA de 30k não aquece com 4k)
    r, job = rodar(gen, prompt_de(tokenizer, max(tamanhos)), 8)
    print(f"aquecimento: {r['prompt_tokens']} tok em {r['time_prefill'] * 1000:.0f} ms (descartado)")
    for n in tamanhos:
        r, job = rodar(gen, prompt_de(tokenizer, n), args.novos)
        resumo(f"{rot} · {n}", r, job, draft is not None)
    print("FIM_PERFIL_GERADOR")


if __name__ == "__main__":
    main()
