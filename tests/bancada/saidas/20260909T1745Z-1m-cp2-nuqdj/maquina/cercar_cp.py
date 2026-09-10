"""
Cerca o tamanho de prompt em que o GERADOR sob CP quebra, num processo só (uma carga do modelo).

    python3 cercar_cp.py /workspace/<modelo> --dcp 2 --backend nccl --chunk 4096 --tamanhos 1000,2000,...

Espelha o caminho do TabbyAPI: Generator + Job, cache paginado, chunk de prefill de 4096.
Cada tamanho gera 8 tokens em greedy e imprime; a primeira quebra (CUDA illegal address)
derruba o processo, e o ultimo "OK" impresso e o teto que passou.
"""
import sys, os, argparse, time
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler.presets import ArgmaxSampler

FRASE = ("A luz do sol atravessa a atmosfera e as moleculas de ar espalham mais os comprimentos de "
         "onda curtos do que os longos, e por isso o ceu parece azul ao meio-dia. ")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_dir")
    p.add_argument("--dcp", type = int, default = 2)
    p.add_argument("--backend", default = "nccl")
    p.add_argument("--chunk", type = int, default = 4096)
    p.add_argument("--cache", type = int, default = 65536)
    p.add_argument("--tamanhos", default = "1000,2000,2500,3000,4096,4200,6000,8000")
    p.add_argument("--novos", type = int, default = 8)
    a = p.parse_args()

    config = Config.from_directory(a.model_dir)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = a.cache)
    t0 = time.time()
    model.load(tensor_p = True, tp_backend = a.backend, progressbar = False, max_chunk_size = a.chunk,
               tp_options = {"dcp": a.dcp} if a.dcp > 1 else None)
    print(f"carga: {time.time() - t0:.0f} s · dispositivos {model.active_devices} · cp_world {getattr(model, 'cp_world', 1)} · chunk {a.chunk}", flush = True)
    tok = Tokenizer.from_config(config)
    gen = Generator(model = model, cache = cache, tokenizer = tok, max_batch_size = 4, max_chunk_size = a.chunk)
    print(f"pagina logica {gen.page_tokens} · {gen.pagetable.max_pages} paginas", flush = True)

    n_frase = len(tok.encode(FRASE)[0])
    for alvo in [int(x) for x in a.tamanhos.split(",")]:
        texto = FRASE * (alvo // n_frase + 1)
        ids = tok.encode(model.default_chat_prompt(texto + "Em uma frase: por que o ceu e azul?"), encode_special_tokens = True)
        ids = ids[:, :alvo] if ids.shape[-1] > alvo else ids
        t0 = time.time()
        job = Job(input_ids = ids, max_new_tokens = a.novos, sampler = ArgmaxSampler(), identifier = str(alvo))
        gen.enqueue(job)
        saida = []
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                if r.get("stage") == "prefill": continue
                if r.get("token_ids") is not None: saida.extend(r["token_ids"].flatten().tolist())
        torch.cuda.synchronize()
        print(f"OK  {ids.shape[-1]:>7} tokens · {time.time() - t0:.1f} s · {tok.decode(torch.tensor([saida]))[0][:40]!r}", flush = True)
    print("FIM: todos passaram", flush = True)


if __name__ == "__main__":
    main()
