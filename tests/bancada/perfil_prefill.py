"""
Perfil do prefill (e de alguns passos de decode) por módulo, no corte, numa placa.

    python tests/bancada/perfil_prefill.py -m /workspace/corte --prefill-tokens 4096
    python tests/bancada/perfil_prefill.py -m /workspace/corte --prefill-tokens 4096 --tp

Envolve o `forward` de cada filho de cada bloco (atenção, MLP, normas, hyper-connections) e dos
netos (projeções, experts) com eventos CUDA e soma o tempo por classe. Sai uma tabela
"módulo · chamadas · ms · %" para o prefill e outra para o decode. Em TP os módulos rodam nos
processos dos ranks, então o perfil por módulo fica vazio e só o total vale: a diferença entre
o total em TP e o total numa placa é o que a comunicação e a divisão custam.
"""
import sys, os, argparse, time, collections
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from exllamav3 import Config, Model, Cache, Tokenizer

ENCHIMENTO = "A luz do sol atravessa a atmosfera e as moléculas de ar espalham mais os comprimentos de onda curtos do que os longos. "
FILHOS = ("attn_hc", "attn_norm", "attn", "attn_post_norm", "mlp_hc", "mlp_norm", "mlp", "mlp_post_norm", "input_norm")


class Perfil:
    def __init__(self):
        self.eventos = collections.defaultdict(list)

    def envolver(self, obj, metodo, rotulo):
        orig = getattr(obj, metodo)
        def f(*a, **k):
            e0, e1 = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
            e0.record()
            y = orig(*a, **k)
            e1.record()
            self.eventos[rotulo].append((e0, e1))
            return y
        setattr(obj, metodo, f)

    def zerar(self):
        self.eventos.clear()

    def tabela(self, titulo, total_ms):
        torch.cuda.synchronize()
        linhas = sorted(((sum(a.elapsed_time(b) for a, b in ev), len(ev), r) for r, ev in self.eventos.items()), reverse = True)
        print(f"\n{titulo}: total {total_ms:.1f} ms")
        print(f"{'módulo':<48} {'chamadas':>8} {'ms':>9} {'%':>6}")
        for ms, n, r in linhas:
            if "/" not in r:
                print(f"{r:<48} {n:>8} {ms:>9.1f} {100 * ms / total_ms:>5.1f}%")
        print("  dentro dos módulos (netos, incluídos acima):")
        for ms, n, r in [l for l in linhas if "/" in l[2]][:20]:
            print(f"  {r:<46} {n:>8} {ms:>9.1f} {100 * ms / total_ms:>5.1f}%")


def instrumentar(model, perfil):
    def nome(m):
        return f"{m.__class__.__name__}:{m.key.split('.')[-1]}" if getattr(m, "key", None) else m.__class__.__name__
    n = 0
    for m in model.modules:
        if not any(getattr(m, f, None) is not None for f in ("attn", "mlp")):
            perfil.envolver(m, "forward", nome(m)); n += 1
            continue
        for f in FILHOS:
            filho = getattr(m, f, None)
            if filho is None:
                continue
            if hasattr(filho, "mix"):
                perfil.envolver(filho, "mix", f"{filho.__class__.__name__}.mix")
                perfil.envolver(filho, "apply_", f"{filho.__class__.__name__}.apply_"); n += 2
                continue
            rot = filho.__class__.__name__
            perfil.envolver(filho, "forward", rot); n += 1
            for neto in getattr(filho, "modules", []):
                perfil.envolver(neto, "forward", f"{rot}/{nome(neto)}"); n += 1
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--model_dir", required = True)
    p.add_argument("--tp", action = "store_true")
    p.add_argument("--prefill-tokens", type = int, default = 4096)
    p.add_argument("--decode", type = int, default = 8, help = "passos de decode perfilados depois do prefill")
    p.add_argument("--cache", type = int, default = 8192)
    args = p.parse_args()

    config = Config.from_directory(args.model_dir)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = args.cache)
    model.load(tensor_p = args.tp, progressbar = True)
    tokenizer = Tokenizer.from_config(config)
    perfil = Perfil()
    if not args.tp:
        print(f"instrumentados {instrumentar(model, perfil)} pontos")

    n_ench = len(tokenizer.encode(ENCHIMENTO)[0])
    ids = tokenizer.encode(ENCHIMENTO * (args.prefill_tokens // n_ench + 1), encode_special_tokens = True)[:, :args.prefill_tokens]
    assert ids.shape[-1] + args.decode + 8 <= args.cache
    print(f"prompt: {ids.shape[-1]} tokens · dispositivos {model.active_devices}")

    # o prefill devolve os estados recorrentes (KDA) em params; o decode precisa deles
    def params(past_len, rs = None):
        p = {"attn_mode": "flash_attn", "cache": cache, "past_len": past_len, "batch_shape": (1, args.cache)}
        if rs is not None:
            p["recurrent_states"] = rs
        return p

    def liberar(rs):
        for r in rs or []:
            r.free()

    # aquecimento: compila Triton e grava grafos; não conta
    p = params(0)
    model.prefill(input_ids = ids[:, :64], params = p)
    rs = p.get("recurrent_states")
    model.forward(input_ids = ids[:, 64:65], params = params(64, rs))
    torch.cuda.synchronize(); liberar(rs); perfil.zerar()

    p = params(0)
    t0 = time.time()
    model.prefill(input_ids = ids[:, :-1], params = p)
    torch.cuda.synchronize()
    rs = p.get("recurrent_states")
    ms = (time.time() - t0) * 1000
    print(f"prefill: {ids.shape[-1] - 1} tokens em {ms:.0f} ms = {(ids.shape[-1] - 1) / ms * 1000:.0f} tok/s")
    if not args.tp:
        perfil.tabela("PREFILL por módulo", ms)
    perfil.zerar()

    x = ids
    torch.cuda.synchronize(); t0 = time.time()
    for i in range(args.decode):
        logits = model.forward(input_ids = x[:, -1:], params = params(x.shape[-1] - 1, rs))
        nxt = logits[0, -1].argmax().item()
        x = torch.cat((x, torch.tensor([[nxt]], dtype = x.dtype)), dim = -1)
    torch.cuda.synchronize()
    ms = (time.time() - t0) * 1000
    print(f"decode: {args.decode} passos em {ms:.0f} ms = {ms / args.decode:.1f} ms/token = {args.decode / ms * 1000:.1f} tok/s")
    if not args.tp:
        perfil.tabela(f"DECODE por módulo ({args.decode} passos)", ms)
    liberar(rs)
    print("FIM_PERFIL")


if __name__ == "__main__":
    main()
