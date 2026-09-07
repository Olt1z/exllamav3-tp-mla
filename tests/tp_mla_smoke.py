"""
Teste de fumaça do tensor parallel na MLAttention.

Gera N tokens greedy a partir de um prompt fixo com forward direto (prefill + decode token a
token) e grava os logits de cada passo. Rodado duas vezes:

    # linha de base, uma placa
    python tests/tp_mla_smoke.py -m /modelo --save base.pt
    # ruído do motor (o ExLlamaV3 não é determinístico bit a bit)
    python tests/tp_mla_smoke.py -m /modelo --compare base.pt
    # TP em todas as placas, mesmos tokens da base (teacher forcing), compara os logits
    python tests/tp_mla_smoke.py -m /modelo --tp --compare base.pt --max-kl 0.01
    # repetir os três com --prefill-tokens 3000 para exercitar o indexador DSA esparso

Sem --prefill-tokens o contexto fica abaixo de index_topk e o DSA roda denso: o k_norm do
indexador, o top-k por rank e o kernel esparso só são provados com o contexto longo.

Com --compare, os tokens alimentados são os da base, então os logits são comparáveis posição
a posição: KL média/máxima de base -> atual, diferença absoluta máxima e concordância do top-1.
Durante o decode amostra o uso de cada GPU pelo nvidia-smi e mede tokens/s. Sai com 1 se
--max-kl for dado e a KL média passar dele.
"""
import sys, os, argparse, subprocess, threading, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from exllamav3 import Config, Model, Cache, Tokenizer

PERGUNTA = "Explique em três frases por que o céu é azul e por que o pôr do sol é avermelhado."
# Texto de enchimento para levar o contexto acima de index_topk e exercitar o caminho esparso
# do indexador DSA (com --prefill-tokens); abaixo desse limite a seleção é all-inclusive
ENCHIMENTO = (
    "A luz do sol atravessa a atmosfera e as moléculas de ar espalham mais os comprimentos de "
    "onda curtos do que os longos. "
)


class GpuSampler(threading.Thread):
    def __init__(self, period = 0.2):
        super().__init__(daemon = True)
        self.period = period
        self.samples = []
        self.stop = threading.Event()

    def run(self):
        while not self.stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output = True, text = True, timeout = 2,
                ).stdout
                self.samples.append([int(v) for v in out.split()])
            except Exception:
                pass
            time.sleep(self.period)

    def mean_per_gpu(self):
        if not self.samples:
            return []
        n = min(len(s) for s in self.samples)
        return [sum(s[i] for s in self.samples) / len(self.samples) for i in range(n)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--model_dir", required = True)
    p.add_argument("--tp", action = "store_true", help = "carregar com tensor_p = True")
    p.add_argument("--tokens", type = int, default = 64)
    p.add_argument("--cache", type = int, default = 4096)
    p.add_argument("--prefill-tokens", type = int, default = 0,
                   help = "enche o prompt até este tamanho (ex.: 3000 passa do index_topk 2048 do GLM-5.3)")
    p.add_argument("--save", help = "grava tokens e logits do decode neste arquivo")
    p.add_argument("--compare", help = "arquivo de --save para comparar (teacher forcing)")
    p.add_argument("--max-kl", type = float, help = "falha se a KL média passar deste valor")
    p.add_argument("--tp-moe-ts", action = "store_true", help = "tensor split nos experts em vez de expert parallel")
    p.add_argument("--cache-bits", type = int, default = 0, help = "cache quantizado (2 a 8 bits); na MLA é a largura do latente")
    p.add_argument("--tp-dev-limits", default = None,
                   help = "paralelismo máximo por classe, ex.: 'attn=1' (só experts/MLP divididos) ou 'moe=1,mlp=1,linear=1' (só a atenção dividida)")
    p.add_argument("--simular-fio-bf16", action = "store_true",
                   help = "numa placa só: arredonda a saída fp32 de cada atenção e MLP para bf16, como o all_reduce do TP faz "
                          "(all_reduce_cpu.cu: 'fp32 payloads keep the bf16 wire'); dá a régua justa para comparar com o TP")
    args = p.parse_args()
    tp_dev_limits = None
    if args.tp_dev_limits:
        tp_dev_limits = {k: int(v) for k, v in (par.split("=") for par in args.tp_dev_limits.split(","))}

    base = torch.load(args.compare) if args.compare else None
    n_tokens = base["tokens"].shape[0] if base is not None else args.tokens

    config = Config.from_directory(args.model_dir)
    model = Model.from_config(config)
    if args.cache_bits:
        from exllamav3.cache import CacheLayer_quant
        cache = Cache(model, max_num_tokens = args.cache, layer_type = CacheLayer_quant,
                      k_bits = args.cache_bits, v_bits = args.cache_bits)
    else:
        cache = Cache(model, max_num_tokens = args.cache)
    camadas = getattr(cache, "layers", None) or getattr(cache, "cache_layers", None)
    if camadas:
        try:
            print(f"cache: {type(camadas[0]).__name__}, {sum(c.storage_size() for c in camadas) / 2**20:.1f} MiB para {args.cache} tokens")
        except Exception as e:
            print(f"cache: {type(camadas[0]).__name__} ({e})")
    t0 = time.time()
    model.load(
        tensor_p = args.tp, progressbar = True, verbose = args.tp,
        tp_options = {"moe_tensor_split": True} if args.tp_moe_ts else None,
        tp_dev_limits = tp_dev_limits,
    )
    print(f"carga: {time.time() - t0:.0f} s, dispositivos {model.active_devices}")
    if args.simular_fio_bf16:
        n = 0
        for blk in model.modules:
            for nome in ("attn", "mlp"):
                sub = getattr(blk, nome, None)
                if sub is None or not hasattr(sub, "forward"):
                    continue
                def fio(orig):
                    def f(x, params, *a, **k):
                        y = orig(x, params, *a, **k)
                        return y.to(torch.bfloat16).to(y.dtype) if y.dtype == torch.float32 else y
                    return f
                sub.forward = fio(sub.forward)
                n += 1
        print(f"fio bf16 simulado em {n} submódulos")
    tokenizer = Tokenizer.from_config(config)

    # Template de chat da própria arquitetura: o mesmo teste serve ao GLM, ao Flash e ao DeepSeek
    pergunta = PERGUNTA
    if args.prefill_tokens:
        n_ench = len(tokenizer.encode(ENCHIMENTO)[0])
        pergunta = ENCHIMENTO * (args.prefill_tokens // n_ench + 1) + PERGUNTA
    ids = tokenizer.encode(model.default_chat_prompt(pergunta), encode_special_tokens = True)
    assert ids.shape[-1] + n_tokens <= args.cache, "prompt + tokens não cabem no --cache"
    print(f"prompt: {ids.shape[-1]} tokens")
    params = {"attn_mode": "flash_attn", "cache": cache, "past_len": 0, "batch_shape": (1, args.cache)}
    model.prefill(input_ids = ids[:, :-1], params = params)
    recurrent_states = params.get("recurrent_states")

    logits_all = []
    tokens = []
    sampler = GpuSampler()
    sampler.start()
    # Os primeiros passos pagam compilação de kernels e captura de grafos; o tok/s mede os demais
    AQUECIMENTO = min(16, n_tokens // 2)
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(n_tokens):
        if i == AQUECIMENTO:
            torch.cuda.synchronize()
            t0 = time.time()
        params = {
            "attn_mode": "flash_attn", "cache": cache, "past_len": ids.shape[-1] - 1,
            "batch_shape": (1, args.cache), "recurrent_states": recurrent_states,
        }
        logits = model.forward(input_ids = ids[:, -1:], params = params)[0, -1].float().cpu()
        logits_all.append(logits)
        nxt = base["tokens"][i].item() if base is not None else int(logits.argmax())
        tokens.append(nxt)
        ids = torch.cat((ids, torch.tensor([[nxt]], dtype = ids.dtype)), dim = -1)
    torch.cuda.synchronize()
    dt = time.time() - t0
    sampler.stop.set()
    if recurrent_states:
        for rs in recurrent_states:
            rs.free()

    logits_all = torch.stack(logits_all)
    tokens = torch.tensor(tokens)
    print("texto:", tokenizer.decode(tokens.unsqueeze(0))[0].replace("\n", "\\n"))
    medidos = n_tokens - AQUECIMENTO
    print(f"decode: {medidos} tokens (após {AQUECIMENTO} de aquecimento) em {dt:.2f} s = {medidos / dt:.2f} tok/s")
    print("uso médio por GPU (%):", [f"{u:.0f}" for u in sampler.mean_per_gpu()])

    if args.save:
        torch.save({"tokens": tokens, "logits": logits_all}, args.save)
        print("gravado:", args.save)

    if base is not None:
        v = min(logits_all.shape[1], base["logits"].shape[1])
        a, b = base["logits"][:, :v], logits_all[:, :v]
        lp_a, lp_b = a.log_softmax(-1), b.log_softmax(-1)
        kl = (lp_a.exp() * (lp_a - lp_b)).sum(-1)
        top1 = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
        print(f"KL média {kl.mean():.5f}  KL máx {kl.max():.5f}  |dif| máx {(a - b).abs().max():.3f}  top-1 igual {top1:.1%}")
        if args.max_kl is not None and kl.mean() > args.max_kl:
            print(f"FALHOU: KL média {kl.mean():.5f} > {args.max_kl}")
            sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
