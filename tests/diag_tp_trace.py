"""
Rastreia o estado oculto módulo a módulo (entrada e saída de cada módulo de topo) num prefill
seguido de um passo de decode, no caminho de uma placa ou no TP, e grava em arquivo. Rodar duas
vezes e comparar com diag_tp_compare.py acha o primeiro módulo em que os dois caminhos divergem.
No modo TP também mede se o all_reduce do backend altera um tensor com um rank só (fp32 e fp16).

    CUDA_VISIBLE_DEVICES=0 python3 tests/diag_tp_trace.py /workspace/corte /workspace/ls.pt
    CUDA_VISIBLE_DEVICES=0 python3 tests/diag_tp_trace.py /workspace/corte /workspace/tp.pt --tp
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

PERGUNTA = "Explique em três frases por que o céu é azul e por que o pôr do sol é avermelhado."


def main():
    from exllamav3 import Config, Model, Cache, Tokenizer
    tp = "--tp" in sys.argv
    config = Config.from_directory(sys.argv[1])
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = 4096)
    model.load(tensor_p = tp)
    if tp:
        ctx = model.mp_parent_conn[model.tp_output_device].local_context
        mods = ctx["modules"]
        backend = ctx["backend"]
        for dt in (torch.float32, torch.float16):
            t = torch.randn(1, 1, config.hidden_size, dtype = dt, device = model.tp_output_device) * 3
            t0 = t.clone()
            backend.all_reduce(t)
            print(f"all_reduce com um rank, {dt}: |dif| máx {(t.float() - t0.float()).abs().max():.6f}  "
                  f"rel {(t.float() - t0.float()).norm() / t0.float().norm():.2e}")
    else:
        mods = model.modules

    rec = {}
    def wrap(i, m):
        orig = m.forward
        def f(x, params, *a, **k):
            xin = x.detach().float().cpu() if torch.is_tensor(x) else None
            y = orig(x, params, *a, **k)
            rec.setdefault(i, []).append({
                "nome": f"{type(m).__name__} {getattr(m, 'key', '')}",
                "in": xin,
                "out": y.detach().float().cpu() if torch.is_tensor(y) else None,
                "params": {k2: (v if isinstance(v, (int, float, bool, str)) else
                                (tuple(v.shape), str(v.dtype), v.flatten()[:4].tolist()) if torch.is_tensor(v) else type(v).__name__)
                           for k2, v in params.items() if k2 in ("position", "positions", "cache_seqlens", "block_table", "prefill", "causal", "attn_mode")},
            })
            return y
        m.forward = f
    for i, m in enumerate(mods):
        wrap(i, m)

    tok = Tokenizer.from_config(config)
    ids = tok.encode(model.default_chat_prompt(PERGUNTA), encode_special_tokens = True)
    params = {"attn_mode": "flash_attn", "cache": cache, "past_len": 0, "batch_shape": (1, 4096)}
    model.prefill(input_ids = ids[:, :-1], params = params)
    params = {"attn_mode": "flash_attn", "cache": cache, "past_len": ids.shape[-1] - 1, "batch_shape": (1, 4096)}
    logits = model.forward(input_ids = ids[:, -1:], params = params)
    torch.save({"rec": rec, "logits": logits.float().cpu(), "tp": tp}, sys.argv[2])
    print("gravado", sys.argv[2], "módulos", len(mods), "chamadas", sum(len(v) for v in rec.values()))


if __name__ == "__main__":
    main()
