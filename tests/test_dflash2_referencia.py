"""
Prova do leitor DFlash 2 do fork contra a implementação de referência (z-lab/dflash, transformers).

    python tests/test_dflash2_referencia.py --alvo /workspace/corte --dflash2 /workspace/dflash2 \
        --referencia /workspace/dflash-ref/dflash/model.py [--ctx 200] [--tokens 64]

A. Mesmas entradas nos dois lados: taps aleatórios do alvo para `ctx` posições e o bloco
   [âncora, máscaras]. Compara o estado final do rascunho (posições 1..7) e o caminho que o
   seletor escolhe. O lm_head e o embedding são os do alvo (o corte), nos dois lados.
B. Mecânica no gerador do ExLlamaV3: o corte como alvo e o DFlash 2 como rascunho, 64 tokens
   greedy, contra o corte sozinho. O corte gera lixo (4 camadas) e o rascunho foi treinado para
   o modelo inteiro, então a aceitação aqui não diz nada; só prova que o caminho roda de ponta a
   ponta sem quebrar e que a saída continua a do alvo (lossless).
"""
import sys, os, argparse, importlib.util, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator
from exllamav3.constants import PAGE_SIZE


def carregar_referencia(caminho):
    spec = importlib.util.spec_from_file_location("dflash_ref_model", caminho)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alvo", required = True, help = "modelo alvo (o corte do Flash serve)")
    p.add_argument("--dflash2", required = True, help = "checkpoint incoai/*-DFlash2")
    p.add_argument("--referencia", required = True, help = "dflash/model.py do z-lab/dflash")
    p.add_argument("--ctx", type = int, default = 200)
    p.add_argument("--tokens", type = int, default = 64)
    p.add_argument("--cache", type = int, default = 4096)
    args = p.parse_args()
    dev = torch.device("cuda:0")

    # --- alvo e rascunho no fork --------------------------------------------------------------
    cfg_d = Config.from_directory(args.dflash2)
    draft = Model.from_config(cfg_d)
    dcache = Cache(draft, max_num_tokens = args.cache)
    draft.load(device = dev, progressbar = True)
    print(f"rascunho: {cfg_d.arch_string}, {cfg_d.num_hidden_layers} camadas, bloco {cfg_d.block_size}, "
          f"taps {cfg_d.target_layer_ids} (com tap_shift), top-k {cfg_d.selector_top_k}, rank {cfg_d.selector_rank}")

    cfg_t = Config.from_directory(args.alvo)
    target = Model.from_config(cfg_t)
    tcache = Cache(target, max_num_tokens = args.cache, max_history = draft.caps.get("default_draft_size", 4))
    target.load(device = dev, progressbar = True)
    tokenizer = Tokenizer.from_config(cfg_t)
    draft.attach_to(target)
    assert cfg_t.hidden_size * len(cfg_d.target_layer_ids) == draft.input_layer.target_state_size, \
        "hidden do alvo não casa com o fc do rascunho"

    # --- A. mesmas entradas, fork × referência ------------------------------------------------
    torch.manual_seed(0)
    L, H = args.ctx, cfg_t.hidden_size
    taps = [torch.randn(1, L, H, device = dev, dtype = torch.half) * 4 for _ in cfg_d.target_layer_ids]
    ancora = torch.tensor([[1234]], dtype = torch.long)
    pages = (L + cfg_d.block_size + PAGE_SIZE - 1) // PAGE_SIZE
    bt = torch.arange(pages, dtype = torch.int32).view(1, -1)

    draft.update_kv_from_target(list(taps), dcache, {"block_table": bt, "cache_seqlens": torch.tensor([0], dtype = torch.int32)})
    params = {"attn_mode": "flash_attn", "block_table": bt, "cache": dcache,
              "cache_seqlens": torch.tensor([L], dtype = torch.int32), "export_draft_conf": True}
    torch.cuda.synchronize(); t0 = time.time()
    state = draft.forward(input_ids = ancora, params = params)
    ids = draft.sample_from_state(state, params)
    torch.cuda.synchronize()
    print(f"fork: estado {tuple(state.shape)} {state.dtype}, ids {ids[0].tolist()}, "
          f"conf {[round(c, 2) for c in params['draft_conf'][0].tolist()]}, {(time.time() - t0) * 1000:.1f} ms")

    ref_mod = carregar_referencia(args.referencia)
    ref = ref_mod.DFlash2DraftModel.from_pretrained(args.dflash2, dtype = torch.bfloat16).to(dev).eval()
    ref_ids = getattr(ref, "target_layer_ids")
    print(f"referência: taps {ref_ids} (crus) → fork usa {cfg_d.target_layer_ids}")
    bloco = torch.cat((ancora, torch.full((1, cfg_d.block_size - 1), cfg_d.mask_token_id, dtype = torch.long)), dim = 1)
    emb = target.modules[0].forward(bloco.to(dev), {}).to(dev)
    th = torch.cat(taps, dim = -1).to(torch.bfloat16)
    pos = torch.arange(L + cfg_d.block_size, device = dev)[None]
    from transformers import DynamicCache
    past = DynamicCache(config = ref.config)
    with torch.inference_mode():
        out_ref = ref(target_hidden = th, noise_embedding = emb.to(torch.bfloat16), position_ids = pos,
                      past_key_values = past, use_cache = True)
        lm = target.modules[target.logit_layer_idx]
        cabeca = lambda h: lm.forward(h.to(torch.half), {})[..., :cfg_t.vocab_size].float()
        tok_ref, cands_ref, _ = ref.propose(out_ref[:, 1:], bloco[:, 0].to(dev), cabeca, 0.0)

    a, b = state[:, 1:].float(), out_ref[:, 1:].float()
    dif = (a - b).abs()
    rel = dif.max() / b.abs().max().clamp_min(1e-6)
    cos = torch.nn.functional.cosine_similarity(a.flatten(1), b.flatten(1)).item()
    iguais = (ids[:, 1:].to(dev) == tok_ref).float().mean().item()
    print(f"A. estado: |dif| máx {dif.max():.4f} · rel {rel:.2e} · cosseno {cos:.5f}")
    print(f"A. caminho: fork {ids[0, 1:].tolist()} · ref {tok_ref[0].tolist()} · iguais {iguais:.0%}")
    ok_a = cos > 0.999 and iguais >= 6 / 7

    # --- B. gerador de ponta a ponta -------------------------------------------------------------
    del ref, past, out_ref
    torch.cuda.empty_cache()
    prompt = "Explique em três frases por que o céu é azul."
    from exllamav3.generator.sampler.presets import ArgmaxSampler
    gen = Generator(model = target, cache = tcache, tokenizer = tokenizer, draft_model = draft, draft_cache = dcache,
                    max_batch_size = 1, record_draft_stats = True)
    t0 = time.time()
    com = gen.generate(prompt = prompt, max_new_tokens = args.tokens, sampler = ArgmaxSampler(), completion_only = True)
    t_com = time.time() - t0
    stats = []
    for job in getattr(gen, "finished_jobs", []) or []:
        stats += getattr(job, "draft_stats", []) or []
    del gen
    tcache2 = Cache(target, max_num_tokens = args.cache)
    gen2 = Generator(model = target, cache = tcache2, tokenizer = tokenizer, max_batch_size = 1)
    t0 = time.time()
    sem = gen2.generate(prompt = prompt, max_new_tokens = args.tokens, sampler = ArgmaxSampler(), completion_only = True)
    t_sem = time.time() - t0
    ids_com = tokenizer.encode(com)[0].tolist()
    ids_sem = tokenizer.encode(sem)[0].tolist()
    n = min(len(ids_com), len(ids_sem))
    iguais_b = sum(1 for i in range(n) if ids_com[i] == ids_sem[i]) / max(n, 1)
    print(f"B. com rascunho: {len(ids_com)} tokens em {t_com:.2f} s · sem: {len(ids_sem)} em {t_sem:.2f} s · "
          f"tokens iguais {iguais_b:.0%} (o motor não é determinístico; o corte gera lixo)")
    if stats:
        aceitos = sum(s[2] for s in stats); janelas = sum(s[1] for s in stats)
        print(f"B. rascunho: {len(stats)} rodadas, {aceitos}/{janelas} aceitos")
    print("texto com rascunho:", com[:200].replace("\n", "\\n"))
    print("OK" if ok_a else "FALHOU: estado ou caminho divergem da referência")
    sys.exit(0 if ok_a else 1)


if __name__ == "__main__":
    main()
