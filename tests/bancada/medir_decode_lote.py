"""
Tempo por PASSO de decode com vários pedidos simultâneos, pelo Generator do jeito que o TabbyAPI usa
(alvo em TP, cabeça MTP como rascunho carregada fora do TP), com e sem as correções do decode em lote.

    # o que o hub roda: 8x RTX PRO 6000, nccl por PCIe
    NCCL_P2P_LEVEL=SYS python tests/bancada/medir_decode_lote.py -m /workspace/pesos \\
        --lotes 1,2,4,8,16 --modos base,lote --componentes --syncs

    # rápido, sem o perfil por componente (só parede por passo)
    NCCL_P2P_LEVEL=SYS python tests/bancada/medir_decode_lote.py -m /workspace/pesos --lotes 1,4

A pergunta: em 02/10/2026 o decode com 4 pedidos custava ~250 ms por passo contra ~21 ms com 1 (12x
para 4x o trabalho). Este roteiro mede o passo (gen.iterate() inteiro: rascunho MTP, verificação, amostragem,
absorção da MTP, rewinds) com o lote CHEIO -- todos os jobs já prefilados, nenhum na fila --, em contexto
curto (--prompt tokens de prompt, --novos gerados, sem stop: todo job vai até --novos).

Modos (alternados na MESMA carga, por globais de módulo em todos os ranks e atributos do gerador):
  base   tudo como no ramo de integração
  mla    + grafo da MLA kpool em lote no regime denso (EXL3_BC_MLA_KPOOL_LOTE; pede a extensão
           compilada deste ramo -- com a antiga o passo cai no eager e a contagem "recusa_ext" acusa)
         + plano agrupado do indexador num kernel só no eager (EXL3_POOL_KERNEL_EAGER)
  lote   mla + rewinds recorrentes numa ida aos ranks (EXL3_REWIND_LOTE)
             + absorção da MTP num prefill por comprimento aceito (EXL3_MTP_PREFILL_LOTE)

Saída por (lote, modo): ms por passo (mediana, p90), tokens por passo, tok/s somado e por pedido,
aceitação do rascunho, e as fases medidas no processo principal (host): rascunho, verificação,
absorção MTP, despachos ao TP. Com --componentes, o tempo por componente em cada rank (GPU exclusivo e host
inclusivo, util/perfil_componentes.py), e com --syncs quantas sincronizações host-placa cada componente fez
por passo. No fim, uma tabela base x lote e o JSON (--saida).
"""
import sys, os, argparse, time, json, datetime, statistics, random, collections
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler.presets import ArgmaxSampler
from exllamav3.util import perfil_componentes as pc

MODOS = {
    "base": dict(kpool_lote = False, pool_kernel = False, rewind_lote = False, mtp_prefill_lote = False),
    "mla":  dict(kpool_lote = True,  pool_kernel = True,  rewind_lote = False, mtp_prefill_lote = False),
    "lote": dict(kpool_lote = True,  pool_kernel = True,  rewind_lote = True,  mtp_prefill_lote = True),
}

ENCHIMENTO = ("A luz do sol atravessa a atmosfera e as moleculas de ar espalham mais os comprimentos de onda "
              "curtos do que os longos. Escreva uma explicacao longa e detalhada sobre isso. ")


def prompt_de(tokenizer, n, nonce):
    ids = tokenizer.encode(f"[{nonce:09d}] " + ENCHIMENTO * (n // 20 + 2), encode_special_tokens = True)
    return ids[:, :n]


class Fases:
    """Tempo de host por fase no processo principal, só enquanto `medindo`."""

    def __init__(self, sincronizar):
        self.medindo = False
        self.sincronizar = sincronizar
        self.t = collections.defaultdict(float)
        self.n = collections.Counter()

    def envolver(self, obj, metodo, rotulo):
        orig = getattr(obj, metodo)
        fs = self

        def f(*a, **k):
            if not fs.medindo:
                return orig(*a, **k)
            t0 = time.perf_counter()
            try:
                return orig(*a, **k)
            finally:
                if fs.sincronizar:
                    torch.cuda.synchronize()
                fs.t[rotulo] += time.perf_counter() - t0
                fs.n[rotulo] += 1
        setattr(obj, metodo, f)

    def zerar(self):
        self.t.clear()
        self.n.clear()


def em_todos(model, fn, *args):
    if model.loaded_tp:
        return model.tp_worker_dispatch_wait_multi(model.active_devices, fn, args)
    return [fn({"device": torch.device("cuda:0"), "modules": model.modules}, *args)]


def definir_modo(model, gen, modo):
    m = MODOS[modo]
    r = em_todos(model, pc.mp_definir_decode_lote, m["kpool_lote"], m["pool_kernel"])
    # o processo principal roda a cabeça MTP; no TP o rank de saída já é ele, sem TP também
    pc.mp_definir_decode_lote({}, m["kpool_lote"], m["pool_kernel"])
    gen.rewind_lote = m["rewind_lote"]
    gen.mtp_prefill_lote = m["mtp_prefill_lote"]
    return r


def rodar(gen, model, tokenizer, bsz, args, medir, fases, perf_local):
    """Um lote de bsz jobs até o fim. Devolve as medidas dos passos com o lote cheio (descontados os
    --aquecer primeiros), ou None se não medir."""
    jobs = [Job(input_ids = prompt_de(tokenizer, args.prompt, random.randrange(10**9)),
                max_new_tokens = args.novos if medir else args.novos_aquecimento,
                sampler = ArgmaxSampler()) for _ in range(bsz)]
    for j in jobs:
        gen.enqueue(j)
    cheios = 0
    passos = []
    perfil_ligado = False

    def ligar_perfil(ligado):
        if args.componentes and model.loaded_tp:
            em_todos(model, pc.mp_perfil_ativar, ligado)
        if perf_local is not None:
            pc.mp_perfil_ativar(perf_local, ligado)

    while gen.num_remaining_jobs():
        prontos = sum(1 for j in gen.active_jobs if j.is_prefill_done())
        cheio = prontos == bsz and gen.num_pending_jobs() == 0
        conta = medir and cheio and cheios >= args.aquecer
        if cheio:
            cheios += 1
        if conta != perfil_ligado:
            ligar_perfil(conta)
            perfil_ligado = conta
        fases.medindo = conta
        antes = sum(max(j.new_tokens, 0) for j in jobs)
        t0 = time.perf_counter()
        for r in gen.iterate():
            if r.get("stage") == "error":
                raise RuntimeError(f"job falhou: {r.get('error')}")
        dt = time.perf_counter() - t0
        if conta:
            passos.append((dt, sum(max(j.new_tokens, 0) for j in jobs) - antes))
    fases.medindo = False
    if perfil_ligado:
        ligar_perfil(False)
    if not medir:
        return None
    if not passos:
        raise RuntimeError(f"lote {bsz}: nenhum passo cheio medido (aumente --novos ou baixe --aquecer)")
    ms = sorted(dt * 1000 for dt, _ in passos)
    tok = sum(n for _, n in passos)
    tempo = sum(dt for dt, _ in passos)
    ac = sum(j.accepted_draft_tokens for j in jobs)
    rj = sum(j.rejected_draft_tokens for j in jobs)
    return {
        "passos": len(passos),
        "ms_mediana": statistics.median(ms),
        "ms_p10": ms[len(ms) // 10],
        "ms_p90": ms[min(len(ms) - 1, len(ms) * 9 // 10)],
        "tokens_por_passo": tok / len(passos),
        "tok_s_soma": tok / tempo,
        "tok_s_por_pedido": tok / tempo / bsz,
        "aceitacao": ac / max(1, ac + rj),
    }


def colher_componentes(model, perf_local, passos, syncs):
    out = {}
    if model.loaded_tp:
        if syncs:
            out["syncs_por_rank"] = [{k: v / passos for k, v in d.items()}
                                     for d in em_todos(model, pc.mp_perfil_syncs)]
        tabs = em_todos(model, pc.mp_perfil_colher)
        out["por_rank"] = [{k: [v[0] / passos, v[1] / passos, v[2] / passos, v[3] / passos]
                            for k, v in t.items()} for t in tabs]
        out["kpool_lote"] = em_todos(model, pc.mp_contagem_kpool_lote)
    elif syncs:
        out["syncs_por_rank"] = [{k: v / passos for k, v in pc.mp_perfil_syncs({}).items()}]
    if perf_local is not None:
        t = pc.mp_perfil_colher(perf_local)
        out["mtp"] = {k: [v[0] / passos, v[1] / passos, v[2] / passos, v[3] / passos] for k, v in t.items()}
    return out


def imprimir(bsz, modo, r, fases_ms, comp):
    print(f"  lote {bsz:>2} {modo:<5} {r['ms_mediana']:>8.1f} ms/passo (p10 {r['ms_p10']:.1f} p90 {r['ms_p90']:.1f}, "
          f"{r['passos']} passos) · {r['tokens_por_passo']:.2f} tok/passo · {r['tok_s_soma']:>6.1f} tok/s soma · "
          f"{r['tok_s_por_pedido']:>5.1f} por pedido · aceitação {100 * r['aceitacao']:.0f} %")
    if fases_ms:
        print("      fases (host, ms/passo): " + " · ".join(f"{k} {v:.1f}" for k, v in fases_ms.items()))
    if not comp:
        return
    if "por_rank" in comp:
        rot = sorted({k for t in comp["por_rank"] for k in t}, key = lambda k: -max(t.get(k, [0, 0, 0, 0])[2] for t in comp["por_rank"]))
        print(f"      {'componente':<22} {'chamadas':>8} {'GPU excl máx':>13} {'host incl máx':>14}")
        for k in rot:
            n = max(t.get(k, [0, 0, 0, 0])[0] for t in comp["por_rank"])
            g = max(t.get(k, [0, 0, 0, 0])[2] for t in comp["por_rank"])
            h = max(t.get(k, [0, 0, 0, 0])[3] for t in comp["por_rank"])
            print(f"      {k:<22} {n:>8.1f} {g:>13.2f} {h:>14.2f}")
    if comp.get("mtp"):
        print("      cabeça MTP (processo principal): " + " · ".join(
            f"{k} GPU {v[2]:.2f} host {v[3]:.2f}" for k, v in sorted(comp["mtp"].items(), key = lambda kv: -kv[1][3])))
    if comp.get("syncs_por_rank"):
        s0 = comp["syncs_por_rank"][-1]   # rank de saída = processo principal (gerador + MTP)
        outros = comp["syncs_por_rank"][:-1]
        print("      syncs/passo, processo principal: " + (" · ".join(f"{k} {v:.1f}" for k, v in sorted(s0.items(), key = lambda kv: -kv[1])) or "nenhuma"))
        if outros:
            soma = collections.Counter()
            for d in outros:
                for k, v in d.items():
                    soma[k] = max(soma[k], v)
            print("      syncs/passo, demais ranks (máx): " + (" · ".join(f"{k} {v:.1f}" for k, v in soma.most_common()) or "nenhuma"))
    if comp.get("kpool_lote"):
        print(f"      MLA kpool bsz>1 (rank de saída): {comp['kpool_lote'][-1]}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--model_dir", required = True)
    p.add_argument("--lotes", default = "1,2,4,8,16")
    p.add_argument("--modos", default = "base,lote", help = "de: " + ",".join(MODOS))
    p.add_argument("--prompt", type = int, default = 44, help = "tokens de prompt por pedido")
    p.add_argument("--novos", type = int, default = 160, help = "tokens gerados por pedido na rodada medida")
    p.add_argument("--novos-aquecimento", type = int, default = 24)
    p.add_argument("--aquecer", type = int, default = 4, help = "passos cheios descartados no começo da rodada medida")
    p.add_argument("--rascunho", type = int, default = 3, help = "tokens de rascunho MTP (0 = sem rascunho)")
    p.add_argument("--rascunho-device", type = int, default = None, help = "placa da cabeça MTP (padrão: autosplit, como o TabbyAPI)")
    p.add_argument("--backend", default = "nccl")
    p.add_argument("--sem-tp", action = "store_true")
    p.add_argument("--cache", type = int, default = 32768)
    p.add_argument("--cache-bits", type = int, default = 0, help = "cache quantizado (o mesmo cache_mode da produção)")
    p.add_argument("--chunk", type = int, default = 2048)
    p.add_argument("--componentes", action = "store_true", help = "perfil por componente em cada rank (eventos CUDA)")
    p.add_argument("--syncs", action = "store_true", help = "conta sincronizações host-placa por componente")
    p.add_argument("--fases-sincronizadas", action = "store_true",
                   help = "synchronize no fim de cada fase do gerador (atribui a GPU à fase; muda a sobreposição)")
    p.add_argument("--saida", default = None)
    args = p.parse_args()
    lotes = [int(x) for x in args.lotes.split(",")]
    modos = args.modos.split(",")
    for m in modos:
        assert m in MODOS, f"modo desconhecido: {m}"
    max_lote = max(lotes)
    assert max_lote * (args.prompt + args.novos + 2 * 256) <= args.cache, "cache pequeno para o maior lote"

    torch.manual_seed(0)
    random.seed(0)
    cfg = Config.from_directory(args.model_dir)
    model = Model.from_config(cfg)
    extra = {}
    if args.cache_bits:
        from exllamav3.cache import CacheLayer_quant
        extra = dict(layer_type = CacheLayer_quant, k_bits = args.cache_bits, v_bits = args.cache_bits)
    cache = Cache(model, max_num_tokens = args.cache, max_batch_size = max_lote, max_history = args.rascunho, **extra)
    draft = dcache = None
    if args.rascunho:
        draft = Model.from_config(cfg, component = "mtp")
        dcache = Cache(draft, max_num_tokens = args.cache, max_batch_size = max_lote, max_history = args.rascunho, **extra)
    t0 = time.time()
    # Ordem do TabbyAPI: a cabeça MTP primeiro, fora do TP; depois o alvo
    if draft is not None:
        if args.rascunho_device is not None:
            draft.load(device = torch.device(f"cuda:{args.rascunho_device}"), progressbar = True)
        else:
            draft.load(progressbar = True)
    if args.sem_tp:
        model.load(progressbar = True)
    else:
        model.load(tensor_p = True, tp_backend = args.backend, progressbar = True,
                   max_chunk_size = args.chunk, max_batch_size = max_lote)
    print(f"carga: {time.time() - t0:.0f} s · dispositivos {model.active_devices} · TP {model.loaded_tp} · "
          f"rascunho {'MTP x' + str(args.rascunho) if draft else 'não'}")
    tokenizer = Tokenizer.from_config(cfg)
    gen = Generator(model = model, cache = cache, tokenizer = tokenizer, max_batch_size = max_lote,
                    max_chunk_size = args.chunk, draft_model = draft, draft_cache = dcache,
                    num_draft_tokens = args.rascunho or None)

    fases = Fases(args.fases_sincronizadas)
    fases.envolver(gen, "iterate_gen", "iterate_gen")
    fases.envolver(model, "forward", "verificacao")
    if draft is not None:
        fases.envolver(gen, "iterate_draftmodel_mtp_gen", "rascunho")
        fases.envolver(draft, "forward", "rascunho.forward")
        fases.envolver(draft, "prefill", "mtp_prefill")
    if model.loaded_tp:
        fases.envolver(model, "tp_dispatch_all", "tp_dispatch_all")
        fases.envolver(model, "tp_dispatch_lm_head_argmax", "lm_head_argmax")

    perf_local = None
    if args.componentes:
        if model.loaded_tp:
            n = em_todos(model, pc.mp_perfil_instrumentar)
            print(f"instrumentados: {n} pontos por rank")
            em_todos(model, pc.mp_perfil_ativar, False)
        if draft is not None:
            dev = next((m.device for m in draft.modules if getattr(m, "device", None) is not None), "cuda:0")
            perf_local = pc.instrumentar_local(draft.modules, dev)
            pc.mp_perfil_ativar(perf_local, False)
    if args.syncs:
        if model.loaded_tp:
            em_todos(model, pc.mp_perfil_contar_syncs, True)
        else:
            pc.mp_perfil_contar_syncs({}, True)

    resultado = {"modelo": args.model_dir, "dispositivos": list(model.active_devices), "args": vars(args),
                 "lotes": {}}
    for bsz in lotes:
        resultado["lotes"][bsz] = {}
        for modo in modos:
            estado = definir_modo(model, gen, modo)
            rodar(gen, model, tokenizer, bsz, args, False, fases, None)   # compila / captura as formas
            if args.componentes and model.loaded_tp:
                em_todos(model, pc.mp_perfil_zerar)
                em_todos(model, pc.mp_contagem_kpool_lote)
            if perf_local is not None:
                pc.mp_perfil_zerar(perf_local)
            if args.syncs:
                em_todos(model, pc.mp_perfil_syncs)
            fases.zerar()
            r = rodar(gen, model, tokenizer, bsz, args, True, fases, perf_local)
            fases_ms = {k: v * 1000 / r["passos"] for k, v in fases.t.items()}
            comp = colher_componentes(model, perf_local, r["passos"], args.syncs) \
                if (args.componentes or args.syncs) else None
            imprimir(bsz, modo, r, fases_ms, comp)
            resultado["lotes"][bsz][modo] = {"medidas": r, "fases_ms_por_passo": fases_ms,
                                             "flags_por_rank": estado, "componentes": comp}

    print("\n=== resumo: ms por passo / tok/s somado ===")
    cab = f"{'lote':>4}" + "".join(f" {m + ' ms':>11} {m + ' tok/s':>12}" for m in modos)
    if "base" in modos and len(modos) > 1:
        cab += "".join(f" {'x ' + m:>8}" for m in modos if m != "base")
    print(cab)
    for bsz in lotes:
        lin = f"{bsz:>4}"
        for m in modos:
            r = resultado["lotes"][bsz][m]["medidas"]
            lin += f" {r['ms_mediana']:>11.1f} {r['tok_s_soma']:>12.1f}"
        if "base" in modos and len(modos) > 1:
            b = resultado["lotes"][bsz]["base"]["medidas"]["tok_s_soma"]
            for m in modos:
                if m != "base":
                    lin += f" {resultado['lotes'][bsz][m]['medidas']['tok_s_soma'] / b:>8.2f}"
        print(lin)

    saida = args.saida or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "saidas",
        datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") + "-decode-lote")
    os.makedirs(saida, exist_ok = True)
    with open(os.path.join(saida, "resultado.json"), "w") as f:
        json.dump(resultado, f, indent = 1, default = str)
    print(f"\nresultado: {saida}/resultado.json")
    if args.componentes and model.loaded_tp:
        em_todos(model, pc.mp_perfil_desinstrumentar)
    if args.syncs and model.loaded_tp:
        em_todos(model, pc.mp_perfil_contar_syncs, False)
    print("FIM_DECODE_LOTE")


if __name__ == "__main__":
    main()
