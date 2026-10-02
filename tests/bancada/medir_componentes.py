"""
Tempo do prefill por COMPONENTE em TP, a contexto longo, com e sem o indexador DSA dividido.

    # 4x RTX PRO 6000 por PCIe, como a produção (o hub liga NCCL_P2P_LEVEL=SYS)
    NCCL_P2P_LEVEL=SYS python tests/bancada/medir_componentes.py -m /workspace/modelo \\
        --contextos 65536,917504 --chunk 4096 --medir 2 --conferir

    # só 64k, rápido (uns 2 min de enchimento)
    NCCL_P2P_LEVEL=SYS python tests/bancada/medir_componentes.py -m /workspace/modelo --contextos 65536

    # linha do tempo no nsys, com as regiões NVTX de cada componente em cada rank
    NCCL_P2P_LEVEL=SYS nsys profile -t cuda,nvtx,osrt --trace-fork-before-exec=true \\
        --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown -o prefill_900k \\
        python tests/bancada/medir_componentes.py -m /workspace/modelo --contextos 917504 --nvtx --nsys

O que mede. Enche o cache até cada contexto C pelo caminho de sempre (chunks de --chunk), guarda o
estado recorrente do KDA em C e roda --medir chunks a partir de C duas vezes -- indexador REPLICADO
(o de hoje) e DIVIDIDO (EXL3_INDEXADOR_DIVIDIDO) --, restaurando o estado do KDA entre as duas e
reescrevendo as mesmas posições do cache, então as duas rodadas veem exatamente o mesmo contexto.
Antes de cada rodada vai um chunk de aquecimento que não conta (primeiro all-gather int32 do NCCL,
alocações novas).

Em cada rank, cada componente é cercado por eventos CUDA (exllamav3/util/perfil_componentes.py):
indexador (chaves, pontuação + top-k, all-gather das faixas), atenção esparsa, resto da MLA, KDA,
MoE, hyper-connections, normas e os coletivos. Sai, por componente, o tempo EXCLUSIVO por chunk no
rank mais lento e a média entre ranks, nas duas rodadas, e o tok/s de parede.

Como ler, para decidir o próximo passo:
  - `all_reduce` alto em relação ao resto = vale sobrepor o all-reduce ao cálculo (async_op, stream
    de comunicação). O número já inclui a espera pelo rank mais lento.
  - `hc` alto = vale o sequence-parallel das hyper-connections (hoje replicadas: cada rank faz o HC
    das 4096 linhas inteiras).
  - `indexador_topk` replicado vs dividido = o ganho deste branch; `indexador_reunir` + `all_gather`
    é o que ele custa.

--conferir: digest dos índices de cada camada "full" em cada rank. No dividido os ranks TÊM de bater
byte a byte (sai com 1 se não); no replicado conta quantas camadas divergem entre ranks (o
quase-empate que o dividido elimina) e, entre os modos, a sobreposição média das seleções de uma
camada.

Resultado também em JSON (--saida, padrão tests/bancada/saidas/<data>-componentes/).
"""
import sys, os, argparse, time, json, datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.util import perfil_componentes as pc

FRASES = (
    "A luz do sol atravessa a atmosfera e as moléculas de ar espalham mais os comprimentos de onda curtos do que os longos.",
    "No inverno o rio baixa e as pedras do leito aparecem, cobertas de musgo e de folhas secas trazidas pela chuva.",
    "O relatório trimestral mostrou queda nas vendas do norte e alta no sul, compensada pelo câmbio favorável.",
    "Para compilar o projeto é preciso instalar as dependências, configurar o caminho do compilador e rodar os testes.",
    "A receita pede três ovos, duzentos gramas de farinha, uma xícara de leite e uma pitada de sal.",
    "Os astrônomos observaram a estrela durante seis noites seguidas e registraram variações de brilho a cada hora.",
    "O tribunal adiou a sessão porque uma das testemunhas não compareceu e o advogado pediu nova data.",
)

ORDEM = ("indexador_topk", "indexador_chaves", "indexador_reunir", "all_gather", "atencao_esparsa",
         "mla", "kda", "moe", "mlp_denso", "hc", "normas", "all_reduce", "reduce_scatter")


def texto_de(n_tokens, tokenizer, arquivo = None):
    if arquivo:
        base = open(arquivo, encoding = "utf-8").read()
        ids = tokenizer.encode(base, encode_special_tokens = False)
        reps = n_tokens // ids.shape[-1] + 1
        return ids.repeat(1, reps)[:, :n_tokens]
    # Parágrafos que não se repetem literalmente: texto repetido enche o indexador de empates
    amostra = "".join(f"§{i}: {FRASES[i % len(FRASES)]} (nota {i * 7919 % 10007}). " for i in range(200))
    por_par = tokenizer.encode(amostra).shape[-1] / 200
    n_par = int(n_tokens / por_par * 1.1) + 200
    s = "".join(f"§{i}: {FRASES[(i * 3 + i // 7) % len(FRASES)]} (nota {i * 7919 % 10007}). " for i in range(n_par))
    ids = tokenizer.encode(s, encode_special_tokens = False)
    assert ids.shape[-1] >= n_tokens, "texto curto demais, aumente a margem"
    return ids[:, :n_tokens]


def em_todos(model, fn, *args):
    return model.tp_worker_dispatch_wait_multi(model.active_devices, fn, args)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--model_dir", required = True)
    p.add_argument("--contextos", default = "65536,917504", help = "posições onde medir, separadas por vírgula")
    p.add_argument("--chunk", type = int, default = 4096, help = "chunk de prefill (o do hub é 4096)")
    p.add_argument("--medir", type = int, default = 2, help = "chunks medidos por modo em cada contexto")
    p.add_argument("--backend", default = "nccl", help = "backend do TP (o indexador dividido só liga no nccl)")
    p.add_argument("--cache", type = int, default = 0, help = "tokens no cache; 0 = o necessário")
    p.add_argument("--cache-bits", type = int, default = 0, help = "cache quantizado (largura do latente na MLA)")
    p.add_argument("--min-linhas", type = int, default = None, help = "EXL3_INDEXADOR_DIVIDIDO_MIN_LINHAS para a rodada dividida")
    p.add_argument("--modos", default = "replicado,dividido")
    p.add_argument("--texto", default = None, help = "arquivo de texto repetido como prompt (padrão: sintético variado)")
    p.add_argument("--conferir", action = "store_true", help = "compara os índices entre ranks e entre modos")
    p.add_argument("--nvtx", action = "store_true", help = "regiões NVTX por componente (para o nsys)")
    p.add_argument("--nsys", action = "store_true", help = "cudaProfilerStart/Stop só em volta das rodadas medidas")
    p.add_argument("--tp-options", default = None, help = "ex.: 'moe_tensor_split=1'")
    p.add_argument("--saida", default = None)
    args = p.parse_args()

    contextos = sorted(int(c) for c in args.contextos.split(","))
    modos = args.modos.split(",")
    for c in contextos:
        assert c % args.chunk == 0, f"contexto {c} não é múltiplo do chunk {args.chunk}"
    total = contextos[-1] + (args.medir + 1) * args.chunk
    cache_tokens = args.cache or -(-(total + 256) // 1024) * 1024

    config = Config.from_directory(args.model_dir)
    model = Model.from_config(config)
    if args.cache_bits:
        from exllamav3.cache import CacheLayer_quant
        cache = Cache(model, max_num_tokens = cache_tokens, layer_type = CacheLayer_quant,
                      k_bits = args.cache_bits, v_bits = args.cache_bits)
    else:
        cache = Cache(model, max_num_tokens = cache_tokens)
    tp_options = None
    if args.tp_options:
        tp_options = {k: int(v) for k, v in (par.split("=") for par in args.tp_options.split(","))}
    t0 = time.time()
    model.load(tensor_p = True, tp_backend = args.backend, progressbar = True, tp_options = tp_options)
    print(f"carga: {time.time() - t0:.0f} s · dispositivos {model.active_devices} · cache {cache_tokens} tokens")
    tokenizer = Tokenizer.from_config(config)

    mla = em_todos(model, pc.mp_descrever_mla)[0]
    full = [l for l, modo, *_ in mla if modo == "full"]
    shared = [l for l, modo, *_ in mla if modo == "shared"]
    ok = all(m[5] for m in mla if m[1] == "full")
    print(f"MLA: {len(mla)} camadas, {len(full)} full {full}, {len(shared)} shared · "
          f"tp_mundo {mla[0][3] if mla else '?'} · cp {mla[0][6] if mla else '?'} · coletivo ok {ok}")
    if "dividido" in modos and not ok:
        print("AVISO: o backend não libera o indexador dividido (só nccl); a rodada 'dividida' vai "
              "medir o caminho replicado")

    n_pts = em_todos(model, pc.mp_perfil_instrumentar)
    print(f"instrumentados: {n_pts} pontos por rank")
    if args.nvtx:
        em_todos(model, pc.mp_perfil_nvtx, True)

    ids = texto_de(total, tokenizer, args.texto)
    print(f"prompt: {ids.shape[-1]} tokens")

    estado = {"rs": None, "guardados": []}

    def prefill(a, b):
        for i in range(a, b, args.chunk):
            j = min(i + args.chunk, b)
            params = {"attn_mode": "flash_attn", "cache": cache, "past_len": i,
                      "batch_shape": (1, cache_tokens), "recurrent_states": estado["rs"]}
            model.prefill(input_ids = ids[:, i:j], params = params)
            estado["rs"] = params.get("recurrent_states")

    def guardar(C):
        g = [r.stash() if hasattr(r, "stash") else None for r in (estado["rs"] or [])]
        estado["guardados"].append(g)
        return g

    def descartar_guardados():
        # O stash em TP fica nos ranks, sob um handle; sem isto cada contexto deixa uma cópia
        # do estado do KDA para trás
        from exllamav3.cache.recurrent import mp_cache_recurrent_del
        for g in estado["guardados"]:
            for s in g:
                if s is not None and "tp_handle" in s:
                    try:
                        model.tp_dispatch_all(mp_cache_recurrent_del, (id(cache), s["tp_handle"]))
                    except Exception as e:
                        print(f"(stash não descartado: {e})")
        estado["guardados"].clear()

    def restaurar(C, guardado):
        for r, s in zip(estado["rs"] or [], guardado):
            r.position = C
            r.last_history = 0
            if s is not None:
                r.unstash(s)

    def definir(modo):
        dividir = modo == "dividido"
        em_todos(model, pc.mp_definir_indexador_dividido, dividir, args.min_linhas if dividir else None)

    resultado = {"modelo": args.model_dir, "dispositivos": list(model.active_devices), "chunk": args.chunk,
                 "backend": args.backend, "mla": mla, "contextos": {}}
    falhou = False
    pos = 0
    definir("replicado")
    for C in contextos:
        em_todos(model, pc.mp_perfil_ativar, False)
        torch.cuda.synchronize()
        t0 = time.time()
        prefill(pos, C)
        torch.cuda.synchronize()
        dt = time.time() - t0
        if C > pos:
            print(f"\nenchimento {pos} -> {C}: {dt:.1f} s = {(C - pos) / dt:.0f} tok/s (replicado, sem perfil)")
        pos = C
        em_todos(model, pc.mp_perfil_zerar)
        em_todos(model, pc.mp_perfil_ativar, True)
        guardado = guardar(C)

        por_modo = {}
        for modo in modos:
            definir(modo)
            restaurar(C, guardado)
            prefill(C, C + args.chunk)                    # aquecimento, não conta
            restaurar(C, guardado)
            torch.cuda.synchronize()
            em_todos(model, pc.mp_perfil_zerar)
            if args.conferir:
                em_todos(model, pc.mp_perfil_conferir, True, full[0] if full else None)
            if args.nsys:
                em_todos(model, pc.mp_cuda_profiler, True)
            t0 = time.time()
            prefill(C, C + args.medir * args.chunk)
            torch.cuda.synchronize()
            parede = (time.time() - t0) * 1000 / args.medir
            if args.nsys:
                em_todos(model, pc.mp_cuda_profiler, False)
            dig, idx_saida = None, []
            if args.conferir:
                # Digests de todos os ranks; as cópias inteiras da camada guardada só do rank de
                # saída (são dezenas de MB por chunk, e atravessam o pipe)
                dig = em_todos(model, pc.mp_perfil_digests, False)
                idx_saida = model.tp_dispatch_master(pc.mp_perfil_digests, (True,))["indices"]
                em_todos(model, pc.mp_perfil_conferir, False)
            tabs = em_todos(model, pc.mp_perfil_colher)
            por_modo[modo] = {"parede_ms_por_chunk": parede, "tok_s": args.chunk / parede * 1000,
                              "por_rank": tabs, "digests": dig, "indices": idx_saida}
        restaurar(C, guardado)
        definir("replicado")

        imprimir(C, args, por_modo, modos)
        if args.conferir:
            falhou |= conferir(C, por_modo, modos)
        resultado["contextos"][C] = {
            m: {"parede_ms_por_chunk": v["parede_ms_por_chunk"], "tok_s": v["tok_s"], "por_rank": v["por_rank"]}
            for m, v in por_modo.items()
        }

    descartar_guardados()
    saida = args.saida or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "saidas",
        datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") + "-componentes")
    os.makedirs(saida, exist_ok = True)
    with open(os.path.join(saida, "resultado.json"), "w") as f:
        json.dump(resultado, f, indent = 1, default = str)
    print(f"\nresultado: {saida}/resultado.json")
    em_todos(model, pc.mp_perfil_desinstrumentar)
    print("FALHOU" if falhou else "FIM_COMPONENTES")
    sys.exit(1 if falhou else 0)


def imprimir(C, args, por_modo, modos):
    print(f"\n=== contexto {C} ({C / 1024:.0f}k) · {args.medir} chunks de {args.chunk} por modo ===")
    rotulos = set()
    for v in por_modo.values():
        for t in v["por_rank"]:
            rotulos |= set(t)
    rotulos = [r for r in ORDEM if r in rotulos] + sorted(r for r in rotulos if r not in ORDEM)

    def ms(v, r, idx):
        # ms exclusivos por chunk: (máximo entre ranks, média entre ranks)
        xs = [t.get(r, (0, 0.0, 0.0))[idx] / args.medir for t in v["por_rank"]]
        return max(xs), sum(xs) / len(xs)

    cab = f"{'componente (ms/chunk, exclusivo)':<34}"
    for m in modos:
        cab += f" {m + ' máx':>15} {'méd':>8}"
    if len(modos) == 2:
        cab += f" {'Δ máx':>9}"
    print(cab)
    for r in rotulos:
        linha = f"{r:<34}"
        vals = []
        for m in modos:
            mx, md = ms(por_modo[m], r, 2)
            vals.append(mx)
            linha += f" {mx:>15.1f} {md:>8.1f}"
        if len(modos) == 2:
            linha += f" {vals[1] - vals[0]:>+9.1f}"
        print(linha)
    linha = f"{'coberto (soma dos exclusivos)':<34}"
    for m in modos:
        cob = [sum(x[2] for x in t.values()) / args.medir for t in por_modo[m]["por_rank"]]
        linha += f" {max(cob):>15.1f} {sum(cob) / len(cob):>8.1f}"
    print(linha)
    linha = f"{'parede':<34}"
    for m in modos:
        linha += f" {por_modo[m]['parede_ms_por_chunk']:>15.1f} {'':>8}"
    print(linha)
    linha = f"{'tok/s':<34}"
    for m in modos:
        linha += f" {por_modo[m]['tok_s']:>15.0f} {'':>8}"
    print(linha)
    if len(modos) == 2:
        a, b = (por_modo[m]["parede_ms_por_chunk"] for m in modos)
        print(f"{modos[1]} / {modos[0]}: {a / b:.3f}x de tok/s")
    # Inclusivos que interessam para a decisão: a MLA inteira e o indexador inteiro
    for m in modos:
        t0 = por_modo[m]["por_rank"][0]
        idx = sum(t0.get(r, (0, 0, 0))[1] for r in ("indexador_chaves", "indexador_topk", "indexador_reunir"))
        print(f"  [{m}] rank 0, inclusivo por chunk: mla {t0.get('mla', (0, 0, 0))[1] / args.medir:.1f} ms, "
              f"indexador inteiro {idx / args.medir:.1f} ms, all_reduce {t0.get('all_reduce', (0, 0, 0))[1] / args.medir:.1f} ms "
              f"em {t0.get('all_reduce', (0, 0, 0))[0] // args.medir} chamadas")


def conferir(C, por_modo, modos) -> bool:
    falhou = False
    for m in modos:
        dig = por_modo[m]["digests"]
        if not dig:
            continue
        d0 = dig[0]["digests"]
        diverge = 0
        for r in dig[1:]:
            diverge += sum(1 for a, b in zip(d0, r["digests"]) if a != b)
            if len(r["digests"]) != len(d0):
                print(f"  [{m}] rank com {len(r['digests'])} seleções contra {len(d0)} do rank 0")
                falhou = True
        print(f"  [{m}] seleções 'full' por rank: {len(d0)} · divergências entre ranks: {diverge}")
        if m == "dividido" and diverge:
            print("  FALHA: no dividido todo rank tem de ter os mesmos índices")
            falhou = True
    if "replicado" in por_modo and "dividido" in por_modo and por_modo["dividido"]["digests"]:
        a = por_modo["replicado"]["digests"][0]
        b = por_modo["dividido"]["digests"][0]
        iguais = sum(1 for x, y in zip(a["digests"], b["digests"]) if x == y)
        print(f"  replicado x dividido (rank 0): {iguais}/{len(a['digests'])} seleções idênticas")
        # Cópias inteiras de uma camada "full", do rank de saída, nos dois modos
        for ia, ib in zip(por_modo["replicado"]["indices"], por_modo["dividido"]["indices"]):
            # Sobreposição por linha, em amostra (o conjunto inteiro são milhões de entradas)
            linhas = range(0, ia.shape[0], max(1, ia.shape[0] // 256))
            sob = []
            for i in linhas:
                sa = set(x for x in ia[i].tolist() if x >= 0)
                sb = set(x for x in ib[i].tolist() if x >= 0)
                if sa or sb:
                    sob.append(len(sa & sb) / max(len(sa | sb), 1))
            if sob:
                print(f"  sobreposição média (Jaccard) da camada guardada: {sum(sob) / len(sob):.5f}, "
                      f"mínima {min(sob):.5f}")
    return falhou


if __name__ == "__main__":
    main()
