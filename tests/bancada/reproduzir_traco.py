"""
Onde os experts deveriam morar, respondido offline a partir de UM traço.

    python3 tests/bancada/reproduzir_traco.py traco.bin
    python3 tests/bancada/reproduzir_traco.py traco.bin --bytes-por-expert 8847360
    python3 tests/bancada/reproduzir_traco.py treino.bin --avaliar trabalho.bin

Roteamento não depende de colocação: o mesmo prompt escolhe os mesmos experts esteja o
expert na VRAM ou na arena do worker. Então uma execução grava a sequência
(`EXL3_MOE_CPU_TRACE_OUT`, ver `_TracoDeRoteamento`) e ela responde, aqui, a família inteira
de perguntas sobre colocação — em qualquer fração, sob qualquer política — sem uma segunda
máquina alugada. A ideia é do `FareedKhan-dev/kimi-k3-in-c` (`tools/sim_cache.py`).

O EIXO AQUI É OUTRO, e é a razão de existir um arquivo em vez de usar o dele. Lá a pergunta
é "quanta RAM de arena", com um LRU transmitindo do disco. Aqui não há cache nem despejo: o
worker guarda TODOS os experts, e a colocação decide quais também vivem na placa. Uma
"falha" não é um carregamento, é uma seleção que a CPU vai atender — e o custo é o
`gemv` do worker mais a leitura da RAM do host. A grandeza que interessa, então, é a
FRAÇÃO DAS SELEÇÕES QUE CAI NA CPU, e os bytes que ela arrasta por token.

As políticas comparadas:

  índice        os k últimos por id de checkpoint vão para a CPU. É o que o motor faz sem
                perfil e sem varredura — a linha de base honesta.
  dinâmica      a varredura periódica do fork (`_split_sweep_layer`): promove o mais quente
                da cauda contra o mais frio da cabeça enquanto a razão passa a histerese E a
                massa passa o piso, decaindo as contagens depois. Começa na identidade, que
                é a colocação por índice.
  popularidade  os k menos selecionados vão para a CPU, ranqueados por um perfil.
  ótimo         os k menos selecionados NESTE traço. Não é política: é o TETO de qualquer
                colocação estática, e serve para saber quando parar de otimizar.

DUAS HONESTIDADES, herdadas de quem teve a ideia e sem as quais o número mente:

  - `popularidade` ranqueada pelo PRÓPRIO traço é conhecimento do futuro, e vira igual ao
    ótimo. Por isso o padrão é dividir o traço ao meio: ranqueia na primeira metade, mede na
    segunda. Com `--avaliar`, ranqueia num traço e mede em OUTRO — que é o caso real (perfil
    colhido de conversa, aplicado a trabalho de código) e a pergunta do item 1c do plano.
  - a `dinâmica` também não vê o futuro: ela é simulada passo a passo, como o motor a roda.
"""
import argparse, json, os

import numpy as np

# 3 bpw de trellis, 3 projeções, hidden 5120 x moe_intermediate 1536: o GLM-5.3 nosso.
BYTES_POR_EXPERT = 3 * 3 * 5120 * 1536 // 8


def carregar(caminho):
    """O binário de `_TracoDeRoteamento`: int32 plano, pares (camada, expert)."""
    cru = np.fromfile(caminho, dtype = np.int32)
    if cru.size % 2:
        raise SystemExit(f"{caminho}: número ímpar de int32, não é um traço")
    return cru.reshape(-1, 2)


def contagens(pares, n_experts):
    """Seleções por (camada, expert), como matriz camadas x experts."""
    camadas = int(pares[:, 0].max()) + 1
    h = np.zeros((camadas, n_experts), dtype = np.int64)
    np.add.at(h, (pares[:, 0], pares[:, 1]), 1)
    return h


def na_cpu_por_indice(_h_perfil, n_experts, k):
    """Os k últimos por id, em toda camada. Independe do perfil, de propósito."""
    return lambda camada: set(range(n_experts - k, n_experts))


def na_cpu_por_popularidade(h_perfil, n_experts, k):
    """Os k menos selecionados de cada camada, segundo o perfil que foi passado."""
    ordem = np.argsort(h_perfil, axis = 1, kind = "stable")   # frio -> quente
    return lambda camada: set(ordem[camada, :k].tolist())


def custo_estatico(pares, na_cpu):
    """Quantas seleções o worker atende, com a colocação fixa durante todo o traço."""
    cache, n = {}, 0
    for camada, expert in pares:
        c = int(camada)
        s = cache.get(c)
        if s is None:
            s = cache[c] = na_cpu(c)
        if int(expert) in s:
            n += 1
    return n


def custo_dinamico(pares, n_experts, k, intervalo, hist_, piso_, orcamento):
    """A varredura do fork, passo a passo.

    Começa na identidade (= colocação por índice) e, a cada `intervalo` passos de decode,
    cada camada troca o mais quente da cauda pelo mais frio da cabeça enquanto a razão passa
    a histerese E a contagem quente passa o piso de massa absoluto. Depois decai as contagens
    pela metade, que é o que faz a colocação seguir o roteamento RECENTE.

    O piso existe no motor por um motivo que vale repetir: com janelas curtas as contagens
    são poucos acertos, e um expert da cauda com 3 contra um da cabeça com 1 passa qualquer
    razão pura -- a varredura churna em ruído para sempre, e cada churn é uma leitura do
    checkpoint.
    """
    camadas = int(pares[:, 0].max()) + 1
    primeiro = n_experts - k
    # mapa[c][r] = posição física do expert r; >= primeiro significa "mora na CPU".
    mapa = np.tile(np.arange(n_experts), (camadas, 1))
    hist = np.zeros((camadas, n_experts), dtype = np.float64)
    na_cpu = 0
    passos = 0
    camada_anterior = -1
    for camada, expert in pares:
        c, r = int(camada), int(expert)
        if mapa[c, r] >= primeiro:
            na_cpu += 1
        hist[c, r] += 1
        # Um "passo de decode" é uma volta completa pelas camadas; o motor conta no primeiro
        # módulo registrado, então a borda é a mesma: voltar para a primeira camada.
        if c < camada_anterior:
            passos += 1
            if passos % intervalo == 0:
                _varrer(mapa, hist, primeiro, n_experts, hist_, piso_, orcamento)
                hist *= 0.5
        camada_anterior = c
    return na_cpu


def _varrer(mapa, hist, primeiro, n_experts, hist_, piso_, orcamento):
    for c in range(mapa.shape[0]):
        piso = piso_ * float(hist[c].sum()) / n_experts
        cabeca = sorted((hist[c, r], r) for r in range(n_experts) if mapa[c, r] < primeiro)
        cauda = sorted(((hist[c, r], r) for r in range(n_experts) if mapa[c, r] >= primeiro),
                       reverse = True)
        trocas = 0
        for (c_frio, r_frio), (c_quente, r_quente) in zip(cabeca, cauda):
            if trocas >= orcamento or c_quente < max(hist_ * max(c_frio, 1.0), piso):
                break
            mapa[c, r_frio], mapa[c, r_quente] = mapa[c, r_quente], mapa[c, r_frio]
            trocas += 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("traco", help="o traço que ranqueia o perfil (e mede, se não houver --avaliar)")
    p.add_argument("--avaliar", help="traço em que MEDIR, quando for outro (item 1c do plano)")
    p.add_argument("--bytes-por-expert", type = int, default = BYTES_POR_EXPERT)
    p.add_argument("--banda-gbs", type = float, default = 77.0,
                   help="banda de RAM do host, para a coluna de tempo")
    p.add_argument("--fracoes", default = "0.1,0.25,0.5,0.75,0.81,0.9")
    p.add_argument("--intervalo", type = int, default = 32)
    p.add_argument("--histerese", type = float, default = 2.0)
    p.add_argument("--piso", type = float, default = 2.0)
    p.add_argument("--orcamento", type = int, default = 128)
    # O traço não marca fronteira de token, e deduzi-la de médias dá um número que parece
    # medido e é adivinhado. É constante do modelo (`num_experts_per_tok`): entra como flag.
    p.add_argument("--top-k", type = int, default = 8, help="experts por token por camada")
    a = p.parse_args()

    perfil = carregar(a.traco)
    medir = carregar(a.avaliar) if a.avaliar else None
    if medir is None:
        # Sem um segundo traço, treino e teste saem do mesmo: primeira metade ranqueia,
        # segunda mede. Ranquear e medir no mesmo pedaço responde outra pergunta (o teto).
        meio = (len(perfil) // 2) & ~1
        perfil, medir = perfil[:meio], perfil[meio:]
        origem = "metade do mesmo traço"
    else:
        origem = os.path.basename(a.traco)

    n_experts = int(max(perfil[:, 1].max(), medir[:, 1].max())) + 1
    camadas = int(medir[:, 0].max()) + 1
    lado = f"{a.avaliar or a.traco}.json"
    nomes = json.load(open(lado))["camadas"] if os.path.exists(lado) else {}

    tokens = max(len(medir) // (camadas * a.top_k), 1)
    print(f"traço de medida: {len(medir)} seleções, {camadas} camadas, {n_experts} experts, "
          f"~{tokens} token(s)")
    if nomes:
        print(f"camadas: {nomes.get('0', '?')} .. {nomes.get(str(camadas - 1), '?')}")
    print(f"perfil ranqueado em: {origem}\n")

    h_perfil = contagens(perfil, n_experts)
    h_medida = contagens(medir, n_experts)

    print("%-9s %11s %11s %13s %9s %12s %11s" %
          ("NA CPU", "ÍNDICE", "DINÂMICA", "POPULARIDADE", "ÓTIMO", "GB/TOKEN", "MS/TOKEN"))
    print("-" * 82)
    for f in [float(x) for x in a.fracoes.split(",")]:
        k = int(round(n_experts * f))
        if not 0 < k < n_experts:
            continue
        idx = custo_estatico(medir, na_cpu_por_indice(h_perfil, n_experts, k))
        din = custo_dinamico(medir, n_experts, k, a.intervalo, a.histerese, a.piso, a.orcamento)
        pop = custo_estatico(medir, na_cpu_por_popularidade(h_perfil, n_experts, k))
        oti = custo_estatico(medir, na_cpu_por_popularidade(h_medida, n_experts, k))
        gb = pop * a.bytes_por_expert / 1e9 / tokens
        ms = gb * 1000.0 / a.banda_gbs
        print("%-9s %10.2f%% %10.2f%% %12.2f%% %8.2f%% %12.3f %11.2f" %
              ("%.0f%%" % (100 * f), 100 * idx / len(medir), 100 * din / len(medir),
               100 * pop / len(medir), 100 * oti / len(medir), gb, ms))
    print("-" * 82)
    print("GB/TOKEN e MS/TOKEN são da coluna POPULARIDADE, que é a política proponível;\n"
          "ÓTIMO ranqueia no próprio traço de medida, então é conhecimento do futuro — teto,\n"
          "não previsão. A distância entre POPULARIDADE e ÓTIMO é o que um perfil melhor\n"
          "ainda poderia comprar; a distância entre ÍNDICE e POPULARIDADE é o que já está\n"
          "na mesa, e não custa um byte de VRAM.")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
