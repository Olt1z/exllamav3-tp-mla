"""
O plano de TP sob context parallel, provado SEM GPU.

    python3 tests/bancada/provar_plano_cp.py

O alocador é Python puro, então esta é a única peça do CP que não custa aluguel para testar — e é
justamente onde um erro se esconde num número plausível. As duas propriedades:

  1. **Placas do mesmo grupo recebem a MESMA faixa de canais.** Elas repartem a sequência entre si,
     não o modelo. Se acumulassem faixas diferentes, cada uma teria um pedaço distinto das cabeças
     e o modelo carregaria normalmente, produzindo saída errada em silêncio.
  2. **A união das faixas distintas cobre todas as cabeças, sem buraco e sem sobreposição.**

E o caso `dcp = 1` tem de reproduzir exatamente o comportamento de hoje, canal por canal.

Não importa torch: o módulo do alocador é carregado com `ratio_split` recortado do util, para o
teste rodar em qualquer máquina.
"""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]


def carregar_alocador():
    """Carrega TPAllocator sem arrastar torch para dentro do processo."""
    fonte = (RAIZ / "exllamav3/model/model_tp_alloc.py").read_text()
    fonte = fonte.replace("from ..util.misc import ratio_split", "")
    misc = (RAIZ / "exllamav3/util/misc.py").read_text()
    i = misc.index("def ratio_split")
    j = misc.index("\ndef ", i + 5)
    ns = {}
    exec(misc[i:j], ns)
    exec(fonte, ns)
    return ns["TPAllocator"], ns["TPAllocation"]


def main():
    TPAllocator, TPAllocation = carregar_alocador()
    falhas = []

    def componente(canais):
        return TPAllocation(key = "attn", channel_width = 1, channel_unit = "heads",
                            storage_per_device = 10, storage_to_split = 100,
                            channels_to_split = canais)

    CABECAS = 64
    for placas, memoria in ((4, [1000] * 4), (4, [1000, 1000, 600, 600]), (8, [1000] * 8)):
        for dcp in [d for d in (1, 2, 4, 8) if d <= placas and placas % d == 0]:
            a = TPAllocator([componente(CABECAS)], num_tokens = 1, output_num_tokens = 1,
                            dcp = dcp)
            a.initial_split(list(memoria))
            plano = a.compile_tp_plan()
            faixas = [plano[d]["attn"][:2] for d in range(placas)]

            # 1. mesma faixa dentro do grupo
            ok_grupo = all(len(set(faixas[g * dcp:(g + 1) * dcp])) == 1
                           for g in range(placas // dcp))
            # 2. cobertura completa, sem buraco nem sobreposicao
            distintas = sorted(set(faixas))
            ok_cob = (distintas[0][0] == 0 and distintas[-1][1] == CABECAS
                      and all(e1 == b2 for (_, e1), (b2, _) in zip(distintas, distintas[1:]))
                      and len(distintas) == placas // dcp)

            marca = "OK" if (ok_grupo and ok_cob) else "FALHOU"
            print(f"{placas} placas {memoria} dcp {dcp}: "
                  f"{len(distintas)} grupo(s), faixas {distintas} {marca}")
            if not ok_grupo:
                falhas.append(f"{placas}/{dcp}: faixas diferentes no grupo")
            if not ok_cob:
                falhas.append(f"{placas}/{dcp}: cobertura {distintas}")

    # dcp = 1 tem de ser identico ao comportamento de hoje
    base = TPAllocator([componente(CABECAS)], num_tokens = 1, output_num_tokens = 1, dcp = 1)
    base.initial_split([1000, 1000, 600, 600])
    hoje = base.compile_tp_plan()
    esperado = [(0, 20), (20, 40), (40, 52), (52, 64)]
    obtido = [hoje[d]["attn"][:2] for d in range(4)]
    ok_hoje = all(o == e for o, e in zip(obtido, esperado))
    print(f"\ndcp 1 em memoria desigual: {obtido} "
          f"{'OK (proporcional a memoria, como sempre foi)' if ok_hoje else f'MUDOU, esperado {esperado}'}")
    if not ok_hoje:
        falhas.append("dcp 1 mudou de comportamento")

    # grau que nao divide tem de falhar alto
    try:
        ruim = TPAllocator([componente(CABECAS)], num_tokens = 1, output_num_tokens = 1, dcp = 3)
        ruim.initial_split([1000] * 4)
        print("dcp 3 em 4 placas foi ACEITO — devia ter recusado")
        falhas.append("dcp que nao divide foi aceito")
    except RuntimeError:
        print("dcp 3 em 4 placas recusado, como tem de ser")

    print()
    if falhas:
        print(f"FALHOU em {len(falhas)}: {'; '.join(falhas)}")
        raise SystemExit(1)
    print("PLANO OK: mesma faixa no grupo, cobertura completa, dcp 1 inalterado")


if __name__ == "__main__":
    main()
