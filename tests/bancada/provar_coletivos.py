"""
Paridade e custo das duas primitivas novas do backend NATIVO contra o torch.distributed.

    torchrun --nproc_per_node=2 tests/bancada/provar_coletivos.py
    torchrun --nproc_per_node=4 tests/bancada/provar_coletivos.py --repeticoes 300

O portão da etapa 2b do context parallel. `pg_all_reduce_kernel` já era um anel de duas fases —
as primeiras `num_ranks-1` iterações acumulam, o que É reduce-scatter, e as últimas copiam, o que
É all-gather. Este teste prova que a metade solta faz o mesmo que a primitiva do NCCL.

Sobre exatidão, e a diferença importa:

  all_gather      é CÓPIA. Tem de bater BIT A BIT, em qualquer dtype. Se não bater, é defeito.
  reduce_scatter  é SOMA, e o anel soma numa ordem diferente do NCCL. Em ponto flutuante isso
                  NÃO é bit a bit e exigir isso seria um portão errado. Testa-se com int32, onde
                  a soma é associativa e a exatidão é obrigatória, e com fp32 sob tolerância.

As geometrias são as reais do CP: o lse (minúsculo, e é ele que pega o cálculo de `threads` do
upstream de jeito errado — 256 B por rank contra um estágio de 512) e a saída pesada de atenção.
"""
import argparse, os, time
import torch
import torch.distributed as dist

from exllamav3.model.model_tp_backend import TPBackendNCCL


def medir(fn, repeticoes, aquecimento = 10):
    for _ in range(aquecimento):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(repeticoes):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeticoes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cabecas", type = int, default = 64)
    p.add_argument("--latente", type = int, default = 512)
    p.add_argument("--repeticoes", type = int, default = 200)
    p.add_argument("--qlens", type = str, default = "1,8")
    a = p.parse_args()

    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)

    devices = list(range(int(os.environ["WORLD_SIZE"])))
    b = TPBackendNCCL(
        device = local,
        active_devices = devices,
        output_device = 0,
        init_method = "env://",
        master = (rank == 0),
        uuid = os.environ.get("MASTER_PORT", "prova2b"),
    )
    nativo = b.fallback
    N = len(devices)
    H, D = a.cabecas, a.latente

    if rank == 0:
        print(f"\n{N} placas · {torch.cuda.get_device_name(local)} · H {H} · latente {D}\n")
        print(f"{'caso':<34} {'paridade':>22} {'nccl':>9} {'nativo':>9} {'razão':>7}")

    falhas = []

    def comparar(nome, ex_nccl, ex_nativo, saida_nccl, saida_nativo, exato):
        ex_nccl(); ex_nativo()
        torch.cuda.synchronize()
        if exato:
            ok = torch.equal(saida_nccl, saida_nativo)
            veredito = "bit a bit" if ok else "DIVERGIU"
        else:
            d = (saida_nccl.float() - saida_nativo.float()).abs().max().item()
            ok = d <= 1e-5
            veredito = f"max {d:.2e}" + ("" if ok else "  DIVERGIU")
        t_n = medir(ex_nccl, a.repeticoes)
        t_a = medir(ex_nativo, a.repeticoes)
        if rank == 0:
            print(f"{nome:<34} {veredito:>22} {t_n*1e6:>7.1f} µs {t_a*1e6:>7.1f} µs "
                  f"{t_a/t_n:>7.2f}")
        if not ok:
            falhas.append(nome)
        dist.barrier()

    for q in [int(x) for x in a.qlens.split(",")]:
        # 1. all_gather do lse: (q*H,) por rank -> (N, q*H). É o caso pequeno, e é o que quebra
        #    se o estágio do anel não dividir a fatia
        lse = torch.randn((q * H,), device = local, dtype = torch.float32) + rank
        g_nccl = torch.empty((N, q * H), device = local, dtype = torch.float32)
        g_nat = torch.empty((N, q * H), device = local, dtype = torch.float32)
        comparar(f"q{q} all_gather lse ({q*H*4} B/rank)",
                 lambda: b.all_gather(g_nccl, lse),
                 lambda: nativo.all_gather(g_nat, lse),
                 g_nccl, g_nat, exato = True)

        # 2. reduce_scatter da saída pesada, em int32: a soma é associativa, então a exatidão é
        #    obrigatória e uma divergência aqui é defeito, não arredondamento
        pesada_i = torch.randint(-1000, 1000, (H, q, D), device = local, dtype = torch.int32)
        r_nccl_i = torch.empty((H // N, q, D), device = local, dtype = torch.int32)
        r_nat_i = torch.empty((H // N, q, D), device = local, dtype = torch.int32)
        comparar(f"q{q} reduce_scatter int32 ({H*q*D*4//N} B/rank)",
                 lambda: b.reduce_scatter(r_nccl_i, pesada_i.clone()),
                 lambda: nativo.reduce_scatter(r_nat_i, pesada_i.clone()),
                 r_nccl_i, r_nat_i, exato = True)

        # 3. o mesmo em fp32, que é o que o CP trafega. Ordem de soma diferente do NCCL, então
        #    tolerância, não igualdade
        pesada = torch.randn((H, q, D), device = local, dtype = torch.float32)
        r_nccl = torch.empty((H // N, q, D), device = local, dtype = torch.float32)
        r_nat = torch.empty((H // N, q, D), device = local, dtype = torch.float32)
        comparar(f"q{q} reduce_scatter fp32 ({H*q*D*4//N} B/rank)",
                 lambda: b.reduce_scatter(r_nccl, pesada.clone()),
                 lambda: nativo.reduce_scatter(r_nat, pesada.clone()),
                 r_nccl, r_nat, exato = False)

    if rank == 0:
        if falhas:
            print(f"\nFALHOU em {len(falhas)} caso(s): {', '.join(falhas)}\n")
        else:
            print("\nPARIDADE OK nos dois coletivos. A coluna 'razão' contra 1,20 é o portão de\n"
                  "latência da etapa 2b; acima disso o nativo não paga o que economiza em\n"
                  "despacho e o CP deve continuar no NCCL.\n")

    b.close()
    raise SystemExit(1 if falhas else 0)


if __name__ == "__main__":
    main()
