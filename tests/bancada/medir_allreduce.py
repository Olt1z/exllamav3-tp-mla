"""
Quanto custa um all-reduce do TP, isolado do modelo.

    torchrun --nproc_per_node=2 tests/bancada/medir_allreduce.py
    torchrun --nproc_per_node=2 tests/bancada/medir_allreduce.py --hidden 4096 --camadas 90

O decode do GLM-5.3-Flash faz `--camadas` all-reduces por token (um na atenção e outro no
MLP de cada camada), cada um sobre o hidden state de UM token — 8 KB. O custo aí não é banda,
é latência: a coletiva é serializada por dependência de dados, uma espera a outra.

Mede quatro coisas, e a diferença entre elas é o que sobra para ganhar:

  bf16 serial   o que o backend NCCL faz hoje (`async_op = False`, dependência real)
  fp32 serial   o desvio de model_tp_backend.py:120 — 3 kernels extras por coletiva
  bf16 paralelo mesmo volume sem dependência, ou seja o piso da coletiva
  enfileirar     o tempo que a CPU gasta só para lançar, sem esperar a GPU

Cada linha reporta o custo por token (× camadas), que é o que se compara com o passo de decode.
"""
import argparse, os, time
import torch
import torch.distributed as dist


def medir(fn, repeticoes, aquecimento = 20):
    for _ in range(aquecimento):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(repeticoes):
        fn()
    cpu = time.perf_counter() - t0          # só enfileirar, sem esperar a GPU
    torch.cuda.synchronize()
    parede = time.perf_counter() - t0       # com a GPU tendo terminado
    return parede / repeticoes, cpu / repeticoes


def caminho_fp32(t):
    """O que TPBackendNative/NCCL.all_reduce faz quando o tensor é fp32."""
    temp = t.to(torch.bfloat16)
    dist.all_reduce(temp, async_op = False)
    temp = temp.to(torch.float32)
    t.copy_(temp)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden", type = int, default = 4096, help = "hidden_size do modelo")
    p.add_argument("--camadas", type = int, default = 90, help = "all-reduces por token")
    p.add_argument("--repeticoes", type = int, default = 300)
    p.add_argument("--tokens", type = str, default = "1,4,8,2048", help = "tokens por coletiva")
    a = p.parse_args()

    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")
    mundo = dist.get_world_size()

    if rank == 0:
        print(f"\n{mundo} placas · {torch.cuda.get_device_name(local)} · hidden {a.hidden} · "
              f"{a.camadas} all-reduces por token\n")
        # P2P decide se a coletiva anda entre placas ou desce ao host
        for i in range(1, mundo):
            ok = torch.cuda.can_device_access_peer(0, i)
            print(f"  P2P 0↔{i}: {'sim' if ok else 'NÃO — a coletiva desce à memória do host'}")
        print(f"\n{'tokens':>7} {'bytes':>9} {'bf16 serial':>13} {'fp32 serial':>13} "
              f"{'bf16 paralelo':>14} {'enfileirar':>11}   por token (bf16 serial)")

    for n in [int(x) for x in a.tokens.split(",")]:
        t16 = torch.randn((n, a.hidden), device = local, dtype = torch.bfloat16)
        t32 = torch.randn((n, a.hidden), device = local, dtype = torch.float32)
        # 8 buffers distintos: sem dependência entre eles, a fila da GPU pode sobrepor
        soltos = [torch.randn((n, a.hidden), device = local, dtype = torch.bfloat16) for _ in range(8)]
        i = [0]

        def solto():
            dist.all_reduce(soltos[i[0] % 8], async_op = False)
            i[0] += 1

        serial16, cpu16 = medir(lambda: dist.all_reduce(t16, async_op = False), a.repeticoes)
        serial32, _ = medir(lambda: caminho_fp32(t32), a.repeticoes)
        paralelo, _ = medir(solto, a.repeticoes)

        if rank == 0:
            b = n * a.hidden * 2
            print(f"{n:>7} {b:>8} B {serial16 * 1e6:>10.1f} µs {serial32 * 1e6:>10.1f} µs "
                  f"{paralelo * 1e6:>11.1f} µs {cpu16 * 1e6:>8.1f} µs   "
                  f"{serial16 * a.camadas * 1e3:>6.2f} ms")
        dist.barrier()

    if rank == 0:
        print("\nO passo de decode medido na etapa 4 foi 26 ms. A coluna da direita é quanto\n"
              "desse passo é só coletiva; 'bf16 paralelo' é o piso se elas pudessem se sobrepor.\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
