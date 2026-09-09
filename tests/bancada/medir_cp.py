"""
Quanto custa o coletivo do context parallel, isolado do modelo.

    torchrun --nproc_per_node=2 tests/bancada/medir_cp.py
    torchrun --nproc_per_node=4 tests/bancada/medir_cp.py --cabecas 64 --camadas-com-cache 11

O portão da etapa 1 do plano de context parallel. A pergunta NÃO é "quantos bytes", é "quantos
lançamentos": a prova 18 mediu que o coletivo em decode é limitado pela CPU emitir, não pela banda.

Sob CP cada camada MLA ganha UM coletivo a mais, sobre as parciais de atenção `(acc, m, l)` em
fp32, com geometria `(q_len, H_g, kv_lora_rank + 2)`. Três formas de fazer, e a diferença entre
elas é o que decide se vale escrever C++ novo:

  v1 all_reduce   escrever a parcial no slot do rank num buffer zerado e somar. É um all-gather
                  disfarçado, e é a ÚNICA que roda com o que a extensão já tem (pg_all_reduce é
                  soma, sem parâmetro de operação; não existe all_gather nem reduce_scatter).
  v2 all_gather   um pg_all_gather novo. Mesmo resultado, N× menos bytes no fio.
  v3 AG(lse)+RS   o estado da arte: all-gathera só o lse (minúsculo), reduce-scatter da saída já
                  entrega o head-scatter que o o_proj queria. Dois coletivos, o menor volume.

Cada linha reporta o custo por coletivo E o custo por token extrapolado para
`--camadas-com-cache`, que é o que se compara
com o passo de decode. `q_len = 8` é o rascunho do DFlash 2, que multiplica o payload por 8.
"""
import argparse, os, time
import torch
import torch.distributed as dist

from medir_allreduce import medir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cabecas", type = int, default = 64, help = "H_g: cabeças que o rank calcula sob CP")
    p.add_argument("--latente", type = int, default = 512, help = "kv_lora_rank")
    p.add_argument("--camadas-com-cache", type = int, default = 1,
                   help = "quantas camadas do modelo tem cache; NAO e constante -- sai da "
                          "contagem de modulos. O default 1 reporta o custo POR CAMADA, que e "
                          "a grandeza que nao depende de modelo nenhum")
    p.add_argument("--hidden", type = int, default = 4096, help = "para a linha de base do all-reduce de hoje")
    p.add_argument("--repeticoes", type = int, default = 300)
    p.add_argument("--qlens", type = str, default = "1,8", help = "q_len por passo; 8 é o rascunho do DFlash 2")
    a = p.parse_args()

    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")
    N = dist.get_world_size()

    H, D = a.cabecas, a.latente
    if H % N:
        raise SystemExit(f"H_g={H} tem de dividir por {N} ranks (restrição dura do CP)")

    if rank == 0:
        print(f"\n{N} placas · {torch.cuda.get_device_name(local)} · H_g {H} · latente {D} · "
              f"{a.camadas_com_cache} camadas MLA\n")
        for i in range(1, N):
            ok = torch.cuda.can_device_access_peer(0, i)
            print(f"  P2P 0↔{i}: {'sim' if ok else 'NÃO — a coletiva desce à memória do host'}")

        # A linha de base: o all-reduce que JÁ existe hoje, por camada, sobre o hidden de um token
        base = torch.randn((1, a.hidden), device = local, dtype = torch.bfloat16)
        b_parede, b_cpu = medir(lambda: dist.all_reduce(base, async_op = False), a.repeticoes)
        print(f"\n  linha de base · all-reduce de hoje ({1 * a.hidden * 2} B bf16): "
              f"{b_parede * 1e6:.1f} µs parede, {b_cpu * 1e6:.1f} µs só para enfileirar")
    else:
        base = torch.randn((1, a.hidden), device = local, dtype = torch.bfloat16)
        medir(lambda: dist.all_reduce(base, async_op = False), a.repeticoes)
    dist.barrier()

    if rank == 0:
        print(f"\n{'q_len':>6} {'variante':<14} {'no fio':>10} {'parede':>10} {'enfileirar':>11} "
              f"{'por token':>10}   (× {a.camadas_com_cache} camadas MLA)")

    for q in [int(x) for x in a.qlens.split(",")]:
        # v1: buffer com um slot por rank, zerado fora do próprio slot, somado
        buf = torch.zeros((N, q, H, D + 2), device = local, dtype = torch.float32)
        # v2/v3: a parcial que o rank produz
        parcial = torch.randn((q, H, D), device = local, dtype = torch.float32)
        lse = torch.randn((q, H), device = local, dtype = torch.float32)
        # destinos
        gather_out = torch.empty((N, q, H, D + 2), device = local, dtype = torch.float32)
        gather_in = torch.empty((q, H, D + 2), device = local, dtype = torch.float32)
        lse_out = torch.empty((N, q, H), device = local, dtype = torch.float32)
        # o reduce-scatter fatia a dimensão externa: a parcial tem de estar head-major primeiro
        rs_in = torch.empty((N, q, H // N, D), device = local, dtype = torch.float32)
        rs_out = torch.empty((q, H // N, D), device = local, dtype = torch.float32)

        def v1():
            dist.all_reduce(buf, async_op = False)

        def v2():
            dist.all_gather_into_tensor(gather_out, gather_in, async_op = False)

        def v3():
            dist.all_gather_into_tensor(lse_out, lse, async_op = False)
            dist.reduce_scatter_tensor(rs_out, rs_in, async_op = False)

        # o custo local que o v3 paga e os outros não: transpor a parcial para head-major
        def transpor():
            rs_in.copy_(parcial.reshape(q, N, H // N, D).transpose(0, 1))

        medidas = [
            ("v1 all_reduce", buf.numel() * 4, medir(v1, a.repeticoes)),
            ("v2 all_gather", gather_in.numel() * 4 * (N - 1), medir(v2, a.repeticoes)),
            ("v3 AG(lse)+RS", (lse.numel() * (N - 1) + rs_in.numel() * (N - 1) // N) * 4, medir(v3, a.repeticoes)),
            ("  v3 transpor", 0, medir(transpor, a.repeticoes)),
        ]

        if rank == 0:
            for nome, fio, (parede, cpu) in medidas:
                print(f"{q:>6} {nome:<14} {fio / 1024:>8.0f} KiB {parede * 1e6:>8.1f} µs "
                      f"{cpu * 1e6:>8.1f} µs {parede * a.camadas_com_cache * 1e3:>8.2f} ms")
        dist.barrier()

    if rank == 0:
        print(f"\nPortão do plano: se a coluna 'por token' passar de ~2 ms, a v1 não fecha e o\n"
              f"caminho vira pg_all_gather antes de qualquer outra coisa. O passo de decode\n"
              f"medido na etapa 4 foi 26 ms, com ~90 all-reduces de linha de base.\n"
              f"Atenção ao fp32: o nosso all_reduce estreita para bf16 acima de 1 MiB.\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
