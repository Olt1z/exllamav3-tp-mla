"""
Quanto custa contar as atribuições por expert: torch.bincount contra scatter_add_.

    python3 tests/bancada/medir_contagem.py

O bincount do CUDA lê `self.max()` de volta ao host para dimensionar a saída, mesmo com
minlength já fixando o tamanho (SummaryOps.cu), e isso é uma sincronização por camada MoE no
caminho de decode -- 42 por token no GLM-5.3-Flash. O scatter_add_ dá a contagem idêntica sem
sair da GPU. Aqui os dois são medidos isolados, porque no corte de 4 camadas (uma só é MoE) a
diferença ficaria dentro do ruído do passo inteiro.

A coluna que decide é `enfileirar`: é o tempo que a CPU gasta na chamada, e é ele que a
sincronização infla.
"""
import time
import torch

# formato real do decode: top-8 de 288 experts por token, e a rodada de verificação do rascunho
CASOS = (("decode, 1 token x top-8", 8), ("rascunho, 8 tokens x top-8", 64))
EXPERTS = 288
CAMADAS_MOE = 42


def medir(fn, repeticoes = 500, aquecimento = 50):
    for _ in range(aquecimento):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeticoes):
        fn()
    cpu = time.perf_counter() - t0          # só a chamada, sem esperar a GPU
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeticoes, cpu / repeticoes


def main():
    print(f"\n{torch.cuda.get_device_name(0)} · {EXPERTS} experts · "
          f"{CAMADAS_MOE} camadas MoE por token\n")
    print(f"{'caso':<28} {'como':<14} {'parede':>10} {'enfileirar':>12}   por token")
    for rotulo, n in CASOS:
        ids = torch.randint(0, EXPERTS, (n,), device = "cuda")

        def com_bincount():
            return torch.bincount(ids, minlength = EXPERTS + 1)

        def com_scatter():
            c = torch.zeros(EXPERTS + 1, dtype = torch.long, device = ids.device)
            c.scatter_add_(0, ids.long(), torch.ones_like(ids, dtype = torch.long))
            return c

        assert torch.equal(com_bincount(), com_scatter()), "as duas contagens têm de ser idênticas"

        for nome, fn in (("bincount", com_bincount), ("scatter_add_", com_scatter)):
            parede, cpu = medir(fn)
            print(f"{rotulo:<28} {nome:<14} {parede * 1e6:>7.1f} µs {cpu * 1e6:>9.1f} µs   "
                  f"{parede * CAMADAS_MOE * 1e3:>6.2f} ms")
        print()
    print("A diferença entre as duas linhas 'por token' é o que a troca economiza no passo,\n"
          "contra os 26 ms medidos no motor cru.\n")


if __name__ == "__main__":
    main()
