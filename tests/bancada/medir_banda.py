"""
Banda real do host: DRAM da CPU, PCIe, e as duas EM CONTENÇÃO.

    python3 tests/bancada/medir_banda.py
    python3 tests/bancada/medir_banda.py --threads 96 --segundos 3

O planejador do hub estima a banda da RAM por fórmula (`bandaDaRamDoHost`: núcleos × 4,8,
teto 600), e essa reta passa por apenas DOIS pontos medidos, ambos de EPYC 9654. Onde a
fórmula erra, ela erra no termo que DOMINA o decode com experts na RAM — o fallback de
77 GB/s já se mostrou 7,8× abaixo do real numa máquina.

Mede três coisas, e a terceira é a que não dá para deduzir das outras duas:

  dram      leitura sequencial da RAM pela CPU, working set acima da LLC
  pcie      cópia pinned <-> placa, nos dois sentidos
  contenção as duas ao MESMO TEMPO, disputando a mesma DRAM

A contenção importa porque é o regime em que o decode vive: o worker de CPU lê experts da
RAM enquanto a placa lê os dela e o PCIe move ativação. Nem "contenção total" (a CPU fica
com dram − pcie) nem "nenhuma contenção" (as duas somam) preveem o resultado; por isso se
mede o par.
"""
import argparse, os, threading, time
import torch

GIB = 1024 ** 3


def banda_dram(tensores, segundos, parar = None):
    """Leitura sequencial: soma tensores grandes até o prazo. Devolve GB/s."""
    lidos, t0 = 0, time.perf_counter()
    i = 0
    while time.perf_counter() - t0 < segundos and not (parar and parar.is_set()):
        t = tensores[i % len(tensores)]
        t.sum()
        lidos += t.numel() * t.element_size()
        i += 1
    return lidos / (time.perf_counter() - t0) / 1e9


def banda_pcie(host, dev, segundos, sentido, parar = None):
    """Cópia pinned <-> placa. `torch.cuda._sleep` segura a GPU para a CPU não enfileirar
    tudo antes do cronômetro e a medida virar taxa de enfileiramento em vez de banda."""
    if hasattr(torch.cuda, "_sleep"):
        torch.cuda._sleep(10 ** 7)
    movidos, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < segundos and not (parar and parar.is_set()):
        if sentido == "h2d":
            dev.copy_(host, non_blocking = True)
        else:
            host.copy_(dev, non_blocking = True)
        torch.cuda.synchronize()   # o DMA tem de estar em voo, não só enfileirado
        movidos += host.numel() * host.element_size()
    return movidos / (time.perf_counter() - t0) / 1e9


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--threads", type = int, default = 0, help = "0 = metade dos núcleos")
    p.add_argument("--segundos", type = float, default = 2.0)
    p.add_argument("--gib", type = float, default = 2.0, help = "working set por buffer")
    a = p.parse_args()

    nucleos = os.cpu_count() or 1
    threads = a.threads or max(1, nucleos // 2)
    torch.set_num_threads(threads)
    print(f"\n{nucleos} vCPU · {threads} threads · working set {a.gib} GiB por buffer\n")

    n = int(a.gib * GIB / 4)
    # quatro buffers em rodízio: o working set tem de passar da LLC, senão mede cache
    tensores = [torch.empty(n, dtype = torch.float32) for _ in range(4)]
    for t in tensores:
        t.uniform_()

    dram = banda_dram(tensores, a.segundos)
    print(f"dram (leitura, {threads} threads) {dram:>8.1f} GB/s")

    if not torch.cuda.is_available():
        print("\nsem placa: PCIe e contenção não medidos")
        return

    host = torch.empty(int(0.5 * GIB / 2), dtype = torch.float16, pin_memory = True)
    dev = torch.empty_like(host, device = "cuda")
    h2d = banda_pcie(host, dev, a.segundos, "h2d")
    d2h = banda_pcie(host, dev, a.segundos, "d2h")
    print(f"pcie h2d                      {h2d:>8.1f} GB/s")
    print(f"pcie d2h                      {d2h:>8.1f} GB/s")

    # as duas ao mesmo tempo: cada lado reporta os SEUS bytes sobre o SEU tempo
    parar = threading.Event()
    saida = {}
    def lado_dram():
        saida["dram"] = banda_dram(tensores, a.segundos, parar)
    th = threading.Thread(target = lado_dram)
    th.start()
    saida["pcie"] = banda_pcie(host, dev, a.segundos, "h2d", parar)
    parar.set()
    th.join()
    print(f"\ncontenção: dram {saida['dram']:>8.1f} GB/s   pcie {saida['pcie']:>8.1f} GB/s")

    perda_dram = 100 * (1 - saida["dram"] / dram) if dram else 0
    perda_pcie = 100 * (1 - saida["pcie"] / h2d) if h2d else 0
    print(f"           perda {perda_dram:>7.0f} %          perda {perda_pcie:>7.0f} %")

    modelo = "desconhecida"
    try:
        for linha in open("/proc/cpuinfo"):
            if linha.startswith("model name"):
                modelo = linha.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    print(f"\ncpu: {modelo}  ·  {nucleos} vCPU")
    print("Comparar com `bandaDaRamDoHost` do hub (cpu-do-host.ts) usando ESTE nome e estes\n"
          "núcleos. Não reproduzir a fórmula aqui: ela tem um ramo por família, e a versão\n"
          "simplificada que ficava nesta linha errou por 5× num EPYC 7C13 ao aplicar a curva\n"
          "do Zen 4 a um Milan.\n")


if __name__ == "__main__":
    main()
