# Backend do TP decidido por medida, e a banda real de um EPYC 7C13 (08/09/2026)

Bancada: instância 50290038, 2× RTX 3090, EPYC 7C13 (128 vCPU, 126 GB), $0,31/h, ~25 min, ~$0,15.
Destruída ao fim. Corte do Flash TR3, TP2, prova `SO_18`, fork 56267b0.

## Decode (tok/s, média de 3 rodadas)

| backend | 4096 tok | 16384 tok |
|---|---|---|
| nccl | 253,7 | 252,0 |
| **native** | **260,1** | **257,9** |
| native + reduce na GPU | 259,2 | 258,6 |

## Prefill (tok/s, média de 3 rodadas)

| backend | 4096 tok | 16384 tok |
|---|---|---|
| nccl | 17.674 | **18.789** |
| native | **17.706** | 17.930 |
| native + reduce na GPU | 14.837 | 15.149 |

## O que isso decide

1. **O nativo ganha +2,5 % de decode, e o número reproduz.** A prova 16 mediu +2,5 % num EPYC
   7B13; esta mediu +2,5 % num 7C13. Duas máquinas, dois hosts, mesmo resultado.
2. **No prefill o nativo empata em chunk 4096** (17.706 contra 17.674) e perde 4,6 % em 16384.
   A perda de 5 % que a prova 16 viu era do host, não do backend: o `chunk-size` de produção é
   4096, e nesse tamanho não há perda.
3. **O caminho de GPU do upstream SOBE, e é pior**: −16 % de prefill (14.837 contra 17.706), sem
   mexer no decode (o limiar de 2 MiB mantém a coletiva pequena na CPU, como projetado). Ele
   funciona, não trava, não corrompe — é só mais lento que copiar em bloco e somar na CPU.
   **Isso é a explicação mais provável de o upstream tê-lo deixado comentado**, e fecha a
   pergunta: não há ganho escondido ali. Manter desligado.

## Banda do host

```
128 vCPU · 64 threads · working set 2,0 GiB por buffer
dram (leitura, 64 threads)    103,0 GB/s
pcie h2d                       24,9 GB/s
pcie d2h                       22,4 GB/s
contenção: dram  91,2 GB/s  ·  pcie  12,8 GB/s
           perda    11 %          perda    48 %
```

**Sob contenção quem sofre é o PCIe (−48 %), não a DRAM (−11 %).** É o regime real do decode com
experts na RAM, e não dava para deduzir das duas medidas isoladas — foi por isso que a medição
entrou no fluxo padrão de toda bancada.

**Defeito achado por causa desta medida:** o `EPYC 7C13` não casava em nenhum regex de
`cpu-do-host.ts` e caía no padrão de 77 GB/s, contra 103 medidos. O regex `7[\dB]\d[23]` só
previa a letra `B`; os SKUs de nuvem da AMD (`7C13`, `7V13`, `7R32`, `7T83`, `7J13`) ficavam
todos de fora, e são os mais comuns nas ofertas da Vast. Corrigido para `7[\dA-Z]\d[23]`.

**Armadilha da própria prova:** o `medir_banda.py` imprimia "a fórmula do hub" reproduzindo uma
versão simplificada dela, e mentiu por 5× aqui (disse 600 GB/s, +482 %) ao aplicar a curva do
Zen 4 a um Milan. A fórmula real tem um ramo por família. O script agora imprime só o nome da CPU
e os núcleos, e manda comparar com a função de verdade.
