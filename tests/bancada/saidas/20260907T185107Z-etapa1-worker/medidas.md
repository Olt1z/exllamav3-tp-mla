# Etapa 1: teto do worker de CPU — 07/09/2026, RTX 6000 Ada + EPYC 9654 (48 vCPU, 118 GB), corte nativo mul1

Banda crua (torch.sum de 4 GiB): 6 thr 105,9 · 12 thr 142,8 · 24 thr 207,6 · 36 thr 232,2 · 48 thr 210,0 GB/s.

Bytes por expert (4 bpw): 12.6 MB. Job = um token numa camada; experts na CPU por job = 8 × N/288 (colocação estática).

| variante | thr | experts/job | prep_gu | gemv_gu+gemv_d | outros | total µs | GB/s no gemv | decode tok/s |
|---|---|---|---|---|---|---|---|---|
| SPLIT=32 | 24 | 0.89 | 98 | 94 | 32 | 225 | 119 | 279.0 |
| SPLIT=64 | 24 | 1.78 | 91 | 133 | 32 | 256 | 168 | 278.9 |
| SPLIT=128 | 24 | 3.56 | 96 | 268 | 32 | 396 | 167 | 266.3 |
| SPLIT=192 | 24 | 5.33 | 85 | 337 | 32 | 455 | 199 | 261.0 |
| SPLIT=256 | 24 | 7.11 | 108 | 495 | 31 | 634 | 181 | 250.5 |
| OFFLOAD=1 (camada inteira) | 24 | 8.00 | 95 | 625 | 43 | 763 | 161 | 238.1 |
| SPLIT=144 | 12 | 4.00 | 126 | 473 | 29 | 628 | 106 | 250.2 |
| SPLIT=144 | 24 | 4.00 | 116 | 239 | 33 | 388 | 210 | 267.2 |
| SPLIT=144 | 36 | 4.00 | 85 | 141 | 54 | 280 | 358 | 274.6 |
| SPLIT=144 | 48 | 4.00 | 304 | 153 | 37 | 494 | 329 | 242.1 |
| SPLIT=144 swap dinâmico | 24 | 4.00 | 118 | 233 | 32 | 384 | 216 | 260.5 |

Linha de base tudo na placa: 283,9 tok/s (decode de 1100 tokens, prompt de 4k).

Leituras: (1) o gemv escala linear nos experts e roda a 160–200 GB/s com 24 threads, ou seja, 78–98 % da banda crua do host: o worker é limitado por MEMÓRIA, não por desempacotar a treliça. (2) `prep_gu` é um custo FIXO de ~90–120 µs por camada e por token, independente de N — com 42 camadas MoE são ~4 ms por token. (3) 36 threads (3/4 dos núcleos) rende mais que o padrão de metade; 48 estoura o `prep_gu` (3×) por disputa com o processo principal. (4) O swap dinâmico não rendeu neste prompt. (5) Os 232 GB/s crus deste host contra os ~77 efetivos inferidos no EPYC 9V84 (VM de 40 vCPU) dizem que a ESCOLHA DO HOST é a maior alavanca, ~3×; a fórmula do hub precisa de banda por host, não de constante. Acima de 36 threads os GB/s aparentes passam da banda crua: parte dos experts quentes cabe no L3 de 768 MB, então os números do corte são teto otimista.
