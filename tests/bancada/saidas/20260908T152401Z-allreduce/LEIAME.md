# Quanto custa um all-reduce do TP (08/09/2026)

Bancada: instância 50281568, 4× RTX 3090, EPYC 7B13, $1,08/h, ~7 min, ~$0,13. Destruída ao fim.
Instrumento: `tests/bancada/medir_allreduce.py` (prova `SO_15`, só torch — não compila a extensão
nem baixa o corte). Fork b90a33a.

**As quatro placas não têm P2P entre si** (`nvidia-smi topo -m`: PHB/NODE), então toda coletiva
desce à memória do host. É o caso comum em host alugado sem NVLink, e o que o hub assume ao
forçar o backend `nccl`.

## Decode: 1 token, 8 KB (µs por all-reduce)

| placas | bf16 | fp32 | bf16 sem dependência | só enfileirar |
|---|---|---|---|---|
| 2 | 64,6 | 89,9 | 29,4 | 64,5 |
| 3 | 63,5 | 85,7 | 40,6 | 63,5 |
| 4 | 36,8 | 82,9 | 34,5 | 36,6 |

## Prefill: 2048 tokens, 16 MB (ms por all-reduce)

| placas | bf16 | fp32 |
|---|---|---|
| 2 | 4,32 | 4,52 |
| 3 | 8,54 | 8,65 |
| 4 | 12,14 | 12,21 |

## O que isso diz

1. **A coletiva cresce com o número de placas.** No prefill, 2 → 4 placas triplica o custo
   (4,3 → 12,1 ms). Sem P2P o custo é proporcional aos participantes, e essa é a explicação de
   por que **TP4 perde para TP2** no decode (etapa 4: +7–14 % para duas placas).
2. **O desvio fp32 é caro no decode e barato no prefill.** As três conversões de
   `model_tp_backend.py` custam ~50 µs fixos por coletiva: metade dos ~85 µs do decode, e 2 % dos
   4,3 ms do prefill. Nas 90 coletivas por token do GLM-5.3 dá **~4 ms**, contra um passo de 26.
3. **O gargalo do decode é lançar, não transferir.** `só enfileirar` ≈ `bf16` para payload
   pequeno: a CPU não consegue emitir mais rápido do que a coletiva anda. Casa com o decode
   depender do clock da CPU (74 tok/s no EPYC 9555 contra 42 no Bergamo, mesma placa).
4. **Sobrepor coletivas não renderia.** `bf16 sem dependência` empata com o serial em 4 e 8
   tokens; a dependência de dados não é o que limita. Essa saída está fechada.

O ruído da primeira linha de cada tabela é visível (1 token sai mais lento que 4 em 2 e 3
placas): 20 aquecimentos não bastam para o NCCL estabilizar buffers por tamanho. As colunas
`fp32` são estáveis em todas as leituras, e é sobre elas que a conclusão 2 se apoia.

## Consequência no código

`TPBackendNCCL.all_reduce` passou a estreitar o fio para bf16 **só acima de 1 MiB**
(`EXLLAMA_TP_LIMIAR_FIO_BF16`). O backend nativo nunca teve o desvio: `pg_all_reduce_cpu` reduz
fp32 no host com `PARCK_MODE_FLOAT`.

**Não provado ainda:** o ganho em tok/s no modelo e a KL. A mudança altera a numérica do decode
em TP — para melhor, porque tira um arredondamento, mas precisa de medida antes de produção.
