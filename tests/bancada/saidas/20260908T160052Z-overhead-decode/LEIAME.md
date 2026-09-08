# Overhead do decode: backend, fio do all-reduce, contagem por expert (08/09/2026)

Bancada: instância 50285454, 2× RTX 3090, EPYC 7B13, $0,55/h, ~35 min, ~$0,45. Destruída ao fim.
Corte do Flash TR3 (4 camadas: 3 KDA + 1 MLA/MoE), TP2, prova `SO_16`, fork dfd4a8e.

## 16a. Contagem por expert, isolada (µs por chamada)

| | parede | enfileirar | × 42 camadas |
|---|---|---|---|
| `torch.bincount` | 54,7 | 54,7 | 2,30 ms |
| `scatter_add_` | 21,5 | 21,4 | 0,90 ms |

`enfileirar ≈ parede` nos dois: é tempo de CPU, não de GPU. O bincount custa 2,5×.

## 16b. Decode no corte, TP2 (média de 3 rodadas, tok/s)

| backend | fio | decode | prefill |
|---|---|---|---|
| nccl | bf16 | 272,0 | **17.789** |
| nccl | fp32 | 275,1 | 14.138 |
| native | bf16 | **280,8** | 16.842 |
| native | fp32 | 280,0 | 16.824 |

## 16c. Com o bincount de volta (nccl, fp32)

274,1 tok/s contra 275,1 do `scatter_add_`: +0,35 %, dentro do ruído — esperado, porque o corte
tem UMA camada MoE contra 42 do inteiro.

## O que isso decide

1. **O backend nativo ganha do NCCL no decode**: +2,5 %, consistente nas 6 rodadas de cada. São
   ~8 µs a menos por coletiva, o que no inteiro (90 por token) daria 0,7–1,3 ms dos 26. Mas ele
   perde 5 % no prefill, e trocar exige revisitar o incidente da A100 de 05/09 (o all-reduce
   nativo ATIVO, `pg_all_reduce_cpu`, passa pela memória do host e não usa P2P; quem usa é o
   `pg_all_reduce` de GPU, desabilitado por comentário do upstream).
2. **O limiar de 1 MiB é necessário, não opcional**: forçar fp32 em tudo custa **−20,5 %** de
   prefill no NCCL (17.789 → 14.138). O nativo não muda, porque nunca converteu.
3. **O microbenchmark superestima de 2 a 3×.** A prova 15 previa 4,5 ms de ganho com o fio fp32;
   dentro do modelo o decode subiu 1,1 %. Idem para a contagem: o isolado prevê 0,9 % no corte,
   medido 0,35 %. Provável causa: no modelo a coletiva se sobrepõe ao resto da fila, então o
   custo marginal é menor que o isolado. **Corrigir para baixo toda extrapolação de prova
   isolada.**

**Não medido:** o comportamento PADRÃO (limiar 1 MiB, que é bf16 no prefill e fp32 no decode).
Os dois regimes foram medidos isolados e o padrão pega o melhor de cada, mas isso é inferência.
