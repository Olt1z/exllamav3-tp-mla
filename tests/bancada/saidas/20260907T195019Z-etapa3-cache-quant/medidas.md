# Etapa 3: cache quantizado no latente da MLA — 07/09/2026, 2× RTX 3090 + EPYC 7532, corte nativo mul1

Caminho DENSO (prefill 1.500 + 128 tokens < index_topk 2.048), KL contra a base fp16 numa placa; TP pelo fio bf16.

| arranjo | camada de cache | 8.192 tokens | decode tok/s | KL média | KL máx | top-1 |
|---|---|---|---|---|---|---|
| 14a. base fp16, uma placa | MLA_fp16 | 12,5 MiB | 247,8 | — | — | — |
| 14b. uma placa, Q8 | MLA_quant | 8,8 MiB (−30 %) | 259,9 | **0,00014** | 0,00292 | 100 % |
| 14b. uma placa, Q6 | MLA_quant | 7,8 MiB (−38 %) | 260,2 | 0,00087 | 0,00322 | 100 % |
| 14b. uma placa, Q4 | MLA_quant | 6,8 MiB (−46 %) | 260,8 | 0,00125 | 0,00344 | 100 % |
| 14c. TP2 canais, fp16 | MLA_fp16 | 12,5 MiB | 290,1 | 0,00028 | 0,00306 | 100 % |
| 14d. TP2 canais, Q8 | MLA_quant | 8,8 MiB | 291,7 | 0,00026 | 0,00302 | 100 % |
| **14e. TP2 canais, Q8, SPLIT=144 (arranjo alvo)** | MLA_quant | 8,8 MiB | 275,2 | **0,00033** | 0,00317 | 100 % |
| 14f. o mesmo pelo ambiente (EXL3_TP_MOE_TENSOR_SPLIT=1), caminho do TabbyAPI | MLA_quant | 8,8 MiB | 273,6 | 0,00036 | 0,00312 | 100 % |

Três leituras. (1) **Q8 no latente é de graça em qualidade**: KL 0,00014 numa placa, abaixo do ruído do próprio TP em fp16 (0,00028); Q6 e Q4 custam 3–5× isso e seguem com top-1 igual. (2) **O cache não cai pela metade, cai ~30 % no Q8**: só o latente (512 × fp16) é quantizado; a chave do indexador DSA e as escalas ficam em fp16 (8,8 de 12,5 MiB). No Flash inteiro (22.784 B/token, 11 camadas) o Q8 tira ~25 %, não 50 %: 22,3 → ~16,7 GiB a 1M, e a fração de experts na RAM em 2 placas com TP fica em ~22 %, não 14 %. (3) **Caminho ESPARSO não aceita cache quantizado**: primeira rodada, 14g com prompt de 16k, `mla_attn.py:1015` "sparse DSA over a quantized MLA cache is not supported yet" — acima de `index_topk` o Flash fica em fp16. O arranjo completo TP2 + Q8 + split funciona e é exato no denso, inclusive pelo ambiente, que é como o TabbyAPI do hub chega nele.
