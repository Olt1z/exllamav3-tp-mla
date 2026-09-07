# Prova do merge da v1.4.8 (commit 1e4da3d) — 07/09/2026, 2× RTX 3090, host de 16 threads AVX2, corte nativo mul1

Passo 13 (TP + experts na RAM) e passo 14 com **prefill de 3.000 > index_topk 2.048: caminho ESPARSO da DSA**, KL contra a base numa placa; TP pelo fio bf16.

| arranjo | KL média | KL máx | top-1 |
|---|---|---|---|
| 13b. uma placa, SPLIT=144 | 0,00005 | 0,00011 | 100 % |
| 13c. TP2 canais fp16 | travou no all-reduce (o travamento conhecido do 1º processo TP, 600 s); o mesmo arranjo passou em 14c | | |
| 13d. TP2 canais, SPLIT=144 (worker uma vez, rank de saída) | 0,00011 | 0,00487 | 100 % |
| 13e. TP2 expert-parallel + SPLIT (avisa, carrega inteiro) | 0,00006 | 0,00492 | 100 % |
| 14b. uma placa, Q8, esparso | **0,00001** | 0,00019 | 100 % |
| 14b. uma placa, Q6, esparso | 0,00003 | 0,00066 | 100 % |
| 14b. uma placa, Q4, esparso | 0,00027 | 0,00407 | 99,2 % |
| 14c. TP2 canais fp16, esparso | 0,00006 | 0,00324 | 99,2 % |
| 14d. TP2 canais Q8, esparso | 0,00009 | 0,00353 | 100 % |
| **14e. TP2 canais + Q8 + SPLIT=144, esparso (arranjo alvo)** | **0,00010** | 0,00353 | 99,2 % |
| 14f. o mesmo pelo ambiente (caminho do TabbyAPI) | 0,00008 | 0,00327 | 99,2 % |

14g (gerador, TP2 + Q8 + split): prompt de 16k → prefill 14,7k tok/s, decode 115 tok/s — na 1.4.7 este caso abortava em `mla_attn.py:1015`. Velocidades deste host não valem (16 threads AVX2, CPU-bound: 1 placa split 110 tok/s, TP2 sem split 257, TP2 split 131).

Conclusão: a 1.4.8 destrava o cache quantizado na atenção esparsa e ele compõe com o nosso TP + split sem perda de exatidão. O bloqueio da etapa 3 caiu.
