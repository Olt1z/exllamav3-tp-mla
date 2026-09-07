# Medidas da etapa 9 (07/09/2026)

## R1 — corte, 1× RTX 5090 (50109943, Coreia, $0,51/h), 503 × 2048, fases (EXL3_CONVERT_TIMING)
- densas (0–2): load 0,1 · capture 16,6 · quantize 10,8 · save 0,1 · advance 33,0 → 61 s  (camada 0: capture 51,6 por aquecimento)
- MoE (3): load 2,1 · capture 163,6 · quantize 301,7 · save 1,7 · advance 39,5 → 516 s (659 s na máquina anterior)
- grupos: concat n=16 K=4 134M pesos 5,4 s (24,8 M pesos/s); batched n=16 K=4 134M 5,5 s; K=8 100M 3,4 s (29,6 M/s)
- hc_expand advance 23,2 s; embed advance 4,2 s
- quantize da MoE ≈ 36 × 5,4 + 18 × 5,5 + ~6 = 300 s ✔

## R2 — mesma máquina, -cr 250 (fases)
- densas: capture 8,4 · quantize 10,8 · advance 17 → 37 s (era 61)
- MoE: load 1,7 · capture 80,3 · quantize 301,3 · save 1,1 · advance 19,9 → 411 s (era 516); RSS 34,8 GB (era 66,6)
- quantize por grupo (n=16, 134M): finalize 0,0 · regularize 0,1 · g_scale 0,9 · ldlq 4,3–4,4 · post 0,0
- quantização total invariante às linhas: 54 grupos → ~235 s de LDLQ + ~49 s de g_scale
- KL vs BF16 (mesma placa): 503 linhas 0,00249/0,00795 · 250 linhas 0,00276/0,00806 (+11 % na média)
- decode dos dois artefatos ~446 tok/s; BF16 254 tok/s

## Passo 11 — mesmo host (2× RTX 5090 Letônia, 50114439, $0,95/h; placas limitadas a 400/425 W, fábrica 575/600), 250 linhas
- solo (1 placa): MoE capture 98,6 · quantize 392,5 (7,1 s por grupo) · advance 8,7; densas 26 s; total 706 s
- dois processos independentes, um por placa: p0 quantize 392,9 (idêntico ao solo) → ZERO disputa de host; p1 morreu no meio da camada 3 sem erro no log (provável OOM de host com dois estados pinados)
- um processo em 2 placas: MoE load 11,0 · capture 50,2 · quantize 214,7 (7,0 s por grupo) · advance 6,3 → 283 s (solo 501 s, 1,77×); densas 15 s; total 419 s (1,68×)
- conclusão: o "1,3×" de antes comparava hosts diferentes (Coreia sem limite de energia: 5,4 s por grupo; Letônia a 400 W: 7,1 s). O caminho paralelo por threads já escala ~1,8× por placa; processo por placa não tem o que resolver.
- custo fixo que sobra em multi-GPU: carga das réplicas por camada (11 s) e o aquecimento da camada 0 (85 s na rodada anterior)
