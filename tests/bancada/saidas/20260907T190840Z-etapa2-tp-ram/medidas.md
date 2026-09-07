# Etapa 2: TP + experts na RAM, modo de canais — 07/09/2026, 2× RTX 3090 + EPYC 7532 (64 vCPU, AVX2, sem AVX-512), corte nativo mul1

| arranjo | worker | decode tok/s | KL média contra 13a | KL máx | top-1 |
|---|---|---|---|---|---|
| 13a. uma placa, tudo na placa (base) | — | 247,6 | — | — | — |
| 13b. uma placa, SPLIT=144 (caminho antigo) | 1×, static | 178,8 | 0,00005 | 0,00018 | 100 % |
| 13c. TP2 canais, tudo na placa (repetição; a 1ª travou no all-reduce, o travamento conhecido do primeiro processo TP) | — | 295,0 | 0,00006 | 0,00487 | 100 % |
| **13d. TP2 canais, SPLIT=144 (o remendo)** | **1×, "TP output rank"** | 233,2 | **0,00007** | 0,00020 | 100 % |
| 13e. TP2 expert-parallel, SPLIT=144 | aviso, carrega inteiro | 261,0 | 0,00010 | 0,00484 | 100 % |
| 13f. gerador, TP2 canais sem split | — | 229,1 | | | |
| 13f. gerador, TP2 canais SPLIT=144 | 1× | 174,1 | | | |
| 13f. gerador, uma placa SPLIT=144 | 1× | 191,5 | | | |

Primeira rodada: todo carregamento em TP quebrou com "list indices must be integers, not dict" — `compile_tp_plan` devolve lista por device e o `tp_export` iterou os elementos como chaves (corrigido em 00a2e1c); a repetição rodou com `git pull` na mesma máquina, sem recompilar.

O que a prova diz: o desenho fecha — todo rank importa já encolhido para [0, 144) e mascara a cauda como um shard de TP; só o rank de saída registra a cauda no worker (uma linha "CPU split experts (worker, static, TP output rank)") e soma a parcela antes do all-reduce; a KL fica no nível do split de uma placa. Velocidade aqui não diz nada: kernel AVX2 num Zen 2, uma camada MoE de 4. Medir no EPYC 9654 e no modelo inteiro.
