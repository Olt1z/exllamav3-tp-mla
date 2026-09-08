<h1 align="center">ExLlamaV3 · <code>tp-mla</code></h1>

<p align="center">
  <a href="https://github.com/turboderp-org/exllamav3"><img alt="upstream" src="https://img.shields.io/badge/upstream-exllamav3%20v1.4.8-1f6feb"></a>
  <img alt="cuda" src="https://img.shields.io/badge/CUDA-12.4%2B-76b900">
  <img alt="python" src="https://img.shields.io/badge/Python-3.10%2B-3776ab">
  <a href="#licença"><img alt="license" src="https://img.shields.io/badge/license-MIT-blue"></a>
</p>

<p align="center">
  <b>Tensor parallel para os modelos de atenção latente</b> que o ExLlamaV3 ainda gera<br>
  numa placa por vez — mais um grafo CUDA que aceita atenção em 16 bits,<br>
  um rascunho DFlash&nbsp;2 e um conversor mais rápido.
</p>

<p align="center">
  Todo o resto é o upstream, intocado.
  Leia a <a href="https://github.com/turboderp-org/exllamav3#readme">documentação original</a> para o que não estiver aqui.
</p>

---

## Em resumo

| | O que faz | Efeito medido |
|---|---|---|
| **TP na `MLAttention`** | Fatia as cabeças de query entre as placas e replica o latente | Destrava GLM-5.3, GLM-5.3-Flash, DeepSeek V3/V3.1/R1, Kimi K2 e Mistral-4 em várias placas |
| **Atenção fp16 no grafo CUDA** | `BC_MLAttention` e `BC_GatedDeltaNetSplit` aceitam projeções em 16 bits | **+37 % a +69 %** de decode, KL ≤ 0,00007 |
| **Rascunho DFlash 2** | Arquitetura de rascunho própria, com taps no modelo alvo | 2,9–4,4 tokens aceitos por rodada; ganha do MTP em todos os prompts |
| **Conversor mais rápido** | Captura sem sincronizar, estado em memória pinada, cronômetro por fase | **1,7×** mais rápido e KL melhor (0,00249 contra 0,00422) |

---

## Tensor parallel para atenção latente

Na v1.4.8 o padrão do motor é `supports_tp: True`, e a maioria das famílias já gera em várias
placas. As que recusavam faziam isso por causa de uma peça só: a `MLAttention`, cujos
`make_tp_allocation` e `tp_export` levantavam `NotImplementedError`.

O desenho é **replicar o pequeno, fatiar as cabeças**:

| Peça | Em cada rank |
|---|---|
| `q_a_proj`, `kv_a_proj_with_mqa`, normas, indexador DSA, k-pool, rope | replicados |
| `q_b_proj`, `w_uk_flat`, `w_uv_flat` | colunas das cabeças do rank |
| `o_proj` | linhas das cabeças do rank, seguido de all-reduce |
| caches MLA/DSA | uma instância por rank |

Toda dimensão sai do módulo — `num_q_heads`, `qk_nope_head_dim`, `qk_rope_head_dim`,
`v_head_dim` —, nunca de constante de um modelo. É o que faz a mesma implementação servir ao
GLM-5.3 (rope 64, nope 192), ao Flash (NoPE, nope 256, k-pool) e ao DeepSeek V3.

O mesmo trabalho destravou a variante **KDA** do `GatedDeltaNet` (projeções `q`/`k`/`v`
separadas, portas `f`/`g`, sem `z_proj`), que o Flash usa em 34 das suas 45 camadas.

Cada rank guarda uma cópia inteira do cache latente. É MQA: uma cabeça só, sem por onde fatiar
por cabeça — a mesma escolha que vLLM e TensorRT-LLM fazem.

### Prazo dos coletivos, configurável

O backend nativo aborta o grupo quando um rank espera demais num coletivo. O upstream fixa 90 s
dentro do kernel; aqui o prazo vive no `PGContext` e vem de `EXLLAMA_TP_SYNC_TIMEOUT`. A primeira
geração compila os kernels Triton dos caminhos MLA e KDA em cada rank, e o mestre ainda compila
os do rascunho — 90 s estouram numa máquina sã.

---

## Atenção em 16 bits dentro do grafo CUDA

Antes, o caminho de grafo exigia projeções EXL3, e qualquer checkpoint com atenção em 16 bits
decodificava em modo eager. Agora `BC_MLAttention` (`q_a`, `q`, `kv_a`, `o`, `idx_wq_b`) e
`BC_GatedDeltaNetSplit` no modo KDA (`qkv`, `o`) aceitam `BC_LinearFP16`.

Um GEMM EXL3 é um nó de kernel com sites patcháveis; um fp16 é um nó cuBLAS sem site nenhum,
então roda entre buffers estáticos: `x` copiado uma vez para `x_st` no topo do grafo, saída do
`o_proj` em `y_st` copiada para `y` no fim, operandos com `R_pad = max(R, 8)` linhas — abaixo de
M = 8 o cuBLASLt escolhe um kernel cerca de 13× mais lento.

<div align="center">

| Arranjo | eager | grafo | ganho | KL | top-1 |
|---|---|---|---|---|---|
| 1 placa, contexto curto | 140 tok/s | 192 tok/s | **+37 %** | 0,00001 | 100 % |
| 1 placa, 3.000 tokens | 125 tok/s | 185 tok/s | **+48 %** | 0,00006 | 100 % |
| TP2, contexto curto | 137 tok/s | 195 tok/s | **+42 %** | 0,00007 | 100 % |
| TP2, 3.000 tokens | 117 tok/s | 198 tok/s | **+69 %** | 0,00002 | 100 % |

</div>

---

## Rascunho DFlash 2

Arquitetura de rascunho própria (`exllamav3/architecture/dflash2.py` e
`modules/arch_specific/dflash2.py`), que lê estados do modelo alvo por taps em vez de rodar um
modelo separado do zero. Funciona com o alvo em tensor parallel, com os ids crus dos taps — a
mesma convenção da implementação de referência e do SGLang.

Aceita **2,9 a 4,4 tokens por rodada** e ganha do MTP em todos os prompts da régua na mesma
máquina: +10 a 15 % em prompt curto, +50 a 65 % em prompt longo, +29 % no difícil.

---

## Conversor mais rápido

O `convert_model.py` do upstream ganhou quatro mudanças que valem em qualquer modelo:

- **Captura sem sincronizar o host.** `capture_H` não faz mais um `sync` por chamada; linhas com
  `inf`/`nan` viram zero e são descontadas por um contador que vive no device.
- **Estado de calibração em memória pinada** (`EXL3_PIN_STATE`, ligado por padrão). O avanço
  entre camadas caiu de 40 s para 15 s no corte medido.
- **Cronômetro por fase e por grupo** (`EXL3_CONVERT_TIMING=1`): carga, captura, quantização,
  gravação e avanço, com subfases dentro da quantização.
- **Passo opcional na etapa fina da busca de escala** (`EXL3_GSCALE_STAGE2_STRIDE`).

Mais três correções de robustez: o `ckpt_old` é apagado antes de compilar (o pico de disco
derrubava conversões grandes); o `compile` não grava o mesmo tensor em dois shards; e o
`quantization_config.bits` sai inteiro, com o valor exato em `bpw`, que é o que o validador do
Hugging Face Hub exige.

Resultado no mesmo corte e com a mesma receita: **16,8 min contra 28** do caminho anterior, com
KL **0,00249 contra 0,00422**. Duas placas no mesmo host escalam **1,8×** na quantização.

---

## Experts na RAM do host

Esta parte é **do upstream** — o fork apenas a documenta e a mediu de ponta a ponta, porque ela
quase não aparece na documentação original. Ela deixa um MoE grande caber numa placa que não
comportaria os pesos: os experts roteados moram na RAM e um worker de CPU os calcula.

| Variável | O que significa |
|---|---|
| `EXL3_MOE_CPU_OFFLOAD=N` | **N camadas MoE inteiras** vão para a CPU. Não é interruptor: `=1` manda só a primeira camada elegível |
| `EXL3_MOE_CPU_SPLIT=N` | Os **N experts de cauda de cada camada** vão para a CPU, com colocação dinâmica entre quente e frio |
| `EXL3_MOE_CPU_THREADS` | Threads do worker; o padrão é metade dos núcleos |

Exigências: experts com codebook **`mul1`**. Sob **tensor parallel** o split funciona desde 07/09
no modo de canais (`EXL3_TP_MOE_TENSOR_SPLIT=1` junto com `EXL3_MOE_CPU_SPLIT`), contribuição
deste fork; o offload de camada inteira (`EXL3_MOE_CPU_OFFLOAD`) continua só na divisão por
camadas, porque é lá que a camada existe inteira num lugar só.

### Quais experts moram na RAM

`EXL3_MOE_CPU_SPLIT=N` manda os **N últimos por índice**, que é uma escolha arbitrária: o
roteamento de um MoE é muito desigual, e mandar experts quentes para a memória lenta custa caro em
cada token. Dois caminhos corrigem isso, e o segundo ganhou o lado que faltava:

| Variável | Padrão | Para quê |
|---|---|---|
| `EXL3_MOE_CPU_SWAP` | `1` | Colocação dinâmica: promove o expert mais quente da CPU ao slot do mais frio da GPU |
| `EXL3_MOE_CPU_SWAP_INTERVAL` | `128` | Passos de decode entre varreduras |
| `EXL3_MOE_CPU_SWAP_FLOOR` | `8.0` | Piso para promover, em múltiplos da expectativa uniforme |
| `EXL3_MOE_CPU_SWAP_HYST` | `2.0` | Razão quente/frio exigida para trocar |
| `EXL3_MOE_CPU_SPLIT_STATS` | — | Colocação **estática** por um perfil de roteamento; exige `EXL3_MOE_CPU_SWAP=0` |
| `EXL3_MOE_CPU_SPLIT_STATS_OUT` | — | **Escreve** esse perfil enquanto o modelo serve (deste fork) |

Os padrões do modo dinâmico são conservadores: com 288 experts, o piso de 8× a expectativa
uniforme exige 28 acertos num mesmo expert dentro da janela de 128 passos, e respostas curtas de
agente terminam antes. Para uso interativo vale afrouxar — `EXL3_MOE_CPU_SWAP_INTERVAL=32`,
`EXL3_MOE_CPU_SWAP_FLOOR=2`.

O caminho estático é o mais forte, porque a colocação já nasce certa e vale para toda máquina
futura. Só faltava produzir o arquivo: **`EXL3_MOE_CPU_SPLIT_STATS_OUT=<caminho>`** despeja as
contagens por camada, indexadas por id de roteador, no formato que `EXL3_MOE_CPU_SPLIT_STATS` lê
de volta. Escreve depois de cada varredura, e não no encerramento, porque máquina alugada
costuma morrer sem desligar limpo; a troca é atômica. Sirva um dia de trabalho **real** com o
despejo ligado — perfil de prompt sintético roteia perto do uniforme e não ensina nada — e depois
suba com `EXL3_MOE_CPU_SWAP=0` e `EXL3_MOE_CPU_SPLIT_STATS` apontando para ele.

Medido num modelo de 321 B (180 GB em EXL3 4 bpw) numa placa de 94 GB com 314 GB de RAM:

<div align="center">

| Experts de cada camada na RAM | Uso da placa | Decode | Prefill |
|---|---|---|---|
| 288 de 288 (tudo) | ~13 GB | 15,0 tok/s | 1.395–1.458 tok/s |
| 224 (78 %) | ~50 GB | 19,8–21,3 tok/s | 1.580–1.634 tok/s |
| 192 (67 %) | ~65 GB | 22,1–23,6 tok/s | 1.685–1.752 tok/s |
| 160 (56 %) | 81 GB | 24,5–25,9 tok/s | 1.819–1.888 tok/s |

</div>

**O prefill paga menos que o decode, mas paga.** O motor faz streaming dos experts pela placa na
leitura do prompt em vez de calculá-los na CPU, e num corte de poucas camadas o custo some. No
modelo inteiro ele aparece: medido em 08/09 no mesmo modelo de 321 B, em 2 placas com 28,8 % dos
experts na RAM, o prefill ficou em **1,65k tok/s** contra 4,2–4,8k em 4 placas com tudo na VRAM.
Metade das placas explica cerca de metade da diferença; o resto é o offload. O decode continua
sendo onde dói, e é quase linear na fração que está na RAM.

---

## Variáveis de ambiente

As que este fork acrescenta ou torna configuráveis:

| Variável | Padrão | Para quê |
|---|---|---|
| `EXLLAMA_TP_SYNC_TIMEOUT` | `90` | Prazo dos coletivos nativos, em segundos |
| `EXL3_BC_GDN` | `1` | Caminho fundido do KDA no grafo CUDA |
| `EXL3_BC_GDN_TRACE` | — | Traço do caminho do KDA |
| `EXL3_BC_GDN_EAGER` | — | Roda o caminho fundido sem captura, para comparar |
| `EXL3_PIN_STATE` | `1` | Estado de calibração em memória pinada, na conversão |
| `EXL3_CONVERT_TIMING` | — | Tempo por fase e por grupo, na conversão |
| `EXL3_GSCALE_STAGE2_STRIDE` | — | Passo da etapa fina da busca de escala |
| `EXL3_TP_MOE_TENSOR_SPLIT` | — | Experts na RAM sob tensor parallel, no modo de canais |
| `EXL3_MOE_CPU_SPLIT_STATS_OUT` | — | Despeja o perfil de roteamento por camada, para a colocação estática |

---

## Instalar

Torch com CUDA 12.4 ou mais novo instalado à parte, depois:

```sh
pip install -r requirements.txt
MAX_JOBS=4 pip install -e .
```

`TORCH_CUDA_ARCH_LIST` restringe a compilação a uma arquitetura e economiza minutos:
`8.6` para RTX 3090, `8.9` para 4090, `12.0` para RTX PRO 6000 e 5090.

Para instalar num commit fixo, como um servidor faria:

```sh
pip install git+https://github.com/Olt1z/exllamav3-tp-mla@<commit>
```

---

## Testes

```sh
python tests/tp_mla_smoke.py --tp        # carrega em 2 placas e compara com a placa única
python tests/test_tp_mla_import.py       # importação exata da MLAttention fatiada
python tests/test_dflash2_referencia.py  # o rascunho contra a implementação de referência
```

O teste de fumaça gera 64 tokens greedy e compara os logits com a placa única. A comparação é
por KL contra o ruído medido entre duas execuções idênticas: o ExLlamaV3 não é determinístico bit
a bit, e cerca de 9 % das posições mudam entre leituras iguais.

---

## Licença

MIT, como o upstream. Todo o mérito do motor é do
[turboderp](https://github.com/turboderp-org/exllamav3) e da comunidade do ExLlama; este fork
acrescenta as peças acima e devolve o que for genérico.
