# ExLlamaV3 · tp-mla (fork bl4ck0ut)

Fork de [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) com um objetivo:
**tensor parallel para os principais modelos open source que hoje geram com uma placa por vez
no ExLlamaV3/TabbyAPI**, para eles rodarem com todas as placas ao mesmo tempo em máquinas
baratas de 24 GB. Começa pelo GLM-5.3, mas o alvo é a lista inteira abaixo.

Tudo o mais é o upstream, intocado. O que não estiver descrito aqui, leia na
[documentação original](https://github.com/turboderp-org/exllamav3#readme).

## Quem já tem TP e quem não tem

Na v1.4.6 o padrão do motor é `supports_tp: True`. Llama, Qwen 3 e Qwen 3 MoE, Qwen 3.5,
Mixtral, GLM 4.5/4.6, Gemma 3/4, Mistral 3, DeepSeek V4, GPT-OSS, MiniMax-M2, Nemotron e
os demais sem flag explícita já geram em várias placas. Os que declaram `False`, agrupados
pela peça que falta:

| Peça sem TP | Famílias travadas | Fase |
| --- | --- | --- |
| `MLAttention` (`make_tp_allocation`/`tp_export` levantam `NotImplementedError`) | GLM-5.3 (`GlmMoeDsaForCausalLM`), GLM-5.3-Flash (`Glm5NextForConditionalGeneration`), DeepSeek V3 / V3.1 / R1 e Kimi K2 (`DeepseekV3ForCausalLM`), Mistral-4 | 1 |
| flag `False` com `GatedDeltaNet`, que já tem TP no módulo ("TODO: Enable TP for linear attn") | Qwen 3-Next (`Qwen3NextForCausalLM`), Qwen 3.8-Flash-Next (`Qwen4ExpForConditionalGeneration`) | 2 |
| normas de Q/K que abrangem todas as cabeças; híbridos sem TP | Olmo 3.1, Olmo-Hybrid, LFM 2.5, AfMoE | 3 |
| cabeças MTP (`*_mtp.py`) | todas | depois das fases 1 e 2 |

## Fase 1: TP na `MLAttention`

A infraestrutura de TP é genérica e já cobre a `Attention` comum, o `GatedDeltaNet` na variante
do Qwen3-Next, os experts, as hyper-connections e os caches MLA/DSA, que replicam o latente por
rank. Os buracos são a própria `MLAttention` e, descoberto em 05/09 na carga do TR3, a variante
KDA do `GatedDeltaNet` (Flash: `q/k/v_proj` separados, portas `f/g`, sem `z_proj`), que caía em
`assert self.z_proj is not None`. Desenho: **replicar o pequeno, fatiar as cabeças.**

| Peça | Em cada rank |
| --- | --- |
| `q_a_proj`, `kv_a_proj_with_mqa`, norms, indexador DSA, k-pool, rope | replicados |
| `q_b_proj`, `w_uk_flat`, `w_uv_flat` | colunas das cabeças do rank |
| `o_proj` | linhas das cabeças do rank, seguido de all-reduce |
| caches MLA/DSA | instância por rank (já era assim no upstream) |

Toda dimensão vem do módulo (`num_q_heads`, `qk_nope_head_dim`, `qk_rope_head_dim`,
`v_head_dim`), nunca de constante de um modelo: é o que faz a mesma implementação servir ao
GLM-5.3 (rope 64, nope 192), ao Flash (NoPE, nope 256, k-pool) e ao DeepSeek V3.

Custo aceito: cada rank guarda o latente inteiro (576 valores por token por camada no GLM-5.3).
É o que vLLM e TensorRT-LLM fazem.

## Andamento

- [x] 1. `make_tp_allocation` na `MLAttention`: divisão por cabeças de query
- [x] 2. `tp_export`: replicado × fatiado, incluindo os tensores do k-pool
- [x] 3. `tp_import`: `MLAttention` local com as cabeças do rank, com e sem rope
- [x] 4. `forward`: all-reduce depois do `o_proj` e cache por rank
- [x] 5. `supports_tp: True` no `glm_moe_dsa` (escrito; só conta como provado depois do 6)
- [x] 6. Teste de fumaça com 2 placas no corte de 4 camadas do GLM-5.3 (05/09, 2× RTX 3090; ver "Resultado da bancada")
- [x] 6b. Corte do Flash (3 KDA + 1 MLA), mesmo teste, flag no `glm5_next` (06/09, 2× RTX 3090; ver "Resultado da bancada, Flash")
- [ ] 6c. DeepSeek V3 e Mistral-4: corte, mesmo teste, flag
- [ ] 6d. Decode em grafo CUDA no rank TP: a guarda `has_split_cache` saiu de `build_bc_mla` (era legado; o upstream já tinha tirado a mesma da Attention em ccd5626); falta provar no corte (KL igual) e medir tokens/s antes/depois
- [ ] 7. PR para o upstream (fase 1)
- [ ] 8. Fase 2: ligar a flag do Qwen 3-Next e do Qwen 3.8-Flash-Next e provar com corte
- [ ] 9. Fase 3: Olmo, LFM 2.5, AfMoE

Portão de cada família: `tests/tp_mla_smoke.py`. Carrega o modelo com `tensor_p=True` em 2
placas, gera 64 tokens greedy e compara os logits com a placa única (o ExLlamaV3 não é
determinístico bit a bit; a comparação é por KL contra o ruído medido entre duas execuções
iguais), e confere que as duas placas passam de 30 % de uso durante a geração. Modelos
grandes são provados por um corte de poucas camadas quantizado pelo hub.

## Resultado da bancada (05/09/2026, 2× RTX 3090, corte de 4 camadas do GLM-5.3)

- **Importação exata.** `tests/test_tp_mla_import.py` e `tests/test_tp_mla_cache.py`: a `MLAttention`
  reconstruída por `tp_export`/`tp_import` é bit a bit igual à original, filho a filho, no caminho
  sem cache e com cache, denso e DSA esparso, prefill e decode, inclusive a camada `shared`.
- **Fatia por cabeças correta.** A soma dos forwards de dois módulos com metade das cabeças difere
  do módulo inteiro por 1e-3 a 3e-2 relativo: é o arredondamento do kernel EXL3 com a soma
  dividida (no prefill MHA em fp32 a diferença cai para 2e-5), não erro de índice.
- **A divergência do TP não é da atenção.** TP em 2 placas deu KL 0,020 contra uma placa, mas o TP
  com *uma* placa, sem fatiar nada, deu 0,023. Causa: o all-reduce do backend nativo manda tensor
  fp32 por um fio bf16 (`all_reduce_cpu.cu`: "fp32 payloads keep the bf16 wire for range"), e o
  GLM emite atenção e MLP em fp32. Simular esse arredondamento numa placa
  (`tp_mla_smoke.py --simular-fio-bf16`) reproduz KL 0,025; TP contra a simulação fica em 0,003.
  O Llama, com saídas fp16, dá 0,00005 no mesmo teste. Régua justa para modelos de saída fp32:
  comparar o TP com a simulação, não com a placa única crua.
- **Os artefatos da nossa esteira não carregavam.** O `kv_b_proj` saía quantizado, e o motor exige
  o peso cru. `tests/bancada/reparar_kv_b_proj.py` desquantiza e o hub passou a manter esse
  tensor fp16 (classe `kv-b`). O corte publicado já está reparado.
- Velocidade no corte não diz nada útil (4 camadas, latência de lançamento): 180 tok/s numa placa,
  230 em duas. A medida que vale é no modelo inteiro, etapa 7.

## Resultado da bancada, Flash (06/09/2026, 2× RTX 3090, corte de 4 camadas do TR3)

Corte por bytes do `brandonmusic/GLM-5.3-Flash-tr3-4bpw`, sem requantizar: camadas 0–3 (3 KDA
densas + 1 MLA NoPE com k-pool e MoE de 288 experts), publicado em
`Olt1z/GLM-5.3-Flash-tr3-4bpw-corte-4L`. `tests/test_tp_block_import.py` apontou o defeito em
minutos: a atenção KDA importada com o plano cheio diferia do original em 73–91 % relativo, MLA e
MoE eram exatos. Causa: `GatedRMSNorm.tp_export` não levava `gate_activation`, e a norma de saída
do KDA (sigmoid) virava silu no rank. Com a flag, KDA importado exato (rel ≤ 2,6e-6) e o TP em 2
placas contra a base: KL 0,00009 no prompt curto, 0,00002 com 2.974 tokens, top-1 igual em 100 %.

Antes disso, no modelo inteiro em 4× RTX PRO 6000 (TabbyAPI pelo hub), a carga e a geração em TP4
só chegaram ao fim depois de mais três consertos: `ulimit -n` (120 shards abertos), `HyperHead`
em modo de média (GLM-5.3 colapsa os fluxos por média, sem tensores) e as cabeças MTP do GLM-5.3 e
do Flash amostrando pelo `lm_head` dos ranks (`tp_dispatch_lm_head_argmax`).

**Modelo inteiro, 06/09/2026, 4× RTX PRO 6000 S (sem NVLink), TabbyAPI pelo hub, commit dfc2ad8,
backend NCCL, `--draft-mode mtp`:** saída correta (raciocínio e resposta coerentes em português).
Carga em TP4 em 1 min 05 s. Decode de 596 tokens a 63,8 tok/s (44,1 na primeira rodada quente),
draft MTP aceito em 48–66 %; prefill de 5.321 tokens a 312 tok/s, seguido de decode a 45 tok/s
com a resposta certa sobre o texto. A primeira requisição gasta ~50 s compilando os kernels
Triton. Referência do README do TR3: 145–151 tok/s em 2× PRO 6000 com DFlash2, outro drafter
e sem TP.

**Mesma máquina, com e sem `--tensor-parallel` (06/09/2026, 01:55Z, mesmos prompts, mesma
chave):** sem TP o autosplit reparte por camadas (95 + 96 + 6 + 0 GB, o cache de 1M enche as duas
primeiras placas) e decodifica 596 tokens a 54,1–54,7 tok/s; em TP4, 63,8 tok/s (+18 %). No prompt
de 5.321 tokens o autosplit estoura a memória da placa 1 (`torch.OutOfMemoryError` no transiente do
KDA, 93,3 de 95 GiB ocupados); em TP4 o mesmo prompt entra a 312 tok/s, porque cache e transientes
se dividem pelas quatro placas. A carga é mais rápida sem TP (50 s contra 65 s). Esses números são
com o decode da MLA em modo eager no rank: `build_bc_mla` recusava o grafo CUDA quando
`has_split_cache`. A guarda era legado do primeiro caminho C++ da MLA (d420170); o upstream tirou a
equivalente da Attention em ccd5626 ("TP shards are eligible: the shard owns its split cache
layers directly ... the output all-reduce runs after the captured block returns") e Attention,
MLP, MoE, KDA e DSv4 já decodificam em grafo dentro do rank. A MLA passa a fazer o mesmo depois de
dfc2ad8; o all-reduce fica fora do grafo, como nos outros. Ainda não provado nem medido (bancada).

## Bancada de desempenho (06/09/2026, 03:00Z, 2× RTX 3090, corte de 4 camadas do TR3)

Instância 50023236 ($0,28/h, ~40 min), commits 7b77298 a 68c7df3, logs em
`Olt1z/quantizacao-bl4ck0ut/saidas/tp-mla/20260906T024850Z/`. Teste por bloco idêntico ao de
00:45Z. O 4c caiu no primeiro prefill com `Synchronization timeout`: um rank compilando Triton
passou dos 90 s do coletivo nativo (a bancada agora exporta `EXLLAMA_TP_SYNC_TIMEOUT=600`); o 4d,
com o cache do Triton quente, passou com KL 0,00002 e top-1 igual em 100 %.

**O grafo CUDA não entra no TR3, com ou sem TP.** Com `EXL3_BC_ATTN_TRACE=1`, `build_bc_mla` diz
`DECLINED layer 3` também numa placa só. A causa não é a guarda `has_split_cache` (tirada em
7b77298; A/B na mesma máquina: 118 tok/s com e 117 sem), é o artefato: no
`brandonmusic/GLM-5.3-Flash-tr3-4bpw` só os experts estão em EXL3 (37.152 tensores `trellis`,
zero na atenção). KDA, MLA, MLPs densos, hyper-connections e embeddings estão em 16 bits
(`.weight`), e `_proj_ok` / `is_quantized_kda` exigem projeções EXL3 com classe BC. Logo os 45
blocos de atenção decodificam em modo eager, em qualquer arranjo. Nos artefatos EXL3 da nossa
esteira (`Olt1z/GLM-5.3-podado-4L-EXL3-balanced-bl4ck0ut`) `q_a/q_b/o_proj` são trellis, mas
`kv_a_proj_with_mqa` (576 colunas, não múltiplo de 128) fica em 16 bits, e `_proj_ok` o exige em
EXL3: o BC-MLA recusa a família GLM-5.3 inteira. O caminho para o grafo é o BC-MLA e o BC-KDA
aceitarem projeções fp16 (GEMM cuBLAS dentro da captura, como o `bc_attn` já faz para o gate
headwise), não o TP.

**Perfil do corte, aquecido, `perfil_prefill.py` (4 camadas: 3 KDA + 1 MLA/MoE; o prefill para no
último módulo com cache, então o MoE da camada 3 não roda no prefill):**

| Medida | 1× 3090 | TP2 (2× 3090) |
| --- | --- | --- |
| prefill 3.990 tokens | 219 ms (18,2k tok/s) | 218 ms (18,3k tok/s) |
| decode, por token | 13,5 ms (74 tok/s) | 9,6 ms (104 tok/s) |

Prefill numa placa, por módulo: GatedDeltaNet 35 % (`qkv_proj` fp16 sozinho 15 %), MLAttention
25 %, GatedMLP 24 % (as três projeções fp16, 8 % cada), embeddings e hyper-connections 9 %. TP2 não
ganha nada no prefill do corte. Decode numa placa, por token e camada: KDA 1,3 ms, MLA 3,0 ms,
GatedMLP 0,5 ms, `lm_head` 1,5 ms; a soma dos módulos fecha com o total, e cada projeção fp16 de
GEMV gasta ~0,3 ms onde a banda de memória daria 0,05 ms: é lançamento de kernel, não cálculo. É o
retrato de um decode eager, e o que o grafo resolveria. O primeiro perfil, sem aquecimento no mesmo
tamanho, marcou 10,6 s de "prefill" que eram compilação Triton do scan do KDA em 4k tokens; a
lição está no script.

## Etapa 4 do plano de desempenho (06/09/2026, 04:00–05:05Z): o inteiro medido

Logs em `Olt1z/quantizacao-bl4ck0ut/saidas/tp-mla/20260906T035300Z-etapa4/`. Duas máquinas de
4× RTX PRO 6000 S pelo hub (a primeira, em leilão, foi tomada por outro locatário no meio; a
segunda a preço fixo). Mesma placa, mesmo software, hosts diferentes, e é isso que aparece:

| Host | CPU | decode curto | prefill 5,4k | prefill 30k | decode 30k |
| --- | --- | --- | --- | --- | --- |
| 148856 (EPYC 9555, 4,4 GHz) | Turin 64c | 73–75 tok/s | 4.223 T/s | 4.837 T/s | 92 tok/s |
| 81589 (EPYC 9754 Bergamo) | Zen4c 128c | 42 tok/s | 3.342 T/s | 1.338 T/s | 52 tok/s |

**O decode é limitado pela CPU e pelo barramento, não pela placa.** O motor cru em TP4 gasta 26
ms por passo (38 tok/s sem rascunho) no host rápido, contra um piso de banda de ~2,5 ms: é eager
(atenção em 16 bits, grafo recusado) e faz 90 all-reduces por token por PCIe. Um núcleo Zen4c
de clock baixo derruba 40 % do tok/s com a mesma GPU. Para o hub, `cpu_name`/`cpu_ghz` da oferta
pesam tanto quanto a placa neste motor.

**TP2 basta e não perde (host lento, mesma máquina, TabbyAPI relançado à mão com
`CUDA_VISIBLE_DEVICES=0,1`, contexto 256k):** carga em 140 s, 91,2 + 89,9 GB de 95,6 ocupados.

| Arranjo | decode curto | prefill 5,4k | decode 5,4k | prefill 30k | decode 30k |
| --- | --- | --- | --- | --- | --- |
| TP4, 1M | 42 tok/s | 3.342 T/s | 47 tok/s | 1.338 T/s | 52 tok/s |
| TP2, 256k | 46–47 tok/s | 3.044 T/s | 54 tok/s | 3.382 T/s | 56 tok/s |

Metade dos all-reduces vale mais do que o dobro de placas: decode +7 a +14 %, prefill igual
(−9 % em 5k, e o 30k em TP4 neste host foi anômalo). Duas PRO 6000 servem o TR3 a 256k.

**Onde o tempo vai no inteiro** (`perfil_prefill.py`, autosplit, host lento, 3.990 tokens em
1.417 ms; decode 62,8 ms/token):

| Módulo | prefill | decode |
| --- | --- | --- |
| BlockSparseMLP (MoE, 41 camadas) | 58 % | 19 % |
| GatedDeltaNet (KDA, 34) | 20 % | 34 % |
| MLAttention (11, com indexador DSA) | 12 % | 28 % (1,6 ms por camada por token) |
| HyperConnection mix+apply, RMSNorm | 7 % | 10 % |

Em TP4 NCCL o mesmo prefill de 3.990 tokens fez 743 ms (5,4k tok/s no motor cru, host rápido).

**O que o card do TR3 faz de diferente:** 145 tok/s com 5,4 tokens aceitos por passo são ~27
passos de forward por segundo; nós fazemos 73 tok/s com ~2 aceitos, ~36 passos por segundo. O
forward do ExLlamaV3 em TP4 é mais rápido que o do vLLM do card; a diferença é o drafter.

## Etapa 5 do plano de desempenho (06/09/2026, 16:36–17:09Z): DFlash 2 no inteiro

Logs e JSONs em `tests/bancada/saidas/20260906T1636Z-etapa5/`. Uma 4× RTX PRO 6000 S pelo hub
(instância 50077792, Texas, Xeon 6767P, $7,01/h a preço fixo, 33 min, ~$4), TR3 do Flash a 1M,
TabbyAPI do hub com `--draft-mode model` (rascunho `incoai/GLM-5.3-Flash-DFlash2`), depois
relançado à mão na MESMA máquina para os outros arranjos. Régua `medir_tabby.py`, temperatura 0,
linha `Metrics` do servidor:

| Arranjo | curto (decode · aceito) | longo 5,4k (prefill · decode · aceito) | difícil 30k (prefill · decode · aceito) |
| --- | --- | --- | --- |
| DFlash 2, TP4 (hub, tap_shift 1) | 45–52 tok/s · 27–29 % | 2.852 T/s · 79 tok/s · 65 % | 1.199 T/s · 68 tok/s · 49 % |
| DFlash 2, TP4, tap_shift 0 | 50–57 tok/s · 28–33 % | 2.781 T/s · 72 tok/s · 65 % | – |
| DFlash 2, autosplit (sem TP) | 42 tok/s · 25–26 % | 2.110 T/s · 76 tok/s · 63 % | – |
| MTP, TP4 | 41–49 tok/s · 59–60 % | 2.780 T/s · 48 tok/s · 86 % | 2.801 T/s · 53 tok/s · 96 % |

**O rascunho carrega e funciona de ponta a ponta pelo hub** (primeira carga real pelo TabbyAPI:
"Loading draft modules 7/7", 7 tokens rascunhados por rodada) e **ganha do MTP em todos os
prompts nesta máquina**: +10–15 % no curto, +50–65 % no longo, +29 % no difícil. Mas não chega
nem perto do card (145 tok/s): aceita 2,9 tokens por rodada no curto e 4,4 no difícil (o card
diz 5,4 + 1), e cada rodada custa ~65 ms contra 26 ms do passo cru, porque verificar 8 tokens
no MoE eager lê ~6× mais experts. Prompt de código sem raciocínio não foi medido: o template
ignorou `enable_thinking: false` e as duas leituras (700 e 800 tokens de raciocínio) deram 32 %
e 45 % — a variação entre leituras idênticas é grande.

**Este host é lento para o decode** (Xeon 6767P, 3,9 GHz de turbo): o MTP fez 41–53 tok/s
aqui contra 73–92 no EPYC 9555 da etapa 4, mesma placa. A comparação vale dentro da máquina.

**Três hipóteses de integração testadas e descartadas:** (1) TP: sem TP a aceitação é a mesma;
(2) índice dos taps: tap_shift 0 e 1 aceitam igual — o fork fica com 0, que é o que a referência
(`hidden_states[layer_id + 1]`) e a captura do SGLang para o glm5_next (PR 36708: entrada da
camada `layer_id + 1`, média dos 4 fluxos mHC) descrevem, e é o que `export_state_layers` já
exporta; (3) representação: a média dos fluxos é a mesma do SGLang. O que a prova com taps
aleatórios não cobre e ainda não foi comparado: o caminho do gerador com taps reais (KV do
rascunho atualizado por rodada, posições, janela bilateral do `is_causal: false`).

**Achado extra:** com o DFlash 2 o prefill de 30k caiu para 1.199 T/s (MTP na mesma máquina:
2.801; em 5k os dois empatam). Superlinear no contexto — a exportação de estados ou o
`update_kv_from_target` sobre 30k tokens; medir com `perfil_prefill.py --export`.

**A exatidão do difícil (soma e agulha) falhou nos dois arranjos**: é do modelo a 4 bpw com 30k
de contexto, não do rascunho (o alvo verifica, o decode é lossless).

## Etapa 2 do plano de desempenho (06/09/2026, 17:24–17:54Z): prefill pelo gerador, no corte

Duas bancadas de 2× RTX 3090 ($0,24/h, ~$0,15 no total), corte de 4 camadas do TR3,
`tests/bancada/perfil_gerador.py` (prefill e decode pelo GERADOR, como o TabbyAPI: chunks,
exportação de estados por chunk, rascunho), 4k/16k/30k, aquecido no maior tamanho. Logs em
`Olt1z/quantizacao-bl4ck0ut/saidas/tp-mla/20260906T172411Z/` e `.../20260906T173814Z/`.

| Arranjo | 16k | 30k |
| --- | --- | --- |
| 1 placa, chunk 4096 | 17.150–17.190 tok/s | 17.690–17.750 tok/s |
| 1 placa, chunk 8192 | 17.020–17.050 | 17.340–17.390 |
| TP2 NCCL, chunk 4096 | 15.600 | 16.080–16.100 |
| TP2 NCCL, chunk 8192 | 15.390–15.480 | 15.750–15.800 |
| TP2 nativo, chunk 4096 | 15.410–15.470 | 15.200–15.360 |
| 1 placa + DFlash 2 (taps remapeados 0,1,2,3,3) | 12.140 (+0,40 s) | 11.020 (+1,03 s) |
| TP2 NCCL + DFlash 2 | 11.150 (+0,43 s) | 10.250 (+1,07 s) |

**Chunk 8192 não rende** (−1 a −2 % em todos os arranjos; no corte o MoE é 1 camada em 4, então
um ganho grande no inteiro é improvável e um pequeno não paga máquina de 4× PRO 6000 para medir).
**NCCL e nativo empatam** em TP2, e TP2 fica 9 % abaixo de uma placa no prefill, como no inteiro.
**O rascunho custa 24–34 µs por token de prefill** (exportação dos 5 taps + `update_kv_from_target`),
igual numa placa e em TP2 — a exportação pelos ranks não multiplica. Os +14 s do TR3 inteiro em 30k
com o DFlash 2 (etapa 5, 1.199 contra 2.801 T/s) não são isso: aquela foi a primeira leitura de 30k
da máquina e a régua só aquecia com o prompt de 5,4k, então o Triton compilou dentro da medida (o
MTP, medido depois no mesmo host, já achou os kernels no cache). A régua passa a aquecer com cada
tamanho que mede. Com taps remapeados a numérica do rascunho é lixo (0 aceito, o gerador corta o
rascunho na primeira rodada), então o decode com rascunho no corte não diz nada.

**O que sobra para o prefill**: nada barato. O motor cru já faz 4,2–4,8k tok/s no inteiro em TP4;
o card faz 6,2k com FlashInfer e DeepGEMM. O chunk, o backend e o TP não mudam isso.

## Etapa 3 do plano de desempenho (06/09/2026, 18:00–19:20Z): decode em grafo CUDA com atenção fp16

Antes, `build_bc_mla` e `is_quantized_kda` exigiam projeções EXL3, e todo checkpoint com atenção em
16 bits (o TR3 do Flash inteiro; nos nossos EXL3 o `kv_a_proj` de 576) decodificava eager. Agora
`BC_MLAttention` (q_a, q, kv_a, o, idx_wq_b) e `BC_GatedDeltaNetSplit` no modo KDA (qkv, o) aceitam
`BC_LinearFP16`: um GEMM EXL3 é um nó de kernel com sites patcháveis (A/C), um fp16 é um nó cuBLAS
sem site nenhum, então roda entre buffers estáticos — `x` copiado uma vez para `x_st` no topo do
grafo (`copy2d_gr`, patchável), saída do o_proj em `y_st` copiada para `y` no fim; operandos com
R_pad = max(R, 8) linhas (cuBLASLt escolhe kernel ~13× mais lento abaixo de M = 8). Helper
`linear_gr(exl3, fp16, x, y, xh, graph)` em `libtorch/linear.{h,cpp}`. Interruptores novos
`EXL3_BC_GDN=0`, `EXL3_BC_GDN_TRACE=1` e `EXL3_BC_GDN_EAGER=1`. Commits f98420d..a42908c.

Bancada de 2× RTX 3090 (instância 50085313, ~$0,40), corte de 4 camadas do TR3 (3 KDA + 1
MLA/MoE, toda a atenção fp16), `tp_mla_smoke.py` 128 tokens, eager × grafo por teacher forcing.
Saídas em `tests/bancada/saidas/20260906T182201Z-etapa3/`:

| Arranjo | eager | grafo | ganho | KL média | top-1 |
| --- | --- | --- | --- | --- | --- |
| 1 placa, contexto curto | 140 tok/s | 192 tok/s | +37 % | 0,00001 | 100 % |
| 1 placa, 3000 tokens (DSA esparso) | 125 tok/s | 185 tok/s | +48 % | 0,00006 | 100 % |
| TP2 NCCL, curto | 137 tok/s | 195 tok/s | +42 % | 0,00007 | 100 % |
| TP2 NCCL, 3000 tokens | 117 tok/s | 198 tok/s | +69 % | 0,00002 | 100 % |

**Dois defeitos achados e consertados no caminho.** (1) `LinearFP16.unswap_cpu` trocava o peso
pela cópia na placa e deixava `inner.bc` apontando para a cópia do host (carga fatiada): agora
religa o BC. (2) O import TP do `GatedRMSNorm` construía o BC do norm com cinco argumentos, sem a
flag do gate sigmoid do KDA que o `load()` passa: no rank, o caminho fundido do KDA aplicava silu
e toda camada linear divergia (KL 1,37, top-1 0 %), com estados recorrentes certos e todos os GEMMs
certos — achado por instrumentação estágio a estágio, porque o BC-KDA em TP nunca tinha rodado (a
fork só ganhou TP no glm5_next em 05/09 e os artefatos fp16 eram sempre recusados). É preexistente
e independente do fp16.

Falta o TR3 inteiro pelo hub, com a régua: o ganho de 37–69 % no corte (4 camadas) precisa ser
medido nas 45, onde o MoE (19 % do decode) já rodava em grafo e o custo por token era 26 ms.

## Etapa 8 do plano de desempenho (06/09/2026, 22:11–22:30Z): experts na RAM, no corte em mul1

Bancada de 1× RTX 6000 Ada (49 GB) num host EPYC 9654 (48 núcleos visíveis, AVX-512 VBMI, 118 GB de RAM,
instância 50103047, $0,63/h, ~$0,25), corte de 4 camadas do Flash quantizado em EXL3 4 bpw **mul1** pela esteira
do hub (`Olt1z/GLM-5.3-Flash-podado-4L-EXL3-4.0bpw-bl4ck0ut`, reparado na máquina: ver abaixo). Offload nativo
do ExLlamaV3 por variável de ambiente (`EXL3_MOE_CPU_OFFLOAD=1` = camada MoE inteira na CPU;
`EXL3_MOE_CPU_SPLIT=N` = os N experts de cauda de cada camada na CPU, colocação dinâmica), modo layer-split,
24 threads de worker (padrão: metade dos núcleos). `perfil_gerador.py`, decode de 128 tokens.
Saídas em `tests/bancada/saidas/20260906T2211Z-etapa8-corte/`.

| Arranjo (1 camada MoE de 288 experts) | decode | prefill 16k | prefill 30k |
| --- | --- | --- | --- |
| tudo na placa | 254 tok/s (3,94 ms/token) | 30,5k tok/s | 31,6k tok/s |
| 64 experts na CPU | 242–251 | 28,1k | 30,9k |
| 128 na CPU | 231–245 | 29,2k | 29,9k |
| 192 na CPU | 227–236 | 28,6k | 29,9k |
| 256 na CPU | 219–230 | 26,6k | 29,7k |
| camada MoE inteira na CPU | 215–216 (4,65 ms/token) | 29,8k | 30,0k |

**O que isso diz.** (1) **O prefill não paga**: com a camada inteira na RAM ele fica em 30k tok/s contra 31,6k,
porque o motor faz streaming dos experts pela placa no prefill em vez de calculá-los na CPU. A conta do plano
("600–1.200 tok/s") estava errada para melhor. (2) **A camada MoE inteira na CPU custa ~0,7 ms a mais por token**
neste host (8 de 288 experts ativos, mul1, 24 threads); descontando o que a placa gastava nela, a camada na CPU
sai em ~1 ms por token. Extrapolando para as 41 camadas MoE do Flash: ~40 ms por token só de experts na RAM,
ou seja, um teto de ~25 tok/s antes do resto da rede (atenção e densas na placa, ~20–30 ms num arranjo de uma
placa) → **~15 tok/s com todos os experts na RAM, e proporcionalmente mais com `moe_cpu_split` parcial**
(metade dos experts na CPU ≈ 0,4 ms/camada ≈ 25–30 tok/s). Um host com mais canais de memória e mais threads
sobe isso; é a régua a repetir no inteiro. (3) A colocação dinâmica (hot/cold) já funciona: o worker anuncia
`[160..288) of 288` e o intervalo migra conforme o roteamento.

**Três defeitos achados no caminho, todos fora do offload.** O artefato da esteira para o Flash não carregava no
motor: (a) as hyper-connections saíram com os nomes de módulo do transformers 5.16 (`attn_hc.fn`) em vez dos do
checkpoint (`hc_attn_fn`), idem `self_attn.forget_gate.{f_a_proj,f_b_proj,A_log,dt_bias}` → `reparar_nomes.py`
(36 tensores); (b) q/k/v do KDA foram quantizados em separado, com `suh` distintos, e o motor os funde num
`qkv_proj` → `reparar_qkv_kda.py` restaura os 9 pesos em BF16 do corte BF16; (c) no fork, `Linear.load` tentava o
carregador EXL3 com a `alt_key` em lista e estourava antes do fp16 → corrigido em d59e9c8. Os reparos estão no
plano da esteira rápida (`plans/2026-09-06-esteira-de-quantizacao-rapida.md`).

## Régua de desempenho

Três prompts fixos, sempre os mesmos, e uma linha por prompt. É o "antes" e o "depois" de toda
mudança de desempenho. Roda de dentro da máquina (o proxy corta pedidos acima de 100 s); com
`--log` lê a linha `Metrics` que o TabbyAPI escreve por pedido, que é a única fonte da aceitação
do draft:

```sh
python tests/bancada/medir_tabby.py http://127.0.0.1:5000 $TABBY_API_TOKEN --log /caminho/do/stdout.log
```

Linha de base, 06/09/2026 04:13Z, TR3 do Flash, 4× RTX PRO 6000 S (instância 50027193), TP4,
NCCL, `--draft-mode mtp`, hub em dfc2ad8, medida pelo script com nonce por pedido e lida das
linhas `Metrics` do servidor:

| Prompt | Entrada | Saída | Prefill | Decode | Draft aceito |
| --- | --- | --- | --- | --- | --- |
| curto ×2 | 67–69 tok | 596 tok | – | 73–75 tok/s | 57–60 % |
| longo | 5.448 tok | 196 tok | 4.223 tok/s | 82 tok/s | 85 % |
| difícil (1.000 registros) | 30.135 tok | 1.996 tok | 4.837 tok/s | 92 tok/s | 97 % |

Duas armadilhas que a régua aprendeu na mesma sessão: os 312 tok/s de prefill medidos à mão em
06/09 01:46Z eram a compilação Triton dos kernels de chunk na primeira leitura longa (5.436
tokens em 11,4 s; a segunda leitura sem cache fez 5.448 em 1,3 s), e a segunda rodada do mesmo
prompt bate no cache de prompt do TabbyAPI (5.376 de 5.436 tokens em cache) e mede um prefill
de 60 tokens. Por isso o script aquece com um prompt longo e prefixa cada pedido com um nonce.

Referência a bater (card do TR3, vLLM customizado, 2× PRO 6000 WS, DFlash2, grafos CUDA):
145–151 tok/s de decode e 6,2k tok/s de prefill.

O perfil por módulo (`tests/bancada/perfil_prefill.py`, etapa 5 da bancada) diz onde o prefill
e o decode gastam o tempo no corte, numa placa; em TP só o total vale, porque os módulos rodam
nos ranks.

## Timeout dos coletivos nativos

O backend nativo aborta o grupo inteiro quando um rank espera mais que o prazo num coletivo
(`## Synchronization timeout in kernel: ...`). O upstream fixa 90 s dentro do kernel. Aqui o prazo
vive no `PGContext` e vem de `EXLLAMA_TP_SYNC_TIMEOUT` (segundos, padrão 90): a primeira geração
compila os kernels Triton dos caminhos MLA e KDA em cada rank, e o rank mestre ainda compila os do
draft MTP, então um rank atrasado pode estourar 90 s numa máquina sã. O backend NCCL só usa esse
prazo nos coletivos que caem no fallback nativo (broadcast e gather).

**Travamento conhecido, ainda sem causa:** o primeiro processo TP numa máquina recém-instalada
trava de vez em quando no primeiro prefill, com os dois ranks girando em `pg_all_reduce_cpu_kernel`
e o auxiliar em `CPU reduce wait timeout`, até o prazo estourar. Visto em 06/09/2026 duas vezes
(hosts e commits diferentes, v1.4.6 e v1.4.7); nas duas o processo TP seguinte, na mesma máquina,
passou com KL normal. A hipótese é a primeira compilação Triton dentro de um forward em TP: uma
alocação bloqueante na CPU de um rank enquanto o kernel de espera do outro gira na GPU. A bancada
contorna com prazo curto e até três tentativas no 4c. O hub, com NCCL, carregou o TR3 inteiro
três vezes no mesmo dia sem travar.

## Branches e upstream

- `tp-mla`: o trabalho, sobre a tag `v1.4.7` do upstream desde 06/09/2026 (nasceu na `v1.4.6`;
  o rebase teve um conflito só, o conserto do `tp_import_split_n` do fp16 que o upstream fez
  igual, e o `timeout.cuh` ficou com o nosso prazo configurável mais o abort pegajoso deles).
- `tp-mla-v1.4.6`: a história anterior ao rebase, mantida porque o hub fixou commits dela
  (dfc2ad8) e eles precisam continuar alcançáveis.
- `master`: espelho do upstream, sem commits nossos.

O fork do GitHub vem sem as tags. Antes de qualquer checkout por tag:

```sh
git remote add upstream https://github.com/turboderp-org/exllamav3.git
git fetch upstream --tags
```

Rebase sobre cada tag nova do upstream até o merge do PR:

```sh
git fetch upstream --tags
git branch tp-mla-v1.4.7 tp-mla     # guarda a história que o hub fixa
git rebase v1.4.8 tp-mla
```

## Instalar (bancada)

Torch com CUDA 12.4+ instalado à parte, depois:

```sh
pip install -r requirements.txt
MAX_JOBS=4 pip install -e .        # compila a extensão para a placa da máquina
```

`TORCH_CUDA_ARCH_LIST` restringe a compilação a uma arquitetura (`8.6` para RTX 3090,
`8.9` para 4090, `12.0` para RTX PRO 6000/5090).

## Como o hub usa

O hub bl4ck0ut instala o ExLlamaV3 por `pip install git+<repo>@<commit>` no arranque da máquina.
Desde 05/09/2026 (`EXL3_REPO`/`EXL3_COMMIT` em `apps/api/src/data/exllama-capabilities.ts`) o
repo é este fork e o commit é o a4e4960; a lista de arquiteturas sem TP é gerada do mesmo
commit (`scripts/gen-exllama-registry.py`), e o TabbyAPI recebe `--tensor-parallel true` quando
a arquitetura não está nela e há mais de uma placa. Cada família só sai da lista depois do
teste de fumaça: o GLM-5.3 saiu em 05/09; o Flash sai na etapa 6b.

## Licença

MIT, como o upstream. Todo o mérito do motor é do turboderp e da comunidade do ExLlama.
