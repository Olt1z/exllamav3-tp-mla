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
- [ ] 6d. Decode em grafo CUDA no rank TP (hoje desligado por `has_split_cache`) e tokens/s antes/depois
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
do Flash amostrando pelo `lm_head` dos ranks (`tp_dispatch_lm_head_argmax`). A velocidade em TP4
ainda não foi medida com a saída correta: etapa 7.

## Timeout dos coletivos nativos

O backend nativo aborta o grupo inteiro quando um rank espera mais que o prazo num coletivo
(`## Synchronization timeout in kernel: ...`). O upstream fixa 90 s dentro do kernel. Aqui o prazo
vive no `PGContext` e vem de `EXLLAMA_TP_SYNC_TIMEOUT` (segundos, padrão 90): a primeira geração
compila os kernels Triton dos caminhos MLA e KDA em cada rank, e o rank mestre ainda compila os do
draft MTP, então um rank atrasado pode estourar 90 s numa máquina sã. O backend NCCL só usa esse
prazo nos coletivos que caem no fallback nativo (broadcast e gather).

## Branches e upstream

- `tp-mla`: o trabalho, criado da tag `v1.4.6` do upstream.
- `master`: espelho do upstream, sem commits nossos.

O fork do GitHub vem sem as tags. Antes de qualquer checkout por tag:

```sh
git remote add upstream https://github.com/turboderp-org/exllamav3.git
git fetch upstream --tags
```

Rebase sobre cada tag nova do upstream até o merge do PR:

```sh
git fetch upstream --tags
git rebase v1.4.7 tp-mla
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
