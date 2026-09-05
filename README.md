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

A infraestrutura de TP é genérica e já cobre a `Attention` comum, o `GatedDeltaNet`, os
experts, as hyper-connections e os caches MLA/DSA, que replicam o latente por rank. O buraco é
a própria `MLAttention`. Desenho: **replicar o pequeno, fatiar as cabeças.**

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
- [ ] 6. Teste de fumaça com 2 placas no corte de 4 camadas do GLM-5.3
- [ ] 6b. Corte do Flash (3 KDA + 1 MLA), mesmo teste, flag no `glm5_next`
- [ ] 6c. DeepSeek V3 e Mistral-4: corte, mesmo teste, flag
- [ ] 7. PR para o upstream (fase 1)
- [ ] 8. Fase 2: ligar a flag do Qwen 3-Next e do Qwen 3.8-Flash-Next e provar com corte
- [ ] 9. Fase 3: Olmo, LFM 2.5, AfMoE

Portão de cada família: `tests/tp_mla_smoke.py`. Carrega o modelo com `tensor_p=True` em 2
placas, gera 64 tokens greedy e compara os logits com a placa única (o ExLlamaV3 não é
determinístico bit a bit; a comparação é por KL contra o ruído medido entre duas execuções
iguais), e confere que as duas placas passam de 30 % de uso durante a geração. Modelos
grandes são provados por um corte de poucas camadas quantizado pelo hub.

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

O hub bl4ck0ut baixa o ExLlamaV3 por tarball do GitHub no arranque da máquina; quando a
família passar no teste de fumaça, o commit fixado passa a apontar para este fork e o
TabbyAPI é lançado com `tensor_parallel: true` quando há mais de uma placa.

## Licença

MIT, como o upstream. Todo o mérito do motor é do turboderp e da comunidade do ExLlama.
