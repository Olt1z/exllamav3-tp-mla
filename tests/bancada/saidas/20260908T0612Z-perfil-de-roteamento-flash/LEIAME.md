# Perfil de roteamento — GLM-5.3-Flash ABLITERATED EXL3 4.0bpw

Colhido em 08/09/2026 na instância 50236668 (1× RTX PRO 6000 S, Xeon Platinum 8559C), pelo
`EXL3_MOE_CPU_SPLIT_STATS_OUT` do fork, servindo trabalho real pelo hub — Playground, agente e a
régua dos três prompts. **42 camadas MoE, 48,3 milhões de seleções de expert.**

Contagens por **id de roteador** (a ordem do checkpoint), nunca por slot físico: vale como
colocação estática em qualquer máquina, com qualquer arranjo de placas.

## Como usar

    EXL3_MOE_CPU_SPLIT_STATS=<caminho deste arquivo>

O hub já aponta as duas variáveis (`_STATS` e `_STATS_OUT`) para
`/workspace/perfil-de-roteamento.json`: se o arquivo existe, ele manda; se não, a colocação
dinâmica age e o colhe. Arquivo de outro modelo é recusado pela largura (288 experts).

## O que ele mostra

Mandando para a RAM os **198 experts mais frios** de cada camada, em vez dos 198 últimos por
índice, a fração das leituras que cai na memória lenta vai de **68,8 % para 22,3 %** — 3,1× menos,
sem mudar quantos experts moram lá nem tocar no contexto.

Medido na mesma máquina, régua dos três prompts, colocação por índice contra o swap já aquecido:

| prompt | por índice | com a colocação aprendida | ganho |
|---|---|---|---|
| curto | 8,1–8,4 tok/s | 9,8–10,0 | +19 a +21 % |
| longo | 12,5 | 14,6 | +17 % |
| difícil (30k) | 11,8 | 14,4 | +22 % |

O prefill não se move (42→45, 461→470, 585→588): a colocação só toca o decode. A aceitação do
DFlash 2 no prompt difícil subiu de 47 % para 60 %.

**É piso, não teto:** a rodada "por índice" já aquece o swap enquanto roda, então a distância real
entre colocação arbitrária e colocação por frequência é maior.
