"""
Reaproveitamento de prompt em modelo hibrido (atencao linear + atencao com K/V), sem torch.

Modelo hibrido so retoma o cache de prefixo onde ha um checkpoint do estado recorrente. Este modulo
junta os interruptores (lidos do ambiente uma vez, todos desligados por padrao) e a logica pura que
decide ONDE guardar os checkpoints, para ser testada sem GPU nem torch.

Interruptores:

  EXL3_REPLAY=1
      Um prompt identico a um guardado comeca a gerar sem prefill. O primeiro token sai do passo de
      geracao, que ja processa o ULTIMO token do prompt (o prefill vai so ate o penultimo); basta
      restaurar o estado do fim do prefill -- o checkpoint parcial do fim do prompt, com o carry do
      MTP -- sem deixar um token sobrando para o prefill. Nao ha linha de logits guardada: o passo de
      geracao recalcula a mesma linha a partir do mesmo estado, e a saida e a da primeira vez.
      Tambem sobe o limite de checkpoints parciais (um por conversa) de 4 para 32.

  EXL3_PONTOS_DE_GUARDA=1
      Checkpoints recorrentes tambem em pontos estaveis compartilhados entre conversas, alinhados a
      pagina: o primeiro token de abertura de turno do usuario (`<|user|>` no GLM: o fim do bloco de
      sistema) e o fim do prefixo de K/V que o prompt ja divide com outra conversa em cache. Mais as
      posicoes que o chamador pedir (`Job(pontos_de_guarda = [...])`, sempre honradas).

  EXL3_GUARDA_TOKENS="<|user|>"
      Os tokens (texto exato, separados por virgula) cuja primeira ocorrencia e um ponto de guarda.

  EXL3_INTERVALO_PP=8192
      Intervalo dos checkpoints longe do fim do prompt (padrao do gerador: 32.768).

  EXL3_MAX_PARCIAIS=N
      Checkpoints parciais (fim de prompt no meio da pagina) guardados; padrao 4, ou 32 com replay.

  EXL3_RECORRENTE_GIB=G / EXL3_RECORRENTE_MAX=N
      Teto em bytes (substitui o `recurrent_cache_size` do gerador) e em quantidade dos checkpoints
      na RAM do host.

Com replay ou pontos de guarda ligados, o LRU dos checkpoints despeja primeiro os de GERACAO
(tirados depois do fim do prompt), que quase nunca voltam a servir: o turno seguinte de um agente
nao repete o raciocinio, e uma nova tentativa sorteia outra resposta. Sem isso, uma resposta longa
(um laco, justamente o que dispara a nova tentativa) gera um checkpoint a cada 2.048 tokens e
empurra para fora os do prompt -- e a nova tentativa comeca do zero.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

MAX_PONTOS = 4                 # pontos de guarda por job, no maximo
MAX_PARCIAIS_REPLAY = 32       # checkpoints parciais guardados com EXL3_REPLAY=1


def _ligado(env, nome: str) -> bool:
    return (env.get(nome, "") or "").strip().lower() not in ("", "0", "off", "false", "no")


def _inteiro(env, nome: str, minimo: int = 0) -> int | None:
    v = (env.get(nome, "") or "").strip()
    if not v:
        return None
    n = int(v)
    if n < minimo:
        raise ValueError(f"{nome}: {minimo} ou mais, nao {v!r}")
    return n


@dataclass
class ConfigReuso:
    replay: bool = False
    guarda: bool = False
    tokens_de_guarda: list[str] = field(default_factory = lambda: ["<|user|>"])
    intervalo_pp: int | None = None
    max_parciais: int | None = None
    recorrente_bytes: int | None = None
    recorrente_max: int = 0

    @property
    def proteger_prompt(self) -> bool:
        """O LRU dos checkpoints despeja os de geracao antes dos do prompt."""
        return self.replay or self.guarda

    def limite_parciais(self, padrao: int) -> int:
        if self.max_parciais is not None:
            return self.max_parciais
        return max(padrao, MAX_PARCIAIS_REPLAY) if self.replay else padrao


def ler_config(env = None) -> ConfigReuso:
    env = os.environ if env is None else env
    tokens = [t.strip() for t in (env.get("EXL3_GUARDA_TOKENS") or "<|user|>").split(",") if t.strip()]
    gib = (env.get("EXL3_RECORRENTE_GIB", "") or "").strip()
    return ConfigReuso(
        replay = _ligado(env, "EXL3_REPLAY"),
        guarda = _ligado(env, "EXL3_PONTOS_DE_GUARDA"),
        tokens_de_guarda = tokens,
        intervalo_pp = _inteiro(env, "EXL3_INTERVALO_PP", 1),
        max_parciais = _inteiro(env, "EXL3_MAX_PARCIAIS", 1),
        recorrente_bytes = int(float(gib) * 1024**3) if gib else None,
        recorrente_max = _inteiro(env, "EXL3_RECORRENTE_MAX", 0) or 0,
    )


# Lido uma vez: o interruptor e para medir, nao para trocar em voo. Os testes trocam este objeto.
CONFIG = ler_config()


def prefixo_de_kv(paginas_cheias: list[bool]) -> int:
    """Quantas paginas iniciais ja tem K/V valido (a cadeia de hashes do prompt achada no cache)."""
    n = 0
    for cheia in paginas_cheias:
        if not cheia:
            break
        n += 1
    return n


def pontos_de_guarda(
    page_tokens: int,
    inicio: int,
    fim_prefill: int,
    explicitos = (),
    aberturas = (),
    prefixo_kv: int = 0,
    max_pontos: int = MAX_PONTOS,
) -> list[int]:
    """
    As posicoes, alinhadas a pagina, onde um job guarda o estado recorrente alem dos checkpoints de
    intervalo e do fim do prompt.

    `inicio`: onde o prefill comeca (o prefixo ja em cache); `fim_prefill`: ate onde ele vai (o
    penultimo token do prompt); `explicitos`: pedidos pelo chamador; `aberturas`: primeira ocorrencia
    de cada token de guarda (o fim do bloco de sistema); `prefixo_kv`: posicao ate onde o K/V do
    prompt ja existe no cache (outra conversa com o mesmo comeco), em tokens.

    Cada candidato desce para a fronteira de pagina e so fica se cai depois de `inicio` e antes da
    ultima fronteira do prompt (que o prefill ja guarda sempre). Prioridade, se passar de
    `max_pontos`: explicitos, depois aberturas, depois o prefixo de K/V.
    """
    if page_tokens <= 0 or max_pontos <= 0:
        return []
    ultima = fim_prefill // page_tokens * page_tokens
    escolhidos: list[int] = []
    for grupo in (explicitos, aberturas, (prefixo_kv,) if prefixo_kv else ()):
        for p in grupo:
            g = int(p) // page_tokens * page_tokens
            if inicio < g < ultima and g not in escolhidos and len(escolhidos) < max_pontos:
                escolhidos.append(g)
    return sorted(escolhidos)


def limite_do_reuso_parcial(restante: int, fim_do_prompt: bool, replay: bool, tem_carry: bool,
                            precisa_carry: bool) -> int:
    """
    Quantos tokens, no maximo, o checkpoint parcial pode pular de um trecho de `restante` tokens.

    Sem replay sobra sempre um token para o prefill (o MTP precisava de um token real do alvo para o
    carry). Com replay, no trecho que termina no fim do prompt, pode pular tudo -- desde que o carry
    do MTP venha junto quando o rascunho e MTP.
    """
    if replay and fim_do_prompt and (tem_carry or not precisa_carry):
        return restante
    return restante - 1
