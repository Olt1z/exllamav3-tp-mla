"""
Rascunho por cópia (prompt lookup): quando os últimos tokens do contexto já apareceram antes, os
tokens que vieram depois daquela ocorrência viram o rascunho da rodada.

Resposta que cita ou edita o prompt (ou se repete) é barata de rascunhar: os últimos
``casamento`` tokens do contexto (o token pendente incluído) são procurados no prompt e na
resposta até aqui, e o que seguiu a ocorrência mais recente vira o rascunho. O rascunho só
propõe; a verificação continua amostrando cada posição como o caminho serial, então a saída é
a mesma.

Inspirado nos patches 0007 (glm-copy-drafts) e 0032 (glm-code-copy-drafts) do TensorFold:
- casamento de 8 tokens para ocorrências no prompt (``EXL3_COPIA_MIN_PROMPT``);
- ocorrência que começa DENTRO da resposta precisa casar 16 tokens (``EXL3_COPIA_MIN_RESPOSTA``):
  código repete clichês curtos (``    def __init__(self``) cuja continuação muda, e lá as cópias
  tiradas da própria resposta mantinham 23 % dos rascunhos contra 74 % das tiradas do prompt;
- até 15 rascunhos por rodada (``EXL3_COPIA_MAX``), o teto da janela de decode da MLA
  (``mla_attn.MAX_DECODE_QLEN`` = 16 linhas: o token pendente e 15 rascunhos).

A busca não varre o contexto a cada rodada (a 380k tokens e 16 jobs isso pesaria): os n-gramas
do contexto inicial vão para um vetor ordenado de hashes (busca binária), e os que chegam depois
para um dicionário. Cada rodada custa uma busca e a conferência dos candidatos, não O(contexto).

Só numpy, sem torch: testável sem placa (tests/test_rascunho_copia.py).
"""

from __future__ import annotations

import bisect
import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np

# Janela de verificação da MLA em modo decode: o token pendente + 15 rascunhos. Acima disso o
# kernel muda para o de prefill (ver mla_attn.MAX_DECODE_QLEN), então a cópia para aqui
TETO_RASCUNHOS = 15

_P = 0x100000001B3                 # primo do FNV-64; o hash é polinomial módulo 2^64
_MASK = (1 << 64) - 1
_MAX_CANDIDATOS = 2048             # acima disso confere os 16 primeiros e os mais recentes


def _hash_py(tokens) -> int:
    h = 0
    for t in tokens:
        h = (h * _P + int(t)) & _MASK
    return h


def _hashes_np(ctx: np.ndarray, n: int) -> np.ndarray:
    """Hash de cada n-grama de ``ctx`` (início 0 .. len - n), igual ao de ``_hash_py``."""
    m = ctx.shape[0] - n + 1
    if m <= 0:
        return np.empty((0,), dtype = np.uint64)
    c = ctx.astype(np.uint64)
    h = np.zeros((m,), dtype = np.uint64)
    p = np.uint64(_P)
    with np.errstate(over = "ignore"):
        for k in range(n):
            h = h * p + c[k:k + m]
    return h


@dataclass(frozen = True)
class CopiaConfig:
    """Ajustes do rascunho por cópia, lidos do ambiente (desligado por padrão)."""
    casamento: int = 8              # EXL3_COPIA_MIN_PROMPT
    casamento_resposta: int = 16    # EXL3_COPIA_MIN_RESPOSTA (0: igual ao do prompt)
    maximo: int = 15                # EXL3_COPIA_MAX
    linhas_lote: int = 16           # EXL3_COPIA_LINHAS_LOTE: linhas de verificação do lote que a cópia pode pedir
    passo: int = 4                  # EXL3_COPIA_PASSO: janelas estendidas pela cópia arredondadas a múltiplos disto

    @classmethod
    def do_ambiente(cls, env = None) -> CopiaConfig | None:
        """EXL3_RASCUNHO_COPIA=1 liga; os demais EXL3_COPIA_* ajustam. None: desligado."""
        env = os.environ if env is None else env
        liga = env.get("EXL3_RASCUNHO_COPIA", "0").strip()
        if liga in ("", "0"):
            return None
        if liga != "1":
            raise ValueError(f"EXL3_RASCUNHO_COPIA: 0 ou 1, não {liga!r}")

        def numero(nome: str, padrao: int, minimo: int, maximo: int) -> int:
            texto = env.get(nome, "").strip()
            if texto == "":
                return padrao
            if not texto.isdecimal() or not minimo <= int(texto) <= maximo:
                raise ValueError(f"{nome}: de {minimo} a {maximo}, não {texto!r}")
            return int(texto)

        casamento = numero("EXL3_COPIA_MIN_PROMPT", 8, 2, 64)
        resposta = numero("EXL3_COPIA_MIN_RESPOSTA", 16, 0, 64)
        if resposta and resposta < casamento:
            raise ValueError(f"EXL3_COPIA_MIN_RESPOSTA: 0 (igual ao do prompt) ou ao menos "
                             f"EXL3_COPIA_MIN_PROMPT ({casamento}), não {resposta}")
        return cls(
            casamento = casamento,
            casamento_resposta = resposta,
            maximo = numero("EXL3_COPIA_MAX", 15, 1, TETO_RASCUNHOS),
            linhas_lote = numero("EXL3_COPIA_LINHAS_LOTE", 16, 1, 1024),
            passo = numero("EXL3_COPIA_PASSO", 4, 1, 16),
        )

    def limite_por_job(self, jobs: int, rascunho_normal: int) -> int:
        """Rascunhos de cópia que cada job pode propor com ``jobs`` no lote.

        A janela é comum ao lote: um job com cópia longa faz todos verificarem aquele tanto de
        linhas. O orçamento é ``linhas_lote`` linhas no lote inteiro (16: um job sozinho vai a 15
        rascunhos, dois a 7, quatro ou mais ficam na janela do rascunho normal), e nunca abaixo
        da janela que o rascunho normal já usa."""
        if jobs < 1:
            return 0
        por_job = self.linhas_lote // jobs - 1
        return max(0, min(self.maximo, max(rascunho_normal, por_job)))

    def indice(self, contexto, prompt: int) -> CopiaIndice:
        return CopiaIndice(contexto, prompt, self.casamento, self.casamento_resposta, self.maximo)


class CopiaIndice:
    """Contexto de um job (prompt, resposta e o token pendente) e as propostas de cópia dele.

    ``prompt``: quantos tokens iniciais são o prompt original; ocorrências que começam nele ou
    depois são da resposta e precisam casar ``casamento_resposta`` tokens."""

    def __init__(self, contexto, prompt: int, casamento: int = 8, casamento_resposta: int = 16,
                 maximo: int = TETO_RASCUNHOS):
        if casamento < 1 or maximo < 1:
            raise ValueError("casamento e maximo têm de ser positivos")
        self.casamento = int(casamento)
        self.casamento_resposta = int(casamento_resposta) if int(casamento_resposta) > self.casamento else 0
        self.maximo = int(maximo)
        self.prompt = int(prompt)
        ctx = np.asarray(contexto, dtype = np.int64).reshape(-1)
        self.buf = np.empty((max(1024, 2 * ctx.shape[0]),), dtype = np.int64)
        self.buf[:ctx.shape[0]] = ctx
        self.L = int(ctx.shape[0])
        self._indexar_base()

    # ---- índice

    def _indexar_base(self):
        """n-gramas do contexto atual inteiro num vetor ordenado; os seguintes vão para ``extra``."""
        n = self.casamento
        h = _hashes_np(self.buf[:self.L], n)
        ordem = np.argsort(h, kind = "stable")         # estável: posições crescentes no mesmo hash
        self.base_h = h[ordem]
        self.base_p = ordem.astype(np.int64)
        self.base_n = int(h.shape[0])                  # inícios 0 .. base_n - 1 estão na base
        self.base_L = self.L
        self.extra: dict[int, list[int]] = {}
        self.extra_log: list[tuple[int, int]] = []     # (hash, início) na ordem de chegada

    def __len__(self) -> int:
        return self.L

    def tokens(self) -> list[int]:
        return self.buf[:self.L].tolist()

    def estender(self, tokens: Sequence[int]):
        """Tokens aceitos (o último é o pendente da próxima rodada) entram no contexto."""
        tokens = np.asarray(tokens, dtype = np.int64).reshape(-1)
        k = int(tokens.shape[0])
        if not k:
            return
        if self.L + k > self.buf.shape[0]:
            maior = np.empty((2 * (self.L + k),), dtype = np.int64)
            maior[:self.L] = self.buf[:self.L]
            self.buf = maior
        self.buf[self.L:self.L + k] = tokens
        n = self.casamento
        antigo = self.L
        self.L += k
        for fim in range(antigo, self.L):
            s = fim - n + 1
            if s < self.base_n:                         # já na base (ou antes do começo)
                continue
            h = _hash_py(self.buf[s:s + n].tolist())
            self.extra.setdefault(h, []).append(s)
            self.extra_log.append((h, s))

    def truncar(self, novo: int):
        """Volta o contexto a ``novo`` tokens (strings banidas rebobinam a sequência)."""
        novo = max(0, int(novo))
        if novo >= self.L:
            return
        if novo < self.base_L:
            self.L = novo
            self._indexar_base()
            return
        n = self.casamento
        while self.extra_log and self.extra_log[-1][1] + n > novo:
            h, s = self.extra_log.pop()
            lista = self.extra[h]
            lista.pop()
            if not lista:
                del self.extra[h]
        self.L = novo

    # ---- busca

    def ocorrencias(self) -> np.ndarray:
        """Inícios crescentes das ocorrências anteriores dos últimos ``casamento`` tokens que
        deixam ao menos um token depois delas (o próprio sufixo fica fora). Ocorrências que
        começam na resposta precisam casar ``casamento_resposta`` tokens."""
        L, n = self.L, self.casamento
        if L <= n:
            return np.empty((0,), dtype = np.int64)
        q = self.buf[L - n:L]
        hq = _hash_py(q.tolist())
        lim = L - n                                    # deixa um token depois; tira o sufixo
        # Os dois baldes ja estao em ordem crescente de inicio: o da base sai do argsort estavel
        # (posicoes crescentes no mesmo hash) e o do extra so tem inicios >= base_n, na ordem de
        # chegada. Corta e recorta por busca binaria e fatia, sem passar pelo balde inteiro: num
        # contexto repetitivo (1M tokens iguais) o balde tem 1M posicoes, e o filtro com np.sort
        # de antes custava ~20 ms por job a cada rodada
        b = np.empty((0,), dtype = np.int64)
        if self.base_n:
            hq_np = np.uint64(hq)
            lo = int(np.searchsorted(self.base_h, hq_np, side = "left"))
            hi = int(np.searchsorted(self.base_h, hq_np, side = "right"))
            if hi > lo:
                b = self.base_p[lo:hi]
                b = b[:int(np.searchsorted(b, lim, side = "left"))]
        e = self.extra.get(hq) or []
        ne = bisect.bisect_left(e, lim)
        nb = int(b.shape[0])
        if nb + ne == 0:
            return np.empty((0,), dtype = np.int64)
        k_ini, k_fim = 16, _MAX_CANDIDATOS - 16       # acima do teto: os 16 primeiros e os mais recentes
        if nb + ne <= k_ini + k_fim:
            c = np.concatenate((b, np.asarray(e[:ne], dtype = np.int64)))
        else:
            ini = b[:k_ini] if nb >= k_ini else \
                np.concatenate((b, np.asarray(e[:k_ini - nb], dtype = np.int64)))
            fim = np.asarray(e[ne - k_fim:ne], dtype = np.int64) if ne >= k_fim else \
                np.concatenate((b[nb - (k_fim - ne):], np.asarray(e[:ne], dtype = np.int64)))
            c = np.concatenate((ini, fim))
        # Colisão de hash: confere os tokens
        ok = (self.buf[c[:, None] + np.arange(n)] == q[None, :]).all(axis = 1)
        c = c[ok]
        extra = self.casamento_resposta - n if self.casamento_resposta else 0
        if extra and c.size:
            dentro = c >= self.prompt
            if dentro.any():
                bom = np.ones(c.shape, dtype = bool)
                ci = c[dentro]
                if L - n - extra < 0:
                    bom[dentro] = False
                else:
                    validos = ci >= extra
                    ci_v = np.where(validos, ci, extra)
                    antes_c = self.buf[ci_v[:, None] - np.arange(1, extra + 1)]
                    antes_q = self.buf[L - n - np.arange(1, extra + 1)]
                    bom[dentro] = validos & (antes_c == antes_q[None, :]).all(axis = 1)
                c = c[bom]
        return np.sort(c)

    def propor(self, espaco: int | None = None, teto_id: int | None = None) -> list[int]:
        """Até ``min(maximo, espaco)`` rascunhos: o que seguiu a ocorrência mais recente que tem
        esse tanto de tokens depois dela, senão o máximo depois de qualquer uma (a mais antiga);
        [] quando o sufixo nunca apareceu antes. ``teto_id``: corta no primeiro id fora do
        vocabulário (marcadores de imagem não podem ir para a verificação como texto)."""
        k = self.maximo if espaco is None else min(self.maximo, int(espaco))
        if k < 1:
            return []
        c = self.ocorrencias()
        if not c.size:
            return []
        L, n = self.L, self.casamento
        cheias = c[c <= L - n - k]
        s = int(cheias[-1]) if cheias.size else int(c[0])
        out = self.buf[s + n:min(s + n + k, L)].tolist()
        if teto_id is not None:
            for i, t in enumerate(out):
                if t < 0 or t >= teto_id:
                    return out[:i]
        return out


def combinar_janela(copias: list, modelo: list, pendentes: list, limite: int, passo: int = 1):
    """Monta as linhas de rascunho do lote.

    ``copias[j]``: a proposta de cópia do job j ([] sem cópia); ``modelo[j]``: o rascunho do
    MTP/DFlash2 para o job j ([] quando o rascunhador não rodou ou nada propôs); ``pendentes[j]``:
    o token pendente do job j (enche uma linha vazia); ``limite``: o
    teto de rascunhos por job neste lote (``CopiaConfig.limite_por_job``).

    Cada job usa a cópia quando tem uma, senão o rascunho do modelo. A janela é comum ao lote
    (o forward do alvo pede o mesmo número de linhas para todos): a largura é a maior proposta,
    e com ``passo`` > 1 uma janela que a cópia estendeu além do rascunho do modelo arredonda as
    linhas (pendente + rascunhos) para o próximo múltiplo de ``passo``, dentro do ``limite``,
    para o decode repetir poucas formas de grafo. As linhas mais curtas são completadas
    repetindo o último token delas (o pendente, numa linha vazia); posições completadas são
    rascunhos como os outros, aceitas só se baterem com a amostra.

    Devolve (linhas, largura, usa_copia), com ``linhas[j]`` de comprimento ``largura`` e
    ``usa_copia[j]`` dizendo se a linha do job j veio da cópia."""
    assert len(copias) == len(modelo) == len(pendentes)
    usa = [bool(c) for c in copias]
    linhas = [list(c[:limite]) if u else list(m) for c, m, u in zip(copias, modelo, usa)]
    largura_modelo = max((len(m) for m in modelo), default = 0)
    largura = max((len(x) for x in linhas), default = 0)
    if any(usa) and passo > 1 and largura > largura_modelo:
        arredondada = -(-(largura + 1) // passo) * passo - 1
        largura = max(largura, min(arredondada, max(limite, largura_modelo)))
    saida = []
    for x, p in zip(linhas, pendentes):
        if len(x) < largura:
            x = x + [x[-1] if x else int(p)] * (largura - len(x))
        saida.append(x[:largura])
    return saida, largura, usa
