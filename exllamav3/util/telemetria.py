"""
Marcação do passo de decode: anel do spdlog (C++) e NVTX, do mesmo ponto.

Existe porque o hub sabe que o serviço é limitado por lançamento de kernel —
medido em 11/09/2026 numa H200: sempre que a GPU trabalhava, a CPU do motor
estava em 100,0% de um núcleo e a GPU em 48,9% — e não sabe ONDE, dentro do
passo, o tempo vai.

Duas saídas do mesmo ponto de marcação, porque respondem perguntas diferentes:

  - o anel (`ext.tel_*`) guarda os eventos em memória e só despeja quando o
    passo estoura o limiar. Serve para o evento RARO: 5 passos lentos em 101.
  - o NVTX marca a região na linha do tempo da CUDA, e é o que o `nsys` e o
    Perfetto desenham. Serve para ver o passo INTEIRO, com os kernels.

Tudo desligado por padrão: sem `EXL3_TEL=1` cada função aqui é um `return`.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import torch

from ..ext import exllamav3_ext as ext

# Lido uma vez. O caminho quente não pode consultar o ambiente por passo.
_LIGADA = os.environ.get("EXL3_TEL", "0") != "0"

# O NVTX custa uma chamada por região mesmo sem profiler anexado, então segue a
# mesma chave — quem quer o traço liga `EXL3_TEL=1` e roda sob `nsys`.
_NVTX = _LIGADA and os.environ.get("EXL3_TEL_NVTX", "1") != "0"


def ligada() -> bool:
    return _LIGADA


def tem_spdlog() -> bool:
    """A extensão foi compilada com spdlog? Sem ele o anel não existe."""
    try:
        return bool(ext.tel_tem_spdlog())
    except AttributeError:
        # Build antiga, de antes da telemetria: não é erro, é ausência.
        return False


@contextmanager
def passo(nome: str = "decode"):
    """
    Um passo de decode inteiro.

    Fecha mesmo com exceção no meio: um passo que estourou por erro é
    justamente o que se quer ver no anel, e deixá-lo aberto faria o próximo
    passo medir a soma dos dois.
    """
    if not _LIGADA:
        yield
        return
    if _NVTX:
        torch.cuda.nvtx.range_push(nome)
    ext.tel_passo_inicio()
    try:
        yield
    finally:
        ext.tel_passo_fim()
        if _NVTX:
            torch.cuda.nvtx.range_pop()


def evento(nome: str) -> None:
    """Um marco dentro do passo. `nome` aparece no despejo com o delta."""
    if not _LIGADA:
        return
    ext.tel_evento(nome)
    if _NVTX:
        torch.cuda.nvtx.mark(nome)


def despejar() -> None:
    """Despeja o anel agora, sem esperar limiar. Para a captura sob demanda."""
    if _LIGADA:
        ext.tel_despejar()
