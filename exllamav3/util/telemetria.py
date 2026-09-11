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
def passo(nome: str = "decode", medir: bool = True):
    """
    Um passo de decode inteiro.

    `medir=False` passa reto, e existe por um motivo específico: `Model.forward`
    é o passo de decode, mas também é o chunk de PREFILL e o forward do modelo
    de RASCUNHO, e os três passam pela mesma função. Medir os três juntos
    estraga as duas coisas que esta telemetria produz — a mediana da duração
    vira a de uma mistura de três populações, e o limiar automático (2x a média
    móvel) passa a comparar decode com prefill, que é dezenas de vezes mais
    caro: todo prefill despejaria o anel, e a média inflada calaria justamente
    as anomalias de decode que se está caçando.

    Os eventos de dentro se cuidam sozinhos: o anel só grava com passo aberto.

    Fecha mesmo com exceção no meio: um passo que estourou por erro é
    justamente o que se quer ver no anel, e deixá-lo aberto faria o próximo
    passo medir a soma dos dois.
    """
    if not _LIGADA or not medir:
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


def regiao_inicio(nome: str) -> None:
    """
    Abre uma região NVTX. Fecha com `regiao_fim`.

    Só NVTX por padrão: isto marca CADA MÓDULO do passo — umas dezenas por
    passo — e alimentar o anel com elas faria cada despejo sair com centenas de
    linhas, que é o tamanho que já provou expulsar as linhas `Metrics` da janela
    de log do hub. `EXL3_TEL_MODULOS=1` põe o mesmo marco no anel mesmo assim
    (ver `marcando_modulos`), e o C++ compensa baixando o contexto do despejo
    para 0. Quem lê a região NVTX é o `nsys`; quem lê o marco é o log.

    Sem profiler anexado o custo é uma chamada que não faz nada — por isso o
    `if` fica no chamador, e não aqui dentro.
    """
    if _NVTX:
        torch.cuda.nvtx.range_push(nome)


def regiao_fim() -> None:
    if _NVTX:
        torch.cuda.nvtx.range_pop()


def marcando_regioes() -> bool:
    """O chamador lê isto UMA vez e guarda: o laço de módulos é o caminho mais
    quente do motor, e nele até a chamada de função que devolve `False` custa."""
    return _NVTX


def marcando_modulos() -> bool:
    """
    `EXL3_TEL_MODULOS=1`: um marco por módulo no ANEL, e não só no NVTX.

    É o que faz o despejo do passo lento dizer `47:Attention` em vez de
    `forward_ls`, sem profiler nenhum. A decisão e o contexto padrão (0) moram
    no C++, que é quem lê o ambiente; aqui só se pergunta uma vez.
    """
    if not _LIGADA:
        return False
    try:
        return bool(ext.tel_marca_modulos())
    except AttributeError:
        # Build antiga: a chave não existe, e não é erro.
        return False


def despejar() -> None:
    """Despeja o anel agora, sem esperar limiar. Para a captura sob demanda."""
    if _LIGADA:
        ext.tel_despejar()
