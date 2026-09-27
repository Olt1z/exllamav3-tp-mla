"""
Falha do rank da saida no meio de um passo com coletivos encerra o processo (`_tp_abortar`).

O pseudo-worker roda no processo principal e recebe o comando por ultimo: quando ele levanta, os outros
ranks ja estao nos coletivos do passo e ficam la. Seguir para o proximo pedido travava o grupo ate o
watchdog do NCCL (10/09/2026, 50449460: OOM no `cp_combinar`, SeqNum 16847 com tamanhos diferentes).

Sem GPU: o despacho e trocado por stubs, e `os._exit` por uma excecao que o teste captura.
"""
from types import SimpleNamespace

import pytest

from exllamav3.model import model_tp


class Saiu(Exception):
    pass


def _modelo(falha):
    enviados = []
    m = object.__new__(model_tp.Model_TPMixin) if hasattr(model_tp, "Model_TPMixin") else SimpleNamespace()
    m.active_devices = [0, 1, 2, 3]
    m.tp_output_device = 3
    m.mp_parent_conn = {d: SimpleNamespace(send_bytes = lambda b, d = d: enviados.append(d)) for d in range(3)}

    def despacho(device, fn, args):
        if device == -1:
            return
        if falha:
            raise RuntimeError("CUDA out of memory (simulado)")
    m.tp_worker_dispatch = despacho
    m.tp_worker_result = lambda device: object()
    m.prepare_inputs_for_tp = lambda x, params: (x, {})
    m.restore_tp_params = lambda params, reserve: None
    return m, enviados


@pytest.fixture
def saida(monkeypatch):
    codigos = []
    def _exit(c):
        codigos.append(c)
        raise Saiu()
    monkeypatch.setattr(model_tp._os, "_exit", _exit)
    return codigos


@pytest.mark.parametrize("metodo", ["forward_tp", "prefill_tp"])
def test_falha_do_rank_da_saida_encerra(metodo, saida, monkeypatch):
    monkeypatch.setattr(model_tp, "_ABORTAR_NA_FALHA", True)
    m, enviados = _modelo(falha = True)
    with pytest.raises(Saiu):
        getattr(model_tp.Model_TPMixin, metodo)(m, "x", {}, 0, [])
    assert saida == [model_tp.TP_DESSINCRONIZADO]
    # Os outros ranks ja tinham recebido o comando: e isso que os deixa presos nos coletivos
    assert enviados == [0, 1, 2]


def test_interruptor_devolve_o_comportamento_antigo(saida, monkeypatch):
    monkeypatch.setattr(model_tp, "_ABORTAR_NA_FALHA", False)
    m, _ = _modelo(falha = True)
    with pytest.raises(RuntimeError, match = "out of memory"):
        model_tp.Model_TPMixin.forward_tp(m, "x", {}, 0, [])
    assert saida == []


def test_warmup_nao_encerra(saida, monkeypatch):
    monkeypatch.setattr(model_tp, "_ABORTAR_NA_FALHA", True)
    m, _ = _modelo(falha = True)
    with pytest.raises(RuntimeError):
        model_tp.Model_TPMixin.forward_tp(m, "x", {"tp_warmup": True}, 0, [])
    assert saida == []


def test_sem_falha_nada_muda(saida, monkeypatch):
    monkeypatch.setattr(model_tp, "_ABORTAR_NA_FALHA", True)
    m, enviados = _modelo(falha = False)
    model_tp.Model_TPMixin.forward_tp(m, "x", {}, 0, [])
    assert saida == [] and enviados == [0, 1, 2]
