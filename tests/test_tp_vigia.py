"""
Robustez do TP (model_tp_vigia): o rank filho que falha num comando com coletivos encerra a si mesmo, a
thread que vigia os filhos encerra o processo principal na hora, e EXL3_TP_VIGIA_S despeja as pilhas de
todas as threads quando um passo trava.

A logica pura (sem torch) e carregada pelo caminho do arquivo, e roda sem GPU. Os testes do laco do
worker (model_tp_fn) e do processo principal (model_tp) precisam do pacote importavel, com torch.

    python3 -m pytest tests/test_tp_vigia.py -v
"""
import importlib.util
import multiprocessing
import os
import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace

import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQUIVO = os.path.join(RAIZ, "exllamav3", "model", "model_tp_vigia.py")


def _carregar():
    spec = importlib.util.spec_from_file_location("model_tp_vigia_isolado", ARQUIVO)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


vigia = _carregar()


def _fn(nome):
    def f(*args):
        raise RuntimeError("CUDA out of memory (simulado)")
    f.__name__ = nome
    return f


# -- quais comandos tem coletivo ---------------------------------------------------------------------------

def test_forward_tem_coletivo_e_warmup_nao():
    fwd = _fn("mp_model_forward")
    assert vigia.comando_tem_coletivo(fwd, (None, {}, 0, False, None))
    assert vigia.comando_tem_coletivo(fwd, (None, {"tp_warmup": False}, 0, True, None))
    assert not vigia.comando_tem_coletivo(fwd, (None, {"tp_warmup": True}, 0, False, None))


def test_lm_head_so_com_gather():
    for nome in ("mp_model_forward_lm_head_argmax", "mp_model_forward_lm_head_topk"):
        f = _fn(nome)
        assert vigia.comando_tem_coletivo(f, (None, {}, 0, [0, 1], [1, 1]))
        assert not vigia.comando_tem_coletivo(f, (None, {}, 0, None, None))


def test_reduce_na_cpu_tem_coletivo_e_carga_nao():
    assert vigia.comando_tem_coletivo(_fn("mp_cpu_reduce"), ())
    for nome in ("mp_model_append", "mp_set_plan", "mp_cache_page_copy", "mp_rotate_cache_pages",
                 "mp_cpu_cache_store", "touch_device_measure_vram"):
        assert not vigia.comando_tem_coletivo(_fn(nome), (None, {}, 0, [0], [1]))


# -- o filho que falha encerra a si mesmo ------------------------------------------------------------------

def test_filho_abortar_sai_com_71_e_diz_por_que(capsys):
    codigos = []
    try:
        raise RuntimeError("CUDA out of memory (simulado)")
    except RuntimeError as e:
        vigia.filho_abortar(2, _fn("mp_model_forward"), e, sair = codigos.append)
    assert codigos == [vigia.TP_FILHO_FALHOU] == [71]
    err = capsys.readouterr().err
    assert "device 2" in err and "mp_model_forward" in err and "out of memory" in err
    assert "Traceback" in err


def test_descrever_saida():
    assert "71" in vigia.descrever_saida(71) and "coletivos" in vigia.descrever_saida(71)
    assert "SIGKILL" in vigia.descrever_saida(-9) and "OOM" in vigia.descrever_saida(-9)
    assert "SIGSEGV" in vigia.descrever_saida(-11)
    assert vigia.descrever_saida(None) == "codigo desconhecido"


# -- a vigia dos filhos ------------------------------------------------------------------------------------

def _filho_falso(sentinela, codigo):
    return SimpleNamespace(sentinel = sentinela, exitcode = codigo, join = lambda t = None: None)


def test_vigia_dos_filhos_avisa_qual_morreu():
    mortes = []
    filhos = {0: _filho_falso(10, None), 1: _filho_falso(11, 71), -1: _filho_falso(12, None)}
    v = vigia.VigiaDosFilhos(filhos, lambda d, c: mortes.append((d, c)), esperar = lambda s: [11])
    v.iniciar()
    v.thread.join(5)
    assert mortes == [(1, 71)]


def test_vigia_dos_filhos_parada_nao_avisa():
    mortes = []
    v = vigia.VigiaDosFilhos({0: _filho_falso(10, 0)}, lambda d, c: mortes.append((d, c)),
                             esperar = lambda s: (time.sleep(0.2), [10])[1])
    v.iniciar()
    v.parar()                                   # o desligamento: os filhos saem de proposito
    v.thread.join(5)
    assert mortes == []


def test_vigia_dos_filhos_sem_filhos_nao_cria_thread():
    v = vigia.VigiaDosFilhos({}, lambda d, c: None)
    v.iniciar()
    assert v.thread is None


@pytest.mark.skipif(sys.platform != "linux", reason = "fork")
def test_vigia_dos_filhos_com_processo_de_verdade():
    ctx = multiprocessing.get_context("fork")
    vivo = ctx.Process(target = time.sleep, args = (30,))
    morre = ctx.Process(target = os._exit, args = (vigia.TP_FILHO_FALHOU,))
    vivo.start()
    mortes = []
    try:
        v = vigia.VigiaDosFilhos({0: vivo, 3: morre}, lambda d, c: mortes.append((d, c)))
        morre.start()
        v.iniciar()
        t0 = time.monotonic()
        v.thread.join(10)
        # Na hora, e nao depois do timeout de um coletivo
        assert time.monotonic() - t0 < 5
        assert mortes == [(3, vigia.TP_FILHO_FALHOU)]
    finally:
        vivo.terminate()
        vivo.join()


# -- EXL3_TP_VIGIA_S ---------------------------------------------------------------------------------------

def test_vigia_segundos():
    assert vigia.vigia_segundos("") == 0
    assert vigia.vigia_segundos("0") == 0
    assert vigia.vigia_segundos("30") == 30
    assert vigia.vigia_segundos(" 2.5 ") == 2.5
    for ruim in ("-1", "trinta"):
        with pytest.raises(ValueError, match = "EXL3_TP_VIGIA_S"):
            vigia.vigia_segundos(ruim)


class FhFalso:
    def __init__(self, falha = None):
        self.armou, self.cancelou, self.falha = [], 0, falha

    def dump_traceback_later(self, timeout, repeat, file, exit):
        if self.falha:
            raise self.falha
        self.armou.append((timeout, repeat, exit))

    def cancel_dump_traceback_later(self):
        self.cancelou += 1


def test_vigia_do_passo_desligada_nao_toca_no_faulthandler():
    fh = FhFalso()
    v = vigia.VigiaDoPasso(0, fh = fh)
    with v:
        pass
    assert not v.ligado and fh.armou == [] and fh.cancelou == 0


def test_vigia_do_passo_reentrante_arma_uma_vez_e_sem_repetir():
    fh = FhFalso()
    v = vigia.VigiaDoPasso(30, fh = fh, arquivo = sys.__stderr__)
    with v:                                     # passo do gerador
        with v:                                 # forward_tp dentro dele
            pass
        assert fh.cancelou == 0                 # o prazo segue contando do inicio do passo
    assert fh.armou == [(30, False, False)] and fh.cancelou == 1
    with v:
        pass
    assert len(fh.armou) == 2 and fh.cancelou == 2


def test_vigia_do_passo_desarma_na_excecao():
    fh = FhFalso()
    v = vigia.VigiaDoPasso(30, fh = fh, arquivo = sys.__stderr__)
    with pytest.raises(RuntimeError):
        with v:
            raise RuntimeError("falhou no passo")
    assert fh.cancelou == 1 and v.profundidade == 0


def test_vigia_do_passo_sem_descritor_se_desliga(capsys):
    v = vigia.VigiaDoPasso(30, fh = FhFalso(falha = ValueError("sem fileno")), arquivo = sys.__stderr__)
    with v:
        pass
    assert not v.ligado and v.profundidade == 0
    assert "vigia desligada" in capsys.readouterr().err


def test_vigiar_passo_devolve_o_resultado(monkeypatch):
    fh = FhFalso()
    monkeypatch.setattr(vigia, "VIGIA_DO_PASSO", vigia.VigiaDoPasso(30, fh = fh, arquivo = sys.__stderr__))

    @vigia.vigiar_passo
    def passo(x, y = 1):
        return x + y

    assert passo(1, y = 2) == 3 and passo.__name__ == "passo"
    assert len(fh.armou) == 1 and fh.cancelou == 1


def test_passo_travado_despeja_as_pilhas_uma_vez():
    # De ponta a ponta, com o faulthandler de verdade: um passo de 2 s com vigia de 0,4 s despeja as
    # pilhas uma vez so (repeat = False), com a funcao travada nelas; o passo rapido seguinte, nada
    script = textwrap.dedent(f"""
        import importlib.util, threading, time
        spec = importlib.util.spec_from_file_location("v", {ARQUIVO!r})
        v = importlib.util.module_from_spec(spec); spec.loader.exec_module(v)
        assert v.VIGIA_DO_PASSO.ligado

        def outra_thread():
            time.sleep(5)
        threading.Thread(target = outra_thread, daemon = True).start()

        @v.vigiar_passo
        def passo_travado():
            time.sleep(2)

        @v.vigiar_passo
        def passo_rapido():
            pass

        passo_travado()
        passo_rapido()
        time.sleep(0.8)
    """)
    env = dict(os.environ, EXL3_TP_VIGIA_S = "0.4")
    r = subprocess.run([sys.executable, "-c", script], env = env, capture_output = True, text = True, timeout = 60)
    assert r.returncode == 0, r.stderr
    assert r.stderr.count("Timeout (") == 1, r.stderr
    assert "passo_travado" in r.stderr and "outra_thread" in r.stderr


# -- integracao com o laco do worker e com o processo principal (precisam do pacote, com torch) ------------

class Saiu(Exception):
    pass


def _importar(nome):
    # Sem torch, o pacote levanta RuntimeError no import (e nao ImportError, que o importorskip pegaria)
    try:
        return importlib.import_module(nome)
    except (ImportError, RuntimeError) as e:
        pytest.skip(f"{nome}: {e}")


class ConnFalsa:
    def __init__(self, msgs):
        self.msgs = list(msgs) + ["quit"]
        self.enviados = []

    def poll(self, t = 0):
        return True

    def recv(self):
        return self.msgs.pop(0)

    def send(self, x):
        self.enviados.append(x)


def _rodar_worker(model_tp_fn, msgs, monkeypatch):
    from unittest.mock import patch
    import signal
    abortos = []

    def abortar(device, func, e):
        abortos.append((device, func.__name__, str(e)))
        raise Saiu()

    monkeypatch.setattr(model_tp_fn, "filho_abortar", abortar)
    monkeypatch.setattr(model_tp_fn, "ABORTAR_NA_FALHA", True)
    conn = ConnFalsa(msgs)
    fechar = SimpleNamespace(close = lambda: None)
    with (
        patch.object(model_tp_fn, "set_t0"),
        patch.object(model_tp_fn, "log_tp"),
        patch.object(model_tp_fn, "install_parent_death_signal", return_value = False),
        patch.object(model_tp_fn, "init_pg", return_value = {"backend": fechar}),
        patch.object(model_tp_fn, "SMConsumer", return_value = fechar),
        patch.object(model_tp_fn.torch.cuda, "synchronize"),
        patch.object(signal, "signal"),
    ):
        try:
            model_tp_fn.mp_model_worker(conn, 1, [1, 0], 0, {"type": "native"}, {}, 0.0)
        except Saiu:
            pass
    return conn, abortos


def test_worker_filho_falha_no_forward_encerra(monkeypatch):
    model_tp_fn = _importar("exllamav3.model.model_tp_fn")
    conn, abortos = _rodar_worker(model_tp_fn, [(_fn("mp_model_forward"), (None, {}, 0, False, None))], monkeypatch)
    assert abortos == [(1, "mp_model_forward", "CUDA out of memory (simulado)")]
    assert conn.enviados == []                  # nao devolve pelo pipe: o principal esta preso no coletivo


def test_worker_falha_sem_coletivo_devolve_a_excecao(monkeypatch):
    model_tp_fn = _importar("exllamav3.model.model_tp_fn")
    msgs = [
        (_fn("mp_model_forward"), (None, {"tp_warmup": True}, 0, False, None)),    # warmup: sem coletivo
        (_fn("mp_model_append"), ({},)),                                           # carga
    ]
    conn, abortos = _rodar_worker(model_tp_fn, msgs, monkeypatch)
    assert abortos == []
    assert [type(e) for e in conn.enviados] == [RuntimeError, RuntimeError]


def test_principal_encerra_quando_um_filho_morre(monkeypatch, capsys):
    model_tp = _importar("exllamav3.model.model_tp")
    codigos = []

    def _exit(c):
        codigos.append(c)
        raise Saiu()

    monkeypatch.setattr(model_tp._os, "_exit", _exit)
    with pytest.raises(Saiu):
        model_tp._tp_filho_morreu(2, 71)
    assert codigos == [model_tp.TP_DESSINCRONIZADO] == [70]
    err = capsys.readouterr().err
    assert "device 2" in err and "71" in err
    with pytest.raises(Saiu):
        model_tp._tp_filho_morreu(-1, -9)
    assert "ajudante de CPU" in capsys.readouterr().err


def test_destroy_para_a_vigia_antes_do_quit(monkeypatch):
    model_tp = _importar("exllamav3.model.model_tp")
    ordem = []
    m = object.__new__(model_tp.Model_TPMixin)
    m.tp_vigia_dos_filhos = SimpleNamespace(parar = lambda: ordem.append("parar"))
    m.tp_pending_acks = []
    m.tp_output_device = 0

    def quit_():
        ordem.append("quit")

    m.mp_parent_conn = [SimpleNamespace(quit = quit_)]
    m.mp_child_conn = [None]
    m.mp_children = [None]
    m.tp_producer = SimpleNamespace(close = lambda: None)
    model_tp.Model_TPMixin.destroy_tp_context(m)
    assert ordem == ["parar", "quit"] and m.tp_vigia_dos_filhos is None
