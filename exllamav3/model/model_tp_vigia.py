"""
Robustez do tensor parallel: falhar rapido e alto em vez de ficar preso num coletivo, e deixar as pilhas
no log quando um passo trava.

Sem torch de proposito: o modulo e importado pelo processo principal e pelos filhos, e os testes o carregam
pelo caminho do arquivo numa maquina sem GPU.

Tres pecas:

- `comando_tem_coletivo`: se o comando despachado a um rank entra em coletivos (o forward fora do warmup, o
  gather do lm_head, o reduce na CPU). Um rank que falha num desses deixa os outros presos no coletivo.

- `filho_abortar` + `VigiaDosFilhos`: o rank FILHO que falha num comando com coletivos imprime o motivo e
  encerra a si mesmo (codigo 71); uma thread do processo principal espera nos sentinelas dos processos
  filhos e, quando um morre fora do desligamento, encerra o processo principal (codigo 70). Antes, o filho
  so devolvia a excecao pelo pipe, que o principal so le DEPOIS do coletivo -- e o coletivo so solta no
  timeout (90 s no nativo, ou o watchdog do NCCL, 600 s+ em 10/09/2026).

  Por que assim, e nao uma thread lendo os pipes dos filhos: o pipe e do despacho (a thread principal le
  dele em `tp_worker_result`/`tp_drain_acks`), dois leitores no mesmo pipe embaralham as mensagens, e os
  filhos mandam acks `None` legitimos enquanto o principal ainda esta no forward -- "chegou algo no pipe"
  nao quer dizer "falhou". A morte do processo e inequivoca, nao disputa o pipe, e de quebra pega o filho
  morto por segfault ou pelo OOM killer, que nem chega a mandar excecao. E por que nao o filho matar o pai
  com um sinal: no conteiner o processo principal costuma ser o PID 1, que ignora SIGKILL vindo de dentro.

- `VigiaDoPasso` (EXL3_TP_VIGIA_S, padrao 0 = desligado): se um passo do gerador (ou um forward TP fora
  dele) passa de N s, despeja as pilhas de todas as threads no stderr, uma vez por travamento. Usa o
  `faulthandler.dump_traceback_later`, cuja thread e de C: despeja mesmo com a thread principal presa
  segurando o GIL. O cabecalho no log e o do faulthandler: "Timeout (h:mm:ss)!".
"""
from __future__ import annotations

import faulthandler
import functools
import os
import sys
import threading
import traceback


# Decisao de operacao, lida uma vez no import (o filho herda o ambiente). `EXL3_TP_ABORTAR_NA_FALHA=0`
# volta ao comportamento antigo: nem o rank da saida, nem os filhos encerram o processo.
ABORTAR_NA_FALHA = os.environ.get("EXL3_TP_ABORTAR_NA_FALHA", "1") != "0"

# Processo principal encerrado porque o grupo ficou dessincronizado (rank da saida falhou, ou um filho morreu)
TP_DESSINCRONIZADO = 70
# Rank filho que encerrou a si mesmo depois de falhar num comando com coletivos
TP_FILHO_FALHOU = 71

# Comandos que entram em coletivos. Pelo nome, para nao importar as funcoes (e o torch) daqui.
_COMANDOS_COM_COLETIVO = {
    "mp_model_forward",                     # fwd_barrier, all-reduce, gather; menos no warmup (TPBackendNull)
    "mp_model_forward_lm_head_argmax",      # gather dos maximos parciais, quando ha mais de um rank
    "mp_model_forward_lm_head_topk",        # idem
    "mp_cpu_reduce",                        # o ajudante de CPU do backend nativo: sem ele, o reduce nao anda
}


def comando_tem_coletivo(func, args: tuple) -> bool:
    """
    Se o comando `func(local_context, *args)` despachado a um rank entra em coletivos.

    mp_model_forward: args = (shared_input, params, last_kv_module_idx, prefill, single_idx); o warmup
    (params["tp_warmup"]) roda sem coletivo. lm_head argmax/topk: args[3] e gather_devices; None e o caso
    de rank unico, sem gather.
    """
    nome = getattr(func, "__name__", None)
    if nome not in _COMANDOS_COM_COLETIVO:
        return False
    if nome == "mp_model_forward":
        params = args[1] if len(args) > 1 else None
        return not (isinstance(params, dict) and params.get("tp_warmup"))
    if nome.startswith("mp_model_forward_lm_head_"):
        return len(args) > 3 and args[3] is not None
    return True


def _escrever(texto: str):
    try:
        sys.stdout.flush()
    except Exception:
        pass
    try:
        sys.stderr.write(texto)
        sys.stderr.flush()
    except Exception:
        pass


def filho_abortar(device: int, func, e: BaseException, sair = os._exit):
    """
    O rank filho do `device` falhou num comando com coletivos: imprime o motivo e encerra a si mesmo.

    Devolver a excecao pelo pipe nao adianta: os outros ranks (o principal inclusive) estao dentro dos
    coletivos desse passo e so leem o pipe depois deles. Morto, o filho e visto na hora pela
    `VigiaDosFilhos` do processo principal, que encerra tudo; e o PDEATHSIG leva os outros filhos.
    """
    nome = getattr(func, "__name__", repr(func))
    _escrever(
        "".join(traceback.format_exception(type(e), e, e.__traceback__))
        + f"[exl3] rank filho (device {device}): {nome} falhou no meio de um passo com coletivos "
        f"({type(e).__name__}: {e}). Os outros ranks estao presos nos coletivos desse passo; encerrando "
        f"este processo (codigo {TP_FILHO_FALHOU}) para o principal encerrar o grupo na hora, em vez de "
        f"esperar o timeout do coletivo (EXL3_TP_ABORTAR_NA_FALHA=0 desliga).\n"
    )
    sair(TP_FILHO_FALHOU)


def descrever_saida(codigo: int | None) -> str:
    if codigo is None:
        return "codigo desconhecido"
    if codigo == TP_FILHO_FALHOU:
        return f"codigo {codigo}: falhou num comando com coletivos, o motivo esta logo acima no log"
    if codigo < 0:
        import signal
        try:
            nome = signal.Signals(-codigo).name
        except ValueError:
            nome = f"sinal {-codigo}"
        return f"morto por {nome}" + (" (OOM killer?)" if -codigo == signal.SIGKILL else "")
    return f"codigo {codigo}"


class VigiaDosFilhos:
    """
    Thread do processo principal que espera nos sentinelas dos processos filhos e chama `ao_morrer(device,
    codigo)` quando um morre fora do desligamento. `parar()` antes de mandar "quit" aos filhos.

    A espera e um select() nos sentinelas: nao custa nada por passo e nao toca nos pipes do despacho. Ela
    so precisa do GIL para reagir; a thread principal presa num coletivo o solta (as esperas do torch e do
    NCCL liberam o GIL). Se nem assim, cai-se no timeout do coletivo, como antes.
    """

    def __init__(self, filhos: dict, ao_morrer, esperar = None):
        if esperar is None:
            from multiprocessing.connection import wait as esperar
        self.filhos = dict(filhos)            # device -> multiprocessing.Process
        self.ao_morrer = ao_morrer
        self.esperar = esperar
        self.parando = threading.Event()
        self.thread = None

    def iniciar(self):
        if not self.filhos:
            return
        self.thread = threading.Thread(target = self._rodar, name = "exl3-tp-vigia-dos-filhos", daemon = True)
        self.thread.start()

    def parar(self):
        self.parando.set()

    def _rodar(self):
        por_sentinela = {p.sentinel: d for d, p in self.filhos.items()}
        try:
            prontos = self.esperar(list(por_sentinela))
        except Exception:
            return                            # sentinelas fechados: o contexto TP esta sendo desmontado
        if self.parando.is_set() or not prontos:
            return
        device = por_sentinela[prontos[0]]
        filho = self.filhos[device]
        try:
            filho.join(1)                     # o sentinela abre na saida; o codigo vem do waitpid
            codigo = filho.exitcode
        except Exception:
            codigo = None
        if self.parando.is_set():
            return
        self.ao_morrer(device, codigo)


def vigia_segundos(valor: str | None = None) -> float:
    """EXL3_TP_VIGIA_S: segundos que um passo pode levar antes de as pilhas irem para o log (0: desligado)."""
    valor = os.environ.get("EXL3_TP_VIGIA_S", "") if valor is None else valor
    try:
        s = 0.0 if valor.strip() == "" else float(valor)
    except ValueError:
        s = -1.0
    if s < 0:
        raise ValueError(f"EXL3_TP_VIGIA_S: segundos (0 desliga), nao {valor!r}")
    return s


class VigiaDoPasso:
    """
    Arma o despejo de pilhas do faulthandler no inicio de um passo e o desarma no fim. Reentrante: so o
    nivel mais externo arma e desarma (o faulthandler tem um temporizador so, e o forward TP roda dentro do
    passo do gerador), entao o prazo conta do inicio do passo inteiro.

    `repeat = False`: um despejo por travamento; o passo seguinte arma de novo. Rearmar custa criar e
    juntar a thread do faulthandler (dezenas de us) por passo, e so quando ligado.
    """

    def __init__(self, segundos: float, fh = faulthandler, arquivo = None):
        self.segundos = segundos
        self.fh = fh
        self.arquivo = arquivo
        self.profundidade = 0
        self.trava = threading.Lock()

    @property
    def ligado(self) -> bool:
        return self.segundos > 0

    def armar(self):
        if self.segundos <= 0:
            return
        with self.trava:
            self.profundidade += 1
            if self.profundidade > 1:
                return
            try:
                self.fh.dump_traceback_later(
                    self.segundos,
                    repeat = False,
                    file = self.arquivo if self.arquivo is not None else sys.__stderr__,
                    exit = False,
                )
            except (ValueError, OSError, AttributeError) as e:
                # Sem descritor para escrever (stderr trocado por algo sem fileno): sem vigia
                self.segundos = 0
                self.profundidade = 0
                _escrever(f"[exl3] EXL3_TP_VIGIA_S: vigia desligada, o faulthandler nao pode escrever ({e})\n")

    def desarmar(self):
        if self.segundos <= 0:
            return
        with self.trava:
            if self.profundidade == 0:
                return
            self.profundidade -= 1
            if self.profundidade == 0:
                self.fh.cancel_dump_traceback_later()

    def __enter__(self):
        self.armar()
        return self

    def __exit__(self, *exc):
        self.desarmar()
        return False


VIGIA_DO_PASSO = VigiaDoPasso(vigia_segundos())


def vigiar_passo(fn):
    """Decorador: `fn` roda sob a VIGIA_DO_PASSO. Desligada (o padrao), e uma comparacao a mais por chamada."""
    @functools.wraps(fn)
    def vigiado(*args, **kwargs):
        if not VIGIA_DO_PASSO.ligado:
            return fn(*args, **kwargs)
        with VIGIA_DO_PASSO:
            return fn(*args, **kwargs)
    return vigiado
