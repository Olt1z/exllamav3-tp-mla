"""
Traço de roteamento: a sequência (camada, expert) que faz experimento de colocação sair de
graça.

Roteamento não depende de colocação — o mesmo prompt escolhe os mesmos experts esteja o
expert na VRAM ou na arena do worker. Então UMA execução grava a sequência e ela responde,
offline, a família inteira de perguntas sobre onde os experts deveriam morar. A ideia e o
formato do arquivo vêm do `FareedKhan-dev/kimi-k3-in-c`.

O que este teste protege é o que torna o traço UTILIZÁVEL, e cada asserção corresponde a uma
forma de o arquivo mentir em silêncio:

  - ordem global preservada entre camadas (um cache se replica em ordem, ou não se replica);
  - ids do CHECKPOINT mesmo quando o roteador fala em espaço permutado (senão um traço com
    perfil e um sem não são comparáveis, que é justamente o A/B que isto serve);
  - despejo parcial ao encher, porque máquina alugada morre sem desligar direito;
  - o formato que `tools/sim_cache.py` lê: int32 plano, pares em ordem de pedido.

Roda na CPU, sem modelo e sem placa.
"""
import json
import os
import sys
import tempfile
import types

import torch

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

# A extensão C++/CUDA é compilada por JIT no import, e nada aqui a usa: `_TracoDeRoteamento`
# só mexe em tensores. Sem este atalho o teste exigiria `ninja`, um compilador e, na prática,
# uma placa — o oposto do que ele existe para provar.
_pkg = types.ModuleType("exllamav3")
_pkg.__path__ = [os.path.join(RAIZ, "exllamav3")]
sys.modules.setdefault("exllamav3", _pkg)
_ext = types.ModuleType("exllamav3.ext")
_ext.exllamav3_ext = types.SimpleNamespace()
sys.modules.setdefault("exllamav3.ext", _ext)
_mods = types.ModuleType("exllamav3.modules")
_mods.__path__ = [os.path.join(RAIZ, "exllamav3", "modules")]
sys.modules.setdefault("exllamav3.modules", _mods)

from exllamav3.modules.block_sparse_mlp_cpu import _TracoDeRoteamento, traco_de_roteamento


class _CamadaFalsa:
    """O mínimo que `anotar` lê de um módulo: índice, nome e a permutação, se houver."""

    def __init__(self, idx, key, perm = None):
        self.cpu_layer_idx = idx
        self.key = key
        self._split_perm = perm


def _pares(caminho):
    import numpy as np
    return np.fromfile(caminho, dtype = np.int32).reshape(-1, 2)


def test_ordem_global_entre_camadas():
    with tempfile.TemporaryDirectory() as d:
        alvo = os.path.join(d, "traco.bin")
        t = _TracoDeRoteamento(alvo, 64, torch.device("cpu"))
        a, b = _CamadaFalsa(0, "model.layers.3.mlp"), _CamadaFalsa(1, "model.layers.4.mlp")
        # Dois "tokens": cada um visita as camadas na mesma ordem.
        t.anotar(a, torch.tensor([7, 9]))
        t.anotar(b, torch.tensor([1]))
        t.anotar(a, torch.tensor([7, 3]))
        t.anotar(b, torch.tensor([2]))
        t.despejar()

        p = _pares(alvo)
        assert p.tolist() == [[0, 7], [0, 9], [1, 1], [0, 7], [0, 3], [1, 2]], p.tolist()

        lado = json.load(open(alvo + ".json"))
        assert lado["pares"] == 6
        assert lado["camadas"]["1"] == "model.layers.4.mlp"


def test_ids_voltam_para_o_espaco_do_checkpoint():
    """Sob perfil estático o roteador fala em ordem quente-para-fria; o traço não pode.

    `perm[r]` é o expert do checkpoint que ocupa a posição r. Sem esta tradução, um traço
    colhido COM perfil e um colhido SEM nomeiam experts diferentes com o mesmo número, e a
    comparação entre os dois — o A/B inteiro — mede a permutação em vez da colocação.
    """
    with tempfile.TemporaryDirectory() as d:
        alvo = os.path.join(d, "traco.bin")
        t = _TracoDeRoteamento(alvo, 64, torch.device("cpu"))
        # A camada 0 permutada, a camada 1 não: as duas convivem no mesmo arquivo.
        permutada = _CamadaFalsa(0, "L0", perm = [40, 41, 42, 43])
        crua = _CamadaFalsa(1, "L1")
        t.anotar(permutada, torch.tensor([0, 3]))
        t.anotar(crua, torch.tensor([0, 3]))
        t.despejar()

        p = _pares(alvo).tolist()
        assert p == [[0, 40], [0, 43], [1, 0], [1, 3]], p
        assert json.load(open(alvo + ".json"))["espaco"] == "ids do checkpoint"


def test_despeja_sozinho_ao_encher():
    """Máquina alugada costuma ser destruída sem desligar direito.

    Um traço que só existe no `unload` é um traço que nunca existe — o mesmo motivo pelo qual
    `despejar_stats_de_roteamento` grava a cada varredura em vez de na saída.
    """
    with tempfile.TemporaryDirectory() as d:
        alvo = os.path.join(d, "traco.bin")
        t = _TracoDeRoteamento(alvo, 4, torch.device("cpu"))
        for _ in range(3):
            t.anotar(_CamadaFalsa(0, "L0"), torch.tensor([1, 2]))
        # Três anotações de 2 pares em capacidade 4: a terceira só cabe depois de despejar.
        assert os.path.exists(alvo), "encher o buffer tinha de ter gravado o que já havia"
        t.despejar()
        assert len(_pares(alvo)) == 6


def test_um_traco_por_processo():
    """O buffer é do processo, não da camada: é o que dá a ordem global de graça."""
    with tempfile.TemporaryDirectory() as d:
        class _Cfg:
            pass

        class _Mod:
            def __init__(self, cfg):
                self.config = cfg
                self.device = "cpu"

        cfg = _Cfg()
        cfg.infer_params = _Cfg()
        os.environ["EXL3_MOE_CPU_TRACE_OUT"] = os.path.join(d, "t.bin")
        os.environ["EXL3_MOE_CPU_TRACE_MAX"] = "16"
        try:
            primeiro = traco_de_roteamento(_Mod(cfg))
            segundo = traco_de_roteamento(_Mod(cfg))
            assert primeiro is not None and primeiro is segundo
        finally:
            del os.environ["EXL3_MOE_CPU_TRACE_OUT"]
            del os.environ["EXL3_MOE_CPU_TRACE_MAX"]

        # Sem a variável, nada é alocado e o caminho quente paga só um getenv.
        cfg2 = _Cfg()
        cfg2.infer_params = _Cfg()
        assert traco_de_roteamento(_Mod(cfg2)) is None


if __name__ == "__main__":
    for nome, fn in sorted(globals().items()):
        if nome.startswith("test_"):
            fn()
            print(f"ok  {nome}")
