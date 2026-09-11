"""
Marcos por módulo no anel (`EXL3_TEL_MODULOS=1`), no laço genérico.

Roda na máquina, com a extensão compilada:

    EXL3_TEL=1 EXL3_TEL_MODULOS=1 EXL3_TEL_NVTX=0 python -m pytest tests/test_telemetria_modulos.py

Sem as duas primeiras o teste pula, porque a decisão é lida do ambiente uma
vez, na importação. O `EXL3_TEL_NVTX=0` é para o teste não depender de CUDA:
com ele ligado o laço chama `torch.cuda.nvtx.range_push`, que levanta sem
placa — e o teste ainda protege esse caminho com monkeypatch.

O que ele tranca: o laço de `forward_ls` grava UM marco por módulo, com o nome
de `_nome_da_regiao` (`12:Attention`), DEPOIS do forward — o delta até o marco é
o custo daquele módulo, e um marco antes do forward atribuiria o custo ao
vizinho. E com a chave desligada o laço não chama nada.
"""

import os
import types

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("EXL3_TEL", "0") == "0" or os.environ.get("EXL3_TEL_MODULOS", "0") == "0",
    reason = "precisa de EXL3_TEL=1 EXL3_TEL_MODULOS=1 no ambiente",
)


class ModuloFalso:
    caps = {}

    def __init__(self, gravados):
        self.gravados = gravados

    def prepare_for_device(self, x, params):
        return x

    def forward(self, x, params):
        # O marco deste módulo ainda NÃO existe quando o forward roda.
        self.gravados.append(("forward", type(self).__name__))
        return x


class Attention(ModuloFalso):
    pass


class MLP(ModuloFalso):
    pass


def test_um_marco_por_modulo_depois_do_forward(monkeypatch):
    from exllamav3.model import model_ls
    from exllamav3.util import telemetria as tel

    assert model_ls._MARCAR_MODULOS, "a chave está no ambiente mas o laço não a leu"

    gravados = []
    monkeypatch.setattr(tel, "evento", lambda nome: gravados.append(("marco", nome)))
    monkeypatch.setattr(tel, "regiao_inicio", lambda nome: None)
    monkeypatch.setattr(tel, "regiao_fim", lambda: None)

    modelo = types.SimpleNamespace(
        config = types.SimpleNamespace(),
        fwd_modules = [(Attention(gravados), 0, 3), (MLP(gravados), 0, 4), (Attention(gravados), 1, 3)],
    )
    x = torch.zeros(1, 1, 8)
    model_ls.Model_LSMixin.forward_ls(modelo, x, {})

    assert gravados == [
        ("forward", "Attention"), ("marco", "3:Attention"),
        ("forward", "MLP"), ("marco", "4:MLP"),
        ("forward", "Attention"), ("marco", "3#1:Attention"),
    ]
