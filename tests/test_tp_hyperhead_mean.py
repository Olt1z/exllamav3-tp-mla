"""
HyperHead no modo de média (GLM-5.3-Flash) sobrevive a tp_export -> tp_import.

Medido em 05/09/2026 no TR3 do Flash em 4× RTX PRO 6000: a primeira geração em TP caía em
`self.fn.half()` com fn = None, porque a flag `mean` não viajava nos kwargs e o módulo importado
tomava o caminho ponderado. Roda em CPU, sem placa:

    python3 tests/test_tp_hyperhead_mean.py
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from exllamav3.modules.hyperconnections import HyperHead


class ProdutorNulo:
    def send(self, tensor, cache_id = None):
        assert tensor is None, "no modo de média não há tensor para exportar"
        return None


class ConsumidorProibido:
    def recv(self, *a, **k):
        raise AssertionError("no modo de média nada deve ser recebido")


head = HyperHead(config = None, key = "hc_head", hc_mult = 4, rms_norm_eps = 1e-5, hc_eps = 1e-3, mean = True)
exported = head.tp_export(plan = {}, producer = ProdutorNulo())
assert exported["kwargs"]["mean"] is True, exported["kwargs"]

local = {"device": -1, "consumer": ConsumidorProibido()}
imported = HyperHead.tp_import(local, exported, plan = {})
assert imported.mean is True and imported.fn is None

x = torch.randn(2, 3, 4, 16)
assert torch.equal(imported.forward(x, {}), x.mean(dim = 2))
print("ok: HyperHead(mean) exporta e importa sem tensores e colapsa por média")
