"""
Transbordo da arena de inferencia para arquivo em /dev/shm (`SMProducer._transbordar`).

Em 27/09/2026 a `y3psf` (GLM-5.3-Flash, TP4, 360k de contexto) morreu no primeiro pedido com imagem: a
tabela de frequencias do mrope passa dos 64 MiB da arena, caiu no `share_memory_()` do torch, e os tres
ranks filhos nao acharam o segmento ao desempacotar o comando ("unable to open shared memory object").

Sem GPU: o consumidor e montado a mao, sem registrar a arena no CUDA.
"""
import os

import torch

from exllamav3.model.model_tp_shared import SMConsumer, SMProducer


def _consumidor(produtor):
    c = object.__new__(SMConsumer)
    c.pin_memory = False
    c.device = None
    c.producer = produtor
    c.cached_cpu_tensors = {}
    c.cache_size = 0
    c.arena = torch.as_tensor(produtor.buf)
    return c


def test_o_que_nao_cabe_vai_para_arquivo_e_chega_inteiro():
    p = SMProducer(buffer_size = 64 * 1024, transbordo_em_arquivo = True)
    try:
        grande = torch.randn(1, 3000, 32)          # 384 KB: nao cabe em 64 KB
        d = p.send(grande)
        assert d["method"] == "arquivo"
        assert os.path.exists(d["path"])
        recebido = _consumidor(p).recv(d)
        assert torch.equal(recebido, grande)

        # O clear() so roda depois de os ranks confirmarem o passo; o arquivo some, o recebido fica
        p.clear()
        assert not os.path.exists(d["path"])
        assert torch.equal(recebido, grande)
    finally:
        p.close()


def test_o_que_cabe_continua_na_arena():
    p = SMProducer(buffer_size = 64 * 1024, transbordo_em_arquivo = True)
    try:
        pequeno = torch.arange(100, dtype = torch.int64)
        d = p.send(pequeno)
        assert d["method"] == "buffer"
        assert p.segmentos == []
        assert torch.equal(_consumidor(p).recv(d), pequeno)
    finally:
        p.close()


def test_close_apaga_o_que_sobrou():
    p = SMProducer(buffer_size = 64 * 1024, transbordo_em_arquivo = True)
    d = p.send(torch.zeros(200_000, dtype = torch.uint8))
    assert os.path.exists(d["path"])
    p.close()
    assert not os.path.exists(d["path"])


def test_arena_de_carga_mantem_o_caminho_antigo():
    p = SMProducer(buffer_size = 64 * 1024)
    try:
        d = p.send(torch.zeros(200_000, dtype = torch.uint8))
        assert d["method"] == "share_memory"
        assert p.segmentos == []
    finally:
        p.close()
