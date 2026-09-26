"""
O cache do rascunho sob context parallel (`Cache.tabela_fisica`).

Sob CP o gerador conta paginas LOGICAS de PAGE_SIZE * cp_world tokens, e o rascunho -- que nao
reparte a sequencia -- guarda cada uma em cp_world paginas fisicas de PAGE_SIZE. Sem a traducao, o
kernel do cache Q8 do rascunho escreveu fora da memoria com um prompt de 9.605 tokens (Xid 31,
`q_cache.cu:228`, 26/09/2026, GLM-5.3-Flash + DFlash 2, EXL3_DCP=4, 4x A100).

Sem GPU: os tres metodos sao so aritmetica de indice e visao de tensor.
"""
from types import SimpleNamespace

import torch

from exllamav3.cache.cache import Cache
from exllamav3.constants import PAGE_SIZE


def _cache(r):
    c = object.__new__(Cache)
    c.paginas_por_logica = r
    return c


def test_fora_de_cp_a_tabela_e_a_mesma():
    bt = torch.tensor([[3, 7]], dtype = torch.int32)
    assert _cache(1).tabela_fisica(bt) is bt


def test_pagina_logica_vira_r_fisicas_em_ordem():
    bt = torch.tensor([[3, 7], [0, 1]], dtype = torch.int32)
    f = _cache(4).tabela_fisica(bt)
    assert f.dtype == torch.int32
    assert f.tolist() == [[12, 13, 14, 15, 28, 29, 30, 31], [0, 1, 2, 3, 4, 5, 6, 7]]
    # A posicao p do rascunho cai em f[p // PAGE_SIZE]: p = 1.024 (inicio da logica 7) -> fisica 28
    assert f[0, 1024 // PAGE_SIZE].item() == 7 * 4
    # E a entrada nao muda: o gerador reaproveita o block_index entre as voltas
    assert bt.tolist() == [[3, 7], [0, 1]]


def test_visao_logica_e_a_mesma_memoria():
    t = torch.arange(8 * PAGE_SIZE * 2, dtype = torch.float16).view(8, PAGE_SIZE, 2)
    v = _cache(4).por_pagina_logica(t)
    assert v.shape == (2, 4 * PAGE_SIZE, 2)
    assert v.data_ptr() == t.data_ptr()
    assert torch.equal(v[1], torch.cat([t[4], t[5], t[6], t[7]]))


def test_copy_page_reparte_os_tokens_pelas_fisicas():
    chamadas = []
    camada = SimpleNamespace(copy_page = lambda src, de, para, n: chamadas.append((de, para, n)))
    c = _cache(4)
    c.model = SimpleNamespace(loaded_tp = False)
    c.num_layers = 1
    c.layers = {(0, 0): camada}
    # Pagina logica 2 -> 5, com 600 tokens: 256 + 256 + 88, e a quarta fisica nao e tocada
    Cache.copy_page(c, c, 2, 5, 600)
    assert chamadas == [(8, 20, PAGE_SIZE), (9, 21, PAGE_SIZE), (10, 22, 600 - 2 * PAGE_SIZE)]
