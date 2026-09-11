"""
Geometria do pool da DSA sob context parallel (cache/dsa.py).

Roda na máquina, com torch: `python -m pytest tests/test_dsa_cache_cp.py`.

O que tranca: sob CP a página do gerador tem PAGE_SIZE x cp_world tokens, e o pool
(replicado) tem de contar as entradas por página nessa largura -- senão um job de T tokens
recebe T / (256 x world) páginas e o pool acha que elas guardam metade ("DSA pool overflow:
entry 4096 beyond block table (48 pages)", dipz5, 11/09/2026). A capacidade não muda.
"""

from types import SimpleNamespace

import pytest

from exllamav3.cache.dsa import CacheLayer_dsa
from exllamav3.constants import PAGE_SIZE


def _atencao(m):
    return SimpleNamespace(compress_rate = m, head_dim = 512, rope_head_dim = 64,
                           index_head_dim = 128, layer_type = "csa", _dcp = 1)


@pytest.mark.parametrize("m", [4, 128])
@pytest.mark.parametrize("world", [1, 2, 4])
def test_pagina_do_pool_acompanha_a_do_gerador(m, world):
    max_tokens = 1 << 20
    cl = CacheLayer_dsa(None, _atencao(m), 0, max_tokens, cp_world = world)
    page_tokens = PAGE_SIZE * world                       # generator.page_tokens
    assert cl.num_pages == max_tokens // page_tokens      # as paginas fisicas do gerador
    assert cl.capacity == max_tokens // m                 # a capacidade nao encolhe: replicado
    # A entrada da posicao global p cai na MESMA pagina que o gerador da a p
    for p in range(0, 200_000, 4_099):
        assert (p // m) // cl.epp == p // page_tokens
    # O job da dipz5: 16384 tokens em 48 paginas de 512 -> tem de caber
    if world == 2 and m == 4:
        assert 16384 // m <= 48 * cl.epp


def test_export_leva_o_grau_da_atencao():
    at = _atencao(4); at._dcp = 2
    cl = CacheLayer_dsa(None, at, 7, 4096)
    assert cl.tp_export(None)["args"]["cp_world"] == 2
