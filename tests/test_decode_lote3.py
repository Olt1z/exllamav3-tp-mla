"""
Decode em lote acima de 8 linhas (3 jobs x q 4 da verificacao do rascunho MTP) sem GPU: o embedding
na CPU em fatias abaixo do grao do ATen (EXL3_EMBEDDING_CPU_SERIAL) e a MoE em sub-lotes de ate
MAX_BSZN linhas pelos kernels fundidos (EXL3_MOE_SUBLOTE_MAX). Roda com a extensao falsa
(`-p stub_ext`); nada aqui lanca kernel.
"""
import pytest
import torch
from torch import nn

from exllamav3.modules import embedding as emb_mod
from exllamav3.modules.embedding import Embedding, linhas_por_fatia
from exllamav3.modules import block_sparse_mlp as bsm
from exllamav3.modules.block_sparse_mlp import BlockSparseMLP, fatias_sublote, MAX_BSZN


# --- embedding serial ---------------------------------------------------------------------------

def _embedding(vocab, largura, dtype, multiplier = 1.0, normalize = False, out_dtype = torch.float):
    e = Embedding.__new__(Embedding)
    e.key = "embed_tokens"
    e.vocab_size = vocab
    e.hidden_size = largura
    e.out_dtype = out_dtype
    e._pinned_staging = {}
    e.multiplier = multiplier
    e.normalize = normalize
    e.device = torch.device("cpu")
    g = torch.Generator().manual_seed(0)
    w = torch.randn((vocab, largura), generator = g).to(dtype)
    e.embedding = nn.Embedding(vocab, largura, device = "meta")
    e.embedding.weight = nn.Parameter(w, requires_grad = False)
    return e


def _rodar(e, ids, serial, out_dtype = None, params = None):
    antes = emb_mod.cpu_serial
    emb_mod.cpu_serial = serial
    try:
        return e.forward(ids, dict(params or {}), out_dtype = out_dtype)
    finally:
        emb_mod.cpu_serial = antes


def test_linhas_por_fatia():
    # GLM-5.3-Flash: largura 4096 -> 8 linhas por grao (o limiar exato do salto de 2 para 3 jobs)
    assert linhas_por_fatia(4096) == 8
    assert linhas_por_fatia(2048) == 16
    assert linhas_por_fatia(5120) == 6
    assert linhas_por_fatia(40000) == 1


@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
@pytest.mark.parametrize("forma", [(1, 1), (2, 4), (3, 4), (4, 4), (1, 9), (16, 1)])
@pytest.mark.parametrize("mult, norm", [(1.0, False), (1.5, False), (1.0, True), (0.75, True)])
def test_serial_igual_ao_padrao(dtype, forma, mult, norm):
    e = _embedding(97, 4096, dtype, multiplier = mult, normalize = norm)
    ids = torch.randint(0, 97, forma)
    a = _rodar(e, ids, False)
    b = _rodar(e, ids, True)
    assert a.dtype == b.dtype == torch.float
    assert a.shape == b.shape == forma + (4096,)
    assert torch.equal(a, b)


@pytest.mark.parametrize("out_dtype", [torch.half, torch.float])
def test_serial_out_dtype(out_dtype):
    # A MTP sem TP pede half direto (qwen3_5_mtp: modules[0].forward(..., out_dtype = half))
    e = _embedding(50, 256, torch.half)
    ids = torch.randint(0, 50, (3, 4))
    a = _rodar(e, ids, False, out_dtype = out_dtype)
    b = _rodar(e, ids, True, out_dtype = out_dtype)
    assert a.dtype == b.dtype == out_dtype
    assert torch.equal(a, b)


def test_serial_fatias_abaixo_do_grao(monkeypatch):
    # Cada index_select pega no maximo um grao de linhas (o ATen so paraleliza com trabalho > grao)
    e = _embedding(64, 4096, torch.half)
    vistos = []
    orig = torch.index_select

    def espiao(w, dim, idx, *a, **k):
        vistos.append(idx.numel())
        return orig(w, dim, idx, *a, **k)

    monkeypatch.setattr(torch, "index_select", espiao)
    _rodar(e, torch.randint(0, 64, (3, 4)), True)
    assert vistos and max(vistos) <= 8 and sum(vistos) == 12
    assert all(n * 4096 <= 32768 for n in vistos)


def test_serial_respeita_limite_de_linhas(monkeypatch):
    # Acima de EXL3_EMBEDDING_CPU_SERIAL_MAX_LINHAS (o prefill) segue o caminho padrao, paralelo
    e = _embedding(64, 128, torch.half)
    monkeypatch.setattr(emb_mod, "cpu_serial_max_linhas", 16)
    assert e._usa_serial(torch.zeros((4, 4), dtype = torch.long)) is False   # desligado
    monkeypatch.setattr(emb_mod, "cpu_serial", True)
    assert e._usa_serial(torch.zeros((4, 4), dtype = torch.long))
    assert not e._usa_serial(torch.zeros((1, 17), dtype = torch.long))


def test_serial_buffer_fixado():
    if not torch.cuda.is_available():
        pytest.skip("pin_memory pede CUDA")
    e = _embedding(64, 4096, torch.half)
    ids = torch.randint(0, 64, (3, 4))
    a = _rodar(e, ids, False, params = {"pinned_staging": True}).clone()
    b = _rodar(e, ids, True, params = {"pinned_staging": True})
    assert b.is_pinned()
    assert torch.equal(a, b)


# --- MoE em sub-lotes ---------------------------------------------------------------------------

def test_fatias_sublote():
    for bsz in range(1, 65):
        f = fatias_sublote(bsz)
        assert f[0][0] == 0 and f[-1][1] == bsz
        assert all(a < b and b - a <= MAX_BSZN for a, b in f)
        assert all(f[i][1] == f[i + 1][0] for i in range(len(f) - 1))
        assert len(f) == (bsz + MAX_BSZN - 1) // MAX_BSZN
        tam = [b - a for a, b in f]
        assert max(tam) - min(tam) <= 1
    assert fatias_sublote(12) == [(0, 6), (6, 12)]
    assert fatias_sublote(16) == [(0, 8), (8, 16)]
    assert fatias_sublote(9) == [(0, 5), (5, 9)]


class _BCMoEFalso:
    """run_bszN de mentira: por linha, out = y * soma(pesos) + soma(experts), no buffer estatico
    compartilhado (como o kernel de verdade, que nao mistura linhas)."""

    def __init__(self, out_bszn):
        self.out_bszn = out_bszn
        self.chamadas = []

    def run_bszN(self, y, sel, w):
        n = y.shape[0]
        assert 1 <= n <= MAX_BSZN
        assert y.is_contiguous() and sel.is_contiguous() and w.is_contiguous()
        assert sel.shape[0] == w.shape[0] == n
        self.chamadas.append(n)
        self.out_bszn[:n] = y.float() * w.float().sum(-1, keepdim = True) + sel.float().sum(-1, keepdim = True)


def _referencia(y, sel, w):
    return y.float() * w.float().sum(-1, keepdim = True) + sel.float().sum(-1, keepdim = True)


@pytest.mark.parametrize("bsz", [9, 12, 16, 24])
def test_sublotes_por_linha(bsz):
    H, k = 32, 8
    m = BlockSparseMLP.__new__(BlockSparseMLP)
    out_bszn = torch.full((MAX_BSZN, H), float("nan"))

    class _Cfg:
        pass

    m.experts_cfg = _Cfg()
    m.experts_cfg.out_bszn = out_bszn
    m.bc = _BCMoEFalso(out_bszn)
    y = torch.randn((bsz, H)).half()
    sel = torch.randint(0, 288, (bsz, k))
    w = torch.rand((bsz, k)).half()
    out = m._run_bszN_sublotes(y, sel, w)
    assert out.shape == (bsz, H) and out.dtype == torch.float
    assert out.data_ptr() != out_bszn.data_ptr()
    assert torch.equal(out, _referencia(y, sel, w))
    assert m.bc.chamadas == [b - a for a, b in fatias_sublote(bsz)]


def test_sublote_padrao_desligado():
    # Sem a variavel o comportamento e o do ramo de integracao (acima de 8 linhas: exl3_moe)
    import os
    if "EXL3_MOE_SUBLOTE_MAX" not in os.environ:
        assert bsm.moe_sublote_max == 0
    if "EXL3_EMBEDDING_CPU_SERIAL" not in os.environ:
        assert emb_mod.cpu_serial is False


def test_despacho_liga_e_desliga():
    from exllamav3.util import perfil_componentes as pc
    a, b = emb_mod.cpu_serial, bsm.moe_sublote_max
    try:
        assert pc.mp_definir_decode_lote3({}, True, 16) == (True, 16)
        assert emb_mod.cpu_serial is True and bsm.moe_sublote_max == 16
        assert pc.mp_definir_decode_lote3({}, False, 0) == (False, 0)
    finally:
        emb_mod.cpu_serial, bsm.moe_sublote_max = a, b
