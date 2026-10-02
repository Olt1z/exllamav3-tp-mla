"""
Decode em lote (bsz > 1) sem GPU: o portao do grafo da MLA kpool, o plano agrupado sem sincronizacao,
os rewinds recorrentes juntos e a absorcao da MTP por comprimento aceito. Roda com a extensao falsa
(tests/stub: `-p stub_ext`); nada aqui lanca kernel.
"""
import collections

import pytest
import torch

from exllamav3.modules.attention_fn import bc_mla
from exllamav3.modules.attention_fn.bc_mla import BCMLA, kpool_lote_aceito
from exllamav3.modules import mla_attn
from exllamav3.modules.mla_attn import MLAttention
from exllamav3.modules import gated_delta_net as gdn
from exllamav3.generator.generator import Generator


# --- portao do grafo kpool --------------------------------------------------------------------

def test_portao_kpool():
    # Sem kpool ou com um job: sempre
    assert kpool_lote_aceito(0, 4, 1, False, False)
    assert kpool_lote_aceito(4, 1, 1, False, True)
    # kpool em lote: so no denso, ligado e com a extensao aceitando
    assert kpool_lote_aceito(4, 4, 0, True, False)
    assert not kpool_lote_aceito(4, 4, 0, False, False)
    assert not kpool_lote_aceito(4, 4, 1, True, False)
    assert not kpool_lote_aceito(4, 4, 0, True, True)


class _BCFalso:
    def __init__(self, erro = None):
        self.erro = erro
        self.chamadas = []

    def run(self, bsz, q_len, *a, **k):
        self.chamadas.append((bsz, q_len, a[-3]))   # regime
        if self.erro is not None:
            raise RuntimeError(self.erro)


def _bcm(bc, cp_world = 1):
    b = BCMLA.__new__(BCMLA)
    b.index_kpool = 4
    b.indexer_mode = "full"
    b.index_topk = 2048
    b.hidden_size = 8
    b.o_dtype = torch.half
    b.cp_world = cp_world
    b.configured = {(n, 4, r) for n in (1, 2, 4, 8) for r in (0, 1)}
    b.kpool_lote_recusado = False
    b.slot_indices = {(n, 4): torch.zeros(1) for n in (1, 2, 4, 8)}
    b.bc = bc
    return b


def _passo(b, bsz, contexto):
    x = torch.zeros((bsz, 4, 8), dtype = torch.half)
    params = {"_mla_host_seqlens": [contexto] * bsz}
    cs = torch.full((bsz,), contexto, dtype = torch.int32)
    bt = torch.zeros((bsz, 16), dtype = torch.int32)
    return b.step(x, params, cs, bt, 0, None, None)


@pytest.fixture
def lote(monkeypatch):
    bc_mla.contagem_kpool_lote.clear()
    def ligar(v):
        monkeypatch.setattr(bc_mla, "kpool_lote", v)
    return ligar


def test_kpool_lote_desligado_recusa(lote):
    lote(False)
    bc = _BCFalso()
    b = _bcm(bc)
    assert _passo(b, 4, 44) is None
    assert bc.chamadas == []
    assert bc_mla.contagem_kpool_lote["desligado"] == 1
    # lote 1 continua no grafo
    assert _passo(b, 1, 44) is not None
    assert bc.chamadas == [(1, 4, 0)]


def test_kpool_lote_denso_vai_ao_grafo(lote):
    lote(True)
    bc = _BCFalso()
    b = _bcm(bc)
    y = _passo(b, 4, 44)
    assert y is not None and tuple(y.shape) == (4, 4, 8)
    assert bc.chamadas == [(4, 4, 0)]
    assert bc_mla.contagem_kpool_lote["grafo"] == 1


def test_kpool_lote_esparso_recusa(lote):
    lote(True)
    bc = _BCFalso()
    b = _bcm(bc)
    assert _passo(b, 4, 4000) is None
    assert bc.chamadas == []
    assert bc_mla.contagem_kpool_lote["recusa_esparso"] == 1


def test_kpool_lote_cp_fica_no_eager(lote):
    lote(True)
    bc = _BCFalso()
    b = _bcm(bc, cp_world = 2)
    assert _passo(b, 4, 44) is None
    assert bc.chamadas == []


def test_kpool_lote_extensao_antiga_cai_no_eager(lote):
    lote(True)
    bc = _BCFalso("BC_MLAttention: kpool indexer requires bsz 1")
    b = _bcm(bc)
    assert _passo(b, 4, 44) is None
    assert b.kpool_lote_recusado
    assert bc_mla.contagem_kpool_lote["recusa_ext"] == 1
    # dai em diante nem tenta
    assert _passo(b, 4, 44) is None
    assert len(bc.chamadas) == 1
    assert bc_mla.contagem_kpool_lote["recusa_ext"] == 2


def test_kpool_lote_outro_erro_sobe(lote):
    lote(True)
    b = _bcm(_BCFalso("BC_MLAttention: slot not configured"))
    with pytest.raises(RuntimeError, match = "slot not configured"):
        _passo(b, 4, 44)


# --- plano agrupado sem sincronizacao ---------------------------------------------------------

class _CamadaFalsa:
    def __init__(self, plane):
        self.plane = plane
        self.escritas = []

    def get_idx(self):
        return self.plane

    def update_pool_direct(self, pool_seqlens, bt, pool_keys):
        self.escritas.append((pool_seqlens.clone(), bt.clone(), pool_keys.clone()))


def _mla_kpool(P = 4, D = 8):
    m = MLAttention.__new__(MLAttention)
    m.index_kpool = P
    m.index_head_dim = D
    g = torch.Generator().manual_seed(1)
    m.idx_kpool_ape = torch.randn((P, D), generator = g)
    return m


@pytest.mark.parametrize("seqlen", [1, 4, 7])
def test_plano_agrupado_igual_sem_torch_tensor(seqlen, monkeypatch):
    P, D, epp, paginas = 4, 8, 16, 12
    m = _mla_kpool(P, D)
    g = torch.Generator().manual_seed(2)
    plane = torch.randn((paginas, epp, 2 * D), generator = g).half()
    bt = torch.tensor([[3, 1, 7, 0], [5, 2, 9, 4], [11, 6, 8, 10]], dtype = torch.int32)
    host = [15, 20, 27]
    cs = torch.tensor(host, dtype = torch.int32)

    antigo = _CamadaFalsa(plane)
    m._update_pool_plane(antigo, bt, host, seqlen)

    # o caminho com cache_seqlens nao pode criar tensor na placa a partir do host
    novo = _CamadaFalsa(plane)
    criados = []
    orig = torch.tensor
    def vigia(*a, **k):
        criados.append(k.get("device"))
        return orig(*a, **k)
    monkeypatch.setattr(torch, "tensor", vigia)
    monkeypatch.setattr(mla_attn, "_pool_kernel_eager", False)
    m._update_pool_plane(novo, bt, host, seqlen, cache_seqlens = cs)
    monkeypatch.setattr(torch, "tensor", orig)

    assert criados == []
    assert len(antigo.escritas) == len(novo.escritas) > 0
    for (s0, b0, k0), (s1, b1, k1) in zip(antigo.escritas, novo.escritas):
        assert s1.dtype == torch.int32 and torch.equal(s0, s1)
        assert torch.equal(b0, b1)
        assert torch.equal(k0, k1)


# --- rewinds juntos ---------------------------------------------------------------------------

def test_rewind_adiado_anota_e_ajusta_posicao():
    st = gdn.GDNState.__new__(gdn.GDNState)
    st.slot, st.position, st.last_history = 5, 100, 3
    pend = []
    st.rewind_adiado(2, pend)
    assert pend == [(5, 3, 2)]
    assert st.position == 98 and st.last_history == 0


def test_juntar_rewind_jobs(monkeypatch):
    def coletar(layers, slot, last_history, num_tokens):
        return {0: ([("c", slot)], [("s", slot)] if num_tokens else []),
                1: ([("c1", slot)], [])}
    monkeypatch.setattr(gdn, "_collect_rewind_jobs", coletar)
    j = gdn._juntar_rewind_jobs([], [(1, 3, 2), (4, 3, 0), (7, 1, 1)])
    assert j[0] == ([("c", 1), ("c", 4), ("c", 7)], [("s", 1), ("s", 7)])
    assert j[1] == ([("c1", 1), ("c1", 4), ("c1", 7)], [])


def test_rewind_em_lote_uma_ida_ao_tp(monkeypatch):
    chamadas = []

    class _Modelo:
        loaded_tp = True
        def tp_dispatch_all(self, fn, args):
            chamadas.append((fn, args))

    class _Cache:
        model = _Modelo()

    c = _Cache()
    gdn.rewind_em_lote(c, [])
    assert chamadas == []
    gdn.rewind_em_lote(c, [(1, 3, 2), (2, 3, 0)])
    assert len(chamadas) == 1
    fn, (cid, pend) = chamadas[0]
    assert fn is gdn.mp_cache_recurrent_rewind_lote and cid == id(c)
    assert pend == [(1, 3, 2), (2, 3, 0)]


def test_rewind_em_lote_sem_tp(monkeypatch):
    despachos = []
    monkeypatch.setattr(gdn, "_collect_rewind_jobs", lambda l, s, h, n: {0: ([s], [s])})
    monkeypatch.setattr(gdn, "_dispatch_rewind_jobs", lambda j: despachos.append(j))

    class _Modelo:
        loaded_tp = False

    class _Cache:
        model = _Modelo()
        def get_all_recurrent_layers(self):
            return {"a": object()}

    gdn.rewind_em_lote(_Cache(), [(1, 3, 2), (2, 3, 1)])
    assert despachos == [{0: ([1, 2], [1, 2])}]


# --- absorcao da MTP por comprimento aceito ---------------------------------------------------

def test_mtp_prefill_em_lote_igual_ao_por_job():
    bsz, W, H, paginas = 5, 4, 3, 6
    g = torch.Generator().manual_seed(3)
    batch_ids = torch.randint(0, 1000, (bsz, W), generator = g)
    block_index = torch.randint(0, 50, (bsz, paginas), generator = g, dtype = torch.int32)
    cache_seqlens = torch.tensor([10, 20, 30, 40, 50], dtype = torch.int32)
    target_hidden = torch.randn((bsz, W, H), generator = g)
    aceitos = [3, 1, 3, 2, 4]          # o job 1 nao aceitou nada: sem prefill

    feitos = []

    class _Rascunho:
        def prefill(self, ids, params):
            feitos.append((ids.clone(), params))

    class _DCache:
        def tabela_fisica(self, bt):
            return bt

    gen = Generator.__new__(Generator)
    gen.draft_model = _Rascunho()
    gen.draft_cache = _DCache()
    grupos = collections.defaultdict(list)
    for j, A in enumerate(aceitos):
        if A > 1:
            grupos[A].append((j, j + 1))
    gen._mtp_prefill_em_lote(dict(grupos), batch_ids, block_index, cache_seqlens, target_hidden)

    assert [f[0].shape[1] + 1 for f in feitos] == [2, 3, 4]
    # cada linha de cada prefill em lote e exatamente a entrada do prefill por job
    for ids, params in feitos:
        A = ids.shape[1] + 1
        linhas = [j for j, a in enumerate(aceitos) if a == A]
        for r, j in enumerate(linhas):
            assert torch.equal(ids[r], batch_ids[j, 1:A])
            assert torch.equal(params["block_table"][r], block_index[j])
            assert params["cache_seqlens"][r].item() == cache_seqlens[j].item() + 1
            assert torch.equal(params["target_hidden"][r], target_hidden[j, :A - 1, :])
        assert params["cache"] is gen.draft_cache


# --- GPU: o kernel do plano agrupado no eager contra o laco por linha -------------------------

class _CamadaPools:
    """Plano por token e plano agrupado de verdade; update_pool_direct escreve em Python."""

    def __init__(self, plane, pool, P):
        self.plane, self.pool, self.P = plane, pool, P

    def get_idx(self):
        return self.plane

    def get_pool(self):
        return self.pool

    def update_pool_direct(self, pool_seqlens, bt, pool_keys):
        epp = self.plane.shape[1]
        ppp = epp // self.P
        p0 = int(pool_seqlens[0].item())
        for i in range(pool_keys.shape[1]):
            p = p0 + i
            pagina = int(bt[0, (p * self.P) // epp].item())
            self.pool[pagina, p % ppp] = pool_keys[0, i]


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "CUDA required")
@pytest.mark.parametrize("seqlen", [1, 4, 7, 300])
def test_kernel_do_plano_agrupado_igual_ao_laco(seqlen, monkeypatch):
    P, D, epp, paginas = 4, 128, 256, 24
    dev = torch.device("cuda:0")
    m = _mla_kpool(P, D)
    m.idx_kpool_ape = m.idx_kpool_ape.to(dev)
    g = torch.Generator().manual_seed(4)
    plane = torch.randn((paginas, epp, 2 * D), generator = g).half().to(dev)
    perm = torch.randperm(paginas, generator = g).to(torch.int32)
    bt = perm[:12].view(3, 4).contiguous().to(dev)
    host = [15, 300, 517]
    cs = torch.tensor(host, dtype = torch.int32, device = dev)

    ref = _CamadaPools(plane, torch.zeros((paginas, epp // P, D), dtype = torch.half, device = dev), P)
    monkeypatch.setattr(mla_attn, "_pool_kernel_eager", False)
    m._update_pool_plane(ref, bt, host, seqlen, cache_seqlens = cs)

    ker = _CamadaPools(plane, torch.zeros_like(ref.pool), P)
    monkeypatch.setattr(mla_attn, "_pool_kernel_eager", True)
    m._update_pool_plane(ker, bt, host, seqlen, cache_seqlens = cs)
    torch.cuda.synchronize()

    # Os pools completos tocados pelo append batem; o kernel ainda escreve o parcial do fim
    for b, pos0 in enumerate(host):
        for p in range(pos0 // P, (pos0 + seqlen) // P):
            pagina = int(bt[b, (p * P) // epp].item())
            torch.testing.assert_close(ker.pool[pagina, p % (epp // P)].float(),
                                       ref.pool[pagina, p % (epp // P)].float(), atol = 2e-3, rtol = 2e-3)


# --- contagem de sincronizacoes do perfil -----------------------------------------------------

def test_contagem_de_syncs_liga_e_devolve_originais():
    from exllamav3.util import perfil_componentes as pc
    antes = (torch.tensor, torch.cuda.synchronize, torch.Tensor.item, torch.Tensor.tolist, torch.Tensor.cpu)
    pc.mp_perfil_contar_syncs({}, True)
    try:
        assert torch.tensor is not antes[0]
        # CPU nao conta
        t = torch.tensor([1, 2], device = "cpu")
        t.sum().item()
        t.tolist()
        t.cpu()
        assert pc.mp_perfil_syncs({}) == {}
        # o rotulo aberto recebe a contagem
        pc._abertos.append("mla")
        pc._contar_sync("teste")
        pc._abertos.pop()
        pc._contar_sync("teste")
        assert pc.mp_perfil_syncs({}) == {"mla": 1, "fora": 1}
        assert pc.mp_perfil_syncs({}) == {}
    finally:
        pc.mp_perfil_contar_syncs({}, False)
    depois = (torch.tensor, torch.cuda.synchronize, torch.Tensor.item, torch.Tensor.tolist, torch.Tensor.cpu)
    assert all(a is b for a, b in zip(antes, depois))
