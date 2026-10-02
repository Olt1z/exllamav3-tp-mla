"""
Indexador DSA dividido entre os ranks do TP (EXL3_INDEXADOR_DIVIDIDO, attention_fn/indexador_dividido.py).

Três camadas de prova:

  1. Lógica pura (sem placa): a partição das linhas do chunk entre ranks -- contígua, cobre tudo,
     fronteiras na grade de 256 do slab, balanceada, inclusive com R não divisível por tp e com
     ranks sem faixa -- e a decisão de dividir.
  2. Ida e volta do empacotar/desempacotar com um all-gather simulado (torch na CPU basta).
  3. Na placa, numa GPU só: o indexador por faixas, montado, é BYTE A BYTE o indexador inteiro
     (caminho normal e caminho em tiles, com e sem k-pool), e o forward da camada com a flag e um
     backend simulado dá os mesmos índices e a mesma saída que sem ela.

O teste em TP de verdade (vários processos, NCCL) roda o roteiro da bancada:

    EXL3_TESTE_MODELO=/workspace/modelo python -m pytest tests/test_indexador_dividido.py -k tp_real -s

que chama tests/bancada/medir_componentes.py --conferir e exige que no modo dividido todo rank
tenha exatamente os mesmos índices.
"""
import sys, os, importlib.util, itertools, subprocess
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pytest

# Carregado pelo caminho, sem passar pelo exllamav3/__init__ (que importa a extensão compilada):
# as provas da lógica pura não precisam de placa nem de build
_ARQ = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "exllamav3", "modules", "attention_fn", "indexador_dividido.py")
_spec = importlib.util.spec_from_file_location("indexador_dividido_puro", _ARQ)
idv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(idv)


# ---------------------------------------------------------------------------------------------------
# 1. Lógica pura

CASOS = [
    (4096, 4), (4096, 3), (4096, 8), (4096, 1), (4096, 2),
    (1000, 4),            # 4 slabs, o último parcial
    (700, 4),             # 3 slabs em 4 ranks: um rank sem faixa
    (300, 4), (256, 4), (1, 4), (257, 2),
    (5000, 8), (3 * 256 + 17, 2), (0, 3), (65536, 4),
]


@pytest.mark.parametrize("linhas,mundo", CASOS)
def test_faixas_particionam(linhas, mundo):
    f = idv.faixas_por_rank(linhas, mundo)
    assert len(f) == mundo
    # contíguas, começam em 0 e terminam em linhas
    assert f[0][0] == 0 and f[-1][1] == linhas
    for (a0, b0), (a1, b1) in zip(f, f[1:]):
        assert b0 == a1
    for a, b in f:
        assert 0 <= a <= b <= linhas
        # fronteiras na grade do slab, exceto o fim do chunk
        assert a % idv.SLAB == 0 or a == linhas
        assert b % idv.SLAB == 0 or b == linhas
    # balanceadas em slabs, e os slabs a mais ficam nos primeiros ranks
    slabs = [-(-(b - a) // idv.SLAB) for a, b in f]
    assert max(slabs) - min(slabs) <= 1
    assert slabs == sorted(slabs, reverse = True)
    assert sum(slabs) == -(-linhas // idv.SLAB)
    assert idv.linhas_max(f) == max(b - a for a, b in f)


def test_faixas_casos_conhecidos():
    assert idv.faixas_por_rank(4096, 4) == [(0, 1024), (1024, 2048), (2048, 3072), (3072, 4096)]
    assert idv.faixas_por_rank(1000, 4) == [(0, 256), (256, 512), (512, 768), (768, 1000)]
    assert idv.faixas_por_rank(700, 4) == [(0, 256), (256, 512), (512, 700), (700, 700)]
    assert idv.faixas_por_rank(1280, 4) == [(0, 512), (512, 768), (768, 1024), (1024, 1280)]
    assert idv.faixas_por_rank(4096, 3) == [(0, 1536), (1536, 2816), (2816, 4096)]


def test_deve_dividir():
    base = dict(ligado = True, mundo = 4, coletivo_ok = True, cp_world = 1, seqlen = 4096,
                min_linhas = 1024, com_cache = True, aquecimento = False)
    assert idv.deve_dividir(**base)
    for chave, valor in [("ligado", False), ("mundo", 1), ("coletivo_ok", False), ("cp_world", 2),
                         ("seqlen", 1023), ("seqlen", 1), ("com_cache", False), ("aquecimento", True)]:
        assert not idv.deve_dividir(**{**base, chave: valor}), chave
    assert idv.deve_dividir(**{**base, "seqlen": 1024})


# ---------------------------------------------------------------------------------------------------
# 2. Empacotar / desempacotar com all-gather simulado

def _simular(indices, bsz, seqlen, mundo, lixo):
    """Cada rank só tem a SUA faixa válida (o resto é lixo); empacota, 'all-gather' por stack,
    desempacota. Devolve a matriz montada no rank 0."""
    import torch
    faixas = idv.faixas_por_rank(seqlen, mundo)
    altura = idv.linhas_max(faixas)
    envios = []
    for r, (a, b) in enumerate(faixas):
        local = lixo.clone()
        v = local.view(bsz, seqlen, -1)
        v[:, a:b] = indices.view(bsz, seqlen, -1)[:, a:b]
        e = idv.empacotar(local, bsz, seqlen, (a, b), altura)
        assert e.shape == (bsz, altura, indices.shape[-1]) and e.is_contiguous()
        # linhas de enchimento saem -1
        assert (e[:, b - a:] == -1).all()
        envios.append(e)
    reunido = torch.stack(envios)
    return idv.desempacotar(reunido, faixas, bsz, seqlen)


@pytest.mark.parametrize("bsz,seqlen,mundo", [
    (1, 4096, 4), (1, 1000, 4), (1, 700, 4), (1, 1280, 4), (2, 1000, 3), (2, 4096, 4), (1, 300, 2),
])
def test_ida_e_volta(bsz, seqlen, mundo):
    torch = pytest.importorskip("torch")
    g = torch.Generator().manual_seed(seqlen * 7 + mundo)
    k_pad = 96
    indices = torch.randint(-1, 50000, (bsz * seqlen, k_pad), generator = g, dtype = torch.int32)
    lixo = torch.randint(-(2 ** 31), 2 ** 31 - 1, (bsz * seqlen, k_pad), generator = g, dtype = torch.int32)
    montado = _simular(indices, bsz, seqlen, mundo, lixo)
    assert montado.shape == indices.shape
    assert torch.equal(montado, indices)


# ---------------------------------------------------------------------------------------------------
# 2b. Os metodos do indexador de mla_attn.py, na CPU, com o kernel de pontuacao e o top-k trocados
# por referencias em torch: as faixas montadas sao exatamente o inteiro, inclusive com rank sem
# faixa e R fora da grade do slab. Prova a logica de slabs/faixas/cauda, nao o kernel

def _extrair_metodo(nome, score_tile):
    import ast, re, textwrap
    torch = pytest.importorskip("torch")
    arq = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "exllamav3", "modules", "mla_attn.py")
    src = open(arq, encoding = "utf-8").read()

    def scores(q, w, keys, q_pos0, cr, bound_max, scores = None, block_table = None, epp = 0):
        R, H, D = q.shape
        T = bound_max
        t = torch.arange(T)
        bt = block_table.reshape(-1).long()
        K = keys[bt[t // epp] * epp + t % epp]
        s = torch.einsum("rhd,td->rht", q.float(), K.float()).relu() * (D ** -0.5)
        s = torch.einsum("rh,rht->rt", w.float() * (H ** -0.5), s).half()
        lim = torch.clamp((q_pos0 + torch.arange(R) + 1) // cr, max = bound_max)
        s = s.masked_fill(t.unsqueeze(0) >= lim.unsqueeze(1), -float("inf"))
        scores[:R, :T] = s
        return scores[:, :T]

    class Ext:
        @staticmethod
        def dsa_topk(sc, out, k, _a, _b):
            out.fill_(-1)
            o = torch.sort(sc.float(), dim = 1, descending = True, stable = True)
            idx = o.indices[:, :k].int()
            out[:, :k] = torch.where(o.values[:, :k] > -float("inf"), idx, idx.new_full((), -1))

    class Cache:
        def __init__(self): self.d = {}
        def get(self, dev, shape, dtype, nome):
            return self.d.setdefault((nome, shape, dtype), torch.empty(shape, dtype = dtype))

    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == nome:
            code = textwrap.dedent(ast.get_source_segment(src, node))
            code = re.sub(r"^(\s*)from \.\S+ import .*$", r"\1pass", code, flags = re.M)
            ns = dict(torch = torch, dsa_indexer_scores = scores, ext = Ext, g_tensor_cache = Cache(),
                      _score_tile = score_tile)
            exec(code, ns)
            return ns[nome]
    raise KeyError(nome)


@pytest.mark.parametrize("kpool,tile", [(0, 32768), (0, 256), (4, 32768), (4, 128)])
def test_metodos_por_faixa_cpu(kpool, tile):
    torch = pytest.importorskip("torch")
    fn = _extrair_metodo("_indexer_topk_kpool" if kpool else "_indexer_topk", tile)

    class Lin:
        def __init__(self, t): self.t = t
        def forward(self, x, params): return self.t

    class M:
        pass

    H, D, topk, epp = 2, 16, 96, 64
    for seqlen, host0, bsz in [(1100, 0, 1), (700, 50, 2), (1280, 300, 1), (513, 1000, 1), (257, 2000, 2), (40, 3000, 1)]:
        g = torch.Generator().manual_seed(seqlen + host0)
        host = [host0 + 13 * b for b in range(bsz)]
        m = M()
        m.index_n_heads, m.index_head_dim, m.index_topk, m.index_kpool = H, D, topk, kpool
        m.index_kpool_tail = True
        m._indexer_rope_ = lambda *a, **k: None
        m.idx_wq_b = Lin(torch.randn(bsz, seqlen, H * D, generator = g).half())
        m.idx_weights = Lin(torch.randn(bsz, seqlen, H, generator = g).half())
        n_ent = (max(host) + seqlen) // (kpool or 1)
        pages = -(-n_ent // epp)
        plane = torch.randn(bsz * pages + 3, epp, D, generator = g).half()
        bt = torch.randperm(bsz * pages + 3, generator = g)[: bsz * pages].int().view(bsz, pages)
        x = torch.zeros(bsz, seqlen, 8)

        def rodar(linhas = None):
            if kpool:
                return fn(m, x, {}, None, bsz, seqlen, host, pool_plane = plane, block_table = bt,
                          linhas = linhas)
            return fn(m, x, {}, None, bsz, seqlen, host, 0, None, None, None, idx_pool = plane,
                      block_table = bt, linhas = linhas)

        inteiro = rodar()
        for mundo in (2, 3, 4, 8):
            faixas = idv.faixas_por_rank(seqlen, mundo)
            altura = idv.linhas_max(faixas)
            reunido = torch.stack([idv.empacotar(rodar(f), bsz, seqlen, f, altura) for f in faixas])
            montado = idv.desempacotar(reunido, faixas, bsz, seqlen)
            assert torch.equal(montado, inteiro), f"seqlen {seqlen} host {host} mundo {mundo}"


# ---------------------------------------------------------------------------------------------------
# 3. Na placa

def _cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("precisa de GPU")
    return torch


def _construir(kpool, topk, seed = 5):
    """Camada DSA "full" com pesos aleatórios, como test_mla_dsa.build_dsa, mais os tensores de
    k-pool do GLM-5.3 quando kpool > 0."""
    torch = _cuda()
    from exllamav3.modules import MLAttention
    from exllamav3.util.rope import RopeSettings, RopeStyle
    from test_mla import FakeConfig

    H, hidden, kv_lora, nope, rope_dim, v_head, q_lora = 8, 512, 512, 128, 64, 128, 256
    idx_heads, idx_dim = 4, 128
    g = torch.Generator(device = "cpu").manual_seed(seed)

    def rnd(*shape, scale = 0.085):
        return (torch.randn(*shape, generator = g) * scale).half()

    key = "model.layers.0.self_attn"
    t = {
        f"{key}.q_a_proj.weight": rnd(q_lora, hidden),
        f"{key}.q_a_layernorm.weight": (torch.randn(q_lora, generator = g) * 0.1 + 1).half(),
        f"{key}.q_b_proj.weight": rnd(H * (nope + rope_dim), q_lora),
        f"{key}.kv_a_proj_with_mqa.weight": rnd(kv_lora + rope_dim, hidden),
        f"{key}.kv_a_layernorm.weight": (torch.randn(kv_lora, generator = g) * 0.1 + 1).half(),
        f"{key}.kv_b_proj.weight": rnd(H * (nope + v_head), kv_lora),
        f"{key}.o_proj.weight": rnd(hidden, H * v_head),
        f"{key}.indexer.wq_b.weight": rnd(idx_heads * idx_dim, q_lora, scale = 0.25),
        f"{key}.indexer.wk.weight": rnd(idx_dim, hidden, scale = 0.25),
        f"{key}.indexer.k_norm.weight": (torch.randn(idx_dim, generator = g) * 0.1 + 1).half(),
        f"{key}.indexer.k_norm.bias": (torch.randn(idx_dim, generator = g) * 0.05).half(),
        f"{key}.indexer.weights_proj.weight": rnd(idx_heads, hidden, scale = 0.25),
    }
    if kpool:
        t[f"{key}.indexer.index_kpool_compress_ape"] = (torch.randn(kpool, idx_dim, generator = g) * 0.1).float()
        t[f"{key}.indexer.index_kpool_compress_gate"] = rnd(idx_dim, hidden, scale = 0.25)
    module = MLAttention(
        config = FakeConfig(t), key = key, layer_idx = 0, hidden_size = hidden,
        num_q_heads = H, kv_lora_rank = kv_lora, qk_nope_head_dim = nope,
        qk_rope_head_dim = rope_dim, v_head_dim = v_head,
        rope_settings = RopeSettings(head_dim = rope_dim, rope_theta = 10000.0, rope_style = RopeStyle.GPTJ),
        q_lora_rank = q_lora, rms_norm_eps = 1e-6,
        indexer_mode = "full", index_n_heads = idx_heads, index_head_dim = idx_dim,
        index_topk = topk, index_kpool = kpool,
    )
    module.load(torch.device("cuda:0"))
    return module


class _BackendSimulado:
    """all_gather de um processo só, em dois tempos. 'gravar': guarda o envio do rank corrente e
    devolve o próprio envio em todas as fatias (índices válidos, saída descartada). 'reproduzir':
    devolve os envios gravados de todos os ranks -- o que o NCCL entregaria."""

    def __init__(self, mundo):
        self.mundo = mundo
        self.modo = "gravar"
        self.rank = 0
        self.envios = {}
        self.chamadas = 0

    def all_gather(self, out, t):
        self.chamadas += 1
        if self.modo == "gravar":
            self.envios[self.rank] = t.clone()
            out.copy_(t.unsqueeze(0).expand_as(out))
        else:
            for r in range(self.mundo):
                out[r].copy_(self.envios[r])

    def all_reduce(self, t, contribution = True):
        raise AssertionError("camada sem tp_reduce não reduz")


def _rodar_chunks(module, layer, x, cortes, params_extra = None, chamar = None):
    torch = _cuda()
    bsz = x.shape[0]
    pages = layer.shape_c[0] // bsz
    bt = torch.arange(pages * bsz, dtype = torch.int32, device = x.device).view(bsz, pages)
    seqlens = torch.zeros((bsz,), dtype = torch.int32, device = x.device)
    saidas = []
    for a, b in zip(cortes, cortes[1:]):
        params = {"attn_mode": "flash_attn", "cache": layer, "block_table": bt,
                  "cache_seqlens": seqlens, "positions": seqlens.clone(), **(params_extra or {})}
        xa = x[:, a:b].contiguous()
        saidas.append(chamar(xa, params, a, b) if chamar else (module.forward(xa, params), params))
        seqlens = seqlens + (b - a)
    return saidas


@pytest.mark.parametrize("kpool", [0, 4])
@pytest.mark.parametrize("tile", [32768, 512])
def test_faixas_montadas_igual_ao_inteiro(kpool, tile, monkeypatch):
    """Chamar o indexador por faixas e montar dá exatamente a matriz do indexador inteiro. tile 512
    força o caminho em tiles (merge do top-k), onde o k_sel de cada slab sai do fim do slab."""
    torch = _cuda()
    from exllamav3.modules import mla_attn
    from exllamav3.cache import CacheLayer_MLA_fp16
    from exllamav3.constants import PAGE_SIZE
    monkeypatch.setattr(mla_attn, "_score_tile", tile)
    topk = 128
    module = _construir(kpool, topk)
    S, bsz = 3000, 1
    layer = CacheLayer_MLA_fp16(None, module, 0, -(-S // PAGE_SIZE) * PAGE_SIZE * bsz)
    layer.alloc(torch.device("cuda:0"))
    x = (torch.randn((bsz, S, module.hidden_size), device = "cuda:0") * 0.5).half()

    nome = "_indexer_topk_kpool" if kpool else "_indexer_topk"
    original = getattr(module, nome)
    capturado = []

    def espiao(*a, **k):
        capturado.append((a, k))
        return original(*a, **k)

    setattr(module, nome, espiao)
    # 1000 linhas (4 slabs, o último parcial) e 2000 (8 slabs): o primeiro chunk começa abaixo do
    # top-k, então os slabs do começo têm k_sel menor que o topk
    _rodar_chunks(module, layer, x, [0, 1000, S])
    assert len(capturado) == 2
    for a, k in capturado:
        seqlen = a[4]
        inteiro = original(*a, **k)
        for mundo in (2, 3, 4, 8):
            faixas = idv.faixas_por_rank(seqlen, mundo)
            montado = torch.full_like(inteiro, -7)
            for f in faixas:
                parcial = original(*a, **{**k, "linhas": f})
                montado[f[0]:f[1]] = parcial[f[0]:f[1]]
            assert torch.equal(montado, inteiro), f"mundo {mundo}, seqlen {seqlen}"


@pytest.mark.parametrize("kpool", [0, 4])
@pytest.mark.parametrize("mundo", [2, 4])
def test_forward_com_flag_igual_sem(kpool, mundo, monkeypatch):
    """A camada inteira, com a flag e um backend simulado, sai com os mesmos índices e a mesma
    saída que sem ela, em todo 'rank'. Exercita a decisão em _attend, _indexador_reunir e a
    montagem."""
    torch = _cuda()
    from exllamav3.modules import mla_attn
    from exllamav3.cache import CacheLayer_MLA_fp16
    from exllamav3.constants import PAGE_SIZE
    topk = 128
    module = _construir(kpool, topk, seed = 11)
    S, bsz = 2800, 1
    cortes = [0, 1024, 1724, S]          # 1024 = 4 slabs; 700 = 3 slabs (um rank sem faixa em 4)
    layer = CacheLayer_MLA_fp16(None, module, 0, -(-S // PAGE_SIZE) * PAGE_SIZE * bsz)
    layer.alloc(torch.device("cuda:0"))
    x = (torch.randn((bsz, S, module.hidden_size), device = "cuda:0") * 0.5).half()

    # Referência: caminho replicado
    monkeypatch.setattr(mla_attn, "_indexador_dividido", False)
    ref = [(y, p["dsa_topk_indices"].clone()) for y, p in _rodar_chunks(module, layer, x, cortes)]

    monkeypatch.setattr(mla_attn, "_indexador_dividido", True)
    monkeypatch.setattr(mla_attn, "_indexador_dividido_min", 256)
    module.tp_mundo, module.tp_coletivo_ok = mundo, True
    be = _BackendSimulado(mundo)

    def chamar(xa, params, a, b):
        params["backend"] = be
        be.envios.clear()
        be.modo = "gravar"
        for r in range(mundo):
            be.rank = module.tp_rank = r
            module.forward(xa, dict(params))
        be.modo = "reproduzir"
        saidas = []
        for r in range(mundo):
            module.tp_rank = r
            p = dict(params)
            saidas.append((module.forward(xa, p), p["dsa_topk_indices"].clone()))
        return saidas

    try:
        res = _rodar_chunks(module, layer, x, cortes, chamar = chamar)
    finally:
        module.tp_mundo, module.tp_rank, module.tp_coletivo_ok = 1, 0, False
    assert be.chamadas == 2 * mundo * (len(cortes) - 1)
    for (y_ref, i_ref), por_rank in zip(ref, res):
        for r, (y, i) in enumerate(por_rank):
            assert torch.equal(i, i_ref), f"rank {r}: índices diferentes do replicado"
            assert torch.equal(i, por_rank[0][1])
            err = (y.float() - y_ref.float()).abs().max().item() / max(y_ref.float().abs().max().item(), 1e-6)
            assert err < 1e-3, f"rank {r}: saída difere {err:.2e}"


def test_aquecimento_e_decode_nao_dividem(monkeypatch):
    """No aquecimento do TP (TPBackendNull, sem coletivo) e em chunk curto, nada de all-gather."""
    torch = _cuda()
    from exllamav3.modules import mla_attn
    from exllamav3.cache import CacheLayer_MLA_fp16
    from exllamav3.constants import PAGE_SIZE
    module = _construir(4, 128, seed = 3)
    S = 1600
    layer = CacheLayer_MLA_fp16(None, module, 0, -(-S // PAGE_SIZE) * PAGE_SIZE)
    layer.alloc(torch.device("cuda:0"))
    x = (torch.randn((1, S, module.hidden_size), device = "cuda:0") * 0.5).half()
    monkeypatch.setattr(mla_attn, "_indexador_dividido", True)
    monkeypatch.setattr(mla_attn, "_indexador_dividido_min", 1024)
    module.tp_mundo, module.tp_coletivo_ok = 4, True
    be = _BackendSimulado(4)
    try:
        # chunk de 1024 em aquecimento, e chunks de 512 e 32 (abaixo do mínimo) fora dele
        _rodar_chunks(module, layer, x, [0, 1024], params_extra = {"backend": be, "tp_warmup": True})
        _rodar_chunks(module, layer, x, [0, 512, 544], params_extra = {"backend": be})
    finally:
        module.tp_mundo, module.tp_coletivo_ok = 1, False
    assert be.chamadas == 0


# ---------------------------------------------------------------------------------------------------
# TP de verdade: vários processos, NCCL, o modelo inteiro

@pytest.mark.skipif(not os.environ.get("EXL3_TESTE_MODELO"), reason = "EXL3_TESTE_MODELO=/caminho/do/modelo")
def test_tp_real_indices_identicos_entre_ranks():
    torch = _cuda()
    if torch.cuda.device_count() < 2:
        pytest.skip("precisa de 2+ GPUs")
    roteiro = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bancada", "medir_componentes.py")
    r = subprocess.run(
        [sys.executable, roteiro, "-m", os.environ["EXL3_TESTE_MODELO"], "--backend", "nccl",
         "--contextos", os.environ.get("EXL3_TESTE_CONTEXTO", "16384"), "--medir", "1", "--conferir"],
        capture_output = True, text = True,
    )
    print(r.stdout[-6000:])
    print(r.stderr[-3000:])
    assert r.returncode == 0
    assert "[dividido]" in r.stdout and "divergências entre ranks: 0" in r.stdout.split("[dividido]")[-1]
