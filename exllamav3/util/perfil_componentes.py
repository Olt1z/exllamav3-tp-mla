"""
Perfil do forward por COMPONENTE dentro dos processos dos ranks do tensor parallel.

O `tests/bancada/perfil_prefill.py` envolve os módulos do processo principal, e em TP eles rodam nos
processos dos ranks -- o perfil por módulo sai vazio e só o total vale. Este arquivo mora na
biblioteca, e não na bancada, porque as funções `mp_*` são despachadas aos ranks por
`Model.tp_worker_dispatch_wait_multi` e o processo filho precisa importá-las pelo nome.

Uso (ver tests/bancada/medir_componentes.py):

    model.tp_worker_dispatch_wait_multi(model.active_devices, mp_perfil_instrumentar, ())
    ... prefill ...
    por_rank = model.tp_worker_dispatch_wait_multi(model.active_devices, mp_perfil_colher, ())

Cada chamada envolvida grava um par de `torch.cuda.Event` no stream corrente do rank. O tempo de um
componente é o intervalo entre os dois eventos NA LINHA DO TEMPO DA GPU: inclui as bolhas em que a
GPU esperou a CPU lançar (o indexador tem laços Python por slab e tile) e, nos coletivos, a espera
pelo rank mais lento -- o "all_reduce" medido é comunicação + desbalanceamento, que é o que se ganha
ao sobrepô-lo ao cálculo.

Os componentes se aninham (o all-reduce da atenção roda dentro do forward da atenção; o indexador
dentro dela). A coleta devolve o tempo INCLUSIVO e o EXCLUSIVO de cada rótulo: exclusivo = o
intervalo menos o dos filhos medidos. A soma dos exclusivos é o tempo coberto, sem dupla contagem.

Nada aqui roda sem ser chamado: instrumentar substitui métodos de instância (o atributo da
instância encobre o da classe), e desinstrumentar os devolve.

Decode em lote (tests/bancada/medir_decode_lote.py): além do tempo na GPU, cada chamada guarda o
tempo de PAREDE no host (perf_counter na entrada e na saída) -- o 4º campo da coleta, inclusivo. No
decode a GPU espera a CPU, então host >> GPU num componente é lançamento Python ou sincronização.
`mp_perfil_contar_syncs` conta, por componente aberto, as chamadas que bloqueiam o host na placa
(.item(), .tolist(), .cpu() de tensor CUDA, torch.tensor(..., device = cuda), synchronize).
`instrumentar_local` faz o mesmo com os módulos de um modelo do processo principal (a cabeça MTP,
que o TabbyAPI carrega fora do TP).
"""
from __future__ import annotations

import collections
import time

import torch

# Rótulo por classe para o forward dos filhos de um bloco. O que não está aqui sai pelo nome da
# classe, para nada ficar fora da tabela.
_ROTULO_POR_CLASSE = {
    "MLAttention": "mla",
    "GatedDeltaNet": "kda",
    "BlockSparseMLP": "moe",
    "GatedMLP": "mlp_denso",
    "MLP": "mlp_denso",
    "RMSNorm": "normas",
    "LayerNorm": "normas",
}

# Métodos internos da MLAttention medidos à parte (rótulo, método)
_MLA_INTERNOS = (
    ("indexador_chaves", "_indexer_keys"),
    ("indexador_chaves", "_update_pool_plane"),
    ("indexador_topk", "_indexer_topk"),
    ("indexador_topk", "_indexer_topk_kpool"),
    ("indexador_reunir", "_indexador_reunir"),
    ("atencao_esparsa", "_attend_sparse"),
)

# Coletivos do backend: (rótulo, método)
_COLETIVOS = (
    ("all_reduce", "all_reduce"),
    ("all_reduce", "broadcast"),
    ("all_gather", "all_gather"),
    ("reduce_scatter", "reduce_scatter"),
)

_FILHOS_DO_BLOCO = ("attn_norm", "attn", "mlp_norm", "mlp", "attn_post_norm", "mlp_post_norm", "input_norm")


class _Registro:
    """Pilha de chamadas abertas e a lista de intervalos fechados, por processo."""

    def __init__(self, device):
        self.device = device
        self.pilha = []          # índices em self.chamadas das chamadas ainda abertas
        self.chamadas = []       # [rótulo, índice do pai ou -1, evento inicial, evento final]
        self.originais = []      # (objeto, nome do método) para desinstrumentar
        self.ativo = True        # desligado, os wrappers só repassam (o enchimento do cache)
        self.nvtx = False        # regiões NVTX com o rótulo, para a linha do tempo do nsys
        self.conferir = False
        self.digests = []        # (layer_idx, digest) dos índices da atenção esparsa "full"
        self.guardar_camada = None
        self.indices_guardados = []

    def envolver(self, obj, metodo, rotulo):
        orig = getattr(obj, metodo, None)
        if orig is None or getattr(orig, "_perfil_componentes", False):
            return 0
        reg = self
        # O device do próprio módulo, quando ele tem um (a cabeça MTP pode estar em outra placa
        # que a do registro); o stream corrente DESSE device é o que mede o trabalho dele
        dev = getattr(obj, "device", None)
        try:
            dev = reg.device if dev is None else torch.device(dev)
            if dev.type != "cuda":
                dev = reg.device
        except (TypeError, RuntimeError):
            dev = reg.device

        def f(*a, **k):
            if not reg.ativo:
                return orig(*a, **k)
            if reg.nvtx:
                torch.cuda.nvtx.range_push(rotulo)
            stream = torch.cuda.current_stream(dev)
            e0 = torch.cuda.Event(enable_timing = True)
            e0.record(stream)
            pai = reg.pilha[-1] if reg.pilha else -1
            idx = len(reg.chamadas)
            reg.chamadas.append([rotulo, pai, e0, None, time.perf_counter(), None])
            reg.pilha.append(idx)
            _abertos.append(rotulo)
            try:
                return orig(*a, **k)
            finally:
                _abertos.pop()
                reg.pilha.pop()
                e1 = torch.cuda.Event(enable_timing = True)
                e1.record(torch.cuda.current_stream(dev))
                reg.chamadas[idx][3] = e1
                reg.chamadas[idx][5] = time.perf_counter()
                if reg.nvtx:
                    torch.cuda.nvtx.range_pop()

        f._perfil_componentes = True
        # O objeto pode já ter um atributo de instância com esse nome (outro wrapper); guarda o
        # que havia no __dict__ para devolver exatamente isso
        self.originais.append((obj, metodo, obj.__dict__.get(metodo, _AUSENTE)))
        setattr(obj, metodo, f)
        return 1

    def desinstrumentar(self):
        for obj, metodo, antes in reversed(self.originais):
            if antes is _AUSENTE:
                try:
                    delattr(obj, metodo)
                except AttributeError:
                    pass
            else:
                setattr(obj, metodo, antes)
        self.originais.clear()


_AUSENTE = object()

# Rótulos abertos NESTE processo, de todos os registros (o do rank e o local da cabeça MTP): a
# contagem de sincronizações atribui cada uma ao componente mais interno aberto, ou "fora"
_abertos = []
_syncs = collections.Counter()
_syncs_originais = []


def _contar_sync(motivo):
    _syncs[(_abertos[-1] if _abertos else "fora")] += 1


def _ligar_contagem_syncs():
    if _syncs_originais:
        return
    T = torch.Tensor

    def em_tensor(nome):
        orig = getattr(T, nome)

        def f(self, *a, **k):
            if self.is_cuda:
                _contar_sync(nome)
            return orig(self, *a, **k)
        _syncs_originais.append((T, nome, orig))
        setattr(T, nome, f)

    for nome in ("item", "tolist", "cpu"):
        em_tensor(nome)

    orig_tensor = torch.tensor

    def tensor(*a, **k):
        d = k.get("device")
        if d is not None and torch.device(d).type == "cuda":
            _contar_sync("torch.tensor")
        return orig_tensor(*a, **k)
    _syncs_originais.append((torch, "tensor", orig_tensor))
    torch.tensor = tensor

    orig_sync = torch.cuda.synchronize

    def synchronize(*a, **k):
        _contar_sync("synchronize")
        return orig_sync(*a, **k)
    _syncs_originais.append((torch.cuda, "synchronize", orig_sync))
    torch.cuda.synchronize = synchronize


def _desligar_contagem_syncs():
    for obj, nome, orig in reversed(_syncs_originais):
        setattr(obj, nome, orig)
    _syncs_originais.clear()


def mp_perfil_contar_syncs(local_context: dict, ligado: bool):
    """Liga/desliga a contagem de sincronizações host-placa neste processo (ver o topo)."""
    if ligado:
        _ligar_contagem_syncs()
    else:
        _desligar_contagem_syncs()
    _syncs.clear()


def _rotulo(m) -> str:
    return _ROTULO_POR_CLASSE.get(type(m).__name__, type(m).__name__)


def _instrumentar_mla(reg: _Registro, m) -> int:
    n = 0
    for rotulo, metodo in _MLA_INTERNOS:
        if hasattr(m, metodo):
            n += reg.envolver(m, metodo, rotulo)
    # Conferência dos índices: digest de cada seleção que entra na atenção esparsa de uma camada
    # "full" (as "shared" reusam a mesma matriz)
    if getattr(m, "indexer_mode", None) == "full":
        orig = m._attend_sparse

        def conferido(q_lat, q_pe, bsz, seqlen, params, ckv_cache, kpe_cache, block_table,
                      indices, *a, **k):
            if reg.ativo and reg.conferir:
                reg.digests.append((m.layer_idx, _digest(indices)))
                if reg.guardar_camada == m.layer_idx:
                    reg.indices_guardados.append(indices.cpu())
            return orig(q_lat, q_pe, bsz, seqlen, params, ckv_cache, kpe_cache, block_table,
                        indices, *a, **k)

        conferido._perfil_componentes = True
        reg.originais.append((m, "_attend_sparse", m.__dict__.get("_attend_sparse", _AUSENTE)))
        m._attend_sparse = conferido
        n += 1
    return n


def _digest(indices: torch.Tensor) -> tuple:
    """Impressão digital da matriz de índices, calculada na placa (32 MB por camada não descem
    ao host). Soma, soma ponderada pela posição e contagem de -1: duas matrizes que diferem em
    um único índice diferem na soma."""
    v = indices.reshape(-1).long()
    pos = torch.arange(v.numel(), device = v.device, dtype = torch.long) % 1000003 + 1
    return (
        tuple(indices.shape),
        int(v.sum().item()),
        int((v * pos).sum().item()),
        int((v < 0).sum().item()),
    )


def _instrumentar_modulos(reg: _Registro, modules, prefixo: str = "") -> int:
    n = 0
    for m in modules:
        filhos = [getattr(m, f, None) for f in _FILHOS_DO_BLOCO]
        if not any(f is not None for f in filhos):
            n += reg.envolver(m, "forward", prefixo + _rotulo(m))
            continue
        for f in filhos:
            if f is None:
                continue
            n += reg.envolver(f, "forward", prefixo + _rotulo(f))
            if type(f).__name__ == "MLAttention" and not prefixo:
                n += _instrumentar_mla(reg, f)
        for nome in ("attn_hc", "mlp_hc"):
            hc = getattr(m, nome, None)
            if hc is not None:
                n += reg.envolver(hc, "mix", prefixo + "hc")
                n += reg.envolver(hc, "apply_", prefixo + "hc")
    return n


def mp_perfil_instrumentar(local_context: dict) -> int:
    """Envolve os componentes deste rank. Idempotente. Devolve quantos pontos foram envolvidos."""
    reg = local_context.get("_perfil_componentes")
    if reg is not None:
        return 0
    device = local_context["device"]
    reg = _Registro(device)
    local_context["_perfil_componentes"] = reg
    n = _instrumentar_modulos(reg, local_context["modules"])
    backend = local_context["backend"]
    for rotulo, metodo in _COLETIVOS:
        if hasattr(backend, metodo):
            n += reg.envolver(backend, metodo, rotulo)
    return n


def instrumentar_local(modules, device, prefixo: str = "mtp.") -> dict:
    """Instrumenta os módulos de um modelo do PROCESSO PRINCIPAL (fora do TP), com rótulos
    prefixados. Devolve um local_context mínimo para usar com mp_perfil_colher / _zerar / _ativar
    / _desinstrumentar diretamente (sem despacho)."""
    reg = _Registro(torch.device(device))
    _instrumentar_modulos(reg, modules, prefixo)
    return {"_perfil_componentes": reg, "device": torch.device(device)}


def mp_perfil_desinstrumentar(local_context: dict):
    reg = local_context.pop("_perfil_componentes", None)
    if reg is not None:
        reg.desinstrumentar()


def mp_perfil_zerar(local_context: dict):
    reg = local_context.get("_perfil_componentes")
    if reg is not None:
        torch.cuda.synchronize(reg.device)
        reg.chamadas.clear()
        reg.pilha.clear()
        reg.digests.clear()
        reg.indices_guardados.clear()
    _syncs.clear()


def mp_perfil_colher(local_context: dict, zerar: bool = True) -> dict:
    """{rótulo: (chamadas, ms inclusivo, ms exclusivo, ms de host inclusivo)} deste rank desde o
    último zerar."""
    reg = local_context.get("_perfil_componentes")
    if reg is None:
        return {}
    torch.cuda.synchronize(reg.device)
    dur = [c[2].elapsed_time(c[3]) if c[3] is not None else 0.0 for c in reg.chamadas]
    host = [(c[5] - c[4]) * 1000.0 if c[5] is not None else 0.0 for c in reg.chamadas]
    filhos = [0.0] * len(dur)
    for i, c in enumerate(reg.chamadas):
        if c[1] >= 0:
            filhos[c[1]] += dur[i]
    tab = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0])
    for i, c in enumerate(reg.chamadas):
        t = tab[c[0]]
        t[0] += 1
        t[2] += dur[i] - filhos[i]
        # Inclusivo só conta a chamada mais externa do rótulo (um rótulo aninhado em si mesmo,
        # como dois indexador_chaves seguidos, não dobra)
        p = c[1]
        while p >= 0 and reg.chamadas[p][0] != c[0]:
            p = reg.chamadas[p][1]
        if p < 0:
            t[1] += dur[i]
            t[3] += host[i]
    out = {k: tuple(v) for k, v in tab.items()}
    if zerar:
        mp_perfil_zerar(local_context)
    return out


def mp_perfil_syncs(local_context: dict, zerar: bool = True) -> dict:
    """{rótulo: sincronizações} deste PROCESSO desde o último zerar (com a contagem ligada). No
    rank de saída, que roda no processo principal, entram as do gerador ("fora") e as da cabeça
    MTP ("mtp.*"). Chamar ANTES de mp_perfil_colher, que zera a contagem."""
    out = dict(_syncs)
    if zerar:
        _syncs.clear()
    return out


def mp_perfil_conferir(local_context: dict, ligado: bool, guardar_camada: int | None = None):
    """Liga a coleta dos digests dos índices (e, opcional, a cópia inteira dos de uma camada)."""
    reg = local_context.get("_perfil_componentes")
    assert reg is not None, "instrumente antes de conferir"
    reg.conferir = ligado
    reg.guardar_camada = guardar_camada


def mp_perfil_digests(local_context: dict, com_indices: bool = False) -> dict:
    """Digests (e, com_indices, as cópias da camada guardada) desde o último zerar. Não limpa:
    a bancada pede os digests a todos os ranks e as cópias, que são grandes, só a um."""
    reg = local_context.get("_perfil_componentes")
    if reg is None:
        return {"digests": [], "indices": []}
    torch.cuda.synchronize(reg.device)
    return {"digests": list(reg.digests),
            "indices": list(reg.indices_guardados) if com_indices else []}


def mp_perfil_ativar(local_context: dict, ativo: bool):
    """Pausa (False) ou retoma a medição sem desinstrumentar: o enchimento de 900k tokens são
    centenas de chunks, e gravar eventos de todos só para jogá-los fora custa memória."""
    reg = local_context.get("_perfil_componentes")
    if reg is not None:
        reg.ativo = bool(ativo)


def mp_perfil_nvtx(local_context: dict, ligado: bool):
    reg = local_context.get("_perfil_componentes")
    if reg is not None:
        reg.nvtx = bool(ligado)


def mp_cuda_profiler(local_context: dict, ligar: bool):
    """cudaProfilerStart/Stop NESTE processo: com `nsys --capture-range=cudaProfilerApi`, cada
    rank é um processo e abre/fecha a própria janela de captura."""
    torch.cuda.synchronize(local_context["device"])
    if ligar:
        torch.cuda.cudart().cudaProfilerStart()
    else:
        torch.cuda.cudart().cudaProfilerStop()


def mp_definir_indexador_dividido(local_context: dict, ligado: bool, min_linhas: int | None = None):
    """Liga/desliga o indexador dividido NESTE rank. Despachar sempre a todos os ranks juntos, entre
    dois forwards: um rank dividindo e outro não trava no all-gather."""
    from ..modules import mla_attn
    mla_attn._indexador_dividido = bool(ligado)
    if min_linhas is not None:
        mla_attn._indexador_dividido_min = int(min_linhas)
    return (mla_attn._indexador_dividido, mla_attn._indexador_dividido_min)


def mp_descrever_mla(local_context: dict) -> list:
    """(layer_idx, indexer_mode, cabeças locais, tp_mundo, tp_rank, tp_coletivo_ok, cp_world) de
    cada MLAttention deste rank: diz quantas camadas "full" o modelo tem e se o caminho dividido
    pode ligar."""
    out = []
    for m in local_context["modules"]:
        a = getattr(m, "attn", None)
        if a is not None and type(a).__name__ == "MLAttention":
            out.append((a.layer_idx, a.indexer_mode, a.num_q_heads, a.tp_mundo, a.tp_rank,
                        a.tp_coletivo_ok, a.cp_world))
    return out


def mp_definir_decode_lote(local_context: dict, kpool_lote: bool, pool_kernel: bool):
    """Liga/desliga, NESTE processo, o grafo da MLA kpool em lote (attention_fn/bc_mla.py:
    kpool_lote) e o kernel do plano agrupado no eager (mla_attn._pool_kernel_eager). Despachar
    a todos os ranks juntos, entre dois forwards (o processo principal também roda a cabeça MTP,
    e o rank de saída é ele mesmo)."""
    from ..modules import mla_attn
    from ..modules.attention_fn import bc_mla
    bc_mla.kpool_lote = bool(kpool_lote)
    mla_attn._pool_kernel_eager = bool(pool_kernel)
    return (bc_mla.kpool_lote, mla_attn._pool_kernel_eager)


def mp_definir_decode_lote3(local_context: dict, embedding_serial: bool, moe_sublote_max: int):
    """Liga/desliga, NESTE processo, o embedding na CPU sem o pool do OpenMP
    (modules/embedding.py: cpu_serial, EXL3_EMBEDDING_CPU_SERIAL) e os sub-lotes da MoE acima de
    MAX_BSZN linhas (block_sparse_mlp.moe_sublote_max, EXL3_MOE_SUBLOTE_MAX; 0 desliga). Despachar
    a todos os ranks juntos, entre dois forwards: a decisao da MoE tem de ser a mesma em todos."""
    from ..modules import embedding, block_sparse_mlp
    embedding.cpu_serial = bool(embedding_serial)
    block_sparse_mlp.moe_sublote_max = int(moe_sublote_max)
    return (embedding.cpu_serial, block_sparse_mlp.moe_sublote_max)


def mp_contagem_kpool_lote(local_context: dict, zerar: bool = True) -> dict:
    """Quantos passos de MLA kpool com bsz > 1 foram ao grafo e por que os outros recusaram
    (desligado / recusa_esparso / recusa_ext = extensão sem o lote)."""
    from ..modules.attention_fn import bc_mla
    out = dict(bc_mla.contagem_kpool_lote)
    if zerar:
        bc_mla.contagem_kpool_lote.clear()
    return out
