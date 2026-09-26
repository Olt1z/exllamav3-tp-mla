from collections import OrderedDict
from ..constants import PAGE_SIZE
from ..util.memory import malloc_trim

# Checkpoint stashes are MB-scale host allocations with LRU (i.e. interleaved) lifetimes —
# exactly the churn glibc retains after free (issue #277). Return memory to the OS once
# enough has been released; per-event cost at this threshold is a few ms. The accumulator
# is per-process, which also gives each tensor-parallel rank its own (their stashes live
# in the child processes)
_TRIM_THRESHOLD = 256 * 1024**2
_freed_bytes = 0

# Quantos checkpoints de pagina PARCIAL (fim de prompt fora da fronteira de pagina) ficam guardados.
# Cada um serve a UMA conversa, e so ate o turno seguinte dela: o turno seguinte grava o seu e o de
# antes vira passado. Quatro cobre conversas alternadas sem deixar o LRU dos checkpoints de pagina
# inteira -- os que retomam um prefixo longo -- ser expulso por eles.
MAX_PARCIAIS = 4

def note_freed(nbytes: int):
    global _freed_bytes
    _freed_bytes += nbytes
    if _freed_bytes >= _TRIM_THRESHOLD:
        _freed_bytes = 0
        malloc_trim()


class RecurrentCache(OrderedDict):
    def __init__(
        self,
        model,
        max_size: int = 4 * 1024**3,
    ):
        super().__init__()
        self.max_size = max_size
        self.current_size = 0
        self.model = model

        # Optionally set by the Generator; enables stranded-first eviction and staleness metrics
        self.pagetable = None
        self.metrics = {
            "stash_evictions": 0,           # checkpoints dropped by LRU pressure
            "stash_evictions_stranded": 0,  # of those, checkpoints that were already unrestorable
            "stash_evictions_live_kv": 0,   # of those, checkpoints whose anchor KV page was still cached
            "stash_pruned": 0,              # stranded checkpoints dropped by prune_stranded()
            "parciais_guardados": 0,        # checkpoints de fim de prompt no meio da pagina
            "parciais_descartados": 0,      # os que sairam pelo limite MAX_PARCIAIS
        }


    def get_stashed(self, key, default = None):
        """
        Fetch state from cache and move it to the end of the queue
        """
        if key in self:
            self.move_to_end(key)
            return self[key]
        return default


    def put(self, key, state, parcial: dict | None = None):
        """
        Add state to cache

        `parcial` marca um checkpoint de fim de prompt no MEIO de uma pagina (ver
        `Job.maybe_stash_recurrent_parcial`): a chave nao e o hash de uma pagina, e sim o do prefixo da
        pagina ate aquela posicao, e o dicionario leva `prev_hash` (a pagina anterior, a ancora da
        cadeia) e `n` (quantos tokens da pagina o estado ja viu).
        """
        if key in self:
            self.move_to_end(key)
        else:
            stashed_state = state.stash()
            if parcial is not None:
                stashed_state["parcial"] = parcial
            state_size = stashed_state["checkpoint_size"]
            while self.update_total_size() + state_size > self.max_size:
                assert self.current_size >= 0, "Not enough space in cache for single state"
                pt = self.pagetable

                # A checkpoint whose anchor page chain has been broken by KV eviction can never be restored by
                # an allocation, so drop stranded checkpoints (oldest first) before restorable ones. This is a
                # pure win: if the conversation returns, the replay prefill recreates the same checkpoint at no
                # extra cost, since the missing pages force a replay past this position either way.
                popped_key = None
                if pt is not None:
                    for k, v in self.items():
                        if self._encalhado(k, v):
                            popped_key = k
                            break
                if popped_key is not None:
                    popped = self.pop(popped_key)
                    self.metrics["stash_evictions_stranded"] += 1
                else:
                    popped_key, popped = self.popitem(last = False)
                    if pt is not None:
                        page = pt.referenced_pages.get(popped_key) or pt.unreferenced_pages.get(popped_key)
                        if page is not None and page.kv_position == PAGE_SIZE:
                            self.metrics["stash_evictions_live_kv"] += 1

                self.metrics["stash_evictions"] += 1
                note_freed(popped["checkpoint_size"])
                if self.model.loaded_tp:
                    self.model.tp_dispatch_all(mp_cache_recurrent_del, (id(self), popped["tp_handle"]))

            self[key] = stashed_state
            self.update_total_size()
            if parcial is not None:
                self.metrics["parciais_guardados"] += 1
                parciais = [k for k, v in self.items() if "parcial" in v]
                for k in parciais[:-MAX_PARCIAIS]:
                    self._descartar(k)
                    self.metrics["parciais_descartados"] += 1


    def _descartar(self, key):
        popped = self.pop(key)
        note_freed(popped["checkpoint_size"])
        if self.model.loaded_tp:
            self.model.tp_dispatch_all(mp_cache_recurrent_del, (id(self), popped["tp_handle"]))
        self.update_total_size()


    def _encalhado(self, key, stashed) -> bool:
        """
        O checkpoint nunca mais pode ser restaurado.

        O de pagina inteira depende da cadeia ate a pagina da chave. O parcial nao tem pagina com o
        hash da chave -- a pagina dele ainda nao esta completa --, entao a ancora e a pagina ANTERIOR;
        sem esta distincao o `is_resumable` diria que todo parcial esta encalhado, e o
        `prune_stranded` do fim de fila apagaria cada um antes do turno seguinte chegar.
        """
        parcial = stashed.get("parcial")
        if parcial is None:
            return not self.pagetable.is_resumable(key)
        prev = parcial["prev_hash"]
        return prev is not None and not self.pagetable.is_resumable(prev)


    def parciais(self, prev_hash):
        """Os checkpoints parciais ancorados em `prev_hash`, como (chave, n, parcial)."""
        for k, v in self.items():
            p = v.get("parcial")
            if p is not None and p["prev_hash"] == prev_hash:
                yield k, p["n"], p


    def prune_stranded(self) -> int:
        """
        Drop all checkpoints whose anchor page chain has been broken by KV eviction. A stranded checkpoint can
        never be restored by an allocation, and if its conversation returns, the replay prefill recreates it at
        no extra cost, so this only frees system RAM that would otherwise sit dead until LRU pressure reaches it.
        Intended to be called when the generator goes idle.
        """
        if self.pagetable is None:
            return 0
        stranded = [k for k, v in self.items() if self._encalhado(k, v)]
        for k in stranded:
            popped = self.pop(k)
            self.metrics["stash_pruned"] += 1
            note_freed(popped["checkpoint_size"])
            if self.model.loaded_tp:
                self.model.tp_dispatch_all(mp_cache_recurrent_del, (id(self), popped["tp_handle"]))
        if stranded:
            self.update_total_size()
        return len(stranded)


    def update_total_size(self):
        seen = set()
        total = 0
        for v in self.values():
            if id(v) in seen:
                continue
            seen.add(id(v))
            total += v["checkpoint_size"]
        self.current_size = total
        return total


# Checkpoint handles key the per-rank recurrent_cache dicts and must be unique across all
# recurrent module types (GDN, short-conv, SWA states all stash through the same dict)
_next_checkpoint_handle = 0

def new_checkpoint_handle() -> int:
    global _next_checkpoint_handle
    h = _next_checkpoint_handle
    _next_checkpoint_handle += 1
    return h


# Per-rank functions for tensor-parallel mode

def mp_cache_recurrent_clear(local_context: dict, cache_id: int, slot: int):
    recurrent_modules = local_context["recurrent_modules"]
    for module in recurrent_modules:
        recurrent_layer = module.tp_recurrent_lookup[cache_id]
        recurrent_layer.clear(slot)


def mp_cache_recurrent_stash(local_context: dict, cache_id: int, cp_handle: int, slot: int, position: int = 0):
    recurrent_modules = local_context["recurrent_modules"]
    recurrent_cache = local_context["recurrent_cache"]
    stashed = []
    for module in recurrent_modules:
        l = module.tp_recurrent_lookup[cache_id]
        stashed.append(l.stash(slot, position))
    recurrent_cache[cp_handle] = stashed


def mp_cache_recurrent_unstash(local_context: dict, cache_id: int, cp_handle: int, slot: int, position: int = 0):
    recurrent_modules = local_context["recurrent_modules"]
    recurrent_cache = local_context["recurrent_cache"]
    stashed = recurrent_cache[cp_handle]
    for module, s in zip(recurrent_modules, stashed):
        l = module.tp_recurrent_lookup[cache_id]
        l.unstash(slot, s, position)


def _stashed_bytes(obj) -> int:
    import torch
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, (list, tuple)):
        return sum(_stashed_bytes(o) for o in obj)
    return 0


def mp_cache_recurrent_del(local_context: dict, cache_id: int, cp_handle: int):
    recurrent_cache = local_context["recurrent_cache"]
    stashed = recurrent_cache.pop(cp_handle)
    note_freed(_stashed_bytes(stashed))
