from ..util.misc import ratio_split
import heapq


def top_k_mask_(lst, k):
    assert 0 < k <= len(lst)
    idx_k_largest = [i for i, _ in heapq.nlargest(k, enumerate(lst), key=lambda t: t[1])]
    keep = set(idx_k_largest)
    for i in range(len(lst)):
        if i not in keep:
            lst[i] = 0


class TPAllocation:

    def __init__(
        self,
        key: str,
        channel_width: int = None,
        channel_unit: str = None,
        storage_per_device: int = 0,
        storage_to_split: int = 0,
        overhead_per_device: int = 0,  # per token
        overhead_to_split: int = 0,  # per token
        recons_temp: int = 0,
        channels_to_split: int = 1,
        limit_key: str = None,
        max_devices: int = None,
        cp: bool = False,
    ):
        self.key = key
        # Context parallel: so um componente que COMBINA entre os ranks do grupo pode receber a
        # mesma faixa de canais em varias placas. Para todos os outros (experts, MLP, atencao sem
        # combine) o grupo nao existe: eles continuam repartidos por placa, senao duas placas com
        # a mesma faixa somam em dobro no all-reduce e metade dos canais fica de fora -- medido
        # na prova 24 (KL 2,4 com o alocador agrupando tudo).
        self.cp = cp
        self.channel_width = channel_width
        self.channel_unit = channel_unit
        self.limit_key = limit_key

        self.storage_per_device = storage_per_device
        self.storage_to_split = storage_to_split
        self.overhead_per_device = overhead_per_device
        self.overhead_to_split = overhead_to_split
        self.recons_temp = recons_temp
        self.channels_to_split = channels_to_split
        self.max_devices = max_devices

        self.current_split = []


class TPAllocator:

    def __init__(
        self,
        components: list[TPAllocation],
        num_tokens: int,
        output_num_tokens: int,
        dev_limits: dict = None,
        dcp: int = 1,
        ordem_dos_ranks: list[int] | None = None,
    ):
        """
        Estimate and plan a tensor-parallel split across devices with uneven capacity.

        Components describe storage that must be replicated, storage that can be divided by channel, and temporary
        per-token overhead. initial_split() allocates each splittable component by the remaining memory ratios on
        the devices, so larger GPUs or less-loaded GPUs receive more channels while smaller devices can receive
        fewer or zero. The allocator tracks persistent storage as a sum and temporary overhead as a per-device max,
        allowing irregular splits when model shards and runtime buffers do not scale uniformly.
        """
        self.components = components
        self.current_split = None
        self.current_usage = None
        self.num_tokens = num_tokens
        self.output_num_tokens = output_num_tokens
        self.dev_limits = dev_limits or {}
        self.estimate_total = None
        self.estimate_storage = None
        self.estimate_overhead = None
        self.num_devices = None
        self.plan = None
        # Context parallel: `dcp` placas por grupo. As placas de um grupo recebem AS MESMAS
        # cabeças (os mesmos canais) e repartem a SEQUÊNCIA entre si. Então o canal é distribuído
        # por GRUPO, não por placa, e a capacidade do grupo é a da placa mais apertada dele --
        # dar a um grupo mais canais do que o seu menor membro aguenta estoura essa placa.
        self.dcp = dcp
        # Os grupos sao fatias CONSECUTIVAS desta lista (ids de placa na ordem dos ranks), e e a
        # MESMA regra do backend (configurar_cp: rank r no grupo r // dcp). O plano e indexado
        # por id fisico e o rank e a posicao em active_devices, que nao vem em ordem ([1, 2, 3, 0]
        # medido): agrupar por id daria grupos diferentes dos do NCCL, e foi isso que a prova 24
        # mediu -- KL 1,1 em dcp 2 com dcp 4 (um grupo so) quase certo.
        self.ordem_dos_ranks = ordem_dos_ranks
        self.grupos = []
        self.grupo_de = {}


    def initial_split(
        self,
        max_mem: list[int],
    ):
        self.num_devices = len(max_mem)
        active_devices = [i for i in range(self.num_devices) if max_mem[i] > 0]
        if not active_devices:
            raise RuntimeError("Insufficient VRAM in split for model and cache")
        ranks = self.ordem_dos_ranks or active_devices
        if len(ranks) % self.dcp:
            raise RuntimeError(
                f"dcp {self.dcp} tem de dividir o numero de placas ativas {len(ranks)}"
            )
        storage_sum = [0] * self.num_devices
        overhead_max = [0] * self.num_devices

        self.grupos = [ranks[g * self.dcp:(g + 1) * self.dcp] for g in range(len(ranks) // self.dcp)]
        self.grupo_de = {d: g for g, gr in enumerate(self.grupos) for d in gr}

        def por_grupo(v):
            """Reduz uma lista por placa a uma por grupo, pelo MENOR membro."""
            return [min(v[d] for d in gr) for gr in self.grupos]

        def por_placa(v):
            """Espalha uma lista por grupo de volta para as placas do grupo; placa fora de
            qualquer grupo (inativa) fica com zero."""
            return [v[self.grupo_de[d]] if d in self.grupo_de else 0 for d in range(self.num_devices)]

        for c in self.components:

            # Remaining computed space per device. If space runs out completely, increase maximum by 10%
            while True:
                rem_mem_s = [max(0, mm - ss - om) for mm, ss, om in zip(max_mem, storage_sum, overhead_max)]
                if sum(rem_mem_s) == 0:
                    max_mem = [m * 11 // 10 for m in max_mem]
                else:
                    break

            # Mask out devices to satisy max split per component type
            if c.max_devices is not None or c.limit_key:
                dev_limit = self.dev_limits.get(c.limit_key, c.max_devices)
                if dev_limit is not None:
                    dev_limit = min(dev_limit, len(active_devices))
                if dev_limit is not None:
                    top_k_mask_(rem_mem_s, dev_limit)

            # Perform split. Sob CP o canal vai para o GRUPO e depois se espalha; sem CP
            # (dcp = 1) `por_grupo`/`por_placa` sao identidade e isto e o codigo de sempre.
            channels = c.channels_to_split
            if self.dcp > 1 and c.cp:
                split = por_placa(ratio_split(channels, por_grupo(rem_mem_s), chunk_size = 1))
            else:
                split = ratio_split(channels, rem_mem_s, chunk_size = 1)
            c.current_split = split

            # Active devices on layer: those that actually received channels. A device with free
            # memory but zero channels holds none of the module, so it must not be charged the
            # per-device storage (replicated MLA caches make that charge large)
            mask = [s > 0 for s in split]

            # Compute storage and overhead given layer and split
            tokens = self.output_num_tokens if c is self.components[-1] else self.num_tokens
            storage = [
                (c.storage_per_device if m else 0)
                + c.storage_to_split * s // channels
                for s, m in zip(split, mask)
            ]
            overhead = [
                (c.overhead_per_device if m else 0)
                + tokens * c.overhead_to_split * s // channels
                + c.recons_temp * s // channels
                for s, m in zip(split, mask)
            ]

            # Compute overall usage
            storage_sum = [ss + s for ss, s in zip(storage_sum, storage)]
            overhead_max = [max(om, o) for om, o in zip(overhead_max, overhead)]

        self.estimate_storage = [ss for ss, om in zip(storage_sum, overhead_max)]
        self.estimate_overhead = [om for ss, om in zip(storage_sum, overhead_max)]
        self.estimate_total = [ss + om for ss, om in zip(storage_sum, overhead_max)]
        return self.estimate_total, self.estimate_storage, self.estimate_overhead


    def print_split(self):
        n_columns = len(self.estimate_total)
        def _divider():
            nonlocal n_columns
            print("    " + "-" * (62 + 10 * n_columns))
        def _columns(t, u, d):
            print(f"    {t:<50}{u:<12}" + "".join([f"{d_:>10}" for d_ in d]))

        print(" -- Model split:")
        _divider()
        _columns("", "Units", [f"CUDA:{i}" for i in range(n_columns)])
        _divider()
        for c in (c for c in self.components if c.channel_unit):
            _columns(c.key, c.channel_unit, [f"{s * c.channel_width}" for s in c.current_split])
        _divider()
        _columns("Storage", "GB", [f"{e / 1024**3:10.2f}" for e in self.estimate_storage])
        _columns("Overhead", "GB", [f"{e / 1024**3:10.2f}" for e in self.estimate_overhead])
        _columns("Total", "GB", [f"{e / 1024**3:10.2f}" for e in self.estimate_total])


    def compile_tp_plan(self):
        """
        Convert per-device channel counts into explicit slice ranges for each allocation key.

        The returned plan is indexed by device and maps each component key to (begin, end, unit), with channel_width
        applied so worker import code can slice tensors in the original channel units.
        """
        plan = []
        for _ in range(self.num_devices):
            plan.append({})
        for c in self.components:
            key = c.key
            idx_end = 0
            cw = c.channel_width or 1
            # Sob CP as placas de um grupo recebem A MESMA faixa de canais e repartem a sequencia
            # entre si; acumular por placa daria a cada uma um pedaco diferente do modelo, que e o
            # oposto do que o CP faz. So para componente com combine (c.cp); com dcp = 1 ou sem
            # combine o passo e 1 e isto e o laco de sempre.
            if self.dcp > 1 and c.cp:
                for gr in self.grupos:
                    idx_beg = idx_end
                    idx_end += c.current_split[gr[0]]
                    for d in gr:
                        plan[d][key] = (idx_beg * cw, idx_end * cw, c.channel_unit)
                for d in range(self.num_devices):
                    if d not in self.grupo_de:
                        plan[d][key] = (idx_end * cw, idx_end * cw, c.channel_unit)
            else:
                for dev in range(self.num_devices):
                    idx_beg = idx_end
                    idx_end += c.current_split[dev]
                    plan[dev][key] = (idx_beg * cw, idx_end * cw, c.channel_unit)
        self.plan = plan
        return self.plan
