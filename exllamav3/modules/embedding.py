from __future__ import annotations
from typing_extensions import override
import os
import torch
from torch import nn
from ..model.config import Config
from ..util.tensor import to2
from . import Module
from ..tokenizer.mm_embedding import FIRST_MM_EMBEDDING_INDEX
from ..model.model_tp_alloc import TPAllocation

# Embedding na CPU sem o pool do OpenMP (decode em lote, 02/10/2026). O at::parallel_for do PyTorch
# so abre uma regiao paralela quando o trabalho passa do grao (at::internal::GRAIN_SIZE = 32768
# elementos; no index_select o grao e 32768 / largura linhas). Com largura 4096 (GLM-5.3-Flash) isso
# acontece exatamente acima de 8 linhas: o lote 2 x q 4 fica serial, o lote 3 x q 4 abre a regiao
# no index_select, no cast para fp32 e na copia para o buffer fixado. Sob TP os 4 processos dos ranks
# abrem a regiao juntos, cada um com um time do tamanho da maquina, e a barreira do fim espera
# threads desescalonadas: ~38 ms por passo no perfil, com a GPU parada (o host do rank preso ali, os
# outros girando no all-reduce). Com EXL3_EMBEDDING_CPU_SERIAL=1, ate
# EXL3_EMBEDDING_CPU_SERIAL_MAX_LINHAS linhas (o decode, a verificacao, a absorcao da MTP) saem em
# fatias de no maximo um grao: cada operacao fica abaixo do limiar e roda serial na thread que chamou.
# Mesmos valores, bit a bit (mesmo gather, mesmo cast). O prefill (mais linhas) segue paralelo.
_GRAO_ATEN = 32768
cpu_serial = os.environ.get("EXL3_EMBEDDING_CPU_SERIAL", "0") == "1"
cpu_serial_max_linhas = int(os.environ.get("EXL3_EMBEDDING_CPU_SERIAL_MAX_LINHAS", 256))


def linhas_por_fatia(largura: int) -> int:
    """Quantas linhas de `largura` elementos cabem num grao do ATen (o maximo que o index_select e
    a copia fazem sem abrir regiao paralela: os dois so paralelizam com trabalho > grao)."""
    return max(1, _GRAO_ATEN // max(1, largura))


class Embedding(Module):

    def __init__(
        self,
        config: Config | None,
        key: str,
        vocab_size: int,
        hidden_size: int,
        out_dtype: torch.dtype | None = torch.float,
        qmap: str | None = None,
        normalize: bool = False,
        multiplier: float = 1.0
    ):
        super().__init__(config, key, None)
        assert qmap is None, "No quant scheme for Embedding"

        self.key = key
        self.embedding = None
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.out_dtype = out_dtype
        self._pinned_staging = {}
        self._numel = vocab_size * hidden_size
        self.normalize = normalize
        self.multiplier = multiplier

        self.caps.update({
            "prefer_cpu": True,
        })

    @override
    def optimizer_targets(self):
        return []

    @override
    def load(self, device: torch.device, **kwargs):
        self.device = device
        weight = self.config.stc.get_tensor(self.key + ".weight", self.device, float2half = True, allow_bf16 = True)
        self._numel = weight.numel()
        self.embedding = nn.Embedding(
            self.vocab_size,
            self.hidden_size,
            device = "meta"
        )
        self.embedding.weight = nn.Parameter(weight)

    @override
    def unload(self):
        self.device = None
        self.embedding = None

    @override
    def get_tensors(self):
        return {
            f"{self.key}.weight": self.embedding.weight.data.contiguous()
        }

    @override
    def weights_numel(self):
        return self._numel
        
    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:

        # Ensure input IDs in params
        if "input_ids" not in params:
            params["input_ids"] = x

        indexed_emb = params.get("indexed_embeddings")
        input_ids = x
        out_dtype = out_dtype or self.out_dtype or x.dtype

        # Indexed embedding masks
        if indexed_emb:
            standard_mask = input_ids < FIRST_MM_EMBEDDING_INDEX
            indexed_masks = [
                (input_ids >= e.first_index) & (input_ids < (e.first_index + e.mm_length))
                for e in indexed_emb
            ]
            indexed_act = [im.any() for im in indexed_masks]
            use_indexed_emb = any(indexed_act)

        # Mixed embeddings when needed
        if indexed_emb and use_indexed_emb:
            bsz, seq_len = input_ids.shape
            combined_emb = torch.empty((bsz, seq_len, self.hidden_size), device = self.device, dtype = out_dtype)

            # Prepare deepstack embedding tensors
            if any(ie.deepstack_embeddings is not None for ie in indexed_emb) and indexed_act:
                assert all(ie.deepstack_embeddings is not None for ie in indexed_emb)
                num_layers = len(indexed_emb[0].deepstack_embeddings)
                assert all(num_layers == len(ie.deepstack_embeddings) is not None for ie in indexed_emb)
                deepstack_emb = [torch.zeros_like(combined_emb) for _ in range(num_layers)]
            else:
                deepstack_emb = None

            # Insert standard embeddings
            if standard_mask.any():
                for i in range(bsz):
                    standard_ids_row = input_ids[i][standard_mask[i]]
                    standard_emb_row = self.embedding(standard_ids_row)
                    combined_emb[i][standard_mask[i]] = standard_emb_row.to(out_dtype)

            # Only normalize standard embeddings
            if self.normalize:
                combined_emb *= combined_emb.shape[-1] ** 0.5

            # Also only scale standard embeddings
            if self.multiplier != 1.0:
                combined_emb *= self.multiplier

            # Insert indexed embeddings
            for im, ie, act in zip(indexed_masks, indexed_emb, indexed_act):
                if not act:
                    continue
                for i in range(bsz):
                    indexed_ids_row = input_ids[i][im[i]] - ie.first_index
                    combined_emb[i][im[i]] = ie.embeddings[indexed_ids_row].to(out_dtype)

                    # Prepare deepstack embeddings
                    if ie.deepstack_embeddings is not None:
                        for layer, de in enumerate(ie.deepstack_embeddings):
                            deepstack_emb[layer][i][im[i]] = de[indexed_ids_row].to(out_dtype)

            # Save deepstack embeddings to params
            if deepstack_emb is not None:
                params["deepstack_emb"] = deepstack_emb

            return combined_emb

        # No indexed embeddings, or none in current batch
        elif self._usa_serial(x):
            return self._forward_serial(x, params, out_dtype)

        else:
            x = self.embedding.forward(x)
            if self.multiplier != 1.0:
                x *= self.multiplier
            x = to2(x, out_dtype, self.out_dtype)
            if self.normalize:
                x *= x.shape[-1] ** 0.5
            # When the embedding resides on the CPU, its output is uploaded to the first
            # device layer; staging it through a reused pinned buffer makes that upload
            # asynchronous. Only callers that guarantee a sync point between forward passes
            # (the generator's decode loop) may set the pinned_staging flag.
            if params.get("pinned_staging") and x.device.type == "cpu":
                key = (x.shape, x.dtype)
                buf = self._pinned_staging.get(key)
                if buf is None:
                    if len(self._pinned_staging) > 8:
                        self._pinned_staging.clear()
                    buf = torch.empty_like(x, pin_memory = True)
                    self._pinned_staging[key] = buf
                buf.copy_(x)
                x = buf
            return x

    def _usa_serial(self, x: torch.Tensor) -> bool:
        if not cpu_serial or x.device.type != "cpu":
            return False
        w = self.embedding.weight
        return (
            w.device.type == "cpu" and
            x.numel() <= cpu_serial_max_linhas and
            w.shape[-1] <= _GRAO_ATEN
        )

    def _forward_serial(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype) -> torch.Tensor:
        """O mesmo que o caminho padrao (gather, multiplier na largura do peso, cast, normalize, e o
        buffer fixado do gerador), em fatias de um grao: nenhuma operacao abre o pool do OpenMP. Ver
        cpu_serial no topo."""
        w = self.embedding.weight.data
        largura = w.shape[-1]
        ids = x.reshape(-1)
        n = ids.numel()
        forma = tuple(x.shape) + (largura,)

        # O destino ja e o buffer fixado quando o gerador pede (um cast a menos que o caminho padrao,
        # que monta x e depois copia para o buffer)
        destino = None
        if params.get("pinned_staging"):
            chave = (torch.Size(forma), out_dtype)
            destino = self._pinned_staging.get(chave)
            if destino is None:
                if len(self._pinned_staging) > 8:
                    self._pinned_staging.clear()
                destino = torch.empty(forma, dtype = out_dtype, pin_memory = True)
                self._pinned_staging[chave] = destino
        if destino is None:
            destino = torch.empty(forma, dtype = out_dtype)

        d = destino.view(n, largura)
        passo = linhas_por_fatia(largura)
        escala = largura ** 0.5
        for a in range(0, n, passo):
            b = min(n, a + passo)
            linhas = torch.index_select(w, 0, ids[a:b])
            if self.multiplier != 1.0:
                linhas *= self.multiplier
            fatia = d[a:b]
            fatia.copy_(linhas)
            if self.normalize:
                fatia *= escala
        return destino

    def make_tp_allocation(self, options: dict) -> list[TPAllocation]:
        return []

    def tp_export(self, plan, producer):
        assert self.device is not None, "Cannot export module for TP before loading."
        return {
            "cls": Embedding,
            "kwargs": {
                "key": self.key,
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "out_dtype": self.out_dtype,
                "normalize": self.normalize,
                "multiplier": self.multiplier,
            },
            "embedding.weight": producer.send(self.embedding.weight),
            "device": self.device
        }

    @staticmethod
    def tp_import(local_context, exported, plan):
        consumer = local_context["consumer"]
        module = Embedding(
            config = None,
            **exported["kwargs"],
        )
        module.device = exported["device"]
        module.embedding = nn.Embedding(
            module.vocab_size,
            module.hidden_size,
            device = "meta"
        )
        emb = consumer.recv(exported["embedding.weight"], cuda = False)
        module.embedding.weight = nn.Parameter(emb)
        return module