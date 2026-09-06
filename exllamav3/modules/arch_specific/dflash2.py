from __future__ import annotations
import torch
from ...model.config import Config
from ...modules import Module, Linear

"""
DFlash 2 (z-lab/dflash, `DFlash2DraftModel`): the three pieces that the original DFlash drafter
(dflash.py) does not have. Reference: dflash/model.py in https://github.com/z-lab/dflash.

- GroupedDynamicCausalConv: a two-tap causal convolution over the draft block, applied before
  and after every attention and MLP sublayer. Each tap is a learned base kernel plus a
  per-position correction predicted from the hidden state, one correction per group of
  `group_size` channels:  conv(x)_t = (base_0 + d_{t,0}) * x_t + (base_1 + d_{t,1}) * x_{t-1}.
  The first position of the block sees a zero left neighbour (the block is [anchor, masks...]).
- ConvSandwich: wraps the block's attention or MLP so TransformerBlock keeps its own forward:
  prepare (conv on the normed input) -> inner -> finish (conv on the output, before the residual).
- CandidateSelector: keeps the top-k tokens at every block position and traces one path through
  them, scoring each adjacent pair (a, b) by  U_t(b) + <A(a) * H(h_t), B(b)>  with two
  256-dim token codebooks and a projection of the draft state. Greedy here (the generator's
  verification is what makes decoding lossless); the sampled variant of the reference is not
  needed for the ExLlamaV3 generator, which verifies against its own sampler.
"""


def grouped_dynamic_convolve(x: torch.Tensor, dynamic: torch.Tensor, base: torch.Tensor, group_size: int):
    """x (b, L, H); dynamic (b, L, K, groups) per-position corrections; base (K, H). fp32 math,
    result in x's dtype. Mirrors _grouped_dynamic_convolve of the reference."""
    b, L, H = x.shape
    groups = H // group_size
    blocks = x.float().view(b, L, groups, group_size)
    dyn = dynamic.float().view(b, L, base.shape[0], groups, 1)
    out = torch.zeros_like(blocks)
    for offset in range(base.shape[0]):
        if offset == 0:
            values = blocks
        else:
            values = torch.cat((torch.zeros_like(blocks[:, :offset]), blocks[:, :-offset]), dim = 1)
        kernel = base[offset].view(1, 1, groups, group_size)
        out = out + (kernel + dyn[:, :, offset]) * values
    return out.view(b, L, H).to(x.dtype)


class GroupedDynamicCausalConv(Module):

    def __init__(self, config: Config, key: str, hidden_size: int, kernel_size: int, group_size: int):
        super().__init__(config, key, None)
        self.module_name = "GroupedDynamicCausalConv"
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.group_size = group_size
        self.groups = hidden_size // group_size
        # checkpoint: <key>.kernel_projection.weight (2 * kernel_size * groups, hidden), <key>.base_kernel (2, kernel_size, hidden)
        self.kernel_projection = Linear(
            config = config,
            key = f"{key}.kernel_projection",
            in_features = hidden_size,
            out_features = 2 * kernel_size * self.groups,
            out_dtype = torch.float,
            pad_to = 1,
        )
        self.register_submodule(self.kernel_projection)
        self.base_kernel = None

    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        self.base_kernel = self.config.stc.get_tensor(f"{self.key}.base_kernel", device, no_defer = True).float().contiguous()
        assert self.base_kernel.shape == (2, self.kernel_size, self.hidden_size), \
            f"{self.key}.base_kernel: esperado (2, {self.kernel_size}, {self.hidden_size}), veio {tuple(self.base_kernel.shape)}"

    def unload(self):
        super().unload()
        self.base_kernel = None

    def get_tensors(self):
        return {f"{self.key}.base_kernel": self.base_kernel}

    def forward(self, x, params, out_dtype = None):
        raise NotImplementedError("use prepare/finish")

    def prepare(self, x: torch.Tensor, params: dict):
        """Conv on the sublayer input; returns (convolved input, corrections kept for finish)."""
        b, L, _ = x.shape
        dyn = self.kernel_projection.forward(x, params, out_dtype = torch.float).view(b, L, 2, self.kernel_size, self.groups)
        return grouped_dynamic_convolve(x, dyn[:, :, 0], self.base_kernel[0], self.group_size), dyn[:, :, 1]

    def finish(self, y: torch.Tensor, dyn: torch.Tensor):
        """Conv on the sublayer output, before the residual add."""
        return grouped_dynamic_convolve(y, dyn, self.base_kernel[1], self.group_size)


class ConvSandwich(Module):
    """prepare -> inner -> finish, so the block's forward stays the generic TransformerBlock one."""

    def __init__(self, config: Config, key: str, inner: Module, conv: GroupedDynamicCausalConv):
        super().__init__(config, key, None)
        self.module_name = "ConvSandwich"
        self.inner = inner
        self.conv = conv
        self.register_submodule(inner)
        self.register_submodule(conv)

    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None):
        y, dyn = self.conv.prepare(x, params)
        y = self.inner.forward(y, params)
        return self.conv.finish(y, dyn)


class CandidateSelector(Module):

    def __init__(self, config: Config, key: str, hidden_size: int, vocab_size: int, rank: int, top_k: int):
        super().__init__(config, key, None)
        self.module_name = "CandidateSelector"
        self.vocab_size = vocab_size
        self.rank = rank
        self.top_k = top_k
        self.hidden_projection = Linear(
            config = config,
            key = f"{key}.hidden_projection",
            in_features = hidden_size,
            out_features = rank,
            out_dtype = torch.float,
            pad_to = 1,
        )
        self.register_submodule(self.hidden_projection)
        # checkpoint: <key>.predecessor_codebook / <key>.successor_codebook (vocab, rank), no ".weight" suffix
        self.predecessor_codebook = None
        self.successor_codebook = None

    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        stc = self.config.stc
        self.predecessor_codebook = stc.get_tensor(f"{self.key}.predecessor_codebook", device, no_defer = True).half().contiguous()
        self.successor_codebook = stc.get_tensor(f"{self.key}.successor_codebook", device, no_defer = True).half().contiguous()
        for nome, cb in (("predecessor", self.predecessor_codebook), ("successor", self.successor_codebook)):
            assert cb.shape == (self.vocab_size, self.rank), \
                f"{self.key}.{nome}_codebook: esperado ({self.vocab_size}, {self.rank}), veio {tuple(cb.shape)}"

    def unload(self):
        super().unload()
        self.predecessor_codebook = None
        self.successor_codebook = None

    def get_tensors(self):
        return {
            f"{self.key}.predecessor_codebook": self.predecessor_codebook,
            f"{self.key}.successor_codebook": self.successor_codebook,
        }

    def forward(self, x, params, out_dtype = None):
        raise NotImplementedError("use select")

    def select(self, hidden: torch.Tensor, unary: torch.Tensor, candidates: torch.Tensor, anchor_ids: torch.Tensor):
        """
        hidden (b, n, H): the drafter's normed state at the n block positions after the anchor
        unary (b, n, k), candidates (b, n, k): top-k logits and token ids at each position
        anchor_ids (b,): the last verified token, predecessor of position 0
        Returns (path (b, n) token ids, confidence (b, n) = unary logit of the chosen token).
        """
        h = self.hidden_projection.forward(hidden, {}, out_dtype = torch.float)  # (b, n, r)
        predecessor = anchor_ids.to(hidden.device)
        path, conf = [], []
        for t in range(hidden.shape[1]):
            gate = self.predecessor_codebook[predecessor].float() * h[:, t]                 # (b, r)
            successors = self.successor_codebook[candidates[:, t]].float()                 # (b, k, r)
            scores = unary[:, t].float() + torch.einsum("br,bkr->bk", gate, successors)     # (b, k)
            index = scores.argmax(dim = -1)
            predecessor = candidates[:, t].gather(-1, index[:, None])[:, 0]
            path.append(predecessor)
            conf.append(unary[:, t].float().gather(-1, index[:, None])[:, 0])
        return torch.stack(path, dim = 1), torch.stack(conf, dim = 1)
