from __future__ import annotations
from typing_extensions import override
import torch

from ..model.config import no_default
from .dflash import DFlashConfig, DFlashModel
from ..modules.arch_specific.dflash2 import GroupedDynamicCausalConv, ConvSandwich, CandidateSelector

# DFlash 2 drafter (z-lab/dflash `DFlash2DraftModel`; checkpoints incoai/GLM-5.3-Flash-DFlash2,
# incoai/GLM-5.3-DFlash2, incoai/Qwen3.8-27B-DFlash2, ...). Same block-diffusion drafter as
# dflash.py (block of [anchor, mask...], K/V of the context derived from the target's hidden
# states, one forward per block, the target's lm_head on top), plus:
#   - a two-tap dynamic causal convolution before and after every attention and MLP sublayer;
#   - a candidate selector that keeps the top-k tokens at every position and traces one coherent
#     path through them (replaces the per-position argmax of sample_from_state);
#   - `is_causal: false` in the config: the sliding-window layers attend both ways inside the
#     block (the reference builds a two-sided window mask), so the window here is (sw, sw).
# Everything else -- target taps (+1 like the original release; the reference reads
# hidden_states[layer_id + 1]), block size, vocab, hidden sizes -- comes from the checkpoint's
# config.json, which is what makes one reader serve every DFlash 2 target.
# The generator is untouched: it still gets one sequence per block and verifies it as before.


class DFlash2Config(DFlashConfig):
    arch_string = "DFlash2DraftModel"
    tap_shift = 1

    def __init__(self, directory: str, **kwargs):
        super().__init__(directory, {"text": DFlash2Model}, **kwargs)
        self.conv_kernel_size = self.read_cfg(int, "dflash_config->conv_kernel_size", no_default)
        self.conv_group_size = self.read_cfg(int, "dflash_config->conv_group_size", no_default)
        self.selector_rank = self.read_cfg(int, "dflash_config->selector_rank", no_default)
        self.selector_top_k = self.read_cfg(int, "dflash_config->selector_top_k", no_default)
        self.draft_vocab_size = self.read_cfg(int, "vocab_size", no_default)
        # None = the reference's default (sliding layers causal); False = two-sided window
        self.is_causal = self.read_cfg(bool, "is_causal", None)
        # Gemma-style targets (the reference reads these with the same defaults)
        self.input_embedding_scale = self.read_cfg(float, ["dflash_config->input_embedding_scale", "input_embedding_scale"], 1.0)
        self.output_multiplier = self.read_cfg(float, ["dflash_config->output_multiplier", "output_multiplier"], 1.0)
        self.final_logit_softcapping = self.read_cfg(float, ["dflash_config->final_logit_softcapping", "final_logit_softcapping"], None)
        assert self.hidden_size % self.conv_group_size == 0, \
            f"DFlash2: hidden_size {self.hidden_size} não é múltiplo de conv_group_size {self.conv_group_size}"


class DFlash2Model(DFlashModel):
    config_class = DFlash2Config

    def __init__(self, config: DFlash2Config, **kwargs):
        super().__init__(config, **kwargs)

        # Wrap each block's attention and MLP with its dynamic convolution. The TransformerBlock
        # keeps calling self.attn.forward / self.mlp.forward; the sandwich does prepare -> inner
        # -> finish around them. self.attn_modules keeps the bare Attention modules, which is what
        # update_kv_from_target needs.
        for idx in range(config.num_hidden_layers):
            blk = self.modules[self.first_block_idx + idx]
            for nome, conv_key in (("attn", "attention_conv"), ("mlp", "mlp_conv")):
                inner = getattr(blk, nome)
                conv = GroupedDynamicCausalConv(
                    config = config,
                    key = f"layers.{idx}.{conv_key}",
                    hidden_size = config.hidden_size,
                    kernel_size = config.conv_kernel_size,
                    group_size = config.conv_group_size,
                )
                sandwich = ConvSandwich(config, f"layers.{idx}.{conv_key}", inner, conv)
                blk.modules[blk.modules.index(inner)] = sandwich
                setattr(blk, nome, sandwich)

            if config.is_causal is False and config.layer_types[idx] == "sliding_attention":
                # ponytail: the reference masks a two-sided window (|q - k| < sw); the fork's window
                # is (left, 0) end to end (AttnArgs.get_window_size, cache.get_kv), so a right side
                # would mean plumbing a tuple through four places. Full attention is identical for
                # contexts up to sw (2048) and only differs beyond it, where the draft sees more
                # context than it was trained on; the target's verification keeps decoding lossless
                # either way. Do the tuple if acceptance measurably drops at long contexts.
                self.attn_modules[idx].sliding_window = -1

        # The selector is not part of the forward chain: loaded after the chain, on the device of
        # the final norm (RMSNorm.load does not recurse into registered children)
        self.candidate_selector = CandidateSelector(
            config = config,
            key = "candidate_selector",
            hidden_size = config.hidden_size,
            vocab_size = config.draft_vocab_size,
            rank = config.selector_rank,
            top_k = config.selector_top_k,
        )


    @override
    def load_gen(self, *args, **kwargs):
        yield from super().load_gen(*args, **kwargs)
        self.candidate_selector.load(torch.device(self.modules[-1].device))


    @override
    def unload(self):
        self.candidate_selector.unload()
        super().unload()


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        # The last real token of the block is the selector's predecessor for position 1
        params["dflash2_anchor_ids"] = input_ids[:, -1].clone()
        return super().prepare_inputs(input_ids, params)


    def draft_logits_topk(self, state: torch.Tensor, params: dict):
        """Top-k (values, ids) of the target's lm_head over the drafter's state, on the master."""
        k = self.candidate_selector.top_k
        target = self.attached_model()
        if not target.loaded_tp:
            lm = target.modules[target.logit_layer_idx]
            logits = lm.forward(lm.prepare_for_device(state, params), params)
            logits = logits[..., :target.config.vocab_size].float()
            logits = self._ajustar_logits(logits)
            v, i = logits.topk(k, dim = -1)
            return v, i
        state = target.tp_producer.send(state)
        v, i = target.tp_dispatch_lm_head_topk((state, {}), k)
        return self._ajustar_logits(v.float()), i

    def _ajustar_logits(self, logits: torch.Tensor):
        cfg = self.config
        if cfg.output_multiplier != 1.0:
            logits = logits * cfg.output_multiplier
        if cfg.final_logit_softcapping:
            logits = torch.tanh(logits / cfg.final_logit_softcapping) * cfg.final_logit_softcapping
        return logits


    @override
    def sample_from_state(self, state: torch.Tensor, params: dict) -> torch.Tensor:
        """state (bsz, block_size, hidden): the drafter's normed output for [anchor, masks...].
        Position 0 is the anchor (the generator drops it); positions 1.. get the selector's path."""
        unary, candidates = self.draft_logits_topk(state, params)              # (b, n, k)
        anchor = params["dflash2_anchor_ids"].to(state.device)
        dev = state.device
        path, conf = self.candidate_selector.select(
            state[:, 1:].to(dev), unary[:, 1:].to(dev), candidates[:, 1:].to(dev), anchor,
        )
        # position 0: the plain argmax, kept only for shape
        p0 = candidates[:, :1].to(dev).gather(-1, unary[:, :1].to(dev).argmax(-1, keepdim = True))[..., 0]
        ids = torch.cat((p0, path), dim = 1)
        if params.get("export_draft_conf"):
            c0 = unary[:, :1].to(dev).max(dim = -1).values
            params["draft_conf"] = torch.cat((c0, conf), dim = 1)
        return ids
