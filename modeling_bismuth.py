"""
modeling_bismuth.py — HuggingFace-compatible wrapper for Bismuth-104M (CrystalFormer v4)
DuoNeural Lab | Archon + Jesse Caldwell + Aura | 2026-10-14

Usage:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        "DuoNeural/bismuth-104m-base", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

Requires: hyp_cdm_transformer.py in the same directory (also in this repo).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import asdict

from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

# -- config lives in configuration_bismuth.py (HF convention) --
try:
    from .configuration_bismuth import BismuthConfig
except ImportError:
    from configuration_bismuth import BismuthConfig

# -- pull the actual model from the companion source file --
try:
    from .hyp_cdm_transformer import (
        CrystalFormer, CrystalFormerConfig as _CrystalFormerDataclass,
        CURVATURE_SCHEDULE, C_MIN, C_MAX, C_INIT,
    )
except ImportError:
    from hyp_cdm_transformer import (
        CrystalFormer, CrystalFormerConfig as _CrystalFormerDataclass,
        CURVATURE_SCHEDULE, C_MIN, C_MAX, C_INIT,
    )


# ── HuggingFace PreTrainedModel wrapper ──────────────────────────────────────

class BismuthForCausalLM(PreTrainedModel):
    """
    Bismuth-104M base language model (CrystalFormer v4).

    First published base LM to use learnable hyperbolic curvature (Poincaré ball /
    Thomas-Wigner connection scan) inside the core transformer stack rather than
    only in embeddings or downstream graph tasks.

    Architecture: [TransformerBlock, TransformerBlock, TransformerBlock, HypCDMBlock] × 3
    - 9 standard causal self-attention + SwiGLU blocks (FlashAttention-2 via SDPA)
    - 3 HypCDMBlock layers: complex recurrent state on hyperbolic manifold
    - Per-layer curvature: L0≈0.88, L1≈0.79, L2≈1.45 (learned during training)
    - Tied embeddings, RoPE positional encoding, RMSNorm throughout
    - 104,529,987 unique parameters (143M total tensor storage, tied embeddings)
    """
    config_class = BismuthConfig
    _no_split_modules = ["HypCDMBlock", "TransformerBlock"]
    supports_gradient_checkpointing = True

    def __init__(self, config: BismuthConfig):
        super().__init__(config)
        # Build the internal CrystalFormerConfig dataclass
        inner_cfg = _CrystalFormerDataclass(
            model_name="Bismuth-104M",
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            cdm_ratio=config.cdm_ratio,
            d_cdm_hidden=config.d_cdm_hidden,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            tie_embeddings=config.tie_embeddings,
            curvature=config.curvature,
            use_moe=config.use_moe,
            moe_n_euclidean=config.moe_n_euclidean,
            moe_n_hyperbolic=config.moe_n_hyperbolic,
            gradient_checkpointing=config.gradient_checkpointing,
            curvature_schedule=config.curvature_schedule,
        )
        self.model = CrystalFormer(inner_cfg)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embedding

    def set_input_embeddings(self, value):
        self.model.embedding = value

    def get_output_embeddings(self):
        return self.model.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.model.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask=None,        # accepted but not used (causal mask is hardcoded)
        labels: torch.LongTensor = None,
        past_key_values=None,       # accepted for API compat; CDM states passed via v_states
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        v_states=None,              # CDM-specific: list of (B, d_h//2) complex tensors
        return_v_states=False,
        **kwargs,
    ):
        out = self.model(input_ids, v_states=v_states, return_v_states=return_v_states)
        logits = out["logits"]

        loss = None
        if labels is not None:
            # standard next-token CE; shift handled by caller (HF convention)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        if not return_dict:
            output = (logits,) + (out.get("v_states"),) if return_v_states else (logits,)
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(loss=loss, logits=logits)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=1.0, top_k=50, **kwargs):
        """Native generate (bypasses HF generate for CDM state threading)."""
        return self.model.generate(
            input_ids, max_new_tokens=max_new_tokens,
            temperature=temperature, top_k=top_k
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        return {"input_ids": input_ids}

    @classmethod
    def from_bismuth_checkpoint(cls, checkpoint_path: str, device="cpu"):
        """
        Load directly from a raw DuoNeural training checkpoint.

        ckpt = torch.load(path, weights_only=False)
        Keys: step, model_state_dict, optimizer_state_dict, config (CrystalFormerConfig)
        """
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        raw_cfg = ckpt["config"]

        hf_config = BismuthConfig(
            vocab_size=raw_cfg.vocab_size,
            d_model=raw_cfg.d_model,
            n_heads=raw_cfg.n_heads,
            n_layers=raw_cfg.n_layers,
            cdm_ratio=raw_cfg.cdm_ratio,
            d_cdm_hidden=raw_cfg.d_cdm_hidden,
            max_seq_len=raw_cfg.max_seq_len,
            dropout=raw_cfg.dropout,
            tie_embeddings=raw_cfg.tie_embeddings,
            curvature=raw_cfg.curvature,
            use_moe=raw_cfg.use_moe,
            curvature_schedule=raw_cfg.curvature_schedule,
        )
        model = cls(hf_config)
        model.model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        print(f"[Bismuth] loaded step={ckpt.get('step','?')}, "
              f"params={model.model.get_num_params()/1e6:.2f}M "
              f"({model.model.get_num_params(non_embedding=True)/1e6:.2f}M non-emb)")
        return model.to(device)


# ── Auto-registration ─────────────────────────────────────────────────────────

try:
    from transformers import AutoConfig, AutoModelForCausalLM
    AutoConfig.register("bismuth", BismuthConfig)
    AutoModelForCausalLM.register(BismuthConfig, BismuthForCausalLM)
except Exception:
    pass  # graceful if transformers version doesn't support register
