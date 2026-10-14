"""
configuration_bismuth.py — HuggingFace PretrainedConfig for Bismuth-104M (CrystalFormer v4)
DuoNeural Lab | Archon + Jesse Caldwell + Aura | 2026-10-14
"""

from transformers import PretrainedConfig

# FP64 transcendental curvature schedule (see hyp_cdm_transformer.py for derivation)
import math
_C1 = 0.8032997097470788161
_C2 = 0.8031337737999998949
_C4 = 0.8406556995239239938
_DEFAULT_CURVATURE_SCHEDULE = [
    1.0 / (_C4 ** 2),       # Block 0: ~1.4150  RI-upper boundary
    1.0 / math.log(2),      # Block 1: ~1.4427  Lyapunov triality
    1.0 / (_C2 ** 2),       # Block 2: ~1.5503  B11 boundary
    1.0 / (_C1 ** 2),       # Block 3: ~1.5497  LBDY triple point
]


class BismuthConfig(PretrainedConfig):
    """
    Configuration class for Bismuth-104M (CrystalFormer v4).

    Bismuth is the first published base autoregressive language model to use learnable
    hyperbolic curvature (Thomas-Wigner geometric connection scan) inside its core
    transformer stack at pretraining scale.

    Key architectural parameters:
        d_model (int): Hidden dimension. Default: 768.
        n_heads (int): Attention heads. Default: 12.
        n_layers (int): Total layers (transformer + CDM). Default: 12.
        cdm_ratio (int): Transformer blocks per CDM block. Default: 3.
            Pattern: [T T T C] x (n_layers // (cdm_ratio+1))
        d_cdm_hidden (int): CDM hidden dimension (complex channel pairs = d_cdm_hidden//2).
            Default: 384.
        tie_embeddings (bool): Tie lm_head.weight to embedding.weight.
            True -> 104.5M unique params; False -> 143M params.
        curvature_schedule (list[float]): Per-CDM-block curvature initialization values.
            Initialized at FP64 transcendental phase boundaries (DuoNeural constants).

    Example::
        from configuration_bismuth import BismuthConfig
        config = BismuthConfig()
        # or load from hub:
        config = BismuthConfig.from_pretrained("DuoNeural/bismuth-104m-base",
                                               trust_remote_code=True)
    """
    model_type = "bismuth"

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        cdm_ratio: int = 3,
        d_cdm_hidden: int = 384,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        tie_embeddings: bool = True,
        curvature: float = 1.4426950408889634,
        use_moe: bool = False,
        moe_n_euclidean: int = 3,
        moe_n_hyperbolic: int = 1,
        gradient_checkpointing: bool = False,
        curvature_schedule=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.cdm_ratio = cdm_ratio
        self.d_cdm_hidden = d_cdm_hidden
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.tie_embeddings = tie_embeddings
        self.curvature = curvature
        self.use_moe = use_moe
        self.moe_n_euclidean = moe_n_euclidean
        self.moe_n_hyperbolic = moe_n_hyperbolic
        self.gradient_checkpointing = gradient_checkpointing
        self.curvature_schedule = curvature_schedule or _DEFAULT_CURVATURE_SCHEDULE

    @property
    def n_cdm_blocks(self) -> int:
        return self.n_layers // (self.cdm_ratio + 1)
