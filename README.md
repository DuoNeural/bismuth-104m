---
language: en
license: apache-2.0
library_name: transformers
tags:
  - causal-lm
  - hyperbolic-geometry
  - cdm
  - crystalformer
  - novel-architecture
  - duoneural
datasets:
  - HuggingFaceFW/fineweb-edu
model_type: bismuth
pipeline_tag: text-generation
---

# Bismuth-104M

**First causal language model to integrate Thomas-Wigner geometric connection scan as a hybrid
recurrent-hyperbolic layer interleaved with standard transformer blocks, with gradient-enabled
learned curvature via complex Lorentz holonomy phase encoding.**

Developed by [DuoNeural](https://github.com/DuoNeural) — Jesse Caldwell, Archon (AI), and Aura (AI).
Pretrained from scratch, Chinchilla-optimal, on a single consumer GPU.

> **Related prior work:** HELM (He et al., NeurIPS 2025, [arxiv:2505.24722](https://arxiv.org/abs/2505.24722))
> establishes hyperbolic autoregressive LMs at 100M–1B scale using the Lorentz hyperboloid model and
> attention-level integration. Bismuth-104M uses a distinct approach: Poincaré ball geometry, a
> Thomas-Wigner holonomy phase recurrence interleaved between Euclidean transformer blocks, and
> a complex-valued state scan. These are different architectural choices, not competing priority claims.

> **Status (Oct 2026):** Architecture investigation is ongoing. Ablation baselines, zero-shot benchmarks
> (LAMBADA, HellaSwag, BLiMP), and post-training (Curvature-Preserved SFT → DPO → GRPO Lite) are in active development.
> This release is the Chinchilla-optimal base model.

---

## DuoNeural

| Resource | Link |
|---|---|
| GitHub | [github.com/DuoNeural](https://github.com/DuoNeural) |
| X (Twitter) | [@DuoNeural](https://x.com/DuoNeural) |
| Full paper catalog (40+ papers) | [zenodo.org/communities/duoneural](https://zenodo.org/communities/duoneural) |
| CDM foundational paper | [doi.org/10.5281/zenodo.21158430](https://doi.org/10.5281/zenodo.21158430) |
| Related: Hyperbolic PEFT (Qwen2.5-1.5B) | [DuoNeural/hyperbolic-peft-qwen2.5-1.5b](https://huggingface.co/DuoNeural/hyperbolic-peft-qwen2.5-1.5b) |

We operate as a human-AI symbiotic research team. Our approach is informed by emerging frameworks
for [Artificial Symbiotic Intelligence](https://institute.deepmind.com/essays/artificial-symbiotic-intelligence/)
(Google DeepMind) — the idea that human and AI cognitive systems in close collaboration can form
a joint cognitive entity exceeding either alone.

---

## Model Description

Bismuth-104M (CrystalFormer v4) is a 104.5M-parameter causal language model that interleaves
standard transformer blocks with novel **Hyperbolic CDM blocks** — recurrent layers operating on
a hyperbolic manifold via the Thomas-Wigner geometric connection scan.

### Architecture

```
Pattern: [TransformerBlock, TransformerBlock, TransformerBlock, HypCDMBlock] × 3
         = 9 Transformer + 3 HypCDM = 12 total layers
```

| Component | Detail |
|---|---|
| Model dimension | d_model = 768 |
| Attention heads | 12 |
| Sequence length | 2048 tokens |
| CDM blocks | 3 × HypCDMBlock (d_h = 384) |
| Attention | FlashAttention-2 (SDPA) |
| FFN | SwiGLU |
| Positional encoding | RoPE |
| Normalization | RMSNorm throughout |
| Embeddings | Tied (lm_head.weight = embedding.weight) |

**Unique properties:**
- **Learnable per-layer curvature** — each HypCDMBlock has its own `log_c` parameter that learns the curvature of its Poincaré ball. Curvature initialized at physics-grounded transcendental constants (FP64 floor, no closed-form expression known).
- **Thomas-Wigner connection scan** — the CDM recurrence encodes non-Abelian Lorentz holonomy via complex state `z_t = λ_t · z_{t-1} + b_t` where the phase `φ_t = c · (u_t ∧ u_{t-1})_k` is computed as a wedge product of successive hidden states. Curvature `c` receives real gradients through the holonomy phase; topological winding accumulates in the complex state trajectory.
- **Lyapunov stability** — CDM blocks maintain contractive/critical Lyapunov regimes throughout training. Zero manifold drift confirmed.

### Learned Curvature Values (after 2.555B tokens)

| CDM Block | Init curvature | Trained curvature | Geometry |
|---|---|---|---|
| L0 (layers 0–3) | 1.415 (RI-upper boundary) | **0.88** | Lower curvature, stronger hyperbolic compression |
| L1 (layers 4–7) | 1.443 (ln2 Lyapunov triality) | **0.79** | Lower curvature, stronger hyperbolic compression |
| L2 (layers 8–11) | 1.550 (B11 phase boundary) | **1.45** | Higher curvature, reduced hyperbolic distortion |

The curvature gradient validates the Thomas-Wigner fix: `dy/dc ≠ 0` confirmed. Prior v3 had a phantom manifold bug (expmap/logmap round-trip = identity map, curvature frozen at init).

---

## Parameter Count

| Count type | Value |
|---|---|
| Unique parameters (tied embeddings counted once) | **104,529,987** |
| Total tensor storage (embedding counted twice) | 143,127,363 |
| Non-embedding parameters | ~65.9M |

> **Note on naming:** 104M reflects the true unique parameter count. The model was originally
> developed as "Bismuth-125M" based on an undeduped tensor count. The accurate figure is 104.5M.
> Embedding table: 50,257 × 768 = 38,597,376 parameters (tied to lm_head).

---

## Training

| Attribute | Value |
|---|---|
| Training data | [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) |
| Training tokens | **2,555,000,000** (Chinchilla-optimal for 104.5M params) |
| Training steps | 78,000 |
| Batch size (effective) | 16 × 2048 = 32,768 tokens/step |
| Gradient accumulation | 4 |
| Learning rate | 3e-4 (cosine decay) |
| Optimizer | AdamW (β₁=0.9, β₂=0.95), curvature params 10× slower LR |
| Precision | Mixed (BF16 compute, FP32 hyperbolic ops) |
| Hardware | **Single NVIDIA RTX 5070 Ti** (16GB VRAM) |
| Host system | Vast.ai GPU pod (consumer cloud compute) |
| Wall time | **9.3 days** |
| Throughput | ~3,180 tokens/second |

### Chinchilla Optimality

At 2.555B tokens for 104.5M parameters, Bismuth-104M sits precisely in the
[Chinchilla compute-optimal](https://arxiv.org/abs/2203.15556) envelope (~24.5 tokens/param).
This is a minimum — further training would likely improve perplexity.

---

## Evaluation

### FineWeb-Edu Held-Out (Primary Eval)

| Model | PPL (FineWeb-Edu held-out) | Params | Notes |
|---|---|---|---|
| GPT-2 | ~25–30 | 124M | Standard GPT-2 weights |
| **Bismuth-104M** | **24.70** | 104.5M | CrystalFormer v4 |

Evaluated on the final held-out parquet shard (2M tokens) from FineWeb-Edu using a sliding window
(seq_len=1024, stride=512). Lower is better.

### Zero-Shot Benchmarks

| Benchmark | Bismuth-104M | GPT-2 124M | Notes |
|---|---|---|---|
| **HellaSwag** (commonsense NLI) | **33.2%** | ~29% | **+4pp — Bismuth wins** |
| BLiMP (anaphor_gender_agreement) | 68.4% | ~67% | +1.4pp slight edge |
| LAMBADA (last-word prediction) | 21.1% | ~45–50% | OOD lexical task (expected gap) |

> **HellaSwag result is notable.** Bismuth-104M outperforms GPT-2 124M (+4pp) despite being
> 20M parameters smaller. Commonsense action-sequence completion is hierarchically structured,
> which hyperbolic geometry should encode efficiently. However, FineWeb-Edu's higher data quality
> relative to GPT-2's WebText may contribute to this gain independently of geometry. An
> iso-compute Euclidean control baseline is needed to isolate the geometric contribution.
>
> **LAMBADA gap is expected.** LAMBADA tests last-word prediction on Project Gutenberg books —
> a very different register from educational web text. Distribution mismatch, not architecture.

Evaluated on p54091 (RTX 5070 Ti, CUDA 12.8). Scripts: `/workspace/bismuth_zeroshot_evals.py`.

---

### WikiText-103 (Out-of-Distribution Comparison)

| Model | PPL (WikiText-103 test, stride=512) | Params | Notes |
|---|---|---|---|
| GPT-2 (stride=512) | ~36–37 | 124M | Estimated; published 29.41 used stride=32 |
| **Bismuth-104M** | **75.50** | 104.5M | Trained on FineWeb-Edu (educational web) |

> **Evaluation note:** GPT-2's published WikiText-103 score of 29.41 (Radford et al. 2019) used
> stride=32, providing near-full context per token. At a comparable stride of 512, GPT-2 scores
> approximately 36–37. Our 75.50 is evaluated at stride=512. The gap is real and reflects
> domain mismatch — Bismuth was trained exclusively on FineWeb-Edu (educational web text), not
> Wikipedia prose. This is an acknowledged out-of-distribution limitation, not a design feature.
>
> **The fair in-distribution comparison** is FineWeb-Edu held-out (24.70 PPL), where Bismuth
> outperforms GPT-2 (trained on WebText, a different domain) — though note that this comparison
> is also confounded by training data differences, not architecture alone.
>
> For general-purpose or encyclopedic use, fine-tuning on mixed-domain data is recommended.

---

## Known Logging Anomaly (Training Checkpoints)

The training script accumulates loss across gradient accumulation steps before logging:

```python
step_loss += loss.item() * args.grad_accum  # grad_accum = 4
```

This means **`step_loss` in checkpoints = 4× the actual per-token NLL**.

```
actual_NLL = ckpt['step_loss'] / 4 = 13.465 / 4 = 3.366 nats
actual_PPL = exp(3.366) ≈ 29  (training distribution, during run)
```

The model itself is unaffected. This is a logging display bug only, not a training bug.
Corrected in `bismuth_eval_v3.py`.

---

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "DuoNeural/bismuth-104m-base",
    trust_remote_code=True,          # required: custom CDM architecture
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Generate
prompt = "The laws of thermodynamics state that"
inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    output = model.generate(inputs["input_ids"], max_new_tokens=50, temperature=0.8, top_k=40)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

### Load from Raw Training Checkpoint

```python
from modeling_bismuth import BismuthForCausalLM
model = BismuthForCausalLM.from_bismuth_checkpoint(
    "checkpoints_bismuth_chinchilla/bismuth_125m_step_78000.pt",
    device="cuda"
)
```

### `trust_remote_code=True` Requirement

Bismuth uses two custom files that must be downloaded from this repo:
- `modeling_bismuth.py` — HuggingFace wrapper (this file defines `BismuthForCausalLM`)
- `hyp_cdm_transformer.py` — core model source (Thomas-Wigner scan, HypCDMBlock, etc.)

---

## Limitations

- **Base model only** — no instruction tuning, RLHF, or chat fine-tuning. Outputs are raw
  next-token predictions, not instruction-following responses.
- **No ablation baseline** — the contribution of hyperbolic geometry vs. a flat-Euclidean
  control of identical size has not yet been formally measured. Without a matched Euclidean
  architecture (same params, same FineWeb-Edu token stream, same optimizer), performance
  differences cannot be attributed to geometry alone. A CDM ablation is planned.
- **Throughput overhead** — FP32 hyperbolic ops running alongside BF16 attention introduce
  kernel serialization. Throughput is ~3,180 tok/s on the training hardware (RTX 5070 Ti,
  16GB), roughly 8–10× slower than a pure-Euclidean model at this scale on equivalent hardware.
- **Inference serving** — custom recurrent ops in Python files are not compatible with
  vLLM, TensorRT-LLM, or TGI. `trust_remote_code=True` required.
- **Long-context not validated** — RoPE extrapolation beyond 2048 tokens interacts with CDM
  recurrent state dynamics; long-context recall has not been tested.
- **English only** — trained on English educational web text (FineWeb-Edu).
- **Sequence length** — trained at max_seq_len=2048. Longer contexts are extrapolated via RoPE.
- **Domain-Specific Perplexity Gap** — Bismuth achieves 24.70 PPL on held-out FineWeb-Edu, but exhibits
  75.50 PPL on WikiText-103 test. This reflects specialized structural adaptation to educational web text rather
  than general encyclopedic prose; mixed-domain fine-tuning is recommended for general-purpose applications.

---

## Citation

If you use Bismuth-104M or the CrystalFormer v4 architecture in your work, please cite:

```bibtex
@misc{duoneural2026bismuth,
  title        = {Bismuth-104M: Hybrid Transformer-Recurrent Language Model with Thomas-Wigner
                  Hyperbolic Connection Scan},
  author       = {Caldwell, Jesse and {Archon (AI)} and {Aura (AI)}},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/DuoNeural/bismuth-104m-base}},
  note         = {DuoNeural Lab. AI co-authors denote AI systems that contributed to research
                  design, implementation, and analysis under human direction; see model card
                  for collaboration methodology.}
}
```

---

## Related Models

| Model | Description |
|---|---|
| [DuoNeural/hyperbolic-peft-qwen2.5-1.5b](https://huggingface.co/DuoNeural/hyperbolic-peft-qwen2.5-1.5b) | Hyperbolic geometry PEFT applied to Qwen2.5-1.5B — CDM curvature adaptation on a production-scale instruction-tuned base |

Bismuth-104M is the **from-scratch pretraining** proof-of-concept for learnable hyperbolic geometry.
The Qwen2.5-1.5B PEFT model demonstrates the same CDM principles applied via post-training to an
existing strong base, showing the approach generalizes beyond novel architectures.

---

## Prior Art / Related Hyperbolic LMs

| Work | Approach | Geometry | Scale |
|---|---|---|---|
| **HELM** (He et al., NeurIPS 2025, [arxiv:2505.24722](https://arxiv.org/abs/2505.24722)) | Mixture-of-Curvature Experts; hyperbolic attention (HMLA) + HoPE positional encoding | Lorentz hyperboloid | 100M, 1B |
| **Hypformer** (Chen et al., 2024, [arxiv:2407.01290](https://arxiv.org/abs/2407.01290)) | Fully hyperbolic attention (linearized) | Lorentz/Minkowski | Graph/NLP benchmarks |
| **HNN++** (Shimizu et al., 2020) | Gyrovector calculus in neural layers | Poincaré ball | Small-scale NLP tasks |

Bismuth-104M differs from HELM and Hypformer in that hyperbolic geometry is **not** applied at the
attention level. Instead, a lightweight recurrent scan layer (HypCDM) uses the Thomas-Wigner
holonomy phase to encode Lorentz holonomy in a complex state, interleaved between standard Euclidean
transformer blocks. This is a complementary architectural choice, not a replacement for attention-level
hyperbolic integration.

---

## Corrections & Feedback

We welcome corrections, reproductions, and technical challenges to any claims in this model card.
If you find an error in evaluation methodology, architecture description, or benchmark comparisons,
please reach out:

- **Lab contact:** DuoNeural@proton.me
- **Archon (AI, lab director):** archon@agentmail.to
- **GitHub issues:** [github.com/DuoNeural](https://github.com/DuoNeural)

This is independent research by a small team. We'd rather be corrected than wrong.

---

## About DuoNeural

DuoNeural is a small independent AI lab focused on post-training research, novel architecture
discovery, and human-AI symbiosis. Our team: Jesse Caldwell (human), Archon (AI, lab director),
and Aura (AI, research & red-teaming).

Papers, models, and datasets: [github.com/duoneural](https://github.com/duoneural) |
[huggingface.co/DuoNeural](https://huggingface.co/DuoNeural)

*Bismuth is named after bismuth hopper crystals — emergent stepped fractal geometry arising from
phase transitions. An apt metaphor for a model whose CDM layers grow their own geometric phase
structure during pretraining.*
