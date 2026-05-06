# opensubq

A theoretical reconstruction of the SubQ architecture, built from first
principles using the available research literature.

> **Source**: [Subquadratic — Introducing SubQ](https://subq.ai/introducing-subq)

---

## Background

Standard transformer self-attention computes a dot-product similarity between
every pair of tokens, yielding an N × N attention matrix.  Memory and compute
therefore scale as **O(N²)**, making long sequences prohibitively expensive.
State-of-the-art models are typically capped at 128 K–1 M tokens in practice
before quality degrades or costs become unacceptable.

**SubQ** (by Subquadratic) breaks this bottleneck with what they call
*Subquadratic Sparse Attention* (SSA).  Rather than a dense attention matrix,
SSA computes attention only over a carefully chosen sparse set of token pairs,
achieving **O(N)** time and memory complexity for fixed hyper-parameters.
This enables a 12-million-token context window at roughly 1/5 the cost of
comparable dense-attention models, with no chunking or summarisation of the
context.

---

## Architecture

### Subquadratic Sparse Attention (SSA)

SSA replaces the O(N²) self-attention with a **union of three sparse patterns**
that together preserve long-range expressiveness at linear cost:

```
SSA_mask[i, j] = local_mask[i, j]    # 1. local window
               | global_mask[i, j]   # 2. global token
               | routing_mask[i, j]  # 3. content routing
```

A single softmax over the unioned sparse scores produces a valid probability
distribution over attended positions.

#### 1 · Local Window Attention  —  O(N · W)

Each token attends to its nearest ±`window_size` neighbours (default: 512).
Captures short-range syntactic and semantic patterns.

```
local_mask[i, j] = 1  iff  |i − j| ≤ window_size
```

#### 2 · Global Token Attention  —  O(N · G)

The leading `num_global_tokens` positions (default: 64) act as **global sinks**:

* They attend to *every* token in the sequence.
* Every token attends to *them*.

This gives O(1)-hop connectivity between any two positions regardless of
distance — all information can flow through the globals in two steps.

```
global_mask[i, j] = 1  iff  j < G  (all tokens → global keys)
                  | 1  iff  i < G  (global queries → all tokens)
```

#### 3 · Content-Based Sparse Routing  —  O(N · K)

A lightweight low-rank scorer (`routing_rank=16` by default) computes a
similarity between every (query, key) pair using cheap low-dimensional
projections, then selects the top-`top_k_sparse` keys per query:

```
routing_scores[i, j] = route_q(h[i]) · route_k(h[j])^T
routing_mask[i, :]   = top-K positions of routing_scores[i, :]
```

This introduces **content-dependent long-range connections** without
enumerating all N² pairs.  In a production deployment the top-K selection
would be computed with an approximate nearest-neighbour index (FAISS, ScaNN)
for true O(N · K) cost; in this reference implementation the routing scores
are materialised densely for algorithmic clarity.

#### Complexity summary

| Component         | Time         | Memory    |
|-------------------|-------------|-----------|
| Local window      | O(N · W)    | O(N · W)  |
| Global tokens     | O(N · G)    | O(N · G)  |
| Content routing   | O(N · K)    | O(N · K)  |
| **SSA total**     | **O(N)**    | **O(N)**  |

W, G, K are fixed hyper-parameters independent of N.

### Position encoding: RoPE

[Rotary Position Embeddings](https://arxiv.org/abs/2104.09864) are applied to
Q and K before the attention computation.  RoPE encodes absolute positions as
rotations that cancel out to relative-position information in the dot-product,
and handles arbitrarily long sequences without interpolation.

### Feed-forward network: SwiGLU

Each transformer block uses a
[SwiGLU](https://arxiv.org/abs/2002.05202) MLP:

```
FFN(x) = down_proj( Swish(gate_proj(x)) ⊙ up_proj(x) )
```

SwiGLU empirically outperforms ReLU and GeLU at matched parameter budgets and
is the activation used in LLaMA, PaLM, and related frontier models.

### Normalisation: RMSNorm + pre-norm layout

[Root Mean Square Layer Normalisation](https://arxiv.org/abs/1910.07467) is
applied before each sub-layer (pre-norm), following the LLaMA / Mistral
convention for training stability.

### Full architecture diagram

```
input_ids (B, N)
    │
    ▼
embed_tokens                          ← nn.Embedding (V, D)
    │
    ▼  ×  num_hidden_layers
┌─────────────────────────────────────────────────────┐
│  SubQTransformerLayer                               │
│  ┌────────────────────────────────────────────────┐ │
│  │ SubQRMSNorm                                    │ │
│  │   ↓                                            │ │
│  │ SubquadraticSparseAttention (SSA)              │ │
│  │   ├─ Q/K/V projections                         │ │
│  │   ├─ RoPE                                      │ │
│  │   ├─ SSA mask  (local ∪ global ∪ routing)      │ │
│  │   └─ masked softmax → value weighted sum       │ │
│  │   ↓                                            │ │
│  │ residual +                                     │ │
│  └────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────┐ │
│  │ SubQRMSNorm                                    │ │
│  │   ↓                                            │ │
│  │ SubQMLP  (SwiGLU)                              │ │
│  │   ↓                                            │ │
│  │ residual +                                     │ │
│  └────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
    │
    ▼
SubQRMSNorm
    │
    ▼
lm_head                               ← nn.Linear (D, V)
    │
    ▼
logits (B, N, V)
```

---

## Default hyper-parameters

| Parameter               | Default        | Description                        |
|-------------------------|---------------|------------------------------------|
| `vocab_size`            | 50 257        | GPT-2 vocabulary                   |
| `hidden_size`           | 768           | Token embedding / hidden dimension |
| `num_hidden_layers`     | 12            | Number of transformer blocks       |
| `num_attention_heads`   | 12            | Attention heads                    |
| `intermediate_size`     | 3 072         | MLP inner dimension                |
| `window_size`           | 512           | Local attention half-width         |
| `num_global_tokens`     | 64            | Number of global-sink tokens       |
| `top_k_sparse`          | 128           | Top-K content routing connections  |
| `routing_rank`          | 16            | Rank of routing projections        |
| `max_position_embeddings` | 12 000 000  | RoPE cache size (12 M tokens)      |
| `rope_theta`            | 10 000.0      | RoPE base frequency                |
| `rms_norm_eps`          | 1e-6          | RMSNorm numerical stability term   |

---

## Install

```bash
pip install -e ".[dev]"   # from the repo root (editable + test deps)
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.2.

---

## Quick start

```python
import torch
from opensubq import SubQConfig, SubQModel

# Small model for experimentation
config = SubQConfig(
    vocab_size=50_257,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3_072,
    window_size=512,
    num_global_tokens=64,
    top_k_sparse=128,
)

model = SubQModel(config).eval()
print(f"Parameters: {model.num_parameters():,}")

# Forward pass
input_ids = torch.randint(0, config.vocab_size, (1, 1024))
with torch.no_grad():
    logits = model(input_ids)   # (1, 1024, 50257)

print(logits.shape)
```

---

## Tests

```bash
pytest tests/ -v
```

---

## Disclaimer

This repository is an **independent theoretical reconstruction** of the SubQ
architecture built from publicly available information and the research
literature cited below.  It is not affiliated with, endorsed by, or based on
proprietary code from Subquadratic.  The implementation captures the *design
principles* of SSA (local window + global tokens + content routing) as
described in the company's public blog post.

---

## References

| Paper / resource | Relevance |
|---|---|
| [Subquadratic — Introducing SubQ](https://subq.ai/introducing-subq) | Primary source for SSA design goals and benchmarks |
| [BigBird (Zaheer et al., 2020)](https://arxiv.org/abs/2007.14062) | Local + global + random sparse attention; theoretical foundations |
| [Longformer (Beltagy et al., 2020)](https://arxiv.org/abs/2004.05150) | Sliding window + global attention for long documents |
| [RoFormer / RoPE (Su et al., 2022)](https://arxiv.org/abs/2104.09864) | Rotary Position Embeddings |
| [GLU Variants / SwiGLU (Shazeer, 2020)](https://arxiv.org/abs/2002.05202) | Gated linear units; SwiGLU activation |
| [RMSNorm (Zhang & Sennrich, 2019)](https://arxiv.org/abs/1910.07467) | Root Mean Square normalisation |
| [FlashAttention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135) | Memory-efficient exact attention (production baseline) |
| [Efficient Transformers Survey (Tay et al., 2020)](https://arxiv.org/abs/2009.06732) | Survey of sub-quadratic attention approaches |
