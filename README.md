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
| `num_attention_heads`   | 12            | Attention heads (query)            |
| `num_key_value_heads`   | same as Q     | KV heads — set < Q heads for GQA  |
| `intermediate_size`     | 3 072         | FFN inner dim (per expert for MoE) |
| `num_experts`           | `None`        | MoE expert count; `None` = dense   |
| `num_experts_per_tok`   | 1             | Active experts per token (top-K)   |
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
pip install -e ".[dev]"          # editable install + test deps (no extra deps)
pip install -e ".[dev,train]"    # also installs tiktoken for GPT-2/4 BPE datasets
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.2.

---

## Model Scales

The repository ships two named `SubQConfig` presets that target distinct
deployment tiers.

### Tier 1 — Mistral 7B scale  *(single A100 80 GB)*

Matches the hyper-parameters of Mistral 7B: 32-layer decoder, hidden size
4 096, SwiGLU FFN with inner dim 14 336.  Uses **Grouped Query Attention**
(32 Q heads / 8 KV heads) to halve the KV-cache footprint compared to
standard MHA.  SubQ's SSA replaces Mistral's fixed sliding-window attention,
adding global-sink tokens and content routing on top of the local window.

```python
from opensubq import SubQConfig, SubQModel

config = SubQConfig.mistral_7b()
# hidden_size=4096, 32 layers, 32 Q / 8 KV heads, dense SwiGLU FFN
# vocab_size=32_000  (Mistral tokeniser)
print(config)
```

### Tier 2 — MiMo-V2-Flash scale  *(multi-GPU cluster)*

Matches the backbone dimensions of MiMo-V2-Flash (Xiaomi, 2025): 48-layer
decoder, hidden size 7 168, 64 Q heads / 8 KV heads.  The dense FFN is
replaced by a **256-expert Sparse MoE** (8 experts active per token via
top-K routing), giving ~15 B active parameters per forward pass out of ~309 B
total.  SubQ's SSA is applied uniformly to every layer, providing the same
O(1)-hop global connectivity as MiMo's interleaved full-attention layers but
at linear cost.

```python
from opensubq import SubQConfig, SubQModel

config = SubQConfig.mimo_v2_flash()
# hidden_size=7168, 48 layers, 64 Q / 8 KV heads
# 256 experts / 8 active  (SparseMoEMLP per layer)
# vocab_size=152_064  (Qwen3 tokeniser)
print(config)
```

### Parameter summary

| Preset | Scale | Layers | Hidden | Q / KV heads | FFN | Vocab |
|---|---|---|---|---|---|---|
| `SubQConfig.mistral_7b()` | ~7 B | 32 | 4 096 | 32 / 8 | Dense SwiGLU | 32 000 |
| `SubQConfig.mimo_v2_flash()` | ~15 B active / 309 B total | 48 | 7 168 | 64 / 8 | Sparse MoE 256 ×, top-8 | 152 064 |

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

## Data & Training

### Data pipeline (`opensubq/data.py`)

| Class / helper | Description |
|---|---|
| `CharDataset` | Byte-level (0–255) tokenisation; no extra deps; matches `vocab_size=256` tiny config |
| `TiktokenDataset` | GPT-2 / GPT-4 BPE via `tiktoken`; requires `pip install tiktoken` |
| `make_synthetic_datasets()` | Reproducible random-token corpus for tests and quick demos |
| `make_split_loaders()` | Returns a `(train_loader, val_loader)` pair |

Both dataset classes produce `(input_ids, labels)` tensors with the autoregressive shift baked in and compatible with `SubQModel.forward(input_ids, labels=labels)`.

```python
from opensubq.data import CharDataset, make_split_loaders

# From a plain-text file (byte-level tokenisation):
train_ds, val_ds = CharDataset.from_file("corpus.txt", seq_len=1024)
train_loader, val_loader = make_split_loaders(train_ds, val_ds, batch_size=8)

# Or use GPT-2 BPE (requires tiktoken):
from opensubq.data import TiktokenDataset
train_ds, val_ds = TiktokenDataset.from_file("corpus.txt", seq_len=1024, encoding="gpt2")
```

### Training loop (`train.py`)

A ready-to-run training script at the repo root.  Features:

- `torch.autocast` mixed-precision — **bfloat16** on CUDA, float32 on CPU
- **AdamW** with cosine LR schedule and linear warmup (`--warmup-frac`)
- Gradient clipping (`--grad-clip`, default 1.0)
- Checkpoint save / resume (`--checkpoint-dir`, `--resume`)
- Eval loss on held-out val split, optional CSV loss log (`--log-file`)

```bash
# Sanity-check: tiny model, synthetic data, CPU, ~5 s:
python train.py --preset tiny --data synthetic --max-steps 100

# Tier-1 training on a real corpus:
python train.py \
    --preset mistral_7b \
    --data file --data-file corpus.txt \
    --seq-len 4096 --batch-size 4 \
    --max-steps 100000 \
    --checkpoint-dir ./ckpts \
    --log-file loss.csv

python train.py --help   # full option list
```

**Presets:** `tiny` (64-dim, 2L, vocab 256), `mistral_7b`, `mimo_v2_flash`.

---

## Tests

```bash
pytest tests/ -v
```

109 tests across attention, model, data pipeline, and training loop.
`TiktokenDataset` tests are automatically skipped when `tiktoken` is not installed.

---

## Roadmap

Progress against the [whitepaper §8 Recommended Roadmap](WHITEPAPER.md#8-recommended-roadmap):

### Phase 1 — Make the reference implementation trainable ✅ Complete

- [x] Add causal mask to SSA forward
- [x] Add autoregressive loss computation
- [x] Wire in a tokeniser + small training corpus (`opensubq/data.py`)
- [x] Write a minimal training loop with bfloat16, AdamW, checkpointing (`train.py`)
- [x] Train a tiny sanity-check model to verify loss decreases

### Phase 2 — Scale to Tier 1 (7 B, single A100)

- [ ] Add `torch.compile` and bfloat16 inference
- [ ] Integrate FlashAttention-2 for the local-window component
- [ ] Add KV-cache for autoregressive decoding
- [ ] Train `SubQConfig.mistral_7b()` on a mid-scale dataset

### Phase 3 — Scale to Tier 2 (MiMo-V2-Flash, cluster)

- [ ] Replace sequential expert dispatch with batched GEMM / megablocks
- [ ] Add auxiliary load-balancing loss
- [ ] Integrate an expert-parallel training framework
- [ ] Add tensor + pipeline parallelism for 309 B total weight distribution
- [ ] Train `SubQConfig.mimo_v2_flash()` on a large-scale dataset

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
| [GQA (Ainslie et al., 2023)](https://arxiv.org/abs/2305.13245) | Grouped Query Attention — fewer KV heads than Q heads |
| [MiMo-V2-Flash (Xiaomi, 2025)](https://arxiv.org/abs/2601.02780) | 309B MoE model; inspiration for the `mimo_v2_flash` scale preset |
| [Mistral 7B (Jiang et al., 2023)](https://arxiv.org/abs/2310.06825) | Dense 7B baseline; inspiration for the `mistral_7b` scale preset |
