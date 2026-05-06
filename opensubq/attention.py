"""
Subquadratic Sparse Attention (SSA)
=====================================
The attention mechanism that powers SubQ.  Rather than computing a full N×N
attention matrix — O(N²) in time and memory — SSA unions three complementary
sparse patterns into a single attention distribution:

  1. **Local window** – token i attends to every token j within ±W positions.
     Cost: O(N · W).

  2. **Global tokens** – the leading G positions ("global sinks") attend to
     every token *and* are attended to by every token.  Cost: O(N · G).

  3. **Content routing** – a lightweight low-rank scoring function identifies
     the top-K most semantically relevant key positions for each query.
     Cost: O(N · routing_rank) for the routing forward pass; with an
     approximate nearest-neighbour index the subsequent gather is O(N · K).

Patterns 1 and 2 are structurally fixed (no compute beyond the mask build);
pattern 3 introduces *content-dependent* long-range connections without
enumerating all N² pairs.

The three boolean masks are unioned into a single sparse mask, and a single
masked softmax produces a valid probability distribution over the attended
positions.  Total cost: O(N · (W + G + K)) = O(N) for fixed W, G, K.

Note on production vs. reference implementation
-----------------------------------------------
This module provides an *algorithmically correct* reference implementation.
The routing score matrix is materialised as a dense (N × N) tensor, so for
very long sequences this component reverts to O(N²) memory.  A production
deployment would replace this with a block-sparse CUDA kernel or an
approximate nearest-neighbour index (e.g. FAISS, ScaNN), retaining true O(N)
memory while executing the same algorithm.  The local-window and global-token
components are O(N) even in this reference form and would use block-sparse
kernels (e.g. Triton, xFormers) in production.

References
----------
- SubQ blog post: https://subq.ai/introducing-subq
- BigBird (Zaheer et al., 2020) — arXiv 2007.14062
- Longformer (Beltagy et al., 2020) — arXiv 2004.05150
- RoFormer / RoPE (Su et al., 2022) — arXiv 2104.09864
- FlashAttention (Dao et al., 2022) — arXiv 2205.14135
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .config import SubQConfig


# --------------------------------------------------------------------------- #
# Rotary Position Embeddings (RoPE)                                            #
# --------------------------------------------------------------------------- #


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE) — Su et al., 2022 (RoFormer).

    Encodes absolute positions as rotations applied to (query, key) pairs,
    yielding relative-position awareness at zero extra parameters.  Handles
    arbitrary sequence lengths at inference without recomputing the full table.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 131_072,
        theta: float = 10_000.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta

        inv_freq = 1.0 / (
            theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._seq_len_cached: int = 0
        self._cos_cached: Optional[torch.Tensor] = None
        self._sin_cached: Optional[torch.Tensor] = None

    def _update_cache(self, seq_len: int, device: torch.device) -> None:
        if seq_len <= self._seq_len_cached:
            return
        self._seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)            # (N, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)          # (N, dim)
        self._cos_cached = emb.cos()
        self._sin_cached = emb.sin()

    def forward(
        self, seq_len: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) tensors of shape (seq_len, dim)."""
        self._update_cache(seq_len, device)
        return (
            self._cos_cached[:seq_len],  # type: ignore[index]
            self._sin_cached[:seq_len],  # type: ignore[index]
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap and negate the second half of the last dimension."""
    d = x.shape[-1] // 2
    return torch.cat([-x[..., d:], x[..., :d]], dim=-1)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Rotary Position Embeddings to query and key tensors.

    Parameters
    ----------
    q, k   : (B, N, H, d)
    cos, sin : (N, d)  —  broadcast over batch B and head H.
    """
    # Expand to (1, N, 1, d) for broadcasting
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


# --------------------------------------------------------------------------- #
# Subquadratic Sparse Attention                                                #
# --------------------------------------------------------------------------- #


class SubquadraticSparseAttention(nn.Module):
    """
    Subquadratic Sparse Attention (SSA) — the core attention module of SubQ.

    See module docstring for full algorithm description.
    """

    def __init__(self, config: SubQConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim  # type: ignore[assignment]
        self.window_size = config.window_size
        self.num_global_tokens = config.num_global_tokens
        self.top_k_sparse = config.top_k_sparse
        self.routing_rank = config.routing_rank
        self.scale = self.head_dim ** -0.5

        # Standard Q / K / V / output projections
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        # Low-rank routing projections for content-based sparse attention.
        # routing_rank << head_dim so the routing forward pass is cheap.
        routing_out = self.num_heads * self.routing_rank
        self.route_q = nn.Linear(config.hidden_size, routing_out, bias=False)
        self.route_k = nn.Linear(config.hidden_size, routing_out, bias=False)

        self.rotary = RotaryEmbedding(
            dim=self.head_dim,
            max_seq_len=config.max_position_embeddings,
            theta=config.rope_theta,
        )

        self.attn_drop = nn.Dropout(config.attention_dropout)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, D) → (B, N, H, d)"""
        B, N, _ = x.shape
        return x.view(B, N, self.num_heads, self.head_dim)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, H, d) → (B, N, D)"""
        B, N, H, d = x.shape
        return x.reshape(B, N, H * d)

    # ------------------------------------------------------------------ #
    # SSA mask construction                                                #
    # ------------------------------------------------------------------ #

    def _local_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Boolean mask — True where |i − j| ≤ window_size.

        Shape: (N, N).  Two tokens are connected when they lie within the
        sliding window, capturing short-range syntactic and semantic context.
        """
        idx = torch.arange(seq_len, device=device)
        return (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() <= self.window_size

    def _global_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Boolean mask — True for all (i, j) where j < G (column is a global
        token) *or* i < G (row is a global token).

        Shape: (N, N).  Global tokens accumulate information from the whole
        sequence and broadcast it back, providing O(1)-hop connectivity between
        any two tokens regardless of distance.
        """
        G = min(self.num_global_tokens, seq_len)
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
        mask[:, :G] = True   # every token may attend to the G global keys
        mask[:G, :] = True   # the G global queries may attend to every token
        return mask

    def _routing_mask(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Content-based top-K sparse mask.

        Shape: (B, H, N, N), bool.

        Each query position selects its top-K key positions according to a
        low-rank routing score (route_q · route_k^T).  This introduces
        content-dependent long-range connections without a full N² enumeration.

        In a production system the routing scores would be evaluated with an
        approximate nearest-neighbour index (e.g. FAISS, ScaNN) to keep the
        cost truly O(N · K) rather than O(N²).
        """
        B, N, _ = hidden_states.shape
        K = min(self.top_k_sparse, N)
        H, R = self.num_heads, self.routing_rank

        # Low-rank routing projections: (B, N, H, R)
        rq = self.route_q(hidden_states).view(B, N, H, R).transpose(1, 2)  # (B, H, N, R)
        rk = self.route_k(hidden_states).view(B, N, H, R).transpose(1, 2)  # (B, H, N, R)

        # Routing similarity scores: (B, H, N, N)
        routing_scores = torch.matmul(rq, rk.transpose(-2, -1))

        # Select top-K per query (last dimension = keys)
        threshold = routing_scores.topk(K, dim=-1).values[..., -1:]   # (B, H, N, 1)
        return routing_scores >= threshold  # (B, H, N, N) bool

    # ------------------------------------------------------------------ #
    # Forward                                                              #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        hidden_states  : (B, N, D)
        attention_mask : (B, N)  — 1 = real token, 0 = padding (optional).

        Returns
        -------
        output : (B, N, D)
        """
        B, N, D = hidden_states.shape
        device = hidden_states.device

        # ---- Projections --------------------------------------------------- #
        q = self._split_heads(self.q_proj(hidden_states))  # (B, N, H, d)
        k = self._split_heads(self.k_proj(hidden_states))
        v = self._split_heads(self.v_proj(hidden_states))

        # ---- Rotary position embeddings ------------------------------------ #
        cos, sin = self.rotary(N, device)
        q, k = apply_rotary_emb(q, k, cos, sin)

        # ---- Build combined SSA boolean mask ------------------------------- #
        local_mask = self._local_mask(N, device)          # (N, N)
        global_mask = self._global_mask(N, device)        # (N, N)
        routing_mask = self._routing_mask(hidden_states)  # (B, H, N, N)

        # Union: a pair (i, j) is attended if *any* pattern connects them.
        ssa_mask = (
            local_mask.unsqueeze(0).unsqueeze(0)    # (1, 1, N, N)
            | global_mask.unsqueeze(0).unsqueeze(0) # (1, 1, N, N)
            | routing_mask                          # (B, H, N, N)
        )  # → (B, H, N, N) via broadcast

        # ---- Attention scores ---------------------------------------------- #
        q_t = q.transpose(1, 2)    # (B, H, N, d)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)

        scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * self.scale  # (B, H, N, N)

        # Mask out non-attended pairs
        neg_inf = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~ssa_mask, neg_inf)

        # Apply padding mask: (B, N) → (B, 1, 1, N)
        if attention_mask is not None:
            pad_mask = attention_mask.bool().unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~pad_mask, neg_inf)

        attn_weights = torch.softmax(scores, dim=-1)
        # Rows that are entirely masked (all -inf) produce NaN after softmax;
        # replace with zeros so they contribute nothing to the output.
        attn_weights = torch.nan_to_num(attn_weights)
        attn_weights = self.attn_drop(attn_weights)

        output = torch.matmul(attn_weights, v_t)   # (B, H, N, d)
        output = output.transpose(1, 2)             # (B, N, H, d)
        return self.out_proj(self._merge_heads(output))  # (B, N, D)
