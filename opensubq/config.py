"""
SubQConfig — hyper-parameter dataclass for the SubQ architecture.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SubQConfig:
    """
    Configuration for the SubQ architecture.

    SubQ introduces **Subquadratic Sparse Attention (SSA)**, replacing the
    standard O(N²) self-attention with a union of three sparse patterns that
    together achieve O(N) complexity, enabling context windows of millions of
    tokens.

    SSA patterns
    ------------
    1. **Local window** – each token attends to its nearest ±``window_size``
       neighbours.  Cost: O(N · window_size).

    2. **Global tokens** – the leading ``num_global_tokens`` positions act as
       "sinks" that attend to and are attended from *every* token.
       Cost: O(N · num_global_tokens).

    3. **Content routing** – a lightweight low-rank scorer identifies the
       top-``top_k_sparse`` most semantically relevant key positions for each
       query.  Cost: O(N · top_k_sparse) with approximate nearest-neighbour
       search; O(N · routing_rank) to compute routing scores.

    References
    ----------
    - SubQ blog post: https://subq.ai/introducing-subq
    - BigBird (Zaheer et al., 2020): global + local + random sparse patterns.
    - Longformer (Beltagy et al., 2020): sliding window + global attention.
    - RoFormer (Su et al., 2022): Rotary Position Embeddings (RoPE).
    """

    # ------------------------------------------------------------------ #
    # Vocabulary / tokeniser
    # ------------------------------------------------------------------ #
    vocab_size: int = 50_257          # GPT-2 vocabulary by default

    # ------------------------------------------------------------------ #
    # Model dimensions
    # ------------------------------------------------------------------ #
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    head_dim: Optional[int] = None    # defaults to hidden_size // num_attention_heads
    intermediate_size: int = 3_072    # FFN inner dimension (typically 4 × hidden_size)

    # ------------------------------------------------------------------ #
    # SSA hyper-parameters
    # ------------------------------------------------------------------ #
    #  1. Local window attention
    window_size: int = 512            # each token attends ±window_size neighbours

    #  2. Global token attention
    num_global_tokens: int = 64       # leading positions act as global "sinks"

    #  3. Content-based sparse routing
    top_k_sparse: int = 128           # top-K keys attended per query via routing
    routing_rank: int = 16            # rank of low-rank routing projections

    # ------------------------------------------------------------------ #
    # Position encoding
    # ------------------------------------------------------------------ #
    max_position_embeddings: int = 12_000_000   # 12 M token design point
    rope_theta: float = 10_000.0                # RoPE base frequency

    # ------------------------------------------------------------------ #
    # Regularisation
    # ------------------------------------------------------------------ #
    dropout: float = 0.0
    attention_dropout: float = 0.0

    # ------------------------------------------------------------------ #
    # Normalisation / initialisation
    # ------------------------------------------------------------------ #
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.head_dim is None:
            if self.hidden_size % self.num_attention_heads != 0:
                raise ValueError(
                    f"hidden_size ({self.hidden_size}) must be divisible by "
                    f"num_attention_heads ({self.num_attention_heads})."
                )
            self.head_dim = self.hidden_size // self.num_attention_heads
