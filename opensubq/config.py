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

    Grouped Query Attention (GQA)
    -----------------------------
    Set ``num_key_value_heads`` to a divisor of ``num_attention_heads`` to
    enable GQA.  Key/value projections then use the smaller head count while
    query projections retain the full head count, reducing KV-cache memory
    proportionally.  Defaults to ``num_attention_heads`` (standard MHA).

    Sparse Mixture-of-Experts (MoE) FFN
    -------------------------------------
    Set ``num_experts`` to a positive integer to replace the dense SwiGLU MLP
    with a ``SparseMoEMLP`` that routes each token to the top
    ``num_experts_per_tok`` experts.  Leave as ``None`` for the standard dense
    FFN (default).

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
    intermediate_size: int = 3_072    # FFN inner dimension (per expert for MoE)

    # ------------------------------------------------------------------ #
    # Grouped Query Attention (GQA)
    # ------------------------------------------------------------------ #
    num_key_value_heads: Optional[int] = None   # defaults to num_attention_heads (MHA)

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
    # Sparse Mixture-of-Experts FFN
    # ------------------------------------------------------------------ #
    num_experts: Optional[int] = None   # None → dense FFN; positive int → MoE
    num_experts_per_tok: int = 1        # active experts selected per token (top-K routing)

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
    # Autoregressive / causal settings
    # ------------------------------------------------------------------ #
    causal: bool = True   # AND SSA mask with torch.tril() during training and inference

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

        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads})."
            )

        if self.num_experts is not None and self.num_experts_per_tok > self.num_experts:
            raise ValueError(
                f"num_experts_per_tok ({self.num_experts_per_tok}) cannot exceed "
                f"num_experts ({self.num_experts})."
            )

    # ------------------------------------------------------------------ #
    # Named scale presets                                                  #
    # ------------------------------------------------------------------ #

    @classmethod
    def mistral_7b(cls) -> "SubQConfig":
        """
        SubQ config matching Mistral-7B scale (~7 B parameters).

        Designed to fit and train on a single A100 80 GB GPU.  Uses Grouped
        Query Attention (32 query heads / 8 KV heads) to reduce the KV-cache
        footprint, and a dense SwiGLU FFN with the same intermediate dimension
        as Mistral 7B.

        Architecture match
        ------------------
        - RMSNorm, SwiGLU, RoPE, no-bias projections: identical to Mistral.
        - Attention: Mistral's sliding window replaced by SubQ's full SSA
          (local window + global sinks + content routing).
        - GQA: 32 Q heads, 8 KV heads (same as Mistral 7B v0.1).
        """
        return cls(
            vocab_size=32_000,
            hidden_size=4_096,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            intermediate_size=14_336,
            window_size=4_096,
            num_global_tokens=64,
            top_k_sparse=128,
            routing_rank=16,
            rope_theta=10_000.0,
            max_position_embeddings=12_000_000,
        )

    @classmethod
    def mimo_v2_flash(cls) -> "SubQConfig":
        """
        SubQ config matching MiMo-V2-Flash scale (~15 B active / 309 B total).

        Intended for multi-GPU cluster training.  Combines SubQ's SSA with a
        256-expert Sparse MoE FFN (8 experts active per token), matching the
        scale and topology of MiMo-V2-Flash (Xiaomi, 2025).

        Architecture match
        ------------------
        - 48 layers, hidden_size=7168, 64 Q heads / 8 KV heads: matches
          MiMo-V2-Flash's transformer backbone dimensions.
        - MoE FFN: 256 total experts, 8 active per token via top-K routing.
          ``intermediate_size`` is the per-expert FFN width.
        - Attention: MiMo's 5:1 SWA/full-attention interleaving is replaced
          by SubQ's SSA applied uniformly to every layer, giving the same
          O(1)-hop global connectivity at linear cost.
        - vocab_size=152_064: Qwen3 tokeniser used by MiMo-V2-Flash.

        Note
        ----
        This preset is an architectural reference.  The per-expert
        ``intermediate_size`` of 2048 approximates MiMo's MoE FFN width;
        actual active-parameter count will differ slightly from the published
        15 B figure due to routing-head and embedding overheads.
        """
        return cls(
            vocab_size=152_064,
            hidden_size=7_168,
            num_hidden_layers=48,
            num_attention_heads=64,
            num_key_value_heads=8,
            intermediate_size=2_048,
            num_experts=256,
            num_experts_per_tok=8,
            window_size=512,
            num_global_tokens=64,
            top_k_sparse=128,
            routing_rank=16,
            rope_theta=10_000.0,
            max_position_embeddings=12_000_000,
        )
