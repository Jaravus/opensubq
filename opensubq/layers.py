"""
SubQ building blocks: RMSNorm and SwiGLU MLP.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SubQConfig


class SubQRMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Normalises by RMS instead of mean + variance, removing the re-centring
    step.  Used in modern LLMs (LLaMA, Mistral, etc.) for its simplicity and
    training stability.

    References
    ----------
    - arXiv 1910.07467 — Root Mean Square Layer Normalization
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in float32 for numerical stability, cast back afterwards
        x_f32 = x.float()
        rms = x_f32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x_f32 * rms).to(x.dtype) * self.weight


class SubQMLP(nn.Module):
    """
    Feed-forward network with SwiGLU activation (Shazeer, 2020).

    SwiGLU replaces the classic ReLU FFN:

        Classic : FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
        SwiGLU  : FFN(x) = (Swish(xW_gate) ⊙ xW_up) W_down

    where Swish(z) = z · σ(z)  (PyTorch: ``nn.SiLU``).

    SwiGLU empirically outperforms ReLU and GeLU at matched parameter budgets
    and is the activation used in LLaMA, PaLM, and related modern LLMs.

    References
    ----------
    - arXiv 2002.05202 — GLU Variants Improve Transformer
    """

    def __init__(self, config: SubQConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act = nn.SiLU()  # Swish ≡ SiLU

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class SparseMoEMLP(nn.Module):
    """
    Sparse Mixture-of-Experts feed-forward network.

    Each token independently selects the top-``num_experts_per_tok`` experts
    by router logit, runs through those experts' SwiGLU MLPs, and returns a
    routing-weight-averaged sum of their outputs.  Active compute per token
    is equivalent to ``num_experts_per_tok`` dense MLPs, while total
    parameter capacity scales with ``num_experts``.

    This matches the token-choice top-K routing used in MiMo-V2-Flash,
    DeepSeek-V3, and related sparse MoE LLMs.

    Parameters
    ----------
    config : SubQConfig
        ``config.num_experts`` must be set (> 0).
        ``config.intermediate_size`` is the per-expert FFN width.
        ``config.num_experts_per_tok`` is the number of active experts (K).
    """

    def __init__(self, config: SubQConfig) -> None:
        super().__init__()
        assert config.num_experts is not None, (
            "SparseMoEMLP requires config.num_experts to be set."
        )
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        # Linear router: maps each token to a score over all experts
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)

        # One SwiGLU expert per slot; each uses config.intermediate_size as its
        # inner dimension, so it is a full SubQMLP at the per-expert width.
        self.experts = nn.ModuleList(
            [SubQMLP(config) for _ in range(config.num_experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, N, D)

        Returns
        -------
        output : (B, N, D)
        """
        B, N, D = x.shape
        x_flat = x.view(B * N, D)                          # (BN, D)

        # Router: select top-K experts per token
        router_logits = self.gate(x_flat)                  # (BN, E)
        routing_weights, selected_experts = torch.topk(
            router_logits, self.num_experts_per_tok, dim=-1
        )                                                   # (BN, K), (BN, K)
        routing_weights = torch.softmax(
            routing_weights.float(), dim=-1
        ).to(x.dtype)                                       # (BN, K)

        output = torch.zeros_like(x_flat)                  # (BN, D)

        # Dispatch each token to its selected experts and accumulate
        for expert_idx, expert in enumerate(self.experts):
            # (BN, K) bool: which slot(s) of each token chose this expert
            slot_mask = selected_experts == expert_idx      # (BN, K)
            # (BN,) bool: any token that uses this expert at all
            any_mask = slot_mask.any(dim=-1)
            if not any_mask.any():
                continue

            expert_tokens = x_flat[any_mask]               # (T, D)
            expert_out = expert(
                expert_tokens.unsqueeze(0)
            ).squeeze(0)                                    # (T, D)

            # Sum routing weights for all K slots that assigned this expert
            # (handles the rare case of a token routing to the same expert twice)
            weights = (
                routing_weights * slot_mask.to(routing_weights.dtype)
            ).sum(dim=-1)                                   # (BN,)

            output[any_mask] += weights[any_mask].unsqueeze(-1) * expert_out

        return output.view(B, N, D)
