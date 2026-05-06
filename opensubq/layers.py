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
