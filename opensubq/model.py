"""
SubQ Transformer model.

Architecture overview
---------------------
  Token embedding
  └─ N × SubQTransformerLayer
       ├─ SubQRMSNorm
       ├─ SubquadraticSparseAttention   ← SSA (local + global + content routing)
       ├─ residual connection
       ├─ SubQRMSNorm
       ├─ SubQMLP  (SwiGLU)
       └─ residual connection
  SubQRMSNorm
  LM head (linear, optional weight-tying with embedding)

The layout is pre-norm decoder-only (GPT / LLaMA style).  Standard
multi-head self-attention is replaced wholesale with SSA, giving O(N)
attention cost instead of O(N²).
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import SubquadraticSparseAttention
from .config import SubQConfig
from .layers import SubQMLP, SubQRMSNorm, SparseMoEMLP


class SubQTransformerLayer(nn.Module):
    """
    Single SubQ transformer block.

    Follows a pre-norm layout:
        x = x + Dropout(SSA(RMSNorm(x)))
        x = x + Dropout(MLP(RMSNorm(x)))
    """

    def __init__(self, config: SubQConfig) -> None:
        super().__init__()
        self.attn_norm = SubQRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn      = SubquadraticSparseAttention(config)
        self.mlp_norm  = SubQRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = (
            SparseMoEMLP(config) if config.num_experts is not None else SubQMLP(config)
        )
        self.drop      = nn.Dropout(config.dropout)

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
        """
        # Attention sub-layer
        residual      = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states = self.attn(hidden_states, attention_mask=attention_mask)
        hidden_states = self.drop(hidden_states)
        hidden_states = residual + hidden_states

        # FFN sub-layer
        residual      = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.drop(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class SubQModel(nn.Module):
    """
    Full SubQ decoder-only language model.

    This is a *theoretical reconstruction* of the SubQ architecture described
    in the Subquadratic blog post (https://subq.ai/introducing-subq), built
    from first principles using the available research literature.

    Key design choices
    ------------------
    * **SSA** replaces standard attention in every layer, giving O(N)
      attention complexity and linear KV-cache growth.
    * **RoPE** (Rotary Position Embeddings) encodes position; handles
      arbitrary sequence lengths without interpolation.
    * **SwiGLU MLP** following the convention of recent frontier LLMs.
    * **Pre-norm RMSNorm** for training stability.
    * Optional **weight tying** between token embedding and LM head.

    Parameters
    ----------
    config : SubQConfig
        All model and SSA hyper-parameters.
    """

    def __init__(self, config: SubQConfig) -> None:
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [SubQTransformerLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm    = SubQRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.apply(self._init_weights)

    # ------------------------------------------------------------------ #
    # Weight initialisation                                                #
    # ------------------------------------------------------------------ #

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    # ------------------------------------------------------------------ #
    # Forward                                                              #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        input_ids      : (B, N)  integer token ids.
        attention_mask : (B, N)  1 = real token, 0 = padding (optional).
        labels         : (B, N)  integer token ids for language-model loss
                         (optional).  When provided the loss is computed with a
                         one-position left shift so that position i predicts
                         position i+1 (standard autoregressive cross-entropy).
                         Token positions where ``labels == -100`` are ignored.

        Returns
        -------
        logits : (B, N, vocab_size)  — when ``labels`` is ``None``.
        (loss, logits) : scalar CE loss and (B, N, vocab_size) logits
                         — when ``labels`` is provided.
        """
        hidden_states = self.embed_tokens(input_ids)   # (B, N, D)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)

        hidden_states = self.norm(hidden_states)        # (B, N, D)
        logits = self.lm_head(hidden_states)            # (B, N, V)

        if labels is None:
            return logits

        # Autoregressive loss: shift so that token i predicts token i+1.
        # logits[:, :-1] predicts labels[:, 1:]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return loss, logits

    # ------------------------------------------------------------------ #
    # Utility                                                              #
    # ------------------------------------------------------------------ #

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Return the total number of (trainable) parameters."""
        params = (
            p for p in self.parameters()
            if (not trainable_only or p.requires_grad)
        )
        return sum(p.numel() for p in params)
