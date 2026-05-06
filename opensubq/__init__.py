"""
opensubq — Theoretical reconstruction of the SubQ sub-quadratic LLM architecture.

Public API
----------
SubQConfig                   – hyper-parameter dataclass
SubquadraticSparseAttention  – SSA attention module
SubQModel                    – full decoder-only language model
"""

from .config import SubQConfig
from .attention import SubquadraticSparseAttention, RotaryEmbedding, apply_rotary_emb
from .layers import SubQRMSNorm, SubQMLP
from .model import SubQModel, SubQTransformerLayer

__all__ = [
    "SubQConfig",
    "SubquadraticSparseAttention",
    "RotaryEmbedding",
    "apply_rotary_emb",
    "SubQRMSNorm",
    "SubQMLP",
    "SubQTransformerLayer",
    "SubQModel",
]
