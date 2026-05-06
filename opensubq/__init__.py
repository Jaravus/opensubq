"""
opensubq — Theoretical reconstruction of the SubQ sub-quadratic LLM architecture.

Public API
----------
SubQConfig                   – hyper-parameter dataclass (includes named scale presets)
SubquadraticSparseAttention  – SSA attention module (supports GQA)
SubQModel                    – full decoder-only language model
SparseMoEMLP                 – sparse mixture-of-experts FFN (MiMo-V2-Flash scale)
"""

from .config import SubQConfig
from .attention import SubquadraticSparseAttention, RotaryEmbedding, apply_rotary_emb
from .layers import SubQRMSNorm, SubQMLP, SparseMoEMLP
from .model import SubQModel, SubQTransformerLayer

__all__ = [
    "SubQConfig",
    "SubquadraticSparseAttention",
    "RotaryEmbedding",
    "apply_rotary_emb",
    "SubQRMSNorm",
    "SubQMLP",
    "SparseMoEMLP",
    "SubQTransformerLayer",
    "SubQModel",
]
