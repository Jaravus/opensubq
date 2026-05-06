"""
opensubq — Theoretical reconstruction of the SubQ sub-quadratic LLM architecture.

Public API
----------
SubQConfig                   – hyper-parameter dataclass (includes named scale presets)
SubquadraticSparseAttention  – SSA attention module (supports GQA)
SubQModel                    – full decoder-only language model
SparseMoEMLP                 – sparse mixture-of-experts FFN (MiMo-V2-Flash scale)
CharDataset                  – character-level (byte-level) dataset; no extra deps
TiktokenDataset              – GPT-2/GPT-4 BPE dataset; requires ``tiktoken``
make_split_loaders           – convenience: return (train_loader, val_loader) pair
make_synthetic_datasets      – synthetic data for tests and quick demos
"""

from .config import SubQConfig
from .attention import SubquadraticSparseAttention, RotaryEmbedding, apply_rotary_emb
from .layers import SubQRMSNorm, SubQMLP, SparseMoEMLP
from .model import SubQModel, SubQTransformerLayer
from .data import (
    CharDataset,
    TiktokenDataset,
    make_split_loaders,
    make_synthetic_corpus,
    make_synthetic_datasets,
)

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
    "CharDataset",
    "TiktokenDataset",
    "make_split_loaders",
    "make_synthetic_corpus",
    "make_synthetic_datasets",
]
