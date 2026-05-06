"""
Tests for named scale presets, GQA support, and SparseMoEMLP.
"""
import dataclasses

import pytest
import torch

from opensubq import SubQConfig, SubQModel, SparseMoEMLP, SubquadraticSparseAttention
from opensubq.layers import SubQMLP


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

def _tiny_moe_config(**overrides):
    """Return a tiny MoE config suitable for CPU tests."""
    defaults = dict(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=32,
        window_size=4,
        num_global_tokens=2,
        top_k_sparse=8,
        routing_rank=4,
        dropout=0.0,
        attention_dropout=0.0,
        num_experts=4,
        num_experts_per_tok=2,
    )
    defaults.update(overrides)
    return SubQConfig(**defaults)


def _tiny_gqa_config(**overrides):
    """Return a tiny GQA config (2 KV heads, 4 Q heads) for CPU tests."""
    defaults = dict(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        window_size=4,
        num_global_tokens=2,
        top_k_sparse=8,
        routing_rank=4,
        dropout=0.0,
        attention_dropout=0.0,
    )
    defaults.update(overrides)
    return SubQConfig(**defaults)


# ------------------------------------------------------------------ #
# SubQConfig validation                                               #
# ------------------------------------------------------------------ #

class TestConfigValidation:
    def test_gqa_invalid_divisibility(self):
        with pytest.raises(ValueError, match="divisible"):
            SubQConfig(
                hidden_size=64,
                num_attention_heads=4,
                num_key_value_heads=3,  # 4 % 3 != 0
            )

    def test_moe_experts_per_tok_exceeds_num_experts(self):
        with pytest.raises(ValueError, match="num_experts_per_tok"):
            SubQConfig(
                hidden_size=64,
                num_attention_heads=4,
                num_experts=2,
                num_experts_per_tok=4,   # 4 > 2
            )

    def test_default_kv_heads_equals_q_heads(self):
        cfg = SubQConfig(hidden_size=64, num_attention_heads=4)
        assert cfg.num_key_value_heads == cfg.num_attention_heads

    def test_explicit_kv_heads_stored(self):
        cfg = _tiny_gqa_config()
        assert cfg.num_key_value_heads == 2
        assert cfg.num_attention_heads == 4

    def test_mha_n_rep_one(self):
        """num_heads == num_kv_heads → n_rep = 1 (standard MHA)."""
        cfg = SubQConfig(hidden_size=64, num_attention_heads=4)
        assert cfg.num_attention_heads // cfg.num_key_value_heads == 1

    def test_gqa_n_rep(self):
        cfg = _tiny_gqa_config()
        assert cfg.num_attention_heads // cfg.num_key_value_heads == 2


# ------------------------------------------------------------------ #
# Named presets — shape / type checks only (no weight loading)       #
# ------------------------------------------------------------------ #

class TestPresets:
    def test_mistral_7b_returns_config(self):
        cfg = SubQConfig.mistral_7b()
        assert isinstance(cfg, SubQConfig)

    def test_mistral_7b_dimensions(self):
        cfg = SubQConfig.mistral_7b()
        assert cfg.hidden_size == 4_096
        assert cfg.num_hidden_layers == 32
        assert cfg.num_attention_heads == 32
        assert cfg.num_key_value_heads == 8
        assert cfg.intermediate_size == 14_336
        assert cfg.vocab_size == 32_000
        assert cfg.num_experts is None  # dense FFN

    def test_mistral_7b_gqa_ratio(self):
        cfg = SubQConfig.mistral_7b()
        assert cfg.num_attention_heads % cfg.num_key_value_heads == 0
        assert cfg.num_attention_heads // cfg.num_key_value_heads == 4

    def test_mimo_v2_flash_returns_config(self):
        cfg = SubQConfig.mimo_v2_flash()
        assert isinstance(cfg, SubQConfig)

    def test_mimo_v2_flash_dimensions(self):
        cfg = SubQConfig.mimo_v2_flash()
        assert cfg.hidden_size == 7_168
        assert cfg.num_hidden_layers == 48
        assert cfg.num_attention_heads == 64
        assert cfg.num_key_value_heads == 8
        assert cfg.num_experts == 256
        assert cfg.num_experts_per_tok == 8
        assert cfg.vocab_size == 152_064

    def test_mimo_v2_flash_gqa_ratio(self):
        cfg = SubQConfig.mimo_v2_flash()
        assert cfg.num_attention_heads % cfg.num_key_value_heads == 0
        assert cfg.num_attention_heads // cfg.num_key_value_heads == 8

    def test_mimo_v2_flash_is_moe(self):
        cfg = SubQConfig.mimo_v2_flash()
        assert cfg.num_experts is not None
        assert cfg.num_experts_per_tok <= cfg.num_experts


# ------------------------------------------------------------------ #
# GQA — attention module                                              #
# ------------------------------------------------------------------ #

class TestGQAAttention:
    def test_kv_projection_size(self):
        cfg = _tiny_gqa_config()
        ssa = SubquadraticSparseAttention(cfg)
        # K / V projections should output num_kv_heads * head_dim, not hidden_size
        kv_out = cfg.num_key_value_heads * cfg.head_dim
        assert ssa.k_proj.out_features == kv_out
        assert ssa.v_proj.out_features == kv_out

    def test_q_projection_size(self):
        cfg = _tiny_gqa_config()
        ssa = SubquadraticSparseAttention(cfg)
        assert ssa.q_proj.out_features == cfg.hidden_size

    def test_repeat_kv_mha_is_noop(self):
        x = torch.randn(2, 10, 4, 16)
        out = SubquadraticSparseAttention._repeat_kv(x, 1)
        assert out is x  # returned unchanged

    def test_repeat_kv_doubles_heads(self):
        x = torch.randn(2, 10, 2, 16)
        out = SubquadraticSparseAttention._repeat_kv(x, 2)
        assert out.shape == (2, 10, 4, 16)
        # Each KV head should be duplicated consecutively
        assert torch.equal(out[:, :, 0], out[:, :, 1])
        assert torch.equal(out[:, :, 2], out[:, :, 3])

    def test_gqa_forward_output_shape(self):
        cfg = _tiny_gqa_config()
        ssa = SubquadraticSparseAttention(cfg).eval()
        B, N, D = 2, 16, cfg.hidden_size
        x = torch.randn(B, N, D)
        out = ssa(x)
        assert out.shape == (B, N, D)

    def test_gqa_forward_no_nan(self):
        cfg = _tiny_gqa_config()
        ssa = SubquadraticSparseAttention(cfg).eval()
        x = torch.randn(2, 14, cfg.hidden_size)
        assert not torch.isnan(ssa(x)).any()

    def test_gqa_model_forward(self):
        cfg = _tiny_gqa_config()
        model = SubQModel(cfg).eval()
        ids = torch.randint(0, cfg.vocab_size, (2, 12))
        logits = model(ids)
        assert logits.shape == (2, 12, cfg.vocab_size)
        assert not torch.isnan(logits).any()


# ------------------------------------------------------------------ #
# SparseMoEMLP                                                        #
# ------------------------------------------------------------------ #

class TestSparseMoEMLP:
    def test_output_shape(self):
        cfg = _tiny_moe_config()
        moe = SparseMoEMLP(cfg).eval()
        B, N, D = 2, 10, cfg.hidden_size
        x = torch.randn(B, N, D)
        out = moe(x)
        assert out.shape == (B, N, D)

    def test_no_nan(self):
        cfg = _tiny_moe_config()
        moe = SparseMoEMLP(cfg).eval()
        x = torch.randn(2, 8, cfg.hidden_size)
        assert not torch.isnan(moe(x)).any()

    def test_requires_num_experts(self):
        cfg = SubQConfig(hidden_size=64, num_attention_heads=4)
        with pytest.raises(AssertionError, match="num_experts"):
            SparseMoEMLP(cfg)

    def test_num_experts_modules(self):
        cfg = _tiny_moe_config()
        moe = SparseMoEMLP(cfg)
        assert len(moe.experts) == cfg.num_experts

    def test_each_expert_is_subqmlp(self):
        cfg = _tiny_moe_config()
        moe = SparseMoEMLP(cfg)
        for expert in moe.experts:
            assert isinstance(expert, SubQMLP)

    def test_gate_shape(self):
        cfg = _tiny_moe_config()
        moe = SparseMoEMLP(cfg)
        assert moe.gate.in_features == cfg.hidden_size
        assert moe.gate.out_features == cfg.num_experts

    def test_single_token(self):
        cfg = _tiny_moe_config()
        moe = SparseMoEMLP(cfg).eval()
        x = torch.randn(1, 1, cfg.hidden_size)
        out = moe(x)
        assert out.shape == (1, 1, cfg.hidden_size)
        assert not torch.isnan(out).any()

    def test_different_experts_can_produce_different_outputs(self):
        """Two distinct inputs should (almost certainly) produce distinct outputs."""
        cfg = _tiny_moe_config()
        moe = SparseMoEMLP(cfg).eval()
        x1 = torch.randn(1, 4, cfg.hidden_size)
        x2 = torch.randn(1, 4, cfg.hidden_size)
        out1 = moe(x1)
        out2 = moe(x2)
        assert not torch.allclose(out1, out2)


# ------------------------------------------------------------------ #
# SubQModel with MoE FFN                                              #
# ------------------------------------------------------------------ #

class TestMoEModel:
    def test_layer_uses_moe(self):
        cfg = _tiny_moe_config()
        model = SubQModel(cfg)
        for layer in model.layers:
            assert isinstance(layer.mlp, SparseMoEMLP)

    def test_dense_layer_uses_dense_mlp(self):
        cfg = SubQConfig(
            vocab_size=256,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
            window_size=4,
            num_global_tokens=2,
            top_k_sparse=8,
            routing_rank=4,
        )
        model = SubQModel(cfg)
        for layer in model.layers:
            assert isinstance(layer.mlp, SubQMLP)

    def test_moe_model_forward_shape(self):
        cfg = _tiny_moe_config()
        model = SubQModel(cfg).eval()
        ids = torch.randint(0, cfg.vocab_size, (2, 12))
        logits = model(ids)
        assert logits.shape == (2, 12, cfg.vocab_size)

    def test_moe_model_no_nan(self):
        cfg = _tiny_moe_config()
        model = SubQModel(cfg).eval()
        ids = torch.randint(0, cfg.vocab_size, (1, 8))
        assert not torch.isnan(model(ids)).any()

    def test_moe_gqa_combined(self):
        """GQA + MoE can run together without error."""
        cfg = _tiny_moe_config(num_attention_heads=4, num_key_value_heads=2)
        model = SubQModel(cfg).eval()
        ids = torch.randint(0, cfg.vocab_size, (2, 10))
        logits = model(ids)
        assert logits.shape == (2, 10, cfg.vocab_size)
        assert not torch.isnan(logits).any()
