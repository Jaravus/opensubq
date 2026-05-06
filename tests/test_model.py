"""
Tests for SubQTransformerLayer and SubQModel.
"""
import pytest
import torch

from opensubq import SubQConfig, SubQModel, SubQTransformerLayer
from opensubq.layers import SubQRMSNorm, SubQMLP


# ------------------------------------------------------------------ #
# Fixtures                                                            #
# ------------------------------------------------------------------ #

@pytest.fixture()
def tiny_config():
    """Very small config for fast CPU tests."""
    return SubQConfig(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        window_size=4,
        num_global_tokens=2,
        top_k_sparse=8,
        routing_rank=4,
        dropout=0.0,
        attention_dropout=0.0,
        tie_word_embeddings=False,
    )


@pytest.fixture()
def model(tiny_config):
    return SubQModel(tiny_config).eval()


# ------------------------------------------------------------------ #
# SubQRMSNorm                                                         #
# ------------------------------------------------------------------ #

class TestSubQRMSNorm:
    def test_output_shape(self):
        norm = SubQRMSNorm(32)
        x = torch.randn(2, 10, 32)
        assert norm(x).shape == x.shape

    def test_unit_scale_is_near_unit_rms(self):
        """With weight=1, output RMS should be ≈ 1."""
        norm = SubQRMSNorm(64)
        x = torch.randn(4, 16, 64)
        out = norm(x)
        rms = out.pow(2).mean(dim=-1).sqrt()
        # RMS should be close to 1 for random Gaussian input
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.2)

    def test_no_bias_parameter(self):
        norm = SubQRMSNorm(32)
        param_names = [n for n, _ in norm.named_parameters()]
        assert "bias" not in param_names
        assert "weight" in param_names


# ------------------------------------------------------------------ #
# SubQMLP                                                             #
# ------------------------------------------------------------------ #

class TestSubQMLP:
    def test_output_shape(self, tiny_config):
        mlp = SubQMLP(tiny_config)
        x = torch.randn(2, 10, tiny_config.hidden_size)
        assert mlp(x).shape == x.shape

    def test_no_nan(self, tiny_config):
        mlp = SubQMLP(tiny_config)
        x = torch.randn(2, 8, tiny_config.hidden_size)
        assert not torch.isnan(mlp(x)).any()


# ------------------------------------------------------------------ #
# SubQTransformerLayer                                                 #
# ------------------------------------------------------------------ #

class TestSubQTransformerLayer:
    def test_output_shape(self, tiny_config):
        layer = SubQTransformerLayer(tiny_config).eval()
        B, N, D = 2, 12, tiny_config.hidden_size
        x = torch.randn(B, N, D)
        out = layer(x)
        assert out.shape == (B, N, D)

    def test_output_shape_with_mask(self, tiny_config):
        layer = SubQTransformerLayer(tiny_config).eval()
        B, N, D = 2, 10, tiny_config.hidden_size
        x = torch.randn(B, N, D)
        mask = torch.ones(B, N, dtype=torch.long)
        mask[1, 6:] = 0
        out = layer(x, attention_mask=mask)
        assert out.shape == (B, N, D)

    def test_residual_connection_present(self, tiny_config):
        """Output must differ from the raw attention / MLP output (residual adds input)."""
        layer = SubQTransformerLayer(tiny_config).eval()
        x = torch.randn(1, 8, tiny_config.hidden_size)
        out = layer(x)
        assert not torch.allclose(out, x)

    def test_no_nan(self, tiny_config):
        layer = SubQTransformerLayer(tiny_config).eval()
        x = torch.randn(2, 10, tiny_config.hidden_size)
        assert not torch.isnan(layer(x)).any()


# ------------------------------------------------------------------ #
# SubQModel                                                            #
# ------------------------------------------------------------------ #

class TestSubQModel:
    def test_logit_shape(self, model, tiny_config):
        B, N = 2, 16
        ids = torch.randint(0, tiny_config.vocab_size, (B, N))
        logits = model(ids)
        assert logits.shape == (B, N, tiny_config.vocab_size)

    def test_logit_shape_with_mask(self, model, tiny_config):
        B, N = 2, 12
        ids = torch.randint(0, tiny_config.vocab_size, (B, N))
        mask = torch.ones(B, N, dtype=torch.long)
        mask[0, 8:] = 0
        logits = model(ids, attention_mask=mask)
        assert logits.shape == (B, N, tiny_config.vocab_size)

    def test_no_nan_in_logits(self, model, tiny_config):
        ids = torch.randint(0, tiny_config.vocab_size, (2, 14))
        assert not torch.isnan(model(ids)).any()

    def test_num_parameters_positive(self, model):
        assert model.num_parameters() > 0

    def test_tie_word_embeddings(self, tiny_config):
        import dataclasses
        cfg = dataclasses.replace(tiny_config, tie_word_embeddings=True)
        m = SubQModel(cfg).eval()
        assert m.lm_head.weight is m.embed_tokens.weight

    def test_no_tie_word_embeddings(self, model):
        assert model.lm_head.weight is not model.embed_tokens.weight

    def test_eval_deterministic(self, model, tiny_config):
        ids = torch.randint(0, tiny_config.vocab_size, (1, 10))
        out1 = model(ids)
        out2 = model(ids)
        assert torch.allclose(out1, out2)

    def test_batch_independence(self, model, tiny_config):
        """Each item in the batch should only depend on its own input tokens."""
        B, N = 3, 8
        ids = torch.randint(0, tiny_config.vocab_size, (B, N))
        logits_batch = model(ids)
        for i in range(B):
            logits_single = model(ids[i : i + 1])
            assert torch.allclose(logits_batch[i : i + 1], logits_single, atol=1e-5)

    def test_different_inputs_give_different_outputs(self, model, tiny_config):
        ids1 = torch.randint(0, tiny_config.vocab_size, (1, 8))
        ids2 = torch.randint(0, tiny_config.vocab_size, (1, 8))
        # It's extremely unlikely two random inputs produce identical outputs
        if not torch.equal(ids1, ids2):
            assert not torch.allclose(model(ids1), model(ids2))

    def test_single_token_input(self, model, tiny_config):
        ids = torch.randint(0, tiny_config.vocab_size, (1, 1))
        logits = model(ids)
        assert logits.shape == (1, 1, tiny_config.vocab_size)
        assert not torch.isnan(logits).any()

    def test_config_stored(self, model, tiny_config):
        assert model.config is tiny_config

    def test_num_layers(self, model, tiny_config):
        assert len(model.layers) == tiny_config.num_hidden_layers

    def test_config_validation_head_dim_mismatch(self):
        with pytest.raises(ValueError, match="divisible"):
            SubQConfig(hidden_size=65, num_attention_heads=4)
