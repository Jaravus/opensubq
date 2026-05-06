"""
Tests for the SubquadraticSparseAttention (SSA) module.
"""
import pytest
import torch

from opensubq import SubQConfig, SubquadraticSparseAttention, RotaryEmbedding, apply_rotary_emb


# ------------------------------------------------------------------ #
# Fixtures                                                            #
# ------------------------------------------------------------------ #

@pytest.fixture()
def small_config():
    """A tiny config suitable for CPU unit tests."""
    return SubQConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        window_size=4,
        num_global_tokens=2,
        top_k_sparse=8,
        routing_rank=4,
        vocab_size=256,
        attention_dropout=0.0,
        dropout=0.0,
    )


@pytest.fixture()
def ssa(small_config):
    return SubquadraticSparseAttention(small_config).eval()


# ------------------------------------------------------------------ #
# RotaryEmbedding                                                     #
# ------------------------------------------------------------------ #

class TestRotaryEmbedding:
    def test_output_shapes(self):
        dim, seq_len = 32, 16
        rope = RotaryEmbedding(dim)
        cos, sin = rope(seq_len, device=torch.device("cpu"))
        assert cos.shape == (seq_len, dim)
        assert sin.shape == (seq_len, dim)

    def test_cached_equals_fresh(self):
        """Calling twice with the same length returns the same tensors."""
        rope = RotaryEmbedding(32)
        cos1, sin1 = rope(10, device=torch.device("cpu"))
        cos2, sin2 = rope(10, device=torch.device("cpu"))
        assert torch.equal(cos1, cos2)
        assert torch.equal(sin1, sin2)

    def test_longer_seq_extends_cache(self):
        rope = RotaryEmbedding(32)
        rope(8, device=torch.device("cpu"))
        cos16, sin16 = rope(16, device=torch.device("cpu"))
        assert cos16.shape[0] == 16

    def test_apply_rotary_emb_preserves_shape(self):
        B, N, H, d = 2, 10, 4, 32
        q = torch.randn(B, N, H, d)
        k = torch.randn(B, N, H, d)
        rope = RotaryEmbedding(d)
        cos, sin = rope(N, device=q.device)
        q_rot, k_rot = apply_rotary_emb(q, k, cos, sin)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_apply_rotary_emb_is_not_identity(self):
        """Rotating should change the values (at non-zero positions)."""
        B, N, H, d = 1, 6, 2, 16
        q = torch.randn(B, N, H, d)
        k = torch.randn(B, N, H, d)
        rope = RotaryEmbedding(d)
        cos, sin = rope(N, device=q.device)
        q_rot, _ = apply_rotary_emb(q, k, cos, sin)
        # Position 0 has cos=1, sin=0 → no change.  Check pos 1+ differs.
        assert not torch.allclose(q_rot[:, 1:], q[:, 1:])


# ------------------------------------------------------------------ #
# SubquadraticSparseAttention — mask helpers                          #
# ------------------------------------------------------------------ #

class TestSSAMasks:
    def test_local_mask_symmetry(self, ssa):
        N = 20
        mask = ssa._local_mask(N, torch.device("cpu"))
        assert mask.shape == (N, N)
        # Symmetric
        assert torch.equal(mask, mask.T)

    def test_local_mask_window(self, ssa):
        N, W = 20, ssa.window_size
        mask = ssa._local_mask(N, torch.device("cpu"))
        for i in range(N):
            for j in range(N):
                expected = abs(i - j) <= W
                assert mask[i, j].item() == expected

    def test_global_mask_coverage(self, ssa):
        N = 20
        G = min(ssa.num_global_tokens, N)
        mask = ssa._global_mask(N, torch.device("cpu"))
        assert mask.shape == (N, N)
        # Every token attends to global keys
        assert mask[:, :G].all()
        # Global queries attend to every token
        assert mask[:G, :].all()

    def test_routing_mask_shape(self, ssa, small_config):
        B, N, D = 2, 16, small_config.hidden_size
        h = torch.randn(B, N, D)
        mask = ssa._routing_mask(h)
        assert mask.shape == (B, small_config.num_attention_heads, N, N)
        assert mask.dtype == torch.bool

    def test_routing_mask_top_k_count(self, ssa, small_config):
        """Each query row should have ≥ top_k_sparse True entries (ties may add more)."""
        B, N = 1, 20
        h = torch.randn(B, N, small_config.hidden_size)
        mask = ssa._routing_mask(h)
        K = min(ssa.top_k_sparse, N)
        row_counts = mask.sum(dim=-1)   # (B, H, N)
        assert (row_counts >= K).all()


# ------------------------------------------------------------------ #
# SubquadraticSparseAttention — forward pass                          #
# ------------------------------------------------------------------ #

class TestSSAForward:
    def test_output_shape(self, ssa, small_config):
        B, N, D = 2, 16, small_config.hidden_size
        x = torch.randn(B, N, D)
        out = ssa(x)
        assert out.shape == (B, N, D)

    def test_output_shape_with_mask(self, ssa, small_config):
        B, N, D = 2, 12, small_config.hidden_size
        x = torch.randn(B, N, D)
        mask = torch.ones(B, N, dtype=torch.long)
        mask[0, 8:] = 0   # first item has padding in positions 8-11
        out = ssa(x, attention_mask=mask)
        assert out.shape == (B, N, D)

    def test_no_nan_in_output(self, ssa, small_config):
        B, N, D = 2, 14, small_config.hidden_size
        x = torch.randn(B, N, D)
        out = ssa(x)
        assert not torch.isnan(out).any()

    def test_no_nan_with_padding_mask(self, ssa, small_config):
        """All-padding rows (all 0 mask) should not produce NaN outputs."""
        B, N, D = 1, 8, small_config.hidden_size
        x = torch.randn(B, N, D)
        # Only first token is real — heavily padded sequence
        mask = torch.zeros(B, N, dtype=torch.long)
        mask[0, 0] = 1
        out = ssa(x, attention_mask=mask)
        assert not torch.isnan(out).any()

    def test_single_token_sequence(self, ssa, small_config):
        B, N, D = 1, 1, small_config.hidden_size
        x = torch.randn(B, N, D)
        out = ssa(x)
        assert out.shape == (B, N, D)
        assert not torch.isnan(out).any()

    def test_batch_independence(self, ssa, small_config):
        """Items in a batch should not influence each other."""
        B, N, D = 3, 10, small_config.hidden_size
        x = torch.randn(B, N, D)
        out_batch = ssa(x)
        for i in range(B):
            out_single = ssa(x[i : i + 1])
            assert torch.allclose(out_batch[i : i + 1], out_single, atol=1e-5)

    def test_eval_deterministic(self, ssa, small_config):
        """In eval mode with dropout=0 the result must be deterministic."""
        B, N, D = 2, 10, small_config.hidden_size
        x = torch.randn(B, N, D)
        out1 = ssa(x)
        out2 = ssa(x)
        assert torch.allclose(out1, out2)

    def test_seq_len_1_through_global_tokens_plus_one(self, ssa, small_config):
        """Edge: seq_len ≤ num_global_tokens — global mask should cover all."""
        D = small_config.hidden_size
        for N in [1, 2, small_config.num_global_tokens, small_config.num_global_tokens + 1]:
            x = torch.randn(1, N, D)
            out = ssa(x)
            assert out.shape == (1, N, D)
            assert not torch.isnan(out).any()
