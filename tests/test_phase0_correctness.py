"""
Phase 0 correctness tests.

Covers the two P0 fixes agreed in the development roadmap:

  1. Causal masking — token i must not attend to any position j > i.
     Tests verify that changing a future token does not affect the
     logits / output at earlier positions.

  2. Autoregressive loss — CrossEntropyLoss with a one-position left
     shift so that token i predicts token i+1.
     Tests verify the loss value, gradient flow, and ignore_index=-100
     handling.
"""
import pytest
import torch
import torch.nn.functional as F

from opensubq import SubQConfig, SubQModel, SubquadraticSparseAttention


# ------------------------------------------------------------------ #
# Shared tiny configs                                                  #
# ------------------------------------------------------------------ #

def _tiny_causal_config(**overrides):
    defaults = dict(
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
        causal=True,
    )
    defaults.update(overrides)
    return SubQConfig(**defaults)


# ------------------------------------------------------------------ #
# 1. Causal masking                                                    #
# ------------------------------------------------------------------ #

class TestCausalMask:
    def test_causal_flag_default_true(self):
        cfg = SubQConfig()
        assert cfg.causal is True

    def test_causal_flag_stored_in_attention(self):
        cfg = _tiny_causal_config(causal=True)
        ssa = SubquadraticSparseAttention(cfg)
        assert ssa.causal is True

    def test_non_causal_flag_stored_in_attention(self):
        cfg = _tiny_causal_config(causal=False)
        ssa = SubquadraticSparseAttention(cfg)
        assert ssa.causal is False

    def test_causal_mask_no_future_leakage_ssa(self):
        """
        Changing a future hidden-state token must not affect any earlier
        position's output from SubquadraticSparseAttention.
        """
        cfg = _tiny_causal_config(causal=True)
        ssa = SubquadraticSparseAttention(cfg).eval()
        B, N, D = 1, 12, cfg.hidden_size

        x = torch.randn(B, N, D)

        out_orig = ssa(x)

        # Perturb tokens from position 6 onward
        x_perturbed = x.clone()
        x_perturbed[:, 6:, :] = torch.randn(B, N - 6, D)
        out_perturbed = ssa(x_perturbed)

        # Positions 0-5 must be identical; positions 6+ may differ
        assert torch.allclose(out_orig[:, :6, :], out_perturbed[:, :6, :], atol=1e-5), (
            "Causal masking violated: earlier positions changed when future tokens were perturbed"
        )

    def test_causal_mask_no_future_leakage_model(self):
        """
        End-to-end: changing future input tokens must not change earlier
        logits in SubQModel with causal=True.
        """
        cfg = _tiny_causal_config(causal=True)
        model = SubQModel(cfg).eval()
        B, N = 1, 16
        ids = torch.randint(0, cfg.vocab_size, (B, N))

        logits_orig = model(ids)

        # Perturb the second half of the sequence
        ids_perturbed = ids.clone()
        ids_perturbed[:, N // 2:] = torch.randint(0, cfg.vocab_size, (B, N // 2))
        logits_perturbed = model(ids_perturbed)

        assert torch.allclose(
            logits_orig[:, : N // 2, :],
            logits_perturbed[:, : N // 2, :],
            atol=1e-5,
        ), "Causal masking violated at model level: earlier logits changed when future tokens were perturbed"

    def test_non_causal_model_future_tokens_do_affect_past(self):
        """
        With causal=False, future tokens should (in general) affect earlier
        positions — confirms causal=True is doing real work.
        """
        cfg = _tiny_causal_config(causal=False)
        model = SubQModel(cfg).eval()
        B, N = 1, 16

        # Try a few random seeds; at least one should show leakage
        leakage_found = False
        for _ in range(5):
            ids = torch.randint(0, cfg.vocab_size, (B, N))
            logits_orig = model(ids)

            ids_perturbed = ids.clone()
            ids_perturbed[:, N // 2:] = torch.randint(0, cfg.vocab_size, (B, N // 2))
            logits_perturbed = model(ids_perturbed)

            if not torch.allclose(logits_orig[:, : N // 2, :], logits_perturbed[:, : N // 2, :], atol=1e-5):
                leakage_found = True
                break

        assert leakage_found, "Expected bidirectional model to show future-token influence; none found"

    def test_causal_ssa_lower_triangle_only(self):
        """
        Internal check: with causal=True the combined mask should be
        lower-triangular (no True entries above the diagonal).
        """
        cfg = _tiny_causal_config(causal=True)
        ssa = SubquadraticSparseAttention(cfg).eval()
        N, D = 12, cfg.hidden_size
        h = torch.randn(1, N, D)

        # Reconstruct the mask the same way the forward pass does
        local_mask = ssa._local_mask(N, h.device)
        global_mask = ssa._global_mask(N, h.device)
        routing_mask = ssa._routing_mask(h)  # (1, H, N, N) — already causal-premasked

        ssa_mask = (
            local_mask.unsqueeze(0).unsqueeze(0)
            | global_mask.unsqueeze(0).unsqueeze(0)
            | routing_mask
        )
        combined = ssa_mask & ssa._causal_mask(N, h.device).unsqueeze(0).unsqueeze(0)

        # No position above the diagonal should be True
        upper_triangle = torch.triu(combined, diagonal=1)
        assert not upper_triangle.any(), (
            "Causal + SSA mask has True entries above the diagonal"
        )


# ------------------------------------------------------------------ #
# 2. Autoregressive loss                                               #
# ------------------------------------------------------------------ #

class TestAutoregressiveLoss:
    def test_no_labels_returns_logits_tensor(self):
        cfg = _tiny_causal_config()
        model = SubQModel(cfg).eval()
        ids = torch.randint(0, cfg.vocab_size, (2, 10))
        out = model(ids)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, 10, cfg.vocab_size)

    def test_with_labels_returns_tuple(self):
        cfg = _tiny_causal_config()
        model = SubQModel(cfg).eval()
        ids = torch.randint(0, cfg.vocab_size, (2, 10))
        loss, logits = model(ids, labels=ids)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0          # scalar
        assert logits.shape == (2, 10, cfg.vocab_size)

    def test_loss_is_finite(self):
        cfg = _tiny_causal_config()
        model = SubQModel(cfg).eval()
        ids = torch.randint(0, cfg.vocab_size, (2, 12))
        loss, _ = model(ids, labels=ids)
        assert torch.isfinite(loss)

    def test_loss_is_positive(self):
        cfg = _tiny_causal_config()
        model = SubQModel(cfg).eval()
        ids = torch.randint(0, cfg.vocab_size, (2, 12))
        loss, _ = model(ids, labels=ids)
        assert loss.item() > 0.0

    def test_loss_matches_manual_cross_entropy(self):
        """Loss from model should equal a manually computed shifted CE loss."""
        cfg = _tiny_causal_config()
        model = SubQModel(cfg).eval()
        B, N = 2, 10
        ids = torch.randint(0, cfg.vocab_size, (B, N))

        loss_model, logits = model(ids, labels=ids)

        # Manual: shift logits left by 1, compare to labels shifted right by 1
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = ids[:, 1:].contiguous()
        loss_manual = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        assert torch.allclose(loss_model, loss_manual, atol=1e-6)

    def test_loss_gradients_flow_to_all_params(self):
        """
        Backward pass through the loss should produce gradients for every
        differentiable parameter.

        Note: route_q and route_k are excluded because the routing mask is
        derived via a non-differentiable top-K threshold comparison (boolean
        mask), so those weights receive no gradient in the reference
        implementation.  This is expected — a production implementation would
        use a straight-through or soft-routing estimator to make them trainable.
        """
        cfg = _tiny_causal_config()
        model = SubQModel(cfg).train()
        ids = torch.randint(0, cfg.vocab_size, (1, 8))
        loss, _ = model(ids, labels=ids)
        loss.backward()

        # Parameters that are intentionally excluded from gradient flow in
        # this reference implementation (discrete mask, no STE).
        non_differentiable = {"route_q.weight", "route_k.weight"}

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # Strip the layer prefix to get the bare parameter name
            bare = ".".join(name.split(".")[-2:])
            if bare in non_differentiable:
                continue
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    def test_ignore_index_minus100(self):
        """Positions with label == -100 should be excluded from the loss."""
        cfg = _tiny_causal_config()
        model = SubQModel(cfg).eval()
        B, N = 1, 10
        ids = torch.randint(0, cfg.vocab_size, (B, N))

        # Labels with no ignore positions
        loss_full, _ = model(ids, labels=ids)

        # Mask out the last few positions
        labels_masked = ids.clone()
        labels_masked[:, -3:] = -100
        loss_partial, _ = model(ids, labels=labels_masked)

        # Both should be finite scalars; they will differ (fewer positions averaged)
        assert torch.isfinite(loss_full)
        assert torch.isfinite(loss_partial)
        assert not torch.allclose(loss_full, loss_partial)

    def test_loss_decreases_with_gradient_steps(self):
        """A few Adam steps on a tiny model should reduce the training loss."""
        cfg = _tiny_causal_config()
        model = SubQModel(cfg).train()
        ids = torch.randint(0, cfg.vocab_size, (1, 8))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            loss, _ = model(ids, labels=ids)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], (
            f"Loss did not decrease after 5 steps: {losses}"
        )
