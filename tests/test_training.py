"""
Phase 1 sanity-check tests: tokeniser + dataset + training loop.

These tests verify:
  - CharDataset indexing and shape contracts
  - make_synthetic_datasets produces correctly shaped tensors
  - make_split_loaders produces DataLoaders that yield (input_ids, labels) pairs
  - Loss decreases over 20 AdamW steps on the tiny config (the key Phase-1
    deliverable: a loss curve showing convergence on a toy task)

TiktokenDataset tests live in tests/test_tiktoken_dataset.py and are
automatically skipped when ``tiktoken`` is not installed.
"""
from __future__ import annotations

import math
from typing import List

import pytest
import torch
from torch.utils.data import DataLoader

from opensubq import SubQConfig, SubQModel
from opensubq.data import (
    CharDataset,
    make_split_loaders,
    make_synthetic_corpus,
    make_synthetic_datasets,
)


# --------------------------------------------------------------------------- #
# Shared fixtures                                                               #
# --------------------------------------------------------------------------- #

@pytest.fixture()
def tiny_config() -> SubQConfig:
    """Fast CPU-friendly config matching the Phase-0 test fixtures."""
    return SubQConfig(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        window_size=32,
        num_global_tokens=2,
        top_k_sparse=16,
        routing_rank=4,
        dropout=0.0,
        attention_dropout=0.0,
    )


@pytest.fixture()
def tiny_model(tiny_config: SubQConfig) -> SubQModel:
    torch.manual_seed(0)
    return SubQModel(tiny_config)


SAMPLE_TEXT = (
    "The quick brown fox jumps over the lazy dog. " * 200
)  # ~9 KB of ASCII — enough for a few windows


# --------------------------------------------------------------------------- #
# CharDataset                                                                   #
# --------------------------------------------------------------------------- #

class TestCharDataset:
    def test_from_text_length(self):
        ds = CharDataset.from_text(SAMPLE_TEXT, seq_len=64)
        expected = max(0, len(SAMPLE_TEXT.encode("utf-8")) - 64)
        assert len(ds) == expected

    def test_item_shapes(self):
        ds = CharDataset.from_text(SAMPLE_TEXT, seq_len=64)
        input_ids, labels = ds[0]
        assert input_ids.shape == (64,)
        assert labels.shape == (64,)
        assert input_ids.dtype == torch.long
        assert labels.dtype == torch.long

    def test_autoregressive_shift(self):
        """labels should be input_ids shifted left by one — consecutive chars."""
        ds = CharDataset.from_text(SAMPLE_TEXT, seq_len=32)
        ids, labs = ds[0]
        ids_next, _ = ds[1]
        # The label at position i equals the input at position i+1.
        assert torch.all(labs[:-1] == ids[1:])
        # The first token of the next window equals the last label of this window.
        assert ids_next[0].item() == labs[-1].item()

    def test_token_ids_in_range(self):
        ds = CharDataset.from_text(SAMPLE_TEXT, seq_len=32)
        ids, labs = ds[0]
        assert ids.min().item() >= 0
        assert ids.max().item() <= 255
        assert labs.min().item() >= 0
        assert labs.max().item() <= 255

    def test_from_text_empty_returns_empty(self):
        ds = CharDataset.from_text("", seq_len=16)
        assert len(ds) == 0

    def test_from_text_too_short_returns_empty(self):
        # "abc" → 3 bytes, seq_len=16, not enough for a window
        ds = CharDataset.from_text("abc", seq_len=16)
        assert len(ds) == 0

    def test_from_file(self, tmp_path):
        p = tmp_path / "corpus.txt"
        p.write_text(SAMPLE_TEXT, encoding="utf-8")
        train_ds, val_ds = CharDataset.from_file(str(p), seq_len=64)
        assert len(train_ds) > 0
        assert len(val_ds) >= 0  # may be 0 for very short text

    def test_reproducible(self):
        ds1 = CharDataset.from_text(SAMPLE_TEXT, seq_len=32)
        ds2 = CharDataset.from_text(SAMPLE_TEXT, seq_len=32)
        ids1, labs1 = ds1[5]
        ids2, labs2 = ds2[5]
        assert torch.equal(ids1, ids2)
        assert torch.equal(labs1, labs2)


# TiktokenDataset tests are in tests/test_tiktoken_dataset.py (skipped when
# tiktoken is not installed).


# --------------------------------------------------------------------------- #
# Synthetic corpus + datasets                                                   #
# --------------------------------------------------------------------------- #

class TestSyntheticData:
    def test_corpus_length(self):
        ids = make_synthetic_corpus(vocab_size=256, total_tokens=1000)
        assert len(ids) == 1000

    def test_corpus_in_range(self):
        ids = make_synthetic_corpus(vocab_size=256, total_tokens=500)
        assert all(0 <= x < 256 for x in ids)

    def test_corpus_reproducible(self):
        a = make_synthetic_corpus(seed=7)
        b = make_synthetic_corpus(seed=7)
        assert a == b

    def test_corpus_different_seeds(self):
        a = make_synthetic_corpus(seed=1)
        b = make_synthetic_corpus(seed=2)
        assert a != b

    def test_make_synthetic_datasets_shapes(self):
        train_ds, val_ds = make_synthetic_datasets(
            vocab_size=256, seq_len=32, total_tokens=2000
        )
        ids, labs = train_ds[0]
        assert ids.shape == (32,)
        assert labs.shape == (32,)

    def test_make_synthetic_datasets_non_empty(self):
        train_ds, val_ds = make_synthetic_datasets(
            vocab_size=256, seq_len=32, total_tokens=5000
        )
        assert len(train_ds) > 0


# --------------------------------------------------------------------------- #
# make_split_loaders                                                            #
# --------------------------------------------------------------------------- #

class TestMakeSplitLoaders:
    def test_yields_correct_shapes(self):
        train_ds, val_ds = make_synthetic_datasets(
            vocab_size=256, seq_len=16, total_tokens=2000
        )
        train_loader, val_loader = make_split_loaders(
            train_ds, val_ds, batch_size=4
        )
        ids, labs = next(iter(train_loader))
        assert ids.shape  == (4, 16)
        assert labs.shape == (4, 16)
        assert ids.dtype  == torch.long
        assert labs.dtype == torch.long

    def test_val_loader_not_empty(self):
        train_ds, val_ds = make_synthetic_datasets(
            vocab_size=256, seq_len=16, total_tokens=5000
        )
        _, val_loader = make_split_loaders(train_ds, val_ds, batch_size=4)
        batches = list(val_loader)
        assert len(batches) > 0


# --------------------------------------------------------------------------- #
# Phase-1 Deliverable: loss decreases on tiny model (20 AdamW steps)           #
# --------------------------------------------------------------------------- #

class TestLossDecreases:
    """
    The key Phase-1 sanity-check: train the tiny model for 20 steps on a
    synthetic corpus and verify that the final loss is strictly lower than
    the initial loss.  This proves the training loop is wired up correctly:
    data flows through the model, gradients propagate, and AdamW updates
    parameters in the right direction.
    """

    NUM_STEPS   = 20    # enough to see a clear drop without being slow
    SEQ_LEN     = 32
    BATCH_SIZE  = 4

    def _run_steps(
        self,
        model: SubQModel,
        loader: DataLoader,
        num_steps: int,
    ) -> List[float]:
        """Return per-step training losses."""
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=3e-3, weight_decay=0.1
        )
        model.train()
        losses: List[float] = []
        data_iter = iter(loader)
        for _ in range(num_steps):
            try:
                ids, labs = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                ids, labs = next(data_iter)
            loss, _ = model(ids, labels=labs)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        return losses

    def test_loss_decreases(self, tiny_model: SubQModel, tiny_config: SubQConfig):
        """Final loss must be strictly lower than initial loss."""
        train_ds, _ = make_synthetic_datasets(
            vocab_size=tiny_config.vocab_size,
            seq_len=self.SEQ_LEN,
            total_tokens=10_000,
            seed=0,
        )
        loader = DataLoader(train_ds, batch_size=self.BATCH_SIZE, shuffle=True)
        losses = self._run_steps(tiny_model, loader, self.NUM_STEPS)
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}\n"
            f"Full loss curve: {losses}"
        )

    def test_loss_is_finite(self, tiny_model: SubQModel, tiny_config: SubQConfig):
        """All step losses must be finite (no NaN / inf explosion)."""
        train_ds, _ = make_synthetic_datasets(
            vocab_size=tiny_config.vocab_size,
            seq_len=self.SEQ_LEN,
            total_tokens=10_000,
            seed=1,
        )
        loader = DataLoader(train_ds, batch_size=self.BATCH_SIZE, shuffle=True)
        losses = self._run_steps(tiny_model, loader, self.NUM_STEPS)
        assert all(math.isfinite(l) for l in losses), (
            f"Non-finite loss encountered: {losses}"
        )

    def test_loss_positive(self, tiny_model: SubQModel, tiny_config: SubQConfig):
        """Cross-entropy loss must be positive."""
        train_ds, _ = make_synthetic_datasets(
            vocab_size=tiny_config.vocab_size,
            seq_len=self.SEQ_LEN,
            total_tokens=10_000,
            seed=2,
        )
        loader = DataLoader(train_ds, batch_size=self.BATCH_SIZE, shuffle=True)
        losses = self._run_steps(tiny_model, loader, self.NUM_STEPS)
        assert all(l > 0 for l in losses), (
            f"Non-positive loss encountered: {losses}"
        )

    def test_gradients_flow(self, tiny_model: SubQModel, tiny_config: SubQConfig):
        """After one backward pass, non-routing parameters should have gradients."""
        train_ds, _ = make_synthetic_datasets(
            vocab_size=tiny_config.vocab_size,
            seq_len=self.SEQ_LEN,
            total_tokens=5_000,
            seed=3,
        )
        loader = DataLoader(train_ds, batch_size=self.BATCH_SIZE)
        tiny_model.train()
        ids, labs = next(iter(loader))
        loss, _ = tiny_model(ids, labels=labs)
        loss.backward()

        # Embedding, projections, MLP should all have gradients.
        params_with_grad = {
            n for n, p in tiny_model.named_parameters()
            if p.grad is not None and p.grad.abs().max().item() > 0
        }
        # route_q / route_k intentionally have no gradient (non-differentiable top-K)
        non_routing = {
            n for n, _ in tiny_model.named_parameters()
            if "route_q" not in n and "route_k" not in n
        }
        assert non_routing.issubset(params_with_grad), (
            f"Missing gradients on: {non_routing - params_with_grad}"
        )

    def test_checkpoint_save_load(
        self, tmp_path, tiny_model: SubQModel, tiny_config: SubQConfig
    ):
        """Saving and loading a checkpoint should reproduce identical outputs."""
        import dataclasses

        train_ds, _ = make_synthetic_datasets(
            vocab_size=tiny_config.vocab_size,
            seq_len=self.SEQ_LEN,
            total_tokens=5_000,
            seed=4,
        )
        loader = DataLoader(train_ds, batch_size=self.BATCH_SIZE, shuffle=True)
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)

        # Train 5 steps
        tiny_model.train()
        data_iter = iter(loader)
        for _ in range(5):
            ids, labs = next(data_iter)
            loss, _ = tiny_model(ids, labels=labs)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # Save checkpoint
        ckpt_path = tmp_path / "test_ckpt.pt"
        torch.save(
            {
                "step": 5,
                "loss": loss.item(),
                "model_state_dict": tiny_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": dataclasses.asdict(tiny_config),
            },
            ckpt_path,
        )

        # Record output before reload
        tiny_model.eval()
        test_ids = torch.randint(0, tiny_config.vocab_size, (1, self.SEQ_LEN))
        with torch.no_grad():
            logits_before = tiny_model(test_ids)

        # Reload into fresh model
        fresh_model = SubQModel(tiny_config).eval()
        ckpt = torch.load(ckpt_path, map_location="cpu")
        fresh_model.load_state_dict(ckpt["model_state_dict"])

        with torch.no_grad():
            logits_after = fresh_model(test_ids)

        assert torch.allclose(logits_before, logits_after, atol=1e-6), (
            "Checkpoint round-trip changed model outputs."
        )
