"""
TiktokenDataset tests — automatically skipped when ``tiktoken`` is not installed.

Install tiktoken with ``pip install tiktoken`` to run these.
"""
import sys

import pytest

# Module-level skip: if tiktoken is unavailable, skip every test in this file.
pytest.importorskip("tiktoken", reason="tiktoken not installed — pip install tiktoken")

import torch  # noqa: E402 (after importorskip guard)

from opensubq.data import TiktokenDataset  # noqa: E402


SAMPLE_TEXT = (
    "The quick brown fox jumps over the lazy dog. " * 200
)


class TestTiktokenDataset:
    def test_from_text_basic(self):
        ds = TiktokenDataset.from_text(SAMPLE_TEXT, seq_len=32, encoding="gpt2")
        assert len(ds) > 0
        ids, labs = ds[0]
        assert ids.shape == (32,)
        assert labs.shape == (32,)
        assert ids.dtype == torch.long
        assert labs.dtype == torch.long

    def test_autoregressive_shift(self):
        ds = TiktokenDataset.from_text(SAMPLE_TEXT, seq_len=16, encoding="gpt2")
        ids, labs = ds[0]
        # labels[i] == input_ids[i+1] for i < seq_len-1
        assert torch.all(labs[:-1] == ids[1:])

    def test_from_file(self, tmp_path):
        p = tmp_path / "corpus.txt"
        p.write_text(SAMPLE_TEXT, encoding="utf-8")
        train_ds, val_ds = TiktokenDataset.from_file(str(p), seq_len=16, encoding="gpt2")
        assert len(train_ds) > 0

    def test_import_error_without_tiktoken(self):
        """TiktokenDataset.from_text must raise ImportError if tiktoken is masked."""
        orig = sys.modules.get("tiktoken")
        sys.modules["tiktoken"] = None  # type: ignore[assignment]
        try:
            with pytest.raises((ImportError, TypeError)):
                TiktokenDataset.from_text("hello world", seq_len=4)
        finally:
            if orig is None:
                sys.modules.pop("tiktoken", None)
            else:
                sys.modules["tiktoken"] = orig
