"""
Data utilities for SubQ training.

Two dataset classes are provided:

``CharDataset``
    Character-level dataset that requires **no extra dependencies** beyond
    PyTorch.  Encodes text as raw byte values (0–255), making it a perfect
    fit for the tiny sanity-check config (``vocab_size=256``).  Useful for
    rapid experimentation and CI tests.

``TiktokenDataset``
    BPE dataset backed by OpenAI's ``tiktoken`` library.  Supports the GPT-2
    (``cl100k_base``) and GPT-4 encodings out of the box, and can be pointed
    at any ``tiktoken``-registered encoding.  Requires ``tiktoken`` to be
    installed (``pip install tiktoken``).

Both classes share the same interface: they are PyTorch ``Dataset`` subclasses
that return fixed-length ``(input_ids, labels)`` pairs, where ``labels`` is
``input_ids`` shifted by one position and padded with ``-100`` at the end so
the autoregressive loss ignores the final token prediction.

Usage
-----
::

    from opensubq.data import CharDataset, make_split_loaders

    train_ds, val_ds = CharDataset.from_file("corpus.txt", seq_len=128, val_frac=0.1)
    train_loader, val_loader = make_split_loaders(train_ds, val_ds, batch_size=8)

    for input_ids, labels in train_loader:
        loss, logits = model(input_ids, labels=labels)
        ...
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


# --------------------------------------------------------------------------- #
# CharDataset — zero-dependency character-level tokenisation                   #
# --------------------------------------------------------------------------- #


class CharDataset(Dataset):
    """
    Character-level (byte-level) text dataset.

    Encodes each character as its UTF-8 byte value (0–255), so it works with
    ``vocab_size=256`` — the same as the tiny sanity-check config.  For ASCII
    corpora the encoding is lossless; for non-ASCII text only the first byte of
    each multi-byte code-point is kept (sufficient for a sanity-check run).

    Parameters
    ----------
    token_ids : list[int]
        The full pre-tokenised corpus as a flat list of integer token ids.
    seq_len   : int
        Length of each training window (context size).  The dataset returns
        ``len(token_ids) - seq_len`` non-overlapping windows; windows that
        would extend past the end of the corpus are silently dropped.
    """

    def __init__(self, token_ids: List[int], seq_len: int) -> None:
        self.data    = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len

    # ------------------------------------------------------------------ #
    # Construction helpers                                                 #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_text(
        cls,
        text: str,
        seq_len: int,
    ) -> "CharDataset":
        """Encode ``text`` at byte level and wrap in a ``CharDataset``."""
        token_ids = [b for b in text.encode("utf-8", errors="replace")]
        return cls(token_ids, seq_len)

    @classmethod
    def from_file(
        cls,
        path: str,
        seq_len: int,
        val_frac: float = 0.1,
    ) -> Tuple["CharDataset", "CharDataset"]:
        """
        Read ``path``, split into train / validation sets, return both.

        Parameters
        ----------
        path     : path to a plain-text UTF-8 file.
        seq_len  : context window length.
        val_frac : fraction of tokens to use for validation (default 10 %).

        Returns
        -------
        (train_dataset, val_dataset)
        """
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        token_ids = [b for b in text.encode("utf-8", errors="replace")]
        split = max(seq_len + 1, int(len(token_ids) * (1.0 - val_frac)))
        return cls(token_ids[:split], seq_len), cls(token_ids[split:], seq_len)

    # ------------------------------------------------------------------ #
    # Dataset protocol                                                     #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.seq_len + 1]   # (seq_len+1,)
        input_ids = chunk[:-1]                            # (seq_len,)
        labels    = chunk[1:].clone()                     # (seq_len,)
        return input_ids, labels


# --------------------------------------------------------------------------- #
# TiktokenDataset — GPT-2 / GPT-4 BPE tokenisation (requires tiktoken)        #
# --------------------------------------------------------------------------- #


class TiktokenDataset(Dataset):
    """
    BPE dataset backed by ``tiktoken``.

    Supports the GPT-2 (``gpt2``) and GPT-4 (``cl100k_base``) encodings.
    Install ``tiktoken`` with::

        pip install tiktoken

    Parameters
    ----------
    token_ids : list[int]
        The pre-tokenised corpus as a flat list of BPE token ids.
    seq_len   : int
        Training window length (number of BPE tokens per example).
    """

    def __init__(self, token_ids: List[int], seq_len: int) -> None:
        self.data    = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len

    # ------------------------------------------------------------------ #
    # Construction helpers                                                 #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_text(
        cls,
        text: str,
        seq_len: int,
        encoding: str = "gpt2",
    ) -> "TiktokenDataset":
        """
        Tokenise ``text`` with tiktoken and return a ``TiktokenDataset``.

        Parameters
        ----------
        text     : raw UTF-8 text.
        seq_len  : context window length in BPE tokens.
        encoding : tiktoken encoding name (e.g. ``"gpt2"``, ``"cl100k_base"``).
        """
        try:
            import tiktoken
        except ImportError as exc:
            raise ImportError(
                "TiktokenDataset requires the 'tiktoken' package.  "
                "Install it with: pip install tiktoken"
            ) from exc

        enc = tiktoken.get_encoding(encoding)
        token_ids = enc.encode_ordinary(text)
        return cls(token_ids, seq_len)

    @classmethod
    def from_file(
        cls,
        path: str,
        seq_len: int,
        encoding: str = "gpt2",
        val_frac: float = 0.1,
    ) -> Tuple["TiktokenDataset", "TiktokenDataset"]:
        """
        Read ``path``, tokenise with tiktoken, split train/val, return both.

        Parameters
        ----------
        path     : path to a plain-text UTF-8 file.
        seq_len  : context window length in BPE tokens.
        encoding : tiktoken encoding name (default ``"gpt2"``).
        val_frac : fraction of tokens for validation (default 10 %).

        Returns
        -------
        (train_dataset, val_dataset)
        """
        try:
            import tiktoken
        except ImportError as exc:
            raise ImportError(
                "TiktokenDataset requires the 'tiktoken' package.  "
                "Install it with: pip install tiktoken"
            ) from exc

        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()

        enc = tiktoken.get_encoding(encoding)
        token_ids = enc.encode_ordinary(text)
        split = max(seq_len + 1, int(len(token_ids) * (1.0 - val_frac)))
        return cls(token_ids[:split], seq_len), cls(token_ids[split:], seq_len)

    # ------------------------------------------------------------------ #
    # Dataset protocol                                                     #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk     = self.data[idx : idx + self.seq_len + 1]
        input_ids = chunk[:-1]
        labels    = chunk[1:].clone()
        return input_ids, labels


# --------------------------------------------------------------------------- #
# Convenience: paired DataLoaders                                               #
# --------------------------------------------------------------------------- #


def make_split_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Return a (train_loader, val_loader) pair from two pre-split datasets.

    Parameters
    ----------
    train_dataset : training split (shuffled).
    val_dataset   : validation split (not shuffled).
    batch_size    : examples per batch (default 8).
    num_workers   : DataLoader worker processes (default 0 — main process).
    pin_memory    : enable pinned memory for faster GPU transfer (default False).

    Returns
    -------
    (train_loader, val_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader


# --------------------------------------------------------------------------- #
# Synthetic corpus helper (for tests and quick demos)                          #
# --------------------------------------------------------------------------- #


def make_synthetic_corpus(
    vocab_size: int = 256,
    total_tokens: int = 50_000,
    seed: int = 42,
) -> List[int]:
    """
    Generate a reproducible pseudo-random token sequence.

    Tokens are drawn independently from ``Uniform[0, vocab_size)``.  This is
    not linguistic data, but it is enough to verify that the training loop
    runs without error and that the loss decreases as the model over-fits a
    small synthetic corpus.

    Parameters
    ----------
    vocab_size   : upper bound of token ids (exclusive).
    total_tokens : number of tokens in the corpus.
    seed         : RNG seed for reproducibility.

    Returns
    -------
    list[int] of length ``total_tokens``.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    ids = torch.randint(0, vocab_size, (total_tokens,), generator=rng)
    return ids.tolist()


def make_synthetic_datasets(
    vocab_size: int = 256,
    seq_len: int = 128,
    total_tokens: int = 50_000,
    val_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[CharDataset, CharDataset]:
    """
    Build train/val ``CharDataset`` pairs from a synthetic token sequence.

    Useful for quick experiments and CI testing where no real corpus is
    available.

    Parameters
    ----------
    vocab_size   : should match ``SubQConfig.vocab_size`` (default 256).
    seq_len      : context window length.
    total_tokens : corpus length in tokens.
    val_frac     : fraction of tokens for validation (default 10 %).
    seed         : RNG seed.

    Returns
    -------
    (train_dataset, val_dataset)
    """
    token_ids = make_synthetic_corpus(vocab_size, total_tokens, seed)
    split = max(seq_len + 1, int(total_tokens * (1.0 - val_frac)))
    return CharDataset(token_ids[:split], seq_len), CharDataset(token_ids[split:], seq_len)
