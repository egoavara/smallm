"""Dataset utilities for language model training."""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Iterator
from tqdm import tqdm

from ..tokenizer.base import Tokenizer


class TextDataset(Dataset):
    """Dataset for language model training.

    Takes tokenized text and creates fixed-length sequences for training.
    Each sample is a (input, target) pair where target is shifted by one position.
    """

    def __init__(
        self,
        tokens: list[int],
        seq_len: int,
        stride: int | None = None,
    ) -> None:
        """Initialize dataset.

        Args:
            tokens: List of token IDs
            seq_len: Sequence length for each sample
            stride: Stride between samples (defaults to seq_len for no overlap)
        """
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.seq_len = seq_len
        self.stride = stride or seq_len

        # Calculate number of samples
        self.n_samples = max(0, (len(tokens) - seq_len) // self.stride)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a training sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (input_ids, target_ids) each of shape (seq_len,)
        """
        start = idx * self.stride
        end = start + self.seq_len + 1  # +1 for target

        chunk = self.tokens[start:end]
        input_ids = chunk[:-1]
        target_ids = chunk[1:]

        return input_ids, target_ids


def load_wikitext(
    tokenizer: Tokenizer,
    split: str = "train",
    seq_len: int = 512,
    stride: int | None = None,
) -> TextDataset:
    """Load WikiText-103 dataset.

    Args:
        tokenizer: Tokenizer to use
        split: Dataset split ("train", "validation", "test")
        seq_len: Sequence length
        stride: Stride between samples

    Returns:
        TextDataset instance
    """
    from datasets import load_dataset

    # Load WikiText-103
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)

    # Concatenate all text
    print(f"Loading WikiText-103 {split} split...")
    texts = []
    for item in tqdm(dataset, desc="Loading"):
        text = item["text"]
        if text.strip():  # Skip empty lines
            texts.append(text)

    full_text = "\n".join(texts)

    # Tokenize
    print("Tokenizing...")
    tokens = tokenizer.encode(full_text)
    print(f"Total tokens: {len(tokens):,}")

    return TextDataset(tokens, seq_len, stride)


def create_dataloader(
    dataset: TextDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a DataLoader for training.

    Args:
        dataset: TextDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for GPU

    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop incomplete batches
    )


class InfiniteDataLoader:
    """Wraps a DataLoader to loop infinitely."""

    def __init__(self, dataloader: DataLoader) -> None:
        self.dataloader = dataloader
        self.iterator: Iterator | None = None

    def __iter__(self) -> "InfiniteDataLoader":
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.iterator is None:
            self.iterator = iter(self.dataloader)

        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            return next(self.iterator)
