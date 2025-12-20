"""Base Tokenizer class."""

from abc import ABC, abstractmethod
from typing import Optional
import json


class Tokenizer(ABC):
    """Abstract base class for tokenizers."""

    special_tokens: dict[str, int]
    inverse_special_tokens: dict[int, str]

    def __init__(self) -> None:
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    @abstractmethod
    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Train the tokenizer on the given text.

        Args:
            text: Training text corpus
            vocab_size: Target vocabulary size
            verbose: Whether to print progress
        """
        pass

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Encode text into token IDs.

        Args:
            text: Text to encode

        Returns:
            List of token IDs
        """
        pass

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text.

        Args:
            ids: List of token IDs

        Returns:
            Decoded text
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save tokenizer to file.

        Args:
            path: Path to save the tokenizer (without extension)
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load tokenizer from file.

        Args:
            path: Path to load the tokenizer from (without extension)
        """
        pass

    def register_special_tokens(self, special_tokens: dict[str, int]) -> None:
        """Register special tokens.

        Args:
            special_tokens: Dictionary mapping special token strings to IDs
        """
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size including special tokens."""
        return self._vocab_size + len(self.special_tokens)

    @property
    @abstractmethod
    def _vocab_size(self) -> int:
        """Return the base vocabulary size (without special tokens)."""
        pass
