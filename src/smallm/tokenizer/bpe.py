"""Basic Byte-Pair Encoding (BPE) Tokenizer.

Reference: https://github.com/karpathy/minbpe
"""

import json
from .base import Tokenizer


def get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
    """Count frequency of consecutive pairs in the list of IDs.

    Args:
        ids: List of token IDs

    Returns:
        Dictionary mapping pairs to their frequency counts
    """
    counts: dict[tuple[int, int], int] = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace all occurrences of pair with new_id in the list.

    Args:
        ids: List of token IDs
        pair: Pair of consecutive IDs to merge
        new_id: New ID to replace the pair with

    Returns:
        New list with pairs replaced
    """
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(new_id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


class BasicBPE(Tokenizer):
    """Basic Byte-Pair Encoding tokenizer.

    This is a simple BPE implementation that works on raw bytes.
    It's educational and matches the algorithm described in the original BPE paper.
    """

    def __init__(self) -> None:
        super().__init__()
        self.merges: dict[tuple[int, int], int] = {}  # (pair) -> merged token id
        self.vocab: dict[int, bytes] = {}  # token id -> token bytes

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Train BPE on the given text.

        Args:
            text: Training text corpus
            vocab_size: Target vocabulary size (must be >= 256)
            verbose: Whether to print progress
        """
        assert vocab_size >= 256, "vocab_size must be at least 256 (for all bytes)"

        num_merges = vocab_size - 256

        # Convert text to bytes
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        # Initialize vocabulary with all single bytes
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.merges = {}

        # Iteratively merge the most frequent pairs
        for i in range(num_merges):
            stats = get_stats(ids)
            if not stats:
                break

            # Find the most frequent pair
            pair = max(stats, key=stats.get)  # type: ignore
            new_id = 256 + i

            # Merge the pair
            ids = merge(ids, pair, new_id)

            # Record the merge
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

            if verbose:
                print(
                    f"merge {i + 1}/{num_merges}: {pair} -> {new_id} "
                    f"({self.vocab[new_id]!r}) had {stats[pair]} occurrences"
                )

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Text to encode

        Returns:
            List of token IDs
        """
        # Handle special tokens first
        if self.special_tokens:
            return self._encode_with_special_tokens(text)

        # Convert to bytes
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        # Apply merges
        while len(ids) >= 2:
            stats = get_stats(ids)
            # Find the pair with the lowest merge index (earliest learned merge)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            new_id = self.merges[pair]
            ids = merge(ids, pair, new_id)

        return ids

    def _encode_with_special_tokens(self, text: str) -> list[int]:
        """Encode text handling special tokens.

        Args:
            text: Text to encode

        Returns:
            List of token IDs
        """
        # Simple approach: split by special tokens, encode each part
        import re

        if not self.special_tokens:
            return self._encode_ordinary(text)

        # Create pattern to match special tokens
        special_pattern = "(" + "|".join(re.escape(k) for k in self.special_tokens) + ")"
        parts = re.split(special_pattern, text)

        ids = []
        for part in parts:
            if part in self.special_tokens:
                ids.append(self.special_tokens[part])
            elif part:
                ids.extend(self._encode_ordinary(part))

        return ids

    def _encode_ordinary(self, text: str) -> list[int]:
        """Encode ordinary text (no special tokens).

        Args:
            text: Text to encode

        Returns:
            List of token IDs
        """
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            new_id = self.merges[pair]
            ids = merge(ids, pair, new_id)

        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text.

        Args:
            ids: List of token IDs

        Returns:
            Decoded text
        """
        text_bytes = b""
        for idx in ids:
            if idx in self.inverse_special_tokens:
                text_bytes += self.inverse_special_tokens[idx].encode("utf-8")
            elif idx in self.vocab:
                text_bytes += self.vocab[idx]
            else:
                raise ValueError(f"Unknown token ID: {idx}")

        return text_bytes.decode("utf-8", errors="replace")

    def save(self, path: str) -> None:
        """Save tokenizer to files.

        Saves two files:
        - {path}.model: The merge rules
        - {path}.vocab: Human-readable vocabulary

        Args:
            path: Base path (without extension)
        """
        # Save model (merges)
        model_data = {
            "merges": [
                {"pair": list(pair), "new_id": new_id}
                for pair, new_id in self.merges.items()
            ],
            "special_tokens": self.special_tokens,
        }
        with open(f"{path}.model", "w", encoding="utf-8") as f:
            json.dump(model_data, f, indent=2)

        # Save vocabulary (human-readable)
        with open(f"{path}.vocab", "w", encoding="utf-8") as f:
            for idx, token_bytes in sorted(self.vocab.items()):
                try:
                    token_str = token_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    token_str = token_bytes.hex()
                f.write(f"{idx}\t{token_str!r}\n")

    def load(self, path: str) -> None:
        """Load tokenizer from files.

        Args:
            path: Base path (without extension)
        """
        # Load model
        with open(f"{path}.model", "r", encoding="utf-8") as f:
            model_data = json.load(f)

        # Rebuild merges
        self.merges = {}
        for item in model_data["merges"]:
            pair = tuple(item["pair"])
            self.merges[pair] = item["new_id"]

        # Rebuild vocabulary
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for pair, new_id in self.merges.items():
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

        # Load special tokens
        if "special_tokens" in model_data:
            self.register_special_tokens(model_data["special_tokens"])

    @property
    def _vocab_size(self) -> int:
        """Return the base vocabulary size."""
        return len(self.vocab)
