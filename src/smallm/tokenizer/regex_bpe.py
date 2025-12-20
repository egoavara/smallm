"""GPT-4 style Regex BPE Tokenizer.

This tokenizer uses regex patterns to pre-split text before applying BPE,
preventing merges across different character categories (words, numbers, punctuation).

Reference:
- https://github.com/karpathy/minbpe
- https://github.com/openai/tiktoken
"""

import json
import regex
from tqdm.auto import tqdm
from .base import Tokenizer
from .bpe import get_stats, merge


# GPT-4 style regex pattern for splitting text
# This pattern captures:
# - Contractions: 's, 'd, 'm, 't, 'll, 've, 're
# - Letters (with optional leading space)
# - Numbers (with optional leading space)
# - Non-letter/number/space (with optional leading space)
# - Whitespace that isn't followed by non-whitespace
# - Remaining whitespace
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexBPE(Tokenizer):
    """GPT-4 style BPE tokenizer with regex pre-tokenization.

    This tokenizer:
    1. Uses regex to split text into chunks (preventing cross-category merges)
    2. Applies BPE to each chunk independently
    3. Supports special tokens
    """

    def __init__(self, pattern: str | None = None) -> None:
        """Initialize the tokenizer.

        Args:
            pattern: Regex pattern for splitting. Defaults to GPT-4 pattern.
        """
        super().__init__()
        self.pattern = pattern or GPT4_SPLIT_PATTERN
        self.compiled_pattern = regex.compile(self.pattern)
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {}

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Train BPE on the given text with regex pre-tokenization.

        Args:
            text: Training text corpus
            vocab_size: Target vocabulary size (must be >= 256)
            verbose: Whether to print progress
        """
        assert vocab_size >= 256, "vocab_size must be at least 256 (for all bytes)"

        num_merges = vocab_size - 256

        # Split text using regex pattern
        text_chunks = regex.findall(self.compiled_pattern, text)

        # Convert each chunk to bytes, then to list of byte IDs
        ids_list: list[list[int]] = [
            list(chunk.encode("utf-8")) for chunk in text_chunks
        ]

        # Initialize vocabulary with all single bytes
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.merges = {}

        # Iteratively merge the most frequent pairs across all chunks
        pbar = tqdm(range(num_merges), desc="BPE Training", disable=not verbose)
        for i in pbar:
            # Count pairs across all chunks
            stats: dict[tuple[int, int], int] = {}
            for ids in ids_list:
                chunk_stats = get_stats(ids)
                for pair, count in chunk_stats.items():
                    stats[pair] = stats.get(pair, 0) + count

            if not stats:
                break

            # Find the most frequent pair
            pair = max(stats, key=stats.get)  # type: ignore
            new_id = 256 + i

            # Merge the pair in all chunks
            ids_list = [merge(ids, pair, new_id) for ids in ids_list]

            # Record the merge
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

            # Update progress bar with current token info
            try:
                token_str = self.vocab[new_id].decode("utf-8")
                if len(token_str) > 10:
                    token_str = token_str[:10] + "..."
            except UnicodeDecodeError:
                token_str = f"0x{self.vocab[new_id].hex()[:8]}"
            pbar.set_postfix({"token": repr(token_str), "count": stats[pair]})

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs with special token handling.

        Args:
            text: Text to encode

        Returns:
            List of token IDs
        """
        if self.special_tokens:
            return self._encode_with_special_tokens(text)
        return self._encode_ordinary(text)

    def _encode_with_special_tokens(self, text: str) -> list[int]:
        """Encode text handling special tokens.

        Args:
            text: Text to encode

        Returns:
            List of token IDs
        """
        # Build pattern to split on special tokens
        special_pattern = "(" + "|".join(
            regex.escape(k) for k in sorted(self.special_tokens, key=len, reverse=True)
        ) + ")"

        parts = regex.split(special_pattern, text)

        ids = []
        for part in parts:
            if part in self.special_tokens:
                ids.append(self.special_tokens[part])
            elif part:
                ids.extend(self._encode_ordinary(part))

        return ids

    def _encode_ordinary(self, text: str) -> list[int]:
        """Encode ordinary text (no special tokens) using regex chunking.

        Args:
            text: Text to encode

        Returns:
            List of token IDs
        """
        # Split text using regex pattern
        text_chunks = regex.findall(self.compiled_pattern, text)

        ids = []
        for chunk in text_chunks:
            chunk_ids = self._encode_chunk(chunk)
            ids.extend(chunk_ids)

        return ids

    def _encode_chunk(self, chunk: str) -> list[int]:
        """Encode a single chunk of text.

        Args:
            chunk: Text chunk to encode

        Returns:
            List of token IDs
        """
        chunk_bytes = chunk.encode("utf-8")
        chunk_ids = list(chunk_bytes)

        while len(chunk_ids) >= 2:
            stats = get_stats(chunk_ids)
            # Find the pair with the lowest merge priority
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            new_id = self.merges[pair]
            chunk_ids = merge(chunk_ids, pair, new_id)

        return chunk_ids

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
        - {path}.model: The merge rules and pattern
        - {path}.vocab: Human-readable vocabulary

        Args:
            path: Base path (without extension)
        """
        # Save model
        model_data = {
            "pattern": self.pattern,
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
                    token_str = f"0x{token_bytes.hex()}"
                f.write(f"{idx}\t{token_str!r}\n")

            # Also write special tokens
            for token, idx in self.special_tokens.items():
                f.write(f"{idx}\t{token!r} [SPECIAL]\n")

    def load(self, path: str) -> None:
        """Load tokenizer from files.

        Args:
            path: Base path (without extension)
        """
        # Load model
        with open(f"{path}.model", "r", encoding="utf-8") as f:
            model_data = json.load(f)

        # Set pattern
        self.pattern = model_data.get("pattern", GPT4_SPLIT_PATTERN)
        self.compiled_pattern = regex.compile(self.pattern)

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
