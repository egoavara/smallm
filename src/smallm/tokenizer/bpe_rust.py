"""Rust-powered BPE Tokenizer wrapper.

Uses the rust_bpe_tokenizer module for maximum performance.
"""

import json
from pathlib import Path

from tqdm.auto import tqdm

try:
    from rust_bpe_tokenizer import RustBPE as _RustBPE
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    _RustBPE = None

from .base import Tokenizer


class RustBPE(Tokenizer):
    """High-performance BPE tokenizer powered by Rust.

    This is a wrapper around the rust_bpe_tokenizer module.
    Falls back to OptimizedBPE if Rust module is not available.
    """

    def __init__(self, pattern: str | None = None) -> None:
        if not RUST_AVAILABLE:
            raise ImportError(
                "rust_bpe_tokenizer not available. "
                "Build it with: cd rust-bpe-tokenizer && maturin develop --release"
            )

        super().__init__()
        self._rust = _RustBPE(pattern)
        self.pattern = pattern
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {}

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Train BPE tokenizer using Rust implementation."""
        num_merges = vocab_size - 256

        if verbose:
            print("=" * 60)
            print("BPE Training Started")
            print("=" * 60)
            print(f"  Input text length: {len(text):,} chars")
            print(f"  Target vocab size: {vocab_size:,}")
            print(f"  Merges to perform: {num_merges:,}")
            print("-" * 60)

            pbar = None
            last_step = -1

            def progress_callback(event: dict) -> None:
                nonlocal pbar, last_step
                name = event.get("event", "")

                if name == "chunking_start":
                    print("[Phase 1/5] Chunking text...")

                elif name == "chunking_done":
                    print(f"  -> Chunks: {event['num_chunks']:,}")
                    print(f"  -> Tokens: {event['num_tokens']:,}")
                    print(f"  -> Avg tokens/chunk: {event['num_tokens'] / max(1, event['num_chunks']):.1f}")

                elif name == "flattening_start":
                    print("[Phase 2/6] Flattening into single array...")

                elif name == "flattening_done":
                    print(f"  -> Array size: {event['array_size']:,}")

                elif name == "indexing_start":
                    print("[Phase 3/6] Building pair position index...")

                elif name == "indexing_done":
                    print(f"  -> Unique pairs: {event['num_pairs']:,}")

                elif name == "heap_built":
                    print("[Phase 4/6] Building priority queue...")
                    print(f"  -> Heap size: {event['heap_size']:,}")

                elif name == "training_start":
                    print("[Phase 5/6] Training merges...")
                    print("-" * 60)
                    pbar = tqdm(total=event["num_merges"], desc="Merging", unit="merge")

                elif name == "merge":
                    if pbar is not None:
                        step = event["step"]
                        pbar.update(step - last_step)
                        last_step = step

                        token = event["token"]
                        display = repr(token[:15] + "..." if len(token) > 15 else token)
                        pbar.set_postfix_str(
                            f"id={event['new_id']}, "
                            f"pair={event['pair']}, "
                            f"token={display}, "
                            f"count={event['count']:,}"
                        )

                elif name == "no_more_pairs":
                    if pbar is not None:
                        pbar.close()
                    print(f"\n  [!] No more pairs to merge at step {event['step']}")

                elif name == "training_done":
                    if pbar is not None:
                        pbar.update(num_merges - last_step - 1)
                        pbar.close()
                    print("-" * 60)
                    print("Training Complete!")
                    print(f"  Final vocab size: {event['vocab_size']:,}")
                    print("=" * 60)

            self._rust.train(text, vocab_size, False, progress_callback)
        else:
            self._rust.train(text, vocab_size, False, None)

        # Sync merges back to Python
        self.merges = {tuple(pair): new_id for pair, new_id in self._rust.get_merges()}

        # Rebuild vocab
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for (pair_a, pair_b), new_id in self.merges.items():
            self.vocab[new_id] = self.vocab[pair_a] + self.vocab[pair_b]

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        return self._rust.encode(text)

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text."""
        return self._rust.decode(ids)

    def register_special_tokens(self, special_tokens: dict[str, int]) -> None:
        """Register special tokens."""
        super().register_special_tokens(special_tokens)
        self._rust.register_special_tokens(special_tokens)

    def save(self, path: str) -> None:
        """Save tokenizer."""
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

        with open(f"{path}.vocab", "w", encoding="utf-8") as f:
            for idx, token_bytes in sorted(self.vocab.items()):
                try:
                    token_str = token_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    token_str = f"0x{token_bytes.hex()}"
                f.write(f"{idx}\t{token_str!r}\n")

            for token, idx in self.special_tokens.items():
                f.write(f"{idx}\t{token!r} [SPECIAL]\n")

    def load(self, path: str) -> None:
        """Load tokenizer."""
        with open(f"{path}.model", "r", encoding="utf-8") as f:
            model_data = json.load(f)

        self.pattern = model_data.get("pattern")

        # Load merges
        merges_list = [
            ((item["pair"][0], item["pair"][1]), item["new_id"])
            for item in model_data["merges"]
        ]
        self._rust.set_merges(merges_list)

        # Sync to Python
        self.merges = {tuple(pair): new_id for pair, new_id in merges_list}

        # Rebuild vocab
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for (pair_a, pair_b), new_id in self.merges.items():
            self.vocab[new_id] = self.vocab[pair_a] + self.vocab[pair_b]

        if "special_tokens" in model_data:
            self.register_special_tokens(model_data["special_tokens"])

    @property
    def _vocab_size(self) -> int:
        return self._rust.vocab_size
