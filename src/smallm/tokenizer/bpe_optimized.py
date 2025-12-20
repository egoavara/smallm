"""Optimized BPE Tokenizer with Numba JIT.

Uses numpy arrays + Numba for fast merge operations.
"""

import json
import heapq
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
from numba import njit, types
from numba.typed import Dict as NumbaDict
import regex
from tqdm.auto import tqdm

from .base import Tokenizer


# ============================================================
# Numba JIT functions
# ============================================================

@njit(cache=True)
def _find_and_merge_numba(
    data: np.ndarray,
    lengths: np.ndarray,
    offsets: np.ndarray,
    pair_a: int,
    pair_b: int,
    new_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find all occurrences of pair and merge them.

    Returns:
        new_data, new_lengths, new_offsets,
        removed_left, removed_right, added_pairs
    """
    n_chunks = len(lengths)

    # First pass: count merges and collect pair changes
    total_merges = 0
    removed_left_list = []
    removed_right_list = []
    added_left_list = []
    added_right_list = []

    for chunk_idx in range(n_chunks):
        start = offsets[chunk_idx]
        length = lengths[chunk_idx]

        i = 0
        while i < length - 1:
            if data[start + i] == pair_a and data[start + i + 1] == pair_b:
                total_merges += 1

                # Record removed pairs
                if i > 0:
                    removed_left_list.append((data[start + i - 1], pair_a))
                if i + 2 < length:
                    next_val = data[start + i + 2]
                    if not (next_val == pair_a and i + 3 < length and data[start + i + 3] == pair_b):
                        removed_right_list.append((pair_b, next_val))

                i += 2  # Skip merged pair
            else:
                i += 1

    # Second pass: create new data with merges applied
    new_total_len = len(data) - total_merges
    new_data = np.empty(new_total_len, dtype=np.int32)
    new_lengths = np.empty(n_chunks, dtype=np.int32)
    new_offsets = np.empty(n_chunks, dtype=np.int32)

    write_pos = 0
    for chunk_idx in range(n_chunks):
        start = offsets[chunk_idx]
        length = lengths[chunk_idx]
        new_offsets[chunk_idx] = write_pos

        chunk_start = write_pos
        i = 0
        while i < length:
            if i < length - 1 and data[start + i] == pair_a and data[start + i + 1] == pair_b:
                # Record new pairs
                if write_pos > chunk_start:
                    added_left_list.append((new_data[write_pos - 1], new_id))

                new_data[write_pos] = new_id
                write_pos += 1
                i += 2

                # Check for right pair
                if i < length:
                    added_right_list.append((new_id, data[start + i]))
            else:
                new_data[write_pos] = data[start + i]
                write_pos += 1
                i += 1

        new_lengths[chunk_idx] = write_pos - chunk_start

    # Convert lists to arrays
    n_removed_left = len(removed_left_list)
    n_removed_right = len(removed_right_list)
    n_added_left = len(added_left_list)
    n_added_right = len(added_right_list)

    removed_pairs = np.empty((n_removed_left + n_removed_right, 2), dtype=np.int32)
    added_pairs = np.empty((n_added_left + n_added_right, 2), dtype=np.int32)

    for i, (a, b) in enumerate(removed_left_list):
        removed_pairs[i, 0] = a
        removed_pairs[i, 1] = b
    for i, (a, b) in enumerate(removed_right_list):
        removed_pairs[n_removed_left + i, 0] = a
        removed_pairs[n_removed_left + i, 1] = b

    for i, (a, b) in enumerate(added_left_list):
        added_pairs[i, 0] = a
        added_pairs[i, 1] = b
    for i, (a, b) in enumerate(added_right_list):
        added_pairs[n_added_left + i, 0] = a
        added_pairs[n_added_left + i, 1] = b

    return new_data, new_lengths, new_offsets, removed_pairs, added_pairs


@njit(cache=True)
def _count_pairs_numba(
    data: np.ndarray,
    lengths: np.ndarray,
    offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Count all pairs in data.

    Returns:
        pairs (N, 2), counts (N,)
    """
    # First pass: count unique pairs
    pair_counts = {}
    n_chunks = len(lengths)

    for chunk_idx in range(n_chunks):
        start = offsets[chunk_idx]
        length = lengths[chunk_idx]

        for i in range(length - 1):
            a, b = data[start + i], data[start + i + 1]
            key = (a, b)
            if key in pair_counts:
                pair_counts[key] += 1
            else:
                pair_counts[key] = 1

    # Convert to arrays
    n_pairs = len(pair_counts)
    pairs = np.empty((n_pairs, 2), dtype=np.int32)
    counts = np.empty(n_pairs, dtype=np.int32)

    i = 0
    for (a, b), count in pair_counts.items():
        pairs[i, 0] = a
        pairs[i, 1] = b
        counts[i] = count
        i += 1

    return pairs, counts


# ============================================================
# Worker functions for preprocessing
# ============================================================

def _split_and_convert_worker(args: tuple[str, str]) -> list[list[int]]:
    """Split text and convert to byte lists."""
    text_chunk, pattern = args
    compiled = regex.compile(pattern)
    chunks = regex.findall(compiled, text_chunk)
    return [list(chunk.encode("utf-8")) for chunk in chunks if chunk]


# ============================================================
# Optimized BPE Tokenizer
# ============================================================

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class OptimizedBPE(Tokenizer):
    """Fast BPE tokenizer with Numba JIT optimization."""

    def __init__(self, pattern: str | None = None) -> None:
        super().__init__()
        self.pattern = pattern or GPT4_SPLIT_PATTERN
        self.compiled_pattern = regex.compile(self.pattern)
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {}

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Train BPE with Numba optimization."""
        assert vocab_size >= 256, "vocab_size must be at least 256"

        num_merges = vocab_size - 256
        n_workers = max(1, mp.cpu_count() - 1)

        if verbose:
            print(f"Using {n_workers} workers for preprocessing")
            print("Numba JIT enabled for merge operations")
            import sys
            sys.stdout.flush()

        # === Phase 1: Parallel preprocessing ===
        chunks = self._preprocess_parallel(text, n_workers, verbose)

        if not chunks:
            if verbose:
                print("No valid chunks found!")
            return

        # Convert to numpy format
        if verbose:
            print("Converting to numpy arrays...")
        data, lengths, offsets = self._to_numpy(chunks)
        del chunks  # Free memory

        if verbose:
            print(f"Total chunks: {len(lengths):,}")
            print(f"Total tokens: {len(data):,}")

        # === Phase 2: Initial pair counting with Numba ===
        if verbose:
            print("Counting pairs (Numba)...")
        pairs_arr, counts_arr = _count_pairs_numba(data, lengths, offsets)

        # Convert to Python dict for heap operations
        pair_counts: dict[tuple[int, int], int] = {}
        for i in range(len(pairs_arr)):
            pair_counts[(int(pairs_arr[i, 0]), int(pairs_arr[i, 1]))] = int(counts_arr[i])

        if verbose:
            print(f"Unique pairs: {len(pair_counts):,}")

        # Initialize vocabulary
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.merges = {}

        # Build max-heap
        if verbose:
            print("Building max-heap...")
        heap: list[tuple[int, tuple[int, int]]] = [
            (-count, pair) for pair, count in pair_counts.items() if count > 0
        ]
        heapq.heapify(heap)

        if verbose:
            print("Starting BPE training (Numba accelerated)...")
            import sys
            sys.stdout.flush()

        # === Phase 3: Training loop with Numba merge ===
        pbar = tqdm(
            range(num_merges),
            desc="BPE Training",
            disable=not verbose,
            miniters=10,
            dynamic_ncols=True,
            leave=True,
        )

        for i in pbar:
            # Find best pair
            pair_a, pair_b, count = self._pop_best_pair(heap, pair_counts)

            if count < 2:
                if verbose:
                    print(f"\nNo more useful pairs at iteration {i}")
                break

            new_id = 256 + i

            # Numba-accelerated merge
            data, lengths, offsets, removed, added = _find_and_merge_numba(
                data, lengths, offsets, pair_a, pair_b, new_id
            )

            # Update pair counts
            for j in range(len(removed)):
                pair = (int(removed[j, 0]), int(removed[j, 1]))
                pair_counts[pair] = pair_counts.get(pair, 1) - 1

            for j in range(len(added)):
                pair = (int(added[j, 0]), int(added[j, 1]))
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
                heapq.heappush(heap, (-pair_counts[pair], pair))

            pair_counts[(pair_a, pair_b)] = 0

            # Record merge
            self.merges[(pair_a, pair_b)] = new_id
            self.vocab[new_id] = self.vocab[pair_a] + self.vocab[pair_b]

            # Update progress bar
            if i % 10 == 0:
                try:
                    token_str = self.vocab[new_id].decode("utf-8")
                    if len(token_str) > 10:
                        token_str = token_str[:10] + "..."
                except UnicodeDecodeError:
                    token_str = f"0x{self.vocab[new_id].hex()[:8]}"

                pbar.set_postfix({
                    "token": repr(token_str),
                    "count": count,
                })

        pbar.close()
        if verbose:
            print(f"\nTraining complete! Vocab size: {len(self.vocab)}")

    def _to_numpy(self, chunks: list[list[int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert list of chunks to flat numpy array with lengths and offsets."""
        total_len = sum(len(c) for c in chunks)
        data = np.empty(total_len, dtype=np.int32)
        lengths = np.empty(len(chunks), dtype=np.int32)
        offsets = np.empty(len(chunks), dtype=np.int32)

        pos = 0
        for i, chunk in enumerate(chunks):
            offsets[i] = pos
            lengths[i] = len(chunk)
            for j, val in enumerate(chunk):
                data[pos + j] = val
            pos += len(chunk)

        return data, lengths, offsets

    def _preprocess_parallel(
        self, text: str, n_workers: int, verbose: bool
    ) -> list[list[int]]:
        """Split and convert to bytes in parallel."""
        text_len = len(text)
        segment_size = max(100000, text_len // (n_workers * 4))

        segments = []
        start = 0
        while start < text_len:
            end = min(start + segment_size, text_len)
            if end < text_len:
                newline_pos = text.find('\n', end)
                if newline_pos != -1 and newline_pos < end + 10000:
                    end = newline_pos + 1
            segments.append(text[start:end])
            start = end

        if verbose:
            print(f"Splitting into {len(segments)} segments...")
            import sys
            sys.stdout.flush()

        all_chunks: list[list[int]] = []
        args = [(seg, self.pattern) for seg in segments]

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            if verbose:
                results = list(tqdm(
                    executor.map(_split_and_convert_worker, args, chunksize=1),
                    total=len(segments),
                    desc="Preprocessing",
                    leave=False,
                ))
            else:
                results = list(executor.map(_split_and_convert_worker, args, chunksize=1))

            for chunk_list in results:
                all_chunks.extend(chunk_list)

        return all_chunks

    def _pop_best_pair(
        self,
        heap: list[tuple[int, tuple[int, int]]],
        pair_counts: dict[tuple[int, int], int],
    ) -> tuple[int, int, int]:
        """Pop best pair, skipping stale entries."""
        while heap:
            neg_count, pair = heapq.heappop(heap)
            actual_count = pair_counts.get(pair, 0)

            if actual_count == -neg_count and actual_count > 0:
                return pair[0], pair[1], actual_count

        return 0, 0, 0

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        if self.special_tokens:
            return self._encode_with_special_tokens(text)
        return self._encode_ordinary(text)

    def _encode_with_special_tokens(self, text: str) -> list[int]:
        """Encode with special token handling."""
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
        """Encode ordinary text."""
        text_chunks = regex.findall(self.compiled_pattern, text)
        ids = []
        for chunk in text_chunks:
            ids.extend(self._encode_chunk(chunk))
        return ids

    def _encode_chunk(self, chunk: str) -> list[int]:
        """Encode a single chunk."""
        ids = list(chunk.encode("utf-8"))

        while len(ids) >= 2:
            best_pair = None
            best_idx = float("inf")

            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])
                if pair in self.merges:
                    idx = self.merges[pair] - 256
                    if idx < best_idx:
                        best_idx = idx
                        best_pair = (i, pair)

            if best_pair is None:
                break

            pos, pair = best_pair
            new_id = self.merges[pair]
            ids = ids[:pos] + [new_id] + ids[pos + 2:]

        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text."""
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

        self.pattern = model_data.get("pattern", GPT4_SPLIT_PATTERN)
        self.compiled_pattern = regex.compile(self.pattern)

        self.merges = {}
        for item in model_data["merges"]:
            pair = tuple(item["pair"])
            self.merges[pair] = item["new_id"]

        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for pair, new_id in self.merges.items():
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

        if "special_tokens" in model_data:
            self.register_special_tokens(model_data["special_tokens"])

    @property
    def _vocab_size(self) -> int:
        return len(self.vocab)
