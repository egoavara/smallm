# %% [markdown]
# # SmallM - Tokenizer Training

# %%
from pathlib import Path
import gc

from config import config, MODES
from smallm.data import iter_texts

BPETokenizer = config.get_bpe_class()

# %%
SAMPLE_RATIO = 0.3
CHUNK_SIZE = 5000

all_datasets = config.all_datasets

print("=== Tokenizer Configuration ===")
print(f"  vocab_size: {config.vocab_size}")
print(f"  sample_ratio: {SAMPLE_RATIO * 100:.0f}%")
print(f"  chunk_size: {CHUNK_SIZE}")
print(f"  datasets: {all_datasets}")

# %%
tokenizer = BPETokenizer()

# %%
print(f"\n{'=' * 50}")
print("Accumulating texts...")
print("=" * 50)

texts = []
total_chunks = 0
total_tokens = 0

for text in iter_texts(all_datasets, sample_ratio=SAMPLE_RATIO):
    texts.append(text)

    if len(texts) >= CHUNK_SIZE:
        chunk_text = "\n".join(texts)
        num_chunks, num_tokens = tokenizer.accumulate(chunk_text)
        total_chunks += num_chunks
        total_tokens += num_tokens
        texts = []
        gc.collect()

        num_pairs, total_tok = tokenizer.get_stats_info()
        print(f"  Accumulated: {total_chunks:,} chunks, {total_tokens:,} tokens, {num_pairs:,} pairs")

if texts:
    chunk_text = "\n".join(texts)
    num_chunks, num_tokens = tokenizer.accumulate(chunk_text)
    total_chunks += num_chunks
    total_tokens += num_tokens
    del texts
    gc.collect()

num_pairs, total_tok = tokenizer.get_stats_info()
print(f"\n✅ Total: {total_chunks:,} chunks, {total_tokens:,} tokens, {num_pairs:,} pairs")

# %%
print(f"\n{'=' * 50}")
print("Finalizing BPE training...")
print("=" * 50)

tokenizer.finalize(config.vocab_size, verbose=True)

# %%
base_vocab = config.vocab_size
special_tokens = {
    "<|endoftext|>": base_vocab,
    "<|pad|>": base_vocab + 1,
}

for i, token in enumerate(config.special_tokens):
    special_tokens[token] = base_vocab + 2 + i

tokenizer.register_special_tokens(special_tokens)

# Save
output_dir = Path(config.tokenizer_dir)
output_dir.mkdir(parents=True, exist_ok=True)
save_path = output_dir / BPETokenizer.__name__
tokenizer.save(str(save_path))

print(f"\n✅ Saved to {save_path}.model")
print(f"   Vocab: {base_vocab} + {len(special_tokens)} special = {tokenizer.vocab_size}")
print(f"   Special tokens: {list(special_tokens.keys())}")

# %%
print("\n=== Test ===")
test_texts = [
    "Hello, world!",
    "Once upon a time",
    "<|im_start|>user\nHello!<|im_end|>",
]
for text in test_texts:
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    print(f"{text!r} → {len(tokens)} tokens → {decoded!r} {'✅' if text == decoded else '❌'}")
