# %% [markdown]
# # SmallM - Tokenizer Training
#
# BPE 토크나이저 학습 스크립트

# %% [markdown]
# ## 1. Setup

# %%
from pathlib import Path
from datasets import load_dataset
from tqdm.auto import tqdm

from config import config

BPETokenizer = config.tokenizer.get_bpe_class()
print(f"Using {BPETokenizer.__name__} tokenizer")

# %% [markdown]
# ## 2. Configuration

# %%
print("=== Tokenizer Configuration ===")
print(f"  vocab_size: {config.tokenizer.vocab_size}")
print(f"  sample_size: {config.tokenizer.sample_size}")
print(f"  output_dir: {config.tokenizer.output_dir}")

# %% [markdown]
# ## 3. Load Data

# %%
print("\nLoading WikiText-103 dataset...")
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

# 샘플 제한 계산 (0이면 전체)
if config.tokenizer.sample_size == 0:
    sample_limit = len(dataset)
else:
    sample_limit = min(config.tokenizer.sample_size, len(dataset))

print(f"Sample limit: {sample_limit:,} / {len(dataset):,}")

# Generator 기반 수집
def collect_texts():
    for i, item in enumerate(dataset):
        if i >= sample_limit:
            break
        text = item["text"]
        if text.strip():
            yield text

texts = list(tqdm(collect_texts(), desc="Collecting"))
full_text = "\n".join(texts)

print(f"Collected samples: {len(texts):,}")
print(f"Total characters: {len(full_text):,}")

# %% [markdown]
# ## 4. Train Tokenizer

# %%
print(f"\nTraining BPE tokenizer (vocab_size={config.tokenizer.vocab_size})...")
print(f"Merges needed: {config.tokenizer.vocab_size - 256}")

tokenizer = BPETokenizer()
tokenizer.train(full_text, config.tokenizer.vocab_size, verbose=True)

# %% [markdown]
# ## 5. Add Special Tokens & Save

# %%
# Special tokens 등록
special_tokens = {
    "<|endoftext|>": config.tokenizer.vocab_size,
    "<|pad|>": config.tokenizer.vocab_size + 1,
}
tokenizer.register_special_tokens(special_tokens)

# 저장 (클래스명으로 파일 구분)
output_dir = Path(config.tokenizer.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
tokenizer_name = BPETokenizer.__name__
save_path = output_dir / tokenizer_name
tokenizer.save(str(save_path))
print(f"\n✅ Tokenizer saved to {save_path}.model")
print(f"   Final vocab size: {tokenizer.vocab_size}")

# %% [markdown]
# ## 6. Test Tokenizer

# %%
print("\n=== Tokenizer Test ===")
test_texts = [
    "Hello, world!",
    "This is a test of the BPE tokenizer.",
    "The quick brown fox jumps over the lazy dog.",
]

for text in test_texts:
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    print(f"\nOriginal: {text!r}")
    print(f"Tokens ({len(tokens)}): {tokens[:15]}{'...' if len(tokens) > 15 else ''}")
    print(f"Decoded: {decoded!r}")
    print(f"Match: {'✅' if text == decoded else '❌'}")

# %%
print("\n🎉 Tokenizer training complete!")
print(f"   Now run train-model.py to train the model.")
