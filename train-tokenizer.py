# %% [markdown]
# # SmallM - Tokenizer Training
#
# BPE 토크나이저 학습 스크립트

# %% [markdown]
# ## 1. Setup

# %%
from pathlib import Path
from tqdm.auto import tqdm

from config import config
from smallm.data.registry import get_dataset_info
from smallm.data.loaders.base import load_hf_dataset, collect_texts

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
# 데이터셋 이름 결정 (혼합 모드면 첫 번째 소스 사용)
dataset_name = (
    config.dataset.sources[0].name
    if config.dataset.sources
    else config.dataset.name
)

# 데이터셋 정보 가져오기
dataset_info = get_dataset_info(dataset_name)
mode_str = " (streaming)" if config.dataset.streaming else ""
print(f"\nLoading {dataset_info.description}{mode_str}...")

# HuggingFace 데이터셋 로드
dataset = load_hf_dataset(
    dataset_info.hf_path,
    dataset_info.hf_subset,
    split=config.dataset.split,
    streaming=config.dataset.streaming,
)

# 텍스트 수집
full_text = collect_texts(
    dataset,
    text_column=dataset_info.text_column,
    max_samples=config.tokenizer.sample_size,
    desc="Collecting",
)

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
