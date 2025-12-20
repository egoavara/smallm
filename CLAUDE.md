# SmallM Project Instructions

## Rust BPE Tokenizer

### Build

```bash
cd rust-bpe-tokenizer
unset CONDA_PREFIX
PATH=".venv/bin:$PATH" VIRTUAL_ENV=".venv" .venv/bin/maturin develop --release
```

또는 프로젝트 루트에서:

```bash
cd /home/egoavara/git/smallm/rust-bpe-tokenizer && \
  unset CONDA_PREFIX && \
  PATH="/home/egoavara/git/smallm/.venv/bin:$PATH" \
  VIRTUAL_ENV="/home/egoavara/git/smallm/.venv" \
  /home/egoavara/git/smallm/.venv/bin/maturin develop --release
```

### Configuration

`config.py`에서 BPE 타입 선택:

```python
@dataclass
class TokenizerConfig:
    bpe_type: str = "rust"  # "rust" or "python"
```

- `"rust"`: RustBPE 사용 (빠름)
- `"python"`: OptimizedBPE 사용 (Rust 빌드 불필요)

### Usage

```python
from config import config

BPETokenizer = config.tokenizer.get_bpe_class()
tokenizer = BPETokenizer()
tokenizer.train(text, vocab_size=4096, verbose=True)
```
