"""OpenWebText dataset loader."""

from ..registry import register_dataset
from .base import load_hf_dataset, collect_texts
from ..dataset import TextDataset
from ...tokenizer.base import Tokenizer


@register_dataset(
    name="openwebtext",
    hf_path="Skylion007/openwebtext",
    hf_subset=None,
    text_column="text",
    description="OpenWebText: Web text corpus, 8M documents",
)
def load_openwebtext_dataset(
    tokenizer: Tokenizer,
    split: str = "train",
    seq_len: int = 512,
    stride: int | None = None,
    max_samples: int = 0,
    **kwargs,
) -> TextDataset:
    """OpenWebText 데이터셋 로드.

    Args:
        tokenizer: 토크나이저
        split: 분할 (train only)
        seq_len: 시퀀스 길이
        stride: 스트라이드 (None이면 seq_len 사용)
        max_samples: 최대 샘플 수 (0 = 전체)

    Returns:
        TextDataset

    Note:
        OpenWebText는 train split만 존재합니다.
    """
    # OpenWebText는 train split만 존재
    if split != "train":
        print(
            f"Warning: OpenWebText only has 'train' split, "
            f"using 'train' instead of '{split}'"
        )
        split = "train"

    dataset = load_hf_dataset("Skylion007/openwebtext", split=split)

    print("Loading OpenWebText...")
    full_text = collect_texts(dataset, "text", max_samples, desc="Loading")

    print("Tokenizing...")
    tokens = tokenizer.encode(full_text)
    print(f"Total tokens: {len(tokens):,}")

    return TextDataset(tokens, seq_len, stride)
