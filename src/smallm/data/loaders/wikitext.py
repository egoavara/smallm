"""WikiText dataset loader."""

from ..registry import register_dataset
from .base import load_hf_dataset, collect_texts
from ..dataset import TextDataset
from ...tokenizer.base import Tokenizer


@register_dataset(
    name="wikitext",
    hf_path="wikitext",
    hf_subset="wikitext-103-raw-v1",
    text_column="text",
    description="WikiText-103: Wikipedia articles, 103M tokens",
)
def load_wikitext_dataset(
    tokenizer: Tokenizer,
    split: str = "train",
    seq_len: int = 512,
    stride: int | None = None,
    max_samples: int = 0,
    **kwargs,
) -> TextDataset:
    """WikiText-103 데이터셋 로드.

    Args:
        tokenizer: 토크나이저
        split: 분할 (train, validation, test)
        seq_len: 시퀀스 길이
        stride: 스트라이드 (None이면 seq_len 사용)
        max_samples: 최대 샘플 수 (0 = 전체)

    Returns:
        TextDataset
    """
    dataset = load_hf_dataset("wikitext", "wikitext-103-raw-v1", split=split)

    print(f"Loading WikiText-103 {split} split...")
    full_text = collect_texts(dataset, "text", max_samples, desc="Loading")

    print("Tokenizing...")
    tokens = tokenizer.encode(full_text)
    print(f"Total tokens: {len(tokens):,}")

    return TextDataset(tokens, seq_len, stride)
