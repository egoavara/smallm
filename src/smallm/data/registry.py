"""Dataset registry for dynamic dataset loading."""

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class DatasetInfo:
    """데이터셋 메타정보."""

    name: str
    hf_path: str
    hf_subset: Optional[str]
    text_column: str
    description: str
    default_split: str = "train"


# 데이터셋 정보 레지스트리
DATASET_INFO: dict[str, DatasetInfo] = {}

# 로더 함수 레지스트리
DATASET_LOADERS: dict[str, Callable] = {}


def register_dataset(
    name: str,
    hf_path: str,
    hf_subset: Optional[str] = None,
    text_column: str = "text",
    description: str = "",
    default_split: str = "train",
):
    """데이터셋 등록 데코레이터.

    Args:
        name: 데이터셋 이름 (wikitext, tinystories 등)
        hf_path: HuggingFace 데이터셋 경로
        hf_subset: 서브셋 이름 (Optional)
        text_column: 텍스트 컬럼명
        description: 데이터셋 설명
        default_split: 기본 분할

    Example:
        @register_dataset(
            name="wikitext",
            hf_path="wikitext",
            hf_subset="wikitext-103-raw-v1",
            description="WikiText-103: Wikipedia articles",
        )
        def load_wikitext_dataset(tokenizer, split, seq_len, ...):
            ...
    """

    def decorator(loader_fn: Callable) -> Callable:
        DATASET_INFO[name] = DatasetInfo(
            name=name,
            hf_path=hf_path,
            hf_subset=hf_subset,
            text_column=text_column,
            description=description,
            default_split=default_split,
        )
        DATASET_LOADERS[name] = loader_fn
        return loader_fn

    return decorator


def get_dataset_loader(name: str) -> Callable:
    """이름으로 데이터셋 로더 함수 반환.

    Args:
        name: 데이터셋 이름

    Returns:
        로더 함수

    Raises:
        ValueError: 알 수 없는 데이터셋 이름
    """
    if name not in DATASET_LOADERS:
        available = ", ".join(DATASET_LOADERS.keys())
        raise ValueError(f"Unknown dataset: {name}. Available: {available}")
    return DATASET_LOADERS[name]


def get_dataset_info(name: str) -> DatasetInfo:
    """이름으로 데이터셋 정보 반환.

    Args:
        name: 데이터셋 이름

    Returns:
        DatasetInfo 객체

    Raises:
        ValueError: 알 수 없는 데이터셋 이름
    """
    if name not in DATASET_INFO:
        available = ", ".join(DATASET_INFO.keys())
        raise ValueError(f"Unknown dataset: {name}. Available: {available}")
    return DATASET_INFO[name]


def list_datasets() -> list[str]:
    """등록된 데이터셋 목록 반환."""
    return list(DATASET_INFO.keys())
