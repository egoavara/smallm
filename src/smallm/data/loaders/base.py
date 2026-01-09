"""Common utilities for dataset loaders."""

from typing import Optional, Union
from datasets import Dataset, IterableDataset, load_dataset
from tqdm import tqdm


def load_hf_dataset(
    path: str,
    name: Optional[str] = None,
    split: str = "train",
    streaming: bool = False,
) -> Union[Dataset, IterableDataset]:
    """HuggingFace 데이터셋 로드 래퍼.

    Args:
        path: HuggingFace 데이터셋 경로
        name: 서브셋 이름 (Optional)
        split: 분할 (train/validation/test)
        streaming: 스트리밍 모드 여부

    Returns:
        HuggingFace Dataset 또는 IterableDataset 객체
    """
    return load_dataset(path, name, split=split, streaming=streaming)


def collect_texts(
    dataset: Union[Dataset, IterableDataset],
    text_column: str = "text",
    max_samples: int = 0,
    desc: str = "Loading",
) -> str:
    """데이터셋에서 텍스트를 수집하여 단일 문자열로 반환.

    Args:
        dataset: HuggingFace Dataset 또는 IterableDataset (streaming)
        text_column: 텍스트 컬럼명
        max_samples: 최대 샘플 수 (0 = 전체, streaming 시 필수 권장)
        desc: tqdm 설명

    Returns:
        모든 텍스트를 '\\n'으로 연결한 문자열
    """
    # 스트리밍 모드 확인
    is_streaming = isinstance(dataset, IterableDataset)

    if is_streaming:
        # 스트리밍 모드: 전체 길이 알 수 없음
        if max_samples == 0:
            print("Warning: streaming=True with max_samples=0 may take very long!")
        total = max_samples if max_samples > 0 else None
    else:
        # 인메모리 모드
        total = min(max_samples, len(dataset)) if max_samples > 0 else len(dataset)

    texts = []
    for i, item in enumerate(tqdm(dataset, desc=desc, total=total)):
        if max_samples > 0 and i >= max_samples:
            break
        text = item[text_column]
        if text and text.strip():
            texts.append(text)

    return "\n".join(texts)
