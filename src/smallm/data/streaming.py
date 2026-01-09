"""Streaming dataset for memory-efficient training."""

from typing import Iterator, Optional, TYPE_CHECKING
import random

import torch
from torch.utils.data import IterableDataset, DataLoader

if TYPE_CHECKING:
    from ..tokenizer.base import Tokenizer


class StreamingTextDataset(IterableDataset):
    """스트리밍 기반 텍스트 데이터셋.

    데이터를 배치 단위로 로드하고 토크나이징하여 메모리 효율적으로 학습합니다.
    셔플 버퍼를 사용하여 랜덤성을 보장합니다.
    """

    def __init__(
        self,
        hf_path: str,
        hf_subset: Optional[str],
        tokenizer: "Tokenizer",
        split: str = "train",
        seq_len: int = 512,
        text_column: str = "text",
        buffer_size: int = 10000,
        shuffle_buffer_size: int = 1000,
    ) -> None:
        """Initialize streaming dataset.

        Args:
            hf_path: HuggingFace 데이터셋 경로
            hf_subset: 서브셋 이름 (Optional)
            tokenizer: 토크나이저
            split: 분할
            seq_len: 시퀀스 길이
            text_column: 텍스트 컬럼명
            buffer_size: 토큰 버퍼 크기 (이 크기만큼 모아서 시퀀스 생성)
            shuffle_buffer_size: 셔플 버퍼 크기 (시퀀스를 모아서 랜덤 추출)
        """
        self.hf_path = hf_path
        self.hf_subset = hf_subset
        self.tokenizer = tokenizer
        self.split = split
        self.seq_len = seq_len
        self.text_column = text_column
        self.buffer_size = buffer_size
        self.shuffle_buffer_size = shuffle_buffer_size

        # 추정 샘플 수 (정확한 계산은 불가능하므로 추정치 사용)
        self._estimated_samples: Optional[int] = None

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """데이터셋 이터레이터 (셔플 버퍼 사용)."""
        from datasets import load_dataset

        # 스트리밍 모드로 데이터셋 로드
        dataset = load_dataset(
            self.hf_path,
            self.hf_subset,
            split=self.split,
            streaming=True,
        )

        # 토큰 버퍼
        token_buffer: list[int] = []
        # 셔플 버퍼 (시퀀스들을 모아서 랜덤 추출)
        shuffle_buffer: list[tuple[torch.Tensor, torch.Tensor]] = []

        for item in dataset:
            text = item[self.text_column]
            if not text or not text.strip():
                continue

            # 텍스트 토크나이징
            tokens = self.tokenizer.encode(text)
            token_buffer.extend(tokens)

            # 버퍼가 충분히 차면 시퀀스 생성
            while len(token_buffer) >= self.seq_len + 1:
                seq = token_buffer[: self.seq_len + 1]
                token_buffer = token_buffer[self.seq_len :]

                input_ids = torch.tensor(seq[:-1], dtype=torch.long)
                target_ids = torch.tensor(seq[1:], dtype=torch.long)

                # 셔플 버퍼에 추가
                shuffle_buffer.append((input_ids, target_ids))

                # 셔플 버퍼가 가득 차면 랜덤하게 하나 추출
                if len(shuffle_buffer) >= self.shuffle_buffer_size:
                    idx = random.randint(0, len(shuffle_buffer) - 1)
                    yield shuffle_buffer.pop(idx)

        # 남은 셔플 버퍼 비우기 (랜덤 순서로)
        random.shuffle(shuffle_buffer)
        for sample in shuffle_buffer:
            yield sample

    def __len__(self) -> int:
        """추정 샘플 수 반환 (정확하지 않음)."""
        if self._estimated_samples is None:
            # 추정치: 일반적인 데이터셋 기준
            self._estimated_samples = 100000
        return self._estimated_samples


class StreamingMixedDataset(IterableDataset):
    """스트리밍 기반 혼합 데이터셋.

    여러 스트리밍 데이터셋을 라운드로빈 방식으로 인터리빙합니다.
    모든 데이터셋의 모든 데이터를 한 번씩 읽습니다.
    """

    def __init__(
        self,
        datasets: list[StreamingTextDataset],
        weights: list[float],
    ) -> None:
        """Initialize mixed streaming dataset.

        Args:
            datasets: StreamingTextDataset 리스트
            weights: 각 데이터셋의 샘플링 가중치 (배치 내 비율 결정)
        """
        if len(datasets) != len(weights):
            raise ValueError("datasets와 weights의 길이가 일치해야 합니다.")

        self.datasets = datasets
        self.weights = weights

        # weights를 정수 비율로 변환 (예: [0.6, 0.4] -> [3, 2])
        min_weight = min(weights)
        self.ratios = [max(1, int(w / min_weight)) for w in weights]

        print(f"\nStreamingMixedDataset created with {len(datasets)} datasets:")
        for i, (ds, w, r) in enumerate(zip(datasets, weights, self.ratios)):
            print(f"  [{i}] {ds.hf_path}, weight={w:.2f}, ratio={r}")

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """라운드로빈으로 모든 데이터셋의 모든 데이터를 순회."""
        # 각 데이터셋의 이터레이터 생성
        iterators = [iter(ds) for ds in self.datasets]
        active = [True] * len(iterators)

        # 각 데이터셋에서 ratio만큼 샘플을 가져오는 라운드로빈
        while any(active):
            for i, (it, ratio) in enumerate(zip(iterators, self.ratios)):
                if not active[i]:
                    continue

                # 해당 데이터셋에서 ratio 개수만큼 샘플 가져오기
                for _ in range(ratio):
                    try:
                        yield next(it)
                    except StopIteration:
                        active[i] = False
                        break

    def __len__(self) -> int:
        """추정 샘플 수."""
        return sum(len(ds) for ds in self.datasets)


def load_streaming_dataset(
    name: str,
    tokenizer: "Tokenizer",
    split: str = "train",
    seq_len: int = 512,
    buffer_size: int = 10000,
    shuffle_buffer_size: int = 1000,
    **kwargs,
) -> StreamingTextDataset:
    """스트리밍 데이터셋 로드.

    Args:
        name: 데이터셋 이름
        tokenizer: 토크나이저
        split: 분할
        seq_len: 시퀀스 길이
        buffer_size: 토큰 버퍼 크기
        shuffle_buffer_size: 셔플 버퍼 크기 (랜덤성 보장)
        **kwargs: 추가 옵션

    Returns:
        StreamingTextDataset
    """
    from .registry import get_dataset_info

    info = get_dataset_info(name)

    return StreamingTextDataset(
        hf_path=info.hf_path,
        hf_subset=info.hf_subset,
        tokenizer=tokenizer,
        split=split,
        seq_len=seq_len,
        text_column=info.text_column,
        buffer_size=buffer_size,
        shuffle_buffer_size=shuffle_buffer_size,
    )


def load_streaming_mixed_dataset(
    sources: list,  # list[DatasetSourceConfig]
    tokenizer: "Tokenizer",
    split: str = "train",
    seq_len: int = 512,
    buffer_size: int = 10000,
    shuffle_buffer_size: int = 1000,
) -> StreamingMixedDataset:
    """스트리밍 혼합 데이터셋 로드.

    Args:
        sources: DatasetSourceConfig 리스트
        tokenizer: 토크나이저
        split: 분할
        seq_len: 시퀀스 길이
        buffer_size: 토큰 버퍼 크기
        shuffle_buffer_size: 셔플 버퍼 크기 (랜덤성 보장)

    Returns:
        StreamingMixedDataset
    """
    datasets: list[StreamingTextDataset] = []
    weights: list[float] = []

    print(f"\nLoading {len(sources)} streaming datasets for mixing...")

    for src in sources:
        print(f"\n--- Setting up {src.name} (weight={src.weight}) ---")
        ds = load_streaming_dataset(
            name=src.name,
            tokenizer=tokenizer,
            split=split,
            seq_len=seq_len,
            buffer_size=buffer_size,
            shuffle_buffer_size=shuffle_buffer_size,
        )
        datasets.append(ds)
        weights.append(src.weight)

    return StreamingMixedDataset(datasets, weights)


def create_streaming_dataloader(
    dataset: IterableDataset,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """스트리밍 데이터셋용 DataLoader 생성.

    Args:
        dataset: IterableDataset
        batch_size: 배치 크기
        num_workers: 워커 수
        pin_memory: GPU 메모리 핀 여부

    Returns:
        DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        # IterableDataset에서는 shuffle 사용 불가
    )
