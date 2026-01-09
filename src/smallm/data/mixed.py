"""Mixed dataset for weighted sampling from multiple datasets."""

import random
from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset

from .dataset import TextDataset

if TYPE_CHECKING:
    from config import DatasetSourceConfig
    from ..tokenizer.base import Tokenizer


class MixedTextDataset(Dataset):
    """가중치 기반 다중 데이터셋 혼합.

    여러 TextDataset에서 가중치에 따라 샘플링합니다.
    """

    def __init__(
        self,
        datasets: list[TextDataset],
        weights: list[float],
    ) -> None:
        """Initialize mixed dataset.

        Args:
            datasets: TextDataset 리스트
            weights: 각 데이터셋의 샘플링 가중치
        """
        if len(datasets) != len(weights):
            raise ValueError("datasets와 weights의 길이가 일치해야 합니다.")

        if len(datasets) == 0:
            raise ValueError("최소 하나의 데이터셋이 필요합니다.")

        self.datasets = datasets
        self.weights = weights

        # weights를 정규화하여 누적 확률로 변환
        total = sum(weights)
        self.probs = [w / total for w in weights]

        # 누적 확률 계산
        self.cum_probs: list[float] = []
        cumsum = 0.0
        for p in self.probs:
            cumsum += p
            self.cum_probs.append(cumsum)

        # 전체 샘플 수
        self._total_samples = sum(len(d) for d in datasets)

        # 데이터셋 정보 출력
        print(f"\nMixedTextDataset created with {len(datasets)} datasets:")
        for i, (ds, w, p) in enumerate(zip(datasets, weights, self.probs)):
            print(f"  [{i}] {len(ds):,} samples, weight={w:.2f}, prob={p:.2%}")
        print(f"  Total: {self._total_samples:,} samples")

    def __len__(self) -> int:
        return self._total_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """가중치에 따라 데이터셋을 선택하고 샘플 반환.

        Note:
            idx는 무시되고, 가중치에 따라 랜덤하게 데이터셋과 샘플이 선택됩니다.
            DataLoader에서 shuffle=True와 함께 사용하면 됩니다.
        """
        # 가중치에 따라 데이터셋 선택
        r = random.random()
        dataset_idx = 0
        for i, cum_prob in enumerate(self.cum_probs):
            if r < cum_prob:
                dataset_idx = i
                break

        dataset = self.datasets[dataset_idx]

        # 선택된 데이터셋에서 랜덤 샘플
        sample_idx = random.randint(0, len(dataset) - 1)
        return dataset[sample_idx]


def load_mixed_dataset(
    sources: list["DatasetSourceConfig"],
    tokenizer: "Tokenizer",
    split: str = "train",
    seq_len: int = 512,
    stride: int | None = None,
) -> MixedTextDataset:
    """여러 데이터셋을 가중치 기반으로 혼합.

    Args:
        sources: DatasetSourceConfig 리스트
        tokenizer: 토크나이저
        split: 분할
        seq_len: 시퀀스 길이
        stride: 스트라이드

    Returns:
        MixedTextDataset
    """
    from .registry import get_dataset_loader

    datasets: list[TextDataset] = []
    weights: list[float] = []

    print(f"\nLoading {len(sources)} datasets for mixing...")

    for src in sources:
        print(f"\n--- Loading {src.name} (weight={src.weight}) ---")
        loader = get_dataset_loader(src.name)
        ds = loader(
            tokenizer=tokenizer,
            split=split,
            seq_len=seq_len,
            stride=stride,
            max_samples=src.max_samples,
        )
        datasets.append(ds)
        weights.append(src.weight)

    return MixedTextDataset(datasets, weights)
