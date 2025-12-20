# %% [markdown]
# # SmallM - Shared Configuration
#
# train-tokenizer.py와 train-model.py에서 공유하는 설정

# %%
import torch
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DatasetSourceConfig:
    """개별 데이터셋 소스 설정."""

    name: str  # wikitext, tinystories, openwebtext
    weight: float = 1.0  # 샘플링 가중치
    max_samples: int = 0  # 0 = 전체


@dataclass
class DatasetConfig:
    """데이터셋 설정."""

    # 단일 데이터셋 사용 시
    name: str = "wikitext"  # wikitext, tinystories, openwebtext

    # 다중 데이터셋 혼합 사용 시 (name 대신 사용)
    sources: Optional[list[DatasetSourceConfig]] = None

    split: str = "train"
    max_samples: int = 0  # 0 = 전체


@dataclass
class TokenizerConfig:
    """토크나이저 설정."""

    vocab_size: int = 4096
    sample_size: int = 100000  # 0 = 전체, 토크나이저는 10만이면 충분
    output_dir: str = "build/tokenizer"  # 저장 디렉토리
    bpe_type: str = "rust"  # "rust" or "python"

    def get_bpe_class(self):
        """BPE 구현체 클래스 반환."""
        from smallm.tokenizer import OptimizedBPE, RustBPE

        if self.bpe_type == "rust":
            return RustBPE
        return OptimizedBPE


@dataclass
class ModelConfig:
    """모델 학습 설정."""

    # Model
    model_size: str = "small"  # tiny, small, medium
    seq_len: int = 256

    # Data
    max_samples: int = 0  # 0 = 전체 데이터 사용

    # Training
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0

    # Paths
    checkpoint_dir: str = "checkpoints"

    # Checkpoint management
    save_best: bool = True  # best.pt 저장 여부
    max_checkpoints: int = 5  # 저장할 최대 체크포인트 수 (loss 기준 상위 n개)
    auto_load_best: bool = True  # 시작 시 자동으로 best.pt 로드

    # Device
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )


@dataclass
class Config:
    """통합 설정."""

    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    @property
    def tokenizer_dir(self) -> str:
        return self.tokenizer.output_dir

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def tokenizer_path(self) -> str:
        """토크나이저 파일 경로 (확장자 제외)."""
        bpe_class = self.tokenizer.get_bpe_class()
        return f"{self.tokenizer.output_dir}/{bpe_class.__name__}"


# 기본 설정 인스턴스
config = Config()


# %%
if __name__ == "__main__":
    print("=== SmallM Configuration ===")
    print("\nTokenizer:")
    for k, v in vars(config.tokenizer).items():
        print(f"  {k}: {v}")

    print("\nModel:")
    for k, v in vars(config.model).items():
        print(f"  {k}: {v}")
