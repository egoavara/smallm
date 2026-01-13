# %% [markdown]
# # SmallM - Configuration

# %%
import json
import torch
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelPreset:
    """모델 크기별 프리셋."""

    seq_len: int
    batch_size: int
    gradient_accumulation_steps: int


@dataclass(frozen=True)
class ModePreset:
    """모드별 프리셋."""

    streaming: bool
    datasets: tuple[str, ...]


# ChatML special tokens (항상 포함)
CHATML_SPECIAL_TOKENS = ("<|im_start|>", "<|im_end|>")


# ========== 프리셋 정의 ==========

MODELS = {
    "tiny": ModelPreset(
        seq_len=256,
        batch_size=32,
        gradient_accumulation_steps=1,
    ),
    "small": ModelPreset(
        seq_len=256,
        batch_size=16,
        gradient_accumulation_steps=2,
    ),
    "medium": ModelPreset(
        seq_len=256,
        batch_size=8,
        gradient_accumulation_steps=4,
    ),
    "large": ModelPreset(
        seq_len=256,
        batch_size=8,
        gradient_accumulation_steps=4,
    ),
}

MODES = {
    "base": ModePreset(
        streaming=True,
        datasets=("wikitext", "tinystories", "openwebtext"),
    ),
    "instruct": ModePreset(
        streaming=True,
        datasets=("openassistant", "alpaca", "dolly", "ultrachat"),
    ),
}

CONFIG_PATH = Path("build/config.json")


# ========== Config ==========


class Config:
    """SmallM 설정. build/config.json에서 mode와 model_size를 로드합니다."""

    def __init__(self):
        self._mode: str = "base"
        self._model_size: str = "medium"
        self.load()

    def load(self):
        """build/config.json에서 설정 로드."""
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH) as f:
                data = json.load(f)
                self._mode = data.get("mode", "base")
                self._model_size = data.get("model_size", "medium")

    def save(self):
        """build/config.json에 설정 저장."""
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_PATH, "w") as f:
            json.dump({"mode": self._mode, "model_size": self._model_size}, f, indent=2)

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str):
        if value not in MODES:
            raise ValueError(f"Invalid mode: {value}. Available: {list(MODES.keys())}")
        self._mode = value
        self.save()

    @property
    def model_size(self) -> str:
        return self._model_size

    @model_size.setter
    def model_size(self, value: str):
        if value not in MODELS:
            raise ValueError(
                f"Invalid model_size: {value}. Available: {list(MODELS.keys())}"
            )
        self._model_size = value
        self.save()

    # 프리셋 접근
    @property
    def model_preset(self) -> ModelPreset:
        return MODELS[self._model_size]

    @property
    def mode_preset(self) -> ModePreset:
        return MODES[self._mode]

    # 프리셋 값 바로가기
    @property
    def seq_len(self) -> int:
        return self.model_preset.seq_len

    @property
    def batch_size(self) -> int:
        return self.model_preset.batch_size

    @property
    def gradient_accumulation_steps(self) -> int:
        return self.model_preset.gradient_accumulation_steps

    @property
    def datasets(self) -> tuple[str, ...]:
        """현재 모드의 데이터셋 목록."""
        return self.mode_preset.datasets

    @property
    def streaming(self) -> bool:
        return self.mode_preset.streaming

    @property
    def special_tokens(self) -> tuple[str, ...]:
        """ChatML special tokens (항상 동일)."""
        return CHATML_SPECIAL_TOKENS

    @property
    def all_datasets(self) -> tuple[str, ...]:
        """토크나이저 학습용: 모든 모드의 데이터셋."""
        all_ds = set()
        for preset in MODES.values():
            all_ds.update(preset.datasets)
        return tuple(sorted(all_ds))

    # 자동 설정
    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def tokenizer_dir(self) -> str:
        return "build/tokenizer"

    @property
    def checkpoint_dir(self) -> str:
        return "checkpoints"

    @property
    def tokenizer_path(self) -> str:
        from smallm.tokenizer import RustBPE

        return f"{self.tokenizer_dir}/{RustBPE.__name__}"

    def get_bpe_class(self):
        from smallm.tokenizer import RustBPE

        return RustBPE

    # 학습 상수
    vocab_size: int = 4096
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    use_amp: bool = True
    amp_dtype: str = "bfloat16"
    gradient_checkpointing: bool = True
    max_checkpoints: int = 5
    save_best: bool = True
    auto_load_best: bool = True


config = Config()


# %%
if __name__ == "__main__":
    print(f"Config path: {CONFIG_PATH}")
    print(f"mode: {config.mode} → {config.mode_preset}")
    print(f"model_size: {config.model_size} → {config.model_preset}")
    print(f"all_datasets: {config.all_datasets}")
