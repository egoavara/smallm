# %% [markdown]
# # SmallM - Model Training
#
# LLaMA 스타일 언어 모델 학습 스크립트
#
# **사전 요구사항**: train-tokenizer.py로 토크나이저를 먼저 학습해야 합니다.

# %% [markdown]
# ## 1. Setup & Imports

# %%
import os
import torch
from pathlib import Path
from IPython.display import display, clear_output
import ipywidgets as widgets
from typing import Optional
import time

from smallm.model import LLaMA, CONFIGS
from smallm.data import (
    load_dataset_by_name,
    load_mixed_dataset,
    create_dataloader,
    load_streaming_dataset,
    load_streaming_mixed_dataset,
    create_streaming_dataloader,
)
from smallm.training import CheckpointManager, TrainingUI
from config import config

# config에서 설정된 BPE 클래스 사용
BPETokenizer = config.tokenizer.get_bpe_class()

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# %% [markdown]
# ## 2. Configuration

# %%
print("=== Model Configuration ===")
print(f"  model_size: {config.model.model_size}")
print(f"  seq_len: {config.model.seq_len}")
print(f"  batch_size: {config.model.batch_size}")
print(f"  learning_rate: {config.model.learning_rate}")
print(f"  device: {config.model.device}")
print(f"  tokenizer_path: {config.tokenizer_path}")
print(f"  save_best: {config.model.save_best}")
print(f"  max_checkpoints: {config.model.max_checkpoints}")
print(f"  auto_load_best: {config.model.auto_load_best}")

# %% [markdown]
# ## 3. Load Tokenizer

# %%
tokenizer_file = Path(f"{config.tokenizer_path}.model")
if not tokenizer_file.exists():
    raise FileNotFoundError(
        f"❌ Tokenizer not found at {tokenizer_file}\n"
        f"   Please run train-tokenizer.py first!"
    )

tokenizer = BPETokenizer()
tokenizer.load(config.tokenizer_path)
print(f"✅ Tokenizer loaded from {config.tokenizer_path}")
print(f"   Vocab size: {tokenizer.vocab_size}")

# %% [markdown]
# ## 4. Training State


# %%
class TrainingState:
    """학습 상태 관리 클래스."""

    def __init__(self):
        self.model: Optional[LLaMA] = None
        self.optimizer: Optional[torch.optim.AdamW] = None
        self.train_loader = None
        self.train_iter = None
        self.checkpoint_manager: Optional[CheckpointManager] = None
        self.ui: Optional[TrainingUI] = None

        self.step = 0
        self.loss_history = []

        self.is_training = False
        self.stop_requested = False

    def reset_iter(self):
        if self.train_loader:
            self.train_iter = iter(self.train_loader)

    def get_batch(self):
        try:
            return next(self.train_iter)
        except (StopIteration, TypeError):
            self.reset_iter()
            return next(self.train_iter)

    def save_checkpoint(self, loss: float) -> str:
        """체크포인트 저장."""
        return self.checkpoint_manager.save(
            step=self.step,
            loss=loss,
            model=self.model,
            optimizer=self.optimizer,
            loss_history=self.loss_history,
        )

    def load_best_checkpoint(self) -> bool:
        """best.pt 로드. 성공 여부 반환."""
        result = self.checkpoint_manager.load_best(self.model, self.optimizer)
        if result[0] is not None:
            self.step = result[0]
            self.loss_history = result[1]
            return True
        return False

    def load_checkpoint(self, path: str):
        """특정 체크포인트 로드."""
        self.step, self.loss_history = self.checkpoint_manager.load_checkpoint(
            path, self.model, self.optimizer
        )
        return self.step


state = TrainingState()
print("Training state initialized.")

# %% [markdown]
# ## 5. Model & Data Setup


# %%
def setup_model():
    """모델 초기화."""
    model_config = CONFIGS[config.model.model_size]
    model_config.vocab_size = tokenizer.vocab_size
    model_config.max_seq_len = config.model.seq_len

    model = LLaMA(model_config).to(config.model.device)

    print(f"\n📦 Model: {config.model.model_size}")
    print(f"   Parameters: {model.count_parameters():,}")

    state.model = model
    state.optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.model.learning_rate,
        weight_decay=config.model.weight_decay,
    )

    # CheckpointManager 초기화 (model_size를 경로에 포함)
    checkpoint_dir = f"{config.model.checkpoint_dir}/{config.model.model_size}"
    state.checkpoint_manager = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        max_checkpoints=config.model.max_checkpoints,
        save_best=config.model.save_best,
        device=config.model.device,
    )

    # auto_load_best가 True이면 best.pt 자동 로드 시도
    if config.model.auto_load_best:
        if state.load_best_checkpoint():
            print(f"   Resuming from step {state.step}")
        else:
            print("   No best.pt found, starting fresh")

    return model


def setup_data(split: str = "train"):
    """데이터 로더 설정."""
    is_streaming = config.dataset.streaming

    if is_streaming:
        # 스트리밍 모드 (메모리 효율적, 셔플 버퍼로 랜덤성 보장)
        if config.dataset.sources:
            dataset = load_streaming_mixed_dataset(
                sources=config.dataset.sources,
                tokenizer=tokenizer,
                split=split,
                seq_len=config.model.seq_len,
                buffer_size=config.dataset.buffer_size,
                shuffle_buffer_size=config.dataset.shuffle_buffer_size,
            )
            dataset_name = "Mixed (streaming)"
        else:
            dataset = load_streaming_dataset(
                name=config.dataset.name,
                tokenizer=tokenizer,
                split=split,
                seq_len=config.model.seq_len,
                buffer_size=config.dataset.buffer_size,
                shuffle_buffer_size=config.dataset.shuffle_buffer_size,
            )
            dataset_name = f"{config.dataset.name} (streaming)"

        state.train_loader = create_streaming_dataloader(
            dataset,
            batch_size=config.model.batch_size,
        )
    else:
        # 인메모리 모드 (기존 방식)
        if config.dataset.sources:
            dataset = load_mixed_dataset(
                sources=config.dataset.sources,
                tokenizer=tokenizer,
                split=split,
                seq_len=config.model.seq_len,
            )
            dataset_name = "Mixed"
        else:
            dataset = load_dataset_by_name(
                name=config.dataset.name,
                tokenizer=tokenizer,
                split=split,
                seq_len=config.model.seq_len,
                max_samples=config.dataset.max_samples,
            )
            dataset_name = config.dataset.name

        state.train_loader = create_dataloader(
            dataset,
            batch_size=config.model.batch_size,
            shuffle=True,
        )

    state.reset_iter()

    print(f"\n📊 Dataset ({dataset_name}): {len(dataset):,} samples")
    print(f"   Batch size: {config.model.batch_size}")
    if not is_streaming:
        print(f"   Steps per epoch: {len(state.train_loader):,}")
    else:
        print("   Mode: Streaming (dynamic loading)")

    return state.train_loader


# %% [markdown]
# ## 6. Training Functions


# %%
def train_step() -> float:
    """단일 학습 스텝."""
    state.model.train()

    x, y = state.get_batch()
    x, y = x.to(config.model.device), y.to(config.model.device)

    state.optimizer.zero_grad()
    _, loss = state.model(x, y)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(state.model.parameters(), config.model.max_grad_norm)
    state.optimizer.step()

    state.step += 1
    loss_val = loss.item()
    state.loss_history.append(loss_val)

    return loss_val


def train(
    num_steps: int = 1000,
    log_interval: int = 100,
    save_interval: int = 500,
    verbose: bool = True,
):
    """학습 루프."""
    if state.model is None:
        raise RuntimeError("Model not initialized. Call setup_model() first.")
    if state.train_loader is None:
        raise RuntimeError("Data not loaded. Call setup_data() first.")

    state.is_training = True
    state.stop_requested = False
    start_step = state.step
    start_time = time.time()

    print(f"\n🚀 Training for {num_steps} steps (from step {start_step})")
    print(f"   Log interval: {log_interval}, Save interval: {save_interval}")
    print("-" * 50)

    try:
        for _ in range(num_steps):
            if state.stop_requested:
                print("\n⏹️ Training stopped by user")
                break

            loss = train_step()

            if state.step % log_interval == 0:
                elapsed = time.time() - start_time
                steps_done = state.step - start_step
                steps_per_sec = steps_done / elapsed if elapsed > 0 else 0

                if verbose:
                    print(
                        f"Step {state.step:6d} | Loss: {loss:.4f} | "
                        f"Speed: {steps_per_sec:.1f} steps/s"
                    )

            if state.step % save_interval == 0:
                # 최근 100스텝의 평균 loss 사용
                recent_losses = state.loss_history[-100:]
                avg_loss = sum(recent_losses) / len(recent_losses)
                saved = state.save_checkpoint(avg_loss)
                if verbose:
                    print(f"   💾 Saved: {saved}")

    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted")

    finally:
        state.is_training = False
        elapsed = time.time() - start_time
        steps_done = state.step - start_step
        print("-" * 50)
        print(f"✅ Completed {steps_done} steps in {elapsed:.1f}s")

        if state.loss_history:
            final_loss = state.loss_history[-1]
            print(f"   Final loss: {final_loss:.4f}")


# %% [markdown]
# ## 7. Generation


# %%
@torch.no_grad()
def generate(
    prompt: str = "",
    max_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
) -> str:
    """텍스트 생성."""
    if state.model is None:
        raise RuntimeError("Model not initialized.")

    state.model.eval()

    if prompt:
        tokens = tokenizer.encode(prompt)
    else:
        tokens = [tokenizer.bos_id] if hasattr(tokenizer, "bos_id") else [1]

    tokens = torch.tensor([tokens], device=config.model.device)

    for _ in range(max_tokens):
        logits, _ = state.model(tokens[:, -config.model.seq_len :])
        logits = logits[:, -1, :] / temperature

        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, -1:]] = float("-inf")

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        tokens = torch.cat([tokens, next_token], dim=1)

        if hasattr(tokenizer, "eos_id") and next_token.item() == tokenizer.eos_id:
            break

    return tokenizer.decode(tokens[0].tolist())


# %% [markdown]
# ## 8. Interactive UI (Jupyter)


# %%
def create_training_ui():
    """Jupyter용 학습 UI 생성 및 표시."""
    if state.model is None:
        raise RuntimeError("Model not initialized. Call setup_model() first.")
    if state.train_loader is None:
        raise RuntimeError("Data not loaded. Call setup_data() first.")

    def train_step_fn() -> float:
        """단일 학습 스텝 (UI용)."""
        state.model.train()
        x, y = state.get_batch()
        x, y = x.to(config.model.device), y.to(config.model.device)

        state.optimizer.zero_grad()
        _, loss = state.model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(state.model.parameters(), config.model.max_grad_norm)
        state.optimizer.step()

        return loss.item()

    state.ui = TrainingUI(
        model=state.model,
        optimizer=state.optimizer,
        checkpoint_manager=state.checkpoint_manager,
        tokenizer=tokenizer,
        train_step_fn=train_step_fn,
        device=config.model.device,
        model_size=config.model.model_size,
    )

    # best.pt에서 로드된 경우 step 동기화
    if state.step > 0:
        state.ui.set_step(state.step, state.loss_history)

    state.ui.display()


# %% [markdown]
# ## 9. Main Entry Point

# %%
def is_jupyter() -> bool:
    """Jupyter 환경인지 확인."""
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


if __name__ == "__main__":
    # 모델 및 데이터 설정
    setup_model()
    setup_data()

    # UI 표시
    create_training_ui()
