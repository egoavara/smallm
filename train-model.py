# %% [markdown]
# # SmallM - Model Training

# %% [markdown]
# ## 1. Setup

# %%
import torch
import time
import ipywidgets as widgets
from pathlib import Path
from typing import Optional
from IPython.display import display, clear_output

from config import config, MODELS, MODES
from smallm.model import LLaMA, CONFIGS
from smallm.data import create_dataloader
from smallm.training import CheckpointManager, TrainingUI

BPETokenizer = config.get_bpe_class()

print(f"PyTorch: {torch.__version__}")
if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)"
    )

# %%
tokenizer_file = Path(f"{config.tokenizer_path}.model")
if not tokenizer_file.exists():
    raise FileNotFoundError(
        f"Tokenizer not found: {tokenizer_file}\nRun train-tokenizer.py first!"
    )

tokenizer = BPETokenizer()
tokenizer.load(config.tokenizer_path)
print(f"✅ Tokenizer loaded (vocab: {tokenizer.vocab_size})")


# %%
class TrainingState:
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
        self.scaler = None
        self.amp_dtype = None
        self.accumulation_step = 0

    def reset_iter(self):
        if self.train_loader is not None:
            self.train_iter = iter(self.train_loader)

    def get_batch(self):
        try:
            return next(self.train_iter)
        except (StopIteration, TypeError):
            self.reset_iter()
            return next(self.train_iter)

    def save_checkpoint(self, loss: float) -> str:
        return self.checkpoint_manager.save(
            step=self.step,
            loss=loss,
            model=self.model,
            optimizer=self.optimizer,
            loss_history=self.loss_history,
        )

    def load_best_checkpoint(self) -> bool:
        result = self.checkpoint_manager.load_best(self.model, self.optimizer)
        if result[0] is not None:
            self.step, self.loss_history = result[0], result[1]
            return True
        return False


state = TrainingState()


# %%
def setup_model():
    config.load()

    model_config = CONFIGS[config.model_size]
    model_config.vocab_size = tokenizer.vocab_size
    model_config.max_seq_len = config.seq_len

    model = LLaMA(model_config).to(config.device)

    if config.gradient_checkpointing:
        model.enable_gradient_checkpointing(True)

    print(
        f"\n📦 Model: {config.mode}/{config.model_size} ({model.count_parameters():,} params)"
    )
    if config.gradient_checkpointing:
        print(f"   Gradient Checkpointing: ON")
    if config.use_amp:
        print(f"   AMP: {config.amp_dtype}")

    state.model = model
    state.optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    if config.use_amp:
        state.amp_dtype = getattr(torch, config.amp_dtype)
        state.scaler = (
            torch.amp.GradScaler("cuda") if config.amp_dtype == "float16" else None
        )

    if config.mode == "instruct":
        base_checkpoint = Path(
            f"{config.checkpoint_dir}/base/{config.model_size}/best.pt"
        )
        if base_checkpoint.exists():
            checkpoint = torch.load(base_checkpoint, map_location=config.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"   🔄 Loaded base model from {base_checkpoint}")
            print(
                f"      (base step: {checkpoint.get('step', '?')}, loss: {checkpoint.get('loss', '?'):.4f})"
            )
        else:
            print(f"   ⚠️ Base model not found: {base_checkpoint}")
            print(f"      Train base model first, or starting from scratch.")

    checkpoint_dir = f"{config.checkpoint_dir}/{config.mode}/{config.model_size}"
    state.checkpoint_manager = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        max_checkpoints=config.max_checkpoints,
        save_best=config.save_best,
        device=config.device,
    )

    state.step = 0
    state.loss_history = []
    if config.auto_load_best and state.load_best_checkpoint():
        print(f"   Resumed from step {state.step}")

    return model


def setup_data():
    config.load()

    state.train_loader = create_dataloader(
        datasets=config.datasets,
        tokenizer=tokenizer,
        seq_len=config.seq_len,
        batch_size=config.batch_size,
    )

    state.reset_iter()
    print(f"\n📊 Dataset: {config.datasets}")
    print(
        f"   Batch: {config.batch_size} × {config.gradient_accumulation_steps} = {config.batch_size * config.gradient_accumulation_steps}"
    )
    return state.train_loader


# %%
def train_step() -> float:
    state.model.train()
    accum = config.gradient_accumulation_steps

    x, y = state.get_batch()
    x, y = x.to(config.device), y.to(config.device)

    if state.amp_dtype:
        with torch.amp.autocast("cuda", dtype=state.amp_dtype):
            _, loss = state.model(x, y)
            loss = loss / accum
    else:
        _, loss = state.model(x, y)
        loss = loss / accum

    if state.scaler:
        state.scaler.scale(loss).backward()
    else:
        loss.backward()

    state.accumulation_step += 1

    if state.accumulation_step >= accum:
        if state.scaler:
            state.scaler.unscale_(state.optimizer)
            torch.nn.utils.clip_grad_norm_(
                state.model.parameters(), config.max_grad_norm
            )
            state.scaler.step(state.optimizer)
            state.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(
                state.model.parameters(), config.max_grad_norm
            )
            state.optimizer.step()

        state.optimizer.zero_grad()
        state.accumulation_step = 0
        state.step += 1

    loss_val = loss.item() * accum
    state.loss_history.append(loss_val)
    return loss_val


def train(num_steps: int = 1000, log_interval: int = 100, save_interval: int = 500):
    if state.model is None or state.train_loader is None:
        raise RuntimeError("Call setup_model() and setup_data() first")

    state.is_training = True
    state.stop_requested = False
    start_step, start_time = state.step, time.time()

    print(f"\n🚀 Training {num_steps} steps from step {start_step}")

    try:
        for _ in range(num_steps):
            if state.stop_requested:
                break
            loss = train_step()

            if state.step % log_interval == 0:
                elapsed = time.time() - start_time
                speed = (state.step - start_step) / elapsed if elapsed > 0 else 0
                print(f"Step {state.step:6d} | Loss: {loss:.4f} | {speed:.1f} steps/s")

            if state.step % save_interval == 0:
                avg_loss = sum(state.loss_history[-100:]) / len(
                    state.loss_history[-100:]
                )
                print(f"   💾 {state.save_checkpoint(avg_loss)}")
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted")
    finally:
        state.is_training = False
        print(
            f"✅ Done. Final loss: {state.loss_history[-1]:.4f}"
            if state.loss_history
            else "✅ Done"
        )


# %%
@torch.no_grad()
def generate(
    prompt: str = "", max_tokens: int = 100, temperature: float = 0.8, top_k: int = 50
) -> str:
    if state.model is None:
        raise RuntimeError("Model not initialized")

    state.model.eval()
    tokens = tokenizer.encode(prompt) if prompt else [1]
    tokens = torch.tensor([tokens], device=config.device)

    for _ in range(max_tokens):
        logits, _ = state.model(tokens[:, -config.seq_len :])
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
# ## 2. Training UI


# %%
def create_training_ui():
    mode_dropdown = widgets.Dropdown(
        options=list(MODES.keys()),
        value=config.mode,
        description="Mode:",
        layout=widgets.Layout(width="200px"),
    )

    model_dropdown = widgets.Dropdown(
        options=list(MODELS.keys()),
        value=config.model_size,
        description="Model:",
        layout=widgets.Layout(width="200px"),
    )

    config_info = widgets.HTML()
    setup_output = widgets.Output()
    training_container = widgets.VBox()

    def update_config_info():
        config.load()
        mode_dropdown.value = config.mode
        model_dropdown.value = config.model_size
        config_info.value = f"""
        <div style="background: #f0f0f0; padding: 10px; border-radius: 5px; margin-top: 10px;">
            <b>Current Config</b><br>
            Mode: <code>{config.mode}</code> → {config.datasets}<br>
            Model: <code>{config.model_size}</code> → seq_len={config.seq_len}, batch={config.batch_size}
        </div>
        """

    def on_mode_change(change):
        config.mode = change["new"]
        update_config_info()

    def on_model_change(change):
        config.model_size = change["new"]
        update_config_info()

    mode_dropdown.observe(on_mode_change, names="value")
    model_dropdown.observe(on_model_change, names="value")

    setup_btn = widgets.Button(
        description="Setup Model & Data",
        button_style="primary",
        layout=widgets.Layout(width="200px"),
    )

    def on_setup_click(_):
        setup_btn.disabled = True
        setup_btn.description = "Setting up..."

        with setup_output:
            clear_output()
            try:
                setup_model()
                setup_data()

                def ui_train_step() -> float:
                    state.model.train()
                    accum = config.gradient_accumulation_steps

                    x, y = state.get_batch()
                    x, y = x.to(config.device), y.to(config.device)

                    if state.amp_dtype:
                        with torch.amp.autocast("cuda", dtype=state.amp_dtype):
                            _, loss = state.model(x, y)
                            loss = loss / accum
                    else:
                        _, loss = state.model(x, y)
                        loss = loss / accum

                    if state.scaler:
                        state.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    state.accumulation_step += 1

                    if state.accumulation_step >= accum:
                        if state.scaler:
                            state.scaler.unscale_(state.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                state.model.parameters(), config.max_grad_norm
                            )
                            state.scaler.step(state.optimizer)
                            state.scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(
                                state.model.parameters(), config.max_grad_norm
                            )
                            state.optimizer.step()

                        state.optimizer.zero_grad()
                        state.accumulation_step = 0

                    return loss.item() * accum

                state.ui = TrainingUI(
                    model=state.model,
                    optimizer=state.optimizer,
                    checkpoint_manager=state.checkpoint_manager,
                    tokenizer=tokenizer,
                    train_step_fn=ui_train_step,
                    device=config.device,
                    model_size=config.model_size,
                )

                if state.step > 0:
                    state.ui.set_step(state.step, state.loss_history)

                training_container.children = [state.ui.ui]
                print("\n✅ Ready to train!")

            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback

                traceback.print_exc()
            finally:
                setup_btn.disabled = False
                setup_btn.description = "Setup Model & Data"

    setup_btn.on_click(on_setup_click)

    update_config_info()

    ui = widgets.VBox(
        [
            widgets.HTML("<h3>1. Select Configuration</h3>"),
            widgets.HBox([mode_dropdown, model_dropdown]),
            config_info,
            widgets.HTML("<br>"),
            setup_btn,
            setup_output,
            widgets.HTML("<h3>2. Training</h3>"),
            training_container,
        ]
    )

    display(ui)


# %%
if __name__ == "__main__":
    create_training_ui()
