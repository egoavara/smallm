# %% [markdown]
# # SmallM - Model Testing
#
# LLaMA 스타일 언어 모델 테스트 스크립트
#
# **사전 요구사항**: train-model.py로 모델을 먼저 학습해야 합니다.

# %% [markdown]
# ## 1. Setup & Imports

# %%
import torch
import threading
import time
from pathlib import Path
from IPython.display import display
import ipywidgets as widgets

from smallm.model import LLaMA, CONFIGS
from smallm.training import CheckpointManager
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
print(f"  device: {config.model.device}")
print(f"  tokenizer_path: {config.tokenizer_path}")

# %% [markdown]
# ## 3. Load Tokenizer

# %%
tokenizer_file = Path(f"{config.tokenizer_path}.model")
if not tokenizer_file.exists():
    raise FileNotFoundError(
        f"Tokenizer not found at {tokenizer_file}\n"
        f"   Please run train-tokenizer.py first!"
    )

tokenizer = BPETokenizer()
tokenizer.load(config.tokenizer_path)
print(f"Tokenizer loaded from {config.tokenizer_path}")
print(f"   Vocab size: {tokenizer.vocab_size}")


# %% [markdown]
# ## 4. Testing UI


# %%
class TestingUI:
    """Jupyter 기반 모델 테스트 UI.

    Features:
        - Checkpoint loader
        - Text generation test
        - Auto-reload on best.pt change
    """

    def __init__(
        self,
        model: LLaMA,
        checkpoint_manager: CheckpointManager,
        tokenizer,
        device: str = "cpu",
        model_size: str = "unknown",
    ):
        self.model = model
        self.checkpoint_manager = checkpoint_manager
        self.tokenizer = tokenizer
        self.device = device
        self.model_size = model_size

        self.step = 0
        self.loss_history = []

        # Auto-reload state
        self._auto_reload_enabled = False
        self._reload_thread = None
        self._stop_reload_thread = False
        self._last_mtime = None

        self._build_ui()

    def _build_ui(self):
        """Build the UI components."""
        # === Model Info Section ===
        param_count = self.model.count_parameters()
        if param_count >= 1_000_000_000:
            param_str = f"{param_count / 1_000_000_000:.1f}B"
        elif param_count >= 1_000_000:
            param_str = f"{param_count / 1_000_000:.1f}M"
        else:
            param_str = f"{param_count / 1_000:.1f}K"

        dtype = next(self.model.parameters()).dtype
        dtype_str = str(dtype).replace("torch.", "")

        cfg = self.model.config
        model_info_html = f"""
        <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
                    color: white; padding: 12px 16px; border-radius: 8px; margin-bottom: 10px;">
            <div style="font-size: 18px; font-weight: bold; margin-bottom: 6px;">
                🧪 {self.model_size.upper()} <span style="font-weight: normal; font-size: 14px;">(params = {param_str}, {dtype_str})</span>
            </div>
            <div style="font-size: 12px; opacity: 0.9;">
                n_layers={cfg.n_layers} | n_heads={cfg.n_heads} | n_kv_heads={cfg.n_kv_heads} |
                d_model={cfg.d_model} | d_ff={cfg.d_ff} | vocab={cfg.vocab_size} | seq_len={cfg.max_seq_len}
            </div>
        </div>
        """
        self.model_info = widgets.HTML(value=model_info_html)

        # === Checkpoint Section ===
        self.checkpoint_dropdown = widgets.Dropdown(
            options=[], description="Checkpoint:", layout=widgets.Layout(width="350px")
        )
        self.load_btn = widgets.Button(
            description="Load",
            button_style="info",
            layout=widgets.Layout(width="100px"),
        )
        self.refresh_btn = widgets.Button(
            description="Refresh",
            button_style="",
            layout=widgets.Layout(width="80px"),
        )

        self.load_btn.on_click(self._on_load_checkpoint)
        self.refresh_btn.on_click(self._on_refresh_checkpoints)

        checkpoint_controls = widgets.HBox(
            [self.checkpoint_dropdown, self.load_btn, self.refresh_btn]
        )

        # === Auto-reload Section ===
        self.auto_reload_checkbox = widgets.Checkbox(
            value=False,
            description="Auto-reload best.pt on change",
            layout=widgets.Layout(width="300px"),
        )
        self.auto_reload_checkbox.observe(self._on_auto_reload_toggle, names="value")

        self.auto_reload_status = widgets.HTML(
            value="<span style='color: gray;'>Auto-reload: OFF</span>"
        )

        auto_reload_controls = widgets.HBox(
            [self.auto_reload_checkbox, self.auto_reload_status]
        )

        # === Generation Section ===
        self.prompt_input = widgets.Textarea(
            value="",
            placeholder="Enter prompt...",
            description="Prompt:",
            layout=widgets.Layout(width="500px", height="80px"),
        )
        self.max_tokens_input = widgets.IntText(
            value=100, description="Max tokens:", layout=widgets.Layout(width="150px")
        )
        self.temperature_input = widgets.FloatText(
            value=0.8, description="Temperature:", layout=widgets.Layout(width="150px")
        )
        self.top_k_input = widgets.IntText(
            value=50, description="Top-K:", layout=widgets.Layout(width="150px")
        )
        self.generate_btn = widgets.Button(
            description="Generate",
            button_style="primary",
            layout=widgets.Layout(width="120px"),
        )

        self.generate_btn.on_click(self._on_generate)

        gen_params = widgets.HBox(
            [self.max_tokens_input, self.temperature_input, self.top_k_input]
        )
        gen_controls = widgets.VBox(
            [self.prompt_input, gen_params, self.generate_btn]
        )

        # === Output Section ===
        self.log_lines = []
        self.log_output = widgets.HTML(
            value="",
            layout=widgets.Layout(
                height="300px", overflow_y="auto", border="1px solid #ccc", padding="5px"
            ),
        )

        # === Status Bar ===
        self.status_label = widgets.Label(value="Ready - No checkpoint loaded")

        # === Layout ===
        checkpoint_section = widgets.VBox(
            [
                widgets.HTML("<b>Checkpoint</b>"),
                checkpoint_controls,
                auto_reload_controls,
            ]
        )

        gen_section = widgets.VBox(
            [
                widgets.HTML("<b>Generate</b>"),
                gen_controls,
            ]
        )

        log_section = widgets.VBox(
            [
                widgets.HTML("<b>Log</b>"),
                self.log_output,
            ]
        )

        self.ui = widgets.VBox(
            [
                self.model_info,
                checkpoint_section,
                widgets.HTML("<hr>"),
                gen_section,
                widgets.HTML("<hr>"),
                log_section,
                self.status_label,
            ]
        )

        # Initial refresh
        self._refresh_checkpoints()

    def _log(self, message: str):
        """Prepend message to log output (newest on top)."""
        import html

        escaped = html.escape(message)
        self.log_lines.insert(0, escaped)
        if len(self.log_lines) > 500:
            self.log_lines = self.log_lines[:500]
        self._render_log()

    def _render_log(self):
        """Render log lines to HTML."""
        html_content = "<pre style='margin:0; font-family:monospace; font-size:12px;'>"
        html_content += "\n".join(self.log_lines)
        html_content += "</pre>"
        self.log_output.value = html_content

    def _update_status(self, message: str):
        """Update status bar."""
        self.status_label.value = message

    def _refresh_checkpoints(self):
        """Refresh checkpoint dropdown."""
        checkpoints = self.checkpoint_manager.list_checkpoints()
        options = [("best.pt", "best")]

        for path, _, _ in checkpoints:
            options.append((f"{path.name}", str(path)))

        self.checkpoint_dropdown.options = options
        if options:
            self.checkpoint_dropdown.value = options[0][1]

    def _on_refresh_checkpoints(self, _):
        """Handle refresh button click."""
        self._refresh_checkpoints()
        self._log("Checkpoint list refreshed")

    def _on_load_checkpoint(self, _):
        """Handle load checkpoint button click."""
        selected = self.checkpoint_dropdown.value
        if not selected:
            self._log("No checkpoint selected")
            return

        self._load_checkpoint(selected)

    def _load_checkpoint(self, selected: str):
        """Load a checkpoint."""
        try:
            if selected == "best":
                result = self.checkpoint_manager.load_best(self.model, None)
                if result[0] is not None:
                    self.step = result[0]
                    self.loss_history = result[1]
                    self._log(f"Loaded best.pt (step: {self.step})")
                    self._update_status(f"Step: {self.step} | best.pt loaded")
                else:
                    self._log("best.pt not found")
            else:
                self.step, self.loss_history = self.checkpoint_manager.load_checkpoint(
                    selected, self.model, None
                )
                self._log(f"Loaded checkpoint (step: {self.step})")
                self._update_status(f"Step: {self.step}")
        except Exception as e:
            self._log(f"Failed to load: {e}")

    def _on_auto_reload_toggle(self, change):
        """Handle auto-reload checkbox toggle."""
        if change["new"]:
            self._start_auto_reload()
        else:
            self._stop_auto_reload()

    def _start_auto_reload(self):
        """Start auto-reload thread."""
        if self._reload_thread is not None:
            return

        self._auto_reload_enabled = True
        self._stop_reload_thread = False

        # Get initial mtime
        best_path = self.checkpoint_manager.checkpoint_dir / "best.pt"
        if best_path.exists():
            self._last_mtime = best_path.stat().st_mtime
        else:
            self._last_mtime = None

        self._reload_thread = threading.Thread(target=self._auto_reload_loop, daemon=True)
        self._reload_thread.start()

        self.auto_reload_status.value = "<span style='color: green;'>Auto-reload: ON (watching best.pt)</span>"
        self._log("Auto-reload enabled - watching best.pt for changes")

    def _stop_auto_reload(self):
        """Stop auto-reload thread."""
        self._auto_reload_enabled = False
        self._stop_reload_thread = True
        self._reload_thread = None

        self.auto_reload_status.value = "<span style='color: gray;'>Auto-reload: OFF</span>"
        self._log("Auto-reload disabled")

    def _auto_reload_loop(self):
        """Background thread to watch for best.pt changes."""
        best_path = self.checkpoint_manager.checkpoint_dir / "best.pt"

        while not self._stop_reload_thread:
            try:
                if best_path.exists():
                    current_mtime = best_path.stat().st_mtime
                    if self._last_mtime is None:
                        # File newly created
                        self._last_mtime = current_mtime
                        self._log("best.pt detected - reloading...")
                        self._load_checkpoint("best")
                    elif current_mtime > self._last_mtime:
                        # File modified
                        self._last_mtime = current_mtime
                        self._log("best.pt changed - reloading...")
                        self._load_checkpoint("best")
                else:
                    self._last_mtime = None
            except Exception as e:
                self._log(f"Auto-reload error: {e}")

            time.sleep(2)  # Check every 2 seconds

    @torch.no_grad()
    def _on_generate(self, _):
        """Handle generate button click."""
        prompt = self.prompt_input.value
        max_tokens = self.max_tokens_input.value
        temperature = self.temperature_input.value
        top_k = self.top_k_input.value

        self._log(f"\n--- Generate (temp={temperature}, top_k={top_k}) ---")
        self._log(f"Prompt: {prompt if prompt else '(empty)'}")

        try:
            self.model.eval()

            if prompt:
                tokens = self.tokenizer.encode(prompt)
            else:
                tokens = (
                    [self.tokenizer.bos_id]
                    if hasattr(self.tokenizer, "bos_id")
                    else [1]
                )

            tokens = torch.tensor([tokens], device=self.device)
            seq_len = self.model.config.max_seq_len

            for _ in range(max_tokens):
                logits, _ = self.model(tokens[:, -seq_len:])
                logits = logits[:, -1, :] / temperature

                # Top-k sampling
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, -1:]] = float("-inf")

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                tokens = torch.cat([tokens, next_token], dim=1)

                if (
                    hasattr(self.tokenizer, "eos_id")
                    and next_token.item() == self.tokenizer.eos_id
                ):
                    break

            result = self.tokenizer.decode(tokens[0].tolist())
            self._log(f"Output: {result}")

        except Exception as e:
            self._log(f"Generation error: {e}")

    def display(self):
        """Display the UI."""
        display(self.ui)

    def cleanup(self):
        """Cleanup resources."""
        self._stop_auto_reload()


# %% [markdown]
# ## 5. Setup Model


# %%
def setup_model():
    """모델 초기화."""
    model_config = CONFIGS[config.model.model_size]
    model_config.vocab_size = tokenizer.vocab_size
    model_config.max_seq_len = config.model.seq_len

    model = LLaMA(model_config).to(config.model.device)

    print(f"\nModel: {config.model.model_size}")
    print(f"   Parameters: {model.count_parameters():,}")

    # CheckpointManager 초기화
    checkpoint_dir = f"{config.model.checkpoint_dir}/{config.model.model_size}"
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        max_checkpoints=config.model.max_checkpoints,
        save_best=config.model.save_best,
        device=config.model.device,
    )

    return model, checkpoint_manager


# %% [markdown]
# ## 6. Main Entry Point

# %%
if __name__ == "__main__":
    model, checkpoint_manager = setup_model()

    ui = TestingUI(
        model=model,
        checkpoint_manager=checkpoint_manager,
        tokenizer=tokenizer,
        device=config.model.device,
        model_size=config.model.model_size,
    )

    # Auto-load best.pt if available
    if config.model.auto_load_best:
        result = checkpoint_manager.load_best(model, None)
        if result[0] is not None:
            ui.step = result[0]
            ui.loss_history = result[1]
            ui._update_status(f"Step: {ui.step} | best.pt loaded")
            ui._log(f"Auto-loaded best.pt (step: {ui.step})")
        else:
            ui._log("No best.pt found")

    ui.display()
