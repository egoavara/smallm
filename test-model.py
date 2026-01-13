# %% [markdown]
# # SmallM - Model Testing

# %% [markdown]
# ## 1. Configuration

# %%
import ipywidgets as widgets
from IPython.display import display

from config import config, MODELS, MODES

# Config 선택 UI
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

def update_config_info():
    config.load()  # 최신 값 로드
    mode_dropdown.value = config.mode
    model_dropdown.value = config.model_size
    config_info.value = f"""
    <div style="background: #f0f0f0; padding: 10px; border-radius: 5px; margin-top: 10px;">
        <b>Current Config</b> (build/config.json)<br>
        Mode: <code>{config.mode}</code><br>
        Model: <code>{config.model_size}</code> → seq_len={config.seq_len}
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

update_config_info()

display(widgets.VBox([
    widgets.HTML("<h3>Select Configuration</h3>"),
    widgets.HBox([mode_dropdown, model_dropdown]),
    config_info,
    widgets.HTML("<small>Changes are saved to <code>build/config.json</code> and shared with train-model.py</small>"),
]))

# %% [markdown]
# ## 2. Setup

# %%
import torch
import threading
import time
from pathlib import Path

from smallm.model import LLaMA, CONFIGS
from smallm.training import CheckpointManager

BPETokenizer = config.get_bpe_class()

print(f"PyTorch: {torch.__version__}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# %%
tokenizer_file = Path(f"{config.tokenizer_path}.model")
if not tokenizer_file.exists():
    raise FileNotFoundError(f"Tokenizer not found: {tokenizer_file}")

tokenizer = BPETokenizer()
tokenizer.load(config.tokenizer_path)
print(f"✅ Tokenizer loaded (vocab: {tokenizer.vocab_size})")


# %%
class TestingUI:
    def __init__(self, model, checkpoint_manager, tokenizer, device, model_size):
        self.model = model
        self.checkpoint_manager = checkpoint_manager
        self.tokenizer = tokenizer
        self.device = device
        self.model_size = model_size
        self.step = 0
        self.loss_history = []
        self._auto_reload_enabled = False
        self._reload_thread = None
        self._stop_reload_thread = False
        self._last_mtime = None
        self._build_ui()

    def _build_ui(self):
        param_count = self.model.count_parameters()
        param_str = f"{param_count/1e9:.1f}B" if param_count >= 1e9 else f"{param_count/1e6:.1f}M"
        dtype_str = str(next(self.model.parameters()).dtype).replace("torch.", "")
        cfg = self.model.config

        self.model_info = widgets.HTML(f"""
        <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
                    color: white; padding: 12px; border-radius: 8px; margin-bottom: 10px;">
            <b>🧪 {self.model_size.upper()}</b> ({param_str}, {dtype_str})<br>
            <small>layers={cfg.n_layers} heads={cfg.n_heads} d_model={cfg.d_model}</small>
        </div>
        """)

        self.checkpoint_dropdown = widgets.Dropdown(options=[], description="Checkpoint:", layout=widgets.Layout(width="350px"))
        self.load_btn = widgets.Button(description="Load", button_style="info", layout=widgets.Layout(width="100px"))
        self.refresh_btn = widgets.Button(description="Refresh", layout=widgets.Layout(width="80px"))
        self.load_btn.on_click(self._on_load_checkpoint)
        self.refresh_btn.on_click(self._on_refresh_checkpoints)

        self.auto_reload_checkbox = widgets.Checkbox(value=False, description="Auto-reload best.pt")
        self.auto_reload_status = widgets.HTML("<span style='color:gray'>OFF</span>")
        self.auto_reload_checkbox.observe(self._on_auto_reload_toggle, names="value")

        self.prompt_input = widgets.Textarea(value="", placeholder="Enter prompt...", description="Prompt:", layout=widgets.Layout(width="500px", height="80px"))
        self.max_tokens_input = widgets.IntText(value=100, description="Max tokens:", layout=widgets.Layout(width="150px"))
        self.temperature_input = widgets.FloatText(value=0.8, description="Temp:", layout=widgets.Layout(width="150px"))
        self.top_k_input = widgets.IntText(value=50, description="Top-K:", layout=widgets.Layout(width="150px"))
        self.generate_btn = widgets.Button(description="Generate", button_style="primary", layout=widgets.Layout(width="120px"))
        self.generate_btn.on_click(self._on_generate)

        self.log_lines = []
        self.log_output = widgets.HTML(layout=widgets.Layout(height="300px", overflow_y="auto", border="1px solid #ccc", padding="5px"))
        self.status_label = widgets.Label(value="Ready")

        self.ui = widgets.VBox([
            self.model_info,
            widgets.HTML("<b>Checkpoint</b>"),
            widgets.HBox([self.checkpoint_dropdown, self.load_btn, self.refresh_btn]),
            widgets.HBox([self.auto_reload_checkbox, self.auto_reload_status]),
            widgets.HTML("<hr><b>Generate</b>"),
            self.prompt_input,
            widgets.HBox([self.max_tokens_input, self.temperature_input, self.top_k_input]),
            self.generate_btn,
            widgets.HTML("<hr><b>Log</b>"),
            self.log_output,
            self.status_label,
        ])
        self._refresh_checkpoints()

    def _log(self, msg):
        import html
        self.log_lines.insert(0, html.escape(msg))
        self.log_lines = self.log_lines[:500]
        self.log_output.value = f"<pre style='margin:0;font-size:12px'>{'<br>'.join(self.log_lines)}</pre>"

    def _update_status(self, msg):
        self.status_label.value = msg

    def _refresh_checkpoints(self):
        checkpoints = self.checkpoint_manager.list_checkpoints()
        options = [("best.pt", "best")] + [(p.name, str(p)) for p, _, _ in checkpoints]
        self.checkpoint_dropdown.options = options
        if options:
            self.checkpoint_dropdown.value = options[0][1]

    def _on_refresh_checkpoints(self, _):
        self._refresh_checkpoints()
        self._log("Refreshed")

    def _on_load_checkpoint(self, _):
        selected = self.checkpoint_dropdown.value
        if not selected:
            return
        self._load_checkpoint(selected)

    def _load_checkpoint(self, selected):
        try:
            if selected == "best":
                result = self.checkpoint_manager.load_best(self.model, None)
                if result[0]:
                    self.step, self.loss_history = result[0], result[1]
                    self._log(f"Loaded best.pt (step {self.step})")
                    self._update_status(f"Step: {self.step}")
            else:
                self.step, self.loss_history = self.checkpoint_manager.load_checkpoint(selected, self.model, None)
                self._log(f"Loaded (step {self.step})")
                self._update_status(f"Step: {self.step}")
        except Exception as e:
            self._log(f"Error: {e}")

    def _on_auto_reload_toggle(self, change):
        if change["new"]:
            self._start_auto_reload()
        else:
            self._stop_auto_reload()

    def _start_auto_reload(self):
        if self._reload_thread:
            return
        self._stop_reload_thread = False
        best_path = self.checkpoint_manager.checkpoint_dir / "best.pt"
        self._last_mtime = best_path.stat().st_mtime if best_path.exists() else None
        self._reload_thread = threading.Thread(target=self._auto_reload_loop, daemon=True)
        self._reload_thread.start()
        self.auto_reload_status.value = "<span style='color:green'>ON</span>"

    def _stop_auto_reload(self):
        self._stop_reload_thread = True
        self._reload_thread = None
        self.auto_reload_status.value = "<span style='color:gray'>OFF</span>"

    def _auto_reload_loop(self):
        best_path = self.checkpoint_manager.checkpoint_dir / "best.pt"
        while not self._stop_reload_thread:
            try:
                if best_path.exists():
                    mtime = best_path.stat().st_mtime
                    if self._last_mtime and mtime > self._last_mtime:
                        self._last_mtime = mtime
                        self._log("best.pt changed - reloading...")
                        self._load_checkpoint("best")
                    elif not self._last_mtime:
                        self._last_mtime = mtime
            except:
                pass
            time.sleep(2)

    @torch.no_grad()
    def _on_generate(self, _):
        prompt = self.prompt_input.value
        max_tokens = self.max_tokens_input.value
        temp = self.temperature_input.value
        top_k = self.top_k_input.value

        self._log(f"--- Generate (temp={temp}, top_k={top_k}) ---")
        try:
            self.model.eval()
            tokens = self.tokenizer.encode(prompt) if prompt else [1]
            tokens = torch.tensor([tokens], device=self.device)
            seq_len = self.model.config.max_seq_len

            for _ in range(max_tokens):
                logits, _ = self.model(tokens[:, -seq_len:])
                logits = logits[:, -1, :] / temp
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, -1:]] = float("-inf")
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                tokens = torch.cat([tokens, next_token], dim=1)
                if hasattr(self.tokenizer, "eos_id") and next_token.item() == self.tokenizer.eos_id:
                    break

            self._log(f"Output: {self.tokenizer.decode(tokens[0].tolist())}")
        except Exception as e:
            self._log(f"Error: {e}")

    def display(self):
        display(self.ui)


# %%
def setup_model():
    model_config = CONFIGS[config.model_size]
    model_config.vocab_size = tokenizer.vocab_size
    model_config.max_seq_len = config.seq_len

    model = LLaMA(model_config).to(config.device)
    print(f"\n📦 Model: {config.model_size} ({model.count_parameters():,} params)")

    checkpoint_dir = f"{config.checkpoint_dir}/{config.mode}/{config.model_size}"
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        max_checkpoints=config.max_checkpoints,
        save_best=config.save_best,
        device=config.device,
    )
    return model, checkpoint_manager


# %%
if __name__ == "__main__":
    model, checkpoint_manager = setup_model()
    ui = TestingUI(model, checkpoint_manager, tokenizer, config.device, config.model_size)

    if config.auto_load_best:
        result = checkpoint_manager.load_best(model, None)
        if result[0]:
            ui.step, ui.loss_history = result[0], result[1]
            ui._update_status(f"Step: {ui.step}")
            ui._log(f"Auto-loaded best.pt (step {ui.step})")

    ui.display()
