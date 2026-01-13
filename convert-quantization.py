# %% [markdown]
# # SmallM - Model Quantization
#
# FP16 및 INT8 양자화 변환 스크립트
#
# **사전 요구사항**: train-model.py로 모델을 먼저 학습해야 합니다.

# %% [markdown]
# ## 1. Setup & Imports

# %%
import torch
from pathlib import Path
from IPython.display import display
import ipywidgets as widgets

from smallm.model import LLaMA, CONFIGS
from smallm.training import CheckpointManager
from smallm.quantization import (
    convert_to_fp16,
    save_fp16_checkpoint,
    quantize_dynamic_int8,
    save_int8_checkpoint,
)
from smallm.quantization.utils import get_model_size_str, get_file_size_str
from config import config

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# %% [markdown]
# ## 2. Configuration

# %%
print("=== Model Configuration ===")
print(f"  model_size: {config.model.model_size}")
print(f"  seq_len: {config.model.seq_len}")
print(f"  device: {config.model.device}")

# %% [markdown]
# ## 3. Quantization UI


# %%
class QuantizationUI:
    """Jupyter 기반 모델 양자화 UI.

    Features:
        - Checkpoint selection
        - FP16 / INT8 quantization options
        - Conversion progress and logging
        - Size comparison display
    """

    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        model_size: str = "unknown",
        output_dir: str = "quantized",
    ):
        self.checkpoint_manager = checkpoint_manager
        self.model_size = model_size
        self.output_dir = Path(output_dir) / model_size

        self.model = None
        self.original_checkpoint = None

        self._build_ui()

    def _build_ui(self):
        """Build the UI components."""
        # === Header Section ===
        header_html = f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; padding: 12px 16px; border-radius: 8px; margin-bottom: 10px;">
            <div style="font-size: 18px; font-weight: bold; margin-bottom: 6px;">
                Quantization Converter
            </div>
            <div style="font-size: 12px; opacity: 0.9;">
                Model: {self.model_size.upper()} | Output: {self.output_dir}
            </div>
        </div>
        """
        self.header = widgets.HTML(value=header_html)

        # === Model Info Section (updated after loading) ===
        self.model_info = widgets.HTML(
            value="<div style='color: gray;'>No model loaded</div>"
        )

        # === Checkpoint Section ===
        self.checkpoint_dropdown = widgets.Dropdown(
            options=[],
            description="Checkpoint:",
            layout=widgets.Layout(width="350px"),
        )
        self.refresh_btn = widgets.Button(
            description="Refresh",
            button_style="",
            layout=widgets.Layout(width="80px"),
        )
        self.load_btn = widgets.Button(
            description="Load",
            button_style="info",
            layout=widgets.Layout(width="100px"),
        )

        self.refresh_btn.on_click(self._on_refresh_checkpoints)
        self.load_btn.on_click(self._on_load_checkpoint)

        checkpoint_controls = widgets.HBox(
            [self.checkpoint_dropdown, self.load_btn, self.refresh_btn]
        )

        # === Quantization Options Section ===
        self.fp16_checkbox = widgets.Checkbox(
            value=True,
            description="FP16 (Half Precision)",
            layout=widgets.Layout(width="250px"),
        )
        self.int8_checkbox = widgets.Checkbox(
            value=True,
            description="INT8 (Dynamic Quantization)",
            layout=widgets.Layout(width="250px"),
        )

        quant_options = widgets.VBox([self.fp16_checkbox, self.int8_checkbox])

        # === Convert Button ===
        self.convert_btn = widgets.Button(
            description="Convert",
            button_style="primary",
            layout=widgets.Layout(width="150px"),
            disabled=True,
        )
        self.convert_btn.on_click(self._on_convert)

        # === Progress Bar ===
        self.progress = widgets.FloatProgress(
            value=0,
            min=0,
            max=100,
            description="Progress:",
            bar_style="info",
            layout=widgets.Layout(width="400px"),
        )
        self.progress.layout.visibility = "hidden"

        # === Output Section ===
        self.log_lines = []
        self.log_output = widgets.HTML(
            value="",
            layout=widgets.Layout(
                height="250px",
                overflow_y="auto",
                border="1px solid #ccc",
                padding="5px",
            ),
        )

        # === Status Bar ===
        self.status_label = widgets.Label(value="Ready - Select a checkpoint to load")

        # === Layout ===
        checkpoint_section = widgets.VBox(
            [
                widgets.HTML("<b>Source Checkpoint</b>"),
                checkpoint_controls,
            ]
        )

        options_section = widgets.VBox(
            [
                widgets.HTML("<b>Quantization Options</b>"),
                quant_options,
                self.convert_btn,
                self.progress,
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
                self.header,
                self.model_info,
                widgets.HTML("<hr>"),
                checkpoint_section,
                widgets.HTML("<hr>"),
                options_section,
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

    def _update_model_info(self):
        """Update model info display."""
        if self.model is None:
            self.model_info.value = "<div style='color: gray;'>No model loaded</div>"
            return

        param_count = self.model.count_parameters()
        if param_count >= 1_000_000_000:
            param_str = f"{param_count / 1_000_000_000:.1f}B"
        elif param_count >= 1_000_000:
            param_str = f"{param_count / 1_000_000:.1f}M"
        else:
            param_str = f"{param_count / 1_000:.1f}K"

        dtype = next(self.model.parameters()).dtype
        dtype_str = str(dtype).replace("torch.", "")
        model_size = get_model_size_str(self.model)

        cfg = self.model.config
        self.model_info.value = f"""
        <div style="background: #f0f0f0; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
            <div style="font-size: 14px; font-weight: bold; margin-bottom: 4px;">
                Loaded Model: {param_str} params ({dtype_str}) - {model_size}
            </div>
            <div style="font-size: 11px; color: #666;">
                n_layers={cfg.n_layers} | n_heads={cfg.n_heads} | n_kv_heads={cfg.n_kv_heads} |
                d_model={cfg.d_model} | d_ff={cfg.d_ff} | vocab={cfg.vocab_size}
            </div>
        </div>
        """

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
            self._log(f"Loading checkpoint: {selected}")
            self._update_status("Loading checkpoint...")

            # Get model config
            model_config = CONFIGS[self.model_size]

            # Load checkpoint to get vocab_size
            if selected == "best":
                checkpoint_path = self.checkpoint_manager.checkpoint_dir / "best.pt"
            else:
                checkpoint_path = Path(selected)

            if not checkpoint_path.exists():
                self._log(f"Checkpoint not found: {checkpoint_path}")
                return

            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            checkpoint["source_path"] = str(checkpoint_path)

            # Update model config from checkpoint
            if "model_config" in checkpoint:
                saved_config = checkpoint["model_config"]
                model_config.vocab_size = saved_config.get(
                    "vocab_size", model_config.vocab_size
                )
                model_config.max_seq_len = saved_config.get(
                    "max_seq_len", model_config.max_seq_len
                )

            # Create model and load weights
            self.model = LLaMA(model_config)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()

            self.original_checkpoint = checkpoint

            # Update UI
            self._update_model_info()
            self.convert_btn.disabled = False

            step = checkpoint.get("step", "?")
            loss = checkpoint.get("loss", "?")
            self._log(f"Loaded: step={step}, loss={loss:.4f}" if isinstance(loss, float) else f"Loaded: step={step}")
            self._update_status(f"Model loaded - Ready to convert")

        except Exception as e:
            self._log(f"Failed to load: {e}")
            self._update_status("Load failed")
            import traceback
            self._log(traceback.format_exc())

    def _on_convert(self, _):
        """Handle convert button click."""
        if self.model is None:
            self._log("No model loaded")
            return

        if not self.fp16_checkbox.value and not self.int8_checkbox.value:
            self._log("Select at least one quantization type")
            return

        self._convert()

    def _convert(self):
        """Perform quantization conversion."""
        self.convert_btn.disabled = True
        self.progress.layout.visibility = "visible"
        self.progress.value = 0

        try:
            # Create output directory
            self.output_dir.mkdir(parents=True, exist_ok=True)

            original_size = get_model_size_str(self.model)
            self._log(f"Original model size: {original_size}")

            total_steps = int(self.fp16_checkbox.value) + int(self.int8_checkbox.value)
            current_step = 0

            # FP16 Conversion
            if self.fp16_checkbox.value:
                self._log("\n--- FP16 Conversion ---")
                self._update_status("Converting to FP16...")

                model_fp16 = convert_to_fp16(self.model, inplace=False)
                fp16_size = get_model_size_str(model_fp16)
                self._log(f"FP16 model size: {fp16_size}")

                # Save
                fp16_path = self.output_dir / "model_fp16.pt"
                save_fp16_checkpoint(
                    model_fp16,
                    str(fp16_path),
                    self.original_checkpoint,
                )

                file_size = get_file_size_str(str(fp16_path))
                self._log(f"Saved: {fp16_path} ({file_size})")

                current_step += 1
                self.progress.value = (current_step / total_steps) * 100

                # Cleanup
                del model_fp16
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # INT8 Conversion
            if self.int8_checkbox.value:
                self._log("\n--- INT8 Conversion ---")
                self._update_status("Converting to INT8...")
                self._log("Note: INT8 quantization is most efficient on CPU")

                model_int8 = quantize_dynamic_int8(self.model, inplace=False)

                # Save
                int8_path = self.output_dir / "model_int8.pt"
                save_int8_checkpoint(
                    model_int8,
                    str(int8_path),
                    self.original_checkpoint,
                )

                file_size = get_file_size_str(str(int8_path))
                self._log(f"Saved: {int8_path} ({file_size})")

                current_step += 1
                self.progress.value = (current_step / total_steps) * 100

                # Cleanup
                del model_int8
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            self._log("\n=== Conversion Complete ===")
            self._log(f"Output directory: {self.output_dir}")
            self._update_status("Conversion complete")

        except Exception as e:
            self._log(f"Conversion failed: {e}")
            self._update_status("Conversion failed")
            import traceback
            self._log(traceback.format_exc())

        finally:
            self.convert_btn.disabled = False
            self.progress.layout.visibility = "hidden"

    def display(self):
        """Display the UI."""
        display(self.ui)


# %% [markdown]
# ## 4. Setup and Run

# %%
def setup():
    """Initialize checkpoint manager."""
    checkpoint_dir = f"{config.model.checkpoint_dir}/{config.model.model_size}"
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        max_checkpoints=config.model.max_checkpoints,
        save_best=config.model.save_best,
        device="cpu",  # Load to CPU for quantization
    )

    return checkpoint_manager


# %% [markdown]
# ## 5. Main Entry Point

# %%
if __name__ == "__main__":
    checkpoint_manager = setup()

    ui = QuantizationUI(
        checkpoint_manager=checkpoint_manager,
        model_size=config.model.model_size,
        output_dir="quantized",
    )

    ui.display()
