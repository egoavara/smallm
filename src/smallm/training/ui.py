"""Training UI for Jupyter notebooks."""

import torch
import ipywidgets as widgets
from IPython.display import display, clear_output
from typing import Optional, Callable, TYPE_CHECKING
import time
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from ..model import LLaMA
    from .checkpoint import CheckpointManager

from ..data import CHATML_USER


class TrainingUI:
    """Jupyter 기반 학습 UI.

    Features:
        - Training controls (start/stop)
        - Checkpoint loader
        - Text generation test
        - Runtime log display
    """

    def __init__(
        self,
        model: "LLaMA",
        optimizer: torch.optim.Optimizer,
        checkpoint_manager: "CheckpointManager",
        tokenizer,
        train_step_fn: Callable[[], float],
        device: str = "cpu",
        model_size: str = "unknown",
        amp_dtype: Optional[str] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ):
        """Initialize Training UI.

        Args:
            model: LLaMA model instance
            optimizer: Optimizer instance
            checkpoint_manager: CheckpointManager instance
            tokenizer: Tokenizer instance
            train_step_fn: Function that performs one training step and returns loss
            device: Device string
            model_size: Model size name (tiny, small, medium)
            amp_dtype: AMP dtype string (e.g., "bfloat16", "float16") or None if disabled
            scheduler: Optional LR scheduler instance
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_manager = checkpoint_manager
        self.tokenizer = tokenizer
        self.train_step_fn = train_step_fn
        self.device = device
        self.model_size = model_size
        self.amp_dtype = amp_dtype

        self.step = 0
        self.loss_history = []
        self.is_training = False
        self.stop_requested = False

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

        # dtype 확인 (AMP 사용 시 AMP dtype 표시)
        if self.amp_dtype:
            dtype_str = self.amp_dtype
        else:
            dtype = next(self.model.parameters()).dtype
            dtype_str = str(dtype).replace("torch.", "")

        cfg = self.model.config
        # Check optimization flags
        grad_ckpt = "✓" if getattr(self.model, 'gradient_checkpointing', False) else "✗"

        model_info_html = f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; padding: 12px 16px; border-radius: 8px; margin-bottom: 10px;">
            <div style="font-size: 18px; font-weight: bold; margin-bottom: 6px;">
                📦 {self.model_size.upper()} <span style="font-weight: normal; font-size: 14px;">(params = {param_str}, {dtype_str})</span>
            </div>
            <div style="font-size: 12px; opacity: 0.9;">
                n_layers={cfg.n_layers} | n_heads={cfg.n_heads} | n_kv_heads={cfg.n_kv_heads} |
                d_model={cfg.d_model} | d_ff={cfg.d_ff} | vocab={cfg.vocab_size} | seq_len={cfg.max_seq_len}
            </div>
            <div style="font-size: 11px; opacity: 0.8; margin-top: 4px;">
                grad_checkpointing={grad_ckpt}
            </div>
        </div>
        """
        self.model_info = widgets.HTML(value=model_info_html)

        # === Training Section ===
        self.train_steps_input = widgets.IntText(
            value=1000, description="Steps:", layout=widgets.Layout(width="150px")
        )
        self.log_interval_input = widgets.IntText(
            value=100, description="Log every:", layout=widgets.Layout(width="150px")
        )
        self.save_interval_input = widgets.IntText(
            value=100, description="Save every:", layout=widgets.Layout(width="150px")
        )

        self.train_btn = widgets.Button(
            description="▶️ Train",
            button_style="success",
            layout=widgets.Layout(width="100px"),
        )
        self.stop_btn = widgets.Button(
            description="⏹️ Stop",
            button_style="danger",
            layout=widgets.Layout(width="100px"),
        )

        self.train_btn.on_click(self._on_train)
        self.stop_btn.on_click(self._on_stop)

        train_controls = widgets.HBox(
            [
                self.train_steps_input,
                self.log_interval_input,
                self.save_interval_input,
            ]
        )
        train_buttons = widgets.HBox([self.train_btn, self.stop_btn])

        # === Checkpoint Section ===
        self.checkpoint_dropdown = widgets.Dropdown(
            options=[], description="Checkpoint:", layout=widgets.Layout(width="350px")
        )
        self.load_btn = widgets.Button(
            description="📂 Load",
            button_style="info",
            layout=widgets.Layout(width="100px"),
        )
        self.refresh_btn = widgets.Button(
            description="🔄", button_style="", layout=widgets.Layout(width="50px")
        )

        self.load_btn.on_click(self._on_load_checkpoint)
        self.refresh_btn.on_click(self._on_refresh_checkpoints)

        checkpoint_controls = widgets.HBox(
            [self.checkpoint_dropdown, self.load_btn, self.refresh_btn]
        )

        # === Generation Section ===
        self.prompt_input = widgets.Textarea(
            value="",
            placeholder="Enter prompt...",
            description="Prompt:",
            layout=widgets.Layout(width="400px", height="80px"),
        )
        self.max_tokens_input = widgets.IntText(
            value=100, description="Max tokens:", layout=widgets.Layout(width="150px")
        )
        self.temperature_input = widgets.FloatText(
            value=0.8, description="Temperature:", layout=widgets.Layout(width="150px")
        )
        self.generate_btn = widgets.Button(
            description="✨ Generate",
            button_style="primary",
            layout=widgets.Layout(width="120px"),
        )
        self.chatml_btn = widgets.Button(
            description="💬 ChatML",
            button_style="warning",
            layout=widgets.Layout(width="100px"),
            tooltip="Convert prompt to ChatML format for instruct mode",
        )

        self.generate_btn.on_click(self._on_generate)
        self.chatml_btn.on_click(self._on_chatml_convert)

        gen_controls = widgets.HBox(
            [
                self.prompt_input,
                self.max_tokens_input,
                self.temperature_input,
                self.generate_btn,
                self.chatml_btn,
            ]
        )

        # === Graph Section ===
        self.graph_btn = widgets.Button(
            description="📈 Show Graph",
            button_style="info",
            layout=widgets.Layout(width="120px"),
        )
        self.graph_window_input = widgets.IntText(
            value=100,
            description="Smooth:",
            layout=widgets.Layout(width="150px"),
        )
        self.graph_recent_input = widgets.Text(
            value="",
            description="Recent:",
            placeholder="all",
            layout=widgets.Layout(width="150px"),
        )
        self.graph_scale_dropdown = widgets.Dropdown(
            options=[("Linear", "linear"), ("Square", "square"), ("Log", "log")],
            value="linear",
            description="Scale:",
            layout=widgets.Layout(width="130px"),
        )
        self.graph_percentile_input = widgets.IntText(
            value=5,
            description="Percentile:",
            layout=widgets.Layout(width="130px"),
        )
        self.graph_output = widgets.Output(
            layout=widgets.Layout(
                height="350px",
                border="1px solid #ccc",
            )
        )
        self.graph_btn.on_click(self._on_show_graph)

        graph_controls = widgets.HBox([self.graph_btn, self.graph_window_input, self.graph_recent_input, self.graph_scale_dropdown, self.graph_percentile_input])

        # === Output Section ===
        self.log_lines = []  # 로그 라인 저장 (최신이 앞)
        self.log_output = widgets.HTML(
            value="",
            layout=widgets.Layout(
                height="300px", overflow_y="auto", border="1px solid #ccc", padding="5px"
            )
        )

        # === Status Bar ===
        self.status_label = widgets.Label(value="Ready")

        # === Layout ===
        train_section = widgets.VBox(
            [
                widgets.HTML("<b>🏋️ Training</b>"),
                train_controls,
                train_buttons,
            ]
        )

        checkpoint_section = widgets.VBox(
            [
                widgets.HTML("<b>💾 Checkpoint</b>"),
                checkpoint_controls,
            ]
        )

        gen_section = widgets.VBox(
            [
                widgets.HTML("<b>✨ Generate</b>"),
                gen_controls,
            ]
        )

        graph_section = widgets.VBox(
            [
                widgets.HTML("<b>📈 Loss Curve</b>"),
                graph_controls,
                self.graph_output,
            ]
        )

        log_section = widgets.VBox(
            [
                widgets.HTML("<b>📋 Log</b>"),
                self.log_output,
            ]
        )

        self.ui = widgets.VBox(
            [
                self.model_info,
                train_section,
                widgets.HTML("<hr>"),
                checkpoint_section,
                widgets.HTML("<hr>"),
                gen_section,
                widgets.HTML("<hr>"),
                graph_section,
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
        # 최대 500줄 유지
        if len(self.log_lines) > 500:
            self.log_lines = self.log_lines[:500]
        self._render_log()

    def _render_log(self):
        """Render log lines to HTML."""
        html_content = "<pre style='margin:0; font-family:monospace; font-size:12px;'>"
        html_content += "\n".join(self.log_lines)
        html_content += "</pre>"
        self.log_output.value = html_content

    def _clear_log(self):
        """Clear log output."""
        self.log_lines = []
        self.log_output.value = ""

    def _update_status(self, message: str):
        """Update status bar."""
        self.status_label.value = message

    def _refresh_checkpoints(self):
        """Refresh checkpoint dropdown."""
        checkpoints = self.checkpoint_manager.list_checkpoints()
        options = [("best.pt", "best")]

        for path, step, loss in checkpoints:
            options.append((f"{path.name}", str(path)))

        self.checkpoint_dropdown.options = options
        if options:
            self.checkpoint_dropdown.value = options[0][1]

    def _on_refresh_checkpoints(self, b):
        """Handle refresh button click."""
        self._refresh_checkpoints()
        self._log("Checkpoint list refreshed")

    def _on_load_checkpoint(self, b):
        """Handle load checkpoint button click."""
        selected = self.checkpoint_dropdown.value
        if not selected:
            self._log("No checkpoint selected")
            return

        try:
            if selected == "best":
                result = self.checkpoint_manager.load_best(
                    self.model, self.optimizer, self.scheduler
                )
                if result[0] is not None:
                    self.step = result[0]
                    self.loss_history = result[1]
                    # Scheduler를 현재 step으로 동기화 (기존 checkpoint에 scheduler state가 없는 경우)
                    if self.scheduler is not None:
                        self.scheduler.last_epoch = self.step
                    self._log(f"✅ Loaded best.pt (step: {self.step})")
                else:
                    self._log("❌ best.pt not found")
            else:
                self.step, self.loss_history = self.checkpoint_manager.load_checkpoint(
                    selected, self.model, self.optimizer, self.scheduler
                )
                # Scheduler를 현재 step으로 동기화
                if self.scheduler is not None:
                    self.scheduler.last_epoch = self.step
                self._log(f"✅ Loaded checkpoint (step: {self.step})")

            self._update_status(f"Step: {self.step}")
        except Exception as e:
            self._log(f"❌ Failed to load: {e}")

    def _on_train(self, b):
        """Handle train button click."""
        if self.is_training:
            self._log("Already training")
            return

        self.is_training = True
        self.stop_requested = False
        self._clear_log()

        num_steps = self.train_steps_input.value
        log_interval = self.log_interval_input.value
        save_interval = self.save_interval_input.value

        self._log(f"🚀 Training for {num_steps} steps (from step {self.step})")
        self._log(f"   Log interval: {log_interval}, Save interval: {save_interval}")
        self._log("-" * 50)

        start_time = time.time()
        start_step = self.step

        try:
            for _ in range(num_steps):
                if self.stop_requested:
                    self._log("\n⏹️ Training stopped by user")
                    break

                loss = self.train_step_fn()
                self.step += 1
                self.loss_history.append(loss)

                if self.step % log_interval == 0:
                    elapsed = time.time() - start_time
                    steps_done = self.step - start_step
                    speed = steps_done / elapsed if elapsed > 0 else 0
                    self._log(
                        f"Step {self.step:6d} | Loss: {loss:.4f} | Speed: {speed:.1f} steps/s"
                    )
                    self._update_status(f"Step: {self.step} | Loss: {loss:.4f}")

                if self.step % save_interval == 0:
                    recent_losses = self.loss_history[-100:]
                    avg_loss = sum(recent_losses) / len(recent_losses)
                    saved = self.checkpoint_manager.save(
                        step=self.step,
                        loss=avg_loss,
                        model=self.model,
                        optimizer=self.optimizer,
                        loss_history=self.loss_history,
                        scheduler=self.scheduler,
                    )
                    self._log(f"   💾 Saved: {saved}")
                    self._refresh_checkpoints()

        except Exception as e:
            self._log(f"❌ Training error: {e}")

        finally:
            self.is_training = False
            elapsed = time.time() - start_time
            steps_done = self.step - start_step
            self._log("-" * 50)
            self._log(f"✅ Completed {steps_done} steps in {elapsed:.1f}s")
            if self.loss_history:
                self._log(f"   Final loss: {self.loss_history[-1]:.4f}")
            self._update_status(f"Step: {self.step} | Done")

    def _on_stop(self, b):
        """Handle stop button click."""
        self.stop_requested = True
        self._update_status("Stopping...")

    def _on_chatml_convert(self, b):
        """Convert current prompt to ChatML format."""
        current_prompt = self.prompt_input.value.strip()
        if not current_prompt:
            self._log("❌ Enter a prompt first")
            return

        # Convert to ChatML format: <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
        chatml_prompt = CHATML_USER.format(content=current_prompt) + "<|im_start|>assistant\n"
        self.prompt_input.value = chatml_prompt
        self._log(f"✅ Converted to ChatML format")

    @torch.no_grad()
    def _on_generate(self, b):
        """Handle generate button click."""
        prompt = self.prompt_input.value
        max_tokens = self.max_tokens_input.value
        temperature = self.temperature_input.value

        self._log(f"\n--- Generate (temp={temperature}) ---")
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
                top_k = 50
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
            self._log(f"❌ Generation error: {e}")

    def display(self):
        """Display the UI."""
        display(self.ui)

    def set_step(self, step: int, loss_history: list = None):
        """Set current step and loss history."""
        self.step = step
        if loss_history is not None:
            self.loss_history = loss_history
        self._update_status(f"Step: {self.step}")

    def _on_show_graph(self, b):
        """Handle show graph button click."""
        self.graph_output.clear_output(wait=True)

        with self.graph_output:
            if not self.loss_history:
                print("No loss history available. Train the model first.")
                return

            window = max(1, self.graph_window_input.value)

            # Parse recent value (empty or non-numeric = all, numeric = last n)
            recent_str = self.graph_recent_input.value.strip()
            if recent_str and recent_str.isdigit() and int(recent_str) > 0:
                recent_n = int(recent_str)
                losses_raw = np.array(self.loss_history[-recent_n:])
                start_step = max(1, len(self.loss_history) - recent_n + 1)
                steps = np.arange(start_step, start_step + len(losses_raw))
            else:
                losses_raw = np.array(self.loss_history)
                steps = np.arange(1, len(losses_raw) + 1)

            # Apply scale transformation to loss values
            scale_type = self.graph_scale_dropdown.value
            if scale_type == "square":
                losses = np.square(losses_raw)
                y_label = "Loss²"
            elif scale_type == "log":
                losses = np.log(np.maximum(losses_raw, 1e-8))
                y_label = "log(Loss)"
            else:
                losses = losses_raw
                y_label = "Loss"

            fig, ax = plt.subplots(figsize=(10, 4))

            # Raw loss (faint)
            ax.plot(steps, losses, alpha=0.3, color="blue", linewidth=0.5, label="Raw")

            # Smoothed loss (moving average)
            if len(losses) >= window:
                smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
                smooth_steps = steps[window - 1:]
                ax.plot(
                    smooth_steps,
                    smoothed,
                    color="blue",
                    linewidth=2,
                    label=f"Smoothed (window={window})",
                )

            ax.set_xlabel("Step")
            ax.set_ylabel(y_label)
            total_steps = len(self.loss_history)
            if recent_str and recent_str.isdigit() and int(recent_str) > 0:
                ax.set_title(f"Training Loss Curve (Recent {len(losses)} of {total_steps} steps)")
            else:
                ax.set_title(f"Training Loss Curve (Total {total_steps} steps)")
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)

            # Set Y-axis range based on percentiles
            pct = max(0, min(49, self.graph_percentile_input.value))
            y_min = np.percentile(losses, pct)
            y_max = np.percentile(losses, 100 - pct)
            y_margin = (y_max - y_min) * 0.1
            ax.set_ylim(y_min - y_margin, y_max + y_margin)

            # Show stats (always show original loss values)
            recent_raw = losses_raw[-min(100, len(losses_raw)):]
            stats_text = (
                f"Final: {losses_raw[-1]:.4f} | "
                f"Min: {losses_raw.min():.4f} | "
                f"Recent avg: {recent_raw.mean():.4f}"
            )
            ax.text(
                0.02, 0.98, stats_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

            plt.tight_layout()
            plt.show()
            plt.close(fig)
