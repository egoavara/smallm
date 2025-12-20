"""Training UI for Jupyter notebooks."""

import torch
import ipywidgets as widgets
from IPython.display import display, clear_output
from typing import Optional, Callable, TYPE_CHECKING
import time

if TYPE_CHECKING:
    from ..model import LLaMA
    from .checkpoint import CheckpointManager


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
    ):
        """Initialize Training UI.

        Args:
            model: LLaMA model instance
            optimizer: Optimizer instance
            checkpoint_manager: CheckpointManager instance
            tokenizer: Tokenizer instance
            train_step_fn: Function that performs one training step and returns loss
            device: Device string
        """
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_manager = checkpoint_manager
        self.tokenizer = tokenizer
        self.train_step_fn = train_step_fn
        self.device = device

        self.step = 0
        self.loss_history = []
        self.is_training = False
        self.stop_requested = False

        self._build_ui()

    def _build_ui(self):
        """Build the UI components."""
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
        self.prompt_input = widgets.Text(
            value="",
            placeholder="Enter prompt...",
            description="Prompt:",
            layout=widgets.Layout(width="400px"),
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

        self.generate_btn.on_click(self._on_generate)

        gen_controls = widgets.HBox(
            [
                self.prompt_input,
                self.max_tokens_input,
                self.temperature_input,
                self.generate_btn,
            ]
        )

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

        log_section = widgets.VBox(
            [
                widgets.HTML("<b>📋 Log</b>"),
                self.log_output,
            ]
        )

        self.ui = widgets.VBox(
            [
                train_section,
                widgets.HTML("<hr>"),
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
                result = self.checkpoint_manager.load_best(self.model, self.optimizer)
                if result[0] is not None:
                    self.step = result[0]
                    self.loss_history = result[1]
                    self._log(f"✅ Loaded best.pt (step: {self.step})")
                else:
                    self._log("❌ best.pt not found")
            else:
                self.step, self.loss_history = self.checkpoint_manager.load_checkpoint(
                    selected, self.model, self.optimizer
                )
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
