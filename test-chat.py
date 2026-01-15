# %% [markdown]
# # SmallM - Chat Testing

# %% [markdown]
# ## 1. Setup

# %%
import torch
import threading
import time
import html
from pathlib import Path
from typing import Optional
import ipywidgets as widgets
from IPython.display import display, clear_output

from config import config, MODELS, MODES
from smallm.model import LLaMA, CONFIGS
from smallm.training import CheckpointManager
from smallm.data import format_chatml

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
class ChatUI:
    """채팅 형태로 모델을 테스트하는 UI."""

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

        # 대화 히스토리
        self.conversation_history: list[dict] = []
        self.system_prompt = ""

        self._build_ui()

    def _build_ui(self):
        param_count = self.model.count_parameters()
        param_str = (
            f"{param_count/1e9:.1f}B"
            if param_count >= 1e9
            else f"{param_count/1e6:.1f}M"
        )
        dtype_str = str(next(self.model.parameters()).dtype).replace("torch.", "")
        cfg = self.model.config

        self.model_info = widgets.HTML(
            f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; padding: 12px; border-radius: 8px; margin-bottom: 10px;">
            <b>💬 Chat Mode - {self.model_size.upper()}</b> ({param_str}, {dtype_str})<br>
            <small>layers={cfg.n_layers} heads={cfg.n_heads} d_model={cfg.d_model}</small>
        </div>
        """
        )

        # Checkpoint 관련
        self.checkpoint_dropdown = widgets.Dropdown(
            options=[], description="Checkpoint:", layout=widgets.Layout(width="350px")
        )
        self.load_btn = widgets.Button(
            description="Load",
            button_style="info",
            layout=widgets.Layout(width="100px"),
        )
        self.refresh_btn = widgets.Button(
            description="Refresh", layout=widgets.Layout(width="80px")
        )
        self.load_btn.on_click(self._on_load_checkpoint)
        self.refresh_btn.on_click(self._on_refresh_checkpoints)

        self.auto_reload_checkbox = widgets.Checkbox(
            value=False, description="Auto-reload best.pt"
        )
        self.auto_reload_status = widgets.HTML("<span style='color:gray'>OFF</span>")
        self.auto_reload_checkbox.observe(self._on_auto_reload_toggle, names="value")

        # 시스템 프롬프트
        self.system_input = widgets.Textarea(
            value="You are a helpful assistant.",
            placeholder="Enter system prompt...",
            description="System:",
            layout=widgets.Layout(width="600px", height="60px"),
        )

        # 채팅 화면
        self.chat_display = widgets.HTML(
            value=self._render_chat(),
            layout=widgets.Layout(
                height="400px",
                overflow_y="auto",
                border="1px solid #ccc",
                padding="10px",
                background_color="white",
            ),
        )

        # 입력
        self.user_input = widgets.Textarea(
            value="",
            placeholder="Type your message...",
            layout=widgets.Layout(width="500px", height="60px"),
        )
        self.send_btn = widgets.Button(
            description="Send",
            button_style="primary",
            layout=widgets.Layout(width="80px"),
        )
        self.send_btn.on_click(self._on_send)

        # 생성 파라미터
        self.max_tokens_input = widgets.IntText(
            value=100, description="Max tokens:", layout=widgets.Layout(width="150px")
        )
        self.temperature_input = widgets.FloatText(
            value=0.8, description="Temp:", layout=widgets.Layout(width="150px")
        )
        self.top_k_input = widgets.IntText(
            value=50, description="Top-K:", layout=widgets.Layout(width="150px")
        )

        # 대화 제어
        self.clear_btn = widgets.Button(
            description="Clear Chat",
            button_style="warning",
            layout=widgets.Layout(width="100px"),
        )
        self.clear_btn.on_click(self._on_clear_chat)

        self.status_label = widgets.Label(value="Ready")

        self.ui = widgets.VBox(
            [
                self.model_info,
                widgets.HTML("<b>Checkpoint</b>"),
                widgets.HBox(
                    [self.checkpoint_dropdown, self.load_btn, self.refresh_btn]
                ),
                widgets.HBox([self.auto_reload_checkbox, self.auto_reload_status]),
                widgets.HTML("<hr><b>System Prompt</b>"),
                self.system_input,
                widgets.HTML("<hr><b>Chat</b>"),
                self.chat_display,
                widgets.HBox([self.user_input, self.send_btn, self.clear_btn]),
                widgets.HBox(
                    [self.max_tokens_input, self.temperature_input, self.top_k_input]
                ),
                self.status_label,
            ]
        )
        self._refresh_checkpoints()

    def _render_chat(self) -> str:
        """대화 히스토리를 HTML로 렌더링."""
        if not self.conversation_history:
            return "<div style='color: #888; text-align: center; padding: 20px;'>Start a conversation...</div>"

        chat_html = ""
        for msg in self.conversation_history:
            role = msg["role"]
            content = html.escape(msg["content"])
            content = content.replace("\n", "<br>")

            if role == "user":
                chat_html += f"""
                <div style="display: flex; justify-content: flex-end; margin: 10px 0;">
                    <div style="background: #007bff; color: white; padding: 10px 15px;
                                border-radius: 15px 15px 0 15px; max-width: 70%;">
                        {content}
                    </div>
                </div>
                """
            else:  # assistant
                chat_html += f"""
                <div style="display: flex; justify-content: flex-start; margin: 10px 0;">
                    <div style="background: #f0f0f0; color: black; padding: 10px 15px;
                                border-radius: 15px 15px 15px 0; max-width: 70%;">
                        {content}
                    </div>
                </div>
                """

        return chat_html

    def _update_chat_display(self):
        """채팅 화면 업데이트."""
        self.chat_display.value = self._render_chat()

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
        self._update_status("Checkpoints refreshed")

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
                    self._update_status(f"Loaded best.pt (step {self.step})")
            else:
                self.step, self.loss_history = self.checkpoint_manager.load_checkpoint(
                    selected, self.model, None
                )
                self._update_status(f"Loaded (step {self.step})")
        except Exception as e:
            self._update_status(f"Error: {e}")

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
        self._reload_thread = threading.Thread(
            target=self._auto_reload_loop, daemon=True
        )
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
                        self._update_status("best.pt changed - reloading...")
                        self._load_checkpoint("best")
                    elif not self._last_mtime:
                        self._last_mtime = mtime
            except:
                pass
            time.sleep(2)

    def _on_clear_chat(self, _):
        """대화 히스토리 초기화."""
        self.conversation_history = []
        self._update_chat_display()
        self._update_status("Chat cleared")

    def _build_chat_prompt(self) -> str:
        """현재 대화 히스토리를 ChatML 형식으로 변환."""
        system = self.system_input.value.strip()
        return format_chatml(self.conversation_history, system=system)

    @torch.no_grad()
    def _on_send(self, _):
        """사용자 메시지 전송 및 응답 생성."""
        user_message = self.user_input.value.strip()
        if not user_message:
            return

        # 사용자 메시지 추가
        self.conversation_history.append({"role": "user", "content": user_message})
        self.user_input.value = ""
        self._update_chat_display()
        self._update_status("Generating...")

        try:
            self.model.eval()

            # ChatML 형식으로 프롬프트 구성 + assistant 시작 태그 추가
            prompt = self._build_chat_prompt() + "<|im_start|>assistant\n"

            tokens = self.tokenizer.encode(prompt)
            tokens = torch.tensor([tokens], device=self.device)
            seq_len = self.model.config.max_seq_len

            max_tokens = self.max_tokens_input.value
            temp = self.temperature_input.value
            top_k = self.top_k_input.value

            generated_tokens = []
            end_token = "<|im_end|>"

            for _ in range(max_tokens):
                logits, _ = self.model(tokens[:, -seq_len:])
                logits = logits[:, -1, :] / temp

                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, -1:]] = float("-inf")

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                tokens = torch.cat([tokens, next_token], dim=1)
                generated_tokens.append(next_token.item())

                # EOS 체크
                if (
                    hasattr(self.tokenizer, "eos_id")
                    and next_token.item() == self.tokenizer.eos_id
                ):
                    break

                # <|im_end|> 체크 - 디코딩해서 확인
                decoded = self.tokenizer.decode(generated_tokens)
                if end_token in decoded:
                    # <|im_end|> 이전까지만 추출
                    decoded = decoded.split(end_token)[0]
                    break

            # 응답 추출
            response = self.tokenizer.decode(generated_tokens)
            if end_token in response:
                response = response.split(end_token)[0]
            response = response.strip()

            # assistant 응답 추가
            self.conversation_history.append({"role": "assistant", "content": response})
            self._update_chat_display()
            self._update_status(f"Generated {len(generated_tokens)} tokens")

        except Exception as e:
            self._update_status(f"Error: {e}")
            # 에러 발생 시 마지막 사용자 메시지 제거
            if (
                self.conversation_history
                and self.conversation_history[-1]["role"] == "user"
            ):
                self.conversation_history.pop()
                self._update_chat_display()

    def display(self):
        display(self.ui)


# %%
class ChatState:
    """채팅 테스트 상태 관리."""

    def __init__(self):
        self.model: Optional[LLaMA] = None
        self.checkpoint_manager: Optional[CheckpointManager] = None
        self.ui: Optional[ChatUI] = None
        self.step = 0
        self.loss_history = []

    def load_best_checkpoint(self) -> bool:
        result = self.checkpoint_manager.load_best(self.model, None)
        if result[0] is not None:
            self.step, self.loss_history = result[0], result[1]
            return True
        return False


state = ChatState()


# %%
def setup_model():
    config.load()  # 최신 설정 로드

    model_config = CONFIGS[config.model_size]
    model_config.vocab_size = tokenizer.vocab_size
    model_config.max_seq_len = config.seq_len

    model = LLaMA(model_config).to(config.device)
    print(
        f"\n📦 Model: {config.mode}/{config.model_size} ({model.count_parameters():,} params)"
    )

    checkpoint_dir = f"{config.checkpoint_dir}/{config.mode}/{config.model_size}"
    state.checkpoint_manager = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        max_checkpoints=config.max_checkpoints,
        save_best=config.save_best,
        device=config.device,
    )

    state.model = model
    state.step = 0
    state.loss_history = []

    if config.auto_load_best and state.load_best_checkpoint():
        print(f"   Loaded best.pt (step {state.step})")

    return model


# %% [markdown]
# ## 2. Chat UI


# %%
def create_chat_ui():
    """통합 채팅 UI: 설정 선택 → Setup → 채팅"""

    # === Config 선택 UI ===
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
    chat_container = widgets.VBox()

    def update_config_info():
        config.load()
        mode_dropdown.value = config.mode
        model_dropdown.value = config.model_size
        config_info.value = f"""
        <div style="background: #f0f0f0; padding: 10px; border-radius: 5px; margin-top: 10px;">
            <b>Current Config</b><br>
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

    setup_btn = widgets.Button(
        description="Setup Model",
        button_style="primary",
        layout=widgets.Layout(width="150px"),
    )

    def on_setup_click(_):
        setup_btn.disabled = True
        setup_btn.description = "Loading..."

        with setup_output:
            clear_output()
            try:
                setup_model()

                # ChatUI 생성
                state.ui = ChatUI(
                    model=state.model,
                    checkpoint_manager=state.checkpoint_manager,
                    tokenizer=tokenizer,
                    device=config.device,
                    model_size=config.model_size,
                )

                if state.step > 0:
                    state.ui.step = state.step
                    state.ui.loss_history = state.loss_history
                    state.ui._update_status(f"Step: {state.step}")

                chat_container.children = [state.ui.ui]
                print("\n✅ Ready to chat!")

            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback

                traceback.print_exc()
            finally:
                setup_btn.disabled = False
                setup_btn.description = "Setup Model"

    setup_btn.on_click(on_setup_click)

    update_config_info()

    # 전체 UI 구성
    ui = widgets.VBox(
        [
            widgets.HTML("<h3>1. Select Configuration</h3>"),
            widgets.HBox([mode_dropdown, model_dropdown]),
            config_info,
            widgets.HTML("<br>"),
            setup_btn,
            setup_output,
            widgets.HTML("<h3>2. Chat</h3>"),
            chat_container,
        ]
    )

    display(ui)


# %%
if __name__ == "__main__":
    create_chat_ui()
