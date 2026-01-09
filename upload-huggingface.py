# %% [markdown]
# # SmallM - Hugging Face Upload/Download
#
# 학습된 모델과 토크나이저를 Hugging Face Hub에 업로드/다운로드하는 UI

# %% [markdown]
# ## 1. Setup & Imports

# %%
from pathlib import Path
import ipywidgets as widgets
from IPython.display import display
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError

from config import config

MODEL_SIZES = ["tiny", "small", "medium"]

print("Hugging Face Upload/Download UI")
print(f"Tokenizer path: {config.tokenizer_path}")
print(f"Checkpoint dir: {config.model.checkpoint_dir}")

# %% [markdown]
# ## 2. Hugging Face UI


# %%
class HuggingFaceUI:
    """Hugging Face 업로드/다운로드 UI."""

    def __init__(self):
        self.api = HfApi()
        self._build_ui()

    def _build_ui(self):
        """UI 구성."""
        # === Header ===
        header_html = """
        <div style="background: linear-gradient(135deg, #ff9a56 0%, #ff6b6b 100%);
                    color: white; padding: 12px 16px; border-radius: 8px; margin-bottom: 10px;">
            <div style="font-size: 18px; font-weight: bold;">
                🤗 Hugging Face Hub
            </div>
            <div style="font-size: 12px; opacity: 0.9;">
                Upload/Download all checkpoints and tokenizer
            </div>
        </div>
        """
        self.header = widgets.HTML(value=header_html)

        # === Repository Settings ===
        self.repo_input = widgets.Text(
            value="egoavara/smallm",
            placeholder="username/repo-name",
            description="Repository:",
            layout=widgets.Layout(width="350px"),
        )

        self.private_checkbox = widgets.Checkbox(
            value=True,
            description="Private repo",
            layout=widgets.Layout(width="150px"),
        )

        # === Buttons ===
        self.upload_btn = widgets.Button(
            description="⬆️ Upload All",
            button_style="success",
            layout=widgets.Layout(width="130px"),
        )
        self.download_btn = widgets.Button(
            description="⬇️ Download All",
            button_style="info",
            layout=widgets.Layout(width="130px"),
        )
        self.check_btn = widgets.Button(
            description="🔍 Check Files",
            button_style="",
            layout=widgets.Layout(width="120px"),
        )

        self.upload_btn.on_click(self._on_upload)
        self.download_btn.on_click(self._on_download)
        self.check_btn.on_click(self._on_check_files)

        # === Log Output ===
        self.log_lines = []
        self.log_output = widgets.HTML(
            value="",
            layout=widgets.Layout(
                height="300px",
                overflow_y="auto",
                border="1px solid #ccc",
                padding="5px",
            ),
        )

        # === Status ===
        self.status_label = widgets.Label(value="Ready")

        # === Layout ===
        repo_section = widgets.VBox(
            [
                widgets.HTML("<b>📁 Repository Settings</b>"),
                widgets.HBox([self.repo_input, self.private_checkbox]),
            ]
        )

        action_section = widgets.VBox(
            [
                widgets.HTML("<b>🚀 Actions</b>"),
                widgets.HBox([self.upload_btn, self.download_btn, self.check_btn]),
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
                self.header,
                repo_section,
                widgets.HTML("<hr>"),
                action_section,
                widgets.HTML("<hr>"),
                log_section,
                self.status_label,
            ]
        )

    def _log(self, message: str):
        """로그 메시지 추가."""
        import html

        escaped = html.escape(message)
        self.log_lines.insert(0, escaped)
        if len(self.log_lines) > 200:
            self.log_lines = self.log_lines[:200]
        self._render_log()

    def _render_log(self):
        """로그 HTML 렌더링."""
        html_content = "<pre style='margin:0; font-family:monospace; font-size:12px;'>"
        html_content += "\n".join(self.log_lines)
        html_content += "</pre>"
        self.log_output.value = html_content

    def _update_status(self, message: str):
        """상태 업데이트."""
        self.status_label.value = message

    def _get_tokenizer_paths(self) -> dict:
        """토크나이저 파일 경로 반환."""
        tokenizer_base = config.tokenizer_path
        return {
            "tokenizer_model": Path(f"{tokenizer_base}.model"),
            "tokenizer_vocab": Path(f"{tokenizer_base}.vocab"),
        }

    def _get_model_path(self, model_size: str) -> Path:
        """특정 모델 사이즈의 best.pt 경로 반환."""
        return Path(config.model.checkpoint_dir) / model_size / "best.pt"

    def _on_check_files(self, b):
        """로컬 파일 상태 확인."""
        self._log("=" * 40)
        self._log("🔍 Checking local files...")

        # 모든 모델 확인
        self._log("\n📦 Models:")
        for model_size in MODEL_SIZES:
            best_pt = self._get_model_path(model_size)
            if best_pt.exists():
                size_mb = best_pt.stat().st_size / (1024 * 1024)
                self._log(f"   ✅ {model_size}/best.pt: {size_mb:.1f} MB")
            else:
                self._log(f"   ❌ {model_size}/best.pt: Not found")

        # 토크나이저 확인
        self._log("\n📝 Tokenizer:")
        paths = self._get_tokenizer_paths()

        if paths["tokenizer_model"].exists():
            size_kb = paths["tokenizer_model"].stat().st_size / 1024
            self._log(f"   ✅ tokenizer.model: {size_kb:.1f} KB")
        else:
            self._log(f"   ❌ tokenizer.model: Not found")

        if paths["tokenizer_vocab"].exists():
            size_kb = paths["tokenizer_vocab"].stat().st_size / 1024
            self._log(f"   ✅ tokenizer.vocab: {size_kb:.1f} KB")
        else:
            self._log(f"   ❌ tokenizer.vocab: Not found")

        self._update_status("File check complete")

    def _on_upload(self, b):
        """Hugging Face에 모든 파일 업로드."""
        repo_id = self.repo_input.value.strip()
        if not repo_id:
            self._log("❌ Please enter a repository name (username/repo-name)")
            return

        self._log("=" * 40)
        self._log(f"⬆️ Uploading all files to {repo_id}...")
        self._update_status("Uploading...")

        try:
            # 레포지토리 생성 (없으면)
            try:
                self.api.create_repo(
                    repo_id=repo_id,
                    private=self.private_checkbox.value,
                    exist_ok=True,
                )
                self._log(f"✅ Repository ready: {repo_id}")
            except Exception as e:
                self._log(f"⚠️ Repo check: {e}")

            uploaded_files = []

            # 모든 모델 업로드
            self._log("\n📦 Uploading models...")
            for model_size in MODEL_SIZES:
                best_pt = self._get_model_path(model_size)
                if best_pt.exists():
                    self._log(f"   Uploading {model_size}/best.pt...")
                    self.api.upload_file(
                        path_or_fileobj=str(best_pt),
                        path_in_repo=f"{model_size}/best.pt",
                        repo_id=repo_id,
                    )
                    uploaded_files.append(f"{model_size}/best.pt")
                    self._log(f"   ✅ Uploaded: {model_size}/best.pt")
                else:
                    self._log(f"   ⏭️ Skipped: {model_size}/best.pt (not found)")

            # 토크나이저 업로드
            self._log("\n📝 Uploading tokenizer...")
            paths = self._get_tokenizer_paths()

            if paths["tokenizer_model"].exists():
                self._log(f"   Uploading tokenizer.model...")
                self.api.upload_file(
                    path_or_fileobj=str(paths["tokenizer_model"]),
                    path_in_repo="tokenizer/tokenizer.model",
                    repo_id=repo_id,
                )
                uploaded_files.append("tokenizer/tokenizer.model")
                self._log(f"   ✅ Uploaded: tokenizer/tokenizer.model")

            if paths["tokenizer_vocab"].exists():
                self._log(f"   Uploading tokenizer.vocab...")
                self.api.upload_file(
                    path_or_fileobj=str(paths["tokenizer_vocab"]),
                    path_in_repo="tokenizer/tokenizer.vocab",
                    repo_id=repo_id,
                )
                uploaded_files.append("tokenizer/tokenizer.vocab")
                self._log(f"   ✅ Uploaded: tokenizer/tokenizer.vocab")

            self._log(f"\n🎉 Upload complete! ({len(uploaded_files)} files)")
            self._log(f"   https://huggingface.co/{repo_id}")
            self._update_status(f"Uploaded {len(uploaded_files)} files")

        except Exception as e:
            self._log(f"❌ Upload failed: {e}")
            self._update_status("Upload failed")

    def _on_download(self, b):
        """Hugging Face에서 모든 파일 다운로드."""
        repo_id = self.repo_input.value.strip()
        if not repo_id:
            self._log("❌ Please enter a repository name (username/repo-name)")
            return

        self._log("=" * 40)
        self._log(f"⬇️ Downloading all files from {repo_id}...")
        self._update_status("Downloading...")

        try:
            downloaded_files = []

            # 모든 모델 다운로드
            self._log("\n📦 Downloading models...")
            for model_size in MODEL_SIZES:
                try:
                    self._log(f"   Downloading {model_size}/best.pt...")
                    checkpoint_dir = Path(config.model.checkpoint_dir) / model_size
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)

                    downloaded_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=f"{model_size}/best.pt",
                        local_dir=config.model.checkpoint_dir,
                    )
                    downloaded_files.append(f"{model_size}/best.pt")
                    self._log(f"   ✅ Downloaded: {model_size}/best.pt")
                except Exception as e:
                    self._log(f"   ⏭️ Skipped: {model_size}/best.pt (not found)")

            # 토크나이저 다운로드
            self._log("\n📝 Downloading tokenizer...")
            tokenizer_dir = Path(config.tokenizer.output_dir)
            tokenizer_dir.mkdir(parents=True, exist_ok=True)
            paths = self._get_tokenizer_paths()

            try:
                self._log(f"   Downloading tokenizer.model...")
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename="tokenizer/tokenizer.model",
                    local_dir=".",
                )
                import shutil

                if Path(downloaded_path).exists():
                    shutil.copy(downloaded_path, paths["tokenizer_model"])
                    downloaded_files.append("tokenizer.model")
                    self._log(f"   ✅ Downloaded: tokenizer.model")
            except Exception as e:
                self._log(f"   ⏭️ Skipped: tokenizer.model (not found)")

            try:
                self._log(f"   Downloading tokenizer.vocab...")
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename="tokenizer/tokenizer.vocab",
                    local_dir=".",
                )
                import shutil

                if Path(downloaded_path).exists():
                    shutil.copy(downloaded_path, paths["tokenizer_vocab"])
                    downloaded_files.append("tokenizer.vocab")
                    self._log(f"   ✅ Downloaded: tokenizer.vocab")
            except Exception as e:
                self._log(f"   ⏭️ Skipped: tokenizer.vocab (not found)")

            self._log(f"\n🎉 Download complete! ({len(downloaded_files)} files)")
            self._update_status(f"Downloaded {len(downloaded_files)} files")

        except RepositoryNotFoundError:
            self._log(f"❌ Repository not found: {repo_id}")
            self._update_status("Repository not found")
        except Exception as e:
            self._log(f"❌ Download failed: {e}")
            self._update_status("Download failed")

    def display(self):
        """UI 표시."""
        display(self.ui)


# %% [markdown]
# ## 3. Display UI

# %%
ui = HuggingFaceUI()
ui.display()
