# %% [markdown]
# # SmallM - Hugging Face Upload/Download

# %%
from pathlib import Path
import ipywidgets as widgets
from IPython.display import display
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError

from config import config, MODELS, MODES

MODEL_SIZES = list(MODELS.keys())
MODE_NAMES = list(MODES.keys())

print("Hugging Face Upload/Download UI")
print(f"Tokenizer: {config.tokenizer_path}")
print(f"Checkpoints: {config.checkpoint_dir}")
print(f"Modes: {MODE_NAMES}")
print(f"Model sizes: {MODEL_SIZES}")


# %%
class HuggingFaceUI:
    def __init__(self):
        self.api = HfApi()
        self._build_ui()

    def _build_ui(self):
        self.header = widgets.HTML("""
        <div style="background: linear-gradient(135deg, #ff9a56 0%, #ff6b6b 100%);
                    color: white; padding: 12px; border-radius: 8px; margin-bottom: 10px;">
            <b>🤗 Hugging Face Hub</b><br>
            <small>Upload/Download checkpoints and tokenizer</small>
        </div>
        """)

        self.repo_input = widgets.Text(value="egoavara/smallm", description="Repository:", layout=widgets.Layout(width="350px"))
        self.private_checkbox = widgets.Checkbox(value=True, description="Private")

        self.upload_btn = widgets.Button(description="⬆️ Upload", button_style="success", layout=widgets.Layout(width="120px"))
        self.download_btn = widgets.Button(description="⬇️ Download", button_style="info", layout=widgets.Layout(width="120px"))
        self.check_btn = widgets.Button(description="🔍 Check", layout=widgets.Layout(width="100px"))

        self.upload_btn.on_click(self._on_upload)
        self.download_btn.on_click(self._on_download)
        self.check_btn.on_click(self._on_check_files)

        self.log_lines = []
        self.log_output = widgets.HTML(layout=widgets.Layout(height="300px", overflow_y="auto", border="1px solid #ccc", padding="5px"))
        self.status_label = widgets.Label(value="Ready")

        self.ui = widgets.VBox([
            self.header,
            widgets.HBox([self.repo_input, self.private_checkbox]),
            widgets.HTML("<hr>"),
            widgets.HBox([self.upload_btn, self.download_btn, self.check_btn]),
            widgets.HTML("<hr>"),
            self.log_output,
            self.status_label,
        ])

    def _log(self, msg):
        import html
        self.log_lines.insert(0, html.escape(msg))
        self.log_lines = self.log_lines[:200]
        self.log_output.value = f"<pre style='margin:0;font-size:12px'>{'<br>'.join(self.log_lines)}</pre>"

    def _get_tokenizer_paths(self):
        return {
            "model": Path(f"{config.tokenizer_path}.model"),
            "vocab": Path(f"{config.tokenizer_path}.vocab"),
        }

    def _get_model_path(self, mode, size):
        return Path(config.checkpoint_dir) / mode / size / "best.pt"

    def _on_check_files(self, _):
        self._log("=" * 40)
        self._log("🔍 Checking local files...")

        for mode in MODE_NAMES:
            self._log(f"\n[{mode}]")
            for size in MODEL_SIZES:
                path = self._get_model_path(mode, size)
                if path.exists():
                    self._log(f"  ✅ {mode}/{size}/best.pt: {path.stat().st_size/1e6:.1f} MB")
                else:
                    self._log(f"  ❌ {mode}/{size}/best.pt: Not found")

        self._log("\n[tokenizer]")
        paths = self._get_tokenizer_paths()
        for name, path in paths.items():
            if path.exists():
                self._log(f"  ✅ tokenizer.{name}: {path.stat().st_size/1024:.1f} KB")
            else:
                self._log(f"  ❌ tokenizer.{name}: Not found")

    def _on_upload(self, _):
        repo_id = self.repo_input.value.strip()
        if not repo_id:
            self._log("❌ Enter repository name")
            return

        self._log(f"⬆️ Uploading to {repo_id}...")
        try:
            self.api.create_repo(repo_id=repo_id, private=self.private_checkbox.value, exist_ok=True)

            for mode in MODE_NAMES:
                for size in MODEL_SIZES:
                    path = self._get_model_path(mode, size)
                    if path.exists():
                        self.api.upload_file(str(path), f"{mode}/{size}/best.pt", repo_id)
                        self._log(f"✅ Uploaded {mode}/{size}/best.pt")

            paths = self._get_tokenizer_paths()
            for name, path in paths.items():
                if path.exists():
                    self.api.upload_file(str(path), f"tokenizer/tokenizer.{name}", repo_id)
                    self._log(f"✅ Uploaded tokenizer.{name}")

            self._log(f"🎉 Done! https://huggingface.co/{repo_id}")
        except Exception as e:
            self._log(f"❌ Error: {e}")

    def _on_download(self, _):
        repo_id = self.repo_input.value.strip()
        if not repo_id:
            self._log("❌ Enter repository name")
            return

        self._log(f"⬇️ Downloading from {repo_id}...")
        try:
            for mode in MODE_NAMES:
                for size in MODEL_SIZES:
                    try:
                        Path(config.checkpoint_dir, mode, size).mkdir(parents=True, exist_ok=True)
                        hf_hub_download(repo_id, f"{mode}/{size}/best.pt", local_dir=config.checkpoint_dir)
                        self._log(f"✅ Downloaded {mode}/{size}/best.pt")
                    except:
                        self._log(f"⏭️ Skipped {mode}/{size}/best.pt")

            tokenizer_dir = Path(config.tokenizer_dir)
            tokenizer_dir.mkdir(parents=True, exist_ok=True)
            paths = self._get_tokenizer_paths()

            for name, local_path in paths.items():
                try:
                    import shutil
                    downloaded = hf_hub_download(repo_id, f"tokenizer/tokenizer.{name}", local_dir=".")
                    shutil.copy(downloaded, local_path)
                    self._log(f"✅ Downloaded tokenizer.{name}")
                except:
                    self._log(f"⏭️ Skipped tokenizer.{name}")

            self._log("🎉 Done!")
        except RepositoryNotFoundError:
            self._log(f"❌ Repository not found: {repo_id}")
        except Exception as e:
            self._log(f"❌ Error: {e}")

    def display(self):
        display(self.ui)


# %%
ui = HuggingFaceUI()
ui.display()
