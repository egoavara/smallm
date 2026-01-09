"""Checkpoint management utilities."""

import re
import torch
from pathlib import Path
from typing import List, Tuple, Optional


class CheckpointManager:
    """체크포인트 관리 클래스 - 최고 성능 N개만 유지."""

    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 5,
        save_best: bool = True,
        device: str = "cpu",
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.save_best = save_best
        self.device = device
        self.best_loss = float("inf")

    def get_best_path(self) -> Path:
        """best.pt 경로 반환."""
        return self.checkpoint_dir / "best.pt"

    def get_checkpoint_path(self, step: int, loss: float) -> Path:
        """체크포인트 파일 경로 생성 (step과 loss 포함)."""
        return self.checkpoint_dir / f"step_{step:06d}_loss_{loss:.4f}.pt"

    def list_checkpoints(self) -> List[Tuple[Path, int, float]]:
        """모든 체크포인트 목록 반환 (path, step, loss) - loss 기준 정렬."""
        checkpoints = []
        pattern = re.compile(r"step_(\d+)_loss_([\d.]+)\.pt")

        for f in self.checkpoint_dir.glob("step_*.pt"):
            match = pattern.match(f.name)
            if match:
                step = int(match.group(1))
                loss = float(match.group(2))
                checkpoints.append((f, step, loss))

        # loss 기준 오름차순 정렬 (낮은 loss가 더 좋음)
        checkpoints.sort(key=lambda x: x[2])
        return checkpoints

    def cleanup_old_checkpoints(self):
        """max_checkpoints 개수를 초과하는 오래된 체크포인트 삭제."""
        checkpoints = self.list_checkpoints()

        if len(checkpoints) > self.max_checkpoints:
            # loss가 높은 (성능이 나쁜) 체크포인트들 삭제
            to_delete = checkpoints[self.max_checkpoints:]
            for path, step, loss in to_delete:
                path.unlink()
                print(f"   🗑️  Deleted: {path.name}")

    def save(
        self,
        step: int,
        loss: float,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_history: list,
    ) -> str:
        """체크포인트 저장 및 관리."""
        checkpoint = {
            "step": step,
            "loss": loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss_history": loss_history,
            "model_config": {
                "n_layers": model.config.n_layers,
                "n_heads": model.config.n_heads,
                "n_kv_heads": model.config.n_kv_heads,
                "d_model": model.config.d_model,
                "d_ff": model.config.d_ff,
                "vocab_size": model.config.vocab_size,
                "max_seq_len": model.config.max_seq_len,
            },
        }

        saved_paths = []

        # best.pt 저장 (현재 loss가 best보다 낮으면)
        if self.save_best and loss < self.best_loss:
            self.best_loss = loss
            best_path = self.get_best_path()
            torch.save(checkpoint, best_path)
            saved_paths.append(f"best.pt (loss: {loss:.4f})")

        # 일반 체크포인트 저장
        ckpt_path = self.get_checkpoint_path(step, loss)
        torch.save(checkpoint, ckpt_path)
        saved_paths.append(ckpt_path.name)

        # 오래된 체크포인트 정리
        self.cleanup_old_checkpoints()

        return ", ".join(saved_paths)

    def load_best(
        self, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Tuple[Optional[int], list]:
        """best.pt 로드. 성공 시 (step, loss_history) 반환, 없으면 (None, [])."""
        best_path = self.get_best_path()
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            step = checkpoint["step"]
            loss = checkpoint.get("loss", float("inf"))
            self.best_loss = loss
            print(f"Loaded best.pt (step: {step}, loss: {loss:.4f})")
            return step, checkpoint.get("loss_history", [])
        return None, []

    def load_checkpoint(
        self, path: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Tuple[int, list]:
        """특정 체크포인트 로드."""
        checkpoint = torch.load(path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        step = checkpoint["step"]
        loss = checkpoint.get("loss", float("inf"))
        if loss < self.best_loss:
            self.best_loss = loss
        return step, checkpoint.get("loss_history", [])
