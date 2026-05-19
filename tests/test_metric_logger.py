"""Test MetricLogger framework — verify no-op khi backend không cài / không bật."""
import sys
from pathlib import Path

# training/ chưa phải Python package — add vào sys.path để import được.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "training"))

from metric_logger import MetricLogger


def test_metric_logger_disabled_is_noop(tmp_path):
    """Cả 2 backend tắt → enabled=False, log_scalar không crash."""
    lg = MetricLogger(save_dir=tmp_path)
    assert lg.enabled is False
    lg.log_scalar("train/loss", 0.5, step=1)
    lg.log_scalars({"a": 1, "b": 2}, step=2)
    lg.log_summary({"final": 0.95})
    lg.close()
    lg.close()  # idempotent


def test_metric_logger_tensorboard_enabled_when_available(tmp_path, monkeypatch):
    """Bật --tensorboard + lib có → SummaryWriter init, file `runs/` được tạo."""
    try:
        import torch  # noqa: F401
        from torch.utils.tensorboard import SummaryWriter  # noqa: F401
    except ImportError:
        # No tensorboard available trong venv test — skip mà không fail.
        import pytest
        pytest.skip("tensorboard không cài — skip test enabled")

    lg = MetricLogger(save_dir=tmp_path, tensorboard=True)
    assert lg.enabled is True
    lg.log_scalar("train/loss", 0.42, step=0)
    lg.log_scalar("train/loss", 0.41, step=1)
    lg.log_summary({"best_val_acc": 0.91})
    lg.close()
    # Folder `runs/` được tạo.
    assert (tmp_path / "runs").exists()
    # SummaryWriter ghi file event nhỏ.
    files = list((tmp_path / "runs").iterdir())
    assert len(files) >= 1


def test_metric_logger_wandb_skipped_when_no_api_key(tmp_path, monkeypatch):
    """Bật --wandb nhưng WANDB_API_KEY không set → skip silently, không raise."""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    # Cố tình bật wandb=True; expect graceful no-op (warning log).
    lg = MetricLogger(save_dir=tmp_path, wandb=True)
    # Có thể vẫn enabled nếu TB module có; mục đích test là không CRASH.
    lg.log_scalar("x", 1, 0)
    lg.close()


def test_metric_logger_close_is_idempotent(tmp_path):
    lg = MetricLogger(save_dir=tmp_path)
    lg.close()
    lg.close()
    lg.close()  # gọi nhiều lần không lỗi
