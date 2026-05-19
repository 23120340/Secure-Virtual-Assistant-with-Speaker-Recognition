"""MetricLogger — wrap TensorBoard + Wandb dispatch trong 1 API duy nhất.

Hai backend đều OPTIONAL — không cài cũng không sao, MetricLogger no-op.
Cách bật:
    1. TensorBoard: `pip install tensorboard` + CLI flag `--tensorboard`.
       Output: `<save_dir>/runs/` → mở UI bằng `tensorboard --logdir <save_dir>/runs`.
       Không cần account, hoàn toàn local.
    2. Wandb: `pip install wandb` + flag `--wandb` + env `WANDB_API_KEY` (free tier
       OK cho personal/academic). Project name lấy từ `--wandb-project`
       (default `secva-ecapa-tdnn`).

Caller dùng:
    logger = MetricLogger(save_dir=save_dir,
                          tensorboard=args.tensorboard,
                          wandb=args.wandb,
                          wandb_project=args.wandb_project,
                          wandb_run_name=f"epoch{args.epochs}-bs{args.batch_size}",
                          hparams=hparams_dict)
    try:
        for step in ...:
            logger.log_scalar("train/loss", loss, step)
        logger.log_summary({"final_val_acc": best_val})
    finally:
        logger.close()

Best-effort: nếu Wandb/TensorBoard fail (network, install missing) → log warning,
KHÔNG raise. Training tiếp tục bình thường.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

_log = logging.getLogger("metric_logger")


class MetricLogger:
    """Dispatch metrics tới TensorBoard và/hoặc Wandb. No-op khi cả 2 disabled."""

    def __init__(self,
                 save_dir: Path,
                 *,
                 tensorboard: bool = False,
                 wandb: bool = False,
                 wandb_project: str = "secva-ecapa-tdnn",
                 wandb_run_name: str | None = None,
                 hparams: dict | None = None):
        self.save_dir = Path(save_dir)
        self._tb_writer = None
        self._wandb_run = None

        if tensorboard:
            self._init_tensorboard()
        if wandb:
            self._init_wandb(project=wandb_project,
                             run_name=wandb_run_name,
                             config=hparams or {})

    # ----- Backend init: TensorBoard ----------------------------------------
    def _init_tensorboard(self):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            _log.warning("TensorBoard chưa cài (pip install tensorboard) — bỏ qua")
            return
        try:
            tb_dir = self.save_dir / "runs"
            tb_dir.mkdir(parents=True, exist_ok=True)
            self._tb_writer = SummaryWriter(log_dir=str(tb_dir))
            _log.info("TensorBoard logging vào %s "
                      "(mở UI: tensorboard --logdir %s)", tb_dir, tb_dir)
        except Exception as e:
            _log.warning("TensorBoard init failed: %s — bỏ qua", e)

    # ----- Backend init: Wandb ----------------------------------------------
    def _init_wandb(self, project: str, run_name: str | None, config: dict):
        try:
            import wandb  # type: ignore
        except ImportError:
            _log.warning("Wandb chưa cài (pip install wandb) — bỏ qua")
            return
        if not os.environ.get("WANDB_API_KEY"):
            _log.warning("WANDB_API_KEY chưa set — bỏ qua wandb. "
                         "Đăng ký free tại https://wandb.ai/ → Settings → API keys.")
            return
        try:
            self._wandb_run = wandb.init(
                project=project,
                name=run_name,
                config=config,
                dir=str(self.save_dir),
                # reinit cho phép gọi lại nhiều lần trong cùng process (script test).
                reinit=True,
            )
            _log.info("Wandb đang log vào project='%s' run='%s' (%s)",
                      project, run_name, self._wandb_run.url)
        except Exception as e:
            _log.warning("Wandb init failed: %s — bỏ qua", e)

    # ----- Public API: log metrics -----------------------------------------
    def log_scalar(self, name: str, value: float, step: int):
        """Log 1 scalar metric. Gọi mỗi batch hoặc mỗi epoch.

        `name` theo convention `<phase>/<metric>`:
            - "train/loss", "train/acc", "train/lr"
            - "val/loss",   "val/acc"
        `step` thường là global_step (batch_idx + epoch * len(loader)).
        """
        # KHÔNG bọc try/except per-call vì nếu backend đã init OK thì add_scalar
        # rất ổn định; pollute log với warning mỗi call sẽ ồn.
        if self._tb_writer is not None:
            self._tb_writer.add_scalar(name, value, step)
        if self._wandb_run is not None:
            self._wandb_run.log({name: value}, step=step)

    def log_scalars(self, metrics: dict, step: int):
        """Convenience: log nhiều scalar 1 lần."""
        for name, value in metrics.items():
            self.log_scalar(name, value, step)

    def log_summary(self, summary: dict):
        """Log final summary (best_val_acc, training_time...) — call ở cuối training."""
        if self._tb_writer is not None:
            for k, v in summary.items():
                # TB hparams cần dict hparams + metric, đây dùng add_text cho gọn.
                self._tb_writer.add_text(f"summary/{k}", str(v))
        if self._wandb_run is not None:
            self._wandb_run.summary.update(summary)

    def close(self):
        """Flush + close. Idempotent — safe gọi nhiều lần."""
        if self._tb_writer is not None:
            try:
                self._tb_writer.flush()
                self._tb_writer.close()
            except Exception:
                pass
            self._tb_writer = None
        if self._wandb_run is not None:
            try:
                self._wandb_run.finish()
            except Exception:
                pass
            self._wandb_run = None

    @property
    def enabled(self) -> bool:
        return self._tb_writer is not None or self._wandb_run is not None
