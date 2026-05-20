"""
Train ECAPA-TDNN on VoxCeleb1 (Speaker Identification, closed-set).
The trained model is reused later for Speaker Verification via cosine similarity
on speaker embeddings.

Usage:
    python train_ecapa.py \
        --data_root /path/to/voxceleb1/wav \
        --split_file /path/to/iden_split.txt \
        --save_dir ./checkpoints \
        --epochs 20 --batch_size 64 --lr 1e-3
"""
import os
import json
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

try:
    from .augmentations import SpecAugment, WaveformAugmenter
except ImportError:
    from augmentations import SpecAugment, WaveformAugmenter


# ==========================================================================
# AAM-Softmax (ArcFace) loss: chuẩn cho speaker recognition hiện đại
# ==========================================================================
class AAMSoftmax(nn.Module):
    """Additive Angular Margin Softmax."""
    def __init__(self, embed_dim, num_classes, margin=0.2, scale=30.0):
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.empty(num_classes, embed_dim))
        nn.init.xavier_normal_(self.weight)

    def forward(self, embeddings, labels):
        # L2-normalize cả embedding và weight để cos_theta = dot product
        emb_n = F.normalize(embeddings, dim=1)
        w_n = F.normalize(self.weight, dim=1)
        cos_theta = F.linear(emb_n, w_n).clamp(-1 + 1e-7, 1 - 1e-7)

        # Cộng margin vào ground-truth class
        theta = torch.acos(cos_theta)
        target_mask = F.one_hot(labels, num_classes=cos_theta.size(1)).bool()
        theta_m = torch.where(target_mask, theta + self.margin, theta)
        logits = self.scale * torch.cos(theta_m)

        loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        return loss, acc


# ==========================================================================
# Dataset: đọc iden_split.txt của VoxCeleb1 (1=train, 2=val, 3=test)
# Mỗi dòng có dạng: "<split_id> idXXXXX/<youtube_id>/<utt>.wav"
# ==========================================================================
class VoxCelebSID(Dataset):
    SPLIT_MAP = {"train": 1, "val": 2, "test": 3}

    def __init__(self, root_dir, split_file, split,
                 sample_rate=16000, duration=3.0, spk2idx=None,
                 waveform_augmenter=None):
        self.root_dir = Path(root_dir)
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * duration)
        self.split = split
        self.waveform_augmenter = waveform_augmenter

        target = self.SPLIT_MAP[split]
        self.files = []
        with open(split_file) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                if int(parts[0]) == target:
                    self.files.append(parts[1])

        # Speaker ID từ phần đầu path: "id10001/.../xxx.wav"
        if spk2idx is None:
            speakers = sorted({p.split("/")[0] for p in self.files})
            self.spk2idx = {s: i for i, s in enumerate(speakers)}
        else:
            self.spk2idx = spk2idx

    def __len__(self):
        return len(self.files)

    def _load(self, path):
        wav, sr = torchaudio.load(str(path))
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        wav = wav.mean(dim=0)  # mono
        # Random crop khi train, center crop khi eval
        if wav.size(0) >= self.num_samples:
            if self.split == "train":
                start = random.randint(0, wav.size(0) - self.num_samples)
            else:
                start = (wav.size(0) - self.num_samples) // 2
            wav = wav[start:start + self.num_samples]
        else:
            wav = F.pad(wav, (0, self.num_samples - wav.size(0)))
        if self.split == "train" and self.waveform_augmenter is not None:
            wav = self.waveform_augmenter(wav)
        return wav

    def __getitem__(self, idx):
        rel = self.files[idx]
        spk = rel.split("/")[0]
        label = self.spk2idx[spk]
        wav = self._load(self.root_dir / rel)
        return wav, label


# ==========================================================================
# Mel-filterbank feature: 80 mel bins, log + CMN
# ==========================================================================
class FbankExtractor(nn.Module):
    def __init__(self, sample_rate=16000, n_mels=80):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=512, win_length=400,
            hop_length=160, n_mels=n_mels,
        )

    def forward(self, x):
        # x: [B, T] -> [B, n_mels, T'] -> log -> CMN -> [B, T', n_mels]
        m = self.mel(x)
        m = torch.log(m + 1e-6)
        m = m - m.mean(dim=-1, keepdim=True)
        return m.transpose(1, 2)


# ==========================================================================
# Train loop
# ==========================================================================
def run_epoch(loader, feat, model, classifier, optimizer, device, train=True,
              *, metric_logger=None, epoch_idx: int = 0, spec_augment=None):
    """Run 1 epoch train hoặc eval.

    `metric_logger` (optional MetricLogger): nếu truyền, log per-batch loss/acc
    với global_step = epoch_idx * len(loader) + batch_idx. Chỉ log khi train.
    """
    if train:
        model.train(); classifier.train()
    else:
        model.eval(); classifier.eval()

    total_loss, total_acc, n = 0.0, 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    desc = "train" if train else "val"
    with ctx:
        for batch_idx, (wav, label) in enumerate(tqdm(loader, desc=desc, leave=False)):
            wav, label = wav.to(device), label.to(device)
            mel = feat(wav)
            if train and spec_augment is not None:
                mel = spec_augment(mel)
            emb = model(mel).squeeze(1)  # [B, 1, D] -> [B, D]
            loss, acc = classifier(emb, label)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(classifier.parameters()), 5.0
                )
                optimizer.step()

            bs = label.size(0)
            total_loss += loss.item() * bs
            total_acc += acc.item() * bs
            n += bs

            # Per-batch logging (chỉ train) — giúp debug spike trong epoch.
            if train and metric_logger is not None and metric_logger.enabled:
                global_step = epoch_idx * len(loader) + batch_idx
                metric_logger.log_scalars({
                    "train/batch_loss": loss.item(),
                    "train/batch_acc":  acc.item(),
                    "train/lr":         optimizer.param_groups[0]["lr"],
                }, step=global_step)
    return total_loss / n, total_acc / n


def main(args):
    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    waveform_augmenter = None
    if args.augment:
        waveform_augmenter = WaveformAugmenter(
            sample_rate=args.sample_rate,
            num_samples=int(args.sample_rate * args.duration),
            musan_root=args.musan_root,
            rir_root=args.rir_root,
            speed_rates=args.speed_rates,
            speed_prob=args.speed_prob,
            noise_prob=args.noise_prob,
            rir_prob=args.rir_prob,
            snr_db=(args.snr_min, args.snr_max),
        )
        print(
            "Augmentation enabled | "
            f"MUSAN files={len(waveform_augmenter.musan_files)} | "
            f"RIR files={len(waveform_augmenter.rir_files)} | "
            f"speed={args.speed_rates}"
        )

    train_ds = VoxCelebSID(
        args.data_root, args.split_file, "train",
        sample_rate=args.sample_rate, duration=args.duration,
        waveform_augmenter=waveform_augmenter,
    )
    # Giữ chung mapping speaker giữa train/val
    val_ds = VoxCelebSID(
        args.data_root, args.split_file, "val",
        sample_rate=args.sample_rate, duration=args.duration,
        spk2idx=train_ds.spk2idx,
    )
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | "
          f"Speakers: {len(train_ds.spk2idx)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    feat = FbankExtractor(sample_rate=args.sample_rate).to(device)
    spec_augment = None
    if args.augment and args.specaugment:
        spec_augment = SpecAugment(
            freq_masks=args.freq_masks,
            time_masks=args.time_masks,
            max_freq_width=args.max_freq_width,
            max_time_width=args.max_time_width,
        ).to(device)
    model = ECAPA_TDNN(input_size=80, lin_neurons=192,
                       channels=[512, 512, 512, 512, 1536]).to(device)
    classifier = AAMSoftmax(192, len(train_ds.spk2idx)).to(device)

    params = list(model.parameters()) + list(classifier.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=2e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "spk2idx.json", "w") as f:
        json.dump(train_ds.spk2idx, f, indent=2)

    # Hparams snapshot: cần cho reproducibility ở báo cáo.
    import sys, subprocess
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_sha = "unknown"
    hparams = {
        "args":          vars(args),
        "n_train":       len(train_ds),
        "n_val":         len(val_ds),
        "n_speakers":    len(train_ds.spk2idx),
        "device":        device,
        "python":        sys.version.split()[0],
        "torch_version": torch.__version__,
        "git_sha":       git_sha,
        "seed":          42,
    }
    with open(save_dir / "hparams.json", "w") as f:
        json.dump(hparams, f, indent=2)

    # Khởi tạo MetricLogger — no-op nếu không bật flag --tensorboard/--wandb.
    # Per-batch loss + per-epoch summary log vào backend tương ứng (Wandb cloud
    # hoặc TensorBoard local). Đọc thêm: training/metric_logger.py.
    from metric_logger import MetricLogger
    metric_logger = MetricLogger(
        save_dir=save_dir,
        tensorboard=args.tensorboard,
        wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=(args.wandb_run_name
                        or f"e{args.epochs}-bs{args.batch_size}-lr{args.lr}"),
        hparams=hparams,
    )

    log = []
    best_val = 0.0
    try:
        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_acc = run_epoch(
                train_loader, feat, model, classifier, optimizer, device,
                train=True, metric_logger=metric_logger, epoch_idx=epoch - 1,
                spec_augment=spec_augment,
            )
            val_loss, val_acc = run_epoch(val_loader, feat, model, classifier,
                                          optimizer, device, train=False)
            scheduler.step()
            print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
                  f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
            log.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": val_loss, "val_acc": val_acc})

            # Per-epoch summary metrics. step=epoch để TB/Wandb có biểu đồ epoch-level
            # song song với per-batch (đã log ở trên với step=global_step).
            metric_logger.log_scalars({
                "epoch/train_loss": tr_loss, "epoch/train_acc": tr_acc,
                "epoch/val_loss":   val_loss, "epoch/val_acc":   val_acc,
            }, step=epoch)

            if val_acc > best_val:
                best_val = val_acc
                torch.save({
                    "model": model.state_dict(),
                    "classifier": classifier.state_dict(),
                    "epoch": epoch, "val_acc": val_acc,
                }, save_dir / "best_model.pt")
                print(f"  ✓ Saved best (val_acc={val_acc:.4f})")
    finally:
        # Final summary + flush — chạy cả khi training fail giữa chừng.
        metric_logger.log_summary({
            "best_val_acc": best_val,
            "total_epochs": len(log),
            "n_speakers":   len(train_ds.spk2idx),
        })
        metric_logger.close()

    with open(save_dir / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nBest val accuracy: {best_val:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True,
                   help="Path tới thư mục chứa wav files (id10001/, id10002/, ...)")
    p.add_argument("--split_file", required=True,
                   help="Path tới iden_split.txt")
    p.add_argument("--save_dir", default="./checkpoints")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--duration", type=float, default=3.0,
                   help="Training crop length in seconds")
    # Training augmentation (optional; best run on Kaggle/GPU retrain)
    p.add_argument("--augment", action="store_true",
                   help="Enable training-only augmentation pipeline")
    p.add_argument("--musan-root", default=None,
                   help="Root folder containing MUSAN speech/music/noise audio")
    p.add_argument("--rir-root", default=None,
                   help="Root folder containing RIR impulse-response audio")
    p.add_argument("--speed-rates", type=float, nargs="+",
                   default=[0.9, 1.0, 1.1],
                   help="Speed perturb factors sampled during training")
    p.add_argument("--speed-prob", type=float, default=1.0)
    p.add_argument("--noise-prob", type=float, default=0.5)
    p.add_argument("--rir-prob", type=float, default=0.3)
    p.add_argument("--snr-min", type=float, default=5.0)
    p.add_argument("--snr-max", type=float, default=20.0)
    p.add_argument("--specaugment", action="store_true",
                   help="Apply SpecAugment on log-mel features during training")
    p.add_argument("--freq-masks", type=int, default=2)
    p.add_argument("--time-masks", type=int, default=2)
    p.add_argument("--max-freq-width", type=int, default=8)
    p.add_argument("--max-time-width", type=int, default=40)
    # ── Metric logging (optional) ────────────────────────────────────────
    # Cả 2 backend đều no-op khi flag tắt (default). Bật khi cần debug/share:
    #   --tensorboard         → log local vào <save_dir>/runs (no API key)
    #   --wandb               → log cloud (cần WANDB_API_KEY env)
    #   --wandb-project NAME  → tên project trên wandb.ai (default secva-ecapa-tdnn)
    #   --wandb-run-name NAME → tên run, default auto theo hparams
    p.add_argument("--tensorboard", action="store_true",
                   help="Bật TensorBoard logging vào <save_dir>/runs (no API key)")
    p.add_argument("--wandb", action="store_true",
                   help="Bật Wandb cloud logging (cần WANDB_API_KEY env)")
    p.add_argument("--wandb-project",
                   default=os.getenv("WANDB_PROJECT", "secva-ecapa-tdnn"),
                   help="Tên project Wandb")
    p.add_argument("--wandb-run-name", default=None,
                   help="Tên run Wandb (default auto theo hparams)")
    main(p.parse_args())
