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
                 sample_rate=16000, duration=3.0, spk2idx=None):
        self.root_dir = Path(root_dir)
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * duration)
        self.split = split

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
def run_epoch(loader, feat, model, classifier, optimizer, device, train=True):
    if train:
        model.train(); classifier.train()
    else:
        model.eval(); classifier.eval()

    total_loss, total_acc, n = 0.0, 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    desc = "train" if train else "val"
    with ctx:
        for wav, label in tqdm(loader, desc=desc, leave=False):
            wav, label = wav.to(device), label.to(device)
            mel = feat(wav)
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
    return total_loss / n, total_acc / n


def main(args):
    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_ds = VoxCelebSID(args.data_root, args.split_file, "train")
    # Giữ chung mapping speaker giữa train/val
    val_ds = VoxCelebSID(args.data_root, args.split_file, "val",
                         spk2idx=train_ds.spk2idx)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | "
          f"Speakers: {len(train_ds.spk2idx)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    feat = FbankExtractor().to(device)
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

    log = []
    best_val = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(train_loader, feat, model, classifier,
                                    optimizer, device, train=True)
        val_loss, val_acc = run_epoch(val_loader, feat, model, classifier,
                                      optimizer, device, train=False)
        scheduler.step()
        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
              f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        log.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                    "val_loss": val_loss, "val_acc": val_acc})

        if val_acc > best_val:
            best_val = val_acc
            torch.save({
                "model": model.state_dict(),
                "classifier": classifier.state_dict(),
                "epoch": epoch, "val_acc": val_acc,
            }, save_dir / "best_model.pt")
            print(f"  ✓ Saved best (val_acc={val_acc:.4f})")

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
    main(p.parse_args())
