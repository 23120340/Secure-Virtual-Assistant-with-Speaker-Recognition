"""
Speaker Verification evaluation: EER + minDCF trên trial pairs (VoxCeleb1-O).

Cách dùng:
    python evaluate_sv.py \
        --ckpt ./checkpoints/best_model.pt \
        --data_root /path/to/voxceleb1/wav \
        --trial_file /path/to/veri_test.txt \
        --out scores.json
"""
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
from sklearn.metrics import roc_curve
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

from train_ecapa import FbankExtractor


# ==========================================================================
# Metrics
# ==========================================================================
def compute_eer(labels, scores):
    """Equal Error Rate (%) và threshold tương ứng."""
    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return eer * 100, float(thr[idx])


def compute_min_dcf(labels, scores, p_target=0.01, c_miss=1.0, c_fa=1.0):
    """minDCF theo NIST SRE convention (p_target=0.01)."""
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    c_det = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    return float((c_det / c_def).min())


# ==========================================================================
# Embedding extraction (full utterance, không crop)
# ==========================================================================
@torch.no_grad()
def extract_embedding(feat, model, wav_path, device, sample_rate=16000):
    wav, sr = torchaudio.load(str(wav_path))
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    wav = wav.mean(dim=0).unsqueeze(0).to(device)
    mel = feat(wav)
    emb = model(mel).squeeze(1)
    emb = F.normalize(emb, dim=1)
    return emb.cpu().numpy()[0]


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    feat = FbankExtractor().to(device).eval()
    model = ECAPA_TDNN(input_size=80, lin_neurons=192,
                       channels=[512, 512, 512, 512, 1536]).to(device).eval()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint từ epoch {ckpt.get('epoch', '?')} "
          f"(val_acc={ckpt.get('val_acc', 0):.4f})")

    # Đọc trial pairs: "<label> <path1> <path2>"
    trials, files = [], set()
    with open(args.trial_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            label, p1, p2 = parts
            trials.append((int(label), p1, p2))
            files.add(p1); files.add(p2)
    print(f"Trials: {len(trials)} | Unique files: {len(files)}")

    # Trích embedding cho từng file (cache để tránh trùng lặp)
    embeddings = {}
    for path in tqdm(sorted(files), desc="Extract embeddings"):
        full = Path(args.data_root) / path
        embeddings[path] = extract_embedding(feat, model, full, device)

    # Cosine score (đã L2-normalized → dot product)
    labels, scores = [], []
    for label, p1, p2 in tqdm(trials, desc="Score trials"):
        s = float(np.dot(embeddings[p1], embeddings[p2]))
        labels.append(label); scores.append(s)
    labels = np.array(labels); scores = np.array(scores)

    eer, thresh = compute_eer(labels, scores)
    min_dcf = compute_min_dcf(labels, scores, p_target=0.01)

    print("\n" + "=" * 50)
    print("Speaker Verification Results")
    print("=" * 50)
    print(f"  EER:                {eer:.2f}%")
    print(f"  Decision threshold: {thresh:.4f}")
    print(f"  minDCF (p=0.01):    {min_dcf:.4f}")
    print("=" * 50)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "eer_percent": eer,
                "threshold": thresh,
                "min_dcf": min_dcf,
                "n_trials": len(trials),
            }, f, indent=2)
        print(f"Đã lưu kết quả vào {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--trial_file", required=True)
    p.add_argument("--out", default="sv_results.json")
    main(p.parse_args())
