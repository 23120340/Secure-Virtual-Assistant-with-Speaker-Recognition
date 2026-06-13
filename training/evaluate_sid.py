"""
Speaker Identification evaluation: Top-1 và Top-5 accuracy trên VoxCeleb1 test split.

Cách dùng:
    python evaluate_sid.py \
        --ckpt ./checkpoints/best_model.pt \
        --spk2idx ./checkpoints/spk2idx.json \
        --data_root /path/to/voxceleb1/wav \
        --split_file /path/to/iden_split.txt
"""
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

from train_ecapa import VoxCelebSID, FbankExtractor, AAMSoftmax


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.spk2idx) as f:
        spk2idx = json.load(f)

    feat = FbankExtractor().to(device).eval()
    model = ECAPA_TDNN(input_size=80, lin_neurons=192,
                       channels=[512, 512, 512, 512, 1536]).to(device).eval()
    classifier = AAMSoftmax(192, len(spk2idx)).to(device).eval()

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    classifier.load_state_dict(ckpt["classifier"])
    print(f"Loaded checkpoint từ epoch {ckpt.get('epoch', '?')}")

    # Test set: dùng cùng spk2idx như khi train
    test_ds = VoxCelebSID(args.data_root, args.split_file, "test",
                          duration=5.0, spk2idx=spk2idx)
    print(f"Test samples: {len(test_ds)}")
    loader = DataLoader(test_ds, batch_size=args.batch_size,
                        num_workers=args.num_workers)

    correct1, correct5, total = 0, 0, 0
    n_classes = len(spk2idx)
    y_true, y_pred = [], []  # gom lại để tính precision/recall/F1 + confusion matrix
    # Pre-normalize trọng số classifier để inference bằng cosine
    w_n = F.normalize(classifier.weight, dim=1)

    with torch.no_grad():
        for wav, label in tqdm(loader, desc="Test"):
            wav, label = wav.to(device), label.to(device)
            mel = feat(wav)
            emb = F.normalize(model(mel).squeeze(1), dim=1)
            logits = emb @ w_n.t()  # cosine similarity với từng speaker

            top1 = logits.argmax(dim=1)
            top5 = logits.topk(5, dim=1).indices
            correct1 += (top1 == label).sum().item()
            correct5 += (top5 == label.unsqueeze(1)).any(dim=1).sum().item()
            total += label.size(0)
            y_true.extend(label.tolist())
            y_pred.extend(top1.tolist())

    # ── Macro / weighted F1 + per-class precision/recall (numpy thuần) ──────
    # Confusion matrix C[t, p] = số mẫu lớp thật t bị đoán thành p.
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)      # số mẫu thật mỗi lớp
    pred_count = cm.sum(axis=0).astype(np.float64)    # số lần được đoán mỗi lớp
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(pred_count > 0, tp / pred_count, 0.0)
        recall = np.where(support > 0, tp / support, 0.0)
        denom = precision + recall
        f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)
    present = support > 0  # chỉ tính trung bình trên các lớp có mặt trong test
    macro_f1 = float(f1[present].mean()) if present.any() else 0.0
    weighted_f1 = float((f1 * support).sum() / support.sum()) if support.sum() else 0.0
    micro_f1 = correct1 / total  # = Top-1 accuracy với single-label multiclass

    print("\n" + "=" * 50)
    print("Speaker Identification Results")
    print("=" * 50)
    print(f"  Test samples:    {total}")
    print(f"  Top-1 accuracy:  {100*correct1/total:.2f}%")
    print(f"  Top-5 accuracy:  {100*correct5/total:.2f}%")
    print(f"  Micro-F1:        {100*micro_f1:.2f}%  (= Top-1 accuracy)")
    print(f"  Macro-F1:        {100*macro_f1:.2f}%")
    print(f"  Weighted-F1:     {100*weighted_f1:.2f}%")
    print("=" * 50)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "top1_accuracy": 100 * correct1 / total,
                "top5_accuracy": 100 * correct5 / total,
                "micro_f1": 100 * micro_f1,
                "macro_f1": 100 * macro_f1,
                "weighted_f1": 100 * weighted_f1,
                "n_test": total,
                "n_speakers": n_classes,
                "per_class": [
                    {"speaker_idx": i, "support": int(support[i]),
                     "precision": float(precision[i]), "recall": float(recall[i]),
                     "f1": float(f1[i])}
                    for i in range(n_classes)
                ],
            }, f, indent=2)

    if args.confusion_out:
        with open(args.confusion_out, "w") as f:
            json.dump({"labels": list(spk2idx.keys()),
                       "matrix": cm.tolist()}, f, indent=2)
        print(f"Confusion matrix -> {args.confusion_out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--spk2idx", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--split_file", required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--out", default="sid_results.json")
    p.add_argument("--confusion-out", default=None,
                   help="Nếu set, lưu confusion matrix (labels + ma trận) ra JSON")
    main(p.parse_args())
