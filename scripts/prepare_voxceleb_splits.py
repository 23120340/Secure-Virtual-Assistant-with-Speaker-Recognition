"""Create SID/SV split files from the VoxCeleb-style files that actually exist.

Use this on Kaggle instead of relying on a pre-generated split when the dataset
variant has missing or renamed utterances.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def collect_files(data_root: Path, min_utts: int) -> dict[str, list[str]]:
    spk2files: dict[str, list[str]] = {}
    for spk_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        rels = []
        for wav in sorted(spk_dir.rglob("*.wav")):
            rels.append(str(wav.relative_to(data_root)).replace("\\", "/"))
        if len(rels) >= min_utts:
            spk2files[spk_dir.name] = rels
    return spk2files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-out", default="/kaggle/working/results/voxceleb_indian/iden_split.txt")
    parser.add_argument("--trial-out", default="/kaggle/working/results/voxceleb_indian/veri_test.txt")
    parser.add_argument("--summary-out", default="/kaggle/working/results/voxceleb_indian/dataset_summary.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--min-utts", type=int, default=3)
    parser.add_argument("--positive-window", type=int, default=3)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    data_root = Path(args.data_root)
    split_out = Path(args.split_out)
    trial_out = Path(args.trial_out)
    summary_out = Path(args.summary_out)
    split_out.parent.mkdir(parents=True, exist_ok=True)
    trial_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)

    spk2files = collect_files(data_root, args.min_utts)
    if not spk2files:
        raise SystemExit(f"No .wav speakers found under {data_root}")

    iden_lines: list[str] = []
    test_by_spk: dict[str, list[str]] = {}
    counts = {"train": 0, "val": 0, "test": 0}

    for spk, files in spk2files.items():
        files = files[:]
        rng.shuffle(files)
        n_total = len(files)
        n_train = max(1, int(args.train_ratio * n_total))
        n_val = max(1, int(args.val_ratio * n_total))
        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]
        if not test_files:
            test_files = val_files[-1:]
            val_files = val_files[:-1] or train_files[-1:]

        for rel in train_files:
            iden_lines.append(f"1 {rel}\n")
        for rel in val_files:
            iden_lines.append(f"2 {rel}\n")
        for rel in test_files:
            iden_lines.append(f"3 {rel}\n")
        counts["train"] += len(train_files)
        counts["val"] += len(val_files)
        counts["test"] += len(test_files)
        test_by_spk[spk] = test_files

    pos: list[tuple[int, str, str]] = []
    neg: list[tuple[int, str, str]] = []
    speakers = sorted(test_by_spk)

    for files in test_by_spk.values():
        for i in range(len(files)):
            for j in range(i + 1, min(i + args.positive_window + 1, len(files))):
                pos.append((1, files[i], files[j]))

    for i in range(len(speakers)):
        for j in range(i + 1, len(speakers)):
            a, b = test_by_spk[speakers[i]], test_by_spk[speakers[j]]
            if a and b:
                neg.append((0, rng.choice(a), rng.choice(b)))

    k = min(len(pos), len(neg))
    trials = rng.sample(pos, k) + rng.sample(neg, k)
    rng.shuffle(trials)

    split_out.write_text("".join(iden_lines), encoding="utf-8")
    trial_out.write_text(
        "".join(f"{label} {p1} {p2}\n" for label, p1, p2 in trials),
        encoding="utf-8",
    )
    summary = {
        "dataset": "VoxCeleb Indian subset",
        "seed": args.seed,
        "data_root": str(data_root),
        "n_speakers": len(spk2files),
        "n_utterances": sum(len(v) for v in spk2files.values()),
        "split_counts": counts,
        "n_trials": len(trials),
        "n_positive_trials": sum(1 for t in trials if t[0] == 1),
        "n_negative_trials": sum(1 for t in trials if t[0] == 0),
    }
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Wrote split: {split_out}")
    print(f"Wrote trials: {trial_out}")


if __name__ == "__main__":
    main()
