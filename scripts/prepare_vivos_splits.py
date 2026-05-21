"""Prepare VIVOS for ECAPA training and create SID/SV split files.

The training dataset class expects relative paths in the form:
    <speaker_id>/<utterance>.wav

VIVOS is commonly distributed as:
    vivos/train/waves/<speaker_id>/*.wav
    vivos/test/waves/<speaker_id>/*.wav

This script copies both subsets into one speaker-root and writes:
    iden_split_vivos.txt  # 1=train, 2=val, 3=test
    veri_test_vivos.txt   # <label> <path1> <path2>
    vivos_summary.json
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


def _find_waves_root(vivos_root: Path, subset: str) -> Path | None:
    candidates = [
        vivos_root / subset / "waves",
        vivos_root / "vivos" / subset / "waves",
        vivos_root / "VIVOS" / subset / "waves",
    ]
    return next((p for p in candidates if p.exists()), None)


def copy_to_speaker_root(vivos_root: Path, out_root: Path) -> dict:
    out_root.mkdir(parents=True, exist_ok=True)
    copied = 0
    speakers: set[str] = set()

    for subset in ("train", "test"):
        waves = _find_waves_root(vivos_root, subset)
        if waves is None:
            continue
        for spk_dir in sorted(p for p in waves.iterdir() if p.is_dir()):
            speakers.add(spk_dir.name)
            dst_spk = out_root / spk_dir.name
            dst_spk.mkdir(exist_ok=True)
            for wav in sorted(spk_dir.glob("*.wav")):
                dst = dst_spk / f"{subset}_{wav.name}"
                if not dst.exists():
                    shutil.copy2(wav, dst)
                    copied += 1

    return {"speakers_seen": len(speakers), "copied_files": copied}


def collect_speaker_files(data_root: Path, min_utts: int) -> dict[str, list[str]]:
    spk2files: dict[str, list[str]] = {}
    for spk_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        files = sorted(spk_dir.glob("*.wav"))
        if len(files) >= min_utts:
            spk2files[spk_dir.name] = [f"{spk_dir.name}/{p.name}" for p in files]
    return spk2files


def filter_readable_audio(
    spk2files: dict[str, list[str]],
    data_root: Path,
    *,
    min_utts: int,
    bad_out: Path | None = None,
) -> tuple[dict[str, list[str]], list[str]]:
    """Keep only files that torchaudio can decode, matching train_ecapa.py."""
    try:
        import torchaudio
    except Exception as exc:
        raise SystemExit(f"--validate-audio requires torchaudio: {exc}") from exc

    clean: dict[str, list[str]] = {}
    bad: list[str] = []
    for spk, rels in spk2files.items():
        kept = []
        for rel in rels:
            try:
                torchaudio.load(str(data_root / rel))
                kept.append(rel)
            except Exception:
                bad.append(rel)
        if len(kept) >= min_utts:
            clean[spk] = kept

    if bad_out is not None:
        bad_out.parent.mkdir(parents=True, exist_ok=True)
        bad_out.write_text("\n".join(bad) + ("\n" if bad else ""), encoding="utf-8")
    return clean, bad


def make_iden_split(
    spk2files: dict[str, list[str]],
    *,
    train_ratio: float,
    val_ratio: float,
    rng: random.Random,
) -> tuple[list[str], dict[str, list[str]], dict]:
    iden_lines: list[str] = []
    test_by_spk: dict[str, list[str]] = {}
    counts = {"train": 0, "val": 0, "test": 0}

    for spk, files in spk2files.items():
        files = files[:]
        rng.shuffle(files)

        n_total = len(files)
        n_train = max(1, int(train_ratio * n_total))
        n_val = max(1, int(val_ratio * n_total))

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

    return iden_lines, test_by_spk, counts


def make_trials(
    test_by_spk: dict[str, list[str]],
    *,
    rng: random.Random,
    positive_window: int,
) -> list[tuple[int, str, str]]:
    pos: list[tuple[int, str, str]] = []
    neg: list[tuple[int, str, str]] = []
    speakers = sorted(test_by_spk)

    for files in test_by_spk.values():
        for i in range(len(files)):
            for j in range(i + 1, min(i + positive_window + 1, len(files))):
                pos.append((1, files[i], files[j]))

    for i in range(len(speakers)):
        for j in range(i + 1, len(speakers)):
            a = test_by_spk[speakers[i]]
            b = test_by_spk[speakers[j]]
            if a and b:
                neg.append((0, rng.choice(a), rng.choice(b)))

    k = min(len(pos), len(neg))
    trials = rng.sample(pos, k) + rng.sample(neg, k)
    rng.shuffle(trials)
    return trials


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vivos-root", required=True,
                        help="Path to VIVOS root, e.g. /kaggle/input/vivos-corpus/vivos")
    parser.add_argument("--out-root", default="/kaggle/working/vivos_speaker_root",
                        help="Output speaker root with <speaker>/<wav>.wav")
    parser.add_argument("--split-out", default="/kaggle/working/iden_split_vivos.txt")
    parser.add_argument("--trial-out", default="/kaggle/working/veri_test_vivos.txt")
    parser.add_argument("--summary-out", default="/kaggle/working/vivos_summary.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--min-utts", type=int, default=3)
    parser.add_argument("--positive-window", type=int, default=3)
    parser.add_argument("--validate-audio", action="store_true",
                        help="Drop wav files that torchaudio cannot decode")
    parser.add_argument("--bad-audio-out", default="/kaggle/working/results/vivos/bad_audio_files.txt")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    vivos_root = Path(args.vivos_root)
    out_root = Path(args.out_root)
    split_out = Path(args.split_out)
    trial_out = Path(args.trial_out)
    summary_out = Path(args.summary_out)

    copy_stats = copy_to_speaker_root(vivos_root, out_root)
    spk2files = collect_speaker_files(out_root, args.min_utts)
    bad_files: list[str] = []
    if args.validate_audio:
        spk2files, bad_files = filter_readable_audio(
            spk2files,
            out_root,
            min_utts=args.min_utts,
            bad_out=Path(args.bad_audio_out),
        )
    if not spk2files:
        raise SystemExit(f"No speakers found under {out_root}. Check --vivos-root.")

    iden_lines, test_by_spk, split_counts = make_iden_split(
        spk2files,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        rng=rng,
    )
    trials = make_trials(
        test_by_spk,
        rng=rng,
        positive_window=args.positive_window,
    )

    split_out.parent.mkdir(parents=True, exist_ok=True)
    trial_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)

    split_out.write_text("".join(iden_lines), encoding="utf-8")
    trial_out.write_text(
        "".join(f"{label} {p1} {p2}\n" for label, p1, p2 in trials),
        encoding="utf-8",
    )

    summary = {
        "dataset": "VIVOS",
        "seed": args.seed,
        "data_root": str(out_root),
        "n_speakers": len(spk2files),
        "n_utterances": sum(len(v) for v in spk2files.values()),
        "split_counts": split_counts,
        "n_trials": len(trials),
        "n_positive_trials": sum(1 for t in trials if t[0] == 1),
        "n_negative_trials": sum(1 for t in trials if t[0] == 0),
        "n_bad_audio_files": len(bad_files),
        "bad_audio_out": args.bad_audio_out if args.validate_audio else "",
        **copy_stats,
    }
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Wrote split: {split_out}")
    print(f"Wrote trials: {trial_out}")
    print(f"Wrote summary: {summary_out}")


if __name__ == "__main__":
    main()
