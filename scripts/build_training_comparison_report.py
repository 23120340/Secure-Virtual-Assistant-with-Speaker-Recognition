"""Build a Markdown comparison report from two training result folders.

Expected files per folder:
    hparams.json
    training_log.json
    sid_results.json
    sv_results.json
    vivos_summary.json or dataset_summary.json (optional)
    iden_split.txt and veri_test.txt (optional)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def count_split(path: Path) -> dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0}
    if not path.exists():
        return counts
    mapping = {"1": "train", "2": "val", "3": "test"}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts and parts[0] in mapping:
            counts[mapping[parts[0]]] += 1
    return counts


def count_trials(path: Path) -> dict[str, int]:
    out = {"trials": 0, "positive": 0, "negative": 0}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        out["trials"] += 1
        if parts[0] == "1":
            out["positive"] += 1
        elif parts[0] == "0":
            out["negative"] += 1
    return out


def best_val_acc(training_log: Any) -> float | None:
    if not isinstance(training_log, list) or not training_log:
        return None
    values = []
    for row in training_log:
        if isinstance(row, dict):
            for key in ("val_acc", "valid_acc", "best_val_acc"):
                if key in row:
                    values.append(float(row[key]))
                    break
    return max(values) if values else None


def fmt(value: Any, *, percent: bool = False, digits: int = 2) -> str:
    if value is None:
        return "TODO"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if percent:
        return f"{value:.{digits}f}%"
    return f"{value:.{digits}f}"


def summarize(name: str, folder: Path) -> dict[str, Any]:
    hparams = load_json(folder / "hparams.json") or {}
    training_log = load_json(folder / "training_log.json")
    sid = load_json(folder / "sid_results.json") or {}
    sv = load_json(folder / "sv_results.json") or {}
    summary = (
        load_json(folder / "dataset_summary.json")
        or load_json(folder / "vivos_summary.json")
        or {}
    )

    split = count_split(folder / "iden_split.txt")
    trials = count_trials(folder / "veri_test.txt")
    args = hparams.get("args", {}) if isinstance(hparams, dict) else {}

    n_speakers = (
        summary.get("n_speakers")
        or hparams.get("n_speakers")
        or sid.get("n_speakers")
    )
    n_utts = (
        summary.get("n_utterances")
        or sum(summary.get("split_counts", {}).values())
        or sum(split.values())
        or None
    )

    return {
        "name": name,
        "folder": str(folder),
        "n_speakers": n_speakers,
        "n_utterances": n_utts,
        "split": summary.get("split_counts") or split,
        "trials": {
            "trials": summary.get("n_trials") or sv.get("n_trials") or trials["trials"],
            "positive": summary.get("n_positive_trials") or trials["positive"],
            "negative": summary.get("n_negative_trials") or trials["negative"],
        },
        "epochs": args.get("epochs") or hparams.get("total_epochs"),
        "batch_size": args.get("batch_size"),
        "lr": args.get("lr"),
        "best_val_acc": best_val_acc(training_log),
        "sid_top1": sid.get("top1_accuracy"),
        "sid_top5": sid.get("top5_accuracy"),
        "sv_eer": sv.get("eer_percent"),
        "sv_threshold": sv.get("threshold"),
        "sv_min_dcf": sv.get("min_dcf"),
    }


def split_text(split: dict[str, int]) -> str:
    return f"{split.get('train', 0)} / {split.get('val', 0)} / {split.get('test', 0)}"


def build_report(vox: dict[str, Any], vivos: dict[str, Any]) -> str:
    rows = []
    for item in (vox, vivos):
        rows.append(
            "| {name} | {spk} | {utt} | {split} | {epochs} | {best} | {top1} | {top5} | {eer} | {dcf} | {thr} |".format(
                name=item["name"],
                spk=item["n_speakers"] or "TODO",
                utt=item["n_utterances"] or "TODO",
                split=split_text(item["split"]),
                epochs=item["epochs"] or "TODO",
                best=fmt(item["best_val_acc"] * 100 if item["best_val_acc"] is not None else None, percent=True),
                top1=fmt(item["sid_top1"], percent=True),
                top5=fmt(item["sid_top5"], percent=True),
                eer=fmt(item["sv_eer"], percent=True),
                dcf=fmt(item["sv_min_dcf"], digits=4),
                thr=fmt(item["sv_threshold"], digits=4),
            )
        )

    return "\n".join([
        "# ECAPA-TDNN Training Dataset Comparison",
        "",
        "Generated from the training artifacts under `training/results/`.",
        "",
        "## Scope",
        "",
        "- Model: ECAPA-TDNN + AAM-Softmax.",
        "- Feature: 80-dim log Mel filterbank with CMN.",
        "- Datasets: VoxCeleb Indian subset and VIVOS.",
        "- MUSAN/RIR are excluded from the main evidence because the noise/RIR datasets are not available in this experiment.",
        "",
        "## Main Results",
        "",
        "| Dataset | Speakers | Utterances | Train / Val / Test | Epochs | Best Val Acc | SID Top-1 | SID Top-5 | SV EER | minDCF | Threshold @ EER |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "## Trial Counts",
        "",
        "| Dataset | Trials | Positive | Negative |",
        "|---|---:|---:|---:|",
        f"| {vox['name']} | {vox['trials']['trials']} | {vox['trials']['positive']} | {vox['trials']['negative']} |",
        f"| {vivos['name']} | {vivos['trials']['trials']} | {vivos['trials']['positive']} | {vivos['trials']['negative']} |",
        "",
        "## Discussion Template",
        "",
        "- Compare whether VIVOS improves Vietnamese-domain speaker recognition or only closed-set SID.",
        "- Explain that EER/minDCF are internal metrics from generated trial pairs, not official VoxCeleb1-O or NIST SRE benchmarks.",
        "- If VIVOS is better, attribute likely causes to language/domain match and possibly smaller/easier speaker set.",
        "- If VoxCeleb is better, attribute likely causes to greater speaker/audio variation despite domain mismatch.",
        "- State that MUSAN/RIR augmentation was implemented in code but excluded from reported evidence due to unavailable auxiliary datasets.",
        "",
        "## Artifact Folders",
        "",
        f"- VoxCeleb Indian: `{vox['folder']}`",
        f"- VIVOS: `{vivos['folder']}`",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voxceleb-dir", default="training/results/voxceleb_indian")
    parser.add_argument("--vivos-dir", default="training/results/vivos")
    parser.add_argument("--out", default="docs/results/training_dataset_comparison.md")
    args = parser.parse_args()

    vox = summarize("VoxCeleb Indian", Path(args.voxceleb_dir))
    vivos = summarize("VIVOS", Path(args.vivos_dir))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_report(vox, vivos), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
