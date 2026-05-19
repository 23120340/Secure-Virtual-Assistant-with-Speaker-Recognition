"""Export unlabeled NLU training candidates from JSONL to CSV.

Usage:
    python scripts/export_nlu_dataset.py
    python scripts/export_nlu_dataset.py --out data/nlu_training_candidates.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import config


DEFAULT_INPUT = config.DATA_DIR / "nlu_training_candidates.jsonl"
DEFAULT_OUTPUT = config.DATA_DIR / "nlu_training_candidates.csv"

FIELDS = [
    "ts",
    "source",
    "request_id",
    "utterance",
    "predicted_intent",
    "predicted_entities",
    "expected_intent",
    "expected_entities",
    "label_status",
    "blocked",
]


def iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc


def export_csv(input_path: Path, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in iter_jsonl(input_path) or []:
            out = {field: row.get(field, "") for field in FIELDS}
            out["predicted_entities"] = json.dumps(
                out.get("predicted_entities") or {},
                ensure_ascii=False,
                sort_keys=True,
            )
            out["expected_entities"] = json.dumps(
                out.get("expected_entities") or {},
                ensure_ascii=False,
                sort_keys=True,
            )
            writer.writerow(out)
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT),
                        help="Path JSONL input")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT),
                        help="Path CSV output")
    args = parser.parse_args()

    count = export_csv(Path(args.input), Path(args.out))
    print(f"Exported {count} rows to {args.out}")


if __name__ == "__main__":
    main()

