"""Add ASR correction rule vào data/asr_corrections.json.

Interactive:
    python scripts/add_asr_correction.py

Non-interactive:
    python scripts/add_asr_correction.py --wrong "đã phía chú" --right "đọc ghi chú"
    python scripts/add_asr_correction.py --wrong "\\bnào,?\\s*ghi chú\\b" --right "đọc ghi chú" --regex

Script chỉ thêm/cập nhật rule, không restart server. Sau khi sửa file, restart
web app để ASR load lại rules.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    sys.stdin.reconfigure(encoding="utf-8")   # type: ignore[attr-defined]
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
CORRECTIONS_PATH = ROOT / "data" / "asr_corrections.json"


def _load_rules(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SystemExit(f"File JSON lỗi: {path} ({e})")
    if not isinstance(raw, list):
        raise SystemExit(f"{path} phải là JSON array.")
    return raw


def _save_rules(path: Path, rules: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(rules, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value or default


def _ask_bool(prompt: str, default: bool = False) -> bool:
    default_text = "y" if default else "n"
    value = _ask(f"{prompt} (y/n)", default_text).lower()
    return value in ("y", "yes", "c", "co", "có")


def _validate_regex(pattern: str) -> None:
    try:
        re.compile(pattern)
    except re.error as e:
        raise SystemExit(f"Regex không hợp lệ: {e}")


def add_or_update_rule(*, wrong: str, right: str, regex: bool = False,
                       ignore_case: bool = True, yes: bool = False,
                       path: Path = CORRECTIONS_PATH) -> str:
    wrong = wrong.strip()
    right = right.strip()
    if not wrong:
        raise SystemExit("Thiếu câu sai/pattern.")
    if not right:
        raise SystemExit("Thiếu câu đúng/replacement.")
    if regex:
        _validate_regex(wrong)

    rules = _load_rules(path)
    new_rule = {
        "pattern": wrong,
        "replacement": right,
        "regex": bool(regex),
        "ignore_case": bool(ignore_case),
    }

    for i, rule in enumerate(rules):
        if not isinstance(rule, dict):
            continue
        same_pattern = rule.get("pattern") == wrong
        same_regex_mode = bool(rule.get("regex", False)) == bool(regex)
        if same_pattern and same_regex_mode:
            if not yes:
                print("Rule đã tồn tại:")
                print(json.dumps(rule, ensure_ascii=False, indent=2))
                if not _ask_bool("Cập nhật replacement/ignore_case?", default=True):
                    return "unchanged"
            rules[i] = new_rule
            _save_rules(path, rules)
            return "updated"

    rules.append(new_rule)
    _save_rules(path, rules)
    return "added"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wrong", help="Câu ASR nhận sai hoặc regex pattern.")
    p.add_argument("--right", help="Câu đúng để thay thế.")
    p.add_argument("--regex", action="store_true",
                   help="Coi --wrong là regex pattern.")
    p.add_argument("--case-sensitive", action="store_true",
                   help="Phân biệt hoa/thường khi match.")
    p.add_argument("-y", "--yes", action="store_true",
                   help="Tự cập nhật nếu rule đã tồn tại.")
    args = p.parse_args()

    wrong = args.wrong
    right = args.right
    regex = args.regex
    ignore_case = not args.case_sensitive

    if not wrong:
        print("Thêm rule sửa chính tả ASR")
        print("Ví dụ: câu sai = 'đã phía chú', câu đúng = 'đọc ghi chú'")
        wrong = _ask("Câu ASR nhận sai")
    if not right:
        right = _ask("Câu đúng")
    if args.wrong is None and not args.regex:
        regex = _ask_bool("Dùng regex cho câu sai?", default=False)
    if args.wrong is None and not args.case_sensitive:
        ignore_case = _ask_bool("Không phân biệt hoa/thường?", default=True)

    result = add_or_update_rule(
        wrong=wrong,
        right=right,
        regex=regex,
        ignore_case=ignore_case,
        yes=args.yes,
    )
    rel = CORRECTIONS_PATH.relative_to(ROOT)
    if result == "added":
        print(f"Đã thêm rule vào {rel}.")
    elif result == "updated":
        print(f"Đã cập nhật rule trong {rel}.")
    else:
        print("Không thay đổi.")
    print("Restart web server để ASR load rule mới.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
