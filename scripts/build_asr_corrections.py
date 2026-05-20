"""Analyze turn_log.jsonl → suggest ASR correction rules + auto-extract vocab.

Workflow:
    1. Đọc `data/turn_log.jsonl` (mỗi line 1 turn event).
    2. Tách 2 nhóm:
       - **Successful turns** (intent != unknown, !blocked): transcript đã được
         Whisper/Gemini hiểu đúng → các phrase trong đó nên vào `asr_user_vocab.txt`.
       - **Unknown/blocked turns**: transcript có thể là ASR error → in ra để
         user xem manual, KHÔNG auto-add rule (avoid false-positive).
    3. In suggestions + counts. User quyết định:
       - Copy phrase candidates vào `data/asr_user_vocab.txt`.
       - Tự viết rule `{pattern, replacement}` cho `data/asr_corrections.json`
         dựa trên transcript bị sai.

Cách dùng:
    python scripts/build_asr_corrections.py
    python scripts/build_asr_corrections.py --write-vocab  # tự ghi vocab file
    python scripts/build_asr_corrections.py --source web_voice  # chỉ voice turns

KHÔNG auto-write correction rules vì cần human judgment (Whisper-error vs
user-intent-different). Vocab thì an toàn auto-write — chỉ cần expose vocab
trong prompt là Whisper bias toward → không phá behavior hiện có.
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

# Đảm bảo print Unicode tiếng Việt không crash trên Windows.
sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

ROOT = Path(__file__).resolve().parent.parent
TURN_LOG = ROOT / "data" / "turn_log.jsonl"
VOCAB_OUT = ROOT / "data" / "asr_user_vocab.txt"
CORR_OUT = ROOT / "data" / "asr_corrections.json"


def _load_events(path: Path) -> list[dict]:
    """Load JSONL, skip malformed lines."""
    if not path.exists():
        print(f"⚠ Không tìm thấy {path} — chạy server + dùng giọng nói trước.")
        return []
    events = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def _norm(s: str) -> str:
    """Lowercase + NFC normalize + strip punctuation cho dedup."""
    s = unicodedata.normalize("NFC", s.lower())
    return "".join(c for c in s if not unicodedata.category(c).startswith("P")).strip()


def _is_success(ev: dict) -> bool:
    """Turn được coi là 'understood' nếu intent xác định + không bị block."""
    return (
        ev.get("intent") not in (None, "", "unknown", "general_question", "greet")
        and not ev.get("blocked")
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default="",
                   help="Lọc theo source (vd 'web_voice'). Bỏ trống = tất cả.")
    p.add_argument("--min-count", type=int, default=2,
                   help="Phrase phải xuất hiện ≥ N lần để được suggest (default 2)")
    p.add_argument("--write-vocab", action="store_true",
                   help="Tự ghi suggestions vào data/asr_user_vocab.txt (append).")
    p.add_argument("--max-suggest", type=int, default=50,
                   help="Tối đa phrase suggest mỗi nhóm (default 50)")
    args = p.parse_args()

    events = _load_events(TURN_LOG)
    if args.source:
        events = [e for e in events if e.get("source") == args.source]
    if not events:
        print("Không có event nào để phân tích.")
        return 1

    print(f"✓ Loaded {len(events)} events từ {TURN_LOG.relative_to(ROOT)}")
    print(f"  Source filter: {args.source or '(tất cả)'}")
    print()

    success_evs = [e for e in events if _is_success(e)]
    unknown_evs = [
        e for e in events
        if e.get("intent") in ("unknown", "general_question", "greet", None, "")
    ]
    blocked_evs = [e for e in events if e.get("blocked")]

    print("📊 Stats:")
    print(f"  Successful (intent OK)    : {len(success_evs)}")
    print(f"  Unknown intent            : {len(unknown_evs)}")
    print(f"  Blocked (SV/auth fail)    : {len(blocked_evs)}")
    print()

    # ───── 1. Vocab candidates từ successful transcripts ─────
    print("🟢 Vocab candidates (từ successful transcripts):")
    print(f"   ↳ phrase xuất hiện ≥ {args.min_count} lần, top {args.max_suggest}.")
    print(f"   ↳ paste những phrase user-specific (tên người, từ chuyên ngành)")
    print(f"     vào {VOCAB_OUT.relative_to(ROOT)} để Whisper bias.")
    print()
    phrase_counter: Counter = Counter()
    for ev in success_evs:
        t = ev.get("transcript", "")
        if not t:
            continue
        norm = _norm(t)
        if not norm:
            continue
        phrase_counter[t.strip()] += 1
    suggested_vocab = []
    for phrase, count in phrase_counter.most_common(args.max_suggest):
        if count < args.min_count:
            break
        print(f"  [{count:3d}×] {phrase!r}")
        suggested_vocab.append(phrase)
    if not suggested_vocab:
        print("  (không có phrase nào đủ frequency — cần thêm voice turn data)")
    print()

    # ───── 2. Sai-transcript candidates (cần human review) ─────
    print("🔴 Suspect ASR errors (intent mơ hồ — cần review):")
    print(f"   ↳ Đây có thể là ASR mis-recognition. Review thủ công xem")
    print(f"     user thực sự muốn nói gì, rồi thêm rule vào")
    print(f"     {CORR_OUT.relative_to(ROOT)} với format:")
    print(f'       {{"pattern": "transcript_bị_sai", "replacement": "câu_đúng"}}')
    print()
    for ev in unknown_evs[-args.max_suggest:]:
        t = ev.get("transcript", "").strip()
        if not t:
            continue
        print(f"  - {t!r}")
    if not unknown_evs:
        print("  (không có turn unknown — ASR + NLU đang khớp tốt)")
    print()

    # ───── 3. Optional: auto-write vocab ─────
    if args.write_vocab and suggested_vocab:
        # Merge với vocab cũ — không duplicate.
        existing: set[str] = set()
        if VOCAB_OUT.exists():
            for line in VOCAB_OUT.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    existing.add(line)
        new_phrases = [p for p in suggested_vocab if p not in existing]
        if new_phrases:
            is_new_file = not VOCAB_OUT.exists() or VOCAB_OUT.stat().st_size == 0
            VOCAB_OUT.parent.mkdir(parents=True, exist_ok=True)
            with VOCAB_OUT.open("a", encoding="utf-8") as f:
                if is_new_file:
                    f.write("# ASR user vocab — 1 phrase/line. Comment bằng #.\n")
                    f.write("# Auto-generated bởi scripts/build_asr_corrections.py.\n\n")
                for p in new_phrases:
                    f.write(p + "\n")
            print(f"✓ Đã append {len(new_phrases)} phrase mới vào "
                  f"{VOCAB_OUT.relative_to(ROOT)}")
            print("  Restart server để Whisper load vocab mới vào initial_prompt.")
        else:
            print(f"✓ Tất cả phrase đã có trong {VOCAB_OUT.relative_to(ROOT)}.")
    elif not args.write_vocab:
        print("💡 Chạy lại với --write-vocab để tự append vocab vào file.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
