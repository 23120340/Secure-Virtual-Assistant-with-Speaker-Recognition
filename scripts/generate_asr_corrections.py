#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sinh tự động bộ ASR correction cho TỪNG chức năng (intent).

Mục tiêu: ~N biến thể lỗi nhận dạng giọng nói (Whisper tiếng Việt) cho mỗi
intent, mỗi biến thể map ngược về CÂU GỐC của chính nó (giữ nguyên entity như
genre / tên người), giúp ASR layer-1 nắn câu lỗi về câu chuẩn để NLU phân loại
đúng — kể cả khi Gemini hết quota và rớt xuống rule-based.

Cách sinh biến thể (mô phỏng lỗi Whisper VN thực tế):
  - Rớt dấu thanh toàn câu / từng từ (lỗi phổ biến nhất).
  - Nhầm phụ âm đầu: đ↔d↔gi↔r, ph↔f↔p, ch↔tr, s↔x, nh↔n↔l, kh↔k↔h, th↔t...
  - Nhầm nguyên âm / âm cuối: ê↔e, ô↔o, ơ↔o, ư↔u, â↔a, ă↔a, ng↔n (cuối)...
  - Tổ hợp nhiều lỗi (seeded để tái lập được).

Rule sinh ra dùng regex NEO TOÀN CHUỖI (^...$, ignore_case) → chỉ kích hoạt khi
cả transcript đúng là biến thể lỗi đó, tránh false-positive khi chạy chuỗi rule.

Chạy:
    python -m scripts.generate_asr_corrections --per-intent 200
    python -m scripts.generate_asr_corrections --per-intent 200 --dry-run
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.intents import INTENTS  # noqa: E402
from core import config  # noqa: E402

CORRECTIONS_PATH = config.DATA_DIR / "asr_corrections.json"

# Intent không phải lệnh hành động/không có ví dụ → bỏ qua.
SKIP_INTENTS = {"unknown", "email_flow"}

# --------------------------------------------------------------------------
# Biến đổi ngữ âm tiếng Việt
# --------------------------------------------------------------------------
# Phụ âm đầu dài phải xét TRƯỚC phụ âm đơn (ngh > ng > n, ...).
_INITIALS_SORTED = ["ngh", "ng", "nh", "gi", "ph", "ch", "tr", "th", "kh",
                    "gh", "qu", "đ", "d", "r", "s", "x", "c", "k", "g",
                    "n", "l", "v", "b", "p", "h", "t", "f"]

# Mỗi phụ âm đầu → list các phụ âm Whisper hay nhầm thành.
INITIAL_CONFUSIONS = {
    "đ":   ["d"],
    "d":   ["đ", "gi", "r"],
    "gi":  ["d", "r", "z"],
    "r":   ["d", "g", "gi"],
    "ph":  ["f", "p"],
    "f":   ["ph"],
    "ch":  ["tr"],
    "tr":  ["ch"],
    "s":   ["x"],
    "x":   ["s"],
    "kh":  ["k", "h", "c"],
    "nh":  ["n", "l"],
    "ng":  ["n"],
    "ngh": ["ng", "n"],
    "th":  ["t"],
    "gh":  ["g"],
    "g":   ["gh"],
    "c":   ["k"],
    "k":   ["c"],
    "qu":  ["q", "kw"],
    "tr":  ["ch", "t"],
    "l":   ["n"],
    "n":   ["l"],
    "v":   ["d", "b"],
    "b":   ["p"],
    "h":   ["kh"],
}

# Nhầm nguyên âm / vần (substring trong 1 từ).
VOWEL_CONFUSIONS = [
    ("iê", "i"), ("yê", "i"), ("uô", "u"), ("ươ", "ư"),
    ("ê", "e"), ("ô", "o"), ("ơ", "o"), ("ư", "u"),
    ("â", "a"), ("ă", "a"), ("oo", "o"),
    ("ay", "ai"), ("ây", "ai"), ("ao", "au"),
    ("uy", "ui"), ("oa", "a"), ("ưa", "ua"), ("uô", "ô"),
    ("ươ", "ơ"), ("ia", "iê"), ("iu", "iêu"), ("ê", "ơ"),
]

# Nhầm âm cuối (cuối từ).
FINAL_CONFUSIONS = [("ng", "n"), ("nh", "n"), ("ch", "t"), ("c", "t"),
                    ("t", "c"), ("c", "k"), ("n", "ng"),  # n↔ng phương ngữ
                    ("nh", "ng")]


def strip_tones(s: str) -> str:
    """Bỏ toàn bộ dấu thanh + đ→d."""
    s = s.replace("đ", "d").replace("Đ", "D")
    nfkd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn")


def _confuse_initial(word: str) -> list[str]:
    low = word.lower()
    out = []
    for init in _INITIALS_SORTED:
        if low.startswith(init) and init in INITIAL_CONFUSIONS:
            for repl in INITIAL_CONFUSIONS[init]:
                out.append(repl + word[len(init):])
            break  # chỉ nhận phụ âm đầu khớp dài nhất
    return out


def _confuse_vowel(word: str) -> list[str]:
    out = []
    for a, b in VOWEL_CONFUSIONS:
        if a in word:
            out.append(word.replace(a, b, 1))
    return out


def _confuse_final(word: str) -> list[str]:
    out = []
    base = strip_tones(word)
    for a, b in FINAL_CONFUSIONS:
        if base.endswith(a):
            out.append(word[: len(word) - len(a)] + b)
    return out


def _key(s: str) -> str:
    """Khoá so khớp CHÍNH XÁC để dedup: thường + gộp khoảng trắng (GIỮ dấu).

    Quan trọng: giữ dấu để biến thể 'rớt dấu hoàn toàn' (lỗi Whisper phổ biến
    nhất) KHÔNG bị coi là trùng câu gốc có dấu → vẫn được sinh ra.
    """
    return re.sub(r"\s+", " ", s.lower()).strip()


def gen_variants(phrase: str, rng: random.Random, target: int) -> list[str]:
    """Sinh tối đa `target` biến thể lỗi (distinct, giữ dấu) của `phrase`."""
    words = phrase.split()
    seen: set[str] = set()
    out: list[str] = []
    base_key = _key(phrase)

    def add(cand: str):
        cand = re.sub(r"\s+", " ", cand).strip()
        key = _key(cand)
        if not key or key == base_key or key in seen:
            return
        seen.add(key)
        out.append(cand)

    # 1) Rớt dấu toàn câu.
    add(strip_tones(phrase))
    # 2) Rớt dấu từng từ.
    for i in range(len(words)):
        ws = list(words)
        ws[i] = strip_tones(ws[i])
        add(" ".join(ws))
    # 3) Nhầm phụ âm đầu / nguyên âm / âm cuối — đổi từng từ một.
    for i, w in enumerate(words):
        for variant in (_confuse_initial(w) + _confuse_vowel(w) + _confuse_final(w)):
            ws = list(words)
            ws[i] = variant
            add(" ".join(ws))
    # 4) Rớt dấu toàn câu + 1 lỗi phụ âm.
    stripped_words = strip_tones(phrase).split()
    for i, w in enumerate(stripped_words):
        for variant in _confuse_initial(w):
            ws = list(stripped_words)
            ws[i] = variant
            add(" ".join(ws))

    # 5) Tổ hợp ngẫu nhiên (seeded) nhiều lỗi để đủ số lượng.
    attempts = 0
    pool = words
    while len(out) < target and attempts < target * 40:
        attempts += 1
        ws = list(pool)
        n_edits = rng.randint(1, max(1, len(ws)))
        idxs = rng.sample(range(len(ws)), k=min(n_edits, len(ws)))
        for i in idxs:
            choices = (_confuse_initial(ws[i]) + _confuse_vowel(ws[i])
                       + _confuse_final(ws[i]) + [strip_tones(ws[i])])
            if choices:
                ws[i] = rng.choice(choices)
        if rng.random() < 0.5:  # đôi khi rớt dấu thêm
            ws = strip_tones(" ".join(ws)).split()
        add(" ".join(ws))

    return out[:target]


# --------------------------------------------------------------------------
# Khai thác từ FILE LOG
# --------------------------------------------------------------------------
LOG_FILES = ["turn_log.jsonl", "nlu_training_candidates.jsonl"]

# Câu lỗi ASR có thật trong log + dạng đúng tương ứng (suy ra thủ công, chắc
# chắn). Map trực tiếp garbled → clean để layer-1 nắn ngay.
CURATED_GARBLED = [
    ("nãy giờ rồi",   "mấy giờ rồi",       "get_time"),
    ("mấy giời rồi",  "mấy giờ rồi",       "get_time"),
    ("kẻ chuyện cười", "kể chuyện cười đi", "tell_joke"),
    ("kể chuyện cừ",  "kể chuyện cười đi", "tell_joke"),
    ("pháp nhạc đi",  "phát nhạc đi",      "play_music"),
    ("pháp nhạc",     "phát nhạc",         "play_music"),
    ("phác nhạc",     "phát nhạc",         "play_music"),
    ("pát nhạc",      "phát nhạc",         "play_music"),
    ("pát nhạc đi",   "phát nhạc đi",      "play_music"),
    ("đọc ký chú",    "đọc ghi chú",       "read_notes"),
    ("đọc kí chú",    "đọc ghi chú",       "read_notes"),
    ("đã phía chú",   "đọc ghi chú",       "read_notes"),
    ("khi chú",       "đọc ghi chú",       "read_notes"),
    ("khi chủ",       "đọc ghi chú",       "read_notes"),
    ("đọc xối dư",    "kiểm tra số dư",    "check_balance"),
    ("đọc xôi dư",    "kiểm tra số dư",    "check_balance"),
    ("số dử",         "số dư",             "check_balance"),
    ("đặc lực",       "đặt lịch",          "add_schedule"),
    ("đặt lịt",       "đặt lịch",          "add_schedule"),
    ("đặt lệch",      "đặt lịch",          "add_schedule"),
    ("gửi meo",       "gửi mail",          "send_email"),
    ("gửi mêu",       "gửi mail",          "send_email"),
    ("soạn meo",      "soạn mail",         "send_email"),
    ("phát nhạt",     "phát nhạc",         "play_music"),
    ("bậc nhạc",      "bật nhạc",          "play_music"),
    ("kê chuyện cười", "kể chuyện cười đi", "tell_joke"),
    ("thời tiếc",     "thời tiết",         "get_weather"),
    ("mấy giờ rồ",    "mấy giờ rồi",       "get_time"),
]


# Cụm 2+ từ nhưng quá chung chung → bỏ để tránh sửa nhầm trong câu hỏi tự do.
SKIP_TRIGGERS = {"việc gì"}


def collect_trigger_phrases() -> dict:
    """Cụm lệnh ĐẶC TRƯNG (>=2 từ) cho từng intent — nguồn sinh biến thể dạng
    SUBSTRING. Lấy từ RuleBasedNLU.KEYWORD_MAP (nguồn duy nhất, luôn đồng bộ).

    Chỉ giữ cụm >=2 từ: cụm 1 từ ("mưa", "nắng", "hi") quá ngắn, dễ khớp nhầm
    bên trong từ/câu khác khi match substring.
    """
    from core.nlu import RuleBasedNLU
    out: dict = {}
    for intent, kws in RuleBasedNLU.KEYWORD_MAP:
        if intent in SKIP_INTENTS:
            continue
        phrases = [k for k in kws
                   if len(k.split()) >= 2 and k.lower() not in SKIP_TRIGGERS]
        if phrases:
            out[intent] = phrases
    return out


def build_rules(per_intent: int, seed: int = 42):
    rng = random.Random(seed)
    triggers = collect_trigger_phrases()
    # Không sinh biến thể trùng KHÍT (giữ dấu) một cụm lệnh chuẩn bất kỳ.
    valid_keys = {_key(p) for ps in triggers.values() for p in ps}

    rules: list = []
    global_seen: set = set()

    def emit(variant: str, replacement: str, intent: str) -> bool:
        key = _key(variant)
        # Chỉ nhận cụm >=2 từ (pattern đủ đặc trưng để match substring an toàn).
        if not key or len(key.split()) < 2:
            return False
        if key in valid_keys or key in global_seen:
            return False
        global_seen.add(key)
        # SUBSTRING có ranh giới từ (\b): sửa đúng cụm lỗi BÊN TRONG câu, giữ
        # nguyên phần còn lại (vd "đặt nịch 5 giờ đi học" → "đặt lịch 5 giờ đi học").
        rules.append({
            "pattern": r"\b" + re.escape(variant) + r"\b",
            "replacement": replacement,
            "regex": True,
            "ignore_case": True,
            "intent": intent,  # metadata để audit; ASR layer bỏ qua key thừa
        })
        return True

    stats = {}

    # ---- 1) Biến thể của CỤM LỆNH (KEYWORD_MAP) — substring ----
    for intent, phrases in triggers.items():
        pools = [gen_variants(p, rng, per_intent) for p in phrases]
        produced = 0
        col = 0
        max_cols = max((len(p) for p in pools), default=0)
        while produced < per_intent and col < max_cols:
            for phrase, pool in zip(phrases, pools):
                if col < len(pool) and emit(pool[col], phrase, intent):
                    produced += 1
                    if produced >= per_intent:
                        break
            col += 1
        stats[intent] = produced

    # ---- 2) Map garbled→clean lấy trực tiếp từ log (lỗi ASR có thật) ----
    curated = 0
    for garbled, clean, intent in CURATED_GARBLED:
        if emit(garbled, clean, intent):
            curated += 1

    return rules, stats, curated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-intent", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    existing = []
    if CORRECTIONS_PATH.exists():
        existing = json.loads(CORRECTIONS_PATH.read_text(encoding="utf-8"))
    existing_patterns = {r.get("pattern") for r in existing}

    generated, stats, curated = build_rules(args.per_intent, args.seed)
    new_rules = [r for r in generated if r["pattern"] not in existing_patterns]

    print("[1] Biến thể CỤM LỆNH (substring \\b...\\b) / intent:")
    for k, v in stats.items():
        flag = "" if v >= args.per_intent else "  (< mục tiêu — ít/ngắn cụm lệnh)"
        print(f"    {k:<16} {v}{flag}")
    print(f"\n[2] Map garbled→clean từ log (substring): {curated} rule")
    print(f"\nĐang có: {len(existing)} rule | Sinh mới: {len(new_rules)} rule "
          f"| Tổng sau merge: {len(existing) + len(new_rules)}")

    if args.dry_run:
        print("\n[dry-run] Không ghi file. Ví dụ 5 rule sinh ra:")
        for r in new_rules[:5]:
            print("  ", json.dumps(r, ensure_ascii=False))
        return

    # Rule cụm-lệnh (substring) đặt TRƯỚC rule legacy: cụm lỗi được nắn sớm,
    # rule legacy phía sau xử lý các trường hợp còn lại.
    merged = new_rules + existing
    CORRECTIONS_PATH.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nĐã ghi {len(merged)} rule vào {CORRECTIONS_PATH}")


if __name__ == "__main__":
    main()
