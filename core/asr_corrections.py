"""Layer correction cho ASR — chống các lỗi nhận dạng giọng nói thường gặp.

Pipeline cải tiến (mỗi layer optional, fall through):

  Whisper raw transcript
    → Layer 1: regex correction map (data/asr_corrections.json)
        Sửa các mis-recognition cố định (vd "đã phía chú" → "đọc ghi chú")
    → Layer 2: user vocab boost trong initial_prompt
        File `data/asr_user_vocab.txt` (1 phrase/dòng) được append vào
        prompt → Whisper bias toward các từ này.
    → Layer 3: Gemini `correct_transcript` (đã có) — fallback ngữ nghĩa.

Vì sao tách thành module riêng:
- Layer 1 chạy mỗi turn (zero cost) — load từ file, apply regex.
- Layer 2 chỉ chạy 1 lần khi khởi tạo ASR.
- Layer 3 đắt (Gemini call) — caller quyết định bật/tắt.

Bài học từ `data/turn_log.jsonl`:
- "Đã phía chú" thường là ASR mis-recognition của "đọc ghi chú".
- "Nào, ghi chú" có thể là "đọc ghi chú" với prefix nhiễu.
- Whisper tiếng Việt tự nhiên dễ nhầm phụ âm đầu (đ/p/n/g) trong
  command-domain ngắn.

Cách user bổ sung corrections thủ công:
1. Mở `data/turn_log.jsonl` → tìm transcript bị sai.
2. Mở `data/asr_corrections.json` (hoặc tạo mới nếu chưa có).
3. Thêm cặp `{"pattern": "đã phía chú", "replacement": "đọc ghi chú"}`.
4. Hoặc dùng regex: `{"pattern": "\\bnào,?\\s*ghi chú\\b", "replacement": "đọc ghi chú",
                    "regex": true, "ignore_case": true}`.
5. Restart server (file load ở startup) — hoặc gọi reload qua admin endpoint
   trong tương lai.

Cách user bổ sung vocab thủ công:
1. Tạo file `data/asr_user_vocab.txt`.
2. Mỗi dòng = 1 phrase tiếng Việt user thường nói (tên người, lệnh hay dùng).
3. Restart server. Vocab tự inject vào Whisper initial_prompt.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import difflib
import unicodedata

from . import config

_log = logging.getLogger("secva.asr.corr")

_CORRECTIONS_PATH = config.DATA_DIR / "asr_corrections.json"
_VOCAB_PATH       = config.DATA_DIR / "asr_user_vocab.txt"


# ==========================================================================
# Helpers chuẩn hoá cho fuzzy snap
# ==========================================================================
def _strip_tones(s: str) -> str:
    """Bỏ dấu thanh + đ→d (để so khớp bỏ qua lỗi sai dấu của Whisper)."""
    s = s.replace("đ", "d").replace("Đ", "D")
    nfkd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn")


def _fuzzy_key(s: str) -> str:
    """Khoá so khớp fuzzy: thường + bỏ dấu + bỏ dấu câu + gộp khoảng trắng."""
    s = _strip_tones(s.lower())
    s = re.sub(r"[^\w\s]", " ", s)        # bỏ dấu câu
    return re.sub(r"\s+", " ", s).strip()


@dataclass
class _Rule:
    """1 quy tắc sửa. `regex=False` thì pattern là chuỗi literal."""
    pattern: str
    replacement: str
    regex: bool = False
    ignore_case: bool = True
    _compiled: Optional[re.Pattern] = None

    def compile(self):
        if self._compiled is not None:
            return
        flags = re.IGNORECASE if self.ignore_case else 0
        if self.regex:
            self._compiled = re.compile(self.pattern, flags)
        else:
            # Literal match — escape special chars, wrap word boundary nếu
            # pattern là từ thông thường (không có space hoặc punctuation).
            esc = re.escape(self.pattern)
            self._compiled = re.compile(esc, flags)

    def apply(self, text: str) -> str:
        self.compile()
        if self._compiled is None:
            return text
        return self._compiled.sub(self.replacement, text)


# ==========================================================================
# Layer 1: regex correction map
# ==========================================================================
_rules_cache: Optional[list[_Rule]] = None


def _load_rules() -> list[_Rule]:
    """Load + cache rules từ `data/asr_corrections.json`.

    Schema:
        [
          {"pattern": "...", "replacement": "...",
           "regex": false (optional, default false),
           "ignore_case": true (optional, default true)}
        ]
    File thiếu hoặc invalid → trả [] (no-op).
    """
    global _rules_cache
    if _rules_cache is not None:
        return _rules_cache
    if not _CORRECTIONS_PATH.exists():
        _rules_cache = []
        return _rules_cache
    try:
        raw = json.loads(_CORRECTIONS_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        _log.warning("asr_corrections.json invalid (%s) — bỏ qua", e)
        _rules_cache = []
        return _rules_cache
    if not isinstance(raw, list):
        _log.warning("asr_corrections.json phải là JSON array — bỏ qua")
        _rules_cache = []
        return _rules_cache
    rules: list[_Rule] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict) or "pattern" not in item or "replacement" not in item:
            _log.warning("asr_corrections.json[%d] thiếu pattern/replacement — bỏ qua", i)
            continue
        rules.append(_Rule(
            pattern=str(item["pattern"]),
            replacement=str(item["replacement"]),
            regex=bool(item.get("regex", False)),
            ignore_case=bool(item.get("ignore_case", True)),
        ))
    _rules_cache = rules
    if rules:
        _log.info("Loaded %d ASR correction rules từ %s",
                  len(rules), _CORRECTIONS_PATH.name)
    return rules


# ==========================================================================
# Layer 1.5: fuzzy snap — nắn cả câu về câu lệnh chuẩn gần nhất
# ==========================================================================
# Câu lệnh chuẩn lấy từ `replacement` của rules (loại general_question).
_snap_cache: Optional[list] = None  # list[(canonical, key, n_words)]


def _load_snap_targets() -> list:
    global _snap_cache
    if _snap_cache is not None:
        return _snap_cache
    targets: dict[str, str] = {}
    if _CORRECTIONS_PATH.exists():
        try:
            raw = json.loads(_CORRECTIONS_PATH.read_text(encoding="utf-8"))
        except Exception:
            raw = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            # KHÔNG snap về câu hỏi tự do → tránh nuốt nhầm general_question.
            if item.get("intent") == "general_question":
                continue
            repl = item.get("replacement")
            if not repl:
                continue
            k = _fuzzy_key(str(repl))
            if k:
                targets.setdefault(k, str(repl))
    _snap_cache = [(canon, k, len(k.split())) for k, canon in targets.items()]
    if _snap_cache:
        _log.info("Loaded %d snap targets cho fuzzy ASR correction", len(_snap_cache))
    return _snap_cache


def _fuzzy_snap(text: str) -> str:
    """Nếu cả câu (chuẩn hoá) gần khớp một câu lệnh chuẩn với độ tương đồng
    >= ngưỡng → trả câu lệnh đó. Ngược lại giữ nguyên. Câu dài (vd câu hỏi tự
    do) bị bỏ qua bằng giới hạn số từ."""
    if not config.ASR_FUZZY_ENABLED:
        return text
    key = _fuzzy_key(text)
    if not key:
        return text
    nwords = len(key.split())
    if nwords > config.ASR_FUZZY_MAX_WORDS:
        return text
    best_canon, best_ratio = None, 0.0
    sm = difflib.SequenceMatcher(autojunk=False)
    sm.set_seq2(key)
    for canon, tkey, twords in _load_snap_targets():
        # CHỈ snap khi BẰNG số từ → chỉ sửa lỗi chính tả từng từ, KHÔNG thêm/bớt
        # từ (tránh nắn "đọc ghi chú của tôi" thành cụm ngắn "ghi chú của tôi").
        if twords != nwords:
            continue
        sm.set_seq1(tkey)
        if sm.quick_ratio() <= best_ratio:
            continue  # cận trên rẻ; không thể tốt hơn best hiện tại
        r = sm.ratio()
        if r > best_ratio:
            best_ratio, best_canon = r, canon
    if best_canon is not None and best_ratio >= config.ASR_FUZZY_THRESHOLD:
        # Snap target là dạng câu lệnh chuẩn, KHÔNG kèm dấu câu cuối. Giữ lại
        # dấu câu cuối của input để không nuốt mất (vd "đọc ghi chú." không bị
        # cụt thành "đọc ghi chú").
        m = re.search(r"\s*([.!?…]+)\s*$", text)
        snapped = best_canon
        if m and not best_canon.endswith((".", "!", "?", "…")):
            snapped = best_canon + m.group(1)
        if _fuzzy_key(best_canon) != key:
            _log.info("fuzzy snap %r → %r (ratio=%.2f)", text, snapped, best_ratio)
        return snapped
    return text


def reload_rules():
    """Force re-load rules + snap targets từ disk (vd: sau khi admin sửa file)."""
    global _rules_cache, _snap_cache
    _rules_cache = None
    _snap_cache = None
    _load_rules()


def apply_corrections(text: str) -> str:
    """Layer 1: áp dụng tất cả regex rules theo thứ tự khai báo, rồi Layer 1.5
    fuzzy snap về câu lệnh chuẩn gần nhất.

    Idempotent với text không match rule nào — trả nguyên text.
    Order matters: rule trước có thể tạo input cho rule sau (chain).
    """
    if not text:
        return text
    out = text
    for rule in _load_rules():
        try:
            out = rule.apply(out)
        except re.error as e:
            _log.warning("rule regex error %r: %s — bỏ qua", rule.pattern, e)
            continue
    return _fuzzy_snap(out)


# ==========================================================================
# Layer 2: user vocab → Whisper initial_prompt boost
# ==========================================================================
def load_user_vocab() -> list[str]:
    """Đọc phrases từ `data/asr_user_vocab.txt`. 1 dòng = 1 phrase.

    Dòng comment (#) hoặc rỗng bị bỏ qua. Trả [] nếu file thiếu.
    """
    if not _VOCAB_PATH.exists():
        return []
    out = []
    try:
        for line in _VOCAB_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(line)
    except Exception as e:
        _log.warning("asr_user_vocab.txt read error: %s — bỏ qua", e)
        return []
    if out:
        _log.info("Loaded %d user vocab phrases từ %s",
                  len(out), _VOCAB_PATH.name)
    return out


# ==========================================================================
# Layer 3 (existing): Gemini correction — gọi `core.asr.correct_transcript`.
# Không duplicate ở đây. Doc đã có ở module asr.
# ==========================================================================


# ==========================================================================
# Helper for tests / debug
# ==========================================================================
def _set_rules_for_test(rules: list[_Rule]):
    """Hook cho unit test — override rules in-memory mà không sửa file."""
    global _rules_cache
    _rules_cache = list(rules)
