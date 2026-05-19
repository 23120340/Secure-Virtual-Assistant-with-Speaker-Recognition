"""Challenge-response cho tác vụ IMPORTANT.

Mục đích: chống replay attack đơn giản. Ngay cả khi attacker ghi âm câu
"đọc ghi chú của tôi" của user thật, attacker không biết câu challenge
random mà server vừa sinh — nên không thể replay pass cả ASR-match lẫn SV
trên audio thứ hai.

Pipeline:
    Turn 1 (audio_1):
        SID + SV pass → BACKEND không dispatch handler ngay.
        Server sinh `phrase` random (vd "ba bốn năm sáu"), tạo `token`,
        lưu state (token, phrase, user_id, intent, entities, expiry) vào session.
        Trả về `action_type="challenge_required"` để frontend prompt user.

    Turn 2 (audio_2):
        Frontend gửi audio_2 + token tới `/api/assistant/challenge`.
        Server:
            1. Kiểm tra token còn hợp lệ (chưa expire, chưa dùng).
            2. ASR audio_2 → so với `phrase` (fuzzy match — Levenshtein
               ratio sau khi normalize, tolerance ~0.3).
            3. SV audio_2 vs user_id đã claim (re-verify giọng).
            4. Cả 2 pass → dispatch handler đã pending; cả 2 fail → block.

Token format: opaque urlsafe random 24 byte. Session lưu mapping
{token: ChallengeState}. TTL ngắn (90s) để force user phản hồi nhanh.
"""
from __future__ import annotations

import re
import secrets
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Optional


# Phrase pool: Vietnamese từ ngắn dễ phát âm (tránh từ khó / homophone).
# Mỗi challenge ghép 4 từ → ~6^4 = 1296 phrase distinct, đủ entropy chống guess.
_DIGITS_VI = [
    "một", "hai", "ba", "bốn", "năm",
    "sáu", "bảy", "tám", "chín", "không",
]
# Một số từ ngắn khác để mix tránh phrase toàn số (ASR đôi khi nhầm số).
_COLORS_VI = ["đỏ", "xanh", "vàng", "tím", "trắng", "đen"]


def gen_phrase(length: int = 4) -> str:
    """Sinh phrase random length từ pool digit + color (mix giảm chance ASR nhầm)."""
    pool = _DIGITS_VI + _COLORS_VI
    return " ".join(secrets.choice(pool) for _ in range(length))


# ==========================================================================
# Token store (in-memory dict; cho production cần Redis)
# ==========================================================================
@dataclass
class ChallengeState:
    """State của 1 challenge đang chờ user phản hồi."""
    phrase: str
    user_id: str
    user_name: str
    intent: str
    entities: dict
    transcript: str          # transcript gốc (để pass lại router)
    sid_score: float
    sv_score: float
    expires_at: float
    used: bool = False


CHALLENGE_TTL_SEC = 90


def make_token() -> str:
    return secrets.token_urlsafe(18)


# ==========================================================================
# Phrase comparison
# ==========================================================================
def _normalize(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace, NFC unicode."""
    s = unicodedata.normalize("NFC", s.lower())
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _edit_distance(a: list, b: list) -> int:
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j-1], dp[j])
            prev = cur
    return dp[m]


def phrase_match(expected: str, hypothesis: str, tolerance: float = 0.34) -> tuple[bool, float]:
    """So sánh phrase user nói với phrase server sinh.

    Dùng word-level Levenshtein normalize bằng len(expected). Trả về
    (match, distance_ratio). Tolerance default 0.34 cho phép 1/3 từ sai
    (vd 1 trên 4 từ); thiếu/sai 1 từ vẫn coi là pass — ASR tiếng Việt
    có WER ~15-25% trên audio mic không chuyên.
    """
    e = _normalize(expected).split()
    h = _normalize(hypothesis).split()
    if not e:
        return False, 1.0
    dist = _edit_distance(e, h)
    ratio = dist / len(e)
    return ratio <= tolerance, ratio
