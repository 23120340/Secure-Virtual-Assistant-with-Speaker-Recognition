"""Reminder utilities — parse thời gian tiếng Việt + quản lý queue reminder.

Mục đích: chuyển free-form text từ user ("9 giờ sáng mai", "30 phút nữa",
"ngày 25 lúc 15h") sang datetime tuyệt đối (ISO format) để lưu vào
`preferences.reminders`.

Không phụ thuộc dateparser/parsedatetime (dep nặng + behavior khó kiểm soát) —
chỉ regex các pattern phổ biến. Parse fail → trả None, caller lưu raw text.

Format reminder entry:
    {
        "id":        "<uuid hex>",
        "content":   "<nội dung>",
        "when_iso":  "2026-05-19T15:00:00+07:00" hoặc None nếu parse fail,
        "when_raw":  "<text gốc user nói>",
        "fired":     False,
        "created_at": "<ISO UTC>",
    }
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

# Việt Nam timezone — naive parsing assume local là VN.
_VN_TZ = timezone(timedelta(hours=7))


def _vn_now() -> datetime:
    return datetime.now(_VN_TZ)


def _replace_time(base: datetime, hour: int, minute: int = 0) -> datetime:
    return base.replace(hour=hour, minute=minute, second=0, microsecond=0)


# ==========================================================================
# Vietnamese time parsing — best-effort, regex-based
# ==========================================================================
_RELATIVE_OFFSETS = [
    # (regex, unit_seconds)
    (re.compile(r"(\d+)\s*phút\s*nữa", re.IGNORECASE), 60),
    (re.compile(r"(\d+)\s*giờ\s*nữa", re.IGNORECASE), 3600),
    (re.compile(r"(\d+)\s*tiếng\s*nữa", re.IGNORECASE), 3600),
    (re.compile(r"(\d+)\s*ngày\s*nữa", re.IGNORECASE), 86400),
    (re.compile(r"(\d+)\s*tuần\s*nữa", re.IGNORECASE), 86400 * 7),
]

_DAY_KEYWORDS = {
    "hôm nay":  0,
    "hôm sau":  1,
    "ngày mai": 1,
    "ngày kia": 2,
    "mai":      1,
}

_TIME_OF_DAY = {
    # Default hour cho từ khóa khi user chỉ nói "sáng/chiều" mà không có giờ cụ thể.
    "sáng":  8,
    "trưa":  12,
    "chiều": 15,
    "tối":   20,
    "đêm":   22,
}


def _parse_explicit_hour(text: str) -> Optional[tuple[int, int]]:
    """Tìm pattern '15 giờ 30 phút', '9h30', '14:00', '8 giờ', '9 giờ sáng'...

    Trả về (hour, minute) hoặc None. KHÔNG xử lý "sáng/chiều" → caller làm.
    """
    # 14:00 / 9:30
    m = re.search(r"\b(\d{1,2})\s*:\s*(\d{2})\b", text)
    if m:
        h, mn = int(m.group(1)), int(m.group(2))
        if 0 <= h <= 23 and 0 <= mn <= 59:
            return h, mn
    # 9h30 / 14h00 / 8h
    m = re.search(r"\b(\d{1,2})\s*h\s*(\d{0,2})\b", text, re.IGNORECASE)
    if m:
        h = int(m.group(1))
        mn = int(m.group(2)) if m.group(2) else 0
        if 0 <= h <= 23 and 0 <= mn <= 59:
            return h, mn
    # "9 giờ 30 phút" / "9 giờ 30" / "9 giờ"
    m = re.search(r"\b(\d{1,2})\s*giờ(?:\s*(\d{1,2}))?\s*(?:phút)?",
                  text, re.IGNORECASE)
    if m:
        h = int(m.group(1))
        mn = int(m.group(2)) if m.group(2) else 0
        if 0 <= h <= 23 and 0 <= mn <= 59:
            return h, mn
    return None


def parse_vi_time(text: str, *, now: datetime | None = None) -> Optional[datetime]:
    """Parse text tiếng Việt → datetime tuyệt đối ở VN timezone.

    Support pattern:
    - "30 phút nữa", "2 giờ nữa", "3 ngày nữa"
    - "9 giờ sáng mai", "14h chiều nay", "10h"
    - "hôm nay 8 giờ tối", "ngày mai 9 giờ"
    - "ngày 25 lúc 15h"

    Trả None nếu không nhận ra → caller lưu raw + cảnh báo user.

    Test với --doctest-modules để verify.
    """
    if not text or not text.strip():
        return None
    t = text.lower().strip()
    base = (now or _vn_now()).astimezone(_VN_TZ)

    # 1) Offset tương đối: "X phút/giờ/ngày/tuần nữa"
    for rx, unit_s in _RELATIVE_OFFSETS:
        m = rx.search(t)
        if m:
            n = int(m.group(1))
            return base + timedelta(seconds=n * unit_s)

    # 2) Day keyword (hôm nay / ngày mai / ngày kia) + time
    day_offset = None
    for kw, off in _DAY_KEYWORDS.items():
        if kw in t:
            day_offset = off
            break

    # 3) Explicit time "9 giờ 30" / "14:00" / "9h"
    hm = _parse_explicit_hour(t)

    # 4) "sáng / chiều / tối / đêm" → adjust period
    period_adj = 0
    if hm and hm[0] < 12:
        if "chiều" in t or "tối" in t:
            period_adj = 12  # 3 chiều → 15h
        elif "đêm" in t:
            period_adj = 12 if hm[0] < 6 else 0
    period_default = None
    for kw, h_default in _TIME_OF_DAY.items():
        if kw in t and hm is None:
            period_default = h_default
            break

    if hm is None and period_default is None and day_offset is None:
        return None

    target = base
    if day_offset is not None:
        target = target + timedelta(days=day_offset)

    if hm is not None:
        h = hm[0] + period_adj
        if h >= 24: h -= 24
        target = _replace_time(target, h, hm[1])
    elif period_default is not None:
        target = _replace_time(target, period_default, 0)

    # Nếu chỉ nói "9 giờ" không kèm ngày, và giờ đó đã qua hôm nay → assume mai.
    if day_offset is None and hm is not None:
        if target <= base:
            target = target + timedelta(days=1)

    return target


# ==========================================================================
# Reminder list management
# ==========================================================================
_MAX_REMINDERS = 100


def make_reminder(content: str, when_text: str) -> dict:
    """Build 1 reminder entry từ content + when_text. Parse best-effort."""
    when_dt = parse_vi_time(when_text)
    return {
        "id":         uuid.uuid4().hex[:12],
        "content":    content.strip(),
        "when_iso":   when_dt.isoformat() if when_dt else None,
        "when_raw":   when_text.strip(),
        "fired":      False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def append_reminder(reminders: list, item: dict) -> list:
    """Append + cap FIFO. Caller chịu trách nhiệm persist."""
    new_list = list(reminders or [])
    new_list.append(item)
    if len(new_list) > _MAX_REMINDERS:
        new_list = new_list[-_MAX_REMINDERS:]
    return new_list


def due_reminders(reminders: list, *, now: datetime | None = None) -> list:
    """Lấy reminder đã đến hạn (when_iso ≤ now) và chưa fired."""
    if not reminders:
        return []
    now_dt = (now or _vn_now()).astimezone(_VN_TZ)
    due = []
    for r in reminders:
        if r.get("fired"):
            continue
        iso = r.get("when_iso")
        if not iso:
            continue  # raw text only — không tự fire, user phải check tay
        try:
            wd = datetime.fromisoformat(iso)
        except ValueError:
            continue
        if wd.astimezone(_VN_TZ) <= now_dt:
            due.append(r)
    return due


def mark_fired(reminders: list, ids: list) -> list:
    """Mark reminders với id trong `ids` là fired=True."""
    id_set = set(ids or [])
    out = []
    for r in (reminders or []):
        if r.get("id") in id_set:
            r = dict(r)
            r["fired"] = True
        out.append(r)
    return out


def format_reminder_for_speech(item: dict) -> str:
    """Format reminder thành câu nói tự nhiên cho TTS."""
    content = item.get("content", "")
    iso = item.get("when_iso")
    if iso:
        try:
            dt = datetime.fromisoformat(iso).astimezone(_VN_TZ)
            when_str = dt.strftime("%H:%M ngày %d/%m")
            return f"Nhắc: {content}, lúc {when_str}."
        except ValueError:
            pass
    raw = item.get("when_raw", "")
    return f"Nhắc: {content} ({raw})." if raw else f"Nhắc: {content}."
