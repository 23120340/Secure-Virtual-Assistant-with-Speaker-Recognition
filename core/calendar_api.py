"""Google Calendar REST API client — đọc events từ user's primary calendar.

Reuse OAuth access token cùng tài khoản đã link Gmail (scope `gmail.send +
calendar.events.readonly`). Khác `core/gmail_api.py`:
- Endpoint: `https://www.googleapis.com/calendar/v3/calendars/<calendarId>/events`
- Method: GET (read-only)
- Time filter: `timeMin`, `timeMax` (RFC 3339) — lấy events trong cửa sổ
  thời gian cụ thể (vd hôm nay).

Khung này chỉ wire phần fetch + parse. Caller (vd `handle_show_schedule`)
tự quyết format trả về user. Khi `config.ENABLE_CALENDAR_INTEGRATION=false`,
endpoint này KHÔNG được gọi (handler check trước).

Retry + circuit breaker chung breaker với Google OAuth (`google-oauth`) để tránh
hammer khi Google API quotas/down — share rate limit fail-fast pattern.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests as _req

from .resilience import retry, CircuitBreaker

_log = logging.getLogger("secva.calendar")

# Reuse breaker name `google-oauth` để 1 fail (vd token expired) share trip
# state với các call OAuth khác → tránh thrash.
_BREAKER = CircuitBreaker("google-oauth", fail_threshold=5, reset_after=60)

_API_ROOT = "https://www.googleapis.com/calendar/v3"

# Default fetch: 10 events trong cửa sổ [now, now+24h]. Caller có thể override.
_DEFAULT_MAX_RESULTS = 10


@_BREAKER
@retry(max_attempts=3, backoff_base=0.5,
       retriable=(_req.ConnectionError, _req.Timeout))
def list_events(access_token: str, *,
                calendar_id: str = "primary",
                time_min: Optional[datetime] = None,
                time_max: Optional[datetime] = None,
                max_results: int = _DEFAULT_MAX_RESULTS) -> list[dict]:
    """Fetch events từ Google Calendar API.

    Args:
        access_token: OAuth access token có scope calendar.events.readonly.
        calendar_id: 'primary' = lịch chính của user. Có thể truyền calendar ID
            khác (vd "abc@group.calendar.google.com") nếu user có nhiều lịch.
        time_min, time_max: cửa sổ thời gian (datetime aware tz). Default
            [now, now+24h] tính bằng UTC.
        max_results: số event tối đa (default 10, Google cap 2500).

    Returns:
        List event dict với schema rút gọn (xem `_normalize_event`):
            {
                "id":      "...",
                "summary": "Tên event",
                "start":   ISO datetime hoặc all-day date string,
                "end":     ISO datetime hoặc all-day date string,
                "location": "...",
                "url":     "https://calendar.google.com/...",
                "all_day": bool,
            }

    Raises:
        requests.HTTPError: 401 (token revoked/expired), 403 (scope thiếu),
            429 (rate limit). Caller handle.
        CircuitOpen: breaker đang open — caller fail-fast.
    """
    now = datetime.now(timezone.utc)
    if time_min is None:
        time_min = now
    if time_max is None:
        time_max = now + timedelta(days=1)

    params = {
        "timeMin":      time_min.isoformat().replace("+00:00", "Z"),
        "timeMax":      time_max.isoformat().replace("+00:00", "Z"),
        "maxResults":   max_results,
        "singleEvents": "true",     # expand recurring → individual instances
        "orderBy":      "startTime",
    }
    r = _req.get(
        f"{_API_ROOT}/calendars/{calendar_id}/events",
        headers={"Authorization": f"Bearer {access_token}"},
        params=params,
        timeout=10,
    )
    r.raise_for_status()
    data = r.json()
    items = data.get("items", []) or []
    return [_normalize_event(e) for e in items]


def _normalize_event(raw: dict) -> dict:
    """Rút gọn Google Calendar event dict → schema dễ render cho user.

    Google trả nhiều field thừa (creator, organizer, attendees, recurrence...).
    Chỉ giữ những gì handler cần để format câu nói.
    """
    start_raw = raw.get("start") or {}
    end_raw = raw.get("end") or {}
    # All-day event chỉ có "date" (YYYY-MM-DD); event giờ cụ thể có "dateTime"
    # (RFC 3339). Phân biệt để format khác nhau.
    start_dt = start_raw.get("dateTime") or start_raw.get("date") or ""
    end_dt   = end_raw.get("dateTime")   or end_raw.get("date")   or ""
    all_day  = "dateTime" not in start_raw and "date" in start_raw
    return {
        "id":       raw.get("id", ""),
        "summary":  raw.get("summary", "(không tiêu đề)"),
        "start":    start_dt,
        "end":      end_dt,
        "location": raw.get("location", ""),
        "url":      raw.get("htmlLink", ""),
        "all_day":  all_day,
    }


def format_events_for_speech(events: list[dict], *,
                             tz_offset_hours: int = 7) -> str:
    """Format danh sách event thành text dễ đọc cho TTS / chat bubble.

    Convert UTC dateTime → giờ địa phương (Việt Nam default +7).
    """
    if not events:
        return ""
    tz = timezone(timedelta(hours=tz_offset_hours))
    lines = []
    for e in events:
        title = e["summary"]
        start = e["start"]
        if e["all_day"]:
            when = f"cả ngày {start}"
        else:
            try:
                dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
                local = dt.astimezone(tz)
                when = local.strftime("%H:%M %d/%m")
            except (ValueError, AttributeError):
                when = start
        loc = f" tại {e['location']}" if e.get("location") else ""
        lines.append(f"{when}: {title}{loc}")
    return "; ".join(lines)
