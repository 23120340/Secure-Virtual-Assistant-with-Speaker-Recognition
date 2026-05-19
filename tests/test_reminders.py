"""Test reminder parser + list management."""
from datetime import datetime, timedelta, timezone

from core.reminders import (
    parse_vi_time, make_reminder, append_reminder, due_reminders, mark_fired,
    format_reminder_for_speech,
)

_VN = timezone(timedelta(hours=7))


def _fixed_now():
    """Cố định now để test deterministic: 2026-05-19 10:00 VN."""
    return datetime(2026, 5, 19, 10, 0, tzinfo=_VN)


# ---- parse_vi_time ----------------------------------------------------------
def test_parse_relative_minutes():
    d = parse_vi_time("30 phút nữa uống thuốc", now=_fixed_now())
    assert d == _fixed_now() + timedelta(minutes=30)


def test_parse_relative_hours():
    d = parse_vi_time("2 giờ nữa kiểm tra", now=_fixed_now())
    assert d == _fixed_now() + timedelta(hours=2)


def test_parse_relative_tiếng():
    """'tiếng' synonym của 'giờ'."""
    d = parse_vi_time("3 tiếng nữa", now=_fixed_now())
    assert d == _fixed_now() + timedelta(hours=3)


def test_parse_relative_days():
    d = parse_vi_time("2 ngày nữa", now=_fixed_now())
    assert d == _fixed_now() + timedelta(days=2)


def test_parse_explicit_hour_colon():
    d = parse_vi_time("14:30 hôm nay xem phim", now=_fixed_now())
    assert d.hour == 14 and d.minute == 30
    assert d.date() == _fixed_now().date()


def test_parse_explicit_hour_h_notation():
    d = parse_vi_time("9h30 sáng mai", now=_fixed_now())
    assert d.hour == 9 and d.minute == 30
    assert d.date() == (_fixed_now() + timedelta(days=1)).date()


def test_parse_explicit_hour_giờ():
    d = parse_vi_time("8 giờ tối nay", now=_fixed_now())
    # "tối" → period_adj = 12 → 8 + 12 = 20
    assert d.hour == 20 and d.minute == 0


def test_parse_passed_today_rolls_to_tomorrow():
    """Nếu nói '8 giờ' mà now=10h sáng cùng ngày → assume mai 8 giờ."""
    d = parse_vi_time("8 giờ đi tập", now=_fixed_now())
    assert d.date() == (_fixed_now() + timedelta(days=1)).date()
    assert d.hour == 8


def test_parse_unknown_returns_none():
    assert parse_vi_time("blabla random", now=_fixed_now()) is None
    assert parse_vi_time("", now=_fixed_now()) is None
    assert parse_vi_time(None, now=_fixed_now()) is None


# ---- reminder list ---------------------------------------------------------
def test_make_reminder_with_valid_time():
    r = make_reminder("uống thuốc", "30 phút nữa")
    assert r["content"] == "uống thuốc"
    assert r["when_iso"] is not None
    assert r["fired"] is False
    assert len(r["id"]) == 12  # uuid hex prefix


def test_make_reminder_unparseable_time_keeps_raw():
    r = make_reminder("nhớ mua sữa", "khi nào đó")
    assert r["when_iso"] is None
    assert r["when_raw"] == "khi nào đó"


def test_append_caps_at_100():
    lst = [{"id": str(i), "content": f"r{i}", "fired": False} for i in range(100)]
    new = append_reminder(lst, make_reminder("new", "1 giờ nữa"))
    assert len(new) == 100  # cap, dropped oldest
    assert new[-1]["content"] == "new"
    assert "0" not in [r.get("id") for r in new]  # FIFO drop


def test_due_reminders_filters_correctly():
    past = (_fixed_now() - timedelta(minutes=5)).isoformat()
    future = (_fixed_now() + timedelta(minutes=5)).isoformat()
    lst = [
        {"id": "a", "content": "past", "when_iso": past, "fired": False},
        {"id": "b", "content": "future", "when_iso": future, "fired": False},
        {"id": "c", "content": "past-fired", "when_iso": past, "fired": True},
        {"id": "d", "content": "no-time", "when_iso": None, "fired": False},
    ]
    due = due_reminders(lst, now=_fixed_now())
    ids = [r["id"] for r in due]
    assert ids == ["a"]


def test_mark_fired():
    lst = [
        {"id": "a", "fired": False},
        {"id": "b", "fired": False},
    ]
    new = mark_fired(lst, ["a"])
    assert new[0]["fired"] is True
    assert new[1]["fired"] is False
    # Không mutate input
    assert lst[0]["fired"] is False


def test_format_for_speech():
    r = make_reminder("uống thuốc", "30 phút nữa")
    s = format_reminder_for_speech(r)
    assert "uống thuốc" in s
    assert "Nhắc" in s
