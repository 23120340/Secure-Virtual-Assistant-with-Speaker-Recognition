"""Test Calendar integration framework — không gọi Google API thật.

Mock `requests.get` để verify endpoint + params + parse logic. Test
`handle_show_schedule` fallback gracefully khi calendar disabled / fail.
"""
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from core import config, calendar_api, handlers


# ---- _normalize_event ------------------------------------------------------
def test_normalize_event_with_datetime():
    raw = {
        "id": "ev1",
        "summary": "Họp design review",
        "start": {"dateTime": "2026-05-19T14:00:00+07:00"},
        "end":   {"dateTime": "2026-05-19T15:00:00+07:00"},
        "location": "Phòng họp A",
        "htmlLink": "https://calendar.google.com/event?eid=ev1",
    }
    out = calendar_api._normalize_event(raw)
    assert out["id"] == "ev1"
    assert out["summary"] == "Họp design review"
    assert out["start"] == "2026-05-19T14:00:00+07:00"
    assert out["all_day"] is False
    assert out["location"] == "Phòng họp A"


def test_normalize_event_all_day():
    raw = {
        "id": "ev2",
        "summary": "Sinh nhật Lan",
        "start": {"date": "2026-05-20"},
        "end":   {"date": "2026-05-21"},
    }
    out = calendar_api._normalize_event(raw)
    assert out["all_day"] is True
    assert out["start"] == "2026-05-20"


def test_normalize_event_missing_fields_defaults_apply():
    out = calendar_api._normalize_event({})
    assert out["summary"] == "(không tiêu đề)"
    assert out["start"] == ""
    assert out["location"] == ""


# ---- format_events_for_speech ---------------------------------------------
def test_format_empty_events():
    assert calendar_api.format_events_for_speech([]) == ""


def test_format_events_basic():
    events = [{
        "summary": "Họp 9h",
        "start":   "2026-05-19T09:00:00+07:00",
        "all_day": False,
        "location": "",
    }]
    s = calendar_api.format_events_for_speech(events)
    assert "Họp 9h" in s
    assert "09:00" in s and "19/05" in s


def test_format_events_with_location():
    events = [{
        "summary": "Coffee chat",
        "start":   "2026-05-19T10:00:00+07:00",
        "all_day": False,
        "location": "Highlands",
    }]
    s = calendar_api.format_events_for_speech(events)
    assert "tại Highlands" in s


def test_format_all_day_event():
    events = [{
        "summary": "Nghỉ lễ",
        "start":   "2026-05-19",
        "all_day": True,
        "location": "",
    }]
    s = calendar_api.format_events_for_speech(events)
    assert "cả ngày" in s.lower() or "2026-05-19" in s


# ---- list_events (mock) ----------------------------------------------------
def test_list_events_calls_correct_endpoint():
    fake_resp = MagicMock()
    fake_resp.json.return_value = {"items": []}
    fake_resp.raise_for_status.return_value = None
    with patch("core.calendar_api._req.get", return_value=fake_resp) as g:
        calendar_api.list_events("fake-token", max_results=5)
    assert g.called
    args, kwargs = g.call_args
    assert "/calendar/v3/calendars/primary/events" in args[0]
    assert kwargs["headers"]["Authorization"] == "Bearer fake-token"
    assert kwargs["params"]["maxResults"] == 5
    assert kwargs["params"]["singleEvents"] == "true"


def test_list_events_returns_normalized_list():
    fake_resp = MagicMock()
    fake_resp.json.return_value = {"items": [
        {"id": "a", "summary": "ev a",
         "start": {"dateTime": "2026-05-19T09:00:00Z"},
         "end":   {"dateTime": "2026-05-19T10:00:00Z"}},
        {"id": "b", "summary": "ev b",
         "start": {"date": "2026-05-20"},
         "end":   {"date": "2026-05-21"}},
    ]}
    fake_resp.raise_for_status.return_value = None
    with patch("core.calendar_api._req.get", return_value=fake_resp):
        out = calendar_api.list_events("fake-token")
    assert len(out) == 2
    assert out[0]["id"] == "a"
    assert out[1]["all_day"] is True


# ---- handle_show_schedule integration with Calendar ----------------------
def test_show_schedule_no_calendar_when_disabled(tmp_db, monkeypatch):
    """ENABLE_CALENDAR_INTEGRATION=False → KHÔNG gọi calendar_api dù user có token."""
    monkeypatch.setattr(config, "ENABLE_CALENDAR_INTEGRATION", False)
    tmp_db.add_user("u1", "Test", np.zeros(192, dtype=np.float32),
                    preferences={"schedule": ["9h họp", "14h review"]})
    user = tmp_db.get_user("u1")
    # Mock calendar_api để verify KHÔNG gọi.
    with patch("core.calendar_api.list_events") as m:
        r = handlers.handle_show_schedule({}, user, db=tmp_db)
    assert "9h họp" in r
    m.assert_not_called()


def test_show_schedule_falls_back_when_calendar_fails(tmp_db, monkeypatch):
    """Calendar API exception → silently fall back local schedule."""
    monkeypatch.setattr(config, "ENABLE_CALENDAR_INTEGRATION", True)
    tmp_db.add_user("u1", "Test", np.zeros(192, dtype=np.float32),
                    preferences={"schedule": ["9h họp"]})
    tmp_db.save_oauth_token("u1", {
        "access_token": "abc", "refresh_token": "r",
        "gmail_address": "x@y", "expiry": 9999999999.0,  # không expired
    })
    user = tmp_db.get_user("u1")
    with patch("core.calendar_api.list_events",
               side_effect=RuntimeError("scope insufficient")):
        r = handlers.handle_show_schedule({}, user, db=tmp_db)
    assert "9h họp" in r
    # Không leak error message
    assert "scope insufficient" not in r


def test_show_schedule_merges_calendar_and_local(tmp_db, monkeypatch):
    """Cả 2 nguồn có dữ liệu → response hiển thị cả 2 section."""
    monkeypatch.setattr(config, "ENABLE_CALENDAR_INTEGRATION", True)
    tmp_db.add_user("u1", "Test", np.zeros(192, dtype=np.float32),
                    preferences={"schedule": ["9h họp local"]})
    tmp_db.save_oauth_token("u1", {
        "access_token": "abc", "refresh_token": "r",
        "gmail_address": "x@y", "expiry": 9999999999.0,
    })
    user = tmp_db.get_user("u1")
    fake_events = [{
        "summary": "Calendar event A",
        "start":   "2026-05-19T10:00:00+07:00",
        "all_day": False, "location": "",
    }]
    with patch("core.calendar_api.list_events", return_value=fake_events):
        r = handlers.handle_show_schedule({}, user, db=tmp_db)
    assert "Calendar event A" in r
    assert "9h họp local" in r
    assert "Google Calendar" in r
    assert "Lịch cá nhân" in r
