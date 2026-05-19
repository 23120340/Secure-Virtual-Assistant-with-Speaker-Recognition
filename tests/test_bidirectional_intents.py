"""Test 3 intent bidirectional mới: add_note / add_schedule / add_contact.

Covers:
- Handler ghi đúng vào preferences (notes / schedule / contacts).
- Cap số entries (FIFO khi vượt).
- add_contact idempotent: cùng name → update email.
- Rule-based NLU classify đúng intent + extract entity tối thiểu.
"""
import numpy as np
import pytest

from core import handlers
from core.intents import INTENTS, AuthLevel
from core.nlu import RuleBasedNLU


# ----- Handler tests ------------------------------------------------------
def _user(tmp_db, prefs=None):
    fake_emb = np.zeros(192, dtype=np.float32)
    tmp_db.add_user("u1", "Test", fake_emb, preferences=prefs or {})
    return tmp_db.get_user("u1")


def test_add_note_appends_to_preferences(tmp_db):
    user = _user(tmp_db)
    r = handlers.handle_add_note({"content": "họp 9h"}, user, db=tmp_db)
    assert "Đã thêm" in r
    refreshed = tmp_db.get_user("u1")
    assert refreshed["preferences"]["notes"] == ["họp 9h"]


def test_add_note_keeps_existing_notes(tmp_db):
    user = _user(tmp_db, prefs={"notes": ["cũ 1", "cũ 2"]})
    handlers.handle_add_note({"content": "mới"}, user, db=tmp_db)
    assert tmp_db.get_user("u1")["preferences"]["notes"] == ["cũ 1", "cũ 2", "mới"]


def test_add_note_caps_at_50(tmp_db):
    # Tạo 51 ghi chú liên tiếp → giữ 50 mới nhất.
    user = _user(tmp_db, prefs={"notes": [f"n{i}" for i in range(50)]})
    handlers.handle_add_note({"content": "n50"}, user, db=tmp_db)
    notes = tmp_db.get_user("u1")["preferences"]["notes"]
    assert len(notes) == 50
    assert notes[-1] == "n50"
    assert "n0" not in notes  # đã bị drop FIFO


def test_add_note_empty_content_rejected(tmp_db):
    user = _user(tmp_db)
    r = handlers.handle_add_note({"content": ""}, user, db=tmp_db)
    assert "chưa nghe" in r.lower() or "nội dung" in r.lower()
    assert tmp_db.get_user("u1")["preferences"].get("notes", []) == []


def test_add_schedule_with_time(tmp_db):
    user = _user(tmp_db)
    handlers.handle_add_schedule(
        {"content": "họp nhóm", "time": "9 giờ"}, user, db=tmp_db)
    entry = tmp_db.get_user("u1")["preferences"]["schedule"][0]
    assert "9 giờ" in entry and "họp nhóm" in entry


def test_add_schedule_without_time(tmp_db):
    user = _user(tmp_db)
    handlers.handle_add_schedule({"content": "đi siêu thị"}, user, db=tmp_db)
    assert tmp_db.get_user("u1")["preferences"]["schedule"] == ["đi siêu thị"]


def test_add_contact_validates_email(tmp_db):
    user = _user(tmp_db)
    r = handlers.handle_add_contact(
        {"name": "Tuấn", "email": "not-an-email"}, user, db=tmp_db)
    assert "không hợp lệ" in r
    assert "contacts" not in tmp_db.get_user("u1")["preferences"]


def test_add_contact_appends(tmp_db):
    user = _user(tmp_db)
    handlers.handle_add_contact(
        {"name": "Tuấn", "email": "tuan@example.com"}, user, db=tmp_db)
    cts = tmp_db.get_user("u1")["preferences"]["contacts"]
    assert len(cts) == 1
    assert cts[0]["name"] == "Tuấn"
    assert cts[0]["email"] == "tuan@example.com"


def test_add_contact_idempotent_updates_email(tmp_db):
    """Thêm cùng name 2 lần với email khác → email được update, không duplicate row."""
    user = _user(tmp_db)
    handlers.handle_add_contact(
        {"name": "Tuấn", "email": "old@example.com"}, user, db=tmp_db)
    user = tmp_db.get_user("u1")
    handlers.handle_add_contact(
        {"name": "Tuấn", "email": "new@example.com"}, user, db=tmp_db)
    cts = tmp_db.get_user("u1")["preferences"]["contacts"]
    assert len(cts) == 1
    assert cts[0]["email"] == "new@example.com"


def test_add_contact_missing_field_rejected(tmp_db):
    user = _user(tmp_db)
    r = handlers.handle_add_contact({"name": "", "email": "x@y.com"}, user, db=tmp_db)
    assert "tên" in r.lower() or "email" in r.lower()
    r2 = handlers.handle_add_contact({"name": "X"}, user, db=tmp_db)
    assert "email" in r2.lower()


def test_handlers_registered_in_map():
    for name in ("add_note", "add_schedule", "add_contact"):
        assert name in handlers.HANDLERS
        assert callable(handlers.HANDLERS[name])


def test_intents_registered_as_important():
    for name in ("add_note", "add_schedule", "add_contact"):
        assert name in INTENTS
        assert INTENTS[name]["level"] == AuthLevel.IMPORTANT


# ----- Rule-based NLU classify ---------------------------------------------
def test_rule_based_classifies_add_note():
    nlu = RuleBasedNLU()
    res = nlu.parse("thêm ghi chú mua sữa")
    assert res["intent"] == "add_note"
    assert "mua sữa" in (res["entities"].get("content") or "")


def test_rule_based_classifies_add_schedule_with_time():
    nlu = RuleBasedNLU()
    res = nlu.parse("thêm lịch họp 9 giờ")
    assert res["intent"] == "add_schedule"
    assert "9 giờ" in (res["entities"].get("time") or "")


def test_rule_based_classifies_add_contact():
    nlu = RuleBasedNLU()
    res = nlu.parse("thêm liên hệ Tuấn email tuan@example.com")
    assert res["intent"] == "add_contact"
    # NOTE: name extraction từ rule-based khá brittle với tiếng Việt — chỉ
    # assert email được extract đúng vì email regex chắc chắn.
    assert res["entities"].get("email") == "tuan@example.com"


def test_rule_based_does_not_confuse_read_with_add():
    """Bug guard: "đọc ghi chú" không được match thành add_note (vì cùng có 'ghi chú')."""
    nlu = RuleBasedNLU()
    assert nlu.parse("đọc ghi chú của tôi")["intent"] == "read_notes"
    assert nlu.parse("xóa ghi chú")["intent"] == "delete_data"
    assert nlu.parse("thêm ghi chú mới")["intent"] == "add_note"
