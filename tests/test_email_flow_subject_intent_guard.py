"""Test email flow KHÔNG bị kill ở step subject/body khi text giống intent khác.

Bug history (user feedback 2026-05-20):
    User soạn email, ở step "subject" nói "đọc ghi chú" → trước đó intent guard
    classify câu này là `read_notes` → exit email_flow → user mất context.
    Subject là free-form text, KHÔNG được parse như intent.
"""
import pytest

# Lazy import — `_continue_email_flow_if_active` cần Flask context.


class _FakeNlu:
    """NLU giả: bất kỳ text nào về intent xác định trừ unknown/general/send_email
    để test intent guard có tôn trọng step không."""
    def __init__(self, intent="read_notes"):
        self._intent = intent

    def parse(self, text):
        return {"intent": self._intent, "entities": {}}


@pytest.fixture
def flask_app(tmp_path, monkeypatch):
    """Minimal Flask app + session để test helper."""
    import os
    os.environ.setdefault("FLASK_SECRET", "x" * 64)
    from flask import Flask, session

    # Stub DB — không cần model thật cho test này.
    class _StubDB:
        def get_user(self, uid):
            return {"user_id": uid, "name": "Tester",
                    "preferences": {"contacts": [{"name": "Lan", "email": "lan@ex.com"}]}}

    app = Flask(__name__)
    app.secret_key = "x" * 64
    app.config["db"] = _StubDB()
    return app


def test_subject_step_does_not_exit_on_intent_match(flask_app):
    """Ở step subject, user nói câu giống lệnh "đọc ghi chú" → flow KHÔNG kill."""
    from web.app import _continue_email_flow_if_active
    nlu = _FakeNlu(intent="read_notes")
    db = flask_app.config["db"]
    with flask_app.test_request_context():
        from flask import session
        session["email_flow"] = {
            "step": "subject",
            "user_id": "u1",
            "recipient_name": "Lan",
            "recipient_email": "lan@ex.com",
            "subject": "",
            "body": "",
            "started_at": 9999999999,  # không expire
        }
        payload, handled = _continue_email_flow_if_active(
            "đọc ghi chú quan trọng", db, nlu)
        # Flow PHẢI tiếp tục — không trả (None, False) như intent guard cũ.
        assert handled is True, "Flow bị kill ở step subject — guard sai"
        assert payload["intent"] == "email_flow"
        # State phải advance sang step body với subject vừa nhập.
        assert session["email_flow"]["step"] == "body"
        assert session["email_flow"]["subject"] == "đọc ghi chú quan trọng"


def test_body_step_does_not_exit_on_intent_match(flask_app):
    """Step body cũng free-form — text "phát nhạc rock" KHÔNG kill flow."""
    from web.app import _continue_email_flow_if_active
    nlu = _FakeNlu(intent="play_music")
    db = flask_app.config["db"]
    with flask_app.test_request_context():
        from flask import session
        session["email_flow"] = {
            "step": "body",
            "user_id": "u1",
            "recipient_name": "Lan",
            "recipient_email": "lan@ex.com",
            "subject": "Hello",
            "body": "",
            "started_at": 9999999999,
        }
        payload, handled = _continue_email_flow_if_active(
            "tối nay đến nhà tôi nghe nhạc rock nhé", db, nlu)
        assert handled is True
        assert session["email_flow"]["step"] == "confirm"


def test_recipient_step_still_exits_on_other_intent(flask_app):
    """Step recipient VẪN có guard — user nói "phát nhạc" thay vì tên → exit."""
    from web.app import _continue_email_flow_if_active
    nlu = _FakeNlu(intent="play_music")
    db = flask_app.config["db"]
    with flask_app.test_request_context():
        from flask import session
        session["email_flow"] = {
            "step": "recipient",
            "user_id": "u1",
            "recipient_name": "",
            "recipient_email": "",
            "subject": "",
            "body": "",
            "started_at": 9999999999,
        }
        payload, handled = _continue_email_flow_if_active("phát nhạc đi", db, nlu)
        assert handled is False, "Recipient step phải có guard"
        assert "email_flow" not in session


def test_email_step_does_not_exit_on_intent_match(flask_app):
    """Step email (sau khi recipient không tìm thấy) cũng KHÔNG có guard."""
    from web.app import _continue_email_flow_if_active
    nlu = _FakeNlu(intent="read_notes")  # Whisper hay mis-recognize email thành intent
    db = flask_app.config["db"]
    with flask_app.test_request_context():
        from flask import session
        session["email_flow"] = {
            "step": "email",
            "user_id": "u1",
            "recipient_name": "Phong",
            "recipient_email": "",
            "subject": "",
            "body": "",
            "started_at": 9999999999,
        }
        payload, handled = _continue_email_flow_if_active(
            "phong@example.com", db, nlu)
        assert handled is True
        assert session["email_flow"]["step"] == "subject"
        assert session["email_flow"]["recipient_email"] == "phong@example.com"


# ---- Recipient not found prompt ---------------------------------------------
def test_recipient_not_found_offers_both_options():
    """Khi name không match contact + không phải email format → prompt mention
    cả 2 option (đọc lại tên RÕ HƠN, hoặc đọc email trực tiếp)."""
    from core.email_flow import continue_flow, start_flow
    contacts = [{"name": "Lan", "email": "lan@ex.com"}]
    _, state = start_flow("XXX", contacts)  # tên không có trong danh bạ
    assert state["step"] == "email"  # advance sang step email để retry
    # Test continue_flow path khi nhập tên không tồn tại trong step "email":
    msg, new_state, _ = continue_flow("YYY", state, contacts)
    # Message phải mention cả 2 options
    assert any(kw in msg.lower() for kw in ["đọc lại", "đọc rõ", "đọc / nhập",
                                            "đọc tên rõ hơn"])
    assert "email" in msg.lower()


def test_email_step_can_recover_with_new_name():
    """User vào step "email" rồi đọc TÊN khác (rõ hơn) → lookup lại contacts."""
    from core.email_flow import continue_flow
    contacts = [{"name": "Phong", "email": "phong@ex.com"}]
    state = {
        "step": "email", "user_id": "u1",
        "recipient_name": "blah",
        "recipient_email": "", "subject": "", "body": "",
        "started_at": 9999999999,
    }
    msg, new_state, _ = continue_flow("Phong", state, contacts)
    assert new_state["step"] == "subject"
    assert new_state["recipient_email"] == "phong@ex.com"
