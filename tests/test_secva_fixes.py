"""Regression tests cho 5 fix theo feedback người dùng (2026-06-19).

1. ASR/NLU: lệnh IMPORTANT bị đọc sai → trước rơi vào general_question, KHÔNG
   kích hoạt xác thực. Nay parse_with_correction thử sửa cả khi general_question.
2. Challenge: đọc sai cụm từ được đọc lại (challenge_retry), KHÔNG block cứng;
   send_email qua challenge mở luồng soạn (start_email_flow) thay vì gửi thẳng.
4. Email: bước subject/body KHÔNG bị hủy khi nội dung chứa từ giống lệnh
   ("không"/"thôi"/...); chỉ hủy khi nói lệnh hủy dứt khoát.
5. Confirm: nhận từ ngắn "có"/"không" + lỗi ASR của "xác nhận" (Xét nhật, ...).
"""
import numpy as np
import pytest

from core import email_flow as ef


# ==========================================================================
# Issue 5 — interpret_confirmation
# ==========================================================================
@pytest.mark.parametrize("text", [
    "có", "Có.", "Có ạ", "xác nhận", "Xác nhận!", "đồng ý", "ok", "OK",
    "gửi đi", "gửi luôn", "vâng", "đúng rồi", "dạ", "ừ", "được", "chốt",
    # Lỗi ASR thực tế của "xác nhận" (từ data/turn_log.jsonl) — 2 âm:
    "Xét nhật", "Sét nhật", "Sạch nhật", "Sạch nhạc", "Xe nhật", "Sáng nhật",
])
def test_confirm_yes(text):
    assert ef.interpret_confirmation(text) == "yes"


@pytest.mark.parametrize("text", [
    "không", "Không.", "hủy", "hủy bỏ", "thôi", "no", "không gửi",
    "dừng lại", "bỏ qua", "đừng gửi", "không gửi nữa",
])
def test_confirm_no(text):
    assert ef.interpret_confirmation(text) == "no"


@pytest.mark.parametrize("text", [
    # AN TOÀN: câu mơ hồ KHÔNG được tự suy → hỏi lại (tránh gửi nhầm).
    "đọc ghi chú quan trọng", "phát nhạc rock", "tôi mua xe nhật bản mới",
    "hôm nay trời đẹp", "",
    # Tên/kính ngữ/cụm trùng âm KHÔNG được hiểu thành "có" sau khi bỏ dấu:
    "cô Sang", "Cô Dung", "anh Dũng", "đá bóng", "đọc lại nội dung",
    "công ty", "cơ hội tốt", "có thể mai", "Song nhi", "Sang nhé", "Xa nhà",
    "Sài Gòn nhật",
])
def test_confirm_ambiguous_returns_none(text):
    assert ef.interpret_confirmation(text) is None


@pytest.mark.parametrize("text", [
    # NGUY HIỂM nếu hiểu nhầm thành "yes": người đang muốn HỦY → tuyệt đối
    # không được trả "yes" (sẽ gửi email ngoài ý muốn). NO có ưu tiên tuyệt đối.
    "có lẽ không", "dạ thôi", "có chứ không hủy", "thôi không gửi",
])
def test_confirm_negation_never_yes(text):
    assert ef.interpret_confirmation(text) != "yes"


# ==========================================================================
# Issue 4 — cancel theo bước
# ==========================================================================
def _state(step, **kw):
    base = {"step": step, "user_id": "u1", "recipient_name": "Lan",
            "recipient_email": "lan@ex.com", "subject": "S", "body": "",
            "started_at": 9_999_999_999}
    base.update(kw)
    return base


@pytest.mark.parametrize("step,text,next_step", [
    ("body", "Không đến được tối nay, hẹn dịp khác", "confirm"),
    ("subject", "Thôi để mai họp nhé", "body"),
    ("subject", "đọc ghi chú quan trọng", "body"),
    ("body", "dừng xe trước cổng giúp tôi", "confirm"),
])
def test_subject_body_not_cancelled_by_contentish_text(step, text, next_step):
    resp, new_state, done = ef.continue_flow(text, _state(step))
    assert new_state is not None, f"flow bị hủy oan ở step {step}: {text!r}"
    assert new_state["step"] == next_step


@pytest.mark.parametrize("step,text", [
    ("body", "hủy bỏ"), ("subject", "Hủy gửi email"), ("body", "thoát"),
])
def test_subject_body_explicit_cancel_still_works(step, text):
    resp, new_state, done = ef.continue_flow(text, _state(step))
    assert new_state is None
    assert "hủy" in resp.lower()


def test_recipient_step_broad_cancel_still_works():
    resp, new_state, done = ef.continue_flow("không", _state("recipient"))
    assert new_state is None


# ==========================================================================
# Issue 5 — confirm step yes/no qua continue_flow
# ==========================================================================
def test_confirm_step_short_yes_sends():
    resp, new_state, done = ef.continue_flow("có", _state("confirm", body="B"))
    assert done is True


def test_confirm_step_misasr_xacnhan_sends():
    resp, new_state, done = ef.continue_flow("Xét nhật", _state("confirm", body="B"))
    assert done is True


def test_confirm_step_no_cancels():
    resp, new_state, done = ef.continue_flow("không", _state("confirm", body="B"))
    assert new_state is None
    assert done is False


# ==========================================================================
# Issue 1/2 — parse_with_correction thử sửa cả khi general_question
# ==========================================================================
class _FakeNLU:
    """Trả general_question cho câu sai, read_notes cho câu đã sửa."""
    def __init__(self, mapping):
        self._map = mapping

    def parse(self, text):
        return self._map.get(text, {"intent": "general_question",
                                     "entities": {"query": text}})


def test_parse_with_correction_recovers_from_general_question(monkeypatch):
    from core import nlu as nlu_mod
    from core import config
    monkeypatch.setattr(config, "GEMINI_API_KEY", "fake-key")
    # correct_transcript "sửa" câu sai thành câu đúng.
    monkeypatch.setattr("core.asr.correct_transcript",
                        lambda text, force=False: "đọc ghi chú của tôi")
    fake = _FakeNLU({"đọc ghi chú của tôi": {"intent": "read_notes", "entities": {}}})
    final_text, result = nlu_mod.parse_with_correction("đọc ghê chú của tôi", fake)
    assert result["intent"] == "read_notes", "câu lệnh sai phải được sửa & nhận lại"
    assert final_text == "đọc ghi chú của tôi"


def test_parse_with_correction_keeps_real_general_question(monkeypatch):
    from core import nlu as nlu_mod
    from core import config
    monkeypatch.setattr(config, "GEMINI_API_KEY", "fake-key")
    # Sửa không đổi gì (câu hỏi thật) → giữ general_question, không đổi transcript.
    monkeypatch.setattr("core.asr.correct_transcript",
                        lambda text, force=False: text)
    fake = _FakeNLU({})
    final_text, result = nlu_mod.parse_with_correction("thủ đô nước Pháp là gì", fake)
    assert result["intent"] == "general_question"
    assert final_text == "thủ đô nước Pháp là gì"


def test_parse_with_correction_skips_gemini_for_genuine_question(monkeypatch):
    """Câu hỏi thật (có dấu hiệu hỏi / dài) KHÔNG được tốn thêm Gemini call."""
    from core import nlu as nlu_mod
    from core import config
    monkeypatch.setattr(config, "GEMINI_API_KEY", "fake-key")
    calls = {"n": 0}

    def _spy(text, force=False):
        calls["n"] += 1
        return text
    monkeypatch.setattr("core.asr.correct_transcript", _spy)
    fake = _FakeNLU({})
    for q in ["thủ đô nước Pháp là gì", "tại sao trời mưa",
              "bạn nghĩ sao về việc học lập trình mỗi ngày một chút nhé"]:
        nlu_mod.parse_with_correction(q, fake)
    assert calls["n"] == 0, "Không được gọi Gemini-correct cho câu hỏi tự do"


# ==========================================================================
# Issue 2 — complete_challenge: retry + send_email → start_email_flow
# ==========================================================================
class _FakeDB:
    def get_user(self, uid):
        return {"user_id": uid, "name": "Tester", "preferences": {}}


class _FakeSpk:
    def __init__(self, sv_pass=True):
        self.db = _FakeDB()
        self._sv_pass = sv_pass

    def verify(self, audio, uid):
        return (self._sv_pass, 0.9 if self._sv_pass else 0.1)

    def incremental_update_centroid(self, audio, uid, alpha=0.1):
        return (False, None)


def _challenge_state(intent, phrase):
    return {"intent": intent, "entities": {"recipient": "Lan"}, "user_id": "u1",
            "user_name": "Tester", "phrase": phrase, "token": "tok123",
            "sid_score": 0.8, "sv_score": 0.9}


def test_challenge_phrase_mismatch_is_retryable_not_hard_block():
    from core.router import Router
    r = Router(_FakeSpk(sv_pass=True))
    audio = np.zeros(16000, dtype=np.float32)
    state = _challenge_state("read_notes", "ba bốn năm sáu")
    # Đọc sai hẳn cụm từ → phải là challenge_retry, KHÔNG phải block SV.
    res = r.complete_challenge(audio, "chín tám bảy hai", state)
    assert res.action_type == "challenge_retry"
    assert res.sv_passed is not False  # không phải fail giọng
    assert res.action_data.get("token") == "tok123"


def test_challenge_sv_fail_is_hard_block():
    from core.router import Router
    r = Router(_FakeSpk(sv_pass=False))
    audio = np.zeros(16000, dtype=np.float32)
    state = _challenge_state("read_notes", "ba bốn năm sáu")
    res = r.complete_challenge(audio, "ba bốn năm sáu", state)  # phrase đúng
    assert res.blocked is True
    assert res.sv_passed is False
    assert res.action_type != "challenge_retry"


def test_challenge_send_email_opens_compose_flow():
    from core.router import Router
    r = Router(_FakeSpk(sv_pass=True))
    audio = np.zeros(16000, dtype=np.float32)
    state = _challenge_state("send_email", "ba bốn năm sáu")
    res = r.complete_challenge(audio, "ba bốn năm sáu", state)
    assert res.blocked is False
    assert res.action_type == "start_email_flow"
    assert res.action_data.get("recipient") == "Lan"


def test_challenge_read_notes_dispatches_handler():
    from core.router import Router
    r = Router(_FakeSpk(sv_pass=True))
    audio = np.zeros(16000, dtype=np.float32)
    state = _challenge_state("read_notes", "ba bốn năm sáu")
    res = r.complete_challenge(audio, "ba bốn năm sáu", state, extra_context={"db": _FakeDB()})
    assert res.blocked is False
    assert res.action_type is None
    assert "ghi chú" in res.response.lower()


def test_challenge_not_interrupted_by_other_intent():
    """Issue: 'xin chào'/'xem thời tiết' khi đang challenge KHÔNG được nhảy sang
    greet/get_weather — chúng đi qua complete_challenge → đọc lại (retry)."""
    from core.router import Router
    r = Router(_FakeSpk(sv_pass=True))
    audio = np.zeros(16000, dtype=np.float32)
    for utter in ["xin chào", "xem thời tiết hôm nay", "mấy giờ rồi"]:
        state = _challenge_state("read_notes", "ba bốn năm sáu")
        res = r.complete_challenge(audio, utter, state)
        assert res.action_type == "challenge_retry", f"{utter!r} không được phá challenge"


# ==========================================================================
# Issue (mới) — intent nhẹ KHÔNG phá email flow ở step recipient
# ==========================================================================
class _FakeNLU2:
    def __init__(self, intent):
        self._i = intent

    def parse(self, text):
        return {"intent": self._i, "entities": {}}


class _StubDB2:
    def get_user(self, uid):
        return {"user_id": uid, "name": "Tester",
                "preferences": {"contacts": [{"name": "Lan", "email": "lan@ex.com"}]}}


import os as _os
_os.environ.setdefault("FLASK_SECRET", "x" * 64)


def _recipient_state():
    return {"step": "recipient", "user_id": "u1", "recipient_name": "",
            "recipient_email": "", "subject": "", "body": "",
            "started_at": 9_999_999_999}


@pytest.mark.parametrize("intent", ["greet", "get_weather", "get_time", "tell_joke"])
def test_recipient_step_light_intents_do_not_kill_flow(intent):
    from flask import Flask, session
    from web.app import _continue_email_flow_if_active
    app = Flask(__name__)
    app.secret_key = "x" * 64
    with app.test_request_context():
        session["email_flow"] = _recipient_state()
        payload, handled = _continue_email_flow_if_active(
            "xin chào", _StubDB2(), _FakeNLU2(intent))
        assert handled is True, f"intent nhẹ {intent} không được phá flow"
        assert "email_flow" in session


@pytest.mark.parametrize("intent", ["play_music", "read_notes", "delete_data"])
def test_recipient_step_action_intents_still_exit(intent):
    from flask import Flask, session
    from web.app import _continue_email_flow_if_active
    app = Flask(__name__)
    app.secret_key = "x" * 64
    with app.test_request_context():
        session["email_flow"] = _recipient_state()
        payload, handled = _continue_email_flow_if_active(
            "làm gì đó", _StubDB2(), _FakeNLU2(intent))
        assert handled is False, f"lệnh hành động {intent} phải thoát flow"
        assert "email_flow" not in session


# ==========================================================================
# Issue (mới) — voice↔text email: cùng một state machine, đổi mode không gãy
# ==========================================================================
def test_email_flow_mode_agnostic_voice_to_text():
    """Bắt đầu (recipient→subject) như voice rồi 'chuyển text' tiếp tục body→confirm
    → cùng continue_flow, state liền mạch."""
    from core.email_flow import continue_flow
    contacts = [{"name": "Lan", "email": "lan@ex.com"}]
    # voice: recipient
    _, st, _ = continue_flow("Lan", _state("recipient", recipient_name="", recipient_email="", subject=""), contacts)
    assert st["step"] == "subject"
    # voice: subject
    _, st, _ = continue_flow("Báo cáo tuần", st, contacts)
    assert st["step"] == "body"
    # text: body (mode khác, cùng state)
    _, st, _ = continue_flow("Nội dung họp 9h sáng mai", st, contacts)
    assert st["step"] == "confirm"
    # text: confirm bằng "có" (kèm dấu câu)
    _, st2, done = continue_flow("Có.", st, contacts)
    assert done is True


# ==========================================================================
# Issue (mới) — dấu câu cuối không làm sai kiểm tra cancel/confirm
# ==========================================================================
@pytest.mark.parametrize("text,expect", [
    ("không!", "no"), ("Không.", "no"), ("hủy?", "no"), ("Thôi…", "no"),
    ("Có!", "yes"), ("Xác nhận.", "yes"), ("đồng ý!", "yes"), ("Có?", "yes"),
])
def test_confirm_trailing_punctuation(text, expect):
    assert ef.interpret_confirmation(text) == expect


@pytest.mark.parametrize("text", ["không!", "Không.", "hủy?", "Thôi…", "bỏ qua!"])
def test_is_cancel_trailing_punctuation(text):
    assert ef._is_cancel(text) is True


@pytest.mark.parametrize("text", ["hủy bỏ!", "Hủy gửi email.", "thoát…"])
def test_is_explicit_cancel_trailing_punctuation(text):
    assert ef._is_explicit_cancel(text) is True


# ==========================================================================
# Issue (mới) — tách recipient/subject/body ngay từ câu lệnh đầu
# ==========================================================================
_CONTACTS = [{"name": "Nguyễn Trọng Phúc", "email": "phuc@ex.com"}]


def test_start_flow_full_command_goes_to_confirm():
    """'gửi email cho X, tiêu đề Y, nội dung Z' → thẳng tới xác nhận."""
    q, s = ef.start_flow("Nguyễn Trọng Phúc", _CONTACTS,
                         subject="Xin chào", body="nội dung xin chào")
    assert s["step"] == "confirm"
    assert s["subject"] == "Xin chào"
    assert s["body"] == "nội dung xin chào"
    assert "xác nhận" in q.lower()


def test_start_flow_recipient_only_asks_subject():
    q, s = ef.start_flow("Nguyễn Trọng Phúc", _CONTACTS)
    assert s["step"] == "subject"
    assert s["recipient_email"] == "phuc@ex.com"
    assert "chủ đề" in q.lower()


def test_start_flow_recipient_subject_asks_body():
    q, s = ef.start_flow("Nguyễn Trọng Phúc", _CONTACTS, subject="Báo cáo tuần")
    assert s["step"] == "body"
    assert s["subject"] == "Báo cáo tuần"
    assert "nội dung" in q.lower()


def test_start_flow_not_found_keeps_subject_body_for_later():
    """Tên không có trong danh bạ → hỏi email, NHƯNG giữ subject/body đã tách."""
    q, s = ef.start_flow("Người Lạ", _CONTACTS, subject="S", body="B")
    assert s["step"] == "email"
    assert s["subject"] == "S" and s["body"] == "B"
    # Sau khi nhập email → đã đủ → thẳng tới confirm (không hỏi lại subject/body).
    q2, s2, done = ef.continue_flow("nguoilas@ex.com", s, _CONTACTS)
    assert s2["step"] == "confirm"


class _DBWithUsers:
    def __init__(self, users):
        self._users = users

    def find_user_by_name_substring(self, q):
        q = (q or "").lower().strip()
        hits = [u for u in self._users if q and q in u["name"].lower()]
        return hits[0] if len(hits) == 1 else None


def test_email_contacts_resolves_registered_user():
    """Người nhận là user đã đăng ký (không nằm trong contacts cá nhân) vẫn giải
    được email → không phải hỏi lại địa chỉ."""
    from web.app import _email_contacts
    db = _DBWithUsers([{"name": "Nguyễn Trọng Phúc",
                        "preferences": {"email": "phuc@ex.com"}}])
    user = {"preferences": {"contacts": []}}
    contacts = _email_contacts(db, user, "Nguyễn Trọng Phúc")
    assert any(c.get("email") == "phuc@ex.com" for c in contacts)
    # start_flow với danh bạ đã augment → giải được recipient, hỏi chủ đề.
    q, s = ef.start_flow("Nguyễn Trọng Phúc", contacts)
    assert s["step"] == "subject"


def test_email_contacts_ambiguous_or_missing_not_added():
    from web.app import _email_contacts
    db = _DBWithUsers([
        {"name": "Phúc A", "preferences": {"email": "a@ex.com"}},
        {"name": "Phúc B", "preferences": {"email": "b@ex.com"}},
    ])
    user = {"preferences": {"contacts": []}}
    # "Phúc" khớp 2 user → ambiguous → KHÔNG thêm (find_user trả None).
    contacts = _email_contacts(db, user, "Phúc")
    assert contacts == []
