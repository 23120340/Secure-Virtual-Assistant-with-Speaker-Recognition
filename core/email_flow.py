"""State machine cho luồng soạn email nhiều bước."""
import difflib
import re
import time

# "không" được coi như cancel ở bước confirm (bao gồm trong _CANCEL).
# Cancel match phải là exact/startswith để câu như "có không sao đâu" không bị trigger.
_CANCEL = {"thôi", "hủy", "hủy bỏ", "cancel", "dừng", "không gửi",
           "bỏ qua", "thoát", "không", "no"}
_YES    = {"có", "đúng", "ok", "oke", "gửi", "gửi đi", "xác nhận", "yes", "đồng ý", "gửi ngay"}

# Flow timeout — tránh state cookie sống mãi nếu user bỏ giữa chừng
FLOW_TIMEOUT_SECONDS = 600  # 10 phút


def _is_cancel(text: str) -> bool:
    t = text.lower().strip()
    return any(t == w or t.startswith(w + " ") for w in _CANCEL) and len(t) < 40


def _is_affirmative(text: str) -> bool:
    t = text.lower().strip()
    return any(t == w or t.startswith(w + " ") for w in _YES) and len(t) < 40


def _email_preview(state: dict) -> str:
    body_preview = state["body"][:120] + ("..." if len(state["body"]) > 120 else "")
    return (
        f"Xác nhận gửi email:\n"
        f"  Đến   : {state['recipient_name']} ({state['recipient_email']})\n"
        f"  Chủ đề: {state['subject']}\n"
        f"  Nội dung: {body_preview}\n"
        'Nói "có" để gửi, "không" để hủy.'
    )


def _looks_like_email(text: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", text.strip()))


def _find_contact(query: str, contacts: list) -> dict | None:
    """Tìm liên hệ theo tên — ưu tiên fuzzy match unique.

    Chống nhầm contact: trước đây dùng 2-chiều substring (q in name OR name in q)
    nên "anh tuan" sẽ match "anh" — nguy hiểm trong context gửi email.
    Hiện tại:
      1. Exact match (case-insensitive) → return.
      2. difflib.get_close_matches (ratio ≥ 0.7) → chỉ nhận khi unique.
      3. Substring 1-chiều (q in name) → chỉ nhận khi unique.
    """
    q = query.lower().strip()
    if not q:
        return None
    names = [c.get("name", "").lower() for c in contacts]

    # 1. Exact
    for c, n in zip(contacts, names):
        if q == n:
            return c

    # 2. Fuzzy unique
    matches = difflib.get_close_matches(q, names, n=2, cutoff=0.7)
    if len(matches) == 1:
        return contacts[names.index(matches[0])]
    if len(matches) > 1:
        return None  # ambiguous

    # 3. Substring 1-chiều unique
    candidates = [c for c, n in zip(contacts, names) if q in n]
    if len(candidates) == 1:
        return candidates[0]
    return None


def start_flow(initial_recipient: str = "", contacts: list = None) -> tuple:
    """Khởi động flow soạn email.

    Trả về (câu_hỏi_đầu_tiên, state_dict).
    state_dict có các key: step, user_id, recipient_name, recipient_email, subject, body.
    """
    contacts = contacts or []
    state = {
        "step": "recipient",
        "user_id": None,
        "recipient_name": "",
        "recipient_email": "",
        "subject": "",
        "body": "",
        "started_at": time.time(),
    }

    if initial_recipient:
        contact = _find_contact(initial_recipient, contacts)
        if contact:
            state.update(step="subject",
                         recipient_name=contact["name"],
                         recipient_email=contact["email"])
            return (f"Gửi cho {contact['name']} ({contact['email']}). "
                    "Chủ đề email là gì?"), state
        if _looks_like_email(initial_recipient):
            state.update(step="subject",
                         recipient_name=initial_recipient,
                         recipient_email=initial_recipient)
            return (f"Đã nhận địa chỉ {initial_recipient}. "
                    "Chủ đề email là gì?"), state
        # Tên nhưng không tìm được → hỏi email HOẶC tên lại
        state.update(step="email", recipient_name=initial_recipient)
        return (f"Không tìm thấy '{initial_recipient}' trong danh bạ. "
                "Bạn có thể đọc lại tên rõ hơn, hoặc đọc / nhập trực tiếp "
                "địa chỉ email người nhận (vd: ten@gmail.com):"), state

    return "Bạn muốn gửi email cho ai?", state


def continue_flow(text: str, state: dict, contacts: list = None) -> tuple:
    """Tiếp tục flow với đầu vào người dùng.

    Trả về (response, new_state, is_done).
    - new_state=None, is_done=False  → đã hủy, xóa session
    - new_state=state, is_done=False → tiếp tục, lưu new_state vào session
    - new_state=state, is_done=True  → đủ dữ liệu, gửi email rồi xóa session
    """
    contacts = contacts or []
    state = dict(state)

    # Timeout — flow state sống quá 10 phút thì coi như stale
    started_at = state.get("started_at", 0)
    if started_at and (time.time() - started_at) > FLOW_TIMEOUT_SECONDS:
        return "Phiên soạn email đã hết hạn. Bạn có thể bắt đầu lại.", None, False

    if _is_cancel(text):
        return "Đã hủy gửi email.", None, False

    step = state["step"]

    if step == "recipient":
        t = text.strip()
        contact = _find_contact(t, contacts)
        if contact:
            state.update(step="subject",
                         recipient_name=contact["name"],
                         recipient_email=contact["email"])
            return (f"Gửi cho {contact['name']} ({contact['email']}). "
                    "Chủ đề email là gì?"), state, False
        if _looks_like_email(t):
            state.update(step="subject", recipient_name=t, recipient_email=t)
            return f"Đã nhận địa chỉ {t}. Chủ đề email là gì?", state, False
        state.update(step="email", recipient_name=t)
        return (f"Không tìm thấy '{t}' trong danh bạ. "
                "Bạn có thể đọc lại tên rõ hơn, hoặc đọc / nhập trực tiếp "
                "địa chỉ email người nhận (vd: ten@gmail.com):"), state, False

    if step == "email":
        t = text.strip()
        if _looks_like_email(t):
            state.update(step="subject", recipient_email=t)
            return f"Đã nhận địa chỉ {t}. Chủ đề email là gì?", state, False
        # User nhập tên mới (không phải email) → thử lookup contacts lại.
        # Trường hợp: ở step "recipient" tên đầu không match, user đọc lại tên
        # rõ hơn ở step "email" → vẫn cần tìm trong danh bạ.
        contact = _find_contact(t, contacts)
        if contact:
            state.update(step="subject",
                         recipient_name=contact["name"],
                         recipient_email=contact["email"])
            return (f"Tìm thấy {contact['name']} ({contact['email']}). "
                    "Chủ đề email là gì?"), state, False
        return ("Không nhận diện được tên hoặc email. "
                "Hãy đọc rõ tên người trong danh bạ, hoặc đọc địa chỉ email "
                "(vd: ten@gmail.com):"), state, False

    if step == "subject":
        subj = text.strip()
        if len(subj) < 2:
            return "Chủ đề quá ngắn. Hãy nói lại chủ đề:", state, False
        state.update(step="body", subject=subj)
        return "Nội dung email là gì?", state, False

    if step == "body":
        body = text.strip()
        if len(body) < 2:
            return "Nội dung quá ngắn. Hãy nói lại nội dung:", state, False
        state["body"] = body
        state["step"] = "confirm"
        return _email_preview(state), state, False

    if step == "confirm":
        # Xác nhận trước — câu như "có không sao đâu" sẽ pass affirmative
        if _is_affirmative(text):
            return "", state, True
        # Cancel ("không" đã nằm trong _CANCEL với exact/startswith match)
        if _is_cancel(text):
            return "Đã hủy gửi email.", None, False
        return 'Mình chưa hiểu. Nói "có" để gửi hoặc "không" để hủy.', state, False

    return "Có lỗi trong quá trình soạn email.", None, False
