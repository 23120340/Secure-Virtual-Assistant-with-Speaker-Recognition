"""State machine cho luồng soạn email nhiều bước."""
import re

_CANCEL = {"thôi", "hủy", "hủy bỏ", "cancel", "dừng", "không gửi", "bỏ qua", "thoát"}
_YES    = {"có", "đúng", "ok", "oke", "gửi", "gửi đi", "xác nhận", "yes", "đồng ý", "gửi ngay"}


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
    """Tìm liên hệ theo tên, không phân biệt hoa/thường, khớp một phần."""
    q = query.lower().strip()
    if not q:
        return None
    for c in contacts:
        name = c.get("name", "").lower()
        if q == name or q in name or name in q:
            return c
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
        # Tên nhưng không tìm được → hỏi email
        state.update(step="email", recipient_name=initial_recipient)
        return (f"Không tìm thấy '{initial_recipient}' trong danh bạ. "
                "Vui lòng nhập địa chỉ email của người nhận:"), state

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
                "Vui lòng nhập địa chỉ email người nhận:"), state, False

    if step == "email":
        t = text.strip()
        if _looks_like_email(t):
            state.update(step="subject", recipient_email=t)
            return f"Đã nhận địa chỉ {t}. Chủ đề email là gì?", state, False
        return ("Địa chỉ email không đúng định dạng. "
                "Vui lòng nhập lại (vd: ten@gmail.com):"), state, False

    if step == "subject":
        state.update(step="body", subject=text.strip())
        return "Nội dung email là gì?", state, False

    if step == "body":
        state["body"] = text.strip()
        state["step"] = "confirm"
        return _email_preview(state), state, False

    if step == "confirm":
        t = text.lower().strip()
        # Từ chối / hủy
        if _is_cancel(text) or any(w in t.split() for w in ("không", "no", "thôi")):
            return "Đã hủy gửi email.", None, False
        # Xác nhận gửi
        if _is_affirmative(text):
            return "", state, True
        return 'Mình chưa hiểu. Nói "có" để gửi hoặc "không" để hủy.', state, False

    return "Có lỗi trong quá trình soạn email.", None, False
