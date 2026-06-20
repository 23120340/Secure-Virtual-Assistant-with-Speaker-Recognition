"""State machine cho luồng soạn email nhiều bước."""
import difflib
import re
import time
import unicodedata

# "không" được coi như cancel ở bước confirm (bao gồm trong _CANCEL).
# Cancel match phải là exact/startswith để câu như "có không sao đâu" không bị trigger.
_CANCEL = {"thôi", "hủy", "hủy bỏ", "cancel", "dừng", "không gửi",
           "bỏ qua", "thoát", "không", "no"}

# Hủy DỨT KHOÁT — chỉ dùng ở bước thu nội dung tự do (subject/body) để KHÔNG
# nuốt nhầm câu nội dung email vô tình chứa từ "không"/"thôi"/"dừng".
# Phải khớp CHÍNH XÁC cả câu (sau chuẩn hoá) → user thật sự muốn hủy mới hủy.
_EXPLICIT_CANCEL = {
    "hủy", "huỷ", "hủy bỏ", "huỷ bỏ", "hủy gửi", "huỷ gửi",
    "hủy gửi email", "huỷ gửi email", "hủy email", "huỷ email",
    "thoát", "dừng lại", "cancel", "không gửi nữa", "thôi không gửi",
    "thôi không gửi nữa", "bỏ qua email", "dừng gửi email",
}

# Flow timeout — tránh state cookie sống mãi nếu user bỏ giữa chừng
FLOW_TIMEOUT_SECONDS = 600  # 10 phút


# Dấu câu hay dính ở đầu/cuối transcript ASR (kể cả "…", "!", "?", phẩy).
_EDGE_PUNCT = " \t\r\n.,!?;:…\"'`)(][}{"


def _clean(text: str) -> str:
    """Thường hoá + bỏ dấu câu hai đầu — dùng cho mọi so khớp lệnh/hủy/xác nhận.

    ASR hay trả "Không.", "Hủy!", "Có?"… nếu chỉ .lower() mà không bỏ dấu câu
    cuối thì so khớp chính xác bị trượt.
    """
    return (text or "").lower().strip().strip(_EDGE_PUNCT)


def _strip_tones(s: str) -> str:
    """Bỏ dấu thanh + đ→d để so khớp bỏ qua lỗi sai dấu của ASR."""
    s = s.replace("đ", "d").replace("Đ", "D")
    nfkd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn")


def _confirm_key(text: str) -> str:
    """Khoá so khớp xác nhận: thường + bỏ dấu + bỏ dấu câu + gộp khoảng trắng."""
    s = _strip_tones((text or "").lower())
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _is_cancel(text: str) -> bool:
    t = _clean(text)
    return any(t == w or t.startswith(w + " ") for w in _CANCEL) and len(t) < 40


def _is_explicit_cancel(text: str) -> bool:
    """Hủy dứt khoát (khớp chính xác toàn câu) — an toàn cho subject/body."""
    return _clean(text) in _EXPLICIT_CANCEL


# ==========================================================================
# Nhận diện xác nhận ở bước confirm.
#
# Đây là cổng trước hành động KHÔNG hoàn tác (gửi email) → ưu tiên AN TOÀN:
#   - NO có ưu tiên tuyệt đối (chứa từ chối ở đâu cũng → "no").
#   - YES chỉ chấp nhận khi RÕ RÀNG (khớp chính xác / mở đầu bằng động từ
#     khẳng định không mơ hồ). KHÔNG dùng từ đơn dễ trùng tên/kính ngữ
#     ("có" vs "cô", "đúng" vs "dừng"/"dùng") theo kiểu bỏ dấu.
#   - Lỗi ASR của "xác nhận" (vd "Xét nhật", "Sạch nhạc") bắt bằng heuristic
#     CHẶT (đúng 2 âm: âm đầu hay-nhầm + âm cuối là nhận/nhật/nhạc).
#   - Mọi câu không rõ → None → hỏi lại (KHÔNG tự suy ra để tránh gửi nhầm).
# Khớp giữ NGUYÊN DẤU để tránh va chạm khi bỏ dấu (có/cô, đúng/dừng/dùng).
# ==========================================================================
# YES khớp CHÍNH XÁC toàn câu (đã chuẩn hoá, còn dấu).
_YES_EXACT = {
    "có", "có ạ", "có chứ", "dạ", "dạ có", "vâng", "vâng ạ", "ừ", "ừm",
    "đúng", "đúng rồi", "được", "được rồi", "chuẩn", "chốt", "yes", "ô kê",
}
# YES khi MỞ ĐẦU bằng các động từ/cụm khẳng định ít mơ hồ.
_YES_STARTS = (
    "xác nhận", "đồng ý", "gửi", "ok", "oke", "okê", "okay", "đúng rồi",
    "chính xác", "chốt", "được rồi", "xác minh",
)
# NO khớp CHÍNH XÁC toàn câu.
_NO_EXACT = {"no", "cancel"}
# NO khi MỞ ĐẦU.
_NO_STARTS = (
    "không", "đừng", "thôi", "hủy", "huỷ", "khoan", "dừng", "bỏ qua",
    "thoát", "khỏi", "đừng gửi", "không gửi",
)
# Bất kỳ token nào trong câu thuộc đây → là tín hiệu TỪ CHỐI (ưu tiên tuyệt đối).
_NO_TOKENS = {"không", "đừng", "thôi", "hủy", "huỷ", "khoan", "dừng", "khỏi"}

# Heuristic lỗi ASR của "xác nhận" — bỏ dấu, ĐÚNG 2 âm.
_CONFIRM_FIRST_SYL = {"xac", "sac", "xet", "set", "xe", "sai", "sang",
                      "song", "sat", "sach", "xach", "xa"}
_CONFIRM_LAST_SYL = {"nhan", "nhat", "nhac"}


def _norm_toned(text: str) -> str:
    """Chuẩn hoá GIỮ DẤU: thường + bỏ dấu câu + gộp khoảng trắng."""
    s = re.sub(r"[^\w\s]", " ", (text or "").lower(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def interpret_confirmation(text: str):
    """Diễn giải câu xác nhận ở bước confirm → "yes" | "no" | None.

    None nghĩa là "không chắc" → caller hỏi lại, tuyệt đối KHÔNG tự gửi.
    """
    nkey = _norm_toned(text)
    if not nkey:
        return None
    ntoks = nkey.split()

    # ── NO: ưu tiên tuyệt đối (khớp chính xác / mở đầu / chứa token từ chối). ──
    no_sig = (
        nkey in _NO_EXACT
        or any(nkey == p or nkey.startswith(p + " ") for p in _NO_STARTS)
        or any(t in _NO_TOKENS for t in ntoks)
    )

    # ── YES: chỉ khi rõ ràng. ──
    yes_sig = (
        nkey in _YES_EXACT
        or any(nkey == p or nkey.startswith(p + " ") for p in _YES_STARTS)
    )
    # Lỗi ASR của "xác nhận": đúng 2 âm, bỏ dấu, đầu hay-nhầm + cuối nhận/nhật/nhạc.
    if not yes_sig:
        stoks = _confirm_key(text).split()
        if (len(stoks) == 2 and stoks[0] in _CONFIRM_FIRST_SYL
                and stoks[1] in _CONFIRM_LAST_SYL):
            yes_sig = True

    if no_sig and yes_sig:
        return None          # mơ hồ (vd "có chứ không hủy") → hỏi lại, không gửi
    if no_sig:
        return "no"
    if yes_sig:
        return "yes"
    return None


def _email_preview(state: dict) -> str:
    body_preview = state["body"][:120] + ("..." if len(state["body"]) > 120 else "")
    return (
        f"Xác nhận gửi email:\n"
        f"  Đến   : {state['recipient_name']} ({state['recipient_email']})\n"
        f"  Chủ đề: {state['subject']}\n"
        f"  Nội dung: {body_preview}\n"
        'Nói "xác nhận" để gửi, "hủy bỏ" để hủy.'
    )


def _looks_like_email(text: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", text.strip()))


STOP_WORDS = {"gửi", "cho", "nhắn", "tin", "gọi", "liên", "hệ", "với", "đến", "tới", "email"}

def extract_name_from_query(query: str) -> str:
    words = query.strip().split()
    filtered = [w for w in words if w.lower() not in STOP_WORDS]
    return " ".join(filtered) if filtered else query.strip()    

def _find_contact(query: str, contacts: list) -> dict | None:
    """Tìm liên hệ theo tên — ưu tiên fuzzy match unique.

    Chống nhầm contact: trước đây dùng 2-chiều substring (q in name OR name in q)
    nên "anh tuan" sẽ match "anh" — nguy hiểm trong context gửi email.
    Hiện tại:
      1. Exact match (case-insensitive) → return.
      2. difflib.get_close_matches (ratio ≥ 0.7) → chỉ nhận khi unique.
      3. Substring 1-chiều (q in name) → chỉ nhận khi unique.
    """
    q = extract_name_from_query(query).lower().strip()
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


def _advance_after_recipient(state: dict) -> str:
    """Đã có recipient_name + recipient_email → nhảy tới bước CÒN THIẾU đầu tiên
    (subject → body → confirm), tôn trọng subject/body đã tách sẵn từ câu lệnh.

    Nhờ vậy "gửi email cho X, tiêu đề Y, nội dung Z" đi thẳng tới xác nhận; còn
    "gửi email cho X" thì chỉ hỏi phần còn thiếu (chủ đề), không hỏi lại từ đầu.
    """
    name  = state["recipient_name"]
    email = state["recipient_email"]
    to = f"{name} ({email})" if name and name != email else email
    if not (state.get("subject") or "").strip():
        state["step"] = "subject"
        return f"Gửi cho {to}. Chủ đề email là gì?"
    if not (state.get("body") or "").strip():
        state["step"] = "body"
        return (f"Gửi cho {to}, chủ đề \"{state['subject']}\". "
                "Nội dung email là gì?")
    state["step"] = "confirm"
    return _email_preview(state)


def start_flow(initial_recipient: str = "", contacts: list = None,
               subject: str = "", body: str = "") -> tuple:
    """Khởi động flow soạn email, tách sẵn các phần đã có trong câu lệnh.

    Args:
        initial_recipient: tên/email người nhận trích từ câu lệnh.
        contacts: danh bạ để giải tên → email.
        subject, body: chủ đề/nội dung nếu user đã nói luôn trong câu lệnh
            (vd "gửi email cho A, tiêu đề B, nội dung C").

    Trả về (câu_hỏi_kế_tiếp, state_dict). Nếu đã đủ recipient+subject+body →
    state ở bước "confirm" và câu trả về là bản xem trước để xác nhận.
    """
    contacts = contacts or []
    state = {
        "step": "recipient",
        "user_id": None,
        "recipient_name": "",
        "recipient_email": "",
        "subject": (subject or "").strip(),
        "body": (body or "").strip(),
        "started_at": time.time(),
    }

    if initial_recipient:
        contact = _find_contact(initial_recipient, contacts)
        if contact:
            state.update(recipient_name=contact["name"],
                         recipient_email=contact["email"])
            return _advance_after_recipient(state), state
        if _looks_like_email(initial_recipient):
            state.update(recipient_name=initial_recipient,
                         recipient_email=initial_recipient)
            return _advance_after_recipient(state), state
        # Tên nhưng không tìm được → hỏi email HOẶC tên lại (giữ subject/body
        # đã tách để dùng lại sau khi có email).
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

    step = state["step"]

    # Cancel detection theo bước:
    #   - subject / body: nội dung TỰ DO → chỉ hủy khi user nói lệnh hủy DỨT KHOÁT
    #     (khớp chính xác). Tránh nuốt nhầm câu nội dung chứa "không"/"thôi"/"dừng".
    #   - confirm: xử lý yes/no trong block confirm bên dưới (interpret_confirmation).
    #   - recipient / email: dùng tập cancel rộng như cũ.
    if step in ("subject", "body"):
        if _is_explicit_cancel(text):
            return "Đã hủy gửi email.", None, False
    elif step != "confirm":
        if _is_cancel(text):
            return "Đã hủy gửi email.", None, False

    if step == "recipient":
        t = text.strip()
        contact = _find_contact(t, contacts)
        if contact:
            state.update(recipient_name=contact["name"],
                         recipient_email=contact["email"])
            return _advance_after_recipient(state), state, False
        if _looks_like_email(t):
            state.update(recipient_name=t, recipient_email=t)
            return _advance_after_recipient(state), state, False
        state.update(step="email", recipient_name=t)
        return (f"Không tìm thấy '{t}' trong danh bạ. "
                "Bạn có thể đọc lại tên rõ hơn, hoặc đọc / nhập trực tiếp "
                "địa chỉ email người nhận (vd: ten@gmail.com):"), state, False

    if step == "email":
        t = text.strip()
        if _looks_like_email(t):
            state.update(recipient_email=t)
            return _advance_after_recipient(state), state, False
        # User nhập tên mới (không phải email) → thử lookup contacts lại.
        # Trường hợp: ở step "recipient" tên đầu không match, user đọc lại tên
        # rõ hơn ở step "email" → vẫn cần tìm trong danh bạ.
        contact = _find_contact(t, contacts)
        if contact:
            state.update(recipient_name=contact["name"],
                         recipient_email=contact["email"])
            return _advance_after_recipient(state), state, False
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
        # Nhận diện xác nhận robust với từ ngắn ("có"/"không") + lỗi ASR.
        verdict = interpret_confirmation(text)
        if verdict == "yes":
            return "", state, True
        if verdict == "no":
            return "Đã hủy gửi email.", None, False
        return ('Mình chưa rõ. Nói "có" / "xác nhận" để gửi, '
                'hoặc "không" / "hủy" để hủy.'), state, False

    return "Có lỗi trong quá trình soạn email.", None, False
