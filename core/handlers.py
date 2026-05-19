"""Handlers cho từng intent. Mỗi handler nhận:
    entities: dict
    user: dict | None  (None = guest)
    **kwargs: extra context (db, ...)
Trả về: response_text (str) HOẶC HandlerResult (cho intent cần signal action).
"""
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional


@dataclass
class HandlerResult:
    """Structured return type cho handler cần signal extra action.

    Caller (web/app.py) check isinstance(r, HandlerResult) → unpack action_*.
    Handlers cũ chỉ return str vẫn hoạt động bình thường (backward compat).
    """
    text: str
    action_type: Optional[str] = None
    action_data: Optional[dict] = None


# Email rate limit: max EMAIL_RATE_MAX email / EMAIL_RATE_WINDOW giây / user.
# User authenticated có thể gửi từ Gmail API → chống spam nếu account bị compromise.
EMAIL_RATE_MAX    = 10
EMAIL_RATE_WINDOW = 3600  # 1 giờ
# Trước đây dùng in-process dict (defaultdict(list)) — reset mỗi restart và
# không share giữa worker. Đã chuyển sang SQLite (xem UserDB.count_emails_sent_within
# / oldest_email_within / record_email_sent).


def _email_rate_check(user_id: str, db) -> tuple[bool, int]:
    """Returns (allowed, wait_min). allowed=False → từ chối, có quota khi nào.

    `db`: UserDB instance (caller phải truyền). Khi db=None (vd: unit test)
    fallback allowed=True để không break test.
    """
    if db is None:
        return True, 0
    count = db.count_emails_sent_within(user_id, EMAIL_RATE_WINDOW)
    if count < EMAIL_RATE_MAX:
        return True, 0
    oldest = db.oldest_email_within(user_id, EMAIL_RATE_WINDOW)
    if oldest is None:
        return True, 0
    wait_min = int((oldest + EMAIL_RATE_WINDOW - time.time()) / 60) + 1
    return False, max(wait_min, 1)


def _email_rate_record(user_id: str, db):
    if db is not None:
        db.record_email_sent(user_id)


# ==========================================================================
# NORMAL handlers
# ==========================================================================
def handle_get_time(entities, user, **kwargs) -> str:
    now = datetime.now()
    return f"Bây giờ là {now.strftime('%H giờ %M phút')}, ngày {now.strftime('%d/%m/%Y')}."


def handle_get_weather(entities, user, **kwargs) -> str:
    # Demo mode — chưa tích hợp API thời tiết thật (OpenWeatherMap / Visual Crossing).
    location = entities.get("location", "đây")
    conditions = ["nắng đẹp", "có mưa nhẹ", "nhiều mây", "trời quang"]
    temp = random.randint(22, 33)
    return (f"[Demo] Thời tiết ở {location} hiện tại {random.choice(conditions)}, "
            f"nhiệt độ khoảng {temp} độ C.")


def handle_tell_joke(entities, user, **kwargs) -> str:
    jokes = [
        "Tại sao máy tính không bao giờ đói? Vì nó luôn có chip.",
        "Hai con bò gặp nhau trên đường, một con hỏi: Cậu có sợ bệnh bò điên không? Con kia trả lời: Sao tớ phải sợ, tớ là con gà mà.",
        "Học sinh: Thầy ơi em không biết câu này. Thầy: Em ngồi xuống đi. Học sinh: Em đang ngồi rồi ạ. Thầy: Vậy thì em đứng lên để nghe lời thầy.",
    ]
    return random.choice(jokes)


def handle_general_question(entities, user, **kwargs) -> str:
    from .nlu import get_chat
    query = entities.get("query", "").strip()
    if not query:
        return "Bạn hỏi gì vậy? Mình chưa nghe rõ câu hỏi."
    chat = get_chat()
    if chat is None:
        return (f"Câu hỏi hay đấy, nhưng mình chưa được kết nối AI để trả lời. "
                "Cần cấu hình GEMINI_API_KEY.")
    user_name = user["name"] if user else ""
    try:
        return chat.answer(query, user_name)
    except Exception:
        return "Mình tạm thời không trả lời được. Bạn thử lại sau nhé."


# ==========================================================================
# IMPORTANT handlers — chỉ chạy SAU khi SV pass
# ==========================================================================
def handle_read_notes(entities, user, **kwargs) -> str:
    if user is None:
        return "Chưa xác thực được người dùng."
    name = user["name"]
    notes = user["preferences"].get("notes", [])
    if not notes:
        return f"{name} ơi, bạn chưa có ghi chú nào cả."
    head = "\n".join(f"- {n}" for n in notes[:3])
    return f"Ghi chú của {name}:\n{head}"


def handle_send_email(entities, user, **kwargs) -> str:
    """Gửi email qua Gmail API (OAuth 2.0).

    Nếu user chưa xác thực → trả về OAUTH_RESP_PREFIX + auth_url để web/app.py
    chuyển thành action_type="oauth_required" với nút đăng nhập Google.
    """
    if user is None:
        return "Mình chưa nhận ra giọng bạn nên không thể gửi email."

    from .oauth import refresh_access_token
    from .gmail_api import send_email as _gmail_send

    db      = kwargs.get("db")
    user_id = user["user_id"]

    # Rate limit: chặn trước khi tốn cycle (lookup recipient, OAuth refresh, ...)
    allowed, wait_min = _email_rate_check(user_id, db)
    if not allowed:
        return (f"Bạn đã gửi {EMAIL_RATE_MAX} email trong giờ vừa qua. "
                f"Vui lòng đợi khoảng {wait_min} phút rồi thử lại.")

    # ── 1. Kiểm tra OAuth token ─────────────────────────────────────────────
    token_data = db.get_oauth_token(user_id) if db else None

    if not token_data:
        # KHÔNG build_auth_url ở đây — URL với state=user_id là predictable.
        # Frontend dùng identified_user_id để redirect qua /auth/google route,
        # nơi server generate nonce + PKCE đúng chuẩn (web/app.py:auth_google).
        return HandlerResult(
            text=(f"{user['name']} ơi, bạn chưa xác thực Gmail. "
                  "Nhấn nút 'Đăng nhập Google' trên màn hình để tiếp tục."),
            action_type="oauth_required",
            action_data={"user_id": user_id},
        )

    # ── 2. Refresh nếu access_token hết hạn ────────────────────────────────
    if time.time() > token_data.get("expiry", 0):
        try:
            updated = refresh_access_token(token_data["refresh_token"])
            token_data.update(updated)
            if db:
                db.save_oauth_token(user_id, token_data)
        except Exception:
            if db:
                db.delete_oauth_token(user_id)
            return HandlerResult(
                text="Token Gmail hết hạn. Vui lòng đăng nhập lại.",
                action_type="oauth_required",
                action_data={"user_id": user_id},
            )

    access_token  = token_data["access_token"]
    gmail_address = token_data.get("gmail_address", "")

    # ── 3. Lấy thông tin người nhận ─────────────────────────────────────────
    recipient_name  = entities.get("recipient", "")
    recipient_email = entities.get("recipient_email", "")
    subject = entities.get("subject", "Tin nhắn từ Trợ lý Ảo")
    body    = entities.get("body", entities.get("content", ""))

    if not recipient_email and db and recipient_name:
        match = db.find_user_by_name_substring(recipient_name)
        if match:
            recipient_email = match["preferences"].get("email", "")

    if not recipient_email:
        if recipient_name:
            return (f"Không tìm thấy email của '{recipient_name}' trong hệ thống. "
                    "Hãy đảm bảo người đó đã đăng ký email trong hồ sơ "
                    "(hoặc nói tên đầy đủ nếu trùng tên).")
        return "Bạn muốn gửi email cho ai? Hãy nói tên người nhận."

    # ── 4. Gửi qua Gmail API ────────────────────────────────────────────────
    try:
        _gmail_send(
            access_token=access_token,
            to=recipient_email,
            subject=subject,
            body=body or f"Email từ {user['name']} qua Trợ lý Ảo.",
            from_name=user["name"],
        )
        _email_rate_record(user_id, db)
        from_note = f" (từ {gmail_address})" if gmail_address else ""
        return (f"Đã gửi email thành công đến "
                f"{recipient_name or recipient_email}!{from_note}")
    except Exception as e:
        detail = str(e)
        try:
            resp = getattr(e, "response", None)
            if resp is not None:
                err_json = resp.json().get("error", {})
                msg      = err_json.get("message", "")
                reasons  = [x.get("reason", "") for x in err_json.get("errors", [])]
                reason   = next((r for r in reasons if r), "")
                if msg:
                    detail = msg
                if reason == "insufficientPermissions":
                    detail += " — token thiếu scope gmail.send. Vào trang quản lý để Hủy xác thực rồi Xác thực Gmail lại."
                elif reason == "forbidden":
                    detail += " — tài khoản không được phép dùng Gmail API (kiểm tra Test Users trên Google Cloud)."
        except Exception:
            pass
        return f"Gửi email thất bại: {detail}"


def handle_check_balance(entities, user, **kwargs) -> str:
    if user is None:
        return "Mình chưa nhận ra giọng bạn nên không kiểm tra được số dư."
    balance = user["preferences"].get("balance")
    if balance is None:
        return f"{user['name']} chưa thiết lập số dư trong hồ sơ."
    return f"Số dư của {user['name']} là {balance:,} đồng."


_DELETABLE_TARGETS = {
    "ghi chú": "notes", "ghi chu": "notes",
    "lịch": "schedule", "lich": "schedule",
    "tất cả": "all", "tat ca": "all",
}


def handle_delete_data(entities, user, **kwargs) -> str:
    if user is None:
        return "Chưa nhận ra giọng — không thể xóa dữ liệu."

    raw = entities.get("target", "").lower().strip()
    target_key = _DELETABLE_TARGETS.get(raw)
    if not target_key:
        return (f"Mình không hiểu '{raw or 'dữ liệu'}'. "
                "Bạn muốn xóa: ghi chú, lịch, hay tất cả?")

    db = kwargs.get("db")
    if db is None:
        return "Không thể xóa dữ liệu — thiếu kết nối database."

    prefs = dict(user["preferences"])
    if target_key == "notes":
        prefs.pop("notes", None)
        label = "ghi chú"
    elif target_key == "schedule":
        prefs.pop("schedule", None)
        label = "lịch"
    else:  # all
        prefs.clear()
        label = "toàn bộ dữ liệu cá nhân"

    db.update_preferences(user["user_id"], prefs)
    return f"Đã xóa {label} của {user['name']}."


def handle_open_files(entities, user, **kwargs) -> str | HandlerResult:
    if user is None:
        return "Chưa xác thực được người dùng — không thể mở file."
    file_count = kwargs.get("file_count")
    if file_count is None:
        try:
            from . import config
            user_dir = config.USER_FILES_DIR / user["user_id"]
            file_count = sum(1 for p in user_dir.iterdir() if p.is_file()) if user_dir.exists() else 0
        except Exception:
            file_count = None

    if file_count == 0:
        text = f"{user['name']} chưa có file nào đã upload. Mình sẽ mở trang file để bạn thêm file."
    elif file_count is None:
        text = f"Đã xác thực thành công. Đang mở file của {user['name']}."
    else:
        text = f"Đã xác thực thành công. Đang mở {file_count} file của {user['name']}."

    return HandlerResult(
        text=text,
        action_type="show_files",
        action_data={"user_id": user["user_id"], "file_count": file_count},
    )


# Cap để tránh user bị spam khiến preferences phình to vô hạn.
_MAX_NOTES = 50
_MAX_SCHEDULE = 50
_MAX_CONTACTS = 100


def handle_add_note(entities, user, **kwargs) -> str:
    """Thêm ghi chú mới vào preferences.notes của user (FIFO khi vượt cap)."""
    if user is None:
        return "Chưa nhận ra giọng — không thể thêm ghi chú."
    content = (entities.get("content") or "").strip()
    if not content:
        return "Bạn muốn ghi chú nội dung gì? Mình chưa nghe rõ."
    db = kwargs.get("db")
    if db is None:
        return "Không thể lưu ghi chú — thiếu kết nối database."

    prefs = dict(user["preferences"])
    notes = list(prefs.get("notes", []))
    notes.append(content)
    if len(notes) > _MAX_NOTES:
        notes = notes[-_MAX_NOTES:]  # giữ N gần nhất
    prefs["notes"] = notes
    db.update_preferences(user["user_id"], prefs)
    return f"Đã thêm ghi chú cho {user['name']}: \"{content}\". Tổng cộng {len(notes)} ghi chú."


def handle_add_schedule(entities, user, **kwargs) -> str:
    """Thêm 1 mục lịch. Nội dung + thời gian (nếu có) ghép vào 1 string."""
    if user is None:
        return "Chưa nhận ra giọng — không thể thêm lịch."
    content = (entities.get("content") or "").strip()
    time_str = (entities.get("time") or "").strip()
    if not content:
        return "Bạn muốn thêm lịch nội dung gì? Mình chưa rõ."
    db = kwargs.get("db")
    if db is None:
        return "Không thể lưu lịch — thiếu kết nối database."

    entry = f"{time_str} - {content}" if time_str else content
    prefs = dict(user["preferences"])
    schedule = list(prefs.get("schedule", []))
    schedule.append(entry)
    if len(schedule) > _MAX_SCHEDULE:
        schedule = schedule[-_MAX_SCHEDULE:]
    prefs["schedule"] = schedule
    db.update_preferences(user["user_id"], prefs)
    return f"Đã thêm vào lịch của {user['name']}: \"{entry}\"."


def handle_set_reminder(entities, user, **kwargs) -> str:
    """Đặt nhắc việc — parse thời gian VN + lưu vào preferences.reminders."""
    if user is None:
        return "Chưa nhận ra giọng — không thể đặt nhắc."
    content = (entities.get("content") or "").strip()
    when_text = (entities.get("when") or "").strip()
    if not content:
        return "Bạn muốn nhắc nội dung gì? Mình chưa nghe rõ."
    if not when_text:
        return "Bạn muốn nhắc lúc nào? Ví dụ: '9 giờ sáng mai', '30 phút nữa'."
    db = kwargs.get("db")
    if db is None:
        return "Không thể lưu nhắc — thiếu kết nối database."

    from . import reminders as _rm
    item = _rm.make_reminder(content, when_text)
    prefs = dict(user["preferences"])
    prefs["reminders"] = _rm.append_reminder(prefs.get("reminders", []), item)
    db.update_preferences(user["user_id"], prefs)

    if item["when_iso"]:
        from datetime import datetime
        dt = datetime.fromisoformat(item["when_iso"])
        when_str = dt.strftime("%H:%M ngày %d/%m")
        return f"Đã đặt nhắc cho {user['name']}: \"{content}\" lúc {when_str}."
    else:
        # Parser fail → lưu raw, user vẫn thấy được khi list.
        return (f"Đã ghi nhắc \"{content}\" với thời gian '{when_text}', "
                f"nhưng mình chưa parse được giờ chính xác. Bạn có thể sửa "
                f"trong trang Thông tin → preferences.reminders.")


def handle_list_reminders(entities, user, **kwargs) -> str:
    """Liệt kê reminder của user: pending + due. PERSONAL — dùng SID."""
    if user is None:
        return "Mình chưa nhận ra giọng nên không truy cập được nhắc việc."
    reminders = user["preferences"].get("reminders") or []
    if not reminders:
        return f"{user['name']} ơi, bạn chưa có nhắc việc nào."

    from . import reminders as _rm
    due = _rm.due_reminders(reminders)
    pending = [r for r in reminders if not r.get("fired")]
    lines = []
    if due:
        lines.append(f"🔔 Đến hạn ({len(due)}):")
        for r in due[:5]:
            lines.append("  - " + _rm.format_reminder_for_speech(r))
    not_due = [r for r in pending if r not in due]
    if not_due:
        lines.append(f"⏳ Sắp tới ({len(not_due)}):")
        for r in not_due[:5]:
            lines.append("  - " + _rm.format_reminder_for_speech(r))
    if not lines:
        return f"{user['name']} ơi, không có nhắc việc nào đang chờ."
    return "\n".join(lines)


def handle_add_contact(entities, user, **kwargs) -> str:
    """Thêm contact mới (name + email) vào preferences.contacts."""
    if user is None:
        return "Chưa nhận ra giọng — không thể thêm liên hệ."
    name = (entities.get("name") or "").strip()
    email = (entities.get("email") or "").strip()
    if not name or not email:
        return ("Cần cả tên và email để thêm liên hệ. Bạn có thể nói "
                "\"thêm liên hệ Tuấn email tuan@example.com\".")
    # Validate email format đơn giản — không strict RFC.
    if "@" not in email or " " in email or len(email) > 254:
        return f"Email '{email}' không hợp lệ."
    db = kwargs.get("db")
    if db is None:
        return "Không thể lưu liên hệ — thiếu kết nối database."

    prefs = dict(user["preferences"])
    contacts = list(prefs.get("contacts", []))
    # Nếu name đã tồn tại → update email (idempotent), không duplicate.
    for c in contacts:
        if (c.get("name") or "").strip().lower() == name.lower():
            c["email"] = email
            db.update_preferences(user["user_id"], prefs)
            return f"Đã cập nhật email của {name} thành {email}."
    contacts.append({"name": name, "email": email})
    if len(contacts) > _MAX_CONTACTS:
        contacts = contacts[-_MAX_CONTACTS:]
    prefs["contacts"] = contacts
    db.update_preferences(user["user_id"], prefs)
    return f"Đã thêm liên hệ {name} ({email}) vào danh bạ."


# ==========================================================================
# PERSONAL handlers — dùng user info để cá nhân hóa
# ==========================================================================
def handle_greet(entities, user, **kwargs) -> str:
    if user is None:
        return "Xin chào! Bạn là khách. Đăng ký giọng nói để mình nhận diện được nhé."
    return f"Chào {user['name']}, mình nhận ra giọng bạn rồi. Hôm nay mình giúp gì được?"


def handle_play_music(entities, user, **kwargs) -> str:
    if user is None:
        genre = entities.get("genre", "v-pop")
        return f"Phát một bản nhạc {genre} cho khách nghe đây."

    fav = user["preferences"].get("favorite_genre", "pop")
    genre = entities.get("genre", fav)
    favorite_artist = user["preferences"].get("favorite_artist")
    extra = f" — đặc biệt là của {favorite_artist}" if favorite_artist else ""
    return f"Phát nhạc {genre} cho {user['name']}{extra}."


def handle_show_schedule(entities, user, **kwargs) -> str:
    """Show lịch — merge local `preferences.schedule` + Google Calendar events.

    Khi `config.ENABLE_CALENDAR_INTEGRATION=true` AND user đã link OAuth có
    scope `calendar.events.readonly` → fetch events 24h tới từ Google. Nếu
    fail (token expired, scope thiếu, network) → fall back về local schedule.
    """
    if user is None:
        return "Mình chưa nhận ra giọng bạn nên không truy cập được lịch cá nhân."

    from . import config
    name = user["name"]
    local_items = list(user["preferences"].get("schedule") or [])
    gcal_items: list[str] = []

    # Thử fetch Google Calendar nếu integration bật + có token.
    if config.ENABLE_CALENDAR_INTEGRATION:
        db = kwargs.get("db")
        if db is not None:
            token_data = db.get_oauth_token(user["user_id"])
            if token_data:
                gcal_items = _fetch_calendar_events(token_data, db, user["user_id"])

    if not local_items and not gcal_items:
        return f"{name} ơi, hôm nay không có việc gì trong lịch."

    lines = []
    if gcal_items:
        lines.append(f"📅 Google Calendar ({len(gcal_items)}): "
                     + "; ".join(gcal_items[:3]))
    if local_items:
        lines.append(f"📝 Lịch cá nhân ({len(local_items)}): "
                     + "; ".join(local_items[:3]))
    return f"Lịch của {name} hôm nay:\n" + "\n".join(lines)


def _fetch_calendar_events(token_data: dict, db, user_id: str) -> list[str]:
    """Helper: fetch + format Google Calendar events 24h tới. Trả [] khi fail.

    Tự refresh token nếu hết hạn (giống `handle_send_email`).
    """
    import time as _t
    try:
        from .oauth import refresh_access_token
        from .calendar_api import list_events, format_events_for_speech
    except ImportError:
        return []
    # Refresh nếu access_token hết hạn.
    if _t.time() > token_data.get("expiry", 0):
        try:
            updated = refresh_access_token(token_data["refresh_token"])
            token_data.update(updated)
            db.save_oauth_token(user_id, token_data)
        except Exception:
            return []
    try:
        events = list_events(token_data["access_token"], max_results=10)
    except Exception:
        # Token thiếu scope (insufficientPermissions) hoặc network fail → silently
        # fall back, không spam user "Lỗi calendar".
        return []
    formatted = format_events_for_speech(events)
    if not formatted:
        return []
    # Chuyển từ string "; "-joined thành list để uniform với local schedule.
    return [s.strip() for s in formatted.split(";") if s.strip()]


# ==========================================================================
# Fallback
# ==========================================================================
def handle_unknown(entities, user, **kwargs) -> str:
    return "Mình chưa hiểu ý bạn. Bạn nói rõ hơn được không?"


# ==========================================================================
# Registry
# ==========================================================================
HANDLERS = {
    "get_time": handle_get_time,
    "get_weather": handle_get_weather,
    "tell_joke": handle_tell_joke,
    "general_question": handle_general_question,
    "read_notes": handle_read_notes,
    "send_email": handle_send_email,
    "check_balance": handle_check_balance,
    "delete_data": handle_delete_data,
    "open_files": handle_open_files,
    "add_note": handle_add_note,
    "add_schedule": handle_add_schedule,
    "add_contact": handle_add_contact,
    "set_reminder": handle_set_reminder,
    "list_reminders": handle_list_reminders,
    "greet": handle_greet,
    "play_music": handle_play_music,
    "show_schedule": handle_show_schedule,
    "unknown": handle_unknown,
}
