"""Handlers cho từng intent. Mỗi handler nhận:
    entities: dict
    user: dict | None  (None = guest)
    **kwargs: extra context (db, ...)
Trả về: response_text (str)
"""
import random
from datetime import datetime

from .oauth import OAUTH_RESP_PREFIX, build_auth_url


# ==========================================================================
# NORMAL handlers
# ==========================================================================
def handle_get_time(entities, user, **kwargs) -> str:
    now = datetime.now()
    return f"Bây giờ là {now.strftime('%H giờ %M phút')}, ngày {now.strftime('%d/%m/%Y')}."


def handle_get_weather(entities, user, **kwargs) -> str:
    location = entities.get("location", "đây")
    conditions = ["nắng đẹp", "có mưa nhẹ", "nhiều mây", "trời quang"]
    temp = random.randint(22, 33)
    return f"Thời tiết ở {location} hiện tại {random.choice(conditions)}, nhiệt độ khoảng {temp} độ C."


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
    return chat.answer(query, user_name)


# ==========================================================================
# IMPORTANT handlers — chỉ chạy SAU khi SV pass
# ==========================================================================
def handle_read_notes(entities, user, **kwargs) -> str:
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

    import time
    from .oauth import refresh_access_token
    from .gmail_api import send_email as _gmail_send

    db      = kwargs.get("db")
    user_id = user["user_id"]

    # ── 1. Kiểm tra OAuth token ─────────────────────────────────────────────
    token_data = db.get_oauth_token(user_id) if db else None

    if not token_data:
        try:
            auth_url = build_auth_url(state=user_id)
        except RuntimeError as e:
            return f"Chưa cấu hình Gmail OAuth: {e}"
        return (f"{OAUTH_RESP_PREFIX}{auth_url}\n"
                f"{user['name']} ơi, bạn chưa xác thực Gmail. "
                "Nhấn nút 'Đăng nhập Google' trên màn hình để tiếp tục.")

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
            try:
                auth_url = build_auth_url(state=user_id)
            except RuntimeError:
                return "Token Gmail hết hạn và không thể làm mới. Kiểm tra cấu hình OAuth."
            return (f"{OAUTH_RESP_PREFIX}{auth_url}\n"
                    "Token Gmail hết hạn. Vui lòng đăng nhập lại.")

    access_token  = token_data["access_token"]
    gmail_address = token_data.get("gmail_address", "")

    # ── 3. Lấy thông tin người nhận ─────────────────────────────────────────
    recipient_name  = entities.get("recipient", "")
    recipient_email = entities.get("recipient_email", "")
    subject = entities.get("subject", "Tin nhắn từ Trợ lý Ảo")
    body    = entities.get("body", entities.get("content", ""))

    if not recipient_email and db and recipient_name:
        for u in db.list_users():
            full = db.get_user(u["user_id"])
            if full and recipient_name.lower() in full["name"].lower():
                recipient_email = full["preferences"].get("email", "")
                break

    if not recipient_email:
        if recipient_name:
            return (f"Không tìm thấy email của '{recipient_name}' trong hệ thống. "
                    "Hãy đảm bảo người đó đã đăng ký email trong hồ sơ.")
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
    balance = user["preferences"].get("balance", 12_500_000)
    return f"Số dư tài khoản của {user['name']} là {balance:,} đồng."


def handle_delete_data(entities, user, **kwargs) -> str:
    target = entities.get("target", "dữ liệu")
    return f"Đã xác thực thành công. Mình sẽ xóa {target} của {user['name']}."


def handle_open_files(entities, user, **kwargs) -> str:
    return f"Đã xác thực thành công. Đang mở file của {user['name']}."


# ==========================================================================
# PERSONAL handlers — dùng user info để cá nhân hóa
# ==========================================================================
def handle_greet(entities, user, **kwargs) -> str:
    if user is None:
        return "Xin chào! Bạn là khách. Đăng ký giọng nói để mình nhận diện được nhé."
    return f"Chào {user['name']}, mình nhận ra giọng bạn rồi. Hôm nay mình giúp gì được?"


def handle_play_music(entities, user, **kwargs) -> str:
    if user is None:
        genre = entities.get("genre", "pop")
        return f"Phát một bản nhạc {genre} cho khách nghe đây."

    fav = user["preferences"].get("favorite_genre", "pop")
    genre = entities.get("genre", fav)
    favorite_artist = user["preferences"].get("favorite_artist")
    extra = f" — đặc biệt là của {favorite_artist}" if favorite_artist else ""
    return f"Phát nhạc {genre} cho {user['name']}{extra}."


def handle_show_schedule(entities, user, **kwargs) -> str:
    if user is None:
        return "Mình chưa nhận ra giọng bạn nên không truy cập được lịch cá nhân."
    schedule = user["preferences"].get("schedule", [])
    if not schedule:
        return f"{user['name']} ơi, hôm nay không có việc gì trong lịch."
    items = "; ".join(schedule[:3])
    return f"Lịch của {user['name']} hôm nay: {items}."


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
    "greet": handle_greet,
    "play_music": handle_play_music,
    "show_schedule": handle_show_schedule,
    "unknown": handle_unknown,
}
