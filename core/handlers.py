"""Handlers cho từng intent. Mỗi handler nhận:
    entities: dict
    user: dict | None  (None = guest)
    **kwargs: extra context (db, smtp_config, ...)
Trả về: response_text (str)
"""
import random
import smtplib
from datetime import datetime
from email.mime.text import MIMEText


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
    query = entities.get("query", "")
    return f"Để mình tìm hiểu về '{query}' rồi trả lời bạn sau nhé."


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
    """Gửi email thật qua SMTP nếu config đầy đủ, fallback về thông báo mock."""
    if user is None:
        return "Mình chưa nhận ra giọng bạn nên không thể gửi email."

    sender_name = user["name"]
    sender_email = user["preferences"].get("email", "")
    recipient_name = entities.get("recipient", "")
    subject = entities.get("subject", "Tin nhắn từ Trợ lý Ảo")
    body = entities.get("body", entities.get("content", ""))

    db = kwargs.get("db")
    smtp_cfg = kwargs.get("smtp_config", {})
    smtp_user = smtp_cfg.get("user", "")
    smtp_pass = smtp_cfg.get("password", "")
    smtp_host = smtp_cfg.get("host", "smtp.gmail.com")
    smtp_port = smtp_cfg.get("port", 587)

    # Ưu tiên dùng email đã được flow xác nhận trước
    recipient_email = entities.get("recipient_email", "")

    if not recipient_email:
        # Fallback: tìm theo tên trong DB
        if db and recipient_name:
            for u in db.list_users():
                if recipient_name.lower() in u["name"].lower():
                    recipient_email = u["preferences"].get("email", "")
                    break

    if not recipient_email:
        if recipient_name:
            return (f"Không tìm thấy email của '{recipient_name}' trong hệ thống. "
                    "Hãy đảm bảo người đó đã đăng ký email trong hồ sơ.")
        return "Bạn muốn gửi email cho ai? Hãy nói tên người nhận."

    if not sender_email:
        return (f"{sender_name} ơi, bạn chưa có địa chỉ email trong hồ sơ. "
                "Hãy cập nhật email của bạn trong trang cài đặt người dùng.")

    if not smtp_user or not smtp_pass:
        # SMTP chưa cấu hình — trả về thông báo mô phỏng
        return (f"Đã chuẩn bị email từ {sender_name} ({sender_email}) "
                f"gửi tới {recipient_name} ({recipient_email}). "
                "Máy chủ email chưa được cấu hình nên chưa gửi thật.")

    try:
        msg = MIMEText(body or f"Email từ {sender_name} gửi qua Trợ lý Ảo.", "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = f"{sender_name} <{smtp_user}>"
        msg["To"] = recipient_email

        with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, [recipient_email], msg.as_string())

        return (f"Đã gửi email thành công từ {sender_name} ({sender_email}) "
                f"đến {recipient_name} ({recipient_email})!")
    except smtplib.SMTPAuthenticationError:
        return "Xác thực SMTP thất bại. Kiểm tra lại cấu hình email ứng dụng."
    except smtplib.SMTPException as e:
        return f"Gửi email thất bại: {e}"
    except Exception as e:
        return f"Lỗi khi gửi email: {e}"


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
