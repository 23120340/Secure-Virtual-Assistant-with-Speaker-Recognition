"""Handlers cho từng intent. Mỗi handler nhận:
    entities: dict
    user: dict | None  (None = guest)
Trả về: response_text (str)

Ở giai đoạn này dùng mock data. Sau này có thể thay bằng API thật
(OpenWeather, Gmail API, lịch Google, ...).
"""
import random
from datetime import datetime


# ==========================================================================
# NORMAL handlers
# ==========================================================================
def handle_get_time(entities, user) -> str:
    now = datetime.now()
    return f"Bây giờ là {now.strftime('%H giờ %M phút')}, ngày {now.strftime('%d/%m/%Y')}."


def handle_get_weather(entities, user) -> str:
    location = entities.get("location", "đây")
    # MOCK — sau thay OpenWeather API
    conditions = ["nắng đẹp", "có mưa nhẹ", "nhiều mây", "trời quang"]
    temp = random.randint(22, 33)
    return f"Thời tiết ở {location} hiện tại {random.choice(conditions)}, nhiệt độ khoảng {temp} độ C."


def handle_tell_joke(entities, user) -> str:
    jokes = [
        "Tại sao máy tính không bao giờ đói? Vì nó luôn có chip.",
        "Hai con bò gặp nhau trên đường, một con hỏi: Cậu có sợ bệnh bò điên không? Con kia trả lời: Sao tớ phải sợ, tớ là con gà mà.",
        "Học sinh: Thầy ơi em không biết câu này. Thầy: Em ngồi xuống đi. Học sinh: Em đang ngồi rồi ạ. Thầy: Vậy thì em đứng lên để nghe lời thầy.",
    ]
    return random.choice(jokes)


def handle_general_question(entities, user) -> str:
    # MOCK — production sẽ gọi LLM trả lời
    query = entities.get("query", "")
    return f"Để mình tìm hiểu về '{query}' rồi trả lời bạn sau nhé."


# ==========================================================================
# IMPORTANT handlers — chỉ chạy SAU khi SV pass
# ==========================================================================
def handle_read_notes(entities, user) -> str:
    name = user["name"]
    notes = user["preferences"].get("notes", [])
    if not notes:
        return f"{name} ơi, bạn chưa có ghi chú nào cả."
    head = "\n".join(f"- {n}" for n in notes[:3])
    return f"Ghi chú của {name}:\n{head}"


def handle_send_email(entities, user) -> str:
    recipient = entities.get("recipient", "(chưa rõ)")
    return f"Đã chuẩn bị email gửi {recipient}. Bạn xác nhận lại nội dung trước khi gửi nhé."


def handle_check_balance(entities, user) -> str:
    # MOCK
    balance = user["preferences"].get("balance", 12_500_000)
    return f"Số dư tài khoản của {user['name']} là {balance:,} đồng."


def handle_delete_data(entities, user) -> str:
    target = entities.get("target", "dữ liệu")
    return f"Đã xác thực thành công. Mình sẽ xóa {target} của {user['name']}."


def handle_open_files(entities, user) -> str:
    return f"Đã xác thực thành công. Đang mở file của {user['name']}."


# ==========================================================================
# PERSONAL handlers — dùng user info để cá nhân hóa
# ==========================================================================
def handle_greet(entities, user) -> str:
    if user is None:
        return "Xin chào! Bạn là khách. Đăng ký giọng nói để mình nhận diện được nhé."
    return f"Chào {user['name']}, mình nhận ra giọng bạn rồi. Hôm nay mình giúp gì được?"


def handle_play_music(entities, user) -> str:
    if user is None:
        # Guest → genre default
        genre = entities.get("genre", "pop")
        return f"Phát một bản nhạc {genre} cho khách nghe đây."

    # Lấy gu nhạc từ preferences, ưu tiên entity nếu có
    fav = user["preferences"].get("favorite_genre", "pop")
    genre = entities.get("genre", fav)
    favorite_artist = user["preferences"].get("favorite_artist")
    extra = f" — đặc biệt là của {favorite_artist}" if favorite_artist else ""
    return f"Phát nhạc {genre} cho {user['name']}{extra}."


def handle_show_schedule(entities, user) -> str:
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
def handle_unknown(entities, user) -> str:
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
