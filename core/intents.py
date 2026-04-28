"""Định nghĩa intents và nhóm bảo mật.

Theo yêu cầu đề bài, mỗi intent thuộc 1 trong 3 nhóm:
  - normal:    không cần xác thực (ai cũng dùng được)
  - important: PHẢI xác thực người nói (SV) trước khi thực thi
  - personal:  cần SID để cá nhân hóa response (chào theo tên, lịch riêng, ...)
"""
from enum import Enum


class AuthLevel(str, Enum):
    NORMAL = "normal"
    IMPORTANT = "important"
    PERSONAL = "personal"


# Mỗi intent: name, level, description (cho LLM hiểu), entities cần extract
INTENTS = {
    # --- NORMAL: không cần auth ---
    "get_time": {
        "level": AuthLevel.NORMAL,
        "desc": "Hỏi giờ hiện tại, thời gian bây giờ",
        "entities": [],
        "examples": ["mấy giờ rồi", "bây giờ là mấy giờ", "giờ hiện tại"],
    },
    "get_weather": {
        "level": AuthLevel.NORMAL,
        "desc": "Hỏi về thời tiết ở một địa điểm",
        "entities": ["location"],
        "examples": ["thời tiết Hà Nội hôm nay", "trời ở Sài Gòn thế nào"],
    },
    "tell_joke": {
        "level": AuthLevel.NORMAL,
        "desc": "Yêu cầu kể một câu chuyện cười",
        "entities": [],
        "examples": ["kể chuyện cười đi", "kể một câu cười"],
    },
    "general_question": {
        "level": AuthLevel.NORMAL,
        "desc": "Câu hỏi kiến thức chung không thuộc các intent khác",
        "entities": ["query"],
        "examples": ["thủ đô nước Pháp là gì", "ai phát minh ra điện thoại"],
    },

    # --- IMPORTANT: cần SV ---
    "read_notes": {
        "level": AuthLevel.IMPORTANT,
        "desc": "Đọc ghi chú/nhật ký cá nhân của user",
        "entities": [],
        "examples": ["đọc ghi chú của tôi", "mở nhật ký của tôi"],
    },
    "send_email": {
        "level": AuthLevel.IMPORTANT,
        "desc": "Gửi email cho ai đó",
        "entities": ["recipient", "content"],
        "examples": ["gửi email cho sếp", "soạn mail báo cáo"],
    },
    "check_balance": {
        "level": AuthLevel.IMPORTANT,
        "desc": "Kiểm tra số dư tài khoản (mô phỏng)",
        "entities": [],
        "examples": ["số dư tài khoản của tôi", "tôi còn bao nhiêu tiền"],
    },
    "delete_data": {
        "level": AuthLevel.IMPORTANT,
        "desc": "Xóa dữ liệu, file, hoặc thông tin nhạy cảm",
        "entities": ["target"],
        "examples": ["xóa file báo cáo", "xóa hết ghi chú"],
    },
    "open_files": {
        "level": AuthLevel.IMPORTANT,
        "desc": "Mở, xem danh sách file cá nhân đã lưu trữ",
        "entities": ["filename"],
        "examples": ["mở file của tôi", "xem file cá nhân", "cho tôi xem file", "danh sách file"],
    },

    # --- PERSONAL: cần SID để cá nhân hóa ---
    "greet": {
        "level": AuthLevel.PERSONAL,
        "desc": "Lời chào — phản hồi sẽ chào theo tên người nói",
        "entities": [],
        "examples": ["chào bạn", "hello", "xin chào"],
    },
    "play_music": {
        "level": AuthLevel.PERSONAL,
        "desc": "Phát nhạc — chọn theo gu nhạc đã lưu trong preferences của user",
        "entities": ["genre"],
        "examples": ["mở nhạc đi", "phát nhạc cho tôi nghe", "bật nhạc rock"],
    },
    "show_schedule": {
        "level": AuthLevel.PERSONAL,
        "desc": "Hiển thị lịch/nhắc việc cá nhân của người đang nói",
        "entities": ["date"],
        "examples": ["lịch hôm nay của tôi", "tôi có việc gì hôm nay"],
    },

    # --- Fallback ---
    "unknown": {
        "level": AuthLevel.NORMAL,
        "desc": "Không hiểu ý định, hoặc không thuộc intent nào ở trên",
        "entities": [],
        "examples": [],
    },
}


def get_auth_level(intent_name: str) -> AuthLevel:
    if intent_name not in INTENTS:
        return AuthLevel.NORMAL
    return INTENTS[intent_name]["level"]
