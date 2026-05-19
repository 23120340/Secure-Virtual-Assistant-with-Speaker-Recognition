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


# Mỗi intent: name, level, description (cho LLM hiểu), entities cần extract,
# examples (paraphrase đa dạng để LLM generalize), counter_examples (case
# dễ nhầm sang intent khác — giúp LLM phân biệt biên).
INTENTS = {
    # --- NORMAL: không cần auth ---
    "get_time": {
        "level": AuthLevel.NORMAL,
        "desc": "Hỏi giờ hiện tại, thời gian bây giờ trong ngày",
        "entities": [],
        "examples": [
            "mấy giờ rồi",
            "bây giờ là mấy giờ",
            "giờ hiện tại",
            "cho tôi xem giờ",
            "đồng hồ bao nhiêu",
            "giờ này là mấy giờ rồi nhỉ",
            "nói tôi nghe mấy giờ",
            "hiện giờ là mấy giờ",
            "mấy giờ rồi bạn ơi",
            "bây giờ mấy giờ rồi vậy",
        ],
        "counter_examples": ["lịch hôm nay", "tôi có lịch gì"],
    },
    "get_weather": {
        "level": AuthLevel.NORMAL,
        "desc": "Hỏi về thời tiết, nhiệt độ, mưa nắng ở một địa điểm",
        "entities": ["location"],
        "examples": [
            "thời tiết Hà Nội hôm nay",
            "trời ở Sài Gòn thế nào",
            "hôm nay Đà Nẵng có mưa không",
            "nhiệt độ ngoài trời bao nhiêu",
            "ở Hà Nội đang nắng hay mưa",
            "dự báo thời tiết hôm nay đi",
            "thời tiết hôm nay ra sao",
            "có mưa không bạn",
            "trời nóng không nhỉ",
            "ngày mai trời thế nào",
            "cho tôi xem thời tiết Cần Thơ",
        ],
        "counter_examples": ["trời ơi nóng quá", "tôi mệt"],
    },
    "tell_joke": {
        "level": AuthLevel.NORMAL,
        "desc": "Yêu cầu kể một câu chuyện cười, mẩu hài, để giải trí",
        "entities": [],
        "examples": [
            "kể chuyện cười đi",
            "kể một câu cười",
            "có chuyện cười nào không",
            "kể tôi nghe gì vui đi",
            "nói gì hài hài đi",
            "cho tôi cười với",
            "tấu hài đi bạn",
            "kể joke nào",
            "có gì hài không kể đi",
        ],
        "counter_examples": ["chuyện hôm nay thế nào", "kể cho tôi nghe lịch"],
    },
    "general_question": {
        "level": AuthLevel.NORMAL,
        "desc": "Câu hỏi kiến thức chung, định nghĩa, sự kiện, lời khuyên — "
                "KHÔNG khớp với bất kỳ intent cụ thể nào ở trên/dưới",
        "entities": ["query"],
        "examples": [
            "thủ đô nước Pháp là gì",
            "ai phát minh ra điện thoại",
            "Python là ngôn ngữ gì",
            "lịch sử Việt Nam bắt đầu từ khi nào",
            "làm thế nào để học tiếng Anh nhanh",
            "công thức nấu phở như thế nào",
            "kể tôi nghe về Albert Einstein",
            "giải thích cho tôi blockchain là gì",
            "1 năm có bao nhiêu ngày",
        ],
        "counter_examples": [
            "mấy giờ rồi", "thời tiết Hà Nội", "đọc ghi chú", "phát nhạc",
        ],
    },

    # --- IMPORTANT: cần SV ---
    "read_notes": {
        "level": AuthLevel.IMPORTANT,
        "desc": "Đọc ghi chú/nhật ký cá nhân ĐÃ LƯU của user (read-only)",
        "entities": [],
        "examples": [
            "đọc ghi chú của tôi",
            "mở nhật ký của tôi",
            "đọc nhật ký",
            "xem ghi chú đã lưu",
            "cho tôi nghe ghi chú",
            "đọc note của tôi đi",
            "có ghi chú gì không",
            "ghi chú mới nhất là gì",
            "lấy nhật ký ra đọc",
        ],
        "counter_examples": [
            "xóa ghi chú", "xóa nhật ký", "ghi chú mới", "viết ghi chú",
        ],
    },
    "send_email": {
        "level": AuthLevel.IMPORTANT,
        "desc": "Soạn và gửi email cho người khác",
        "entities": ["recipient", "recipient_email", "subject", "body", "content"],
        "examples": [
            "gửi email cho sếp",
            "soạn mail báo cáo",
            "viết mail cho anh Tuấn",
            "gửi email tới lan@gmail.com",
            "mail cho mẹ nói tối nay con về muộn",
            "gửi thư điện tử cho khách hàng",
            "soạn email mới",
            "tạo email gửi đội nhóm",
            "viết email nội dung họp 9h sáng mai",
            "gửi mail đến địa chỉ abc@example.com",
        ],
        "counter_examples": [
            "đọc email", "kiểm tra hộp thư", "xóa email",
        ],
    },
    "check_balance": {
        "level": AuthLevel.IMPORTANT,
        "desc": "Kiểm tra số dư tài khoản ngân hàng/ví của user (mô phỏng)",
        "entities": [],
        "examples": [
            "số dư tài khoản của tôi",
            "tôi còn bao nhiêu tiền",
            "kiểm tra số dư",
            "tài khoản tôi còn bao nhiêu",
            "số tiền trong ví",
            "balance tôi đang là bao nhiêu",
            "trong tài khoản của tôi còn lại gì",
            "cho tôi biết số dư đi",
            "tôi còn dư bao nhiêu trong tài khoản",
        ],
        "counter_examples": ["nạp tiền vào tài khoản", "chuyển tiền cho ai đó"],
    },
    "delete_data": {
        "level": AuthLevel.IMPORTANT,
        "desc": "Xóa dữ liệu cá nhân: ghi chú, lịch, preferences, thông tin — "
                "KHÔNG bao gồm xóa file (đó là open_files, vì xóa file thực hiện "
                "trong panel files)",
        "entities": ["target"],
        "examples": [
            "xóa hết ghi chú",
            "xóa nhật ký của tôi",
            "xóa lịch hôm nay",
            "xóa dữ liệu cá nhân",
            "xóa tất cả thông tin",
            "xoá ghi chú đi",
            "dọn sạch lịch của tôi",
            "xoá hết preferences",
            "xóa thông tin của tôi khỏi hệ thống",
        ],
        "counter_examples": [
            "xóa file", "xoá file báo cáo", "xóa ảnh", "delete file",
        ],
    },
    "open_files": {
        "level": AuthLevel.IMPORTANT,
        "desc": "Mở, xem, hoặc thao tác (gồm cả XÓA) trên file cá nhân đã upload",
        "entities": ["filename"],
        "examples": [
            "mở file của tôi",
            "xem file cá nhân",
            "cho tôi xem file",
            "danh sách file",
            "xóa file báo cáo",
            "xoá file ảnh",
            "mở thư mục file",
            "file của tôi đâu",
            "tôi có file nào",
            "vào xem các file đã lưu",
            "download file về máy",
        ],
        "counter_examples": [
            "đọc ghi chú", "xóa ghi chú",
        ],
    },
    "add_note": {
        "level": AuthLevel.IMPORTANT,
        "desc": "Thêm 1 ghi chú/nhật ký mới vào danh sách của user. "
                "PHẢI có nội dung — không phải đọc, không phải xoá",
        "entities": ["content"],
        "examples": [
            "ghi chú họp 9h sáng mai",
            "thêm ghi chú mua sữa",
            "tạo note nhớ gọi mẹ",
            "lưu ghi chú là sinh nhật Lan ngày 15",
            "thêm vào danh sách: đi khám răng",
            "note lại: nộp báo cáo trước thứ 6",
            "viết ghi chú mới: học bài chương 3",
            "thêm 1 ghi chú: chuẩn bị slide",
            "thêm note hôm nay phải hoàn thành đồ án",
        ],
        "counter_examples": [
            "đọc ghi chú", "xóa ghi chú", "ghi chú của tôi là gì",
        ],
    },
    "add_schedule": {
        "level": AuthLevel.IMPORTANT,
        "desc": "Thêm 1 mục lịch/cuộc hẹn vào schedule của user. Phải có "
                "nội dung và (tuỳ chọn) thời gian",
        "entities": ["content", "time"],
        "examples": [
            "thêm lịch họp lúc 9 giờ",
            "đặt lịch khám răng ngày mai",
            "thêm vào lịch: đi siêu thị tối nay",
            "lên lịch sinh nhật Lan ngày 15",
            "ghi vào lịch hôm nay: meeting 14h",
            "thêm cuộc hẹn với khách lúc 10h",
            "tạo lịch học bài 8 giờ tối",
            "đặt nhắc 3 giờ chiều có khách",
            "thêm task vào lịch: review code",
        ],
        "counter_examples": [
            "lịch hôm nay", "xóa lịch", "tôi có việc gì",
        ],
    },
    "add_contact": {
        "level": AuthLevel.IMPORTANT,
        "desc": "Thêm 1 contact (tên + email) vào danh bạ để dùng cho gửi email. "
                "Phải có cả tên và địa chỉ email",
        "entities": ["name", "email"],
        "examples": [
            "thêm liên hệ Tuấn email tuan@example.com",
            "lưu contact Lan với mail lan@gmail.com",
            "thêm anh Hùng số mail hung@company.vn vào danh bạ",
            "tạo contact mới: Minh, minh@gmail.com",
            "thêm vào danh bạ: Mai mail mai@yahoo.com",
            "lưu số mail của Phong là phong@example.org",
            "thêm liên hệ mới tên Quân email quan@gmail.com",
        ],
        "counter_examples": [
            "gửi email cho Tuấn", "xoá Tuấn khỏi danh bạ",
        ],
    },

    # --- PERSONAL: cần SID để cá nhân hóa ---
    "greet": {
        "level": AuthLevel.PERSONAL,
        "desc": "Lời chào hỏi xã giao — phản hồi sẽ chào theo tên người nói",
        "entities": [],
        "examples": [
            "chào bạn",
            "hello",
            "xin chào",
            "hi",
            "chào buổi sáng",
            "chào trợ lý",
            "alo alo",
            "hey",
            "chào bạn nhé",
            "chào hệ thống",
        ],
        "counter_examples": ["tạm biệt", "kết thúc"],
    },
    "play_music": {
        "level": AuthLevel.PERSONAL,
        "desc": "Phát nhạc — chọn theo gu nhạc đã lưu trong preferences của user",
        "entities": ["genre"],
        "examples": [
            "mở nhạc đi",
            "phát nhạc cho tôi nghe",
            "bật nhạc rock",
            "cho tôi nghe nhạc",
            "nghe một bài nhạc",
            "play music đi",
            "bật bài hát ballad",
            "mở playlist của tôi",
            "phát nhạc EDM",
            "nghe v-pop một bài",
            "mở nhạc nhẹ thư giãn",
        ],
        "counter_examples": [
            "tắt nhạc", "dừng nhạc", "kể chuyện cười",
        ],
    },
    "show_schedule": {
        "level": AuthLevel.PERSONAL,
        "desc": "Hiển thị lịch/nhắc việc cá nhân của người đang nói",
        "entities": ["date"],
        "examples": [
            "lịch hôm nay của tôi",
            "tôi có việc gì hôm nay",
            "lịch của tôi đâu",
            "có nhắc việc nào không",
            "kiểm tra lịch ngày mai",
            "tuần này tôi có hẹn gì",
            "lịch tuần của tôi",
            "tôi có cuộc họp nào hôm nay không",
            "xem todo list",
        ],
        "counter_examples": [
            "mấy giờ rồi", "xóa lịch",
        ],
    },

    # --- Fallback ---
    "unknown": {
        "level": AuthLevel.NORMAL,
        "desc": "Không hiểu ý định, hoặc không thuộc intent nào ở trên",
        "entities": [],
        "examples": [],
    },
    # Internal synthetic state used by web email composer flow. It is intentionally
    # hidden from LLM/tool declarations and should never be predicted from user text.
    "email_flow": {
        "level": AuthLevel.IMPORTANT,
        "desc": "Internal email composition flow state",
        "entities": [],
        "examples": [],
        "internal": True,
    },
}


def get_auth_level(intent_name: str) -> AuthLevel:
    if intent_name not in INTENTS:
        return AuthLevel.NORMAL
    return INTENTS[intent_name]["level"]
