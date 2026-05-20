"""Test Whisper hallucination detection — YouTube outro phrases.

Whisper được pretrain trên YouTube → khi audio im lặng/quá ngắn, sinh ra
các phrase outro hay gặp ("Hẹn gặp lại mọi người", "Cảm ơn đã xem"...).
Test này confirm pattern blocklist catch các case từ turn_log thực tế.
"""
from core.asr import (
    _is_hallucinated,
    _is_whisper_youtube_hallucination,
)


# ---- Hallucination patterns từ turn_log.jsonl thực tế ----
HALLUCINATION_CASES = [
    "Hẹn gặp lại mọi người",
    "Hẹn gặp lại mọi người.",
    "Hẹn gặp lại với mọi người.",
    "Xin chào tạm biệt và hẹn gặp lại video hôm nay",
    "Xin chào và hẹn gặp lại trong video tiếp theo",
    "Xin chào tất cả các bạn.",
    "Cảm ơn các bạn đã xem video.",
    "Cảm ơn mọi người đã theo dõi.",
    "Hẹn gặp lại các bạn lần sau",
    "Hẹn gặp lại!",
    "Hẹn gặp lại.",
    "Hẹn gặp lại",
    "Đừng quên like và đăng ký kênh",
    "Nhớ subscribe nhé",
    "Chúc các bạn một ngày tốt lành",
    # Single time-token hallucinations (Whisper sinh khi audio ngắn).
    "11h",
    "11 giờ",
    "5h",
    "30 phút",
    "2h.",
]

# Câu thực sự user nói — KHÔNG được trigger blocklist.
LEGITIMATE_CASES = [
    "đọc ghi chú của tôi",
    "gửi email cho Tuấn",
    "phát nhạc rock",
    "xin chào trợ lý",      # greet — không match outro pattern
    "chào bạn",
    "kiểm tra số dư",
    "thời tiết Hà Nội",
    "kể chuyện cười",
    "nhắc tôi 11 giờ uống thuốc",  # "11 giờ" trong câu dài — không match
    "đặt lịch họp 9h sáng mai",    # "9h" trong câu dài — không match
    "tạm biệt nhé",                 # KHÔNG có "hẹn gặp lại" — pass
]


def test_youtube_hallucinations_detected():
    """Tất cả case từ turn_log thực tế phải bị flag là hallucination."""
    failures = []
    for text in HALLUCINATION_CASES:
        if not _is_whisper_youtube_hallucination(text):
            failures.append(text)
    assert not failures, f"Không detect được hallucination: {failures}"


def test_legitimate_commands_not_flagged():
    """Câu lệnh thật của user KHÔNG được flag — tránh false positive."""
    failures = []
    for text in LEGITIMATE_CASES:
        if _is_whisper_youtube_hallucination(text):
            failures.append(text)
    assert not failures, f"False positive cho câu lệnh thật: {failures}"


def test_empty_string_not_flagged():
    assert not _is_whisper_youtube_hallucination("")
    assert not _is_whisper_youtube_hallucination("   ")


def test_is_hallucinated_combines_youtube_check():
    """_is_hallucinated (entry chính) phải gọi youtube check."""
    # n_samples=16000 ≈ 1s — đủ dài để không bị catch bởi length check.
    assert _is_hallucinated("Hẹn gặp lại mọi người.", n_samples=16000)
    assert _is_hallucinated("11h", n_samples=16000)
    assert not _is_hallucinated("đọc ghi chú", n_samples=16000)


def test_youtube_pattern_case_insensitive():
    """Pattern phải case-insensitive — Whisper có thể trả caps."""
    assert _is_whisper_youtube_hallucination("HẸN GẶP LẠI MỌI NGƯỜI")
    assert _is_whisper_youtube_hallucination("hẹn gặp lại mọi người")


def test_short_time_token_blocked():
    """Single time token (Whisper hallucination khi audio ngắn) bị block."""
    for t in ["1h", "2 h", "3h.", "5 giờ", "20 phút", "45s"]:
        assert _is_whisper_youtube_hallucination(t), f"Should block: {t!r}"
