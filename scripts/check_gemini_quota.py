"""Gửi 1 request thử tới Gemini để biết quota còn hay hết.

Chạy:  ./venv/Scripts/python.exe scripts/check_gemini_quota.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from core import config  # noqa: E402 — config tự load .env


def main() -> int:
    key = config.GEMINI_API_KEY
    if not key:
        print("GEMINI_API_KEY rỗng → app đang chạy rule-based, không gọi Gemini.")
        return 1
    print(f"Key đuôi : ...{key[-8:]}")
    print(f"Model    : {config.GEMINI_MODEL}")

    from google import genai

    client = genai.Client(api_key=key)
    try:
        resp = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents="Trả lời đúng 1 từ: ok",
        )
        print("\n[OK] Quota CÒN. Gemini trả lời:", (resp.text or "").strip())
        return 0
    except Exception as e:  # noqa: BLE001 — cần xem full message để phân loại
        detail = str(e)
        print("\n[LỖI]", detail)
        low = detail.lower()
        if "429" in detail or "resource_exhausted" in low or "quota" in low:
            if "perday" in low.replace(" ", "") or "per day" in low or "/d" in low:
                print(">> Hết quota theo NGÀY (RPD). Đợi reset (trưa-chiều VN) hoặc đổi key tài khoản khác / đổi GEMINI_MODEL.")
            elif "perminute" in low.replace(" ", "") or "per minute" in low:
                print(">> Chỉ vượt giới hạn THEO PHÚT (RPM). Đợi ~60s rồi chạy lại là hết.")
            else:
                print(">> 429 nhưng không rõ phút/ngày. Đợi 60s thử lại; nếu vẫn lỗi là hết theo ngày.")
        elif "api_key_invalid" in low or "api key not valid" in low or "permission" in low:
            print(">> Key SAI hoặc bị thu hồi, không phải hết quota.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
