"""Enroll user mới qua CLI.

Cách dùng:
    python scripts/enroll_user.py --user_id minh --name "Nguyễn Văn Minh"
    # → record 5 lần × 4s, lưu vào DB

Hoặc dùng file wav có sẵn:
    python scripts/enroll_user.py --user_id minh --name "Minh" \
        --audio_files sample1.wav sample2.wav sample3.wav
"""
import sys
import argparse
import json
from pathlib import Path

# Thêm parent dir vào path để import core/
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import audio_io, config
from core.database import UserDB, SpeakerManager


ENROLL_PROMPTS = [
    "Trợ lý ơi, hôm nay thời tiết thế nào?",
    "Phát một bản nhạc tôi yêu thích đi.",
    "Cho tôi xem lịch làm việc hôm nay.",
    "Đọc cho tôi ghi chú cuối tuần.",
    "Một hai ba bốn năm sáu bảy tám chín mười.",
]


def enroll_via_mic(user_id: str, name: str, num_samples: int):
    print(f"\n=== Đăng ký user: {name} ({user_id}) ===")
    print(f"Sẽ thu {num_samples} mẫu giọng nói, mỗi mẫu {config.ENROLL_DURATION}s.")
    print("Hãy đọc từng câu prompt khi được yêu cầu.\n")

    audios = []
    user_audio_dir = config.ENROLL_AUDIO_DIR / user_id
    user_audio_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_samples):
        prompt = ENROLL_PROMPTS[i % len(ENROLL_PROMPTS)]
        print(f"\n[{i+1}/{num_samples}] Đọc câu: \"{prompt}\"")
        input("        Nhấn Enter khi sẵn sàng...")
        audio = audio_io.record_and_trim(
            config.ENROLL_DURATION,
            save_path=user_audio_dir / f"sample_{i+1}.wav",
        )
        if audio.size < config.SAMPLE_RATE:  # < 1s sau trim
            print(f"        ⚠ Audio quá ngắn sau VAD ({audio.size/config.SAMPLE_RATE:.2f}s)."
                  " Hãy thử lại.")
            i -= 1
            continue
        audios.append(audio)
        print(f"        ✓ Recorded ({audio.size/config.SAMPLE_RATE:.2f}s)")

    return audios


def enroll_via_files(audio_files: list):
    audios = []
    for f in audio_files:
        path = Path(f)
        if not path.exists():
            raise FileNotFoundError(path)
        audio = audio_io.load_wav(path)
        audio = audio_io.SileroVAD.trim(audio)
        audios.append(audio)
        print(f"  ✓ Loaded {path.name} ({audio.size/config.SAMPLE_RATE:.2f}s)")
    return audios


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--user_id", required=True, help="ID duy nhất, không dấu, không space")
    p.add_argument("--name", required=True, help="Tên hiển thị")
    p.add_argument("--num_samples", type=int, default=config.ENROLL_NUM_SAMPLES)
    p.add_argument("--audio_files", nargs="*", default=None,
                   help="Nếu có thì dùng file thay vì record")
    p.add_argument("--preferences", default="{}",
                   help='JSON string, ví dụ \'{"favorite_genre":"rock","balance":5000000}\'')
    p.add_argument("--preferences_file", default=None,
                   help="Đường dẫn tới file .json chứa preferences (thay thế --preferences)")
    args = p.parse_args()

    # Validate
    if " " in args.user_id:
        raise ValueError("user_id không được chứa khoảng trắng")

    db = UserDB()
    spk_mgr = SpeakerManager(db)

    if db.get_user(args.user_id):
        print(f"User '{args.user_id}' đã tồn tại. Xóa trước nếu muốn đăng ký lại.")
        return

    # Lấy audio
    if args.audio_files:
        audios = enroll_via_files(args.audio_files)
    else:
        audios = enroll_via_mic(args.user_id, args.name, args.num_samples)

    if len(audios) < 2:
        print("Cần ít nhất 2 mẫu audio để tạo embedding ổn định.")
        return

    # Enroll
    if args.preferences_file:
        with open(args.preferences_file, encoding="utf-8") as f:
            prefs = json.load(f)
    else:
        prefs = json.loads(args.preferences)
    centroid = spk_mgr.enroll(args.user_id, args.name, audios, prefs)
    print(f"\n✅ Đăng ký thành công user '{args.name}' với centroid 192-d "
          f"(norm={float((centroid**2).sum()**0.5):.4f})")
    print(f"   Audio mẫu lưu tại: {config.ENROLL_AUDIO_DIR / args.user_id}")
    print(f"   Database: {db.db_path}")


if __name__ == "__main__":
    main()
