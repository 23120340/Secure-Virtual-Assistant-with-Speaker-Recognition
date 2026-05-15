"""Config tập trung cho hệ thống."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

# ==========================================================================
# Paths
# ==========================================================================
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# Speaker encoder checkpoint (từ Tuần 1)
SPEAKER_CKPT = os.getenv("SPEAKER_CKPT", str(ROOT / "checkpoints" / "best_model.pt"))

# Database
DB_PATH = DATA_DIR / "users.db"
ENROLL_AUDIO_DIR = DATA_DIR / "enroll_audio"
ENROLL_AUDIO_DIR.mkdir(exist_ok=True)
USER_FILES_DIR = DATA_DIR / "user_files"
USER_FILES_DIR.mkdir(exist_ok=True)

# ==========================================================================
# Audio
# ==========================================================================
SAMPLE_RATE = 16000
RECORD_DURATION = 5.0       # giây mỗi lần record (cho command)
ENROLL_DURATION = 4.0       # giây mỗi mẫu enroll
ENROLL_NUM_SAMPLES = 5      # số mẫu mỗi user khi đăng ký

# Min audio length sau VAD trim, theo từng context:
#   - Enroll : cần chất lượng cao để build centroid → 1.0s
#   - Turn   : chỉ cần đủ để identify → 0.5s
MIN_AUDIO_SEC_ENROLL = 1.0
MIN_AUDIO_SEC_TURN   = 0.5

# ==========================================================================
# Speaker recognition thresholds
# ==========================================================================
# Cosine similarity để verify (cùng người). Calibrate lại sau khi có data thật.
# Own ECAPA (Tuần 1) và SpeechBrain pretrained (VoxCeleb2) có phân phối điểm
# khác nhau → cần threshold riêng cho mỗi model.
SV_THRESHOLD_OWN = float(os.getenv("SV_THRESHOLD_OWN", "0.45"))
SV_THRESHOLD_PRETRAINED = float(os.getenv("SV_THRESHOLD_PRETRAINED", "0.25"))

SID_MIN_THRESHOLD_OWN = float(os.getenv("SID_MIN_THRESHOLD_OWN", "0.35"))
SID_MIN_THRESHOLD_PRETRAINED = float(os.getenv("SID_MIN_THRESHOLD_PRETRAINED", "0.20"))

# Default (giữ cho backward compat với code chưa biết mode) — ưu tiên own.
SV_THRESHOLD = SV_THRESHOLD_OWN
SID_MIN_THRESHOLD = SID_MIN_THRESHOLD_OWN

# Top1 − Top2 margin tối thiểu để chấp nhận IDENTIFY cho intent IMPORTANT.
# Nếu top1 và top2 sát nhau → có khả năng 2 user giọng giống → block IMPORTANT.
# (Chống case kẻ giả giọng đủ giỏi để Top1 vượt threshold nhưng vẫn cạnh tranh với
# user thật khác.)
SID_MIN_MARGIN = float(os.getenv("SID_MIN_MARGIN", "0.08"))

# ==========================================================================
# ASR
# ==========================================================================
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")  # tiny/base/small/medium/large-v3 — base nhanh cho CPU int8
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "int8")  # int8 cho CPU, float16 cho GPU
ASR_LANGUAGE = "vi"

# Bật/tắt sửa chính tả qua Gemini sau ASR. Mặc định OFF vì tốn 500-2000ms/turn
# và gửi transcript cho Google (privacy). Có cache LRU nên 1 câu lặp lại không gọi 2 lần.
ASR_CORRECT_ENABLED = os.getenv("ASR_CORRECT", "false").lower() in ("true", "1", "yes")

# ==========================================================================
# NLU
# ==========================================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# ==========================================================================
# TTS
# ==========================================================================
TTS_LANG = "vi"

# ==========================================================================
# Google OAuth 2.0 (Gmail API)
# ==========================================================================
GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
# Phải khớp chính xác với URI đã đăng ký trong Google Cloud Console.
# Default HTTPS vì app chạy với --ssl (cert.pem/key.pem) — Google block HTTP redirect
# trên non-localhost. Nếu chạy plain HTTP localhost, override qua env.
GOOGLE_REDIRECT_URI  = os.getenv("GOOGLE_REDIRECT_URI",
                                  "https://localhost:5000/auth/google/callback")

# ==========================================================================
# Admin
# ==========================================================================
ADMIN_PASS = os.getenv("ADMIN_PASS", "")
