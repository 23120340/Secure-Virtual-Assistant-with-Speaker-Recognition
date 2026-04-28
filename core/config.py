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

# ==========================================================================
# Speaker recognition thresholds
# ==========================================================================
# Cosine similarity để verify (cùng người). Calibrate lại sau khi có data thật.
SV_THRESHOLD = 0.45          # > threshold → coi như cùng người
SID_MIN_THRESHOLD = 0.35     # nhỏ hơn → coi là "guest"

# ==========================================================================
# ASR
# ==========================================================================
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")  # tiny/base/small/medium
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "int8")  # int8 cho CPU, float16 cho GPU
ASR_LANGUAGE = "vi"

# ==========================================================================
# NLU
# ==========================================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# ==========================================================================
# TTS
# ==========================================================================
TTS_LANG = "vi"
