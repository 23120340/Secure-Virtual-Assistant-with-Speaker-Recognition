"""Config tập trung cho hệ thống."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

# Quiet Windows/HuggingFace startup noise:
# - SpeechBrain/HF cache may warn about symlinks on Windows; we use COPY where
#   needed, so the warning is not actionable for this app.
# - Transformers 5.x can spawn a background safetensors-conversion thread for
#   .bin-only models such as PhoWhisper. Some repos disable discussions, which
#   causes a noisy 403 traceback even though model loading succeeds.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

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
ENROLL_DURATION = float(os.getenv("ENROLL_DURATION", "5.0"))   # ↑ 4→5s: centroid ổn định hơn
ENROLL_NUM_SAMPLES = int(os.getenv("ENROLL_NUM_SAMPLES", "5"))

# Min audio length sau VAD trim, theo từng context:
#   - Enroll : cần chất lượng cao để build centroid → 1.0s
#   - Turn   : chỉ cần đủ để identify → 0.5s
#   - Confirm: bước xác nhận email chỉ nói từ rất ngắn ("có"/"không") → 0.2s,
#              nếu giữ 0.5s thì "có" sau VAD thường < ngưỡng và bị loại oan.
MIN_AUDIO_SEC_ENROLL  = 1.0
MIN_AUDIO_SEC_TURN    = 0.5
MIN_AUDIO_SEC_CONFIRM = float(os.getenv("MIN_AUDIO_SEC_CONFIRM", "0.2"))

# ==========================================================================
# Speaker model backend
# ==========================================================================
# Backend chọn encoder cho speaker:
#   "ecapa" (default): ECAPA-TDNN từ SpeechBrain hoặc own checkpoint Tuần 1 (192-d)
#   "wavlm" (future, session 3): WavLM-SV pretrained của Microsoft (768-d/1024-d)
#       Cần schema migration vì embedding dim khác.
SPEAKER_BACKEND = os.getenv("SPEAKER_BACKEND", "ecapa").lower()

# ==========================================================================
# Speaker recognition thresholds
# ==========================================================================
# Cosine similarity để verify (cùng người). Calibrate lại sau khi có data thật.
# Own ECAPA (Tuần 1) và SpeechBrain pretrained (VoxCeleb2) có phân phối điểm
# khác nhau → cần threshold riêng cho mỗi model.
SV_THRESHOLD_OWN = float(os.getenv("SV_THRESHOLD_OWN", "0.45"))
SV_THRESHOLD_PRETRAINED = float(os.getenv("SV_THRESHOLD_PRETRAINED", "0.25"))
# WavLM-base-plus-sv: x-vector embedding cosine concentrate ở mức cao (~0.9+
# cho cả same lẫn different speaker khi audio cùng domain). Calibrated trên
# 5 user × 5 audio nội bộ (script: reenroll_backend.py + benchmark):
#   target mean=0.96, std=0.02, min=0.92
#   non-target mean=0.70, std=0.18, max=0.97
#   optimal threshold ~0.95
# Chọn 0.93 = min_target - 1 std margin để cân false-reject/false-accept.
# Re-calibrate khi có nhiều user hơn hoặc môi trường record khác.
SV_THRESHOLD_WAVLM = float(os.getenv("SV_THRESHOLD_WAVLM", "0.93"))

SID_MIN_THRESHOLD_OWN = float(os.getenv("SID_MIN_THRESHOLD_OWN", "0.35"))
SID_MIN_THRESHOLD_PRETRAINED = float(os.getenv("SID_MIN_THRESHOLD_PRETRAINED", "0.20"))
SID_MIN_THRESHOLD_WAVLM = float(os.getenv("SID_MIN_THRESHOLD_WAVLM", "0.90"))

# WavLM model name (chỉ dùng khi SPEAKER_BACKEND=wavlm)
WAVLM_MODEL  = os.getenv("WAVLM_MODEL", "microsoft/wavlm-base-plus-sv")
WAVLM_DEVICE = os.getenv("WAVLM_DEVICE", "")  # rỗng = auto

# Default (giữ cho backward compat với code chưa biết mode) — ưu tiên own.
SV_THRESHOLD = SV_THRESHOLD_OWN
SID_MIN_THRESHOLD = SID_MIN_THRESHOLD_OWN

# Top1 − Top2 margin tối thiểu để chấp nhận IDENTIFY cho intent IMPORTANT.
# Nếu top1 và top2 sát nhau → có khả năng 2 user giọng giống → block IMPORTANT.
# (Chống case kẻ giả giọng đủ giỏi để Top1 vượt threshold nhưng vẫn cạnh tranh với
# user thật khác.)
SID_MIN_MARGIN = float(os.getenv("SID_MIN_MARGIN", "0.08"))

# ==========================================================================
# Speaker pipeline improvements (đều có thể bật/tắt qua env)
# ==========================================================================
# Multi-window embedding: chia audio đủ dài thành N cửa sổ chồng lấp, encode
# từng cửa sổ rồi average → robust hơn 1-pass single encode. Audio quá ngắn tự
# fallback về single encode → an toàn.
SPEAKER_MULTIWINDOW = os.getenv("SPEAKER_MULTIWINDOW", "true").lower() in ("true", "1", "yes")
SPEAKER_WINDOW_SEC  = float(os.getenv("SPEAKER_WINDOW_SEC", "2.0"))
SPEAKER_N_WINDOWS   = int(os.getenv("SPEAKER_N_WINDOWS", "3"))

# AS-Norm-lite: z-normalize raw cosine score theo cohort các user khác trong DB.
# Auto-disable nếu cohort < SPEAKER_ASNORM_MIN_COHORT → tránh statistics không ổn định.
SPEAKER_ASNORM             = os.getenv("SPEAKER_ASNORM", "true").lower() in ("true", "1", "yes")
SPEAKER_ASNORM_MIN_COHORT  = int(os.getenv("SPEAKER_ASNORM_MIN_COHORT", "4"))
SPEAKER_ASNORM_TOP_K       = int(os.getenv("SPEAKER_ASNORM_TOP_K", "5"))
# Z-score threshold cho AS-Norm. Score đã chuẩn hóa → ngưỡng tuyệt đối, không phụ
# thuộc raw cosine distribution. ~2.0 = "vượt cohort trung bình 2 std" — strict.
SPEAKER_ASNORM_Z_THRESHOLD = float(os.getenv("SPEAKER_ASNORM_Z_THRESHOLD", "2.0"))

# Audio pre-processing trước khi encode: DC-removal, RMS-normalize.
# Lưu ý: nếu đổi sang True sau khi đã enroll user, có thể mismatch với centroid cũ
# (centroid built từ audio chưa preprocess). Re-enroll khi đổi flag để chắc chắn.
SPEAKER_PREPROCESS = os.getenv("SPEAKER_PREPROCESS", "false").lower() in ("true", "1", "yes")

# Denoise bằng spectral gating (noisereduce nếu có, else fallback no-op).
# Cũng phải re-enroll khi đổi flag — embedding của audio noise vs denoised khác nhau.
SPEAKER_DENOISE = os.getenv("SPEAKER_DENOISE", "false").lower() in ("true", "1", "yes")

# ==========================================================================
# ASR
# ==========================================================================
# Backend chọn engine ASR:
#   "faster-whisper" (default): faster-whisper với WHISPER_MODEL = "base"/"small"/...
#       hoặc đường dẫn local đến CT2-converted model (vd: PhoWhisper-ct2).
#   "phowhisper": transformers pipeline với vinai/PhoWhisper-{tiny,base,small,medium,large}.
#       Cài thêm: pip install transformers (chưa có trong requirements).
ASR_BACKEND = os.getenv("ASR_BACKEND", "faster-whisper").lower()

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")  # tiny/base/small/medium/large-v3 — base nhanh cho CPU int8
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "int8")  # int8 cho CPU, float16 cho GPU
ASR_LANGUAGE = "vi"

# PhoWhisper-specific (chỉ dùng khi ASR_BACKEND=phowhisper).
# Default 'base' — CPU-friendly (~1× realtime). Trên CPU 'small' khoảng 3× realtime,
# quá chậm cho virtual assistant. Có GPU thì đổi 'small' hoặc 'medium' qua env.
PHOWHISPER_MODEL  = os.getenv("PHOWHISPER_MODEL", "vinai/PhoWhisper-base")
PHOWHISPER_DEVICE = os.getenv("PHOWHISPER_DEVICE", "")  # rỗng = auto (cuda → cpu)

# Bật/tắt sửa chính tả qua Gemini sau ASR. Mặc định OFF vì tốn 500-2000ms/turn
# và gửi transcript cho Google (privacy). Có cache LRU nên 1 câu lặp lại không gọi 2 lần.
ASR_CORRECT_ENABLED = os.getenv("ASR_CORRECT", "false").lower() in ("true", "1", "yes")

# Layer-1.5: fuzzy snap. Nếu cả câu (đã chuẩn hoá, bỏ dấu) gần khớp một câu lệnh
# chuẩn với độ tương đồng >= ngưỡng → nắn về câu lệnh đó. Bù cho việc rule khớp
# chính xác không phủ hết mọi lỗi ASR thật. Offline, không cần Gemini.
ASR_FUZZY_ENABLED   = os.getenv("ASR_FUZZY", "true").lower() in ("true", "1", "yes")
ASR_FUZZY_THRESHOLD = float(os.getenv("ASR_FUZZY_THRESHOLD", "0.86"))
ASR_FUZZY_MAX_WORDS = int(os.getenv("ASR_FUZZY_MAX_WORDS", "8"))

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

# Calendar integration (opt-in): khi True, OAuth flow thêm scope
# `calendar.events.readonly` → user phải re-consent. `show_schedule` handler
# sẽ merge sự kiện Google Calendar hôm nay với `preferences.schedule` local.
# Mặc định False — chỉ bật khi đã configure Google Cloud Console project
# để enable Calendar API + add scope vào OAuth consent screen.
ENABLE_CALENDAR_INTEGRATION = os.getenv(
    "ENABLE_CALENDAR_INTEGRATION", "false"
).lower() in ("true", "1", "yes")

# ==========================================================================
# Admin
# ==========================================================================
ADMIN_PASS = os.getenv("ADMIN_PASS", "")

# ==========================================================================
# Auth policy
# ==========================================================================
# Text-mode IMPORTANT: cho phép password thay thế Speaker Verification.
# Mặc định OFF — text-mode block IMPORTANT (chỉ voice mới qua được SV gate).
# Bật khi cần demo flow hoặc khi user gặp lỗi mic (best-effort 2FA fallback).
# CẢNH BÁO: password fallback KHÔNG phải biometric — chỉ là "know factor".
# Threat model phải document rõ điều này trong báo cáo.
ALLOW_PASSWORD_FOR_IMPORTANT = os.getenv(
    "ALLOW_PASSWORD_FOR_IMPORTANT", "false"
).lower() in ("true", "1", "yes")

# Challenge-response cho IMPORTANT voice flow:
# Khi bật, sau khi SID+SV pass, server KHÔNG dispatch handler ngay — thay vào
# đó sinh phrase random và yêu cầu user đọc lại. Server verify cả ASR-match
# lẫn SV trên audio thứ 2 → chống replay attack đơn giản.
# Mặc định OFF cho demo nhanh; bật cho production / báo cáo bảo mật.
CHALLENGE_RESPONSE_ENABLED = os.getenv(
    "CHALLENGE_RESPONSE_ENABLED", "false"
).lower() in ("true", "1", "yes")

# Số từ trong phrase challenge (mỗi từ từ pool 16 → entropy ~4 bit/từ).
CHALLENGE_PHRASE_LEN = int(os.getenv("CHALLENGE_PHRASE_LEN", "4"))

# Số lần cho phép đọc lại cụm từ challenge khi ASR nghe chưa khớp (trong TTL).
# Sai SV (giọng) là block ngay; chỉ lỗi đọc cụm từ mới được thử lại.
CHALLENGE_MAX_ATTEMPTS = int(os.getenv("CHALLENGE_MAX_ATTEMPTS", "4"))

# Incremental re-enroll sau khi SV pass: cập nhật centroid của user 1 ít theo
# audio mới (chống speaker drift do tuổi / mic / nhiễu thay đổi theo thời gian).
# alpha nhỏ → mỗi lần update chỉ "kéo" centroid alpha% về phía audio mới.
# Mặc định OFF — bật khi demo thấy SV score của user thật giảm dần theo phiên.
SPEAKER_INCREMENTAL_REENROLL = os.getenv(
    "SPEAKER_INCREMENTAL_REENROLL", "false"
).lower() in ("true", "1", "yes")
SPEAKER_INCREMENTAL_ALPHA = float(os.getenv("SPEAKER_INCREMENTAL_ALPHA", "0.1"))

# Encrypt biometric WAV at-rest: khi true, file mẫu giọng `data/enroll_audio/`
# được lưu dưới dạng `<file>.wav.enc` (Fernet ciphertext) thay vì plaintext WAV.
# Tránh leak audio gốc qua filesystem backup / DB dump.
# Mặc định OFF cho backward compat — bật cần migration (encrypt file cũ).
# Key derivation cùng nguồn với OAuth token (TOKEN_ENCRYPTION_KEY → FLASK_SECRET).
ENCRYPT_BIOMETRIC_WAV = os.getenv(
    "ENCRYPT_BIOMETRIC_WAV", "false"
).lower() in ("true", "1", "yes")
