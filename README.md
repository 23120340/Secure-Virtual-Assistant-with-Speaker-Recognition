# Secure Virtual Assistant with Speaker Recognition

Trợ lý ảo tiếng Việt **bảo mật bằng giọng nói** — kết hợp Speaker Identification (SID) và Speaker Verification (SV) dựa trên ECAPA-TDNN để phân quyền truy cập theo cấp độ nhạy cảm của từng tác vụ.

> Hệ thống gồm 3 phần: (1) train mô hình nhận dạng giọng nói, (2) CLI assistant để test pipeline, (3) Flask web app full-stack với Gmail API, music player và file manager cá nhân.

---

## Tính năng chính

- **3 lớp bảo mật theo intent**
  - `NORMAL` — ai cũng dùng được (hỏi giờ, thời tiết, kể chuyện cười, Q&A tổng quát).
  - `IMPORTANT` — yêu cầu SV (verify giọng): đọc ghi chú, gửi email, xem số dư, xoá dữ liệu, mở file cá nhân.
  - `PERSONAL` — cá nhân hoá theo SID: chào theo tên, phát nhạc theo gu, hiển thị lịch riêng.
- **Pipeline voice end-to-end**: Mic → Silero VAD → faster-whisper (ASR) → ECAPA-TDNN (Speaker Encoder) → Gemini NLU → Router (gate auth) → gTTS (TTS).
- **Web app full-stack** (Flask): đăng ký user qua trình duyệt, push-to-talk, text mode, music player (YouTube + nhạc cá nhân), file manager, admin panel.
- **Gmail API tích hợp** (OAuth 2.0): user xác thực Google một lần, sau đó assistant gửi email trực tiếp qua API.
- **Đa kênh fallback**: SV thất bại → cho phép dùng password; Gemini quota hết → rule-based NLU; thiếu checkpoint riêng → tự fallback pretrained SpeechBrain.

---

## Cấu trúc project

```
Secure-Virtual-Assistant-with-Speaker-Recognition/
│
├── training/              # Phần 1 — Train ECAPA-TDNN (chạy trên Kaggle GPU)
│   ├── README.md
│   ├── train_ecapa.py
│   ├── evaluate_sid.py
│   ├── evaluate_sv.py
│   ├── requirements.txt
│   └── data/              # iden_split.txt, veri_test.txt
│
├── core/                  # Business logic dùng chung cho CLI + Web
│   ├── config.py          # paths, thresholds, env vars
│   ├── audio_io.py        # mic record, file I/O, Silero VAD, browser audio decode
│   ├── asr.py             # faster-whisper wrapper + transcript corrector
│   ├── tts.py             # gTTS wrapper (CLI play + web MP3 stream)
│   ├── speaker_encoder.py # ECAPA-TDNN (own ckpt hoặc pretrained SpeechBrain)
│   ├── database.py        # SQLite UserDB + SpeakerManager (enroll/SID/SV)
│   ├── intents.py         # 12 intents × 3 nhóm bảo mật
│   ├── nlu.py             # Gemini NLU + rule-based fallback
│   ├── handlers.py        # logic mỗi intent
│   ├── router.py          # orchestrator + SV/SID gating
│   ├── email_flow.py      # state machine cho luồng soạn email nhiều turn
│   ├── oauth.py           # Google OAuth 2.0 (auth URL, exchange code, refresh)
│   └── gmail_api.py       # gửi email qua Gmail REST API
│
├── cli/                   # Phần 2 — CLI Virtual Assistant
│   ├── README.md
│   ├── enroll_user.py     # đăng ký user qua CLI mic
│   ├── run_assistant.py   # REPL chạy assistant
│   └── test_pipeline.py   # smoke test với file wav (không cần mic)
│
├── web/                   # Phần 3 — Flask Web App
│   ├── README.md
│   ├── app.py             # Flask routes + API
│   ├── gen_cert.py        # tạo TLS self-signed cert (cho mic qua LAN)
│   ├── cert.pem / key.pem # TLS cert (không commit key.pem)
│   ├── requirements.txt
│   ├── templates/
│   │   ├── base.html
│   │   ├── home.html          # danh sách users
│   │   ├── enroll.html        # đăng ký + record N mẫu
│   │   ├── user_detail.html   # xem/sửa preferences, đổi password, xoá
│   │   ├── assistant.html     # chat push-to-talk + text mode
│   │   ├── music.html         # player + playlist + YouTube search
│   │   └── files.html         # file manager (SV-protected)
│   └── static/style.css
│
├── data/                  # Runtime artifacts
│   ├── users.db           # SQLite: users + embeddings + oauth_tokens
│   ├── enroll_audio/      # wav samples của mỗi user
│   ├── user_files/        # file & nhạc upload theo user_id
│   ├── profiles/          # preferences JSON mẫu (minh.json, lan.json)
│   └── turn_log.json      # log mỗi turn cho báo cáo
│
├── checkpoints/
│   └── best_model.pt      # ECAPA-TDNN weights (từ training/)
│
├── requirements.txt       # deps cho CLI (Phần 2)
├── .env                   # API keys (không commit)
└── .env.example           # template cho .env
```

---

## Bắt đầu nhanh

| Phần | Mô tả | Tài liệu chi tiết |
|---|---|---|
| **Phần 1** | Train ECAPA-TDNN trên VoxCeleb1 (Kaggle GPU) | [training/README.md](training/README.md) |
| **Phần 2** | CLI Virtual Assistant — REPL trên terminal | [cli/README.md](cli/README.md) |
| **Phần 3** | Flask Web App — UI + Gmail + Music + Files | [web/README.md](web/README.md) |

### 1. Cài đặt

```bash
# Tạo virtual env
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS

# Cài deps (chọn 1 trong 2)
pip install -r requirements.txt        # đủ cho CLI
pip install -r web/requirements.txt    # đủ cho cả Web + CLI

# System deps:
#   Windows: winget install Gyan.FFmpeg  (cần ffmpeg trong PATH)
#   Linux:   sudo apt install ffmpeg portaudio19-dev
#   macOS:   brew install ffmpeg portaudio
```

### 2. Cấu hình `.env`

```bash
cp .env.example .env
```

Điền tối thiểu:
- `GEMINI_API_KEY` — lấy free tại [aistudio.google.com/apikey](https://aistudio.google.com/apikey). Bỏ trống → fallback rule-based NLU.
- `GOOGLE_CLIENT_ID` + `GOOGLE_CLIENT_SECRET` — chỉ cần nếu dùng tính năng gửi email (xem hướng dẫn trong `.env.example`).
- `ADMIN_PASS` — chỉ cần nếu mở admin panel.

### 3. Speaker checkpoint

Copy `best_model.pt` từ Kaggle (Phần 1) vào `checkpoints/`.
**Chưa có cũng được** — hệ thống tự fallback sang **SpeechBrain pretrained ECAPA-TDNN (VoxCeleb2)**.

### 4. Chạy

```bash
# CLI assistant (REPL)
python cli/run_assistant.py

# Web app (HTTPS, hỗ trợ mic qua LAN)
python web/gen_cert.py          # tạo cert 1 lần
python -m web.app --ssl         # → https://localhost:5000
```

---

## Kiến trúc tổng quan

```
┌──────────────────────────────────────────────────────────────────┐
│   FRONTEND (Browser / CLI)                                       │
│   - MediaRecorder API push-to-talk (web)                         │
│   - sounddevice mic recording (CLI)                              │
└──────────────────────────────────────────────────────────────────┘
                          ↓ audio bytes (WebM/Opus hoặc wav)
┌──────────────────────────────────────────────────────────────────┐
│   AUDIO PIPELINE                                                 │
│   decode → 16kHz mono → Silero VAD trim                          │
└──────────────────────────────────────────────────────────────────┘
                          ↓
        ┌─────────────────┴─────────────────┐
        ↓                                   ↓
┌──────────────────┐               ┌──────────────────────┐
│ faster-whisper   │               │ ECAPA-TDNN encoder   │
│ (medium / int8)  │               │ → 192-d embedding    │
└──────────────────┘               └──────────────────────┘
        ↓                                   ↓
┌──────────────────┐               ┌──────────────────────┐
│ Gemini NLU       │               │ SpeakerManager       │
│ → {intent,       │               │ - identify (SID)     │
│    entities}     │               │ - verify  (SV)       │
└──────────────────┘               └──────────────────────┘
        └─────────────────┬─────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────────┐
│   ROUTER — auth gate theo INTENTS[intent].level                  │
│   - NORMAL    → handler trực tiếp                                │
│   - IMPORTANT → SV check → handler hoặc block                    │
│   - PERSONAL  → handler(user) để cá nhân hoá                     │
└──────────────────────────────────────────────────────────────────┘
                          ↓
                  response text + entities
                          ↓
                 gTTS → MP3 stream / play
```

---

## 3 lớp bảo mật — Intent map

| Nhóm | Auth | Intent | Use case |
|---|---|---|---|
| **NORMAL** | Không | `get_time`, `get_weather`, `tell_joke`, `general_question` | Guest cũng dùng được |
| **IMPORTANT** | SV (hoặc password fallback) | `read_notes`, `send_email`, `check_balance`, `delete_data`, `open_files` | Phải verify đúng giọng / mật khẩu |
| **PERSONAL** | SID | `greet`, `play_music`, `show_schedule` | Identify ai → response cá nhân hoá theo preferences |

**Demo case chấm điểm**:
- `"Mấy giờ rồi?"` — guest cũng trả lời (NORMAL).
- `"Đọc ghi chú của tôi"` giọng Minh → SV pass → đọc notes Minh.
- Cùng câu nhưng người lạ nói → SID = guest → block.
- `"Phát nhạc đi"` giọng Minh thích rock → phát rock; giọng Lan thích ballad → phát ballad.

---

## Quy trình lưu trữ và xác thực

```
Đăng ký (enroll):
   audio_1 ... audio_N (mỗi mẫu 4s, N ≥ 2)
      ↓ Silero VAD trim
      ↓ ECAPA-TDNN encode
   emb_1 ... emb_N           (192-d, L2-normalized)
      ↓ trung bình + L2-normalize
   centroid                  (đại diện user)
      ↓ store
   SQLite users.db           (BLOB ~768 bytes mỗi user)

Một turn (identify + verify):
   audio_query → VAD → ECAPA → emb_query (192-d)

   SID: argmax cosine(emb_query, centroid_i)
        nếu max < SID_MIN_THRESHOLD → guest

   SV (cho IMPORTANT): cosine(emb_query, centroid_claimed)
        nếu score ≥ SV_THRESHOLD → pass
        ngược lại → block (cho phép fallback password)
```

Threshold mặc định (calibrate lại sau khi enroll thật):

```python
# core/config.py
SV_THRESHOLD       = 0.45   # cosine ≥ → cùng người
SID_MIN_THRESHOLD  = 0.35   # < → guest
```

---

## Tech stack

| Lớp | Công nghệ |
|---|---|
| Speaker recognition | ECAPA-TDNN (SpeechBrain), AAM-Softmax loss, cosine similarity |
| ASR | faster-whisper (CTranslate2) — tiếng Việt |
| NLU | Google Gemini 2.0 Flash + rule-based fallback |
| TTS | gTTS (Google Translate TTS) |
| VAD | Silero VAD |
| Web | Flask 3, vanilla JS, MediaRecorder API |
| Auth | Speaker Verification + bcrypt password + Google OAuth 2.0 |
| Music | YouTube via yt-dlp, Deezer search API |
| Storage | SQLite (users, embeddings, OAuth tokens) + filesystem (audio, files) |

---

## Calibrate threshold (khuyến nghị)

Sau khi enroll vài users, chạy assistant nhiều lần và xem `sid_score` / `sv_score`:

- **Genuine** (cùng người): score ≈ 0.55 – 0.85
- **Impostor** (khác người): score ≈ 0.10 – 0.40

Đặt `SV_THRESHOLD` ở giữa hai cụm.
Phương pháp chính thức: chạy `training/evaluate_sv.py` → lấy threshold tại điểm EER.

---

## Troubleshooting

| Triệu chứng | Khắc phục |
|---|---|
| `getUserMedia` block mic | Web phải chạy HTTPS hoặc `localhost`. Dùng `python web/gen_cert.py` + `--ssl`. |
| `OSError: PortAudio library not found` | `sudo apt install portaudio19-dev` (Linux) / `brew install portaudio` (macOS) |
| Whisper quá chậm trên CPU | Đổi `WHISPER_MODEL=tiny` hoặc `base` trong `.env`, hoặc `WHISPER_DEVICE=cuda WHISPER_COMPUTE=float16` |
| `ffmpeg not found` | Cài ffmpeg và đảm bảo trong PATH (pydub cần để decode WebM) |
| SID luôn ra "guest" | Threshold cao, hoặc audio enroll quá ngắn/noisy — check `sid_score` trong log |
| SV pass cả khi sai giọng | Tăng `SV_THRESHOLD` lên 0.5 – 0.6 |
| Gemini quota exceeded | Bỏ `GEMINI_API_KEY` trong `.env` → tự fallback rule-based |
| YouTube search lỗi bot check | Tạo `web/youtube_cookies.txt` từ browser cookies, hoặc đăng nhập Chrome/Edge để yt-dlp tự đọc |

---

## Đóng góp

Mỗi sub-folder có README riêng với hướng dẫn chi tiết hơn — xem [training/](training/README.md), [cli/](cli/README.md), [web/](web/README.md).
