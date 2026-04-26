# Phần 3 – Flask Web App

Virtual Assistant tiếng Việt với 3 lớp bảo mật dựa trên speaker recognition — giao diện web đăng ký user, admin panel, và chat push-to-talk.

## Tổng quan kiến trúc

```
┌──────────────────────────────────────────────────────────────────┐
│   BROWSER (HTML + JS, MediaRecorder API)                         │
│   ┌─────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│   │ /  list      │  │ /enroll      │  │ /assistant           │   │
│   │ users        │  │ record N×4s  │  │ push-to-talk chat    │   │
│   └─────────────┘  └──────────────┘  └──────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
                          ↓ HTTPS (multipart audio)
┌──────────────────────────────────────────────────────────────────┐
│   FLASK BACKEND (web/app.py)                                     │
│   ┌──────────────────────────────────────────────────────────┐   │
│   │   audio bytes → decode (WebM→wav) → VAD trim             │   │
│   │        ↓                              ↓                   │   │
│   │   Whisper ASR              ECAPA-TDNN encoder             │   │
│   │   (text)                   (192-d embedding)              │   │
│   │        ↓                              ↓                   │   │
│   │   Gemini NLU              SpeakerManager (SID + SV)       │   │
│   │   {intent, entities}      identify / verify vs SQLite     │   │
│   │              ↓        ↓                                   │   │
│   │           Router (auth gate per intent)                   │   │
│   │           ├─ NORMAL    → handler                          │   │
│   │           ├─ IMPORTANT → SV check → handler               │   │
│   │           └─ PERSONAL  → handler(user) personalize        │   │
│   │              ↓                                            │   │
│   │           response text → gTTS → MP3 stream → browser     │   │
│   └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

## Cấu trúc thư mục

```
Secure-Virtual-Assistant-with-Speaker-Recognition/
├── core/                       # Business logic (dùng chung CLI + web)
│   ├── config.py
│   ├── audio_io.py             # mic record, file I/O, VAD, browser audio decode
│   ├── asr.py                  # faster-whisper
│   ├── tts.py                  # gTTS (CLI play + web stream)
│   ├── speaker_encoder.py      # ECAPA-TDNN (own ckpt hoặc pretrained)
│   ├── database.py             # UserDB + SpeakerManager (enroll/SID/SV)
│   ├── intents.py              # 12 intents × 3 nhóm bảo mật
│   ├── nlu.py                  # Gemini + rule-based fallback
│   ├── handlers.py             # logic mỗi intent
│   └── router.py               # orchestrator + SV/SID gating
├── web/
│   ├── app.py                  # Flask app
│   ├── gen_cert.py             # tạo self-signed TLS cert (chạy 1 lần)
│   ├── cert.pem                # TLS certificate (import vào Windows Trust Store)
│   ├── key.pem                 # TLS private key (không commit lên git)
│   ├── templates/
│   │   ├── base.html
│   │   ├── home.html           # list users
│   │   ├── enroll.html         # đăng ký + record N mẫu
│   │   ├── user_detail.html    # xem/sửa preferences, xóa user
│   │   └── assistant.html      # chat push-to-talk
│   └── static/style.css
├── cli/
│   ├── enroll_user.py          # enroll qua CLI mic
│   ├── run_assistant.py        # REPL CLI
│   └── test_pipeline.py        # smoke test với file wav
├── checkpoints/                # (copy từ Tuần 1) best_model.pt
├── data/                       # users.db, enroll_audio/, log
├── requirements.txt
└── .env.example
```

## Cài đặt

```bash
# Windows
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install flask flask-cors  # hoặc dùng web/requirements.txt

# Linux / macOS
# python -m venv venv && source venv/bin/activate
# pip install -r web/requirements.txt

# System deps:
#   Linux:  sudo apt install ffmpeg portaudio19-dev
#   macOS:  brew install ffmpeg portaudio
#   Win:    winget install Gyan.FFmpeg  (rồi thêm bin/ vào PATH)

cp .env.example .env  # điền GEMINI_API_KEY (free tại aistudio.google.com)
```

**Copy speaker checkpoint** từ Tuần 1 vào `checkpoints/best_model.pt`.
Nếu chưa có → app tự fallback sang pretrained SpeechBrain ECAPA-TDNN (VoxCeleb2).

## Cách chạy

### Bước 0 — Tạo TLS certificate (chỉ làm 1 lần)

Trình duyệt chỉ cho phép `getUserMedia` (mic) khi trang chạy trên HTTPS hoặc `localhost`.
Để truy cập qua địa chỉ IP trong LAN (ví dụ `192.168.1.9`) cần HTTPS với cert được trust.

```bash
# Sinh cert.pem + key.pem trong thư mục web/
python web/gen_cert.py
```

Sau đó import `web/cert.pem` vào Windows Trusted Root CA (theo hướng dẫn in ra khi chạy lệnh trên):

1. Mở **Run** (`Win+R`) → gõ `certlm.msc` → Enter
2. Mở **Trusted Root Certification Authorities** → **Certificates**
3. Chuột phải → **All Tasks** → **Import…**
4. Chọn file `web/cert.pem`
5. Chọn store **Trusted Root Certification Authorities** → **Finish** → **Yes**
6. Khởi động lại trình duyệt

> Cert có hiệu lực 825 ngày và có Subject Alternative Names cho `localhost`, `127.0.0.1`, và `192.168.1.9`.
> Nếu IP máy đổi, sửa `HOSTNAMES` trong `web/gen_cert.py` và tạo lại.

### Bước 1 — Chạy server HTTPS

```bash
python -m web.app --ssl
# → mở https://localhost:5000
# → mở https://192.168.1.9:5000  (từ thiết bị khác trong LAN)
```

Nếu chưa có `cert.pem`/`key.pem`, server tự dùng `adhoc` (cert tạm thời, thay đổi mỗi lần restart, trình duyệt vẫn hiện cảnh báo).

Đổi port nếu cần:
```bash
python -m web.app --ssl --port 8443
```

### Bước 2 — Sử dụng

Flow:
1. `/enroll` — đăng ký user mới: nhập user_id/name, record 5 mẫu giọng, optional preferences (gu nhạc, balance, lịch...)
2. `/` — xem danh sách users đã đăng ký
3. `/users/<id>` — sửa preferences, xóa user
4. `/assistant` — chat push-to-talk (giữ nút hoặc giữ phím Space để nói, thả ra để gửi)

### CLI mode

```bash
# Enroll (dùng --preferences_file để tránh lỗi quoting trên Windows)
python cli/enroll_user.py --user_id minh --name "Minh" \
    --preferences_file data/profiles/minh.json

# Chạy REPL
python cli/run_assistant.py

# Test với file wav (không cần mic)
python cli/test_pipeline.py --audio data/enroll_audio/minh/sample_1.wav
```

## 3 lớp bảo mật — Use Cases

| Nhóm | Auth | Ví dụ intent | Demo case |
|---|---|---|---|
| **NORMAL** | không | `get_time`, `get_weather`, `tell_joke`, `general_question` | Bất kỳ ai (kể cả guest chưa đăng ký) hỏi giờ → trả lời ngay |
| **IMPORTANT** | SV | `read_notes`, `send_email`, `check_balance`, `delete_data` | Minh nói "Đọc ghi chú của tôi" → SV pass → đọc notes của Minh.<br>Lan nói câu y hệt → SID nhận ra Lan → SV với claimed_user=Lan → đọc notes của Lan.<br>Người lạ nói câu này → SID = guest → block. |
| **PERSONAL** | SID | `greet`, `play_music`, `show_schedule` | Minh: "Phát nhạc" → biết là Minh → phát rock (gu của Minh).<br>Lan: cùng câu → biết là Lan → phát ballad. |

## Quy trình đăng ký (chi tiết cho báo cáo)

```
Web /enroll                           Backend
─────────────                         ────────
User nhập tên + record 5 mẫu                      
(MediaRecorder → WebM/Opus)    →     POST /api/enroll multipart
                                     ├─ Decode WebM → 16kHz wav
                                     ├─ Silero VAD trim mỗi mẫu
                                     ├─ ECAPA encode → emb_1...emb_5 (192-d)
                                     ├─ Centroid = mean(emb_i) → L2 norm
                                     ├─ Lưu SQLite: users(metadata) + 
                                     │              embeddings(BLOB 768 bytes)
                                     └─ Lưu wav samples vào data/enroll_audio/<id>/
                                ←     {ok, n_samples, centroid_norm}
```

## Quy trình một turn (chi tiết)

```
[Browser] giữ Space → MediaRecorder ghi audio
[Browser] thả tay → POST /api/assistant/turn (multipart audio)

[Backend]
  audio_bytes
    → pydub decode → 16kHz mono float32
    → Silero VAD trim
    ┌───────────────────┬───────────────────┐
    ↓                   ↓                   
  Whisper ASR        ECAPA encode → emb_query
    ↓                   ↓
  transcript        SpeakerManager.identify(emb)
    ↓                   ↓
  Gemini NLU      → (uid, name, sid_score)  
    ↓
  intent, entities
    ↓
  Router.handle_turn(audio, transcript, nlu_result)
    │
    ├─ Lookup INTENTS[intent].level
    │
    ├─ if level == NORMAL:    handler(entities, user)
    ├─ if level == IMPORTANT: 
    │     SpeakerManager.verify(emb, claimed_uid)
    │     if score >= 0.45: handler(entities, user)
    │     else:              block, return error
    └─ if level == PERSONAL:  handler(entities, user)  
                              # user info dùng để cá nhân hóa response

  → response_text
  → gTTS → MP3 bytes
  → JSON {transcript, intent, response, ..., tts_url}

[Browser]
  Hiển thị transcript + response + meta (intent, sid_score, sv_score)
  <audio src=tts_url> auto play
```

## API Reference

| Method | Path | Body | Response |
|---|---|---|---|
| GET | `/` | — | HTML home |
| GET | `/enroll` | — | HTML enroll form |
| POST | `/api/enroll` | multipart: user_id, name, preferences (json), sample_0…N | `{ok, user_id, n_samples, centroid_norm}` |
| GET | `/users/<id>` | — | HTML user detail |
| POST | `/api/users/<id>/update` | json: `{preferences: {…}}` | `{ok}` |
| POST | `/api/users/<id>/delete` | — | `{ok}` |
| GET | `/assistant` | — | HTML chat |
| POST | `/api/assistant/turn` | multipart: audio | TurnResult JSON |
| GET | `/api/tts?text=...` | — | MP3 stream |
| GET | `/api/health` | — | system info |

## Calibrate threshold

Mặc định trong `core/config.py`:
```python
SV_THRESHOLD = 0.45         # cosine ≥ 0.45 → cùng người
SID_MIN_THRESHOLD = 0.35    # < 0.35 → guest
```

Sau khi enroll vài users:
1. Mở DevTools tab Network → quan sát `sid_score` và `sv_score` trong response của `/api/assistant/turn`.
2. **Genuine** (cùng người): score ≈ 0.55–0.85.
3. **Impostor** (khác người): score ≈ 0.10–0.40.
4. Đặt `SV_THRESHOLD` ở giữa 2 cụm.

Phương pháp formal: chạy `part1/evaluate_sv.py` (Phần 1) → lấy threshold tại EER → dùng trong production.

## Bug & Workaround

**`getUserMedia` không hoạt động / mic không có quyền** → trình duyệt chỉ cho phép mic trên HTTPS hoặc `localhost`. Làm theo Bước 0 để tạo cert và import vào Windows, sau đó chạy server với `--ssl`.

**`No module named 'sounddevice'`** trên server → đã handle: server không cần audio device, chỉ cần ffmpeg để decode browser audio.

**Whisper quá chậm trên CPU** → đổi `WHISPER_MODEL=tiny` hoặc `base` trong `.env`. Hoặc dùng GPU: `WHISPER_DEVICE=cuda WHISPER_COMPUTE=float16`.

**`ffmpeg not found`** → cài ffmpeg, đảm bảo có trong PATH. pydub cần nó để decode WebM.

**SID luôn ra "guest"** → check threshold + đảm bảo audio enroll dài >= 2s sau VAD.

**Browser block autoplay TTS** → user click vào trang trước khi gửi turn đầu tiên (browser cần user gesture).

## Checklist nộp bài

- [ ] **Phần 1** — Mã nguồn `train_ecapa.py`, `evaluate_sid.py`, `evaluate_sv.py` + checkpoint + log + kết quả EER/Top-1.
- [ ] **Phần 2 + 3** — toàn bộ thư mục: `core/`, `cli/`, `web/`, requirements, README.
- [ ] **Database** — `data/users.db` mẫu (đã enroll vài users) + audio mẫu trong `data/enroll_audio/`.
- [ ] **Demo video** — quay 3 use case (NORMAL / IMPORTANT pass / IMPORTANT fail / PERSONAL phân biệt 2 user).
- [ ] **Báo cáo** — bao gồm:
   - Mô tả mô hình + dataset (Phần 1)
   - Sơ đồ kiến trúc + flow đăng ký + flow xử lý turn
   - Kết quả thực nghiệm: EER, Top-1, threshold đã calibrate
   - Phân tích lỗi (false accept / false reject)
   - Use case + screenshot UI
- [ ] Nếu file lớn (checkpoint, audio mẫu) → upload Google Drive, nộp link.
