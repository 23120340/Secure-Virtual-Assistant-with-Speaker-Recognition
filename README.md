# Secure Virtual Assistant with Speaker Recognition

Trợ lý ảo tiếng Việt có phân quyền bằng giọng nói, kết hợp **Speaker Identification (SID)** và **Speaker Verification (SV)** để quyết định tác vụ nào được phép chạy theo mức độ nhạy cảm.

Đồ án gồm hai yêu cầu chính theo đề:

- **YC1 - Train & evaluate speaker model**: ECAPA-TDNN cho speaker identification và speaker verification.
- **YC2 - Virtual assistant tích hợp SV/SID**: trợ lý có voice interaction, enrollment, user management và các tác vụ `NORMAL` / `IMPORTANT` / `PERSONAL`.

> Trạng thái hiện tại: phần runtime/web đã khá đầy đủ; phần còn thiếu quan trọng trước khi nộp là **artifact kết quả training/evaluation thật** và **calibration benchmark thật** trong `training/results/` hoặc `docs/results/`. Xem checklist chi tiết ở [docs/PROJECT_REVIEW_AND_TODO.md](docs/PROJECT_REVIEW_AND_TODO.md).

---

## Tính Năng Chính

- **3 lớp bảo mật theo intent**
  - `NORMAL`: không cần xác thực, ví dụ hỏi giờ, thời tiết demo, kể chuyện cười, hỏi đáp tổng quát.
  - `IMPORTANT`: cần speaker verification, ví dụ đọc ghi chú, gửi email, xem số dư demo, xoá dữ liệu, mở file cá nhân, thêm ghi chú/lịch/liên hệ.
  - `PERSONAL`: cần speaker identification để cá nhân hoá, ví dụ chào theo tên, phát nhạc theo gu, xem lịch cá nhân.
- **Voice pipeline end-to-end**: browser/CLI audio -> decode 16 kHz mono -> Silero VAD -> ASR -> speaker encoder -> NLU -> router auth gate -> handler -> TTS.
- **ASR backend có thể đổi qua `.env`**:
  - `faster-whisper` mặc định.
  - `phowhisper` cho PhoWhisper/VinAI nếu cài thêm dependency phù hợp.
- **Speaker backend có thể đổi qua `.env`**:
  - `ecapa` mặc định: dùng checkpoint tự train nếu có, fallback SpeechBrain pretrained nếu thiếu.
  - `wavlm`: dùng `microsoft/wavlm-base-plus-sv`.
- **Web app Flask**: enrollment bằng trình duyệt, push-to-talk assistant, text mode, file manager, music player, admin panel, Google OAuth/Gmail send.
- **Security/polish đã có**: password PBKDF2, OAuth PKCE + nonce, Fernet token encryption, rate limit, audit log JSONL, request ID, health/readiness endpoints, cascade delete, consent banner, data export, challenge-response opt-in.
- **Test suite**: `97` unit/integration tests hiện pass với `pytest`.

---

## Cấu Trúc Repo

```text
Secure-Virtual-Assistant-with-Speaker-Recognition/
├── core/                    # Logic dùng chung CLI + Web
│   ├── audio_io.py          # Decode audio, VAD, preprocess
│   ├── asr.py               # faster-whisper / PhoWhisper
│   ├── audit.py             # JSONL audit log
│   ├── challenge.py         # Challenge-response cho IMPORTANT voice flow
│   ├── config.py            # Env/config/thresholds
│   ├── database.py          # SQLite UserDB + SpeakerManager
│   ├── email_flow.py        # Multi-turn email composer
│   ├── gmail_api.py         # Gmail REST API send
│   ├── handlers.py          # Intent handlers
│   ├── intents.py           # AuthLevel + intent registry
│   ├── nlu.py               # Gemini function-calling + rule-based fallback
│   ├── oauth.py             # Google OAuth 2.0
│   ├── resilience.py        # Retry + circuit breaker
│   ├── router.py            # SID/SV gating + dispatch
│   ├── speaker_encoder.py   # ECAPA / WavLM encoder
│   ├── tts.py               # gTTS stream, multi-language
│   └── turn_logging.py      # JSONL turn logs + NLU training candidates
├── training/                # YC1: train/evaluate ECAPA-TDNN
│   ├── train_ecapa.py
│   ├── evaluate_sid.py
│   ├── evaluate_sv.py
│   ├── MODEL_CARD.md
│   ├── README.md
│   ├── data/
│   └── results/             # Cần commit artifact thật trước khi nộp
├── web/                     # YC2: Flask web app
│   ├── app.py
│   ├── gen_cert.py
│   ├── README.md
│   ├── templates/
│   └── static/
├── cli/                     # CLI enrollment/assistant/smoke test
├── scripts/                 # benchmark, re-enroll backend, password migration
├── tests/                   # pytest suite
├── docs/
│   ├── PROJECT_REVIEW_AND_TODO.md
│   ├── Secure_Virtual_Assistant_with_Speaker_Recognition.pdf
│   └── results/             # Runtime benchmark/calibration artifacts
├── data/                    # Runtime DB/audio/files/logs, không nên commit dữ liệu nhạy cảm
├── checkpoints/             # best_model.pt, dùng Git LFS nếu commit checkpoint
├── .env.example             # Khung config
├── requirements.txt
├── requirements.lock.txt
└── README.md
```

---

## Cài Đặt Nhanh

### 1. Tạo môi trường

```powershell
python -m venv venv
./venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

Nếu chạy web đầy đủ mà thiếu dependency, dùng thêm:

```powershell
pip install -r web/requirements.txt
```

System dependency cần có:

- **FFmpeg** để decode WebM/Opus từ browser.
- **PortAudio** nếu dùng CLI mic qua `sounddevice`.

Windows:

```powershell
winget install Gyan.FFmpeg
```

### 2. Cấu hình `.env`

Repo có sẵn `.env.example`. Tạo `.env` theo file đó:

```powershell
Copy-Item .env.example .env
```

Các biến quan trọng:

| Biến | Mục đích |
|---|---|
| `FLASK_SECRET` | Secret cho session Flask. Không để trống khi demo ổn định. |
| `TOKEN_ENCRYPTION_KEY` | Key riêng để mã hoá OAuth token. Nếu trống, fallback derive từ `FLASK_SECRET`. |
| `GEMINI_API_KEY` | Dùng Gemini NLU/Q&A. Nếu trống, fallback rule-based NLU. |
| `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `GOOGLE_REDIRECT_URI` | OAuth Gmail API cho intent gửi email. |
| `ADMIN_PASS` | Mật khẩu admin panel. |
| `ASR_BACKEND` | `faster-whisper` hoặc `phowhisper`. |
| `SPEAKER_BACKEND` | `ecapa` hoặc `wavlm`. |
| `ALLOW_PASSWORD_FOR_IMPORTANT` | Mặc định `false`: text mode block IMPORTANT. |
| `CHALLENGE_RESPONSE_ENABLED` | Mặc định `false`: bật nếu muốn demo chống replay đơn giản. |
| `SPEAKER_INCREMENTAL_REENROLL` | Mặc định `false`: cập nhật centroid sau SV pass. |

Sinh `TOKEN_ENCRYPTION_KEY`:

```powershell
./venv/Scripts/python.exe -c "import secrets,base64; print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode())"
```

### 3. Checkpoint speaker model

Đặt checkpoint tự train vào:

```text
checkpoints/best_model.pt
```

Nếu chưa có, app vẫn chạy bằng fallback SpeechBrain pretrained ECAPA. Tuy nhiên để đáp ứng YC1, vẫn cần training/evaluation artifact thật từ `training/`.

---

## Chạy Ứng Dụng

### Web app

Chạy HTTP local nhanh:

```powershell
$env:FLASK_DEV_HTTP="true"
python -m web.app
```

Mở:

```text
http://localhost:5000
```

Chạy HTTPS để browser cho phép microphone ổn định hơn:

```powershell
python web/gen_cert.py
python -m web.app --ssl
```

Mở:

```text
https://localhost:5000
```

Các trang chính:

| Path | Chức năng |
|---|---|
| `/` | Danh sách user |
| `/enroll` | Đăng ký user, record nhiều mẫu giọng |
| `/users/<id>` | Xem/sửa user, preferences, password, export/delete |
| `/assistant` | Push-to-talk assistant + text mode |
| `/music` | Music player |
| `/files` | File manager sau xác thực |
| `/healthz` | Liveness |
| `/readyz` | Readiness |
| `/api/health` | Health/info legacy |

### CLI

```powershell
python cli/enroll_user.py --user_id minh --name "Minh" --preferences_file data/profiles/minh.json
python cli/run_assistant.py
python cli/test_pipeline.py --audio data/enroll_audio/minh/sample_1.wav
```

---

## Intent Và AuthLevel

| Nhóm | Auth | Intent |
|---|---|---|
| `NORMAL` | Không cần xác thực | `get_time`, `get_weather`, `tell_joke`, `general_question` |
| `IMPORTANT` | Speaker Verification | `read_notes`, `send_email`, `check_balance`, `delete_data`, `open_files`, `add_note`, `add_schedule`, `add_contact` |
| `PERSONAL` | Speaker Identification | `greet`, `play_music`, `show_schedule` |

Lưu ý cho báo cáo:

- `get_weather` hiện là demo/stub, chưa gọi API thời tiết thật.
- `check_balance` đọc `preferences.balance`, không kết nối ngân hàng thật.
- `send_email` gọi Gmail API thật với OAuth scope `gmail.send`.
- Text mode không có audio để verify giọng, nên mặc định block `IMPORTANT`. Nếu bật `ALLOW_PASSWORD_FOR_IMPORTANT=true`, password fallback là chế độ demo/convenience, không phải biometric gate.

---

## Security Model Tóm Tắt

- Password user/admin: PBKDF2-HMAC-SHA256, salt ngẫu nhiên.
- Quên mật khẩu: không khôi phục password cũ; admin có thể reset password mới qua Admin Panel sau khi xác thực `ADMIN_PASS`.
- OAuth token: Fernet encryption; ưu tiên `TOKEN_ENCRYPTION_KEY`, fallback HKDF từ `FLASK_SECRET`.
- OAuth flow: server-side nonce + PKCE, callback escape HTML.
- CSRF guard: yêu cầu `X-Requested-With` cho request thay đổi trạng thái.
- Session cookie: `HttpOnly`, `SameSite=Lax`, `Secure` trừ khi `FLASK_DEV_HTTP=true`.
- Rate limit: password endpoints, enroll, assistant text, challenge response.
- Audit log: JSONL ở `data/audit.log`, không log nội dung nhạy cảm dài.
- Turn log để train/debug: `data/turn_log.jsonl`, append mỗi lượt web/CLI với `schema_version`, transcript đã redact email/số điện thoại, intent/entity dự đoán, auth level, SID/SV score, trạng thái block và source.
- Candidate NLU dataset: `data/nlu_training_candidates.jsonl`, dùng cho labeling thủ công. Export CSV bằng:

```powershell
python scripts/export_nlu_dataset.py --out data/nlu_training_candidates.csv
```
- File path safety: `_resolve_child_path(...).relative_to(...)`, không dùng string `startswith`.
- Privacy: consent banner khi enroll, cascade delete WAV/files/tokens/embeddings, data export ZIP password-gated.

Challenge-response cho IMPORTANT voice flow là opt-in:

```env
CHALLENGE_RESPONSE_ENABLED=true
```

Khi bật, sau khi SID+SV pass, server yêu cầu user đọc lại phrase random. Handler chỉ chạy nếu audio thứ hai vừa khớp phrase vừa pass SV.

---

## Training Và Evaluation YC1

Chi tiết ở [training/README.md](training/README.md).

Dataset hiện được mô tả là VoxCeleb1 Indian subset:

- 24 speakers.
- `training/data/iden_split.txt`: 4857 utterance.
- `training/data/veri_test.txt`: 552 trial pairs.
- Có caveat methodology: split theo utterance có thể leak session, trial pair nhỏ, bias Indian subset, chưa có augmentation.

Các script chính:

```powershell
python training/train_ecapa.py --help
python training/evaluate_sid.py --help
python training/evaluate_sv.py --help
```

Artifact cần có trước khi nộp:

| Folder | Artifact |
|---|---|
| `training/results/` | `training_log.json`, `spk2idx.json`, `sid_results.json`, `sv_results.json`, khuyến nghị thêm ROC/confusion/loss plots |
| `docs/results/` | `benchmark_ecapa.json`, `benchmark_wavlm.json` hoặc `benchmark_all.json`, `threshold_calibration.md` |

Hiện repo đã có README/checklist placeholder cho hai folder này, nhưng số liệu thật cần được generate bằng Kaggle/GPU và audio enroll demo thật.

---

## Benchmark Và Re-enroll

Xem [scripts/README.md](scripts/README.md).

Chạy benchmark runtime sau khi enroll vài user:

```powershell
python scripts/benchmark.py --out docs/results/benchmark_ecapa.json
python scripts/benchmark.py --all --out docs/results/benchmark_all.json
```

File benchmark JSON có schema `secva.benchmark.v1`, metadata backend, summary WER/SID/SV và từng sample để phân tích lỗi/calibration.

Nếu đổi speaker backend:

```powershell
$env:SPEAKER_BACKEND="wavlm"
python scripts/reenroll_backend.py
python scripts/benchmark.py --speaker-only
```

---

## Test

Chạy toàn bộ test suite:

```powershell
./venv/Scripts/python.exe -m pytest tests/ -q --basetemp .pytest_tmp
```

Kết quả gần nhất:

```text
94 passed, 4 warnings
```

`--basetemp .pytest_tmp` hữu ích trên máy Windows nếu thư mục temp mặc định bị permission denied.

---

## Troubleshooting

| Triệu chứng | Cách xử lý |
|---|---|
| Browser không cho dùng mic | Dùng `localhost` hoặc chạy HTTPS bằng `python -m web.app --ssl`. |
| `ffmpeg not found` | Cài FFmpeg và đảm bảo có trong `PATH`. |
| Whisper chậm trên CPU | Đổi `WHISPER_MODEL=tiny` hoặc `base`; nếu có GPU dùng `WHISPER_DEVICE=cuda`, `WHISPER_COMPUTE=float16`. |
| SID luôn ra guest | Audio enroll quá ngắn/noisy hoặc threshold cao; kiểm tra `sid_score` trong response/log. |
| SV quá dễ pass | Tăng threshold tương ứng backend trong `.env`, rồi benchmark lại. |
| Gemini lỗi/quota | Bỏ `GEMINI_API_KEY` để fallback rule-based NLU. |
| Gmail OAuth redirect lỗi | `GOOGLE_REDIRECT_URI` trong `.env` phải khớp tuyệt đối với Google Cloud Console. |
| Cookie/session không hoạt động trên HTTP | Set `FLASK_DEV_HTTP=true` khi chạy local HTTP, hoặc dùng HTTPS. |

---

## Checklist Trước Khi Nộp

- [ ] Chạy training thật và commit artifact vào `training/results/`.
- [ ] Chạy evaluation SID/SV thật và có `sid_results.json`, `sv_results.json`.
- [ ] Enroll vài user demo qua web, chạy `scripts/benchmark.py --out docs/results/...`.
- [ ] Cập nhật report với caveat dataset/split/threshold và các chức năng demo/stub.
- [ ] Quay demo đủ 3 nhóm: `NORMAL`, `IMPORTANT pass/fail`, `PERSONAL`.
- [ ] Không commit `.env`, `data/users.db`, OAuth token, audio/biometric data nhạy cảm nếu repo public.

Tài liệu nên đọc tiếp:

- [docs/PROJECT_REVIEW_AND_TODO.md](docs/PROJECT_REVIEW_AND_TODO.md): review tổng hợp và todo theo ưu tiên.
- [training/README.md](training/README.md): hướng dẫn train/evaluate ECAPA-TDNN.
- [web/README.md](web/README.md): hướng dẫn web app chi tiết.
- [scripts/README.md](scripts/README.md): benchmark, re-enroll backend, migration.
- [training/MODEL_CARD.md](training/MODEL_CARD.md): model card và caveat sử dụng.
