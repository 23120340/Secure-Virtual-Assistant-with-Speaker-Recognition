# Phần 2 / Tuần 2 – Virtual Assistant Pipeline (CLI)

End-to-end pipeline: **Mic → ASR → Speaker Encoder → NLU → Router (SV/SID gate) → TTS**. Chạy hoàn toàn trên CLI ở tuần này; tuần 3 mới ráp web UI.

## Cấu trúc

```
Secure-Virtual-Assistant-with-Speaker-Recognition/
├── core/                  # logic dùng chung cho cả CLI lẫn web (tuần 3)
│   ├── config.py          # paths, thresholds, env vars
│   ├── audio_io.py        # record + VAD + load/save wav
│   ├── asr.py             # faster-whisper wrapper
│   ├── tts.py             # gTTS wrapper
│   ├── speaker_encoder.py # ECAPA-TDNN — own ckpt hoặc pretrained
│   ├── database.py        # UserDB + SpeakerManager (enroll/identify/verify)
│   ├── intents.py         # định nghĩa 12 intents × 3 nhóm bảo mật
│   ├── nlu.py             # Gemini NLU + rule-based fallback
│   ├── handlers.py        # logic mỗi intent
│   └── router.py          # orchestrator + SV/SID gating
├── scripts/
│   ├── enroll_user.py     # đăng ký user mới
│   ├── run_assistant.py   # REPL chạy assistant
│   └── test_pipeline.py   # smoke test với file wav (không cần mic)
├── data/                  # (tự tạo) DB sqlite, audio enrollment, log
├── checkpoints/           # best_model.pt (copy từ Tuần 1 / Kaggle)
├── part1/                 # Tuần 1 — ECAPA-TDNN training (xem part1/README.md)
├── requirements.txt
└── .env.example
```

## Cài đặt

```bash
# Windows
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Linux / macOS
# python -m venv venv && source venv/bin/activate
# pip install -r requirements.txt

# Linux có thể cần thêm: sudo apt install ffmpeg portaudio19-dev
# macOS: brew install ffmpeg portaudio

cp .env.example .env       # sửa GEMINI_API_KEY trong này
```

**Copy checkpoint từ Tuần 1** vào `checkpoints/best_model.pt`.
Nếu chưa train xong → vẫn chạy được, hệ thống sẽ tự fallback sang **pretrained SpeechBrain ECAPA-TDNN (VoxCeleb2)** để bạn dev song song.

## 3 nhóm chức năng — đáp ứng yêu cầu đề bài

| Nhóm | Auth | Intents | Behavior |
|---|---|---|---|
| **NORMAL** | Không cần | `get_time`, `get_weather`, `tell_joke`, `general_question` | Ai cũng dùng được, kể cả guest |
| **IMPORTANT** | SV (Speaker Verification) | `read_notes`, `send_email`, `check_balance`, `delete_data` | Phải verify đúng giọng user mới chạy |
| **PERSONAL** | SID (cá nhân hóa) | `greet`, `play_music`, `show_schedule` | Identify ai → response theo preferences user đó |

Use case chấm điểm tốt:
- "Mấy giờ rồi?" (guest cũng được) → NORMAL
- "Đọc ghi chú của tôi" (giọng Minh) → SV pass → đọc notes của Minh
- "Đọc ghi chú của tôi" (Minh dùng giọng impersonate giả) → SV fail → block
- "Phát nhạc đi" (giọng Minh, Minh thích rock) → SID → phát rock
- "Phát nhạc đi" (giọng Lan, Lan thích ballad) → SID → phát ballad

## Demo flow

### 1. Đăng ký 2-3 users

```bash
# User Minh: thích rock, có balance giả
python scripts/enroll_user.py \
    --user_id minh --name "Minh" \
    --preferences '{"favorite_genre":"rock","favorite_artist":"Bức Tường","balance":15000000,"notes":["Họp 9h sáng","Mua sữa","Đặt phòng cuối tuần"],"schedule":["9h họp team","2h gặp khách hàng","tối tập gym"]}'

# User Lan: thích ballad
python scripts/enroll_user.py \
    --user_id lan --name "Lan" \
    --preferences '{"favorite_genre":"ballad","favorite_artist":"Mỹ Tâm","balance":8500000,"notes":["Đi chợ","Gọi mẹ"]}'

# Có thể enroll bằng wav file thay vì record:
python scripts/enroll_user.py --user_id test --name "Test" \
    --audio_files samples/test1.wav samples/test2.wav samples/test3.wav
```

Khi enroll, mỗi mẫu sẽ được:
1. Record 4s từ mic (hoặc đọc file)
2. VAD trim bỏ khoảng lặng
3. Encode 192-d embedding
4. Sau khi có 5 mẫu → trung bình → centroid → lưu DB

### 2. Chạy assistant

```bash
python scripts/run_assistant.py
```

Mỗi turn bấm Enter → nói 5 giây → xem log → nghe response.

```
> [Enter]
  🎙  Recording 5.0s...
  Loading Silero VAD...
────────────────────────────────────────────────────────────
  Transcript:  'phát nhạc đi'
  Intent:      play_music  [personal]
  Entities:    {}
  Speaker:     Minh (uid=minh, sid_score=0.712)
  Response:    Phát nhạc rock cho Minh — đặc biệt là của Bức Tường.
────────────────────────────────────────────────────────────
  🔊 Speaking: ...
```

### 3. Test mode không cần mic (cho Kaggle/CI)

```bash
# Chỉ test NLU + handler (text input thay vì voice)
python scripts/run_assistant.py --text-mode

# Test pipeline với file wav có sẵn
python scripts/test_pipeline.py --audio data/enroll_audio/minh/sample_1.wav
```

## Calibrate threshold (quan trọng!)

Threshold mặc định trong `core/config.py`:
```python
SV_THRESHOLD = 0.45         # > → cùng người
SID_MIN_THRESHOLD = 0.35    # < → guest
```

Sau khi enroll vài users, chạy thử nhiều lần và xem `sid_score` / `sv_score` trong log. Tính toán:
- **Genuine pairs** (cùng người): score nên ~0.55–0.85
- **Impostor pairs** (khác người): score nên ~0.10–0.40

Đặt `SV_THRESHOLD` ở giữa 2 cụm. Phương pháp formal hơn: chạy `evaluate_sv.py` (Tuần 1) trên trial pairs nội bộ → lấy threshold tại EER.

## Quy trình lưu trữ (cho báo cáo)

```
Đăng ký:
   audio_1...audio_5
      ↓ ECAPA-TDNN
   emb_1...emb_5     (mỗi cái 192-d, L2-normalized)
      ↓ trung bình + L2-normalize
   centroid          (192-d, đại diện user)
      ↓ store
   SQLite users.db   (BLOB ~768 bytes)

Verify/Identify:
   audio_query → ECAPA → emb_query (192-d)
   cosine(emb_query, centroid_user) → similarity score
      → SV: score >= 0.45 ?
      → SID: argmax similarity, nếu max < 0.35 thì = guest
```

## Bug thường gặp

**`OSError: PortAudio library not found`** → cài `portaudio19-dev` (Linux) hoặc `brew install portaudio` (macOS).

**`Whisper too slow`** → đổi sang model `tiny` hoặc `base`, hoặc bật GPU bằng `WHISPER_DEVICE=cuda` + `WHISPER_COMPUTE=float16`.

**SID luôn return guest** → threshold quá cao, hoặc audio enroll quá ngắn/noisy. Check `sid_score` trong log.

**SV pass cả khi giọng khác** → threshold quá thấp. Tăng lên 0.5–0.6.

**Gemini quota exceeded** → bỏ key trong `.env`, sẽ tự fallback rule-based.

## Checklist tuần 2

- [ ] Cài requirements + tạo `.env`
- [ ] Copy `best_model.pt` từ Tuần 1 vào `checkpoints/`
- [ ] Enroll ít nhất 2-3 users (kể cả người nhà/bạn để test SID phân biệt)
- [ ] Test 1 intent thuộc mỗi nhóm: NORMAL, IMPORTANT (cả pass + fail), PERSONAL
- [ ] Kiểm tra `data/turn_log.json` ghi đúng — dùng cho báo cáo
- [ ] Calibrate `SV_THRESHOLD` dựa trên scores quan sát được

Sang **Tuần 3**: web đăng ký + admin panel với Flask, gọi cùng `core/`.
