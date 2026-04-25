# Secure Virtual Assistant with Speaker Recognition

End-to-end hệ thống trợ lý ảo bảo mật, nhận diện người dùng qua giọng nói.

## Cấu trúc project

```
Secure-Virtual-Assistant-with-Speaker-Recognition/
│
├── part1/              # Phần 1 — Train ECAPA-TDNN (chạy trên Kaggle)
│   ├── README.md
│   ├── train_ecapa.py
│   ├── evaluate_sid.py
│   ├── evaluate_sv.py
│   ├── requirements.txt
│   └── data/           # iden_split.txt, veri_test.txt
│
├── core/               # Logic dùng chung cho CLI (Phần 2) và Web (Phần 3)
│   ├── config.py
│   ├── audio_io.py     # record, decode_browser_audio, VAD
│   ├── asr.py          # Whisper ASR
│   ├── tts.py          # gTTS + synthesize_to_mp3_bytes
│   ├── speaker_encoder.py
│   ├── database.py     # SQLite UserDB + SpeakerManager
│   ├── intents.py
│   ├── nlu.py          # Gemini / rule-based
│   ├── handlers.py
│   └── router.py
│
├── cli/                # Phần 2 — CLI Virtual Assistant
│   ├── README.md
│   ├── enroll_user.py
│   ├── run_assistant.py
│   └── test_pipeline.py
│
├── web/                # Phần 3 — Flask Web App
│   ├── README.md
│   ├── app.py
│   ├── requirements.txt
│   ├── templates/
│   │   ├── base.html
│   │   ├── home.html
│   │   ├── enroll.html
│   │   ├── assistant.html
│   │   └── user_detail.html
│   └── static/
│       └── style.css
│
├── data/               # Runtime: SQLite DB, audio enrollment, logs
│   ├── users.db
│   ├── enroll_audio/
│   ├── profiles/       # preferences JSON mẫu
│   └── turn_log.json
│
├── checkpoints/        # Model weights (best_model.pt từ Kaggle)
├── requirements.txt    # Dependencies cho CLI (Phần 2)
├── .env                # API keys và config (không commit)
└── .env.example        # Template cho .env
```

## Bắt đầu

| Phần | Mô tả | Hướng dẫn |
|---|---|---|
| Phần 1 | Train ECAPA-TDNN trên Kaggle | [part1/README.md](part1/README.md) |
| Phần 2 | CLI Virtual Assistant | [cli/README.md](cli/README.md) |
| Phần 3 | Flask Web App | [web/README.md](web/README.md) |
