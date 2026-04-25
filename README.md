# Secure Virtual Assistant with Speaker Recognition

End-to-end hệ thống trợ lý ảo bảo mật, nhận diện người dùng qua giọng nói.

## Cấu trúc project

```
Secure-Virtual-Assistant-with-Speaker-Recognition/
├── part1/                 # Phần 1 — ECAPA-TDNN Speaker Recognition
│   ├── README.md          # Hướng dẫn train + evaluate trên Kaggle
│   ├── train_ecapa.py
│   ├── evaluate_sid.py
│   ├── evaluate_sv.py
│   ├── requirements.txt
│   └── data/              # iden_split.txt, veri_test.txt (local)
│
├── core/                  # Phần 2 — Virtual Assistant core logic
│   ├── config.py
│   ├── audio_io.py
│   ├── asr.py
│   ├── tts.py
│   ├── speaker_encoder.py
│   ├── database.py
│   ├── intents.py
│   ├── nlu.py
│   ├── handlers.py
│   └── router.py
│
├── scripts/               # Phần 2 — Entry points
│   ├── enroll_user.py
│   ├── run_assistant.py
│   └── test_pipeline.py
│
├── data/                  # Runtime: SQLite DB, audio enrollment, logs
├── checkpoints/           # Model weights (best_model.pt từ Kaggle)
├── requirements.txt       # Dependencies cho Phần 2
└── .env.example           # Cấu hình API keys
```

## Bắt đầu

- **Phần 1** (train model): xem [part1/README.md](part1/README.md) — chạy trên Kaggle GPU.
- **Phần 2** (chạy assistant): xem [part2_README.md](part2_README.md) — chạy local sau khi có checkpoint.
