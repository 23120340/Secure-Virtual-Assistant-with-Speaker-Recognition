# scripts/

Tiện ích chạy 1 lần — benchmark, migration, re-enroll. Không phải module được import; chạy trực tiếp:

```bash
python scripts/<name>.py [args]
```

## `benchmark.py`

Đo **WER** (ASR) + **SID accuracy** + **SV TPR/FPR** trên audio enroll đã có trong `data/enroll_audio/<user>/sample_*.wav`. Dùng `ENROLL_PROMPTS` (xem `cli/enroll_user.py`) làm ground truth.

```bash
# Test combo hiện tại (theo .env)
python scripts/benchmark.py

# Test combo cụ thể
ASR_BACKEND=phowhisper SPEAKER_BACKEND=wavlm python scripts/benchmark.py

# Tất cả 4 combos (2 ASR × 2 Speaker)
python scripts/benchmark.py --all

# Chỉ ASR / chỉ Speaker
python scripts/benchmark.py --asr-only
python scripts/benchmark.py --speaker-only

# Lưu JSON chuẩn để đưa vào docs/results
python scripts/benchmark.py --all --out docs/results/benchmark_all.json
```

Output JSON có schema `secva.benchmark.v1`, metadata backend, summary WER/SID/SV và từng sample để phân tích lỗi/calibration. Output mẫu cũ xem `data/benchmark_session4.log`. Nếu speaker backend nào chưa enroll cho user nào → script báo và bỏ qua; chạy `reenroll_backend.py` trước.

## `export_nlu_dataset.py`

Xuất `data/nlu_training_candidates.jsonl` sang CSV để label thủ công rồi dùng cho train/evaluate NLU.

```bash
python scripts/export_nlu_dataset.py
python scripts/export_nlu_dataset.py --out data/nlu_training_candidates.csv
```

## `reenroll_backend.py`

Sau khi đổi `SPEAKER_BACKEND` (ví dụ `ecapa → wavlm`), embedding cũ trong DB không dùng được cho backend mới (dim khác, backend_id khác). Script này đọc audio gốc trong `data/enroll_audio/<user>/` và **encode lại** bằng encoder của backend đang chạy, lưu thành embedding mới — KHÔNG động vào embedding cũ.

```bash
# Re-encode tất cả user cho backend đang set trong .env
SPEAKER_BACKEND=wavlm python scripts/reenroll_backend.py

# Chỉ 1 user
SPEAKER_BACKEND=wavlm python scripts/reenroll_backend.py --user minh

# Liệt kê tình trạng, không encode
python scripts/reenroll_backend.py --dry-run
```

DB schema mới (cột `backend_id` trên `embeddings`) cho phép 1 user có nhiều embedding song song — đổi backend giữa các phiên không cần xóa DB.

## `migrate_password_hashes.py`

Migration một lần: SHA-256 unsalted (legacy) → PBKDF2-HMAC-SHA256 + salt. Quét user có `password_hash` dạng cũ (hex 64 ký tự, không có `$`) hoặc rỗng (backdoor cũ), reset password tạm thời và in ra console.

```bash
# Liệt kê trước, không sửa
python scripts/migrate_password_hashes.py --dry-run

# Migrate thật
python scripts/migrate_password_hashes.py
```

Hash SHA-256 không thể decode → password cũ không giữ được; admin báo user đổi lại sau lần đăng nhập đầu.

## `prepare_vivos_splits.py`

Chuẩn hóa VIVOS từ `train/waves/<speaker>` + `test/waves/<speaker>` về dạng
`<speaker>/<wav>.wav`, rồi tạo split SID và trial SV cho `training/train_ecapa.py`:

```bash
python scripts/prepare_vivos_splits.py \
  --vivos-root /kaggle/input/vivos-corpus/vivos \
  --out-root /kaggle/working/vivos_speaker_root \
  --split-out /kaggle/working/iden_split_vivos.txt \
  --trial-out /kaggle/working/veri_test_vivos.txt \
  --summary-out /kaggle/working/results/vivos/vivos_summary.json
```

## `build_training_comparison_report.py`

Đọc artifact training/evaluation của VoxCeleb Indian và VIVOS, sinh bảng Markdown:

```bash
python scripts/build_training_comparison_report.py \
  --voxceleb-dir training/results/voxceleb_indian \
  --vivos-dir training/results/vivos \
  --out docs/results/training_dataset_comparison.md
```

Artifact hiện đã có trong `training/results/`; chạy lại script này khi muốn refresh bảng sau một lần train/evaluate mới.
