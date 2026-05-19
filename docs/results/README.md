# Training & Evaluation Results — Evidence cho báo cáo (YC1)

Đây là nơi commit các artifact thực nghiệm thực tế cần đi kèm báo cáo cuối kỳ.
Đề bài (`docs/Secure_Virtual_Assistant_with_Speaker_Recognition.pdf`) yêu cầu trình bày: bộ dữ liệu, split, mô hình, quy trình huấn luyện, độ đo và **kết quả thực nghiệm**. Repo hiện chỉ có "expected results" trong `training/README.md` — folder này phải có số liệu thật khi nộp.

## Checklist artifact cần commit

### Training-side (`training/results/`)

- [ ] `training_log.json` — output của `train_ecapa.py` (loss/accuracy mỗi epoch, đã có sẵn trong code, chỉ cần copy về).
- [ ] `spk2idx.json` — output của `train_ecapa.py` (mapping speaker → class index).
- [ ] `hparams.json` — argparse args + git SHA + python/torch version. Khuyến nghị bổ sung vào `train_ecapa.py` (1 dòng `json.dump`).
- [ ] `sid_results.json` — output của `evaluate_sid.py --out sid_results.json`. Schema: `{top1_accuracy, top5_accuracy, n_test, n_speakers}`.
- [ ] `sv_results.json` — output của `evaluate_sv.py --out sv_results.json`. Schema: `{eer_percent, threshold, min_dcf, n_trials}`.
- [ ] `roc_curve.png` (tùy chọn nhưng nên có) — plot FPR vs TPR + điểm EER. Code mẫu trong `training/README.md` tip #1.
- [ ] `confusion_matrix.png` (tùy chọn) — SID confusion 24×24.
- [ ] `loss_curves.png` (tùy chọn) — vẽ từ `training_log.json`.

### Runtime calibration (`docs/results/`)

- [ ] `benchmark_ecapa.json` — output của `python scripts/benchmark.py --out docs/results/benchmark_ecapa.json` trên audio enroll thật.
- [ ] `benchmark_wavlm.json` — tương tự cho `SPEAKER_BACKEND=wavlm`.
- [ ] `threshold_calibration.md` — bảng EER thật so với threshold mặc định trong `config.py`. Nếu lệch lớn, justify chọn threshold mới.

## Cách generate (Kaggle)

```bash
# Trên Kaggle notebook (xem training/README.md để biết toàn bộ pipeline)
!python train_ecapa.py --data_root ... --split_file ... --save_dir checkpoints --epochs 15
!python evaluate_sid.py --ckpt checkpoints/best_model.pt --spk2idx checkpoints/spk2idx.json \
    --data_root ... --split_file ... --out sid_results.json
!python evaluate_sv.py --ckpt checkpoints/best_model.pt \
    --data_root ... --trial_file ... --out sv_results.json

# Download zip → giải nén → copy training_log.json, spk2idx.json, sid_results.json,
# sv_results.json vào training/results/ ở local repo.
```

## Cách generate (local benchmark trên audio enroll thật)

```bash
# Sau khi enroll vài user qua web app, chạy:
python scripts/benchmark.py --out docs/results/benchmark_ecapa.json

# Đổi backend:
ASR_BACKEND=phowhisper SPEAKER_BACKEND=wavlm python scripts/benchmark.py \
    --out docs/results/benchmark_wavlm.json

# Đối chiếu 2 backend:
python scripts/benchmark.py --all --out docs/results/benchmark_all.json
```

## Tại sao folder này quan trọng

Reviewer/giảng viên không reproduce được training trên dataset của họ. Số liệu trong báo cáo phải đối chiếu được với artifact đã commit; nếu không có, dễ bị nghi ngờ "expected results" trong `training/README.md` là estimate không thực nghiệm.

Khi sẵn sàng nộp, xoá file `.gitkeep` (nếu có) trong các folder và đảm bảo các JSON/PNG ở trên đã commit.
