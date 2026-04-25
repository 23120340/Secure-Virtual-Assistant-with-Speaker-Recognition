# Phần 1 – ECAPA-TDNN Speaker Recognition

Train ECAPA-TDNN trên VoxCeleb1 cho **cả** Speaker Identification (SID) và Speaker Verification (SV). Cùng một mô hình, hai cách đánh giá.

## Kiến trúc & lý do chọn

- **ECAPA-TDNN (192-dim embedding)**: state-of-the-art baseline, đã được kiểm chứng rộng rãi. Dùng implementation từ SpeechBrain để tránh re-invent the wheel.
- **AAM-Softmax loss (margin=0.2, scale=30)**: chuẩn cho speaker embedding, tách lớp trong không gian góc cosine tốt hơn cross-entropy thường.
- **Mel-filterbank 80 chiều** + log + cepstral mean normalization.
- **Random crop 3s khi train**, full utterance khi extract embedding để verify.
- Sau khi train xong, dùng **cosine similarity** giữa hai embedding để verify.

## Dataset: VoxCeleb1

| Thông số | Giá trị |
|---|---|
| Số speaker | 1251 |
| Số utterance | ~150K |
| File split SID | `iden_split.txt` (1=train, 2=val, 3=test, cùng tập speaker — closed-set) |
| File trial SV | `veri_test.txt` (37,720 cặp, label 1=cùng người, 0=khác) |
| Sampling rate | 16kHz |

**Lấy dataset trên Kaggle**: thêm dataset `nghiapickatlu/voxceleb1` (hoặc tương đương) vào notebook. Hai file split sẽ ở `/kaggle/input/<dataset>/iden_split.txt` và `veri_test.txt`. Audio ở `/kaggle/input/<dataset>/wav/`.

## Setup trên Kaggle (khuyến nghị)

1. Tạo notebook mới, bật **GPU T4 ×2** (Settings → Accelerator).
2. Add VoxCeleb1 dataset.
3. Trong cell đầu:

```bash
!pip install -q speechbrain
!git clone https://github.com/<your-repo>/speaker-recognition.git
%cd speaker-recognition
```

Hoặc upload thẳng 3 file `.py` này vào notebook nếu chưa có repo.

4. Train (giảm `epochs` nếu hết quota):

```bash
!python train_ecapa.py \
    --data_root /kaggle/input/voxceleb1/wav \
    --split_file /kaggle/input/voxceleb1/iden_split.txt \
    --save_dir /kaggle/working/checkpoints \
    --epochs 15 --batch_size 128 --lr 1e-3 --num_workers 2
```

Với T4 ×2 và batch 128, mỗi epoch ~25–35 phút trên VoxCeleb1 train (~138K samples). 15 epochs ≈ 6–9 giờ. Nếu quota chật, có thể train 8–10 epochs cũng đủ ra số đẹp.

5. Đánh giá SID:

```bash
!python evaluate_sid.py \
    --ckpt /kaggle/working/checkpoints/best_model.pt \
    --spk2idx /kaggle/working/checkpoints/spk2idx.json \
    --data_root /kaggle/input/voxceleb1/wav \
    --split_file /kaggle/input/voxceleb1/iden_split.txt
```

6. Đánh giá SV:

```bash
!python evaluate_sv.py \
    --ckpt /kaggle/working/checkpoints/best_model.pt \
    --data_root /kaggle/input/voxceleb1/wav \
    --trial_file /kaggle/input/voxceleb1/veri_test.txt
```

## Kết quả tham khảo

| Metric | Train from scratch (15 epochs, VoxCeleb1) | Pretrained SpeechBrain (VoxCeleb2) |
|---|---|---|
| SID Top-1 | ~85–90% | – |
| SID Top-5 | ~95–98% | – |
| SV EER | ~3–5% | ~0.9% |
| SV minDCF (p=0.01) | ~0.30–0.45 | ~0.10 |

Số liệu thực tế phụ thuộc batch size, augmentation, số epoch. Train 20+ epochs với SpecAugment thường ra EER ~2.5%.

## Tip để báo cáo "đẹp" hơn

1. **Vẽ ROC curve** cho SV: lưu `labels` và `scores` trong `evaluate_sv.py`, plot bằng matplotlib.
2. **Confusion matrix top-10 speakers** cho SID.
3. **Loss/accuracy curves**: file `training_log.json` đã ghi sẵn, plot ra cho dễ nhìn.
4. **Phân tích lỗi**: sample vài cặp trial sai trong SV (cosine cao nhưng khác người, hoặc cosine thấp nhưng cùng người) — gợi ý về độ dài audio, chất lượng thu âm, cross-gender, v.v.
5. **Ablation nhỏ** (nếu có thời gian): so sánh AAM-Softmax vs CE thường, hoặc 80 mel vs 64 mel.

## Fallback nếu compute không đủ

Dùng pretrained ECAPA-TDNN của SpeechBrain (đã train trên VoxCeleb2):

```python
from speechbrain.inference.speaker import EncoderClassifier
encoder = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_ecapa"
)
emb = encoder.encode_batch(wav)  # [B, 1, 192]
```

Và **chỉ làm phần evaluation** (không train). Báo cáo trình bày là "fine-tune sẽ là future work" hoặc fine-tune nhẹ vài epoch trên VoxCeleb1 dev. Tuy nhiên nếu bạn có Kaggle GPU thì train được — full pipeline luôn ấn tượng hơn cho điểm.

## Files

- `train_ecapa.py` — training với AAM-Softmax
- `evaluate_sid.py` — Top-1/Top-5 accuracy
- `evaluate_sv.py` — EER + minDCF trên trial pairs
- `requirements.txt` — dependencies
