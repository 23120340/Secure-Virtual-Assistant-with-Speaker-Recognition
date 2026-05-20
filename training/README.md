# Phần 1 – ECAPA-TDNN Speaker Recognition

Train ECAPA-TDNN trên VoxCeleb1 cho **cả** Speaker Identification (SID) và Speaker Verification (SV). Cùng một mô hình, hai cách đánh giá.

## Vị trí trong repo

```
Secure-Virtual-Assistant-with-Speaker-Recognition/
├── part1/                  ← thư mục này
│   ├── README.md
│   ├── train_ecapa.py
│   ├── evaluate_sid.py
│   ├── evaluate_sv.py
│   ├── requirements.txt
│   └── data/               # iden_split.txt, veri_test.txt (bản local)
├── checkpoints/            # đặt best_model.pt vào đây sau khi train
└── ...
```

## Kiến trúc & lý do chọn

- **ECAPA-TDNN (192-dim embedding)**: state-of-the-art baseline, đã được kiểm chứng rộng rãi. Dùng implementation từ SpeechBrain để tránh re-invent the wheel.
- **AAM-Softmax loss (margin=0.2, scale=30)**: chuẩn cho speaker embedding, tách lớp trong không gian góc cosine tốt hơn cross-entropy thường.
- **Mel-filterbank 80 chiều** + log + cepstral mean normalization.
- **Random crop 3s khi train**, full utterance khi extract embedding để verify.
- Sau khi train xong, dùng **cosine similarity** giữa hai embedding để verify.

## Dataset: VoxCeleb1 (Indian subset)

| Thông số | Giá trị |
|---|---|
| Dataset Kaggle | `gaurav41/voxceleb1-audio-wav-files-for-india-celebrity` |
| Số speaker | 24 (Indian celebrities, ID range `id10002` → `id11209`) |
| Sampling rate | 16kHz |
| Audio root | `/kaggle/input/datasets/gaurav41/voxceleb1-audio-wav-files-for-india-celebrity/vox1_indian/content/vox_indian` |
| Split file đã commit | `training/data/iden_split.txt` (4857 utterance), `training/data/veri_test.txt` (552 trial pair) |

**Lưu ý**: Dataset này không có sẵn `iden_split.txt` và `veri_test.txt` — cần tạo thủ công (xem bước 2 bên dưới). File đã commit trong `training/data/` là output của script bước 2 với seed=42, dùng để reproduce kết quả.

### Caveat methodology (cần ghi rõ trong báo cáo)

1. **Random shuffle utterance** — Split 70/15/15 chia theo utterance, KHÔNG group theo `video_id` (YouTube session). Hậu quả: utterance cùng một session có thể leak qua train/val/test → metric có thể đẹp hơn closed-set thật. Đề chuẩn của VoxCeleb1 là `iden_split.txt` per-utt với cùng speaker xuất hiện ở cả 3 split, và `veri_test.txt` (VoxCeleb1-O) là trial pair từ 40 speaker held-out. Phiên bản đang dùng KHÔNG phải VoxCeleb1-O chính thức.
2. **Trial pair số lượng nhỏ** — Cap positive ở `min(i+4, len(files))` mỗi speaker, sample 1 impostor mỗi cặp speaker → ~24×23/2 = 276 cặp âm. EER trên trial set nhỏ này ít nghĩa thống kê.
3. **Indian subset bias** — Train trên 24 Indian celebrity → có bias domain (accent, gender ratio, microphone). Khi deploy cho user Việt là OOD scenario — báo cáo cần discuss khoảng cách này.
4. **Augmentation cần retrain** — Đã có khung MUSAN noise / RIR reverb / SpecAugment / speed perturb, nhưng metric mới chỉ có ý nghĩa sau khi retrain trên Kaggle GPU và commit artifact EER/minDCF.

## Training augmentation (Task 2 - P2)

`train_ecapa.py` có pipeline augmentation tùy chọn, chỉ chạy lúc training:

| Augmentation | Vị trí | Flag |
|---|---|---|
| MUSAN noise | waveform, sau crop 3s | `--augment --musan-root /path/to/musan` |
| RIR reverb | waveform, sau crop 3s | `--augment --rir-root /path/to/rirs` |
| Speed perturb | waveform, sample 0.9x/1.0x/1.1x | `--augment --speed-rates 0.9 1.0 1.1` |
| SpecAugment | log-mel `[B,T,F]` trước ECAPA | `--augment --specaugment` |

Ví dụ train trên Kaggle khi đã add dataset MUSAN/RIR:

```bash
!python train_ecapa.py \
    --data_root /kaggle/input/datasets/gaurav41/voxceleb1-audio-wav-files-for-india-celebrity/vox1_indian/content/vox_indian \
    --split_file /kaggle/working/iden_split.txt \
    --save_dir /kaggle/working/checkpoints_aug \
    --epochs 20 --batch_size 64 --lr 1e-3 --num_workers 2 \
    --augment --specaugment \
    --musan-root /kaggle/input/musan \
    --rir-root /kaggle/input/rir-noises \
    --speed-rates 0.9 1.0 1.1 \
    --noise-prob 0.5 --rir-prob 0.3 --snr-min 5 --snr-max 20
```

Nếu chưa có MUSAN/RIR path, flag `--augment --specaugment --speed-rates 0.9 1.0 1.1`
vẫn chạy được: script sẽ bỏ qua noise/reverb khi không tìm thấy file audio. Đây là
pipeline retrain từ đầu trên GPU, không làm thay đổi runtime của assistant.

## Setup trên Kaggle

### 1. Chuẩn bị notebook

- Tạo notebook mới, bật **GPU T4 x1** (Settings → Accelerator).
- Add dataset `gaurav41/voxceleb1-audio-wav-files-for-india-celebrity`.
- Upload 4 file từ thư mục `part1/` lên notebook (dùng nút **Upload** trên Kaggle):
  - `train_ecapa.py`
  - `evaluate_sid.py`
  - `evaluate_sv.py`
  - `requirements.txt`

```bash
!pip install -q speechbrain
```

### 2. Tạo split files

Chạy cell Python này để tạo `iden_split.txt` và `veri_test.txt`:

```python
import os, random

random.seed(42)
DATA_ROOT = "/kaggle/input/datasets/gaurav41/voxceleb1-audio-wav-files-for-india-celebrity/vox1_indian/content/vox_indian"

all_files = []
for spk in sorted(os.listdir(DATA_ROOT)):
    spk_dir = os.path.join(DATA_ROOT, spk)
    if not os.path.isdir(spk_dir): continue
    for vid in os.listdir(spk_dir):
        vid_dir = os.path.join(spk_dir, vid)
        if not os.path.isdir(vid_dir): continue
        for f in os.listdir(vid_dir):
            if f.endswith('.wav'):
                all_files.append(f"{spk}/{vid}/{f}")

random.shuffle(all_files)
n = len(all_files)
n_train = int(0.70 * n)
n_val   = int(0.15 * n)

with open("/kaggle/working/iden_split.txt", "w") as f:
    for i, path in enumerate(all_files):
        if i < n_train:               split_id = 1
        elif i < n_train + n_val:     split_id = 2
        else:                         split_id = 3
        f.write(f"{split_id} {path}\n")

test_files = [all_files[i] for i in range(n_train + n_val, n)]
spk2files = {}
for f in test_files:
    spk = f.split("/")[0]
    spk2files.setdefault(spk, []).append(f)

pos, neg = [], []
spks = list(spk2files.keys())
for spk, files in spk2files.items():
    for i in range(len(files)):
        for j in range(i+1, min(i+4, len(files))):
            pos.append((1, files[i], files[j]))
for i in range(len(spks)):
    for j in range(i+1, len(spks)):
        f1 = random.choice(spk2files[spks[i]])
        f2 = random.choice(spk2files[spks[j]])
        neg.append((0, f1, f2))

k = min(len(pos), len(neg))
trials = random.sample(pos, k) + random.sample(neg, k)
random.shuffle(trials)

with open("/kaggle/working/veri_test.txt", "w") as f:
    for label, p1, p2 in trials:
        f.write(f"{label} {p1} {p2}\n")

print(f"Total: {n} | Train: {n_train} | Val: {n_val} | Test: {n-n_train-n_val}")
print(f"Trial pairs (SV): {len(trials)} | Speakers: {len(spk2files)}")
```

### 3. Train

```bash
!python train_ecapa.py \
    --data_root /kaggle/input/datasets/gaurav41/voxceleb1-audio-wav-files-for-india-celebrity/vox1_indian/content/vox_indian \
    --split_file /kaggle/working/iden_split.txt \
    --save_dir /kaggle/working/checkpoints \
    --epochs 15 --batch_size 64 --lr 1e-3 --num_workers 2
```

Với T4 x1 và batch 64, mỗi epoch ~5–10 phút. 15 epochs ≈ 1–2 giờ.

Metric logging optional:

```bash
# Local, miễn phí, không cần account
!python train_ecapa.py ... --tensorboard

# Cloud dashboard, cần pip install wandb + WANDB_API_KEY
!python train_ecapa.py ... --wandb --wandb-project secva-ecapa-tdnn
```

Nếu `tensorboard`/`wandb` chưa cài hoặc thiếu `WANDB_API_KEY`, `MetricLogger` sẽ warning rồi no-op; training vẫn chạy bình thường.

### 4. Kiểm tra output sau train

```python
import os
ckpt_dir = "/kaggle/working/checkpoints"
for f in os.listdir(ckpt_dir):
    size = os.path.getsize(os.path.join(ckpt_dir, f))
    print(f"  {f}  ({size/1024/1024:.1f} MB)")
# Kết quả mong đợi:
#   best_model.pt    (~20-40 MB)
#   spk2idx.json     (~0.0 MB)
#   training_log.json (~0.0 MB)
```

### 5. Đánh giá SID

```bash
!python evaluate_sid.py \
    --ckpt /kaggle/working/checkpoints/best_model.pt \
    --spk2idx /kaggle/working/checkpoints/spk2idx.json \
    --data_root /kaggle/input/datasets/gaurav41/voxceleb1-audio-wav-files-for-india-celebrity/vox1_indian/content/vox_indian \
    --split_file /kaggle/working/iden_split.txt
```

### 6. Đánh giá SV

```bash
!python evaluate_sv.py \
    --ckpt /kaggle/working/checkpoints/best_model.pt \
    --data_root /kaggle/input/datasets/gaurav41/voxceleb1-audio-wav-files-for-india-celebrity/vox1_indian/content/vox_indian \
    --trial_file /kaggle/working/veri_test.txt
```

### 7. Download checkpoint về local

```python
import shutil
shutil.make_archive("/kaggle/working/checkpoints_export", "zip", "/kaggle/working/checkpoints")
print("Tải file checkpoints_export.zip từ tab Output của Kaggle")
```

Giải nén và đặt `best_model.pt` + `spk2idx.json` vào thư mục `checkpoints/` ở root repo để Phần 2 sử dụng.

## Kết quả tham khảo

| Metric | Indian subset (~24 speakers, 15 epochs) | Full VoxCeleb1 (1251 speakers, 15 epochs) |
|---|---|---|
| SID Top-1 | ~95–99% | ~85–90% |
| SID Top-5 | ~99–100% | ~95–98% |
| SV EER | ~1–3% | ~3–5% |
| SV minDCF (p=0.01) | ~0.10–0.25 | ~0.30–0.45 |

Số liệu cao hơn full VoxCeleb1 vì dataset nhỏ hơn (~24 speakers).

## Tip để báo cáo "đẹp" hơn

1. **Vẽ ROC curve** cho SV: lưu `labels` và `scores` trong `evaluate_sv.py`, plot bằng matplotlib.
2. **Confusion matrix** cho SID.
3. **Loss/accuracy curves**: file `training_log.json` đã ghi sẵn, plot ra cho dễ nhìn.
4. **Phân tích lỗi**: sample vài cặp trial sai trong SV — gợi ý về độ dài audio, chất lượng thu âm.
5. **Ablation nhỏ** (nếu có thời gian): so sánh AAM-Softmax vs CE thường, hoặc 80 mel vs 64 mel.

## Fallback nếu compute không đủ

Dùng pretrained ECAPA-TDNN của SpeechBrain (đã train trên VoxCeleb2), bỏ qua bước train:

```python
from speechbrain.inference.speaker import EncoderClassifier
encoder = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_ecapa"
)
emb = encoder.encode_batch(wav)  # [B, 1, 192]
```

Chỉ chạy phần evaluation. Báo cáo trình bày là "fine-tune sẽ là future work".
