# Threshold Calibration Report

- **Sinh tự động bởi**: `scripts/build_calibration_report.py`
- **Nguồn dữ liệu**: audio enroll thật trong `data/enroll_audio/`
- **Generated at (nguồn JSON đầu tiên)**: `2026-05-19T16:26:46+00:00`
- **Speaker backends báo cáo**: ecapa
- **ASR backends báo cáo**: faster-whisper

## 1. Speaker Verification — Calibration

### 1.1 Tổng hợp ở threshold mặc định (trong `core/config.py`)

| Backend | n_users | n_audio | SID acc | SV thr | TPR | FPR | n_pos | n_neg |
|---------|--------:|--------:|--------:|-------:|----:|----:|------:|------:|
| `ecapa` | 5 | 15 | 1.000 | 0.450 | 1.000 | 0.000 | 15 | 60 |

### 1.2 Threshold sweep (tìm EER + recommendation)

#### Backend `ecapa`

- **EER**: 0.00%
- **Threshold tại EER**: **0.05**
- **Threshold đang dùng (`config.py`)**: 0.45
- **n_pos_trials** (genuine): 15
- **n_neg_trials** (max-impostor): 0

⚠️ **Threshold đang CHẶT** hơn EER: hiện 0.45 > EER 0.05. User thật có thể bị reject. Cân nhắc hạ xuống 0.05 nếu false-reject rate cao trong demo.

**Sweep table** (TPR/FPR theo threshold, step 0.05):

| Threshold | TPR | FPR | FNR | |FPR-FNR| |
|----------:|----:|----:|----:|---------:|
| 0.050 | 1.000 | 0.000 | 0.000 | 0.000 | ← EER
| 0.100 | 1.000 | 0.000 | 0.000 | 0.000 |
| 0.150 | 1.000 | 0.000 | 0.000 | 0.000 |
| 0.200 | 1.000 | 0.000 | 0.000 | 0.000 |
| 0.250 | 1.000 | 0.000 | 0.000 | 0.000 |
| 0.300 | 1.000 | 0.000 | 0.000 | 0.000 |
| 0.350 | 1.000 | 0.000 | 0.000 | 0.000 |
| 0.400 | 1.000 | 0.000 | 0.000 | 0.000 |
| 0.450 | 1.000 | 0.000 | 0.000 | 0.000 | ← config
| 0.500 | 1.000 | 0.000 | 0.000 | 0.000 |
| 0.550 | 1.000 | 0.000 | 0.000 | 0.000 |
| 0.600 | 1.000 | 0.000 | 0.000 | 0.000 |
| 0.650 | 1.000 | 0.000 | 0.000 | 0.000 |
| 0.700 | 1.000 | 0.000 | 0.000 | 0.000 |
| 0.750 | 0.733 | 0.000 | 0.267 | 0.267 |
| 0.800 | 0.533 | 0.000 | 0.467 | 0.467 |
| 0.850 | 0.400 | 0.000 | 0.600 | 0.600 |
| 0.900 | 0.067 | 0.000 | 0.933 | 0.933 |
| 0.950 | 0.000 | 0.000 | 1.000 | 1.000 |

### 1.3 SID Confusion Matrix

#### Backend `ecapa`

| True \ Pred | `Test00` | `lan` | `quan` |
|---|---:|---:|---:|
| `Test00` | 5 | 0 | 0 |
| `lan` | 0 | 5 | 0 |
| `quan` | 0 | 0 | 5 |

## 2. ASR — Word Error Rate (WER)

Ground truth: 5 prompts từ enrollment (`ENROLL_PROMPTS` trong `scripts/benchmark.py`). WER tính theo word-level Levenshtein, normalize lowercase + bỏ punctuation.

| Backend | n_samples | WER mean | WER median | Latency mean (s) |
|---------|----------:|---------:|-----------:|-----------------:|
| `faster-whisper` | 20 | 0.768 | 1.000 | 1.25 |

## 3. Methodology caveats (PHẢI ghi vào báo cáo)

- **Dataset rất nhỏ**: chỉ 5 user enrolled, 15 audio total. EER/TPR/FPR ở đây là **indicative**, KHÔNG so sánh trực tiếp với benchmark VoxCeleb hay NIST SRE.
- **Same-session bias**: cả 5 mẫu/user thu liên tiếp cùng 1 session, cùng mic, cùng noise floor — score genuine sẽ optimistic so với enrollment-vs-future-session.
- **Impostor convention**: dùng "max-impostor per audio" (chọn user gần giọng nhất trong số non-target). Conservative cho FPR — phản ánh threat model attacker chọn user gần giọng họ nhất.
- **Ground truth ASR cho user có `sample_0..4`** (thay vì `sample_1..5`): mapping prompt lệch 1 → WER user đó có thể cao bất thường, không phản ánh ASR quality.
- **Threshold recommendation chỉ là gợi ý** từ EER trên dataset hiện tại; production cần audio cross-session + augmentation noise/reverb để robust.
