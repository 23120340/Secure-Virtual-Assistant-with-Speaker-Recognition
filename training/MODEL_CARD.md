# Model Card — ECAPA-TDNN Speaker Embedding (Secure VA Tuần 1)

Theo định dạng model card của Mitchell et al. 2019 (FAccT). Mục tiêu: minh bạch về dataset, scope, hạn chế của model cho reviewer + người dùng downstream.

## Model details

| Trường | Giá trị |
|---|---|
| Tên model | ECAPA-TDNN (own checkpoint, Secure VA Phần 1) |
| Architecture | ECAPA-TDNN từ SpeechBrain `speechbrain.lobes.models.ECAPA_TDNN` |
| Input | Mel-filterbank 80 chiều, log, cepstral mean normalization (CMN) |
| Output | Embedding 192 chiều, L2-normalized |
| Loss khi train | AAM-Softmax (margin=0.2, scale=30) |
| Số parameters | ~6.2M (config `channels=[512,512,512,512,1536]`) |
| Tác giả/owner | Sinh viên đồ án cuối kỳ |
| License weights | Code SpeechBrain Apache-2.0; weight do sinh viên train, không claim license cụ thể |
| Phiên bản | Snapshot của `training/train_ecapa.py` tại commit chứa file này |
| Reproducibility | Seed=42, hparams + git SHA + torch version được dump vào `training/results/hparams.json` |

## Intended use

**Primary**: speaker identification (SID) + speaker verification (SV) cho ứng dụng trợ lý ảo nội bộ — chỉ user đã enroll giọng nói được phép thực hiện IMPORTANT intent.

**Out-of-scope**:
- KHÔNG dùng cho forensic speaker comparison có giá trị pháp lý.
- KHÔNG dùng để gate truy cập tài chính / dữ liệu y tế / dữ liệu công dân thực — model trained trên dataset nhỏ Indian subset, không calibrated cho audience Việt Nam.
- KHÔNG dùng làm yếu tố xác thực DUY NHẤT cho tác vụ critical. Voice biometric dễ bị replay/clone — phải đi kèm 2FA password hoặc challenge-response (xem `core/challenge.py`).

## Training data

| Trường | Giá trị |
|---|---|
| Dataset | VoxCeleb1 Indian subset (Kaggle `gaurav41/voxceleb1-audio-wav-files-for-india-celebrity`) |
| Số speaker | 24 Indian celebrities, speaker ID range `id10002` → `id11209` |
| Sampling rate | 16 kHz mono |
| Tổng số utterance | ~4857 (xem `training/data/iden_split.txt`) |
| Train/Val/Test split | 70/15/15 random utterance shuffle (seed=42) |
| Số trial pair SV | 552 (~276 positive + ~276 negative — xem `training/data/veri_test.txt`) |
| Augmentation | KHÔNG có (chưa add MUSAN / RIR / SpecAugment) |
| Audio duration | Random 3s crop khi train, full utterance khi extract embedding |

### Caveats data

1. **Random shuffle leak**: split không group theo `video_id` (YouTube session) → utterance cùng session có thể leak qua train/val/test, làm metric đẹp giả tạo. Đề chuẩn VoxCeleb1 là `iden_split.txt` per-utt với cùng speaker xuất hiện ở cả 3 split và `veri_test.txt` (VoxCeleb1-O) từ 40 speaker held-out — split của chúng tôi KHÔNG phải VoxCeleb1-O chính thức.
2. **Trial pair số lượng nhỏ**: 552 cặp → EER trên trial set này có sai số thống kê lớn (CI ~±2%).
3. **Demographic bias**: 24 Indian celebrities, không có thông tin chính thức về phân bố gender / age / accent. Tỷ lệ nam/nữ chưa cân bằng (estimate ~70/30 dựa trên public profile).

## Evaluation results

> ⚠️ Folder `training/results/` chứa artifact thực nghiệm khi sinh viên chạy training. Nếu folder rỗng, các con số dưới đây là **expected range** chứ KHÔNG phải kết quả đã đo.

| Metric | Expected (Indian subset, 15 epoch) |
|---|---|
| SID Top-1 accuracy | 95-99% |
| SID Top-5 accuracy | 99-100% |
| SV EER | 1-3% |
| SV minDCF (p=0.01, c_miss=c_fa=1) | 0.10-0.25 |

Threshold runtime đang dùng (xem `core/config.py`):
- ECAPA own checkpoint: `SV_THRESHOLD_OWN = 0.45`, `SID_MIN_THRESHOLD_OWN = 0.35`.
- SpeechBrain pretrained fallback (khác phân phối score): `0.25 / 0.20`.
- WavLM-SV alternative backend: `0.93 / 0.90`.

Thresholds CHƯA được calibrate trên dataset enroll thật — chỉ là default. Khuyến nghị chạy `python scripts/benchmark.py --out training/results/benchmark.json` sau khi enroll user thật rồi điều chỉnh.

## Quantitative analyses

Chưa thực hiện. Khi reviewer cần đánh giá bias:
- Per-speaker accuracy breakdown (24 speaker × confusion matrix).
- Per-gender accuracy (nếu phân loại được gender từ tên speaker).
- Per-duration analysis: accuracy vs audio length sau VAD trim.

## Ethical considerations

- **Biometric data sensitivity**: voice embedding là PII đặc biệt nhạy cảm (GDPR Art.9). Hệ thống lưu WAV mẫu + embedding 192-d trong SQLite local; user có thể yêu cầu xoá toàn bộ qua nút "Xoá tài khoản" (cascade: oauth_tokens + embeddings + WAV + user_files).
- **Replay attack**: thuần cosine SV KHÔNG defend được attacker có ghi âm. Mitigation: bật `CHALLENGE_RESPONSE_ENABLED=true` để buộc user đọc phrase random.
- **Voice cloning**: RVC/Tortoise/XTTS có thể clone giọng từ ~30s mẫu. Mitigation cùng challenge-response, không hoàn toàn defeat được cloning chất lượng cao.
- **Consent**: trang enroll có consent banner mô tả biometric data storage + retention. Yêu cầu user tick consent trước khi xin mic.

## Caveats and recommendations

1. **Trước khi nộp/demo**: chạy training pipeline trên dataset thật, commit `training/results/*.json` để reproducibility được verify.
2. **Trước khi deploy production**: re-train trên dataset large-scale (VoxCeleb1+2 full), add augmentation, re-calibrate threshold per environment.
3. **Cho user-facing app**: phải có disclaimer "Voice authentication is convenience, not a primary security factor" — không thay thế password.
4. **Cho audit**: tất cả SV/SID decision được log vào `data/audit.log` (JSON line, rotate 5MB×5) với `request_id` correlation.

## References

- Desplanques et al. *ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification.* Interspeech 2020.
- Mitchell et al. *Model Cards for Model Reporting.* FAT* 2019.
- SpeechBrain: <https://speechbrain.github.io/>
- VoxCeleb: <https://www.robots.ox.ac.uk/~vgg/data/voxceleb/>
