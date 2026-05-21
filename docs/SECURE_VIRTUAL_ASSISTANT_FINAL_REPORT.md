# Secure Virtual Assistant with Speaker Recognition - Final Report

## 1. Tóm Tắt Đồ Án

Đồ án xây dựng một trợ lý ảo bảo mật có khả năng nhận lệnh bằng giọng nói, nhận diện người nói, xác minh người nói trước các tác vụ nhạy cảm, và cá nhân hóa phản hồi theo từng người dùng. Hệ thống gồm hai phần chính:

- **YC1 - Speaker Recognition**: huấn luyện và đánh giá ECAPA-TDNN cho Speaker Identification (SID) và Speaker Verification (SV).
- **YC2 - Secure Virtual Assistant**: tích hợp speaker recognition vào web assistant, phân quyền intent theo mức `NORMAL`, `PERSONAL`, `IMPORTANT`.

Runtime hiện dùng Flask web app, PhoWhisper/faster-whisper cho ASR, ECAPA-TDNN cho speaker embedding, Gemini cho NLU/general question khi API key còn quota, cùng các fallback cần thiết cho demo.

## 2. Kiến Trúc Hệ Thống

### 2.1 Pipeline Runtime

Luồng xử lý một lượt voice command:

1. Browser ghi âm và gửi audio lên `/api/assistant/turn`.
2. ASR chuyển audio thành transcript tiếng Việt.
3. NLU phân loại intent và extract entities.
4. Speaker module sinh embedding, thực hiện SID/SV khi cần.
5. Router kiểm tra policy theo auth level.
6. Handler thực thi tác vụ.
7. TTS đọc phản hồi về browser.

### 2.2 Nhóm Intent Và Bảo Mật

| Nhóm | Ý nghĩa | Ví dụ | Chính sách |
|---|---|---|---|
| `NORMAL` | Không cần xác thực | hỏi giờ, hỏi thời tiết, câu hỏi tổng quát | ai cũng dùng được |
| `PERSONAL` | Cá nhân hóa theo user | phát nhạc, chào người dùng | dùng SID nếu có |
| `IMPORTANT` | Có dữ liệu nhạy cảm hoặc tác vụ thay đổi dữ liệu | gửi email, đọc/xóa ghi chú, lịch, file, số dư, reminder | cần SV hoặc password fallback ở text mode |

Thiết kế này giúp assistant không chỉ nhận lệnh mà còn kiểm soát rủi ro: cùng một câu lệnh có thể bị chặn nếu không xác thực được người nói.

## 3. Speaker Recognition

### 3.1 Mô Hình

Mô hình huấn luyện chính là ECAPA-TDNN với:

- Input feature: 80-dim log Mel filterbank + cepstral mean normalization.
- Embedding dimension: 192.
- Loss: Additive Angular Margin Softmax (AAM-Softmax), margin `0.2`, scale `30`.
- Training crop: 3 giây.
- Verification score: cosine similarity giữa hai speaker embeddings.

Mô hình được train supervised theo speaker label cho SID, sau đó reuse embedding cho SV.

### 3.2 Dataset Thực Nghiệm

Đồ án dùng hai dataset để tạo evidence:

| Dataset | Domain | Speakers | Utterances | Train / Val / Test | Trial pairs |
|---|---|---:|---:|---:|---:|
| VoxCeleb Indian subset | celebrity speech, English/Indian accent | 24 | 4,857 | 3,389 / 717 / 751 | 552 |
| VIVOS | Vietnamese read speech | 65 | 12,419 | 8,686 / 1,850 / 1,883 | 4,160 |

VIVOS gần domain tiếng Việt của ứng dụng hơn, trong khi VoxCeleb Indian có domain lệch hơn nhưng vẫn hữu ích để so sánh khả năng học speaker embedding trên dữ liệu celebrity speech.

### 3.3 Làm Sạch Dữ Liệu

Với VIVOS, quá trình chuẩn bị phát hiện một file WAV không decode được bởi `torchaudio`:

```text
VIVOSSPK44/train_VIVOSSPK44_157.wav
```

File này được loại khỏi split và lưu vào:

```text
training/results/vivos/bad_audio_files.txt
```

Với VoxCeleb Indian, split được tạo lại từ chính các file tồn tại trên Kaggle để tránh lỗi split cũ trỏ tới file không có.

### 3.4 Kết Quả Training Và Evaluation

Hai thí nghiệm dùng cùng cấu hình:

| Config | Giá trị |
|---|---|
| Epochs | 15 |
| Batch size | 64 |
| Learning rate | 0.001 |
| Sample rate | 16 kHz |
| Crop duration | 3.0s |
| Device | CUDA |
| Augmentation chính | Không dùng trong số liệu chính |

Kết quả:

| Dataset | Best Val Acc | SID Top-1 | SID Top-5 | SV EER | minDCF | Threshold @ EER |
|---|---:|---:|---:|---:|---:|---:|
| VoxCeleb Indian | 93.44% | 91.34% | 98.67% | 2.90% | 0.1486 | 0.3884 |
| VIVOS | 99.19% | 99.31% | 100.00% | 0.96% | 0.0673 | 0.4062 |

### 3.5 Nhận Xét Kết Quả

VIVOS cho kết quả tốt hơn rõ rệt trên cả SID và SV:

- SID Top-1 tăng từ **91.34%** lên **99.31%**.
- SV EER giảm từ **2.90%** xuống **0.96%**.
- minDCF giảm từ **0.1486** xuống **0.0673**.

Nguyên nhân hợp lý:

- VIVOS là tiếng Việt, gần domain người dùng demo hơn.
- VIVOS có nhiều speaker hơn trong thực nghiệm này: 65 so với 24.
- Trial set VIVOS lớn hơn: 4,160 pair so với 552 pair.
- Read speech trong VIVOS có thể sạch và ổn định hơn so với celebrity speech nhiều biến thiên.

Tuy nhiên, EER/minDCF ở đây là metric nội bộ từ split tự tạo, không phải benchmark chuẩn VoxCeleb1-O hoặc NIST SRE. Do đó số liệu dùng để chứng minh pipeline và so sánh nội bộ, không claim ngang các paper benchmark.

## 4. Augmentation

Codebase đã có khung augmentation trong training pipeline:

| Augmentation | Trạng thái |
|---|---|
| MUSAN noise | Đã có hook code, chưa đưa vào metric chính |
| RIR reverb | Đã có hook code, chưa đưa vào metric chính |
| SpecAugment | Đã có hook code |
| Speed perturb | Đã có hook code |

MUSAN/RIR không được đưa vào kết quả chính vì không có dataset noise/RIR hợp lệ trong môi trường Kaggle hiện tại. Báo cáo không claim gain từ MUSAN/RIR. Trọng tâm thực nghiệm được chuyển sang so sánh domain dataset: VoxCeleb Indian vs VIVOS.

## 5. Calibration Runtime

Ngoài training evidence, hệ thống có benchmark runtime trên audio enroll thật trong `docs/results/benchmark_ecapa.json`, sinh báo cáo:

```text
docs/results/threshold_calibration.md
```

Tóm tắt calibration ECAPA runtime:

| Backend | Users | Audio | SID Acc | SV Threshold | TPR | FPR |
|---|---:|---:|---:|---:|---:|---:|
| ECAPA | 5 | 15 | 1.000 | 0.450 | 1.000 | 0.000 |

Threshold sweep cho thấy EER trên bộ enroll demo là 0.00%, nhưng dataset rất nhỏ và same-session, nên chỉ dùng làm tín hiệu demo. Threshold production cần thêm audio cross-session, nhiều user hơn, và noise/reverb thực tế.

## 6. Virtual Assistant Features

Các chức năng chính của assistant:

- Enroll user bằng nhiều mẫu giọng nói.
- Đăng nhập/xác minh bằng password fallback cho text mode.
- Voice assistant và text assistant.
- Chạy checkpoint tự train nếu có; fallback SpeechBrain pretrained nếu thiếu checkpoint.
- PhoWhisper/faster-whisper ASR.
- NLU bằng Gemini function calling, rule-based fallback.
- General question bằng Gemini, có offline fallback cho câu demo phổ biến và thông báo rõ khi API quota/key lỗi.
- Gửi email qua Gmail OAuth.
- Quản lý ghi chú, lịch, reminder, file cá nhân.
- Phát nhạc, playlist cá nhân.
- TTS qua gTTS, có fallback offline.
- Turn log JSONL và NLU candidate log để audit/debug.

## 7. Các Lỗi Đã Xử Lý Trong Quá Trình Hoàn Thiện

Một số vấn đề thực tế đã được phát hiện và xử lý:

- Windows symlink issue của SpeechBrain/HuggingFace: chuyển fetch strategy sang copy.
- PhoWhisper safetensors conversion warning/403 noise: giảm log noise và cấu hình env phù hợp.
- `general_question` bị nhầm sang `get_weather`: thêm guard cho câu hỏi giải thích như “vì sao trời mưa”.
- Gemini quota/API key lỗi: thêm thông báo rõ nguyên nhân và offline fallback.
- Reminder đã set nhưng không báo: thêm polling frontend cho due reminders.
- Player bấm `X` bị chuyển bài: chặn audio error handler khi người dùng chủ động đóng player.
- VIVOS có WAV lỗi decode: lọc file lỗi trước khi tạo split.
- VoxCeleb split cũ mismatch file Kaggle: tạo split mới từ file tồn tại thực tế.

## 8. Evidence Artifacts

Các artifact chính hiện nằm ở:

```text
training/results/voxceleb_indian/
training/results/vivos/
docs/results/training_dataset_comparison.md
docs/results/threshold_calibration.md
docs/results/benchmark_ecapa.json
```

Các file quan trọng:

| File | Vai trò |
|---|---|
| `hparams.json` | cấu hình train, seed, torch/python version |
| `training_log.json` | loss/accuracy theo epoch |
| `spk2idx.json` | mapping speaker to class index |
| `sid_results.json` | Top-1/Top-5 SID |
| `sv_results.json` | EER/minDCF/threshold |
| `iden_split.txt` | split train/val/test |
| `veri_test.txt` | trial pairs cho SV |
| `best_model.pt` | checkpoint train tốt nhất |

## 9. Hạn Chế

- Split train/val/test tự tạo, chưa phải protocol chuẩn chính thức.
- VIVOS là read speech, có thể sạch hơn môi trường thực tế.
- Runtime calibration mới có 5 user, 15 audio, same-session.
- Chưa đánh giá cross-session và cross-device đầy đủ.
- MUSAN/RIR chưa có metric vì thiếu dataset phụ.
- General question phụ thuộc Gemini API quota nếu câu hỏi ngoài offline fallback.

## 10. Kết Luận

Đồ án đã hoàn thiện pipeline từ training ECAPA-TDNN đến tích hợp vào secure virtual assistant. Evidence trên hai dataset cho thấy VIVOS phù hợp hơn với bài toán tiếng Việt: SID Top-1 đạt **99.31%**, SV EER đạt **0.96%**, tốt hơn VoxCeleb Indian subset trong thiết lập nội bộ.

Với kết quả này, checkpoint VIVOS là lựa chọn hợp lý hơn để dùng cho demo tiếng Việt. Hệ thống vẫn cần mở rộng benchmark cross-session, nhiều user hơn, và augmentation noise/reverb khi có MUSAN/RIR để tăng độ tin cậy trong môi trường thực.

## 11. Tài Liệu Giữ Riêng

Một số file Markdown vẫn nên giữ riêng vì là tài liệu vận hành hoặc artifact chuyên biệt:

- `README.md`: hướng dẫn repo tổng quan.
- `web/README.md`: hướng dẫn web app.
- `training/README.md`: hướng dẫn train/evaluate ECAPA.
- `scripts/README.md`: hướng dẫn các script tiện ích.
- `cli/README.md`: hướng dẫn CLI.
- `training/MODEL_CARD.md`: model card và caveat.
- `docs/results/training_dataset_comparison.md`: report bảng sinh tự động từ artifact.
- `docs/results/threshold_calibration.md`: report calibration sinh tự động từ benchmark.
