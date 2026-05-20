# Danh sách chức năng — Secure Virtual Assistant

Hệ thống hiện hỗ trợ **18 intents**, chia 3 nhóm theo `AuthLevel`
(định nghĩa tại [`core/intents.py`](../core/intents.py)).

## 🟢 NORMAL — không cần xác thực

| Intent | Chức năng | Ví dụ |
|---|---|---|
| `get_time` | Hỏi giờ hiện tại | "mấy giờ rồi" |
| `get_weather` | Hỏi thời tiết theo địa điểm | "thời tiết Hà Nội" |
| `tell_joke` | Kể chuyện cười | "kể chuyện cười đi" |
| `general_question` | Hỏi kiến thức chung (qua LLM/Gemini) | "thủ đô Pháp là gì" |

## 🔴 IMPORTANT — bắt buộc Speaker Verification (SV)

| Intent | Chức năng |
|---|---|
| `read_notes` | Đọc ghi chú/nhật ký đã lưu của user |
| `send_email` | Soạn + gửi email qua Gmail OAuth (có `email_flow` đa lượt: recipient → subject → body → confirm) |
| `check_balance` | Kiểm tra số dư tài khoản (mô phỏng) |
| `delete_data` | Xoá ghi chú/lịch/preferences cá nhân |
| `open_files` | Mở/xem/xoá/download file cá nhân đã upload |
| `add_note` | Thêm 1 ghi chú mới |
| `add_schedule` | Thêm 1 mục lịch/cuộc hẹn |
| `add_contact` | Thêm contact (tên + email) vào danh bạ |
| `set_reminder` | Đặt nhắc việc theo thời gian (relative "30 phút nữa" hoặc absolute "9h sáng mai") — đến giờ pop notification |

## 🟡 PERSONAL — cần Speaker Identification (SID) để cá nhân hoá

| Intent | Chức năng |
|---|---|
| `greet` | Chào hỏi — phản hồi theo tên người đang nói |
| `play_music` | Phát nhạc theo gu (genre) trong preferences của user (Deezer search + preview) |
| `show_schedule` | Hiển thị lịch cá nhân |
| `list_reminders` | Liệt kê reminder đang chờ/đến hạn |

## 🛡 Các tính năng hệ thống đi kèm (không phải intent)

- **Voice enroll/re-enroll** — record nhiều sample, tạo embedding ECAPA-TDNN + AS-Norm-lite + multi-window pooling
- **Incremental re-enroll** — bổ sung mẫu giọng vào centroid hiện có (không xoá lịch sử)
- **Challenge-response anti-replay** — IMPORTANT voice flow yêu cầu user đọc phrase 4-từ random (opt-in `CHALLENGE_RESPONSE_ENABLED`)
- **Text-mode fallback** — chat bằng text + mật khẩu khi không có mic (IMPORTANT block default, opt-in `ALLOW_PASSWORD_FOR_IMPORTANT`)
- **Reset mật khẩu qua Gmail** — gửi mã 6 chữ số đến Gmail đã OAuth
- **Admin panel** — list user, masked info, revoke voice + force re-enroll, lock/unlock account
- **Data export** — ZIP toàn bộ data của user (GDPR right to erasure/access)
- **Audit log + Turn log** — JSONL có redaction PII, schema versioned
- **Multi-language TTS** — gTTS (online) + pyttsx3 (offline fallback)

---

**Đặc điểm nổi bật so với Google Assistant/Siri**: tách rõ NORMAL/IMPORTANT/PERSONAL
→ action nguy hiểm (gửi email, xoá data, check balance) bắt buộc voice biometric,
không chỉ tin câu lệnh có giọng "khớp" sơ sơ.
