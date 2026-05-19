# Secure Virtual Assistant — Review tổng hợp & Todo-list

Ngày review (consolidated): 2026-05-19
Lần cập nhật gần nhất: 2026-05-19 (scope guard theo PDF + 4 feature §4.4)

> Tài liệu này hợp nhất 4 nguồn:
> 1. Đề bài gốc: `docs/Secure_Virtual_Assistant_with_Speaker_Recognition.pdf`
> 2. Master review v2→v5: `SecureVA-MASTER-REVIEW.md` (~80 issues, đã fix một phần)
> 3. Bản review 2026-05-19 trước đó (đã sửa Path traversal mục #1)
> 4. Audit chuyên sâu 3 hướng (ML/Pipeline/Security) — chạy lại 2026-05-19
>
> Đọc theo thứ tự: phần 1 (đối chiếu đề) → phần 2 (chất lượng theo tiêu chí big-tech) → phần 3 (đề xuất tính năng) → phần 4 (todo-list ưu tiên).
>
> **Scope guard theo PDF**:
> - Bắt buộc chấm điểm nằm ở §1: YC1 train/evaluate speaker model và YC2 trợ lý ảo tích hợp SV/SID.
> - Các mục §2–§4 mang nhãn `big-tech`, P1/P2/P3 hoặc "wow factor" là khuyến nghị để demo chắc hơn, **không phải yêu cầu bắt buộc của PDF**.
>
> **Tiến độ tổng quát**:
> - **P0 thật sự trước nộp**: còn 2 việc cần chứng minh bằng artifact/log — training evidence và calibration/benchmark.
> - **P1/P2/P3**: optional polish/production-grade; chỉ làm khi không tăng rủi ro demo.
> - **§4.4 feature**: đã thêm 3 bidirectional intent + multi-language TTS + incremental re-enroll.
> - **Tests**: 97 unit + integration test pytest, chạy `./venv/Scripts/python.exe -m pytest tests/ --basetemp .pytest_tmp`.

---

## Mục lục

- [1. Đối chiếu đề bài](#1-đối-chiếu-đề-bài)
  - [1.1 Yêu cầu 1 — Train & evaluate speaker model (5 điểm)](#11-yêu-cầu-1--train--evaluate-speaker-model-5-điểm)
  - [1.2 Yêu cầu 2 — Trợ lý ảo tích hợp SV/SID (5 điểm)](#12-yêu-cầu-2--trợ-lý-ảo-tích-hợp-svsid-5-điểm)
- [2. Review theo tiêu chí big-tech](#2-review-theo-tiêu-chí-big-tech)
  - [2.1 Security](#21-security)
  - [2.2 Reliability & Observability](#22-reliability--observability)
  - [2.3 Maintainability & Code quality](#23-maintainability--code-quality)
  - [2.4 ML/Product Quality](#24-mlproduct-quality)
  - [2.5 Compliance & Privacy](#25-compliance--privacy)
- [3. Đề xuất cải tiến tính năng](#3-đề-xuất-cải-tiến-tính-năng)
  - [3.1 Đào sâu chức năng đã có](#31-đào-sâu-chức-năng-đã-có)
  - [3.2 Thêm chức năng mới (showcase tốt)](#32-thêm-chức-năng-mới-showcase-tốt)
- [4. Todo-list theo ưu tiên](#4-todo-list-theo-ưu-tiên)

---

## 1. Đối chiếu đề bài

### 1.1 Yêu cầu 1 — Train & evaluate speaker model (5 điểm)

**Mức đáp ứng: khá tốt về kỹ thuật, nhưng thiếu evidence thực nghiệm — rủi ro mất điểm khi phản biện.**

#### Đã có

- **Kiến trúc đúng chuẩn**: ECAPA-TDNN (`input_size=80, lin_neurons=192, channels=[512,512,512,512,1536]`) ở [training/train_ecapa.py:188-189](../training/train_ecapa.py#L188-L189). AAM-Softmax `margin=0.2, scale=30` ở [training/train_ecapa.py:32-55](../training/train_ecapa.py#L32-L55) — đúng community standard.
- **Feature pipeline đúng**: 80-mel filterbank + log + CMN, 16kHz ở [training/train_ecapa.py:119-132](../training/train_ecapa.py#L119-L132).
- **Train/eval discipline**: random 3s crop train, center crop eval, full-utt embedding ở SV time, L2-normalize trước cosine. `spk2idx` share giữa train/val.
- **Optimization**: Adam + weight decay 2e-5, gradient clipping 5.0, Cosine LR scheduler, best-on-val checkpointing, seed=42.
- **Đánh giá đủ 2 task**:
  - SID Top-1/Top-5 tại [training/evaluate_sid.py:49-62](../training/evaluate_sid.py#L49-L62).
  - SV EER + minDCF (p_target=0.01, NIST convention) tại [training/evaluate_sv.py:29-44](../training/evaluate_sv.py#L29-L44).
- **Runtime fallback**: code có SpeechBrain ECAPA pretrained khi không có own ckpt, và thêm backend WavLM-SV (`microsoft/wavlm-base-plus-sv`).

#### Thiếu/rủi ro cần xử lý đúng scope

1. **Thiếu artifact kết quả thật trong repo**. Cần có `training_log.json`, `sid_results.json`, `sv_results.json`, `spk2idx.json`, ROC/DET hoặc confusion matrix trong `training/results/` hoặc `docs/results/`. Đây là blocker theo PDF vì phần train/evaluate chiếm 5 điểm.
2. **Dataset/split đã được làm rõ là 24 Indian speakers**, nhưng methodology vẫn cần caveat trong báo cáo: split random theo utterance có thể leak cùng session qua train/val/test; trial pair nhỏ nên EER/minDCF chỉ là minh hoạ nội bộ, không nên claim như VoxCeleb1-O chuẩn.
3. **Threshold mặc định vẫn cần calibration evidence**: `SV_THRESHOLD_OWN = 0.45`, `SV_THRESHOLD_PRETRAINED = 0.25`, `SV_THRESHOLD_WAVLM = 0.93` phải đi kèm benchmark trên audio enroll/demo thật nếu dùng để bảo vệ IMPORTANT task.
4. **Augmentation, TensorBoard/Wandb, Git LFS/Drive, model card** đều tốt cho chất lượng báo cáo hoặc repo hygiene, nhưng không phải blocker bắt buộc nếu thời gian không cho phép.

**Kết luận YC1**: đủ khung kỹ thuật, nhưng trước khi nộp phải ưu tiên **evidence thực nghiệm thật**. Các cải tiến nghiên cứu sâu nên để optional nếu chúng làm tăng thời gian.

---

### 1.2 Yêu cầu 2 — Trợ lý ảo tích hợp SV/SID (5 điểm)

**Mức đáp ứng: tốt — bao phủ đủ checklist của đề.**

| Checklist đề bài | Bằng chứng | Verdict |
|---|---|---|
| Voice interaction (ASR + TTS) | `core/asr.py` faster-whisper + PhoWhisper, hallucination filter [asr.py:48](../core/asr.py#L48); `core/tts.py` gTTS MP3 stream | ✅ |
| Enrollment UI (web) thu mẫu giọng nói | `web/app.py:357-462` GET `/enroll` + POST `/api/enroll` multipart `sample_0..N`, VAD trim mỗi sample, atomic temp-dir → move | ✅ |
| User management | CRUD ở `web/app.py:350-525`, DB ở `core/database.py:178-298`, admin panel ở `app.py:1262-1282` | ✅ |
| ≥ 1 NORMAL task | 4 intent: `get_time`, `get_weather`, `tell_joke`, `general_question` ở [core/intents.py:22-95](../core/intents.py#L22-L95) | ✅ |
| ≥ 1 IMPORTANT task (SV) | 5 intent: `read_notes`, `send_email`, `check_balance`, `delete_data`, `open_files` ở [core/intents.py:97-195](../core/intents.py#L97-L195) | ✅ |
| ≥ 1 PERSONAL task (SID) | 3 intent: `greet`, `play_music`, `show_schedule` ở [core/intents.py:197-255](../core/intents.py#L197-L255) | ✅ |
| Mô tả enrollment flow | Browser N×5s blob → `decode_browser_audio` → Silero VAD → preprocess → `encode_centroid` (quality filter) → DB BLOB + WAV/`data/enroll_audio/` | ✅ |
| Mô tả lưu trữ | SQLite `data/users.db` tables `users/embeddings/oauth_tokens`, schema document ở `database.py:1-7` | ✅ |
| Kiến trúc tổng thể | Có sơ đồ pipeline trong README chính | ✅ |

**Điểm sáng**:
- 3 lớp `AuthLevel` (NORMAL/IMPORTANT/PERSONAL) gọn, dễ extend.
- Multi-backend (ECAPA / WavLM, faster-whisper / PhoWhisper) chuyển qua env, DB schema multi-backend không cần re-enroll.
- Pipeline graceful degradation: Gemini fail → rule-based NLU; OWN ckpt missing → SpeechBrain pretrained; TTS fail → 204 not bogus bytes.
- Fallback nhiều tầng (mục README #21).

**Điểm yếu cần đề cập trong báo cáo** (không sai, nhưng tránh hiểu lầm khi phản biện):
- Một số chức năng vẫn là **demo/stub**: `handle_get_weather` random ở [core/handlers.py:60](../core/handlers.py#L60); `handle_check_balance` lấy từ `preferences.balance` tĩnh. Nên ghi rõ trong báo cáo là "demo use case" thay vì "tích hợp dịch vụ thật".
- **Text mode IMPORTANT đã được block mặc định** qua `ALLOW_PASSWORD_FOR_IMPORTANT=false`; nếu bật fallback bằng password khi demo thì phải ghi rõ đó là convenience/demo mode, không phải biometric security gate.
- **SV/SID dùng cùng speaker embedding** nên vẫn cần giải thích threat model. Challenge-response đã có opt-in cho IMPORTANT voice flow, nhưng replay/anti-spoofing hoàn chỉnh vẫn là giới hạn nghiên cứu nếu không bật hoặc không demo.

**Kết luận YC2**: đáp ứng checklist đề bài. Việc còn lại trước nộp là chứng minh bằng demo/log và tránh claim các stub là dịch vụ thật.

---

## 2. Review theo tiêu chí big-tech (tham khảo/optional)

Phần này là audit theo hướng production-grade, dùng để chọn việc nâng cấp nếu còn thời gian. Nó **không phải checklist bắt buộc của PDF**. Khi phần này xung đột với deadline hoặc độ ổn định demo, ưu tiên §1 và P0 trước.

### 2.1 Security

#### Đã fix (so với master review v5)
- Password: PBKDF2-HMAC-SHA256 + salt (không còn SHA-256 unsalted + backdoor).
- `FLASK_SECRET` không hardcode; fallback random + warning.
- OAuth callback có nonce server-side + PKCE.
- OAuth token Fernet-encrypt từ `FLASK_SECRET`.
- CSRF guard bằng `X-Requested-With`, security headers + CSP.
- File upload whitelist extension + sanitize filename.
- Runtime `torch.load(..., weights_only=True)` ở `core/speaker_encoder.py:54`.
- Silero VAD từ PyPI thay vì `torch.hub.load`.
- Path containment đã thay `startswith` → `Path.relative_to()`.

#### Critical trong audit gốc (đã xử lý hoặc hạ rủi ro ở §4)

1. **Text mode (password) bypass SV cho IMPORTANT** — [web/app.py:683-855](../web/app.py#L683-L855)
   `api_assistant_text` set `sv_passed=True, sid_score=1.0` khi user nhập password. Bất kỳ ai biết password (hoặc brute-force qua 1.3 bên dưới) thực thi mọi IMPORTANT intent mà KHÔNG cần audio. → Voice biometric chỉ là UX, không phải security gate.
   **Fix**: (a) document rõ password là 2nd factor tương đương, hoặc (b) text-mode block IMPORTANT outright, hoặc (c) yêu cầu BOTH voice + password.

2. **OAuth URL predictable state khi gọi từ handler** — [core/handlers.py:134, 155](../core/handlers.py#L134)
   `build_auth_url(state=user_id)` (không nonce/PKCE) trong `handle_send_email`. Mọi luồng OAuth phải qua `/auth/google` để có nonce + PKCE.
   **Fix**: handler chỉ trả relative URL `/auth/google?user_id=...`, không tự gen URL.

3. **Không rate-limit password endpoints** — [web/app.py:466-525, 1147-1176, 1262-1283](../web/app.py#L466)
   `api_update_prefs`, `api_files_verify_password`, `api_user_unlock`, `api_admin_verify` cho unlimited attempt. PBKDF2 (600k iter) chống offline crack nhưng KHÔNG chống online brute force.
   **Fix**: Flask-Limiter, 5/min per IP+user, exponential backoff sau N fail.

#### High trong audit gốc

4. **CSRF chỉ dựa custom header**, không có server-issued token. `X-Requested-With` chỉ safe nếu CORS chặt. `flask-cors` ở `web/requirements.txt` (dead dep, chưa cấu hình) — nếu sau này enable permissive CORS thì mất CSRF.
5. **Session cookie thiếu `Secure / SameSite / HttpOnly`** ở [web/app.py:119-135](../web/app.py#L119-L135). App bind `0.0.0.0`, có `--ssl` flag nhưng KHÔNG force HTTPS — cookie qua plaintext có thể replay.
6. **CSP `'unsafe-inline'` cho script-src** — vô hiệu hoá phần lớn XSS protection. Template dùng inline `<script>` nhiều. Cần extract sang static JS + nonce CSP.
7. **Endpoint public exposes user existence** — `GET /api/users/<id>`, `/api/music/playlist/<id>`, `/api/oauth/status/<id>` không auth → attacker enumerate ID + biết user nào link Gmail.
8. **SSRF lite qua `/api/yt-audio`** — [web/app.py:1079-1109](../web/app.py#L1079-L1109) load cookies từ browser host vào yt-dlp; private video associated với host session bị pull qua server.
9. **`LIKE ?` không escape `%/_`** ở [core/database.py:241-260](../core/database.py#L241-L260) `find_user_by_name_substring` — user input wildcards. **Đã fix 2026-05-19**: escape `\`, `%`, `_` và dùng `LIKE ... ESCAPE '\'`; thêm regression test.

#### Medium trong audit gốc

10. **`delete_user` không cascade delete oauth_tokens, embeddings BLOB, hay WAV** ở `data/enroll_audio/<uid>/`. Re-enroll cùng `user_id` kế thừa OAuth token cũ. User không exercise được "right to erasure" (GDPR).
11. **OAuth callback HTML không escape `error / e`** — [web/app.py:1349, 1382](../web/app.py#L1349) `f"<pre>{error}</pre>"`.
12. **Plaintext OAuth token "backward compat" silently accept** — `_decrypt` rơi về plaintext nếu không có prefix `gAAAA` ([core/database.py:60-74](../core/database.py#L60-L74)). Attacker có write quyền DB downgrade encryption.
13. **Fernet key dẫn xuất từ `FLASK_SECRET` qua single SHA-256** (không salt, không KDF stretching) ở `core/database.py:36-49`.
14. **Voice replay attack chưa được giải quyết** — `SpeakerManager.verify` chỉ cosine + margin. Ghi âm playback pass SV. Cần challenge-response (prompt user đọc câu random) hoặc anti-spoofing model.

#### Low

15. Exception strings forward thẳng vào response → leak path / stack.
16. Admin chỉ password, không 2FA / IP-allowlist.
17. `pydub.AudioSegment.from_file(io.BytesIO(blob))` shell-out ffmpeg trên untrusted bytes — keep ffmpeg up-to-date.
18. Thiếu HSTS / `Strict-Transport-Security`.

---

### 2.2 Reliability & Observability

#### Gaps lớn

- **Không có audit log có cấu trúc** cho SV decision, password fail, OAuth token usage. `core/router.py:101` chỉ log exception. Cho dự án "Secure" — đây là gap lớn nhất.
- **`_email_sent_log` in-process dict** ở `core/handlers.py:33` — rate limit reset mỗi restart, không share giữa worker.
- **Không retry/backoff** cho external call (`requests.get/post` ở `oauth.py`, `gmail_api.py`, `app.py:892, 1438`). Một network hiccup = user thấy error.
- **Không circuit breaker**: Gemini/Gmail/Deezer/yt-dlp/gTTS down → mỗi turn pay 6-15s timeout.
- **Eager model loading blocks startup** ở `web/app.py:138-147`. Không có `/healthz` riêng biệt với `/api/health` — readiness vs liveness conflated.
- **Không idempotency key** cho `/api/files/upload`, `/api/enroll` — flaky network = duplicate WAV.
- **Không request-ID / correlation ID** — khó trace một turn của user qua nhiều log line.
- **Lock encoder serializes forward pass** — concurrent user queue lên; p99 latency spike dưới load.
- **Session 30 min permanent**, không sliding window.

#### Đã có
- Model eager-load một lần, lock quanh forward pass.
- SQLite WAL.
- Per-route request size limit.
- Audio enroll dùng temp dir, move sau khi thành công.

---

### 2.3 Maintainability & Code quality

- **`web/app.py` = 1499 dòng monolith** — tách blueprints: `auth_bp`, `users_bp`, `files_bp`, `music_bp`, `assistant_bp`, `admin_bp`, `oauth_bp`.
- **`web/templates/home.html` 2500+ dòng** HTML + inline JS + inline CSS. Extract JS sang `static/js/*.js` để drop `unsafe-inline`.
- **Email flow code duplicate** giữa `api_assistant_turn` ([web/app.py:568-641](../web/app.py#L568-L641)) và `api_assistant_text` ([web/app.py:703-771](../web/app.py#L703-L771)) — ~70 dòng giống nhau. Extract `_handle_email_flow()`.
- **Magic strings** khắp nơi: `"file_uid"`, `"file_auth_method"`, `"voice"`, `"password"`, `"oauth_nonce"`. Đưa vào `SessionKey` enum/dataclass.
- **Không có test tự động** trong repo. `cli/test_pipeline.py` là smoke script, không phải test suite. Cần pytest under `tests/` cho `database.py`, `oauth.py`, `email_flow.py`, `_safe_filename`, `_resolve_child_path`, `_verify_pw`.
- **`requirements.txt` không pin** — mọi dep dùng `>=`; build non-reproducible. Dùng `pip-compile` → `requirements.lock`. `web/requirements.txt` và root `requirements.txt` duplicate ML deps lệch nhẹ.
- **`flask-cors` declared but never imported** — dead dep.
- **`core/__init__.py` rỗng** — public API surface undefined.
- **Type hints inconsistent** — `_handle_send_email` returns `str | HandlerResult` thay vì chuẩn hoá.
- **Inconsistent error handling**: some routes `try/except → 500 str(e)`, others raise raw, others 400.

#### Bug/inconsistency nhỏ phát hiện thêm

- **`cli/enroll_user.py:52` retry bug**: `i -= 1; continue` không ảnh hưởng `for i in range(num_samples)`. Sample fail-VAD bị drop silent, final count có thể < `num_samples`.
- **`handle_open_files` trả plain string** thay vì `HandlerResult(action_type="show_files")` ở [core/handlers.py:266](../core/handlers.py#L266). **Đã fix 2026-05-19**: handler trả `HandlerResult(action_type="show_files")`, đếm file hiện có và nói rõ khi user chưa upload file.
- **`router.py:13` `from urllib.parse import quote` unused**.
- **`handle_play_music` guest fallback `"pop"`** (`handlers.py:280`) vs web default `"v-pop"` (`app.py:212`) — **đã fix 2026-05-19**: thống nhất default guest là `"v-pop"`.
- **Intent `email_flow` synthetic value** (`app.py:625, 756`) không khai báo trong `INTENTS` dict. **Đã fix 2026-05-19**: khai báo `email_flow` là internal intent và ẩn khỏi Gemini tool declarations.
- **`nlu.py:195` `resp.text.strip()` không guard `None`** — **đã fix 2026-05-19**: guard `(resp.text or "").strip()` và trả fallback thân thiện khi Gemini safety-block/empty response.
- **`api_assistant_text` query DB user 2 lần/turn** (router + `_attach_action_data`).

---

### 2.4 ML/Product Quality

- **Tốt**: multi-window embedding, quality filter centroid, AS-Norm-lite, margin check, backend switch.
- **Thiếu**:
  - Calibration notebook/file kết quả thật cho threshold SID/SV. Default `0.45/0.25/0.93` là guess theo backend.
  - Spoof/replay defense (challenge-response, anti-spoofing).
  - Drift monitoring chất lượng giọng theo thời gian.
  - Bias/fairness: train trên Indian subset → underperform khi deploy cho user Việt (OOD). Cần per-speaker accuracy table + discussion trong báo cáo.
  - Model card: intended use, dataset bias, known failure mode, embedding spec, license của SpeechBrain weight.

---

### 2.5 Compliance & Privacy

- **Biometric data là PII nhạy cảm** (GDPR Art.9, Luật bảo vệ DLCN Việt Nam): `data/embeddings` BLOB + WAV ở `data/enroll_audio/<uid>/` không encrypt at-rest, không retention policy.
- **`delete_user` không xoá WAV file** — chỉ xoá DB row → "right to erasure" không exercise được.
- **Không có consent banner / privacy notice** trước khi xin mic. Cần explicit consent screen lần đầu enroll.
- **Gmail OAuth scope `gmail.send` đúng — minimal**. Tốt.
- **Transparency về độ mạnh auth**: user nghĩ SV "an toàn" — cần banner ở `assistant.html` cho IMPORTANT: "Voice verification is convenience, not equivalent to password".
- **Audit log nếu thêm**: KHÔNG log full email body / note / balance — đó là user data, không phải security signal.
- **Không có data export endpoint** (GDPR Art.20 portability).

---

## 3. Đề xuất cải tiến tính năng

### 3.1 Đào sâu chức năng đã có

- **`handle_get_weather`** ([core/handlers.py:60](../core/handlers.py#L60)) đang `random.choice(conditions)` — wire OpenWeatherMap / Visual Crossing API dùng entity `location`. Cache result theo `(city, hour)`.
- **`handle_check_balance`** đang trả `preferences.balance` tĩnh — thêm transaction history `[{ts, amount, note}]` và surface "3 giao dịch gần nhất".
- **`read_notes`** chỉ show top-3 — thêm intent `add_note` (IMPORTANT) để bidirectional voice-side.
- **`show_schedule`** chỉ display string — pair với `add_schedule` + date entity parsing. Tương lai integrate Google Calendar (reuse OAuth scope expansion đã proven cho Gmail).
- **`play_music`** đã có rich Deezer/YouTube/user-tracks integration nhưng `_attach_action_data` chỉ consult `genre/favorite_artist` — thêm playlist-aware mode (auto-pick từ saved playlist).
- **`handle_open_files`** không biết user có file không — **đã fix 2026-05-19**: handler đếm file trong `USER_FILES_DIR/<uid>` và trả message/action data phù hợp.

### 3.2 Thêm chức năng mới (showcase tốt)

**Ưu tiên cao** (vừa nâng điểm "Secure" vừa demo tốt):

- **Challenge-response cho IMPORTANT**: sau khi SID+SV pass lần đầu, sinh câu ngắn ngẫu nhiên (vd "đọc 7-3-9-1") để user đọc lại; verify BOTH ASR content + giọng nói. Đây là single most impactful security feature.
- **Voice liveness / anti-spoofing**: integrate `ASVspoof` model hoặc đơn giản dùng energy-band check. Pass-fail report trong turn log.
- **Audit log có cấu trúc** (JSON sink loguru): user, intent, auth_method, sv_score, sv_passed, sid_score, ip, ua, outcome. Ẩn nội dung nhạy cảm.
- **Calibration report tự động**: script chạy benchmark trên audio enroll/demo, xuất EER, FAR/FRR theo threshold, recommended threshold per backend. Output PNG plot + Markdown table cho báo cáo.
- **Re-enroll bổ sung sau khi xác thực thành công** (speaker drift handling): mỗi N turn pass SV, lưu thêm embedding mới vào centroid với weight giảm dần.

**Ưu tiên vừa**:

- **`set_reminder` (IMPORTANT)** — reuse `schedule` storage, fits taxonomy.
- **Google Calendar integration** — reuse OAuth pattern, `show_schedule` đọc event thật.
- **`add_contact` (IMPORTANT)** — voice-populate `contacts` đang chỉ edit qua UI (`email_flow.py:41-72`).
- **`read_email` / `summarize_inbox` (IMPORTANT)** — Gmail send đã có, đọc inbox API là extension nhỏ.
- **Multi-language switch** — config hard-code `vi` ở `config.py:125, 146`; expose `preferences.language` per-user.
- **Wake-word mode** — Picovoice Porcupine offline, thay thế "Press Enter" trigger ở `cli/run_assistant.py:70`.
- **Offline TTS fallback** (pyttsx3) — không phụ thuộc gTTS khi mất mạng.
- **UI hiển thị confidence + lý do block** — thân thiện thay vì silent fail.
- **Rate limit theo IP/session** cho password verification, enroll, assistant turn.

**Ưu tiên thấp nhưng có điểm cộng**:

- **Tách text password fallback** thành "developer/demo mode" qua env var, tránh hiểu nhầm SV.
- **Export báo cáo demo**: turn log → bảng markdown cho phụ lục báo cáo.
- **Data export endpoint** `GET /api/users/<id>/export` (password-gated).
- **Consent banner** lần đầu mic access.

---

## 4. Todo-list theo ưu tiên

Format: `- [ ]` chưa làm, `- [x]` đã làm, ưu tiên cao → thấp.

### 4.1 Ưu tiên 🔴 P0 — trước khi nộp/demo

- [ ] **Commit evidence training cho YC1**: chạy `train_ecapa.py` + `evaluate_sid.py` + `evaluate_sv.py` trên dataset thực tế đang dùng; commit `training_log.json`, `sid_results.json`, `sv_results.json`, `spk2idx.json`, DET/ROC plot vào `training/results/` hoặc `docs/results/`.
  - **Đã chuẩn bị**: Tạo `docs/results/README.md` + `training/results/README.md` với checklist artifact. `train_ecapa.py` giờ tự dump `hparams.json` (git SHA + torch version + args) để reproducibility. `scripts/benchmark.py` thêm flag `--out` dump JSON summary.
  - **Còn lại**: chỉ phụ thuộc user/sinh viên chạy training trên Kaggle (cần GPU) và copy artifact về 2 folder trên.
- [x] **Làm rõ dataset đã dùng**: đã verify `iden_split.txt` và `veri_test.txt` chứa đúng 24 Indian speakers (`id10002`→`id11209`), khớp README. Thêm section "Caveat methodology" vào `training/README.md` ghi rõ random-shuffle leak, trial pair nhỏ, Indian bias, không augmentation.
- [x] **Đổi `torch.load(..., weights_only=False)` → `weights_only=True`** ở `training/evaluate_sid.py:36` và `training/evaluate_sv.py:69`.
- [x] **Chuẩn hoá OAuth URL generation**: bỏ `build_auth_url(state=user_id)` trong `core/handlers.py`; handler giờ trả `action_data={"user_id": ...}` và frontend tự redirect qua `/auth/google?user_id=...` để server gen nonce + PKCE đúng chuẩn.
- [x] **Quyết định + document text-mode IMPORTANT**: chọn phương án (c) — password fallback opt-in qua env `ALLOW_PASSWORD_FOR_IMPORTANT` (default `false`). Khi off, text-mode block IMPORTANT outright với message rõ. `.env.example` + `core/config.py` đã document threat model.
- [x] **Rate limit password endpoints**: thêm decorator `@rate_limit` (in-process, sliding window) cho `/api/users/<id>/{update,update-info,delete,unlock}`, `/api/files/verify-password`, `/api/admin/verify`, `/api/enroll`, `/api/assistant/text`. Trả 429 + `Retry-After`. Production sẽ migrate sang Flask-Limiter + Redis (xem P2).
- [x] **Cascade delete khi `delete_user`**: `core/database.py:delete_user` xoá `oauth_tokens` + `embeddings`. `web/app.py:api_delete_user` shutil.rmtree `data/enroll_audio/<uid>/` + `data/user_files/<uid>/` và `_clear_file_session()` nếu chính user đó đang đăng nhập.
- [ ] **Calibration artifact thật trong `docs/results/`**: EER, minDCF, SID Top-1, confusion matrix, ROC từ audio enroll demo. Threshold mặc định hiện là guess (`0.45/0.25/0.93`).
  - **Đã chuẩn bị**: `scripts/benchmark.py --out docs/results/benchmark_<backend>.json` dump JSON schema `secva.benchmark.v1`, có metadata backend, summary và từng sample ASR/SID/SV.
  - **Còn lại**: chạy benchmark trên audio enroll thật (cần sinh viên enroll vài user qua web rồi chạy script).
- [x] Thay path containment `startswith` → `Path.relative_to()` cho file/music routes (đã làm trong bản review trước).

### 4.2 Ưu tiên 🟡 P1 — trong sprint trước demo

- [x] **Security regression tests** (P1-1) — `tests/` với 55 security/regression test pass: password hashing (PBKDF2 + salt + Unicode), path traversal (`_resolve_child_path` + `_safe_filename`), OAuth Fernet encryption (DB không leak plaintext), cascade delete (oauth_tokens orphan), CSRF guard, rate limit 429, text-mode IMPORTANT block, session cookie flags, security headers, GET user không leak prefs, challenge-response phrase match, LIKE wildcard escape, `open_files` action, `email_flow` internal intent, Gemini empty response guard, admin-assisted password reset.
- [ ] **Tách `web/app.py` thành blueprints** (P1-2) — **DEFER → P2** vì refactor lớn không có security/feature impact, risk break cao mà tests còn ít. Đẩy xuống P2 backlog.
- [x] **Extract duplicate email-flow** (P1-3) — `_continue_email_flow_if_active(text, db, nlu)` ở `web/app.py`, dùng chung giữa `/api/assistant/turn` (voice) và `/api/assistant/text` (text). Cắt ~140 dòng duplicate.
- [x] **Challenge-response cho IMPORTANT voice flow** (P1-4) — module `core/challenge.py` (phrase gen 4 từ từ pool digit+color, token urlsafe 18 byte, Levenshtein word-level match tolerance 0.34, TTL 90s). Router `complete_challenge()` re-verify SV+ASR. New endpoint `/api/assistant/challenge-response` (rate-limit 10/60s). Opt-in via env `CHALLENGE_RESPONSE_ENABLED=true`. Audit log `auth.challenge_issued` + `auth.challenge_check`. 11 unit test trong `tests/test_challenge.py`.
- [x] **Audit log có cấu trúc** (P1-5) — module `core/audit.py` JSON-line sink `data/audit.log` (rotate 5MB×5). Event: `auth.password_check` (success/fail + scope + ip + ua_hash), `auth.voice_verify` (sv_score, sid_score, margin), `auth.challenge_*`, `oauth.granted` (gmail_hash, không log plaintext), `data.delete_user` (dirs_removed). Truncate field > 200 char.
- [x] **Turn log / NLU training candidates có schema** — module `core/turn_logging.py` ghi append-only `data/turn_log.jsonl` cho web voice/text/challenge + CLI voice, đồng thời sinh `data/nlu_training_candidates.jsonl` để label thủ công. Có redaction email/số điện thoại/password/token/body email. Thêm `scripts/export_nlu_dataset.py` xuất CSV cho pipeline train.
- [x] **Session cookie hardening** (P1-6) — `SESSION_COOKIE_HTTPONLY=True`, `SAMESITE=Lax`, `SECURE=True` (auto-tắt nếu env `FLASK_DEV_HTTP=true`). HSTS `max-age=180 ngày` set khi request HTTPS.
- [x] **Bảo vệ public endpoints exposes user existence** (P1-7) — `GET /api/users/<id>` trả về preferences rỗng nếu caller chưa unlock; full prefs chỉ qua `/api/users/<id>/unlock`. `GET /api/oauth/status/<id>` trả gmail address chỉ khi caller có file session match user_id, hoặc đang trong OAuth flow active (`oauth_pending_uid`), hoặc vừa hoàn tất OAuth (`oauth_recent` marker 10 phút). Page `/users/<id>` SSR cũng gate prefs.
- [x] **OAuth callback HTML escape** (P1-8) — `html.escape(error)`, `html.escape(str(e))`, `html.escape(user_id)`, `html.escape(gmail_addr)` trong tất cả response HTML của `/auth/google/callback`.
- [x] **CSP nonce** (P1-9, partial) — Per-request nonce qua `g.csp_nonce`, inject vào tất cả `<script>` tag trong 7 template + OAuth callback inline script. `'unsafe-inline'` giữ lại cho onclick attribute compat (full drop cần refactor onclick → addEventListener, P2 backlog).
- [x] **Cập nhật README "Security model"** (P1-10) — Section mới trong `README.md` mô tả 3 lớp AuthLevel, policy text-mode, rate limit cap mỗi endpoint, session hardening, PII protection, demo functions cần khai báo trong báo cáo, threat model limitations.
- [x] **Fix `cli/enroll_user.py:52`** (P1-11) retry bug — đổi `for i in range(N)` + `i -= 1; continue` thành `while i < N` với `max_retries_per_slot=3` để retry slot fail VAD đúng cách.

### 4.3 Ưu tiên 🟢 P2 — polish/production-grade

> Bao gồm 2 task được đẩy từ P1 xuống: blueprint split (P1-2), full drop CSP `'unsafe-inline'` qua onclick → addEventListener refactor (P1-9 phần còn).

- [x] **Encrypt biometric data at-rest / Key derivation hardening** (P2-J, partial) — Phương án tối thiểu đã làm: thêm env `TOKEN_ENCRYPTION_KEY` (dedicated 32-byte key) và HKDF-SHA256 (RFC 5869) làm fallback từ `FLASK_SECRET`. Backward compat: legacy single-SHA-256 key giữ làm `_legacy_fernet` để decrypt token cũ. WAV encryption ở filesystem level đẩy xuống P3 — touches 3+ callers (benchmark, reenroll_backend, data_export), risk break cao.
- [x] **Consent banner** (P2-F) — Section "Thông báo về dữ liệu sinh trắc giọng nói" trên `/enroll`, mô tả storage (WAV + 192-d BLOB + token) + retention + right to erasure. Checkbox gate `record-btn` và `submit-btn`.
- [x] **Data export endpoint** (P2-G) — `POST /api/users/<id>/export` password-gated, rate-limit 3/5min. ZIP chứa `user.json` + `enroll_audio/*` + `user_files/*` + `oauth.json` (gmail/expiry, KHÔNG include token). Audit `data.export_user`.
- [ ] **Augmentation cho training** — MUSAN noise + RIR reverb + SpecAugment + speed perturb. ECAPA-TDNN gain lớn nhất từ đây. **DEFER**: yêu cầu chạy training trên Kaggle GPU, không impact runtime.
- [ ] **Wandb/TensorBoard hook** trong `train_ecapa.py` — per-step loss curve. **DEFER**: optional; `training_log.json` đã có loss/acc theo epoch.
- [x] **Model card** (P2-H) — `training/MODEL_CARD.md` theo format Mitchell et al. 2019: model details, intended use, training data (caveat splits + bias), evaluation expected, ethical considerations (replay/cloning/consent), caveats + recommendations.
- [x] **Pin requirements** (P2-K) — `requirements.lock.txt` từ `pip freeze` (96 dòng), kèm header doc cách regenerate. `pyproject.toml` extras + `pip-audit` đẩy xuống P3.
- [x] **Bỏ `flask-cors` dead dep** (P2-A) — đã xoá khỏi `web/requirements.txt`. (Vẫn còn trong venv local + `requirements.lock.txt` cho đến khi `pip uninstall`.)
- [x] **Best_model.pt sang Git LFS** (P2-B) — `.gitattributes` track `checkpoints/*.{pt,ckpt,safetensors}` qua LFS + chuẩn hoá line-ending cho text file. Reviewer cần `git lfs install && git lfs pull`.
- [x] **Retry + circuit breaker** (P2-I) — `core/resilience.py` (~150 dòng, không dep mới): `@retry(max_attempts, backoff, retriable)` + `CircuitBreaker(name, fail_threshold, reset_after)` 3-state closed/open/half-open. Apply lên `core/oauth.py` (`google-oauth` breaker) + `core/gmail_api.py` (`gmail-send` breaker). 6 unit test pass.
- [x] **Request-ID middleware** (P2-D) — `g.request_id` sinh ở `before_request` (trust upstream `X-Request-ID` nếu match regex), echo qua response header. Auto-attach vào audit log event qua `flask.has_request_context()` check.
- [x] **Healthz vs ready endpoint** (P2-C) — `/healthz` liveness 200 ngay; `/readyz` check DB + encoder + ASR, 503 nếu chưa ready. `/api/health` giữ làm info dump backward compat.
- [x] **Move `_email_sent_log` → SQLite** (P2-E) — bảng `email_send_log(user_id, sent_at)` + indexed. `UserDB.count_emails_sent_within / oldest_email_within / record_email_sent`. Persist qua restart + multi-worker-safe. Auto-cleanup row > 24h.
- [x] **Fix batch bug/inconsistency nhỏ từ audit 2.3** (P2-L) — escape wildcard trong `find_user_by_name_substring`, `handle_open_files` trả `HandlerResult(show_files)` + file count, guest music default thống nhất `"v-pop"`, khai báo `email_flow` internal intent, guard Gemini `resp.text=None`. Thêm `tests/test_regression_todo_fixes.py` (5 test).
- [x] **Admin-assisted reset password** (P2-M) — thêm endpoint `POST /api/admin/users/<id>/reset-password` yêu cầu `ADMIN_PASS` trên từng request, validate password mới ≥ 8 ký tự, không khôi phục/hiển thị password cũ, audit `auth.password_reset`, rate-limit 5/5 phút per IP+user. UI Admin Panel có nút "Reset mật khẩu". Thêm regression test.
- [ ] **`web/templates/home.html` refactor** — 2500 dòng → component-style. **DEFER**: refactor lớn, no security/feature impact.
- [ ] **Blueprint split `web/app.py`** (carry from P1-2) — **DEFER P3**.
- [ ] **Full drop CSP `'unsafe-inline'`** (carry from P1-9) — refactor onclick → addEventListener trong 7+ template. **DEFER P3**.
- [ ] **WAV encryption at-rest** — Fernet-encrypt WAV trong `data/enroll_audio/`. **DEFER P3**: touches benchmark + reenroll_backend + data_export, cần migration script.

### 4.4 Tính năng mới (chọn 1-2 cho demo wow factor)

- [x] **Challenge-response + anti-spoofing** — đã làm phần challenge-response ở P1-4. Anti-spoofing đầy đủ (ASVspoof) vẫn defer.
- [x] **`add_note` / `add_contact` / `add_schedule`** (3 bidirectional voice intent, IMPORTANT) — module `core/intents.py` + `core/handlers.py` + rule-based NLU. Cap entries (50 notes / 50 schedule / 100 contacts). `add_contact` idempotent (cùng name → update email). 14 unit test trong `tests/test_bidirectional_intents.py`.
- [x] **Multi-language switch** qua `preferences.language` — `core/tts.py:_resolve_lang()` + `SUPPORTED_TTS_LANGS` (vi/en/ja/ko/zh-CN/fr/de). `/api/tts` resolve theo `?lang=...` → user pref → default vi. 7 unit test trong `tests/test_multilang_tts.py`.
- [x] **Re-enroll bổ sung sau khi pass SV** — `SpeakerManager.incremental_update_centroid(audio, uid, alpha)` cập nhật centroid theo công thức `new = normalize((1-α)*old + α*emb_new)`. Hook ở Router sau SV pass (skip khi challenge enabled để chờ audio_2). Opt-in `SPEAKER_INCREMENTAL_REENROLL=true`, `SPEAKER_INCREMENTAL_ALPHA=0.1`. Audit event `speaker.incremental_update`. 6 unit test trong `tests/test_incremental_reenroll.py`.
- [ ] **Calendar integration** cho `show_schedule` (reuse OAuth scope expansion).
- [ ] **Reminder intent** + push notification (nếu PWA).
- [ ] **Offline TTS fallback** pyttsx3.
- [ ] **Wake-word** Picovoice Porcupine.

---

## 5. Tài liệu liên quan

- Review chi tiết source-level: `SecureVA-MASTER-REVIEW.md` (Phần 0 → Phần 20, ~80 issues).
- Đề bài: `docs/Secure_Virtual_Assistant_with_Speaker_Recognition.pdf`.
- READMEs: [README chính](../README.md), [training/](../training/README.md), [cli/](../cli/README.md), [web/](../web/README.md), [scripts/](../scripts/README.md).
