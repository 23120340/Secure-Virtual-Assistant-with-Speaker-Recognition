
"""Flask web app cho Virtual Assistant.

Routes:
    GET  /                         → home, list users
    GET  /enroll                   → form đăng ký user mới
    POST /api/enroll               → multipart upload nhiều audio mẫu
    GET  /users/<id>               → user detail + edit preferences
    GET  /api/users/<id>           → trả về user dict + has_password
    POST /api/users/<id>/update    → update preferences (cần password)
    POST /api/users/<id>/update-info → cập nhật name / password
    POST /api/users/<id>/delete    → xóa user (cần password)
    GET  /assistant                → chat UI
    POST /api/assistant/turn       → audio in → JSON {transcript, intent, ...}
    GET  /api/tts?text=...         → MP3 bytes cho browser play
    POST /api/files/verify-password → xác thực bằng password thay vì giọng

Chạy:
    python -m web.app
    # → http://localhost:5000
"""
import hmac
import io
import json
import logging
import os
import re
import secrets
import sys
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from urllib.parse import quote as _quote

# Cấu hình logging cho toàn bộ app — INFO trên stdout, format gọn cho dev.
# Production có thể override qua env LOG_LEVEL=WARNING / ERROR.
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_log = logging.getLogger("secva.app")

from flask import (Flask, render_template, request, jsonify, send_file,
                   redirect, url_for, flash, session)


import unicodedata as _unicodedata


AUDIO_EXTS = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.wma', '.opus'}
DOC_EXTS = {'.pdf', '.doc', '.docx', '.txt', '.md', '.jpg', '.jpeg', '.png',
            '.gif', '.zip', '.rar', '.csv', '.xlsx', '.xls'}


def _safe_filename(filename: str,
                   max_stem_len: int = 100,
                   allowed_exts: set | None = None) -> tuple[str, bool]:
    """Unicode-safe sanitizer — giữ ký tự tiếng Việt, loại bỏ path traversal.

    Returns (safe_name, is_allowed_ext).
    - Normalize Unicode NFC để tránh tấn công homograph qua composed/decomposed.
    - Xóa path separator, shell special, control, null byte.
    - Cắt stem ≤ max_stem_len ký tự (Windows path limit ~260 char total).
    - allowed_exts=None bỏ qua check ext; nếu set → trả về False khi không khớp.
    """
    name = _unicodedata.normalize("NFC", (filename or "").strip())
    # Xóa ký tự nguy hiểm (path sep, shell special, control, null)
    name = re.sub(r'[/\\:*?"<>|\x00-\x1f\x7f]', '_', name)
    # Chống path traversal: ".." → "."
    name = re.sub(r'\.\.+', '.', name)
    # Thu gọn dấu cách/gạch dưới liên tiếp
    name = re.sub(r'[_\s]+', '_', name).strip('_. ')

    if "." in name:
        stem, ext = name.rsplit(".", 1)
        ext = "." + ext.lower()[:10]
    else:
        stem, ext = name, ""

    stem = stem[:max_stem_len] if stem else "file"
    safe = f"{stem}{ext}".rstrip(".")
    is_allowed = (allowed_exts is None or ext in allowed_exts)
    return safe or "file", is_allowed

# Thêm project root vào path để import được core/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import audio_io, config
from core.asr import get_asr, correct_transcript
from core import email_flow as _ef
from core import handlers as _h_module
from core.handlers import handle_send_email as _send_email
from core.oauth import (build_auth_url, exchange_code, gen_pkce_pair,
                        get_user_email)
from core.tts import get_tts
from core.nlu import get_nlu
from core.database import UserDB, SpeakerManager
from core.router import Router


# ==========================================================================
# App init — tất cả model load 1 lần lúc startup (singleton)
# ==========================================================================
def create_app():
    app = Flask(__name__,
                template_folder="templates",
                static_folder="static")
    # Secret key từ env, fallback random nếu chưa set (sessions invalidate mỗi restart)
    flask_secret = os.environ.get("FLASK_SECRET", "").strip()
    if not flask_secret:
        _log.warning("FLASK_SECRET chưa set — dùng random key, sessions sẽ invalidate mỗi restart")
        flask_secret = secrets.token_hex(32)
    app.secret_key = flask_secret
    # Global hard cap 60MB — per-route giới hạn nhỏ hơn qua @max_size(...).
    # Vừa đủ cho upload file/music + buffer; chống DoS POST rác lớn.
    app.config["MAX_CONTENT_LENGTH"] = 60 * 1024 * 1024
    app.permanent_session_lifetime = timedelta(minutes=30)
    # Jinja auto_reload: chỉ bật khi FLASK_DEBUG=1 (development). Production có
    # auto_reload=True làm tăng I/O mỗi request — không cần thiết.
    app.jinja_env.auto_reload = os.environ.get("FLASK_DEBUG", "").strip() in ("1", "true", "True")

    # Eager-load các model — startup chậm 1 chút nhưng request sẽ nhanh
    _log.info("=" * 60)
    _log.info("Loading models (lần đầu mất 30-60s)...")
    db = UserDB()
    spk_mgr = SpeakerManager(db)
    asr = get_asr()
    tts = get_tts()
    nlu = get_nlu()
    router = Router(spk_mgr)
    _log.info("✓ Tất cả model đã sẵn sàng.")
    _log.info("=" * 60)

    # Lưu vào app context để các handler dùng
    app.config["db"] = db
    app.config["spk_mgr"] = spk_mgr
    app.config["asr"] = asr
    app.config["tts"] = tts
    app.config["nlu"] = nlu
    app.config["router"] = router

    register_routes(app)
    return app


# ==========================================================================
# Request size limit (per-route)
# ==========================================================================
def _list_user_files(uid: str) -> list:
    """Liệt kê file của user (dùng cho intent open_files trong cả turn/text mode)."""
    user_dir = config.USER_FILES_DIR / uid
    files = []
    if user_dir.exists():
        for _f in sorted(user_dir.iterdir()):
            if _f.is_file():
                files.append({
                    "name": _f.name,
                    "size": _f.stat().st_size,
                    "modified": datetime.fromtimestamp(
                        _f.stat().st_mtime).strftime("%d/%m/%Y %H:%M"),
                })
    return files


def _list_user_music_tracks(uid: str) -> list:
    """Liệt kê file nhạc user đã upload."""
    music_dir = config.USER_FILES_DIR / uid / "music"
    if not music_dir.exists():
        return []
    return [f.name for f in music_dir.iterdir()
            if f.is_file() and f.suffix.lower() in AUDIO_EXTS]


def _attach_action_data(payload: dict, intent: str, blocked: bool,
                        uid: str | None, user_name: str,
                        entities: dict, user_prefs: dict,
                        auth_method: str) -> dict:
    """Gắn action_type + action_data + set_file_session cho intent play_music / open_files.

    Dùng chung cho api_assistant_turn (auth_method='voice') và api_assistant_text
    (auth_method='password'). Trước đây ~80 dòng duplicate giữa 2 routes.
    """
    if blocked:
        return payload

    if intent == "play_music":
        user_tracks = _list_user_music_tracks(uid) if uid else []
        if user_tracks and uid:
            _set_file_session(uid, user_name, auth_method)
            payload["action_type"] = "play_music"
            payload["action_data"] = {
                "mode": "user_tracks",
                "user_id": uid,
                "tracks": sorted(user_tracks),
            }
        else:
            genre  = entities.get("genre") or user_prefs.get("favorite_genre", "") or "v-pop"
            artist = user_prefs.get("favorite_artist", "")
            payload["action_type"] = "play_music"
            payload["action_data"] = {
                "mode": "youtube",
                "artist": artist,
                "genre": genre,
                "uid": uid or "",
            }

    elif intent == "open_files" and uid:
        files = _list_user_files(uid)
        _set_file_session(uid, user_name, auth_method)
        payload["action_type"] = "show_files"
        payload["action_data"] = {
            "files": files,
            "user_name": user_name,
            "user_id": uid,
        }

    return payload


def _json_body() -> dict:
    """Parse request body as JSON, return {} nếu thiếu hoặc invalid.
    Tránh pattern verbose `_json_body()` lặp lại 7 lần."""
    return request.get_json(silent=True) or {}


def max_size(mb: float):
    """Decorator giới hạn size request cho từng route. Audio turn ≈ 100KB, không
    cần cho phép 50MB như global. Chống DoS upload rác lên /api/assistant/turn.
    """
    limit_bytes = int(mb * 1024 * 1024)

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            cl = request.content_length
            if cl is not None and cl > limit_bytes:
                return jsonify({"error": f"Yêu cầu quá lớn (giới hạn {mb}MB)"}), 413
            return fn(*args, **kwargs)
        return wrapper
    return decorator


# ==========================================================================
# Session helpers
# ==========================================================================
# Phiên file/music sống tối đa 30 phút sau khi xác thực; sau đó người dùng phải
# xác thực lại để truy cập file riêng. Session cookie cũng được set permanent
# với timedelta(minutes=30) ở create_app — đây là check thêm phía server-side.
SESSION_LIFETIME_SEC = 30 * 60


def _set_file_session(uid: str, user_name: str, auth_method: str):
    """Đánh dấu phiên đã xác thực với phương thức cụ thể.

    auth_method: 'voice' nếu qua SID+SV, 'password' nếu qua mật khẩu.
    Bao gồm timestamp để middleware có thể tự expire phiên.
    """
    session["file_uid"]          = uid
    session["file_name"]         = user_name
    session["file_auth_method"]  = auth_method
    session["file_verified_at"]  = time.time()
    session.permanent            = True


def _clear_file_session():
    for k in ("file_uid", "file_name", "file_auth_method", "file_verified_at"):
        session.pop(k, None)


def _get_authed_uid(level: str = "any") -> str | None:
    """Lấy uid đã xác thực, hoặc None nếu chưa/đã hết hạn.

    level='any'      → cả voice và password đều ok (cho file/music).
    level='voice'    → chỉ voice (cho IMPORTANT intent kiểu read_notes/send_email
                       — không dùng password làm bypass biometric).
    """
    uid = session.get("file_uid")
    if not uid:
        return None
    verified_at = session.get("file_verified_at", 0)
    if time.time() - verified_at > SESSION_LIFETIME_SEC:
        _clear_file_session()
        return None
    if level == "voice" and session.get("file_auth_method") != "voice":
        return None
    return uid


# ==========================================================================
# Routes
# ==========================================================================
def register_routes(app):

    # ----- CSRF + Security headers -----
    @app.before_request
    def csrf_protect():
        """Yêu cầu header X-Requested-With cho mọi state-changing request.

        Trình duyệt không cho phép cross-origin form gửi header tùy chỉnh trong
        request đơn giản → kiểm tra header này chặn phần lớn CSRF.
        OAuth callback (/auth/google/callback) là GET nên không bị chặn.
        """
        if request.method not in ("POST", "PUT", "PATCH", "DELETE"):
            return None
        # OAuth callback từ Google: cũng GET, không vào đây.
        # Browser native form không thể set custom header.
        if request.headers.get("X-Requested-With") != "XMLHttpRequest":
            return jsonify({
                "error": "CSRF: thiếu header X-Requested-With"
            }), 403

    # Content-Security-Policy — defence-in-depth chống XSS. 'unsafe-inline' cho
    # script vẫn cần vì code inline đầy template; refactor extract sang static .js
    # rồi có thể bỏ unsafe-inline để tăng độ an toàn.
    _CSP = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://fonts.cdnfonts.com; "
        "font-src https://fonts.gstatic.com https://fonts.cdnfonts.com data:; "
        "img-src 'self' data: https://i.ytimg.com https://*.deezer.com https://*.dzcdn.net; "
        "media-src 'self' blob: https://*.googlevideo.com; "
        "connect-src 'self'; "
        "frame-ancestors 'none';"
    )

    @app.after_request
    def set_security_headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        response.headers.setdefault("Content-Security-Policy", _CSP)
        return response

    # ----- Pages -----
    @app.route("/")
    def home():
        users = app.config["db"].list_users()
        return render_template("home.html", users=users,
                               num_samples=config.ENROLL_NUM_SAMPLES,
                               duration=config.ENROLL_DURATION)

    @app.route("/enroll")
    def enroll_page():
        return render_template("enroll.html",
                               num_samples=config.ENROLL_NUM_SAMPLES,
                               duration=config.ENROLL_DURATION)

    @app.route("/users/<user_id>")
    def user_detail(user_id):
        user = app.config["db"].get_user(user_id)
        if not user:
            flash(f"Không tìm thấy user '{user_id}'", "error")
            return redirect(url_for("home"))
        return render_template("user_detail.html", user=user)

    @app.route("/assistant")
    def assistant_page():
        users = app.config["db"].list_users()
        return render_template("assistant.html", users=users)

    # ----- API: Enrollment -----
    @app.route("/api/enroll", methods=["POST"])
    @max_size(20)
    def api_enroll():
        """Body multipart:
           - user_id: text
           - name: text
           - preferences: JSON text (optional)
           - password: text (optional)
           - sample_0, sample_1, ... : audio blobs (WebM/wav)
        """
        user_id = request.form.get("user_id", "").strip()
        name = request.form.get("name", "").strip()
        prefs_str = request.form.get("preferences", "{}")
        password = request.form.get("password", "").strip()

        # Validate
        if not user_id or " " in user_id:
            return jsonify({"error": "user_id không hợp lệ (không trống, không space)"}), 400
        if not name:
            return jsonify({"error": "Thiếu name"}), 400
        if not password or len(password) < 6:
            return jsonify({"error": "Mật khẩu phải có ít nhất 6 ký tự"}), 400
        try:
            preferences = json.loads(prefs_str)
        except json.JSONDecodeError:
            return jsonify({"error": "preferences không phải JSON hợp lệ"}), 400

        db = app.config["db"]
        spk_mgr = app.config["spk_mgr"]
        if db.get_user(user_id):
            return jsonify({"error": f"User '{user_id}' đã tồn tại"}), 409

        # Decode + VAD trim từng sample. Lưu WAV vào TEMP dir, chỉ move sang vị trí
        # cuối cùng khi DB commit thành công — tránh trường hợp 4 sample đầu thành công,
        # sample 5 fail VAD → 4 file rác kẹt lại trong data/enroll_audio/<user_id>/.
        import tempfile, shutil
        audios = []
        with tempfile.TemporaryDirectory(prefix="enroll_") as _tmp_str:
            tmp_dir = Path(_tmp_str)

            for key in sorted(request.files.keys()):
                if not key.startswith("sample_"):
                    continue
                blob = request.files[key].read()
                try:
                    audio = audio_io.decode_browser_audio(blob)
                except Exception as e:
                    return jsonify({"error": f"Giải mã {key} thất bại: {e}"}), 400

                audio, has_speech = audio_io.SileroVAD.trim(
                    audio, return_speech_detected=True)
                _min_enroll = int(config.SAMPLE_RATE * config.MIN_AUDIO_SEC_ENROLL)
                if not has_speech or audio.size < _min_enroll:
                    return jsonify({
                        "error": f"{key} quá ngắn sau VAD ({audio.size/config.SAMPLE_RATE:.2f}s, "
                                 f"cần ≥ {config.MIN_AUDIO_SEC_ENROLL}s). "
                                 "Hãy đảm bảo nói rõ ràng đủ thời lượng."
                    }), 400

                audio_io.save_wav(audio, tmp_dir / f"{key}.wav")
                audios.append(audio)

            if len(audios) < config.ENROLL_NUM_SAMPLES:
                return jsonify({
                    "error": f"Cần đủ {config.ENROLL_NUM_SAMPLES} mẫu, chỉ có {len(audios)}"
                }), 400

            # Enroll (sẽ commit DB nếu thành công)
            try:
                centroid = spk_mgr.enroll(user_id, name, audios, preferences, password)
            except Exception as e:
                return jsonify({"error": f"Đăng ký giọng thất bại: {e}"}), 500

            # DB commit OK → move WAV từ tmp sang vị trí cuối cùng.
            sample_dir = config.ENROLL_AUDIO_DIR / user_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            for wav_file in tmp_dir.iterdir():
                shutil.move(str(wav_file), sample_dir / wav_file.name)

        return jsonify({
            "ok": True,
            "user_id": user_id,
            "name": name,
            "n_samples": len(audios),
            "centroid_norm": float((centroid**2).sum()**0.5),
        })

    # ----- API: User CRUD -----
    @app.route("/api/users/<user_id>", methods=["GET"])
    def api_get_user(user_id):
        db = app.config["db"]
        user = db.get_user(user_id)
        if not user:
            return jsonify({"error": "User không tồn tại"}), 404
        result = dict(user)
        result["has_password"] = db.has_password(user_id)
        return jsonify(result)

    @app.route("/api/users/<user_id>/update", methods=["POST"])
    def api_update_prefs(user_id):
        db = app.config["db"]
        if not db.get_user(user_id):
            return jsonify({"error": "User không tồn tại"}), 404
        try:
            data = request.get_json()
            preferences = data["preferences"]
            if not isinstance(preferences, dict):
                return jsonify({"error": "preferences phải là object"}), 400
        except (KeyError, TypeError):
            return jsonify({"error": "Body cần field 'preferences'"}), 400

        password = data.get("password", "")
        if not db.check_password(user_id, password):
            return jsonify({"error": "Sai mật khẩu"}), 403

        db.update_preferences(user_id, preferences)
        return jsonify({"ok": True})

    @app.route("/api/users/<user_id>/update-info", methods=["POST"])
    def api_update_user_info(user_id):
        db = app.config["db"]
        if not db.get_user(user_id):
            return jsonify({"error": "User không tồn tại"}), 404
        data = _json_body()
        password = data.get("password", "")
        if not db.check_password(user_id, password):
            return jsonify({"error": "Sai mật khẩu"}), 403

        if "name" in data:
            db.update_user_name(user_id, data["name"])
        new_password = data.get("new_password", "")
        if new_password:
            db.update_password(user_id, new_password)
        return jsonify({"ok": True})

    @app.route("/api/users/<user_id>/delete", methods=["POST"])
    def api_delete_user(user_id):
        db = app.config["db"]
        if not db.get_user(user_id):
            return jsonify({"error": "User không tồn tại"}), 404
        data = _json_body()
        password = data.get("password", "")
        if not db.check_password(user_id, password):
            return jsonify({"error": "Sai mật khẩu"}), 403

        db.delete_user(user_id)
        # Invalidate cache trong SpeakerManager
        app.config["spk_mgr"]._cache = None
        return jsonify({"ok": True})

    # ----- API: Assistant turn -----
    @app.route("/api/assistant/turn", methods=["POST"])
    @max_size(5)
    def api_assistant_turn():
        """Body: multipart với file 'audio' (WebM hoặc wav).

        Trả về:
          {
            transcript, intent, auth_level, entities,
            identified_user_id, identified_user_name, sid_score,
            sv_required, sv_passed, sv_score,
            response, blocked,
            tts_url    # GET URL để stream MP3 response
          }
        """
        if "audio" not in request.files:
            return jsonify({"error": "Thiếu file 'audio'"}), 400

        blob = request.files["audio"].read()
        try:
            audio = audio_io.decode_browser_audio(blob)
        except Exception as e:
            return jsonify({"error": f"Giải mã audio thất bại: {e}"}), 400

        audio, has_speech = audio_io.SileroVAD.trim(audio, return_speech_detected=True)
        _min_turn = int(config.SAMPLE_RATE * config.MIN_AUDIO_SEC_TURN)
        if not has_speech or audio.size < _min_turn:
            return jsonify({"error": "Audio quá ngắn hoặc không có giọng nói"}), 400

        # ASR → NLU → Router
        db     = app.config["db"]
        asr    = app.config["asr"]
        nlu    = app.config["nlu"]
        router = app.config["router"]

        transcript = asr.transcribe(audio)
        if not transcript:
            return jsonify({"error": "ASR không nhận diện được"}), 400

        transcript = correct_transcript(transcript)

        # ── Email flow: tiếp tục nếu đang trong luồng soạn email ──────────
        flow_state = None
        uid = None
        _user = None
        contacts = []
        if "email_flow" in session:
            flow_state = session["email_flow"]
            uid   = flow_state.get("user_id")
            _user = db.get_user(uid) if uid else None
            contacts = (_user["preferences"].get("contacts", [])
                        if _user else [])

            # Intent guard: nếu user nói intent khác (vd "phát nhạc đi"), thoát flow
            # và xử lý intent mới như bình thường. Tránh trường hợp câu đó bị nuốt
            # thành subject/body.
            if flow_state.get("step") in ("recipient", "subject", "body"):
                _quick_nlu = nlu.parse(transcript)
                _new_intent = _quick_nlu.get("intent", "unknown")
                _email_safe = {"send_email", "unknown", "general_question"}
                if _new_intent not in _email_safe:
                    session.pop("email_flow", None)
                    # Fall through xuống NLU/Router thông thường — không continue_flow
                    flow_state = None

        if flow_state is not None and "email_flow" in session:
            resp, new_state, is_done = _ef.continue_flow(
                transcript, flow_state, contacts)

            _oauth_extras = {}
            if is_done:
                ents = {
                    "recipient":       new_state["recipient_name"],
                    "recipient_email": new_state["recipient_email"],
                    "subject":         new_state["subject"],
                    "body":            new_state["body"],
                }
                _send_result = _send_email(ents, _user, db=db)
                # Handler trả HandlerResult nếu cần OAuth, str nếu đã gửi xong
                if isinstance(_send_result, _h_module.HandlerResult):
                    resp = _send_result.text
                    session["email_flow"] = new_state   # giữ để retry sau khi auth
                    flow_active = True
                    _oauth_extras = {"action_type": _send_result.action_type,
                                     "action_data": _send_result.action_data}
                else:
                    resp = _send_result
                    session.pop("email_flow", None)
                    flow_active = False
            elif new_state is None:
                session.pop("email_flow", None)
                flow_active = False
            else:
                session["email_flow"] = new_state
                flow_active = True

            payload = {
                "transcript":           transcript,
                "intent":               "email_flow",
                "auth_level":           "important",
                "entities":             {},
                "identified_user_id":   uid,
                "identified_user_name": _user["name"] if _user else "Khách",
                "sid_score":            1.0,
                "sv_required":          False,
                "sv_passed":            True,
                "sv_score":             None,
                "response":             resp,
                "blocked":              False,
                "flow_active":          flow_active,
                "tts_url":              f"/api/tts?text={_quote(resp)}",
                **_oauth_extras,
            }
            return jsonify(payload)
        # ── End email flow continuation ───────────────────────────────────

        nlu_result = nlu.parse(transcript)
        result = router.handle_turn(audio, transcript, nlu_result,
                                    extra_context={"db": db})

        # Nếu send_email vừa pass SV → bắt đầu flow thay vì gửi ngay
        if result.intent == "send_email" and not result.blocked:
            uid   = result.identified_user_id
            _user = db.get_user(uid) if uid else None
            contacts = (_user["preferences"].get("contacts", [])
                        if _user else [])
            question, flow_state = _ef.start_flow(
                result.entities.get("recipient", ""), contacts)
            flow_state["user_id"] = uid
            session["email_flow"] = flow_state

            payload = asdict(result)
            payload["response"]    = question
            payload["tts_url"]     = f"/api/tts?text={_quote(question)}"
            payload["flow_active"] = True
            return jsonify(payload)

        payload = asdict(result)
        payload["tts_url"] = f"/api/tts?text={_quote(result.response)}"
        payload["flow_active"] = False

        # action_type + action_data cho frontend mở panel tương ứng
        _uid = result.identified_user_id
        _user = db.get_user(_uid) if _uid else None
        _prefs = _user["preferences"] if _user else {}
        _attach_action_data(
            payload, result.intent, result.blocked,
            _uid, result.identified_user_name,
            result.entities, _prefs, auth_method="voice",
        )

        return jsonify(payload)

    # ----- API: Text mode (không cần micro) -----
    @app.route("/api/assistant/text", methods=["POST"])
    @max_size(1)
    def api_assistant_text():
        """Chế độ nhập văn bản thay vì giọng nói.
        Body JSON: { text, user_id (optional), password (optional) }
        - NORMAL/PERSONAL: không cần password
        - IMPORTANT: cần user_id + password (thay thế SV)
        """
        data     = _json_body()
        text     = data.get("text", "").strip()
        user_id  = data.get("user_id", "").strip()
        password = data.get("password", "")

        if not text:
            return jsonify({"error": "Thiếu nội dung lệnh"}), 400

        from core.intents import INTENTS, AuthLevel
        _h = _h_module  # alias dùng trong scope route

        db  = app.config["db"]
        nlu = app.config["nlu"]

        # ── Email flow: tiếp tục nếu đang trong luồng soạn email ──────────
        flow_state = None
        uid = None
        _user = None
        contacts = []
        if "email_flow" in session:
            flow_state = session["email_flow"]
            uid   = flow_state.get("user_id")
            _user = db.get_user(uid) if uid else None
            contacts = (_user["preferences"].get("contacts", [])
                        if _user else [])

            # Intent guard (xem api_assistant_turn): cho phép thoát flow bằng intent khác.
            if flow_state.get("step") in ("recipient", "subject", "body"):
                _quick_nlu = nlu.parse(text)
                _new_intent = _quick_nlu.get("intent", "unknown")
                _email_safe = {"send_email", "unknown", "general_question"}
                if _new_intent not in _email_safe:
                    session.pop("email_flow", None)
                    flow_state = None

        if flow_state is not None and "email_flow" in session:
            resp, new_state, is_done = _ef.continue_flow(text, flow_state, contacts)

            _oauth_extras = {}
            if is_done:
                ents = {
                    "recipient":       new_state["recipient_name"],
                    "recipient_email": new_state["recipient_email"],
                    "subject":         new_state["subject"],
                    "body":            new_state["body"],
                }
                _send_result = _send_email(ents, _user, db=db)
                # Handler trả HandlerResult nếu cần OAuth, str nếu đã gửi xong
                if isinstance(_send_result, _h_module.HandlerResult):
                    resp = _send_result.text
                    session["email_flow"] = new_state   # giữ để retry sau khi auth
                    flow_active = True
                    _oauth_extras = {"action_type": _send_result.action_type,
                                     "action_data": _send_result.action_data}
                else:
                    resp = _send_result
                    session.pop("email_flow", None)
                    flow_active = False
            elif new_state is None:
                session.pop("email_flow", None)
                flow_active = False
            else:
                session["email_flow"] = new_state
                flow_active = True

            payload = {
                "transcript":           text,
                "intent":               "email_flow",
                "auth_level":           "important",
                "entities":             {},
                "identified_user_id":   uid,
                "identified_user_name": _user["name"] if _user else "Khách",
                "sid_score":            1.0 if uid else 0.0,
                "sv_required":          False,
                "sv_passed":            True,
                "sv_score":             None,
                "response":             resp,
                "blocked":              False,
                "flow_active":          flow_active,
                "tts_url":              f"/api/tts?text={_quote(resp)}",
                **_oauth_extras,
            }
            return jsonify(payload)
        # ── End email flow continuation ───────────────────────────────────

        nlu_result = nlu.parse(text)
        intent   = nlu_result["intent"]
        entities = nlu_result["entities"]
        spec     = INTENTS.get(intent, INTENTS["unknown"])
        level    = spec["level"]

        user = db.get_user(user_id) if user_id else None
        uid  = user["user_id"] if user else None
        name = user["name"] if user else "Khách"

        sv_required = (level == AuthLevel.IMPORTANT)
        sv_passed   = None
        blocked     = False
        response    = ""

        if sv_required:
            if not uid:
                blocked  = True
                response = ("Tác vụ này yêu cầu xác thực. "
                            "Vui lòng chọn người dùng và nhập mật khẩu.")
            elif not db.check_password(uid, password):
                blocked  = True
                response = "Sai mật khẩu. Không thể thực hiện tác vụ này."
            else:
                sv_passed = True

        # Nếu send_email pass auth → bắt đầu flow thay vì gửi ngay
        if intent == "send_email" and not blocked:
            _user2 = db.get_user(uid) if uid else None
            contacts = (_user2["preferences"].get("contacts", [])
                        if _user2 else [])
            question, flow_state = _ef.start_flow(
                entities.get("recipient", ""), contacts)
            flow_state["user_id"] = uid
            session["email_flow"] = flow_state

            payload = {
                "transcript":           text,
                "intent":               "send_email",
                "auth_level":           level.value,
                "entities":             entities,
                "identified_user_id":   uid,
                "identified_user_name": name,
                "sid_score":            1.0 if uid else 0.0,
                "sv_required":          sv_required,
                "sv_passed":            sv_passed,
                "sv_score":             None,
                "response":             question,
                "blocked":              False,
                "flow_active":          True,
                "tts_url":              f"/api/tts?text={_quote(question)}",
            }
            return jsonify(payload)

        if not blocked:
            handler  = _h.HANDLERS.get(intent, _h.handle_unknown)
            response = handler(entities, user, db=db)

        payload = {
            "transcript":           text,
            "intent":               intent,
            "auth_level":           level.value,
            "entities":             entities,
            "identified_user_id":   uid,
            "identified_user_name": name,
            "sid_score":            1.0 if uid else 0.0,
            "sv_required":          sv_required,
            "sv_passed":            sv_passed,
            "sv_score":             None,
            "response":             response,
            "blocked":              blocked,
            "flow_active":          False,
            "tts_url":              f"/api/tts?text={_quote(response)}",
        }

        _prefs = user["preferences"] if user else {}
        _attach_action_data(
            payload, intent, blocked,
            uid, name, entities, _prefs, auth_method="password",
        )

        return jsonify(payload)

    # ----- API: TTS streaming -----
    @app.route("/api/tts")
    def api_tts():
        text = request.args.get("text", "").strip()
        if not text:
            return jsonify({"error": "Thiếu param 'text'"}), 400
        # gTTS không xử lý tốt newline → thay bằng dấu cách
        text = " ".join(text.split())
        # Giới hạn độ dài để tránh timeout (gTTS vẫn xử lý ổn với ~500 ký tự)
        if len(text) > 500:
            text = text[:497] + "..."
        try:
            mp3 = app.config["tts"].synthesize_to_mp3_bytes(text)
        except Exception:
            # gTTS lỗi (mất mạng, rate limit...). Trả về 204 No Content thay vì
            # bytes giả "\xff\xe3..." (không phải MP3 hợp lệ, gây lỗi audio player).
            return ("", 204)
        return send_file(io.BytesIO(mp3), mimetype="audio/mpeg",
                         as_attachment=False)

    # ==========================================================================
    # Music routes
    # ==========================================================================
    @app.route("/music")
    def music_page():
        users = app.config["db"].list_users()
        return render_template("music.html", users=users)

    @app.route("/api/music/search")
    def api_music_search():
        q = request.args.get("q", "").strip()
        if not q:
            return jsonify({"tracks": []})
        try:
            import requests as _req
            resp = _req.get("https://api.deezer.com/search",
                            params={"q": q, "limit": 20}, timeout=6)
            tracks = [
                {"id": t["id"], "title": t["title"],
                 "artist": t["artist"]["name"],
                 "cover": t["album"]["cover_small"],
                 "preview": t["preview"]}
                for t in resp.json().get("data", []) if t.get("preview")
            ]
        except Exception as e:
            return jsonify({"error": str(e), "tracks": []})
        return jsonify({"tracks": tracks})

    @app.route("/api/music/playlist/<user_id>")
    def api_playlist_get(user_id):
        user = app.config["db"].get_user(user_id)
        if not user:
            return jsonify({"error": "User không tồn tại"}), 404
        return jsonify({
            "playlist": user["preferences"].get("playlist", []),
            "favorite_genre": user["preferences"].get("favorite_genre", ""),
            "favorite_artist": user["preferences"].get("favorite_artist", ""),
        })

    @app.route("/api/music/playlist/<user_id>/add", methods=["POST"])
    def api_playlist_add(user_id):
        db = app.config["db"]
        user = db.get_user(user_id)
        if not user:
            return jsonify({"error": "User không tồn tại"}), 404
        track = request.get_json()
        if not track or not {"id", "title", "artist", "preview"}.issubset(track):
            return jsonify({"error": "Thiếu field track"}), 400
        playlist = user["preferences"].get("playlist", [])
        if not any(t["id"] == track["id"] for t in playlist):
            playlist.append({k: track[k] for k in
                             ["id", "title", "artist", "cover", "preview"] if k in track})
            db.update_preferences(user_id, {**user["preferences"], "playlist": playlist})
        return jsonify({"ok": True, "count": len(playlist)})

    @app.route("/api/music/playlist/<user_id>/remove", methods=["POST"])
    def api_playlist_remove(user_id):
        db = app.config["db"]
        user = db.get_user(user_id)
        if not user:
            return jsonify({"error": "User không tồn tại"}), 404
        track_id = (_json_body()).get("id")
        playlist = [t for t in user["preferences"].get("playlist", [])
                    if t["id"] != track_id]
        db.update_preferences(user_id, {**user["preferences"], "playlist": playlist})
        return jsonify({"ok": True, "count": len(playlist)})

    # ==========================================================================
    # User music tracks (file-based personal playlist)
    # ==========================================================================
    def _music_dir(uid: str) -> Path:
        d = config.USER_FILES_DIR / uid / "music"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _list_tracks(uid: str) -> list:
        d = _music_dir(uid)
        return sorted(
            [{"name": f.name,
              "size": f.stat().st_size,
              "url": f"/api/music/user-tracks/{uid}/stream/{f.name}"}
             for f in d.iterdir()
             if f.is_file() and f.suffix.lower() in AUDIO_EXTS],
            key=lambda x: x["name"]
        )

    @app.route("/api/music/user-tracks/<user_id>")
    def api_user_tracks_list(user_id):
        if _get_authed_uid("any") != user_id:
            return jsonify({"error": "Chưa xác thực hoặc phiên đã hết hạn"}), 403
        return jsonify({"tracks": _list_tracks(user_id)})

    @app.route("/api/music/user-tracks/<user_id>/upload", methods=["POST"])
    @max_size(50)
    def api_user_tracks_upload(user_id):
        if _get_authed_uid("any") != user_id:
            return jsonify({"error": "Chưa xác thực hoặc phiên đã hết hạn"}), 403
        uploaded = []
        for key in request.files:
            f = request.files[key]
            if not f.filename:
                continue
            fname, is_audio = _safe_filename(f.filename, allowed_exts=AUDIO_EXTS)
            if not is_audio:
                continue
            f.save(_music_dir(user_id) / fname)
            uploaded.append(fname)
        if not uploaded:
            return jsonify({"error": "Không có file audio hợp lệ"}), 400
        return jsonify({"ok": True, "uploaded": uploaded,
                        "tracks": _list_tracks(user_id)})

    @app.route("/api/music/user-tracks/<user_id>/delete/<path:filename>",
               methods=["POST"])
    def api_user_tracks_delete(user_id, filename):
        if _get_authed_uid("any") != user_id:
            return jsonify({"error": "Chưa xác thực hoặc phiên đã hết hạn"}), 403
        base = _music_dir(user_id).resolve()
        fpath = (base / filename).resolve()
        if not str(fpath).startswith(str(base)):
            return jsonify({"error": "Không hợp lệ"}), 400
        if not fpath.exists():
            return jsonify({"error": "File không tồn tại"}), 404
        fpath.unlink()
        return jsonify({"ok": True, "tracks": _list_tracks(user_id)})

    @app.route("/api/music/user-tracks/<user_id>/stream/<path:filename>")
    def api_user_tracks_stream(user_id, filename):
        """Stream audio file — yêu cầu file session hoặc query token."""
        if _get_authed_uid("any") != user_id:
            return jsonify({"error": "Chưa xác thực hoặc phiên đã hết hạn"}), 403
        base = _music_dir(user_id).resolve()
        fpath = (base / filename).resolve()
        if not str(fpath).startswith(str(base)) or not fpath.exists():
            return jsonify({"error": "File không tồn tại"}), 404
        return send_file(fpath, conditional=True)

    def _ydl_base_opts(**extra) -> dict:
        """Build yt-dlp options with cookie auth to bypass YouTube bot check.

        Priority:
          1. youtube_cookies.txt next to this file  (export from browser)
          2. Cookies read live from Chrome / Edge / Firefox on this machine
          3. No cookies (may fail on restricted videos)
        """
        import yt_dlp  # noqa: F401 — ensure import available for callers
        opts = {
            "quiet": True,
            "no_warnings": True,
            **extra,
        }
        cookies_file = Path(__file__).parent / "youtube_cookies.txt"
        if cookies_file.exists():
            opts["cookiefile"] = str(cookies_file)
        else:
            for browser in ("chrome", "edge", "firefox", "chromium", "brave"):
                try:
                    import yt_dlp as _y
                    test_opts = {**opts, "cookiesfrombrowser": (browser,)}
                    with _y.YoutubeDL({**test_opts, "extract_flat": True}) as _ydl:
                        _ydl.cookiejar  # trigger load; raises if browser not found
                    opts["cookiesfrombrowser"] = (browser,)
                    break
                except Exception:
                    continue
        return opts

    # YouTube search proxy (yt-dlp, không cần API key)
    @app.route("/api/music/youtube-search")
    def api_youtube_search():
        q = request.args.get("q", "").strip()
        if not q:
            return jsonify({"videos": []})
        try:
            import yt_dlp
            ydl_opts = _ydl_base_opts(
                extract_flat=True,
                skip_download=True,
            )
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"ytsearch10:{q}", download=False)
            videos = []
            for e in (info.get("entries") or []):
                if not e or not e.get("id"):
                    continue
                thumbs = e.get("thumbnails") or []
                thumb = thumbs[0]["url"] if thumbs else (
                    e.get("thumbnail") or
                    f"https://i.ytimg.com/vi/{e['id']}/default.jpg"
                )
                videos.append({
                    "id":       e["id"],
                    "title":    e.get("title", ""),
                    "channel":  e.get("uploader") or e.get("channel", ""),
                    "thumbnail": thumb,
                    "duration": e.get("duration", 0),
                    "url":      f"https://www.youtube.com/watch?v={e['id']}",
                })
        except Exception as ex:
            return jsonify({"error": str(ex), "videos": []})
        return jsonify({"videos": videos})

    @app.route("/api/music/yt-audio")
    def api_yt_audio():
        """Extract YouTube audio stream URL via yt-dlp and redirect to it."""
        import yt_dlp
        video_id = request.args.get("v", "").strip()
        # YouTube video ID luôn 11 ký tự — cố định độ dài chống injection
        if not video_id or not re.match(r'^[a-zA-Z0-9_-]{11}$', video_id):
            return jsonify({"error": "Video ID không hợp lệ"}), 400
        try:
            ydl_opts = _ydl_base_opts(
                format="bestaudio[ext=webm]/bestaudio[ext=m4a]/bestaudio/best",
                skip_download=True,
            )
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(
                    f"https://www.youtube.com/watch?v={video_id}",
                    download=False
                )
            url = None
            for fmt in sorted(info.get("formats", []),
                               key=lambda f: f.get("tbr") or 0, reverse=True):
                if fmt.get("vcodec") in (None, "none") and fmt.get("url"):
                    url = fmt["url"]
                    break
            if not url:
                url = info.get("url")
            if not url:
                return jsonify({"error": "Không tìm thấy luồng audio"}), 404
            return redirect(url)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ==========================================================================
    # File manager routes (SV-protected via Flask session)
    # ==========================================================================
    @app.route("/files")
    def files_page():
        return render_template("files.html")

    @app.route("/api/files/verify", methods=["POST"])
    @max_size(5)
    def api_files_verify():
        if "audio" not in request.files:
            return jsonify({"error": "Thiếu audio"}), 400
        blob = request.files["audio"].read()
        try:
            aud = audio_io.decode_browser_audio(blob)
        except Exception as e:
            return jsonify({"error": f"Giải mã audio thất bại: {e}"}), 400
        aud, has_speech = audio_io.SileroVAD.trim(aud, return_speech_detected=True)
        if not has_speech or aud.size < int(config.SAMPLE_RATE * config.MIN_AUDIO_SEC_TURN):
            return jsonify({"error": "Audio quá ngắn / không có giọng nói"}), 400
        spk_mgr = app.config["spk_mgr"]
        uid, name, sid_score = spk_mgr.identify(aud)
        if uid is None:
            return jsonify({"ok": False,
                            "error": f"Không nhận ra giọng (score={sid_score:.2f}). "
                                     "Đăng ký trước hoặc nói rõ hơn."}), 403
        sv_passed, sv_score = spk_mgr.verify(aud, uid)
        if not sv_passed:
            return jsonify({"ok": False,
                            "error": f"Xác thực thất bại (score={sv_score:.2f}). "
                                     "Giọng không khớp với dữ liệu đã đăng ký."}), 403
        _set_file_session(uid, name, "voice")
        return jsonify({"ok": True, "user_id": uid, "user_name": name,
                        "sid_score": round(sid_score, 3),
                        "sv_score": round(sv_score, 3)})

    @app.route("/api/files/verify-password", methods=["POST"])
    def api_files_verify_password():
        data = _json_body()
        user_id = data.get("user_id", "")
        password = data.get("password", "")
        db = app.config["db"]
        user = db.get_user(user_id)
        if not user:
            return jsonify({"error": "User không tồn tại"}), 404
        if not db.check_password(user_id, password):
            return jsonify({"error": "Sai mật khẩu"}), 403
        _set_file_session(user_id, user.get("name", user_id), "password")
        return jsonify({"ok": True, "user_name": user.get("name", user_id)})

    @app.route("/api/users/<user_id>/unlock", methods=["POST"])
    def api_user_unlock(user_id):
        """Xác thực mật khẩu, set file session, trả về user info.
        Dùng thay cho update-info khi chỉ cần mở khóa modal."""
        db = app.config["db"]
        user = db.get_user(user_id)
        if not user:
            return jsonify({"error": "User không tồn tại"}), 404
        data = _json_body()
        password = data.get("password", "")
        if not db.check_password(user_id, password):
            return jsonify({"error": "Sai mật khẩu"}), 403
        _set_file_session(user_id, user["name"], "password")
        result = dict(user)
        result["has_password"] = db.has_password(user_id)
        return jsonify({"ok": True, "user": result})

    @app.route("/api/files/logout", methods=["POST"])
    def api_files_logout():
        _clear_file_session()
        return jsonify({"ok": True})

    @app.route("/api/files/list")
    def api_files_list():
        uid = _get_authed_uid("any")
        if not uid:
            return jsonify({"error": "Chưa xác thực hoặc phiên đã hết hạn"}), 403
        user_dir = config.USER_FILES_DIR / uid
        files = []
        if user_dir.exists():
            for f in sorted(user_dir.iterdir()):
                if f.is_file():
                    files.append({
                        "name": f.name,
                        "size": f.stat().st_size,
                        "modified": datetime.fromtimestamp(
                            f.stat().st_mtime).strftime("%d/%m/%Y %H:%M"),
                    })
        return jsonify({"files": files,
                        "user_name": session.get("file_name"),
                        "user_id": uid})

    @app.route("/api/files/upload", methods=["POST"])
    @max_size(50)
    def api_files_upload():
        uid = _get_authed_uid("any")
        if not uid:
            return jsonify({"error": "Chưa xác thực hoặc phiên đã hết hạn"}), 403
        uploaded = []
        rejected = []
        for key in request.files:
            f = request.files[key]
            if not f.filename:
                continue
            # Cho phép file văn bản/ảnh/zip thông thường — chặn .exe/.bat/.html (SVG XSS, etc.)
            fname, is_allowed = _safe_filename(f.filename, allowed_exts=DOC_EXTS)
            if not is_allowed:
                rejected.append(f.filename)
                continue
            user_dir = config.USER_FILES_DIR / uid
            user_dir.mkdir(parents=True, exist_ok=True)
            f.save(user_dir / fname)
            uploaded.append(fname)
        if not uploaded:
            msg = "Không có file hợp lệ"
            if rejected:
                msg += f" (định dạng không hỗ trợ: {', '.join(rejected[:3])})"
            return jsonify({"error": msg}), 400
        resp = {"ok": True, "uploaded": uploaded}
        if rejected:
            resp["rejected"] = rejected
        return jsonify(resp)

    @app.route("/api/files/download/<path:filename>")
    def api_files_download(filename):
        uid = _get_authed_uid("any")
        if not uid:
            return jsonify({"error": "Chưa xác thực hoặc phiên đã hết hạn"}), 403
        base = (config.USER_FILES_DIR / uid).resolve()
        fpath = (base / filename).resolve()
        if not str(fpath).startswith(str(base)):
            return jsonify({"error": "Không hợp lệ"}), 400
        if not fpath.exists():
            return jsonify({"error": "File không tồn tại"}), 404
        return send_file(fpath, as_attachment=True)

    @app.route("/api/files/delete/<path:filename>", methods=["POST"])
    def api_files_delete(filename):
        uid = _get_authed_uid("any")
        if not uid:
            return jsonify({"error": "Chưa xác thực hoặc phiên đã hết hạn"}), 403
        base = (config.USER_FILES_DIR / uid).resolve()
        fpath = (base / filename).resolve()
        if not str(fpath).startswith(str(base)):
            return jsonify({"error": "Không hợp lệ"}), 400
        if not fpath.exists():
            return jsonify({"error": "File không tồn tại"}), 404
        fpath.unlink()
        return jsonify({"ok": True})

    # ----- Admin -----
    @app.route("/api/admin/verify", methods=["POST"])
    @max_size(0.01)
    def api_admin_verify():
        data = _json_body()
        supplied = data.get("password", "")
        if (not config.ADMIN_PASS or
                not hmac.compare_digest(str(supplied), config.ADMIN_PASS)):
            return jsonify({"error": "Sai mật khẩu quản trị"}), 403
        users = app.config["db"].list_users()
        return jsonify({
            "ok": True,
            "count": len(users),
            "users": [
                {
                    "user_id":    u["user_id"],
                    "name":       u["name"],
                    "created_at": u["created_at"][:10],
                }
                for u in users
            ],
        })

    # ==========================================================================
    # Google OAuth 2.0 — Gmail API authentication
    # ==========================================================================
    @app.route("/auth/google")
    def auth_google():
        """Redirect user đến Google consent screen.

        Query param: user_id (bắt buộc) — bind authorization code về đúng speaker profile.
        State được generate ngẫu nhiên + lưu vào session để chống CSRF
        (kẻ tấn công không thể gọi /auth/google?user_id=victim trên máy họ
        rồi callback gắn Gmail attacker vào profile victim).
        """
        user_id = request.args.get("user_id", "").strip()
        if not user_id:
            return "Thiếu user_id", 400
        # Random nonce làm OAuth state, lưu cùng pending user_id vào session.
        # PKCE: gửi code_challenge qua URL, giữ code_verifier server-side. Khi
        # callback exchange code → server gửi verifier kèm → Google verify
        # SHA256(verifier) == challenge. Attacker intercept code không có verifier
        # → không đổi được token.
        nonce = secrets.token_urlsafe(24)
        verifier, challenge = gen_pkce_pair()
        session["oauth_nonce"]         = nonce
        session["oauth_pending_uid"]   = user_id
        session["oauth_code_verifier"] = verifier
        session.permanent = True
        try:
            url = build_auth_url(state=nonce, code_challenge=challenge)
        except RuntimeError as e:
            return str(e), 500
        return redirect(url)

    @app.route("/auth/google/callback")
    def auth_google_callback():
        """Google redirect về đây với code + state sau khi user đồng ý.

        Flow:
          1. Validate state khớp với nonce trong session (CSRF check)
          2. Trao đổi code lấy access_token + refresh_token
          3. Lấy Gmail address của user
          4. Lưu token vào DB (oauth_tokens table)
          5. Cập nhật user preferences với Gmail address
          6. Trả về trang thành công (user có thể đóng tab)
        """
        error          = request.args.get("error")
        code           = request.args.get("code")
        received_state = request.args.get("state", "")

        # Lấy nonce + user_id + verifier từ session (đã set trong auth_google)
        expected_nonce = session.pop("oauth_nonce", None)
        user_id        = session.pop("oauth_pending_uid", None)
        code_verifier  = session.pop("oauth_code_verifier", None)

        # CSRF check: state phải khớp với nonce server-side
        if (expected_nonce is None or
                not hmac.compare_digest(str(received_state), str(expected_nonce))):
            return "<h3>State không hợp lệ (CSRF detected hoặc session expired)</h3>", 400

        # Lỗi từ Google (vd: user từ chối, redirect_uri_mismatch)
        if error:
            return (
                "<!doctype html><html><head><meta charset='utf-8'>"
                "<style>body{font-family:sans-serif;text-align:center;padding:60px}"
                "h2{color:#c62828}pre{background:#f5f5f5;padding:12px;border-radius:6px;"
                "text-align:left;max-width:500px;margin:auto;word-break:break-all}</style></head><body>"
                f"<h2>✗ Xác thực thất bại</h2>"
                f"<pre>{error}</pre>"
                "<p>Kiểm tra lại <b>Authorized Redirect URIs</b> trong Google Cloud Console.<br>"
                "URI phải khớp chính xác với <code>GOOGLE_REDIRECT_URI</code> trong <code>.env</code>.</p>"
                "<button onclick='window.close()' style='margin-top:16px;padding:8px 20px;"
                "background:#1565c0;color:#fff;border:none;border-radius:6px;cursor:pointer'>"
                "Đóng tab</button>"
                "</body></html>"
            ), 400

        if not code or not user_id:
            return "<h3>Thiếu tham số (code hoặc state)</h3>", 400

        _db = app.config["db"]
        user = _db.get_user(user_id)
        if not user:
            return f"<h3>Không tìm thấy user '{user_id}'</h3>", 404

        try:
            tokens     = exchange_code(code, code_verifier=code_verifier)
            gmail_addr = get_user_email(tokens["access_token"])
            tokens["gmail_address"] = gmail_addr
            _db.save_oauth_token(user_id, tokens)

            prefs = dict(user["preferences"])
            prefs["email"] = gmail_addr
            _db.update_preferences(user_id, prefs)
        except Exception as e:
            return (
                "<!doctype html><html><head><meta charset='utf-8'>"
                "<style>body{font-family:sans-serif;text-align:center;padding:60px}"
                "h2{color:#c62828}pre{background:#f5f5f5;padding:12px;border-radius:6px;"
                "text-align:left;max-width:500px;margin:auto;word-break:break-all}</style></head><body>"
                f"<h2>✗ Lỗi trao đổi token</h2><pre>{e}</pre>"
                "<button onclick='window.close()' style='margin-top:16px;padding:8px 20px;"
                "background:#1565c0;color:#fff;border:none;border-radius:6px;cursor:pointer'>"
                "Đóng tab</button></body></html>"
            ), 500

        # Tab tự đóng sau 3s, đồng thời báo hiệu cho cửa sổ mẹ (nếu có)
        return (
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<style>body{font-family:sans-serif;text-align:center;padding:60px}"
            "h2{color:#2e7d32}.badge{display:inline-block;background:#e8f5e9;"
            "border:1px solid #a5d6a7;border-radius:8px;padding:12px 24px;margin:12px 0}"
            ".cnt{font-size:2rem;font-weight:bold;color:#1565c0}</style></head><body>"
            f"<h2>✓ Xác thực Gmail thành công!</h2>"
            f"<div class='badge'>📧 <strong>{gmail_addr}</strong></div>"
            "<p>Tab sẽ tự đóng trong <span id='cnt' class='cnt'>3</span> giây.<br>"
            "Quay lại trang trợ lý và nói <b>\"có\"</b> để gửi email.</p>"
            "<button onclick='window.close()' style='margin-top:8px;padding:8px 20px;"
            "background:#2e7d32;color:#fff;border:none;border-radius:6px;cursor:pointer'>"
            "Đóng ngay</button>"
            "<script>"
            "try{new BroadcastChannel('oauth').postMessage({type:'oauth_done'});}catch(_){}"
            "if(window.opener){try{window.opener.postMessage({type:'oauth_done'},'*');}catch(_){}}"
            "let s=3;const t=setInterval(()=>{s--;document.getElementById('cnt').textContent=s;"
            "if(s<=0){clearInterval(t);window.close();}},1000);"
            "</script>"
            "</body></html>"
        )

    @app.route("/api/oauth/status/<user_id>")
    def api_oauth_status(user_id):
        """Kiểm tra trạng thái OAuth của user — dùng bởi frontend polling."""
        _db  = app.config["db"]
        tok  = _db.get_oauth_token(user_id)
        return jsonify({
            "authenticated": tok is not None,
            "gmail":         tok["gmail_address"] if tok else "",
        })

    @app.route("/api/oauth/debug/<user_id>", methods=["POST"])
    @max_size(0.01)
    def api_oauth_debug(user_id):
        """Kiểm tra token có scope gmail.send không — chỉ admin xem được.

        Endpoint cũ GET public sẽ leak tokeninfo. Yêu cầu admin password trong
        body JSON {password} để dùng.
        """
        data = _json_body()
        supplied = data.get("password", "")
        if (not config.ADMIN_PASS
                or not hmac.compare_digest(str(supplied), config.ADMIN_PASS)):
            return jsonify({"error": "Yêu cầu mật khẩu quản trị"}), 403
        _db = app.config["db"]
        tok = _db.get_oauth_token(user_id)
        if not tok:
            return jsonify({"error": "Chưa có token"}), 404
        import requests as _req
        r = _req.get(
            "https://www.googleapis.com/oauth2/v1/tokeninfo",
            params={"access_token": tok["access_token"]}, timeout=5)
        return jsonify({"tokeninfo": r.json(), "status": r.status_code})

    @app.route("/api/oauth/revoke/<user_id>", methods=["POST"])
    def api_oauth_revoke(user_id):
        """Thu hồi token và xóa khỏi DB — user đăng xuất Gmail."""
        _db = app.config["db"]
        tok = _db.get_oauth_token(user_id)
        if not tok:
            return jsonify({"ok": True, "note": "Không có token để revoke"})
        from core.oauth import revoke_token
        google_ok = revoke_token(tok.get("access_token", ""))
        # Luôn xóa token local kể cả khi Google revoke fail (network down...) — quyết
        # định an toàn: ưu tiên không giữ token chưa thu hồi trên máy. Báo cho client biết.
        _db.delete_oauth_token(user_id)
        return jsonify({"ok": True, "google_revoked": google_ok})

    # ----- Misc -----
    @app.route("/api/health")
    def health():
        return jsonify({
            "ok": True,
            "n_users": len(app.config["db"].list_users()),
            "encoder_mode": app.config["spk_mgr"].encoder.mode,
            "asr_model": config.WHISPER_MODEL,
            "nlu": app.config["nlu"].__class__.__name__,
        })


# ==========================================================================
# Entrypoint
# ==========================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssl", action="store_true",
                        help="Bật HTTPS (cần pyopenssl) — cho phép dùng mic qua IP")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    app = create_app()

    ssl_context = None
    if args.ssl:
        cert_path = Path(__file__).parent / "cert.pem"
        key_path  = Path(__file__).parent / "key.pem"
        if cert_path.exists() and key_path.exists():
            ssl_context = (str(cert_path), str(key_path))
            _log.info("✓ HTTPS enabled (persistent cert) — truy cập qua https://<IP>:%s", args.port)
        else:
            _log.warning("⚠ Chưa có cert.pem / key.pem — dùng adhoc (cert thay đổi mỗi lần restart)")
            _log.warning("  Tạo cert cố định: python web/gen_cert.py")
            ssl_context = "adhoc"

    # threaded=True: I/O routes (static, /api/health, file upload/download) chạy
    # song song. Các model torch (ASR/encoder/VAD) đã có threading.Lock riêng để
    # serialize forward pass, tránh race condition khi nhiều request đến cùng lúc.
    app.run(host="0.0.0.0", port=args.port, debug=False,
            threaded=True, ssl_context=ssl_context)
