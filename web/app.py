
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
import hashlib
import hmac
import html
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
                   redirect, url_for, flash, session, g)


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


def _resolve_child_path(base: Path, child: str) -> Path | None:
    """Resolve user-controlled path and require it to stay under base."""
    base_resolved = base.resolve()
    candidate = (base_resolved / child).resolve()
    try:
        candidate.relative_to(base_resolved)
    except ValueError:
        return None
    return candidate


def _mask_email(email: str) -> str:
    """Mask email cho admin view — show first 3 char + ***@domain.
    Vd: nguyenvanminh@gmail.com → ngu***@gmail.com.
    Tránh dump PII cho admin nhưng vẫn cho admin nhận diện account.
    """
    if not email or "@" not in email:
        return email or ""
    local, _, domain = email.partition("@")
    if len(local) <= 3:
        masked_local = local[:1] + "***" if local else "***"
    else:
        masked_local = local[:3] + "***"
    return f"{masked_local}@{domain}"


def _mask_name(name: str) -> str:
    """Mask tên user. Vd: Nguyễn Văn Minh → Nguyễn V*** M***."""
    if not name:
        return ""
    parts = name.strip().split()
    if len(parts) <= 1:
        return parts[0][:1] + "***" if parts else "***"
    return parts[0] + " " + " ".join(p[:1] + "***" for p in parts[1:])

# Thêm project root vào path để import được core/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import audio_io, config
from core.asr import get_asr, correct_transcript
from core.audit import audit as _audit
from core.turn_logging import log_turn as _log_turn
from core import email_flow as _ef
from core import handlers as _h_module
from core.handlers import handle_send_email as _send_email
from core.oauth import (build_auth_url, exchange_code, gen_pkce_pair,
                        get_user_email)
from core.tts import get_tts
from core.nlu import get_nlu, parse_with_correction
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

    # Session cookie hardening:
    #   HTTPONLY  → JS không đọc được cookie (chống XSS đánh cắp session).
    #   SAMESITE  → Lax: cross-site request không gửi cookie (chống CSRF).
    #   SECURE    → chỉ gửi cookie qua HTTPS. Bật mặc định; tắt khi dev HTTP
    #               qua env FLASK_DEV_HTTP=1 (vd: localhost không SSL).
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    _dev_http = os.environ.get("FLASK_DEV_HTTP", "").strip() in ("1", "true", "True")
    app.config["SESSION_COOKIE_SECURE"]   = not _dev_http
    if _dev_http:
        _log.warning("FLASK_DEV_HTTP=1 → SESSION_COOKIE_SECURE=False (chỉ dùng dev HTTP)")
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


def _log_assistant_turn(payload: dict, *, source: str, auth_method: str):
    """Best-effort structured turn logging with request correlation."""
    return _log_turn(payload, source=source, auth_method=auth_method,
                     request_id=getattr(g, "request_id", ""))


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
# Rate limit (in-process) — chống brute force password endpoints.
# Process-local, đủ cho single-worker dev/demo. Production multi-worker cần
# Redis-backed (Flask-Limiter); xem todo P1.
# ==========================================================================
import threading
from collections import defaultdict, deque

_RL_LOCK = threading.Lock()
_RL_BUCKETS: dict = defaultdict(deque)

_OAUTH_PENDING_LOCK = threading.Lock()
_OAUTH_PENDING: dict[str, dict] = {}


def _oauth_pending_put(state: str, payload: dict):
    now = time.time()
    with _OAUTH_PENDING_LOCK:
        expired = [k for k, v in _OAUTH_PENDING.items()
                   if now - v.get("created_at", 0) > 600]
        for k in expired:
            _OAUTH_PENDING.pop(k, None)
        _OAUTH_PENDING[state] = {**payload, "created_at": now}


def _oauth_pending_pop(state: str) -> dict | None:
    now = time.time()
    with _OAUTH_PENDING_LOCK:
        payload = _OAUTH_PENDING.pop(state, None)
    if payload and now - payload.get("created_at", 0) <= 600:
        return payload
    return None


def _rl_client_ip() -> str:
    # request.remote_addr; trust proxy chỉ khi có header rõ ràng (default off).
    xff = request.headers.get("X-Forwarded-For", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.remote_addr or "unknown"


def rate_limit(max_attempts: int, window_sec: int, scope: str,
               key_user_field: str | None = None):
    """Decorator giới hạn N attempts / window / (ip, user_id).

    scope        : tên bucket (vd "files-verify", "user-info").
    key_user_field: tên route param chứa user_id để key thêm; None → chỉ key IP.

    Trả 429 + Retry-After khi vượt. KHÔNG block lâu — chỉ delay; user-friendly fail.
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            ip = _rl_client_ip()
            uid = kwargs.get(key_user_field, "") if key_user_field else ""
            key = (scope, ip, uid)
            now = time.time()
            with _RL_LOCK:
                bucket = _RL_BUCKETS[key]
                while bucket and now - bucket[0] > window_sec:
                    bucket.popleft()
                if len(bucket) >= max_attempts:
                    retry = int(window_sec - (now - bucket[0])) + 1
                    resp = jsonify({
                        "error": f"Quá nhiều lần thử. Vui lòng đợi {retry}s rồi thử lại.",
                        "retry_after": retry,
                    })
                    resp.status_code = 429
                    resp.headers["Retry-After"] = str(retry)
                    return resp
                bucket.append(now)
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


def _continue_email_flow_if_active(text: str, db, nlu):
    """Nếu session có email_flow đang active, drive flow tiếp.

    Dùng chung cho voice route (`/api/assistant/turn`) và text route
    (`/api/assistant/text`) — logic identical, chỉ khác nguồn `text`.

    Trả về `(payload_dict, True)` nếu đã xử lý email_flow trong turn này;
    caller chỉ cần `return jsonify(payload)`. Trả `(None, False)` nếu không
    có flow active → caller chạy NLU/Router thông thường.
    """
    if "email_flow" not in session:
        return None, False

    flow_state = session["email_flow"]
    uid   = flow_state.get("user_id")
    _user = db.get_user(uid) if uid else None
    contacts = (_user["preferences"].get("contacts", []) if _user else [])

    # Intent guard: nếu user nói intent khác (vd "phát nhạc đi"), thoát flow
    # để tránh câu đó bị nuốt thành recipient/subject/body.
    if flow_state.get("step") in ("recipient", "subject", "body"):
        _quick_nlu = nlu.parse(text)
        if _quick_nlu.get("intent", "unknown") not in (
                "send_email", "unknown", "general_question"):
            session.pop("email_flow", None)
            return None, False

    resp, new_state, is_done = _ef.continue_flow(text, flow_state, contacts)
    _oauth_extras: dict = {}

    if is_done:
        ents = {
            "recipient":       new_state["recipient_name"],
            "recipient_email": new_state["recipient_email"],
            "subject":         new_state["subject"],
            "body":            new_state["body"],
        }
        send_result = _send_email(ents, _user, db=db)
        if isinstance(send_result, _h_module.HandlerResult):
            # Handler cần OAuth → giữ flow state để retry sau khi user xác thực
            resp = send_result.text
            session["email_flow"] = new_state
            flow_active = True
            _oauth_extras = {"action_type": send_result.action_type,
                             "action_data": send_result.action_data}
        else:
            resp = send_result
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
    return payload, True


# ==========================================================================
# Routes
# ==========================================================================
def register_routes(app):

    # ----- JSON error handler cho /api/* -----
    # Bug history: routes /api/assistant/text khi gặp uncaught exception thì Flask
    # trả HTML 500 mặc định → frontend làm `await r.json()` → "Unexpected token '<'"
    # rất khó debug. Convert sang JSON 500 để frontend log/hiển thị lỗi đúng cách.
    @app.errorhandler(Exception)
    def _json_error_for_api(exc):
        from werkzeug.exceptions import HTTPException
        # Cho phép HTTPException (404, 403, 413, 429) đi qua handler riêng.
        if isinstance(exc, HTTPException):
            if request.path.startswith("/api/"):
                resp = jsonify({"error": exc.description or exc.name,
                                "code": exc.code})
                resp.status_code = exc.code or 500
                return resp
            return exc  # non-API → Flask default HTML.
        # Exception thật → log đầy đủ, response gọn (không leak stack).
        _log.exception("Unhandled exception trên %s %s", request.method, request.path)
        rid = getattr(g, "request_id", "")
        if request.path.startswith("/api/"):
            resp = jsonify({
                "error": "Lỗi server. Vui lòng thử lại hoặc liên hệ admin.",
                "request_id": rid,
            })
            resp.status_code = 500
            return resp
        # Non-API request lỗi → trang HTML đơn giản (không leak stack).
        return ("<!doctype html><meta charset='utf-8'>"
                f"<h2>Lỗi server (500)</h2><p>request_id={rid}</p>"), 500

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

    # Per-request CSP nonce + Request-ID correlation. Cả 2 sinh trong cùng hook.
    # Request-ID: dùng để correlate audit log + error trace ngang qua handler.
    #   Header `X-Request-ID` từ client (vd: load balancer/reverse proxy) được
    #   preserve nếu hợp lệ; nếu không có thì sinh mới. Trả lại qua response
    #   header để client thấy được ID đó.
    _RID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")

    @app.before_request
    def _gen_csp_nonce():
        g.csp_nonce = secrets.token_urlsafe(16)
        # Request-ID: trust upstream nếu valid format, else generate.
        incoming = request.headers.get("X-Request-ID", "")
        if incoming and _RID_RE.match(incoming):
            g.request_id = incoming
        else:
            g.request_id = secrets.token_urlsafe(12)

    @app.context_processor
    def _inject_csp_nonce():
        return {"csp_nonce": getattr(g, "csp_nonce", "")}

    def _build_csp(nonce: str) -> str:
        # script-src: chỉ nonce — đã refactor toàn bộ inline onclick handler
        # sang addEventListener (mỗi <script> block đều có nonce). Bỏ
        # 'unsafe-inline' để strict CSP, ngăn XSS chèn `<script>` không nonce
        # hoặc gắn onclick attribute mới.
        # style-src vẫn cần 'unsafe-inline' (Tailwind utility-class style + nhiều
        # `style="..."` attribute inline trong template).
        return (
            "default-src 'self'; "
            f"script-src 'self' 'nonce-{nonce}' https://cdn.tailwindcss.com; "
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
        # Echo lại Request-ID để client/log aggregator có thể trace.
        rid = getattr(g, "request_id", "")
        if rid:
            response.headers.setdefault("X-Request-ID", rid)
        # Dùng nonce per-request thay vì 'unsafe-inline'.
        nonce = getattr(g, "csp_nonce", "")
        response.headers.setdefault("Content-Security-Policy", _build_csp(nonce))
        # HSTS: chỉ set khi request đến qua HTTPS (request.is_secure). Tránh
        # gửi header trên kết nối HTTP — vô nghĩa và confuse browser.
        # max-age=180 ngày, includeSubDomains. KHÔNG có preload — app chỉ dùng
        # nội bộ/demo, không submit lên HSTS preload list.
        if request.is_secure:
            response.headers.setdefault(
                "Strict-Transport-Security",
                "max-age=15552000; includeSubDomains"
            )
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
        # Gate PII: render trang chi tiết với prefs đầy đủ CHỈ khi caller có
        # file session match user_id (đã xác thực qua voice/password).
        # Ngược lại render với prefs rỗng — user vẫn thấy danh tính + nhập
        # password để mở khoá, nhưng KHÔNG leak prefs ra public URL.
        if _get_authed_uid("any") != user_id:
            user = dict(user)
            user["preferences"] = {}
            user["_locked"] = True
        return render_template("user_detail.html", user=user)

    @app.route("/assistant")
    def assistant_page():
        users = app.config["db"].list_users()
        return render_template("assistant.html", users=users)

    # ----- API: Enrollment -----
    @app.route("/api/enroll", methods=["POST"])
    @max_size(20)
    @rate_limit(max_attempts=5, window_sec=600, scope="enroll")
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
        """Trả về public info của user.

        Mặc định KHÔNG include preferences (notes, email, balance, schedule —
        đều là PII). Caller cần preferences phải qua `/api/users/<id>/unlock`
        (đã có sẵn — yêu cầu password) hoặc đã đăng nhập file session.
        """
        db = app.config["db"]
        user = db.get_user(user_id)
        if not user:
            return jsonify({"error": "User không tồn tại"}), 404
        # Public payload: identification + auth-status only.
        # `revoke_pending` được trả về public — user cần thấy ngay (kể cả trước
        # khi unlock) để biết phải re-enroll giọng. Không leak PII.
        result = {
            "user_id":        user["user_id"],
            "name":           user["name"],
            "created_at":     user["created_at"],
            "has_password":   db.has_password(user_id),
            "revoke_pending": bool(user.get("revoke_pending")),
            "preferences":    {},
        }
        # Authed session với chính user này → trả preferences đầy đủ
        # (file session đặt sau khi unlock; same auth scope dùng cho assistant).
        if _get_authed_uid("any") == user_id:
            result["preferences"] = user["preferences"]
        return jsonify(result)

    def _decode_voice_samples_from_request(request):
        """Decode multipart `sample_*` audio blobs → list[np.ndarray] sau VAD.

        Dùng chung cho /api/enroll (lần đầu), /api/users/<id>/reenroll-voice
        (admin Revoke flow), /api/users/<id>/add-voice-samples (incremental).
        Trả về (audios, err_response). Nếu err_response không None, caller chỉ
        cần `return err_response` ngay.
        """
        audios = []
        for key in sorted(request.files.keys()):
            if not key.startswith("sample_"):
                continue
            blob = request.files[key].read()
            try:
                audio = audio_io.decode_browser_audio(blob)
            except Exception as e:
                return None, (jsonify({"error": f"Giải mã {key} thất bại: {e}"}), 400)
            audio, has_speech = audio_io.SileroVAD.trim(
                audio, return_speech_detected=True)
            _min_enroll = int(config.SAMPLE_RATE * config.MIN_AUDIO_SEC_ENROLL)
            if not has_speech or audio.size < _min_enroll:
                return None, (jsonify({
                    "error": f"{key} quá ngắn sau VAD ({audio.size/config.SAMPLE_RATE:.2f}s, "
                             f"cần ≥ {config.MIN_AUDIO_SEC_ENROLL}s). "
                             "Hãy đảm bảo nói rõ ràng đủ thời lượng."
                }), 400)
            audios.append(audio)
        return audios, None

    @app.route("/api/users/<user_id>/reenroll-voice", methods=["POST"])
    @max_size(20)
    @rate_limit(max_attempts=3, window_sec=300,
                scope="user-reenroll-voice", key_user_field="user_id")
    def api_reenroll_voice(user_id):
        """Đăng ký lại giọng nói toàn bộ (REPLACE centroid).

        Dùng khi admin đã Revoke (revoke_pending=True) HOẶC user muốn refresh
        toàn bộ template giọng (vd: bị cảm giọng đổi rõ).

        Body multipart: password (form field) + sample_0..N (audio blobs).
        Yêu cầu password current để chống attacker. Sau khi pass:
        - Encode N audio mới → centroid mới.
        - REPLACE embedding trong DB (KHÔNG average với old — vì old có thể đã xoá).
        - Clear cờ revoke_pending.
        - Replace WAV samples trên disk.
        """
        db = app.config["db"]
        spk_mgr = app.config["spk_mgr"]
        user = db.get_user(user_id)
        if not user:
            return jsonify({"error": "User không tồn tại"}), 404
        password = request.form.get("password", "").strip()
        ip = _rl_client_ip()
        ua = request.headers.get("User-Agent", "")
        if not db.check_password(user_id, password):
            _audit("auth.password_check", user_id=user_id, scope="reenroll-voice",
                   outcome="fail", ip=ip, ua=ua)
            return jsonify({"error": "Sai mật khẩu"}), 403

        audios, err = _decode_voice_samples_from_request(request)
        if err is not None:
            return err
        if len(audios) < 2:
            return jsonify({
                "error": f"Cần ít nhất 2 mẫu (gửi được {len(audios)}). "
                         f"Khuyến nghị {config.ENROLL_NUM_SAMPLES} mẫu."
            }), 400

        # Encode + REPLACE
        try:
            from core import audio_io as _aio
            prepped = [_aio.preprocess_audio(a) for a in audios]
            centroid = spk_mgr.encoder.encode_centroid(prepped)
            db.upsert_embedding(user_id, centroid,
                                backend_id=spk_mgr.encoder.backend_id)
        except Exception as e:
            return jsonify({"error": f"Re-enroll thất bại: {e}"}), 500

        # Clear revoke flag — user đã chứng minh quyền sở hữu giọng nói.
        db.set_revoke_pending(user_id, False)
        spk_mgr.invalidate_cache()

        # Replace WAV files trên disk.
        import shutil, tempfile
        sample_dir = config.ENROLL_AUDIO_DIR / user_id
        # Wipe old + atomic move new in.
        with tempfile.TemporaryDirectory(prefix="reenroll_") as _tmp_str:
            tmp_dir = Path(_tmp_str)
            for idx, a in enumerate(audios, start=1):
                audio_io.save_wav(a, tmp_dir / f"sample_{idx}.wav")
            if sample_dir.exists():
                shutil.rmtree(sample_dir)
            sample_dir.mkdir(parents=True, exist_ok=True)
            for wav in tmp_dir.iterdir():
                shutil.move(str(wav), sample_dir / wav.name)

        _audit("speaker.reenroll", user_id=user_id, actor="self",
               outcome="success", n_samples=len(audios),
               cleared_revoke=True, ip=ip, ua=ua)
        return jsonify({
            "ok": True,
            "user_id": user_id,
            "n_samples": len(audios),
            "revoke_cleared": True,
        })

    @app.route("/api/users/<user_id>/add-voice-samples", methods=["POST"])
    @max_size(20)
    @rate_limit(max_attempts=3, window_sec=300,
                scope="user-add-voice", key_user_field="user_id")
    def api_add_voice_samples(user_id):
        """Bổ sung mẫu giọng nói cho centroid (KHÔNG replace).

        Khác `reenroll-voice`: endpoint này AVERAGE mẫu mới với centroid cũ
        (weight theo alpha). Dùng khi user muốn thêm robustness cho centroid
        mà không xoá template cũ — ví dụ thu thêm mẫu trong môi trường ồn khác.

        Body multipart: password + sample_0..N. Cần ít nhất 1 mẫu.
        - KHÔNG cho phép khi revoke_pending (admin đã thu hồi → phải re-enroll).
        - Mỗi mẫu apply incremental update với alpha = 1/(n_existing+1).
        """
        db = app.config["db"]
        spk_mgr = app.config["spk_mgr"]
        user = db.get_user(user_id)
        if not user:
            return jsonify({"error": "User không tồn tại"}), 404
        if user.get("revoke_pending"):
            return jsonify({"error": "Tài khoản đang chờ Re-enroll. Gọi "
                                     "/api/users/<id>/reenroll-voice thay vì add-voice."}), 409
        password = request.form.get("password", "").strip()
        ip = _rl_client_ip()
        ua = request.headers.get("User-Agent", "")
        if not db.check_password(user_id, password):
            _audit("auth.password_check", user_id=user_id, scope="add-voice",
                   outcome="fail", ip=ip, ua=ua)
            return jsonify({"error": "Sai mật khẩu"}), 403

        audios, err = _decode_voice_samples_from_request(request)
        if err is not None:
            return err
        if not audios:
            return jsonify({"error": "Cần ít nhất 1 mẫu giọng"}), 400

        # Update incremental — mỗi audio kéo centroid theo alpha nhỏ.
        # alpha = 1/(N_existing_estimate + 1) — assume centroid đại diện cho
        # ~ ENROLL_NUM_SAMPLES mẫu gốc nên giảm dần ảnh hưởng mỗi audio thêm.
        alpha = 1.0 / (config.ENROLL_NUM_SAMPLES + 1)
        updated_count = 0
        for a in audios:
            try:
                ok, _ = spk_mgr.incremental_update_centroid(a, user_id, alpha=alpha)
                if ok:
                    updated_count += 1
            except Exception:
                _log.exception("incremental update failed cho %s", user_id)
        # Persist WAV samples mới (append) — tên file unique theo timestamp.
        sample_dir = config.ENROLL_AUDIO_DIR / user_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        for idx, a in enumerate(audios, start=1):
            audio_io.save_wav(a, sample_dir / f"sample_added_{ts}_{idx}.wav")

        _audit("speaker.add_samples", user_id=user_id, actor="self",
               outcome="success", n_samples=updated_count, alpha=alpha,
               ip=ip, ua=ua)
        return jsonify({
            "ok": True,
            "user_id": user_id,
            "n_samples_applied": updated_count,
            "alpha_per_sample": round(alpha, 4),
        })

    @app.route("/api/users/<user_id>/due-reminders", methods=["GET", "POST"])
    @rate_limit(max_attempts=60, window_sec=60,
                scope="due-reminders", key_user_field="user_id")
    def api_due_reminders(user_id):
        """Trả về reminder đã đến hạn và chưa fire.

        - GET: chỉ read (peek). Frontend poll mỗi N giây để hiện notification.
        - POST với body `{ack_ids: [...]}`: mark fired để không pop lại.
          Gate auth: phải có file session match user_id (chống enumeration).
        """
        from core.reminders import due_reminders, mark_fired
        db = app.config["db"]
        user = db.get_user(user_id)
        if not user:
            return jsonify({"error": "User không tồn tại"}), 404
        # GET không cần auth (chỉ ID-based polling sau khi user đã unlock ở UI).
        # POST cần file session match → chống attacker mark-fired reminder của
        # user khác.
        if request.method == "POST":
            if _get_authed_uid("any") != user_id:
                return jsonify({"error": "Chưa xác thực"}), 403
            ids = (_json_body() or {}).get("ack_ids", []) or []
            prefs = dict(user["preferences"])
            prefs["reminders"] = mark_fired(prefs.get("reminders", []), ids)
            db.update_preferences(user_id, prefs)
            return jsonify({"ok": True, "acked": len(ids)})

        reminders = user["preferences"].get("reminders") or []
        due = due_reminders(reminders)
        # Public payload: chỉ id + content + when_iso — đủ cho banner UI.
        return jsonify({
            "due": [
                {"id": r["id"], "content": r["content"],
                 "when_iso": r.get("when_iso")}
                for r in due
            ],
            "total_pending": sum(1 for r in reminders if not r.get("fired")),
        })

    @app.route("/api/users/<user_id>/update", methods=["POST"])
    @rate_limit(max_attempts=5, window_sec=60,
                scope="user-update-prefs", key_user_field="user_id")
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
            _audit("auth.password_check", user_id=user_id, scope="update-prefs",
                   outcome="fail", ip=_rl_client_ip(),
                   ua=request.headers.get("User-Agent", ""))
            return jsonify({"error": "Sai mật khẩu"}), 403

        db.update_preferences(user_id, preferences)
        _audit("auth.password_check", user_id=user_id, scope="update-prefs",
               outcome="success", ip=_rl_client_ip(),
               ua=request.headers.get("User-Agent", ""),
               pref_keys=sorted(preferences.keys()))
        return jsonify({"ok": True})

    @app.route("/api/users/<user_id>/update-info", methods=["POST"])
    @rate_limit(max_attempts=5, window_sec=60,
                scope="user-update-info", key_user_field="user_id")
    def api_update_user_info(user_id):
        """User self-service update (name + optional password).

        Body: {password, name?, new_password?}
        - `password`: current password (verify).
        - `new_password` (tuỳ chọn): nếu set → đổi password. Validate length 8-128,
          khác password cũ. Sau khi đổi → invalidate file session (user phải login lại).
        """
        db = app.config["db"]
        if not db.get_user(user_id):
            return jsonify({"error": "User không tồn tại"}), 404
        data = _json_body()
        password = data.get("password", "")
        ip = _rl_client_ip()
        ua = request.headers.get("User-Agent", "")
        if not db.check_password(user_id, password):
            _audit("auth.password_check", user_id=user_id, scope="update-info",
                   outcome="fail", ip=ip, ua=ua)
            return jsonify({"error": "Sai mật khẩu"}), 403

        if "name" in data:
            db.update_user_name(user_id, data["name"])
        new_password = data.get("new_password", "")
        if new_password:
            # Validate new password — big-tech standard: length 8-128 + khác cũ.
            if len(new_password) < 8:
                return jsonify({"error": "Mật khẩu mới phải có ít nhất 8 ký tự"}), 400
            if len(new_password) > 128:
                return jsonify({"error": "Mật khẩu mới quá dài (max 128)"}), 400
            if new_password == password:
                return jsonify({"error": "Mật khẩu mới phải khác mật khẩu cũ"}), 400
            db.update_password(user_id, new_password)
            # Invalidate file session — buộc login lại với password mới.
            if session.get("file_uid") == user_id:
                _clear_file_session()
            _audit("auth.password_change", user_id=user_id, actor="self",
                   outcome="success", ip=ip, ua=ua)
        return jsonify({"ok": True})

    @app.route("/api/users/<user_id>/change-password", methods=["POST"])
    @rate_limit(max_attempts=5, window_sec=60,
                scope="user-change-password", key_user_field="user_id")
    def api_change_password(user_id):
        """User self-service password change — explicit endpoint riêng cho UI.

        Body: {old_password, new_password, new_password_confirm}
        - old_password phải khớp current.
        - new_password = new_password_confirm.
        - new ≠ old, length 8-128.
        - Audit `auth.password_change`, invalidate file session.
        """
        db = app.config["db"]
        if not db.get_user(user_id):
            return jsonify({"error": "User không tồn tại"}), 404
        data = _json_body()
        old_pw = data.get("old_password", "")
        new_pw = str(data.get("new_password", ""))
        new_pw_confirm = str(data.get("new_password_confirm", ""))
        ip = _rl_client_ip()
        ua = request.headers.get("User-Agent", "")

        if not db.check_password(user_id, old_pw):
            _audit("auth.password_change", user_id=user_id, actor="self",
                   outcome="fail_old_password", ip=ip, ua=ua)
            return jsonify({"error": "Mật khẩu cũ không đúng"}), 403
        if not new_pw or new_pw != new_pw_confirm:
            return jsonify({"error": "Xác nhận mật khẩu mới không khớp"}), 400
        if len(new_pw) < 8:
            return jsonify({"error": "Mật khẩu mới phải có ít nhất 8 ký tự"}), 400
        if len(new_pw) > 128:
            return jsonify({"error": "Mật khẩu mới quá dài (max 128)"}), 400
        if new_pw == old_pw:
            return jsonify({"error": "Mật khẩu mới phải khác mật khẩu cũ"}), 400

        db.update_password(user_id, new_pw)
        if session.get("file_uid") == user_id:
            _clear_file_session()
        _audit("auth.password_change", user_id=user_id, actor="self",
               outcome="success", ip=ip, ua=ua)
        return jsonify({"ok": True})

    @app.route("/api/users/<user_id>/export", methods=["POST"])
    @rate_limit(max_attempts=3, window_sec=300,
                scope="user-export", key_user_field="user_id")
    def api_export_user(user_id):
        """Xuất toàn bộ dữ liệu của user dạng ZIP (GDPR Art.20 portability).

        Body JSON: {password}. Required vì preferences chứa PII (email, notes,
        balance...). Sau khi xác thực, trả về ZIP chứa:
          - user.json        : metadata + preferences
          - enroll_audio/*   : tất cả WAV mẫu enroll
          - user_files/*     : file user đã upload
          - oauth.json       : thông tin OAuth (gmail address, expiry) — KHÔNG
                               include access/refresh token (security: token là
                               secret server-side, không phải user data).

        Trả ZIP qua send_file streaming. Lưu trữ tạm trong tempfile, xoá sau khi gửi.
        """
        db = app.config["db"]
        user = db.get_user(user_id)
        ip = _rl_client_ip()
        ua = request.headers.get("User-Agent", "")
        if not user:
            return jsonify({"error": "User không tồn tại"}), 404
        data = _json_body()
        password = data.get("password", "")
        if not db.check_password(user_id, password):
            _audit("auth.password_check", user_id=user_id, scope="export",
                   outcome="fail", ip=ip, ua=ua)
            return jsonify({"error": "Sai mật khẩu"}), 403

        import io as _io
        import zipfile
        import tempfile

        # Build ZIP in-memory (user data size khá nhỏ: vài MB WAV + ít byte JSON).
        # Nếu user có nhiều file upload → có thể >50MB, nhưng đã có MAX_CONTENT_LENGTH
        # cap. Send via send_file với BytesIO.
        buf = _io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            # 1. Metadata + preferences
            meta = {
                "user_id":    user["user_id"],
                "name":       user["name"],
                "created_at": user["created_at"],
                "preferences": user["preferences"],
                "exported_at": datetime.now().isoformat() + "Z",
                "exported_by_ip": ip,
            }
            zf.writestr("user.json", json.dumps(meta, ensure_ascii=False, indent=2))

            # 2. WAV enroll samples
            enroll_dir = _resolve_child_path(config.ENROLL_AUDIO_DIR, user_id)
            if enroll_dir and enroll_dir.is_dir():
                for wav in enroll_dir.iterdir():
                    if wav.is_file() and wav.suffix.lower() == ".wav":
                        zf.write(wav, f"enroll_audio/{wav.name}")

            # 3. User file uploads
            files_dir = _resolve_child_path(config.USER_FILES_DIR, user_id)
            if files_dir and files_dir.is_dir():
                for f in files_dir.iterdir():
                    if f.is_file():
                        zf.write(f, f"user_files/{f.name}")

            # 4. OAuth metadata (KHÔNG include token)
            tok = db.get_oauth_token(user_id)
            if tok:
                oauth_meta = {
                    "gmail_address": tok.get("gmail_address", ""),
                    "expiry":        tok.get("expiry", 0),
                    "note": ("access_token + refresh_token không export — "
                             "đây là secret server-side, không phải user data."),
                }
                zf.writestr("oauth.json", json.dumps(oauth_meta, ensure_ascii=False, indent=2))

        buf.seek(0)
        _audit("data.export_user", user_id=user_id, outcome="success",
               size_bytes=buf.getbuffer().nbytes, ip=ip, ua=ua)
        return send_file(
            buf,
            mimetype="application/zip",
            as_attachment=True,
            download_name=f"secva-export-{user_id}.zip",
        )

    @app.route("/api/users/<user_id>/delete", methods=["POST"])
    @rate_limit(max_attempts=3, window_sec=300,
                scope="user-delete", key_user_field="user_id")
    def api_delete_user(user_id):
        db = app.config["db"]
        ip = _rl_client_ip()
        ua = request.headers.get("User-Agent", "")
        if not db.get_user(user_id):
            return jsonify({"error": "User không tồn tại"}), 404
        data = _json_body()
        password = data.get("password", "")
        if not db.check_password(user_id, password):
            _audit("auth.password_check", user_id=user_id, scope="delete-user",
                   outcome="fail", ip=ip, ua=ua)
            return jsonify({"error": "Sai mật khẩu"}), 403

        # 1. Xoá DB rows (users + embeddings + oauth_tokens).
        db.delete_user(user_id)
        # 2. Invalidate cache SpeakerManager (đẩy user khỏi identify/verify pool).
        app.config["spk_mgr"]._cache = None
        # 3. Xoá on-disk biometric data + user files (right to erasure / GDPR Art.17).
        #    _resolve_child_path đảm bảo không escape ra ngoài thư mục cha.
        import shutil
        dirs_removed = []
        for base in (config.ENROLL_AUDIO_DIR, config.USER_FILES_DIR):
            target = _resolve_child_path(base, user_id)
            if target and target.exists() and target.is_dir():
                try:
                    shutil.rmtree(target)
                    dirs_removed.append(str(target.relative_to(config.DATA_DIR)))
                except OSError as e:
                    _log.warning("Không xoá được %s: %s", target, e)
        # 4. Invalidate session nếu user vừa xoá đang đang đăng nhập trên browser này.
        if session.get("file_uid") == user_id:
            _clear_file_session()
        _audit("data.delete_user", user_id=user_id, outcome="success",
               dirs_removed=dirs_removed, ip=ip, ua=ua)
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

        # Email flow continuation (dùng chung với text route — xem
        # _continue_email_flow_if_active).
        flow_payload, handled = _continue_email_flow_if_active(transcript, db, nlu)
        if handled:
            _log_assistant_turn(flow_payload, source="web_voice_email_flow",
                                auth_method="voice")
            return jsonify(flow_payload)

        transcript, nlu_result = parse_with_correction(transcript, nlu)
        result = router.handle_turn(audio, transcript, nlu_result,
                                    extra_context={"db": db})

        # Challenge-response: lưu state vào session để turn thứ 2 dispatch handler.
        if result.action_type == "challenge_required" and result.action_data:
            cd = result.action_data
            ch_store = session.get("challenges", {})
            # Cleanup expired tokens
            now = time.time()
            ch_store = {k: v for k, v in ch_store.items()
                        if v.get("expires_at", 0) > now}
            ch_store[cd["token"]] = {
                **cd,
                "expires_at": now + 90,  # 90s TTL
                "used":       False,
            }
            session["challenges"] = ch_store

            payload = asdict(result)
            payload["tts_url"] = f"/api/tts?text={_quote(result.response)}"
            payload["flow_active"] = False
            # Frontend đọc payload.action_data.phrase để show user + record audio_2.
            _log_assistant_turn(payload, source="web_voice_challenge_issued",
                                auth_method="voice")
            return jsonify(payload)

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
            _log_assistant_turn(payload, source="web_voice",
                                auth_method="voice")
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

        _log_assistant_turn(payload, source="web_voice", auth_method="voice")
        return jsonify(payload)

    # ----- API: Challenge-response (turn 2 sau khi SID+SV pass) -----
    @app.route("/api/assistant/challenge-response", methods=["POST"])
    @max_size(5)
    @rate_limit(max_attempts=10, window_sec=60, scope="challenge-response")
    def api_challenge_response():
        """Hoàn tất 1 challenge-response cho IMPORTANT intent.

        Body multipart:
            - audio: file blob (WebM/wav) — user đọc lại phrase server đã sinh.
            - token: form field — token từ payload `action_data.token` của turn 1.

        Server check token còn hợp lệ → ASR audio → so phrase → re-SV →
        dispatch handler đã pending.
        """
        token = (request.form.get("token") or "").strip()
        if not token:
            return jsonify({"error": "Thiếu token"}), 400
        if "audio" not in request.files:
            return jsonify({"error": "Thiếu file 'audio'"}), 400

        ch_store = session.get("challenges", {})
        state = ch_store.get(token)
        if not state:
            return jsonify({"error": "Token không hợp lệ hoặc đã hết hạn"}), 403
        if state.get("used"):
            return jsonify({"error": "Token đã được dùng"}), 403
        if time.time() > state.get("expires_at", 0):
            ch_store.pop(token, None)
            session["challenges"] = ch_store
            return jsonify({"error": "Token đã hết hạn — vui lòng thử lại"}), 403

        # Decode audio thứ 2
        blob = request.files["audio"].read()
        try:
            audio = audio_io.decode_browser_audio(blob)
        except Exception as e:
            return jsonify({"error": f"Giải mã audio thất bại: {e}"}), 400
        audio, has_speech = audio_io.SileroVAD.trim(audio, return_speech_detected=True)
        _min_turn = int(config.SAMPLE_RATE * config.MIN_AUDIO_SEC_TURN)
        if not has_speech or audio.size < _min_turn:
            return jsonify({"error": "Audio quá ngắn hoặc không có giọng"}), 400

        # ASR + complete
        asr    = app.config["asr"]
        router = app.config["router"]
        db     = app.config["db"]
        transcript = asr.transcribe(audio)
        result = router.complete_challenge(audio, transcript, state,
                                           extra_context={"db": db})

        # Mark token used (one-shot), cleanup
        state["used"] = True
        ch_store.pop(token, None)
        session["challenges"] = ch_store

        payload = asdict(result)
        payload["tts_url"] = f"/api/tts?text={_quote(result.response)}"
        payload["flow_active"] = False

        # Attach action_data cho frontend nếu pass + handler có signal
        if not result.blocked:
            _uid = result.identified_user_id
            _user = db.get_user(_uid) if _uid else None
            _prefs = _user["preferences"] if _user else {}
            _attach_action_data(
                payload, result.intent, result.blocked,
                _uid, result.identified_user_name,
                result.entities, _prefs, auth_method="voice",
            )
        _log_assistant_turn(payload, source="web_challenge_response",
                            auth_method="voice")
        return jsonify(payload)

    # ----- API: Text mode (không cần micro) -----
    @app.route("/api/assistant/text", methods=["POST"])
    @max_size(1)
    @rate_limit(max_attempts=30, window_sec=60, scope="assistant-text")
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

        # Email flow continuation (dùng chung với voice route — xem
        # _continue_email_flow_if_active).
        flow_payload, handled = _continue_email_flow_if_active(text, db, nlu)
        if handled:
            _log_assistant_turn(flow_payload, source="web_text_email_flow",
                                auth_method="password")
            return jsonify(flow_payload)

        text, nlu_result = parse_with_correction(text, nlu)
        intent   = nlu_result["intent"]
        entities = nlu_result["entities"]
        spec     = INTENTS.get(intent, INTENTS["unknown"])
        level    = spec["level"]

        user = db.get_user(user_id) if user_id else None
        uid  = user["user_id"] if user else None
        name = user["name"] if user else "Khách"

        sv_required = (level == AuthLevel.IMPORTANT)
        password_allowed_for_intent = (
            config.ALLOW_PASSWORD_FOR_IMPORTANT or intent == "open_files"
        )
        sv_passed   = None
        blocked     = False
        response    = ""

        if sv_required:
            # Guard 0: nếu admin đã Revoke giọng → block IMPORTANT cho tới khi
            # user re-enroll xong. KHÔNG cho password bypass — đây là design
            # intent của Revoke flow.
            if user and user.get("revoke_pending"):
                blocked  = True
                response = (
                    f"Tài khoản {name} đang trong trạng thái chờ đăng ký lại giọng nói "
                    f"(admin đã thu hồi). Vui lòng vào trang user → 'Đăng ký lại giọng nói' "
                    f"để mở khóa tác vụ quan trọng."
                )
            # Text-mode KHÔNG có audio để verify giọng. Mặc định block IMPORTANT
            # — Speaker Verification là biometric gate, password là "know factor"
            # không tương đương. Cho phép password fallback chỉ khi
            # ALLOW_PASSWORD_FOR_IMPORTANT=true (demo mode hoặc mic broken).
            elif not password_allowed_for_intent:
                blocked  = True
                response = ("Tác vụ này yêu cầu xác thực bằng giọng nói (Speaker Verification). "
                            "Hãy chuyển sang chế độ Voice và nói lại yêu cầu.")
            elif not uid:
                blocked  = True
                response = ("Tác vụ này yêu cầu xác thực. "
                            "Vui lòng chọn người dùng và nhập mật khẩu.")
            elif not password:
                # User chưa nhập password (turn đầu tiên với IMPORTANT intent)
                # → prompt nhập password, KHÔNG báo "sai mật khẩu".
                blocked  = True
                response = (f"Tác vụ này cần xác thực. "
                            f"Vui lòng nhập mật khẩu của {name} rồi gửi lại yêu cầu.")
            elif not db.check_password(uid, password):
                blocked  = True
                response = "Sai mật khẩu. Vui lòng kiểm tra và thử lại."
            else:
                # Pass qua password fallback — KHÔNG phải biometric SV.
                # auth_method="password" set ở _attach_action_data để frontend
                # hiển thị warning rõ.
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
            _log_assistant_turn(payload, source="web_text",
                                auth_method="password")
            return jsonify(payload)

        if not blocked:
            handler  = _h.HANDLERS.get(intent, _h.handle_unknown)
            raw_response = handler(entities, user, db=db)
            action_type = None
            action_data = None
            if isinstance(raw_response, _h_module.HandlerResult):
                response = raw_response.text
                action_type = raw_response.action_type
                action_data = raw_response.action_data
            else:
                response = raw_response
        else:
            action_type = None
            action_data = None

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
        if action_type:
            payload["action_type"] = action_type
        if action_data:
            payload["action_data"] = action_data

        _prefs = user["preferences"] if user else {}
        _attach_action_data(
            payload, intent, blocked,
            uid, name, entities, _prefs, auth_method="password",
        )

        _log_assistant_turn(payload, source="web_text", auth_method="password")
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
        # Resolve language: param `lang` ưu tiên, fallback theo authed user
        # preferences.language, cuối cùng default vi.
        lang = (request.args.get("lang", "") or "").strip()
        if not lang:
            _uid = _get_authed_uid("any")
            if _uid:
                _user = app.config["db"].get_user(_uid)
                if _user:
                    lang = (_user["preferences"].get("language") or "").strip()
        try:
            mp3 = app.config["tts"].synthesize_to_mp3_bytes(text, lang=lang or None)
        except Exception:
            # gTTS lỗi (mất mạng, rate limit...). Trả về 204 No Content thay vì
            # bytes giả "\xff\xe3..." (không phải MP3 hợp lệ, gây lỗi audio player).
            return ("", 204)
        if not mp3:
            # Cả gTTS và pyttsx3 fallback đều fail → 204 cho client xử lý gracefully.
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
              "url": f"/api/music/user-tracks/{uid}/stream/{_quote(f.name)}"}
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
        fpath = _resolve_child_path(base, filename)
        if fpath is None:
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
        fpath = _resolve_child_path(base, filename)
        if fpath is None or not fpath.exists():
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
    @rate_limit(max_attempts=5, window_sec=60, scope="files-verify-password")
    def api_files_verify_password():
        data = _json_body()
        user_id = data.get("user_id", "")
        password = data.get("password", "")
        db = app.config["db"]
        user = db.get_user(user_id)
        ip = _rl_client_ip()
        ua = request.headers.get("User-Agent", "")
        if not user:
            return jsonify({"error": "User không tồn tại"}), 404
        if not db.check_password(user_id, password):
            _audit("auth.password_check", user_id=user_id, scope="files",
                   outcome="fail", ip=ip, ua=ua)
            return jsonify({"error": "Sai mật khẩu"}), 403
        _set_file_session(user_id, user.get("name", user_id), "password")
        _audit("auth.password_check", user_id=user_id, scope="files",
               outcome="success", ip=ip, ua=ua)
        return jsonify({"ok": True, "user_name": user.get("name", user_id)})

    @app.route("/api/users/<user_id>/unlock", methods=["POST"])
    @rate_limit(max_attempts=5, window_sec=60,
                scope="user-unlock", key_user_field="user_id")
    def api_user_unlock(user_id):
        """Xác thực mật khẩu, set file session, trả về user info.
        Dùng thay cho update-info khi chỉ cần mở khóa modal."""
        db = app.config["db"]
        user = db.get_user(user_id)
        ip = _rl_client_ip()
        ua = request.headers.get("User-Agent", "")
        if not user:
            return jsonify({"error": "User không tồn tại"}), 404
        data = _json_body()
        password = data.get("password", "")
        if not db.check_password(user_id, password):
            _audit("auth.password_check", user_id=user_id, scope="unlock",
                   outcome="fail", ip=ip, ua=ua)
            return jsonify({"error": "Sai mật khẩu"}), 403
        _set_file_session(user_id, user["name"], "password")
        _audit("auth.password_check", user_id=user_id, scope="unlock",
               outcome="success", ip=ip, ua=ua)
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
        fpath = _resolve_child_path(base, filename)
        if fpath is None:
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
        fpath = _resolve_child_path(base, filename)
        if fpath is None:
            return jsonify({"error": "Không hợp lệ"}), 400
        if not fpath.exists():
            return jsonify({"error": "File không tồn tại"}), 404
        fpath.unlink()
        return jsonify({"ok": True})

    # ----- Admin -----
    @app.route("/api/admin/verify", methods=["POST"])
    @max_size(0.01)
    @rate_limit(max_attempts=5, window_sec=300, scope="admin-verify")
    def api_admin_verify():
        data = _json_body()
        supplied = data.get("password", "")
        if (not config.ADMIN_PASS or
                not hmac.compare_digest(str(supplied), config.ADMIN_PASS)):
            return jsonify({"error": "Sai mật khẩu quản trị"}), 403
        db = app.config["db"]
        users = db.list_users()
        reset_requests = db.list_password_reset_requests(status="pending")
        return jsonify({
            "ok": True,
            "count": len(users),
            "reset_requests": reset_requests,
            "users": [
                {
                    "user_id":    u["user_id"],
                    "name":       u["name"],
                    "created_at": u["created_at"][:10],
                }
                for u in users
            ],
        })

    def _check_admin_pass(supplied: str) -> bool:
        return bool(config.ADMIN_PASS) and hmac.compare_digest(
            str(supplied), config.ADMIN_PASS)

    @app.route("/api/admin/password-reset-requests", methods=["POST"])
    @max_size(0.02)
    @rate_limit(max_attempts=10, window_sec=60, scope="admin-reset-requests")
    def api_admin_password_reset_requests():
        data = _json_body()
        if not _check_admin_pass(data.get("admin_password", "")):
            return jsonify({"error": "Sai mật khẩu quản trị"}), 403
        return jsonify({
            "ok": True,
            "requests": app.config["db"].list_password_reset_requests(status="pending"),
        })

    @app.route("/api/admin/password-reset-requests/<int:request_id>/resolve",
               methods=["POST"])
    @max_size(0.02)
    @rate_limit(max_attempts=10, window_sec=60, scope="admin-reset-request-resolve")
    def api_admin_password_reset_request_resolve(request_id):
        data = _json_body()
        if not _check_admin_pass(data.get("admin_password", "")):
            return jsonify({"error": "Sai mật khẩu quản trị"}), 403
        status = str(data.get("status", "resolved")).strip() or "resolved"
        if status not in ("resolved", "rejected"):
            return jsonify({"error": "Status không hợp lệ"}), 400
        app.config["db"].update_password_reset_request_status(request_id, status)
        _audit("admin.password_reset_request", request_id=request_id,
               outcome=status, ip=_rl_client_ip(),
               ua=request.headers.get("User-Agent", ""))
        return jsonify({"ok": True})

    def _reset_code_hash(code: str) -> str:
        return hashlib.sha256(str(code).encode("utf-8")).hexdigest()

    def _send_password_reset_code(db, user_id: str, token_data: dict,
                                  gmail_addr: str, code: str):
        import time as _t
        from core.gmail_api import send_email
        from core.oauth import refresh_access_token
        if _t.time() > token_data.get("expiry", 0):
            updated = refresh_access_token(token_data["refresh_token"])
            token_data.update(updated)
            db.save_oauth_token(user_id, token_data)
        send_email(
            token_data["access_token"],
            gmail_addr,
            "Mã đặt lại mật khẩu Secure Virtual Assistant",
            "Mã đặt lại mật khẩu của bạn là: "
            f"{code}\n\nMã có hiệu lực trong 10 phút. "
            "Nếu bạn không yêu cầu thao tác này, hãy bỏ qua email.",
            from_name="Secure Virtual Assistant",
        )

    @app.route("/api/users/<user_id>/forgot-password/request", methods=["POST"])
    @max_size(0.01)
    @rate_limit(max_attempts=3, window_sec=300,
                scope="forgot-password", key_user_field="user_id")
    def api_forgot_password_request(user_id):
        """Request reset password.

        - Nếu user đã link Gmail OAuth: gửi mã 6 số về email đó.
        - Nếu chưa link Gmail: tạo request pending cho admin xử lý.
        """
        db = app.config["db"]
        user = db.get_user(user_id)
        # Không leak user enumeration quá nhiều; nhưng UI nội bộ cần biết lỗi rõ.
        if not user:
            return jsonify({"error": "User không tồn tại"}), 404

        token_data = db.get_oauth_token(user_id)
        gmail_addr = (token_data or {}).get("gmail_address", "")
        ip = _rl_client_ip()
        ua = request.headers.get("User-Agent", "")
        if token_data and gmail_addr:
            code = f"{secrets.randbelow(1_000_000):06d}"
            db.save_password_reset_code(
                user_id, _reset_code_hash(code),
                time.time() + 600, time.time(),
            )
            try:
                _send_password_reset_code(db, user_id, token_data, gmail_addr, code)
            except Exception as exc:
                _audit("auth.password_reset_request", user_id=user_id,
                       outcome="email_send_failed", ip=ip, ua=ua,
                       error=str(exc)[:120])
                return jsonify({
                    "error": "Không gửi được email reset. Vui lòng thử lại hoặc gửi yêu cầu cho admin.",
                    "mode": "email_failed",
                }), 503
            _audit("auth.password_reset_request", user_id=user_id,
                   outcome="email_code_sent", ip=ip, ua=ua)
            return jsonify({"ok": True, "mode": "email_code",
                            "email_masked": _mask_email(gmail_addr),
                            "expires_in_sec": 600})

        req_id = db.create_password_reset_request(
            user_id, note="User requested password reset without linked Gmail")
        _audit("auth.password_reset_request", user_id=user_id,
               outcome="admin_pending", request_id=req_id, ip=ip, ua=ua)
        return jsonify({"ok": True, "mode": "admin_request",
                        "request_id": req_id,
                        "message": "Chưa có Gmail liên kết. Đã gửi yêu cầu cho quản trị viên."})

    @app.route("/api/users/<user_id>/forgot-password/confirm", methods=["POST"])
    @max_size(0.02)
    @rate_limit(max_attempts=5, window_sec=600,
                scope="forgot-password-confirm", key_user_field="user_id")
    def api_forgot_password_confirm(user_id):
        db = app.config["db"]
        user = db.get_user(user_id)
        if not user:
            return jsonify({"error": "User không tồn tại"}), 404
        data = _json_body()
        code = str(data.get("code", "")).strip()
        new_password = str(data.get("new_password", ""))
        confirm = str(data.get("new_password_confirm", ""))
        rec = db.get_password_reset_code(user_id)
        if not rec:
            return jsonify({"error": "Chưa có mã reset hoặc mã đã hết hạn"}), 400
        if time.time() > rec["expires_at"]:
            db.delete_password_reset_code(user_id)
            return jsonify({"error": "Mã reset đã hết hạn"}), 400
        if rec["attempts"] >= 5:
            db.delete_password_reset_code(user_id)
            return jsonify({"error": "Nhập sai quá nhiều lần. Vui lòng yêu cầu mã mới."}), 429
        if not hmac.compare_digest(_reset_code_hash(code), rec["code_hash"]):
            db.increment_password_reset_attempts(user_id)
            return jsonify({"error": "Mã reset không đúng"}), 403
        if len(new_password) < 8:
            return jsonify({"error": "Mật khẩu mới phải có ít nhất 8 ký tự"}), 400
        if len(new_password) > 128:
            return jsonify({"error": "Mật khẩu mới quá dài"}), 400
        if new_password != confirm:
            return jsonify({"error": "Xác nhận mật khẩu không khớp"}), 400
        db.update_password(user_id, new_password)
        db.delete_password_reset_code(user_id)
        if session.get("file_uid") == user_id:
            _clear_file_session()
        _audit("auth.password_reset", user_id=user_id, actor="self_email",
               outcome="success", ip=_rl_client_ip(),
               ua=request.headers.get("User-Agent", ""))
        return jsonify({"ok": True})

    @app.route("/api/admin/users/<user_id>/info-masked", methods=["POST"])
    @max_size(0.01)
    @rate_limit(max_attempts=10, window_sec=60,
                scope="admin-info-masked", key_user_field="user_id")
    def api_admin_info_masked(user_id):
        """Admin xem thông tin user với PII đã mask.

        Big-tech style: admin KHÔNG được xem password/email/preferences full.
        Chỉ thấy đủ để confirm danh tính + decide có revoke không. Không
        có endpoint nào trả về password hash cho admin (always-hashed-never-recoverable).

        Body: {admin_password}
        Returns: {user_id, name_masked, email_masked, voice_status, has_password,
                  revoke_pending, created_at}
        """
        data = _json_body()
        ip = _rl_client_ip()
        ua = request.headers.get("User-Agent", "")
        if not _check_admin_pass(data.get("admin_password", "")):
            _audit("admin.access", user_id=user_id, action="info_masked",
                   outcome="fail_admin_auth", ip=ip, ua=ua)
            return jsonify({"error": "Sai mật khẩu quản trị"}), 403

        db = app.config["db"]
        user = db.get_user(user_id)
        if not user:
            return jsonify({"error": "User không tồn tại"}), 404

        prefs = user.get("preferences", {}) or {}
        email = (prefs.get("email") or "").strip()
        # Voice status: dựa trên embedding tồn tại cho backend hiện tại.
        spk_mgr = app.config["spk_mgr"]
        try:
            cache = db.load_all_embeddings(
                expected_dim=spk_mgr.encoder.embedding_dim,
                backend_id=spk_mgr.encoder.backend_id,
            )
            voice_registered = user_id in cache
        except Exception:
            voice_registered = False

        _audit("admin.access", user_id=user_id, action="info_masked",
               outcome="success", ip=ip, ua=ua)
        return jsonify({
            "user_id":        user_id,
            "name_masked":    _mask_name(user["name"]),
            "email_masked":   _mask_email(email),
            "voice_status":   "Registered" if voice_registered else "Not Registered",
            "has_password":   db.has_password(user_id),
            "revoke_pending": bool(user.get("revoke_pending")),
            "created_at":     user["created_at"],
            "backend":        spk_mgr.encoder.backend_id,
        })

    @app.route("/api/admin/users/<user_id>/revoke-voice", methods=["POST"])
    @max_size(0.01)
    @rate_limit(max_attempts=3, window_sec=300,
                scope="admin-revoke-voice", key_user_field="user_id")
    def api_admin_revoke_voice(user_id):
        """Admin "Revoke & Re-enroll" — xoá voice embedding, set cờ revoke_pending.

        Hành vi sau khi gọi:
        - DELETE FROM embeddings WHERE user_id=? (tất cả backend).
        - SET users.revoke_pending = 1 → router block IMPORTANT cho user này.
        - User phải tự gọi /api/users/<id>/reenroll-voice (cần password current) +
          upload mẫu giọng mới để clear cờ.
        - KHÔNG xoá user, KHÔNG xoá password, KHÔNG xoá oauth/preferences/files.

        Body: {admin_password, reason?}
        """
        data = _json_body()
        ip = _rl_client_ip()
        ua = request.headers.get("User-Agent", "")
        if not _check_admin_pass(data.get("admin_password", "")):
            _audit("admin.access", user_id=user_id, action="revoke_voice",
                   outcome="fail_admin_auth", ip=ip, ua=ua)
            return jsonify({"error": "Sai mật khẩu quản trị"}), 403

        db = app.config["db"]
        user = db.get_user(user_id)
        if not user:
            return jsonify({"error": "User không tồn tại"}), 404

        reason = str(data.get("reason", "") or "")[:200]
        # 1) Xoá embedding của TẤT CẢ backend.
        db.revoke_voice_embedding(user_id, backend_id=None)
        # 2) Set cờ revoke_pending.
        db.set_revoke_pending(user_id, True)
        # 3) Invalidate cache SpeakerManager.
        app.config["spk_mgr"].invalidate_cache()
        # 4) Force user logout nếu đang có file session.
        if session.get("file_uid") == user_id:
            _clear_file_session()

        _audit("admin.revoke_voice", user_id=user_id, actor="admin",
               outcome="success", reason=reason, ip=ip, ua=ua)
        return jsonify({
            "ok": True,
            "user_id": user_id,
            "message": ("Đã thu hồi giọng nói. User phải gọi "
                        "/api/users/<id>/reenroll-voice với mật khẩu + mẫu "
                        "giọng mới để kích hoạt lại."),
        })

    @app.route("/api/admin/users/<user_id>/request-reenroll", methods=["POST"])
    @max_size(0.01)
    @rate_limit(max_attempts=5, window_sec=300,
                scope="admin-request-reenroll", key_user_field="user_id")
    def api_admin_request_reenroll(user_id):
        """Admin yêu cầu user re-enroll lại giọng nhưng KHÔNG xoá embedding."""
        data = _json_body()
        ip = _rl_client_ip()
        ua = request.headers.get("User-Agent", "")
        if not _check_admin_pass(data.get("admin_password", "")):
            return jsonify({"error": "Sai mật khẩu quản trị"}), 403
        db = app.config["db"]
        user = db.get_user(user_id)
        if not user:
            return jsonify({"error": "User không tồn tại"}), 404
        reason = str(data.get("reason", "") or "")[:200]
        db.set_revoke_pending(user_id, True)
        if session.get("file_uid") == user_id:
            _clear_file_session()
        _audit("admin.request_reenroll", user_id=user_id, actor="admin",
               outcome="success", reason=reason, ip=ip, ua=ua)
        return jsonify({"ok": True, "user_id": user_id,
                        "message": "Đã yêu cầu user đăng ký lại giọng nói."})

    @app.route("/api/admin/users/<user_id>/delete", methods=["POST"])
    @max_size(0.01)
    @rate_limit(max_attempts=3, window_sec=300,
                scope="admin-delete-user", key_user_field="user_id")
    def api_admin_delete_user(user_id):
        """Admin revoke theo nghĩa xoá hẳn user + dữ liệu liên quan."""
        data = _json_body()
        ip = _rl_client_ip()
        ua = request.headers.get("User-Agent", "")
        if not _check_admin_pass(data.get("admin_password", "")):
            return jsonify({"error": "Sai mật khẩu quản trị"}), 403
        db = app.config["db"]
        if not db.get_user(user_id):
            return jsonify({"error": "User không tồn tại"}), 404

        db.delete_user(user_id)
        app.config["spk_mgr"].invalidate_cache()
        import shutil
        dirs_removed = []
        for base in (config.ENROLL_AUDIO_DIR, config.USER_FILES_DIR):
            target = _resolve_child_path(base, user_id)
            if target and target.exists() and target.is_dir():
                shutil.rmtree(target)
                dirs_removed.append(str(target.relative_to(config.DATA_DIR)))
        if session.get("file_uid") == user_id:
            _clear_file_session()
        _audit("admin.delete_user", user_id=user_id, actor="admin",
               outcome="success", dirs_removed=dirs_removed, ip=ip, ua=ua)
        return jsonify({"ok": True, "user_id": user_id})

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
        # Lưu timestamp để callback có thể chẩn đoán session-lost vs state-mismatch.
        session["oauth_started_at"]    = time.time()
        _oauth_pending_put(nonce, {"user_id": user_id, "code_verifier": verifier})
        session.permanent = True
        # Force session lưu xuống cookie ngay — flask sometimes lazy-writes
        # khi response không có Set-Cookie trigger.
        session.modified = True
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
        started_at     = session.pop("oauth_started_at", None)

        # CSRF / session check — phân biệt 3 trường hợp để user biết hành động:
        #   (a) session không có nonce → cookie không tới (Secure/SameSite/restart)
        #   (b) nonce có nhưng state khác → real CSRF hoặc multiple OAuth tab.
        #   (c) hợp lệ → tiếp tục.
        nonce_ok = False
        if expected_nonce is not None:
            nonce_ok = hmac.compare_digest(str(received_state), str(expected_nonce))
        elif received_state:
            pending = _oauth_pending_pop(received_state)
            if pending:
                expected_nonce = received_state
                user_id = user_id or pending.get("user_id")
                code_verifier = code_verifier or pending.get("code_verifier")
                started_at = pending.get("created_at")
                nonce_ok = True
        if not nonce_ok:
            ip = _rl_client_ip()
            reason = "no_session_nonce" if expected_nonce is None else "state_mismatch"
            _audit("oauth.callback_failed", outcome="csrf", reason=reason,
                   ip=ip, ua=request.headers.get("User-Agent", ""),
                   session_age_s=(round(time.time() - started_at, 1)
                                  if started_at else None))
            # Thông báo cho user theo từng case — tránh "session expired" mơ hồ.
            if expected_nonce is None:
                hint = (
                    "Trình duyệt không gửi session cookie tới callback. Thường do "
                    "1 trong các nguyên nhân: app vừa restart (FLASK_SECRET random "
                    "thay đổi → cookie cũ invalid), cookie bị xoá, hoặc chạy HTTP "
                    "mà session cookie có cờ Secure (set FLASK_DEV_HTTP=true để "
                    "tạm tắt khi dev local HTTP)."
                )
            else:
                hint = ("State trả về từ Google không khớp nonce trong session. "
                        "Có thể bạn mở nhiều tab OAuth song song, hoặc bị CSRF "
                        "thật. Đóng tất cả tab OAuth rồi thử lại từ đầu.")
            return (
                "<!doctype html><html><head><meta charset='utf-8'>"
                "<style>body{font-family:sans-serif;text-align:center;padding:48px;max-width:600px;margin:auto}"
                "h2{color:#c62828}.hint{background:#fff8e1;border:1px solid #ffe082;"
                "padding:12px;border-radius:6px;text-align:left;margin:16px 0;font-size:14px}"
                "</style></head><body>"
                f"<h2>✗ Xác thực Google bị từ chối</h2>"
                f"<p>State không hợp lệ ({html.escape(reason)}).</p>"
                f"<div class='hint'>{html.escape(hint)}</div>"
                "<button onclick='window.close()' style='padding:8px 20px;"
                "background:#1565c0;color:#fff;border:none;border-radius:6px;"
                "cursor:pointer'>Đóng tab</button></body></html>"
            ), 400

        # Lỗi từ Google (vd: user từ chối, redirect_uri_mismatch)
        if error:
            return (
                "<!doctype html><html><head><meta charset='utf-8'>"
                "<style>body{font-family:sans-serif;text-align:center;padding:60px}"
                "h2{color:#c62828}pre{background:#f5f5f5;padding:12px;border-radius:6px;"
                "text-align:left;max-width:500px;margin:auto;word-break:break-all}</style></head><body>"
                f"<h2>✗ Xác thực thất bại</h2>"
                f"<pre>{html.escape(error)}</pre>"
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
            return f"<h3>Không tìm thấy user '{html.escape(user_id)}'</h3>", 404

        try:
            tokens     = exchange_code(code, code_verifier=code_verifier)
            gmail_addr = get_user_email(tokens["access_token"])
            tokens["gmail_address"] = gmail_addr
            _db.save_oauth_token(user_id, tokens)

            prefs = dict(user["preferences"])
            prefs["email"] = gmail_addr
            _db.update_preferences(user_id, prefs)

            # Đánh dấu user vừa hoàn tất OAuth — cho phép /api/oauth/status trả
            # về gmail address trong vài phút polling sau (UX modal đóng + show
            # "Đã xác thực: user@gmail.com"). Sau đó marker tự expire.
            recent = session.get("oauth_recent", {})
            recent[user_id] = time.time()
            # Xoá entry quá cũ (> 10 phút) để cookie không phình ra.
            recent = {u: t for u, t in recent.items() if time.time() - t < 600}
            session["oauth_recent"] = recent

            _audit("oauth.granted", user_id=user_id, outcome="success",
                   gmail_hash=hashlib.sha256(gmail_addr.encode()).hexdigest()[:12],
                   ip=_rl_client_ip(),
                   ua=request.headers.get("User-Agent", ""))
        except Exception as e:
            return (
                "<!doctype html><html><head><meta charset='utf-8'>"
                "<style>body{font-family:sans-serif;text-align:center;padding:60px}"
                "h2{color:#c62828}pre{background:#f5f5f5;padding:12px;border-radius:6px;"
                "text-align:left;max-width:500px;margin:auto;word-break:break-all}</style></head><body>"
                f"<h2>✗ Lỗi trao đổi token</h2><pre>{html.escape(str(e))}</pre>"
                "<button onclick='window.close()' style='margin-top:16px;padding:8px 20px;"
                "background:#1565c0;color:#fff;border:none;border-radius:6px;cursor:pointer'>"
                "Đóng tab</button></body></html>"
            ), 500

        # Tab tự đóng sau 3s, đồng thời báo hiệu cho cửa sổ mẹ (nếu có)
        _nonce = getattr(g, "csp_nonce", "")
        return (
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<style>body{font-family:sans-serif;text-align:center;padding:60px}"
            "h2{color:#2e7d32}.badge{display:inline-block;background:#e8f5e9;"
            "border:1px solid #a5d6a7;border-radius:8px;padding:12px 24px;margin:12px 0}"
            ".cnt{font-size:2rem;font-weight:bold;color:#1565c0}</style></head><body>"
            f"<h2>✓ Xác thực Gmail thành công!</h2>"
            f"<div class='badge'>📧 <strong>{html.escape(gmail_addr)}</strong></div>"
            "<p>Tab sẽ tự đóng trong <span id='cnt' class='cnt'>3</span> giây.<br>"
            "Quay lại trang trợ lý và nói <b>\"có\"</b> để gửi email.</p>"
            "<button onclick='window.close()' style='margin-top:8px;padding:8px 20px;"
            "background:#2e7d32;color:#fff;border:none;border-radius:6px;cursor:pointer'>"
            "Đóng ngay</button>"
            f"<script nonce='{_nonce}'>"
            "try{new BroadcastChannel('oauth').postMessage({type:'oauth_done'});}catch(_){}"
            "if(window.opener){try{window.opener.postMessage({type:'oauth_done'},'*');}catch(_){}}"
            "let s=3;const t=setInterval(()=>{s--;document.getElementById('cnt').textContent=s;"
            "if(s<=0){clearInterval(t);window.close();}},1000);"
            "</script>"
            "</body></html>"
        )

    @app.route("/api/oauth/status/<user_id>")
    def api_oauth_status(user_id):
        """Kiểm tra trạng thái OAuth của user — dùng bởi frontend polling.

        Authenticated flag là public (chỉ là bool, không leak PII).
        Gmail address CHỈ trả khi caller có quyền:
          - file session match user_id (đã unlock bằng password / voice),
          - đang trong OAuth flow của chính user_id này (oauth_pending_uid),
          - vừa hoàn tất OAuth (oauth_recent marker, ~10 min) — UX hiển thị
            "Đã xác thực: user@gmail.com" sau callback.
        Caller không quyền chỉ thấy `gmail = ""`.
        """
        _db  = app.config["db"]
        tok  = _db.get_oauth_token(user_id)
        in_oauth_flow = session.get("oauth_pending_uid") == user_id
        recent_ts     = session.get("oauth_recent", {}).get(user_id, 0)
        recently_done = time.time() - recent_ts < 600  # 10 phút
        can_see_gmail = (
            _get_authed_uid("any") == user_id
            or in_oauth_flow
            or recently_done
        )
        return jsonify({
            "authenticated": tok is not None,
            "gmail":         (tok["gmail_address"] if (tok and can_see_gmail) else ""),
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
    # ----- Liveness vs Readiness vs Info -----
    # Pattern chuẩn Kubernetes / production:
    #   /healthz  → liveness: process còn sống không (đáp ngay, không touch ML)
    #   /readyz   → readiness: model đã load chưa (depends_on init)
    #   /api/health → info dump (legacy, giữ cho backward compat)
    @app.route("/healthz")
    def healthz():
        """Liveness probe — server process còn responsive.
        KHÔNG check model/DB → trả 200 ngay cả khi đang load."""
        return jsonify({"status": "alive"})

    @app.route("/readyz")
    def readyz():
        """Readiness probe — tất cả dependency đã sẵn sàng phục vụ.
        Check: DB query OK, encoder loaded, ASR loaded. 503 nếu chưa ready."""
        try:
            _ = app.config["db"].list_users()
            encoder_ok = app.config["spk_mgr"].encoder is not None
            asr_ok     = app.config["asr"] is not None
            if encoder_ok and asr_ok:
                return jsonify({"status": "ready"})
            return jsonify({"status": "not_ready",
                            "encoder": encoder_ok, "asr": asr_ok}), 503
        except Exception as e:
            return jsonify({"status": "not_ready", "error": str(e)[:120]}), 503

    @app.route("/api/health")
    def health():
        """Info dump — backward compat. Cho debug, không phải probe."""
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

    if not args.ssl and "FLASK_DEV_HTTP" not in os.environ:
        os.environ["FLASK_DEV_HTTP"] = "true"
        _log.warning("Chạy HTTP dev → tự set FLASK_DEV_HTTP=true để session cookie không có cờ Secure")

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
