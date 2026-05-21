"""User database: SQLite cho metadata, embedding lưu dưới dạng BLOB (numpy bytes).

Schema:
    users(user_id PK, name, created_at, preferences_json, password_hash)
    embeddings(user_id FK, embedding BLOB)  -- centroid 192-d
"""
import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import sqlite3
import numpy as np

_log = logging.getLogger("secva.db")
from datetime import datetime, timezone
from hashlib import pbkdf2_hmac
from pathlib import Path
from typing import Optional

from . import config
from . import speaker_encoder
from . import audio_io


# ==========================================================================
# OAuth token encryption (Fernet AES-128-CBC + HMAC-SHA256)
# ==========================================================================
# Key dẫn xuất theo 2 phương án (ưu tiên đầu):
#   1. TOKEN_ENCRYPTION_KEY (env): raw 32 byte key encode base64-urlsafe.
#      Đây là phương án preferred — key dedicated, rotate độc lập với
#      FLASK_SECRET (session cookie signing).
#   2. FLASK_SECRET (env): fallback dùng HKDF-SHA256 với salt cố định project
#      để derive key. Cải tiến so với phiên bản cũ chỉ single SHA-256 — HKDF
#      kéo dài key + bind context, mạnh hơn về cryptographic standard.
# DB leak alone = không đủ để chiếm Gmail (cần thêm key ở env server).
_FERNET = None
# Salt + info HKDF cố định cho project. Không phải secret — public values dùng
# để bind key derivation với scope cụ thể, tránh key reuse cross-context.
_HKDF_SALT = b"secva-oauth-token-v1"
_HKDF_INFO = b"fernet-key"


def _derive_fernet_key_from_secret(secret: str) -> bytes:
    """HKDF-SHA256 derive 32-byte key từ secret. Stronger than single SHA-256.

    Cài đặt thủ công (HMAC-Extract + HMAC-Expand) tránh dependency mới.
    RFC 5869, OKM=32 byte → chỉ 1 round expand cần thiết.
    """
    secret_b = secret.encode("utf-8")
    # Extract: PRK = HMAC-SHA256(salt, IKM)
    prk = hmac.new(_HKDF_SALT, secret_b, hashlib.sha256).digest()
    # Expand: OKM = HMAC-SHA256(PRK, info || 0x01); 32 byte = đúng 1 block.
    okm = hmac.new(prk, _HKDF_INFO + b"\x01", hashlib.sha256).digest()
    return okm


def _fernet():
    """Lazy init Fernet — ưu tiên TOKEN_ENCRYPTION_KEY, fallback FLASK_SECRET+HKDF."""
    global _FERNET
    if _FERNET is None:
        from cryptography.fernet import Fernet
        # Phương án 1: dedicated key.
        raw_key_b64 = os.environ.get("TOKEN_ENCRYPTION_KEY", "").strip()
        if raw_key_b64:
            try:
                # Fernet key phải là 32 byte base64-urlsafe (44 char với padding).
                key = base64.urlsafe_b64decode(raw_key_b64.encode("ascii"))
                if len(key) != 32:
                    raise ValueError(f"key length = {len(key)}, expected 32")
                _FERNET = Fernet(raw_key_b64.encode("ascii"))
                return _FERNET
            except Exception as e:
                raise RuntimeError(
                    f"TOKEN_ENCRYPTION_KEY không hợp lệ ({e}). "
                    "Sinh bằng: python -c \"import secrets,base64; "
                    "print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode())\""
                )
        # Phương án 2: derive từ FLASK_SECRET qua HKDF.
        secret = os.environ.get("FLASK_SECRET", "").strip()
        if not secret:
            raise RuntimeError(
                "Cần set TOKEN_ENCRYPTION_KEY hoặc FLASK_SECRET để mã hoá OAuth tokens. "
                "Khuyến nghị TOKEN_ENCRYPTION_KEY (dedicated key)."
            )
        key = _derive_fernet_key_from_secret(secret)
        _FERNET = Fernet(base64.urlsafe_b64encode(key))
    return _FERNET


def _encrypt(plaintext: str) -> str:
    if not plaintext:
        return ""
    return _fernet().encrypt(plaintext.encode("utf-8")).decode("ascii")


# Legacy key derivation (single SHA-256 từ FLASK_SECRET) — giữ để decrypt
# các token cũ trong DB từ phiên bản trước HKDF migration. Sẽ deprecate sau
# khi tất cả token được re-encrypt (re-save_oauth_token trigger tự nhiên khi
# refresh_access_token hoặc user link Gmail lại).
_LEGACY_FERNET = None


def _legacy_fernet():
    """Fernet derived bằng single SHA-256 (phương án cũ, pre-HKDF)."""
    global _LEGACY_FERNET
    if _LEGACY_FERNET is None:
        from cryptography.fernet import Fernet
        secret = os.environ.get("FLASK_SECRET", "").strip()
        if not secret:
            return None
        key = hashlib.sha256(secret.encode("utf-8")).digest()
        _LEGACY_FERNET = Fernet(base64.urlsafe_b64encode(key))
    return _LEGACY_FERNET


def _decrypt(ciphertext: str) -> str:
    if not ciphertext:
        return ""
    # Backward compat: nếu chuỗi không có dạng Fernet (gAAAA...), giả định là plaintext
    # cũ → vẫn trả về (sẽ được encrypt lại khi save tiếp). Cách này tránh phải migrate
    # cứng nhắc và giữ runtime ổn định khi DB cũ có token plaintext.
    if not ciphertext.startswith("gAAAA"):
        return ciphertext
    try:
        from cryptography.fernet import InvalidToken
        try:
            return _fernet().decrypt(ciphertext.encode("ascii")).decode("utf-8")
        except InvalidToken:
            # Có thể là token encrypted bằng legacy key (single SHA-256).
            # Thử legacy key — nếu pass thì backward-compat OK; nếu fail thì
            # đây thực sự là plaintext bắt đầu gAAAA (rare).
            legacy = _legacy_fernet()
            if legacy is not None:
                try:
                    return legacy.decrypt(ciphertext.encode("ascii")).decode("utf-8")
                except InvalidToken:
                    pass
            return ciphertext
    except Exception:
        return ciphertext

# PBKDF2-HMAC-SHA256 parameters (OWASP 2023 minimum)
_PBKDF2_ITER = 600_000
_SALT_LEN = 16


def _hash_pw(pw: str, salt: bytes = None) -> str:
    """Hash password with PBKDF2-HMAC-SHA256 + random salt.
    Stored format: 'salt_hex$hash_hex'."""
    if salt is None:
        salt = secrets.token_bytes(_SALT_LEN)
    derived = pbkdf2_hmac("sha256", pw.encode("utf-8"), salt, _PBKDF2_ITER)
    return f"{salt.hex()}${derived.hex()}"


def _verify_pw(pw: str, stored: str) -> bool:
    """Verify password against stored hash. Returns False for empty/invalid stored values."""
    if not stored or "$" not in stored:
        return False
    try:
        salt_hex, hash_hex = stored.split("$", 1)
        derived = pbkdf2_hmac("sha256", pw.encode("utf-8"),
                              bytes.fromhex(salt_hex), _PBKDF2_ITER)
        return hmac.compare_digest(derived, bytes.fromhex(hash_hex))
    except (ValueError, TypeError):
        return False


class UserDB:
    def __init__(self, db_path: Path = config.DB_PATH):
        self.db_path = Path(db_path)
        self._init()

    def _conn(self):
        return sqlite3.connect(str(self.db_path))

    def _init(self):
        with self._conn() as c:
            # WAL = Write-Ahead Logging. Cho phép nhiều reader + 1 writer đồng thời,
            # không khoá nhau như rollback journal. Quan trọng với Flask threaded=True.
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")  # fsync ít hơn — đủ an toàn cho app này
            c.execute("PRAGMA foreign_keys=ON")
            c.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    preferences TEXT DEFAULT '{}'
                )
            """)
            # Embeddings table: composite PK (user_id, backend_id) → 1 user có
            # thể có nhiều embedding cho các backend khác nhau (ecapa, wavlm...).
            # Khi đổi SPEAKER_BACKEND, query filter theo backend_id của encoder
            # đang chạy → không cần xóa embedding cũ.
            c.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    user_id TEXT NOT NULL,
                    backend_id TEXT NOT NULL DEFAULT 'ecapa',
                    embedding BLOB NOT NULL,
                    PRIMARY KEY (user_id, backend_id),
                    FOREIGN KEY(user_id) REFERENCES users(user_id)
                )
            """)
            # Migration cho DB cũ (schema chỉ có user_id PK, không có backend_id).
            # SQLite không hỗ trợ DROP/MODIFY PK → phải build table mới + swap.
            cols = [r[1] for r in c.execute("PRAGMA table_info(embeddings)")]
            if "backend_id" not in cols:
                _log.info("Migrating embeddings table: thêm cột backend_id...")
                c.execute("""
                    CREATE TABLE embeddings_new (
                        user_id TEXT NOT NULL,
                        backend_id TEXT NOT NULL DEFAULT 'ecapa',
                        embedding BLOB NOT NULL,
                        PRIMARY KEY (user_id, backend_id),
                        FOREIGN KEY(user_id) REFERENCES users(user_id)
                    )
                """)
                c.execute(
                    "INSERT INTO embeddings_new(user_id, backend_id, embedding) "
                    "SELECT user_id, 'ecapa', embedding FROM embeddings"
                )
                c.execute("DROP TABLE embeddings")
                c.execute("ALTER TABLE embeddings_new RENAME TO embeddings")
                _log.info("Migration done.")
            try:
                c.execute("ALTER TABLE users ADD COLUMN password_hash TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            # `revoke_pending`: cờ admin set sau khi "Revoke & Re-enroll" — block
            # tất cả tác vụ IMPORTANT cho tới khi user tự re-enroll giọng. Reset
            # về 0 sau khi user upload mẫu mới qua /api/users/<id>/reenroll-voice.
            try:
                c.execute("ALTER TABLE users ADD COLUMN revoke_pending "
                          "INTEGER NOT NULL DEFAULT 0")
            except sqlite3.OperationalError:
                pass
            c.execute("""
                CREATE TABLE IF NOT EXISTS oauth_tokens (
                    user_id       TEXT PRIMARY KEY,
                    access_token  TEXT NOT NULL,
                    refresh_token TEXT NOT NULL,
                    gmail_address TEXT NOT NULL DEFAULT '',
                    expiry        REAL NOT NULL DEFAULT 0,
                    FOREIGN KEY(user_id) REFERENCES users(user_id)
                )
            """)
            # email_send_log: track timestamp gửi email cho rate-limit (10/hour).
            # Trước đó là in-process dict ở core/handlers — reset khi restart và
            # không chia sẻ giữa worker. Đẩy xuống SQLite để persist + multi-worker-safe.
            c.execute("""
                CREATE TABLE IF NOT EXISTS email_send_log (
                    user_id  TEXT NOT NULL,
                    sent_at  REAL NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(user_id)
                )
            """)
            c.execute("CREATE INDEX IF NOT EXISTS idx_email_send_log "
                      "ON email_send_log(user_id, sent_at)")
            c.execute("""
                CREATE TABLE IF NOT EXISTS password_reset_codes (
                    user_id      TEXT PRIMARY KEY,
                    code_hash    TEXT NOT NULL,
                    expires_at   REAL NOT NULL,
                    attempts     INTEGER NOT NULL DEFAULT 0,
                    created_at   REAL NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(user_id)
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS password_reset_requests (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id      TEXT NOT NULL,
                    requested_at REAL NOT NULL,
                    status       TEXT NOT NULL DEFAULT 'pending',
                    note         TEXT NOT NULL DEFAULT '',
                    resolved_at  REAL,
                    verification_method TEXT NOT NULL DEFAULT '',
                    admin_note   TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY(user_id) REFERENCES users(user_id)
                )
            """)
            for col, ddl in (
                ("resolved_at", "ALTER TABLE password_reset_requests ADD COLUMN resolved_at REAL"),
                ("verification_method",
                 "ALTER TABLE password_reset_requests ADD COLUMN verification_method TEXT NOT NULL DEFAULT ''"),
                ("admin_note",
                 "ALTER TABLE password_reset_requests ADD COLUMN admin_note TEXT NOT NULL DEFAULT ''"),
            ):
                try:
                    c.execute(ddl)
                except sqlite3.OperationalError:
                    pass
            c.execute("CREATE INDEX IF NOT EXISTS idx_password_reset_requests "
                      "ON password_reset_requests(status, requested_at)")

            c.execute("""
                CREATE TABLE IF NOT EXISTS model_feedback (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id  TEXT NOT NULL,
                    user_id_hash TEXT NOT NULL DEFAULT '',
                    intent      TEXT NOT NULL DEFAULT '',
                    rating      INTEGER NOT NULL,
                    comment     TEXT NOT NULL DEFAULT '',
                    created_at  TEXT NOT NULL
                )
            """)
            c.execute("CREATE INDEX IF NOT EXISTS idx_feedback_request_id "
                      "ON model_feedback(request_id)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_feedback_intent "
                      "ON model_feedback(intent)")
            c.execute("""
                CREATE TABLE IF NOT EXISTS security_alerts (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id     TEXT NOT NULL,
                    alert_type  TEXT NOT NULL,
                    severity    TEXT NOT NULL DEFAULT 'medium',
                    message     TEXT NOT NULL DEFAULT '',
                    evidence    TEXT NOT NULL DEFAULT '{}',
                    status      TEXT NOT NULL DEFAULT 'pending',
                    created_at  REAL NOT NULL,
                    resolved_at REAL,
                    admin_note  TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY(user_id) REFERENCES users(user_id)
                )
            """)
            c.execute("CREATE INDEX IF NOT EXISTS idx_security_alerts_status "
                      "ON security_alerts(status, created_at)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_security_alerts_user "
                      "ON security_alerts(user_id, status)")

    # ----------------------------------------------------------------
    # User CRUD
    # ----------------------------------------------------------------
    def add_user(self, user_id: str, name: str,
                 embedding: np.ndarray, preferences: dict = None,
                 password: str = "", backend_id: str = "ecapa"):
        """Tạo user mới + embedding cho backend chỉ định.

        backend_id default 'ecapa' để tương thích với code cũ. Caller (vd:
        SpeakerManager.enroll) truyền `self.encoder.backend_id` để khớp model.
        """
        prefs = json.dumps(preferences or {})
        pw_hash = _hash_pw(password) if password else ""
        with self._conn() as c:
            c.execute(
                "INSERT INTO users(user_id, name, created_at, preferences, password_hash) "
                "VALUES (?, ?, ?, ?, ?)",
                (user_id, name, datetime.now(timezone.utc).isoformat(), prefs, pw_hash),
            )
            c.execute(
                "INSERT INTO embeddings(user_id, backend_id, embedding) VALUES (?, ?, ?)",
                (user_id, backend_id, embedding.astype(np.float32).tobytes()),
            )

    def upsert_embedding(self, user_id: str, embedding: np.ndarray,
                         backend_id: str = "ecapa"):
        """Insert embedding mới cho (user, backend), hoặc replace nếu đã có.

        Dùng cho re-enrollment: user đã có embedding ECAPA, giờ enroll thêm WavLM
        → không động vào ECAPA, chỉ thêm WavLM row mới.
        """
        with self._conn() as c:
            c.execute(
                "INSERT INTO embeddings(user_id, backend_id, embedding) VALUES (?, ?, ?) "
                "ON CONFLICT(user_id, backend_id) DO UPDATE SET embedding=excluded.embedding",
                (user_id, backend_id, embedding.astype(np.float32).tobytes()),
            )

    def update_embedding(self, user_id: str, embedding: np.ndarray,
                         backend_id: str = "ecapa"):
        with self._conn() as c:
            c.execute("UPDATE embeddings SET embedding=? "
                      "WHERE user_id=? AND backend_id=?",
                      (embedding.astype(np.float32).tobytes(), user_id, backend_id))

    def get_user(self, user_id: str) -> Optional[dict]:
        with self._conn() as c:
            row = c.execute(
                "SELECT name, created_at, preferences, revoke_pending "
                "FROM users WHERE user_id=?",
                (user_id,)
            ).fetchone()
        if not row:
            return None
        return {
            "user_id": user_id,
            "name": row[0],
            "created_at": row[1],
            "preferences": json.loads(row[2]),
            "revoke_pending": bool(row[3]) if row[3] is not None else False,
        }

    def update_preferences(self, user_id: str, preferences: dict):
        with self._conn() as c:
            c.execute("UPDATE users SET preferences=? WHERE user_id=?",
                      (json.dumps(preferences), user_id))

    def list_users(self) -> list:
        with self._conn() as c:
            rows = c.execute("SELECT user_id, name, created_at FROM users").fetchall()
        return [{"user_id": r[0], "name": r[1], "created_at": r[2]} for r in rows]

    @staticmethod
    def _escape_like(value: str) -> str:
        """Escape SQLite LIKE wildcards so user input is treated literally."""
        return (
            (value or "")
            .replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )

    def find_user_by_name_substring(self, query: str) -> Optional[dict]:
        """Tìm 1 user theo substring tên (case-insensitive). Trả về None nếu
        không thấy hoặc match nhiều hơn 1 user (ambiguous).
        Thay thế cho pattern N+1 query list_users() + get_user() trong handlers."""
        q = (query or "").lower().strip()
        if not q:
            return None
        pattern = f"%{self._escape_like(q)}%"
        with self._conn() as c:
            rows = c.execute(
                "SELECT user_id, name, preferences FROM users "
                "WHERE LOWER(name) LIKE ? ESCAPE '\\' LIMIT 2",
                (pattern,)
            ).fetchall()
        if len(rows) != 1:
            return None
        return {
            "user_id": rows[0][0],
            "name": rows[0][1],
            "preferences": json.loads(rows[0][2] or "{}"),
        }

    def delete_user(self, user_id: str):
        """Xoá user + cascade delete embeddings + oauth_tokens (DB rows).

        File on-disk (WAV enrollment, user file uploads) caller phải tự xoá —
        ở web/app.py:api_delete_user. Lý do tách: UserDB chỉ quản lý DB,
        không biết về layout filesystem (config.ENROLL_AUDIO_DIR / config.USER_FILES_DIR).
        """
        with self._conn() as c:
            c.execute("DELETE FROM oauth_tokens WHERE user_id=?", (user_id,))
            c.execute("DELETE FROM embeddings WHERE user_id=?", (user_id,))
            c.execute("DELETE FROM password_reset_codes WHERE user_id=?", (user_id,))
            c.execute("DELETE FROM password_reset_requests WHERE user_id=?", (user_id,))
            c.execute("DELETE FROM security_alerts WHERE user_id=?", (user_id,))
            c.execute("DELETE FROM users WHERE user_id=?", (user_id,))

    # ----------------------------------------------------------------
    # Security alerts
    # ----------------------------------------------------------------
    def add_security_alert(self, user_id: str, alert_type: str, severity: str,
                           message: str, evidence: dict = None,
                           dedupe_window_sec: int = 300) -> int | None:
        now = __import__("time").time()
        evidence_json = json.dumps(evidence or {}, ensure_ascii=False)
        with self._conn() as c:
            row = c.execute(
                """SELECT id FROM security_alerts
                   WHERE user_id=? AND alert_type=? AND status='pending'
                     AND created_at >= ?
                   ORDER BY created_at DESC LIMIT 1""",
                (user_id, alert_type, now - dedupe_window_sec),
            ).fetchone()
            if row:
                c.execute(
                    """UPDATE security_alerts
                       SET severity=?, message=?, evidence=?, created_at=?
                       WHERE id=?""",
                    (severity, message[:500], evidence_json, now, int(row[0])),
                )
                return int(row[0])
            cur = c.execute(
                """INSERT INTO security_alerts
                   (user_id, alert_type, severity, message, evidence, status, created_at)
                   VALUES (?, ?, ?, ?, ?, 'pending', ?)""",
                (user_id, alert_type[:80], severity[:20], message[:500],
                 evidence_json, now),
            )
            return int(cur.lastrowid)

    def list_security_alerts(self, status: str = "pending", limit: int = 50) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                """SELECT a.id, a.user_id, COALESCE(u.name, a.user_id),
                          a.alert_type, a.severity, a.message, a.evidence,
                          a.status, a.created_at
                   FROM security_alerts a
                   LEFT JOIN users u ON u.user_id=a.user_id
                   WHERE a.status=?
                   ORDER BY a.created_at DESC
                   LIMIT ?""",
                (status, limit),
            ).fetchall()
        out = []
        for r in rows:
            try:
                evidence = json.loads(r[6] or "{}")
            except json.JSONDecodeError:
                evidence = {}
            out.append({
                "id": int(r[0]),
                "user_id": r[1],
                "name": r[2],
                "alert_type": r[3],
                "severity": r[4],
                "message": r[5],
                "evidence": evidence,
                "status": r[7],
                "created_at": float(r[8]),
            })
        return out

    def resolve_security_alert(self, alert_id: int, status: str = "resolved",
                               admin_note: str = ""):
        now = __import__("time").time()
        with self._conn() as c:
            c.execute(
                """UPDATE security_alerts
                   SET status=?, resolved_at=?, admin_note=?
                   WHERE id=?""",
                (status[:30], now, admin_note[:500], alert_id),
            )

    def check_password(self, user_id: str, password: str) -> bool:
        """Verify password for user_id.
        Returns False if user not found or no password is set."""
        with self._conn() as c:
            row = c.execute(
                "SELECT password_hash FROM users WHERE user_id=?",
                (user_id,)
            ).fetchone()
        if row is None:
            return False
        return _verify_pw(password, row[0])

    def update_user_name(self, user_id: str, name: str):
        with self._conn() as c:
            c.execute("UPDATE users SET name=? WHERE user_id=?",
                      (name, user_id))

    def update_password(self, user_id: str, new_password: str):
        with self._conn() as c:
            c.execute("UPDATE users SET password_hash=? WHERE user_id=?",
                      (_hash_pw(new_password), user_id))

    # ----------------------------------------------------------------
    # Password reset support
    # ----------------------------------------------------------------
    def save_password_reset_code(self, user_id: str, code_hash: str,
                                 expires_at: float, created_at: float):
        with self._conn() as c:
            c.execute(
                """INSERT INTO password_reset_codes(user_id, code_hash, expires_at,
                                                     attempts, created_at)
                   VALUES (?, ?, ?, 0, ?)
                   ON CONFLICT(user_id) DO UPDATE SET
                       code_hash=excluded.code_hash,
                       expires_at=excluded.expires_at,
                       attempts=0,
                       created_at=excluded.created_at""",
                (user_id, code_hash, expires_at, created_at),
            )

    def get_password_reset_code(self, user_id: str) -> Optional[dict]:
        with self._conn() as c:
            row = c.execute(
                "SELECT code_hash, expires_at, attempts, created_at "
                "FROM password_reset_codes WHERE user_id=?",
                (user_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "code_hash": row[0],
            "expires_at": float(row[1]),
            "attempts": int(row[2]),
            "created_at": float(row[3]),
        }

    def increment_password_reset_attempts(self, user_id: str):
        with self._conn() as c:
            c.execute(
                "UPDATE password_reset_codes SET attempts=attempts+1 WHERE user_id=?",
                (user_id,),
            )

    def delete_password_reset_code(self, user_id: str):
        with self._conn() as c:
            c.execute("DELETE FROM password_reset_codes WHERE user_id=?", (user_id,))

    def create_password_reset_request(self, user_id: str, note: str = "") -> int:
        now = __import__("time").time()
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO password_reset_requests(user_id, requested_at, status, note) "
                "VALUES (?, ?, 'pending', ?)",
                (user_id, now, note[:300]),
            )
            return int(cur.lastrowid)

    def list_password_reset_requests(self, status: str = "pending",
                                     limit: int = 20) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                """SELECT r.id, r.user_id, u.name, r.requested_at, r.status, r.note,
                          r.resolved_at, r.verification_method, r.admin_note
                   FROM password_reset_requests r
                   LEFT JOIN users u ON u.user_id=r.user_id
                   WHERE r.status=?
                   ORDER BY r.requested_at DESC
                   LIMIT ?""",
                (status, limit),
            ).fetchall()
        return [
            {
                "id": r[0],
                "user_id": r[1],
                "name": r[2] or r[1],
                "requested_at": float(r[3]),
                "status": r[4],
                "note": r[5],
                "resolved_at": float(r[6]) if r[6] is not None else None,
                "verification_method": r[7] or "",
                "admin_note": r[8] or "",
            }
            for r in rows
        ]

    def get_password_reset_request(self, request_id: int) -> Optional[dict]:
        with self._conn() as c:
            row = c.execute(
                """SELECT r.id, r.user_id, u.name, r.requested_at, r.status, r.note,
                          r.resolved_at, r.verification_method, r.admin_note
                   FROM password_reset_requests r
                   LEFT JOIN users u ON u.user_id=r.user_id
                   WHERE r.id=?""",
                (request_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "user_id": row[1],
            "name": row[2] or row[1],
            "requested_at": float(row[3]),
            "status": row[4],
            "note": row[5],
            "resolved_at": float(row[6]) if row[6] is not None else None,
            "verification_method": row[7] or "",
            "admin_note": row[8] or "",
        }

    def update_password_reset_request_status(self, request_id: int, status: str,
                                             verification_method: str = "",
                                             admin_note: str = ""):
        now = __import__("time").time()
        with self._conn() as c:
            c.execute(
                """UPDATE password_reset_requests
                   SET status=?, resolved_at=?, verification_method=?, admin_note=?
                   WHERE id=?""",
                (status, now, verification_method[:80], admin_note[:500], request_id),
            )

    def has_password(self, user_id: str) -> bool:
        """Returns True if a non-empty password_hash is stored for user_id."""
        with self._conn() as c:
            row = c.execute(
                "SELECT password_hash FROM users WHERE user_id=?",
                (user_id,)
            ).fetchone()
        if row is None:
            return False
        return bool(row[0])

    # ----------------------------------------------------------------
    # Voice revocation flag (admin Revoke & Re-enroll workflow)
    # ----------------------------------------------------------------
    def set_revoke_pending(self, user_id: str, pending: bool):
        """Admin flip cờ. True = user phải re-enroll giọng trước khi dùng
        IMPORTANT. False = đã re-enroll xong, mở lại quyền."""
        with self._conn() as c:
            c.execute("UPDATE users SET revoke_pending=? WHERE user_id=?",
                      (1 if pending else 0, user_id))

    def is_revoke_pending(self, user_id: str) -> bool:
        with self._conn() as c:
            row = c.execute(
                "SELECT revoke_pending FROM users WHERE user_id=?",
                (user_id,)
            ).fetchone()
        if not row:
            return False
        return bool(row[0])

    def revoke_voice_embedding(self, user_id: str, backend_id: str | None = None):
        """Xoá voice embedding của user — bước đầu của admin "Revoke & Re-enroll".

        backend_id=None → xoá EMBEDDING tất cả backend của user này (cẩn thận).
        Có backend_id → chỉ xoá embedding của backend đó (giữ backend khác).
        Mặc định xoá hết — match semantics "thu hồi giọng nói" admin mong đợi.
        """
        with self._conn() as c:
            if backend_id is None:
                c.execute("DELETE FROM embeddings WHERE user_id=?", (user_id,))
            else:
                c.execute("DELETE FROM embeddings WHERE user_id=? AND backend_id=?",
                          (user_id, backend_id))

    # ----------------------------------------------------------------
    # OAuth token storage
    # ----------------------------------------------------------------
    def save_oauth_token(self, user_id: str, token_data: dict):
        """Lưu hoặc cập nhật token OAuth cho user.

        access_token + refresh_token được mã hoá Fernet trước khi ghi DB.
        DB leak alone không đủ để chiếm Gmail (cần thêm FLASK_SECRET).
        """
        access_enc  = _encrypt(token_data.get("access_token", ""))
        refresh_enc = _encrypt(token_data.get("refresh_token", ""))
        with self._conn() as c:
            c.execute(
                """INSERT INTO oauth_tokens(user_id, access_token, refresh_token,
                                            gmail_address, expiry)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(user_id) DO UPDATE SET
                       access_token  = excluded.access_token,
                       refresh_token = CASE WHEN excluded.refresh_token != ''
                                       THEN excluded.refresh_token
                                       ELSE oauth_tokens.refresh_token END,
                       gmail_address = CASE WHEN excluded.gmail_address != ''
                                       THEN excluded.gmail_address
                                       ELSE oauth_tokens.gmail_address END,
                       expiry        = excluded.expiry""",
                (user_id,
                 access_enc,
                 refresh_enc,
                 token_data.get("gmail_address", ""),
                 token_data.get("expiry", 0.0)),
            )

    def get_oauth_token(self, user_id: str) -> Optional[dict]:
        """Trả về dict token (đã giải mã) hoặc None nếu chưa có."""
        with self._conn() as c:
            row = c.execute(
                "SELECT access_token, refresh_token, gmail_address, expiry "
                "FROM oauth_tokens WHERE user_id=?",
                (user_id,)
            ).fetchone()
        if not row:
            return None
        return {
            "access_token":  _decrypt(row[0]),
            "refresh_token": _decrypt(row[1]),
            "gmail_address": row[2],
            "expiry":        row[3],
        }

    def delete_oauth_token(self, user_id: str):
        """Xóa token (khi revoke hoặc token hết hạn không thể refresh)."""
        with self._conn() as c:
            c.execute("DELETE FROM oauth_tokens WHERE user_id=?", (user_id,))

    # ----------------------------------------------------------------
    # Email rate-limit (persisted)
    # ----------------------------------------------------------------
    def count_emails_sent_within(self, user_id: str, window_sec: float) -> int:
        """Đếm số email user_id đã gửi trong cửa sổ thời gian gần đây.
        Replace pattern dict in-process cũ ở core/handlers — persist + multi-worker-safe."""
        import time as _t
        cutoff = _t.time() - window_sec
        with self._conn() as c:
            row = c.execute(
                "SELECT COUNT(*) FROM email_send_log "
                "WHERE user_id=? AND sent_at >= ?",
                (user_id, cutoff),
            ).fetchone()
        return int(row[0]) if row else 0

    def oldest_email_within(self, user_id: str, window_sec: float) -> float | None:
        """Lấy timestamp của email cũ nhất trong cửa sổ (để tính wait_min)."""
        import time as _t
        cutoff = _t.time() - window_sec
        with self._conn() as c:
            row = c.execute(
                "SELECT MIN(sent_at) FROM email_send_log "
                "WHERE user_id=? AND sent_at >= ?",
                (user_id, cutoff),
            ).fetchone()
        return float(row[0]) if row and row[0] is not None else None

    def record_email_sent(self, user_id: str):
        """Ghi log 1 lần gửi email; cleanup row cũ > 24h để bảng không vô hạn."""
        import time as _t
        now = _t.time()
        with self._conn() as c:
            c.execute(
                "INSERT INTO email_send_log(user_id, sent_at) VALUES (?, ?)",
                (user_id, now),
            )
            # Cleanup occasional: drop row > 24h. 1 trong N lần để giảm I/O.
            if int(now) % 10 == 0:
                c.execute("DELETE FROM email_send_log WHERE sent_at < ?",
                          (now - 86400,))

    # ----------------------------------------------------------------
    # Load tất cả embeddings (cho identification)
    # ----------------------------------------------------------------
    def load_all_embeddings(self, expected_dim: int = 192,
                            backend_id: str = "ecapa") -> dict:
        """Trả về {user_id: (name, embedding)} cho user có embedding khớp
        backend_id chỉ định.

        Args:
            expected_dim: chiều embedding của encoder hiện tại. Sai shape →
                skip + warning (defensive: trường hợp DB lưu nhầm dim).
            backend_id: chỉ load embedding match backend. Đổi backend giữa các
                phiên không cần xóa DB — embedding của backend khác đơn giản
                bị filter ra ở đây.

        User chưa có embedding cho backend hiện tại → bị loại khỏi cache → SID
        sẽ trả Guest. Phải re-enroll qua `scripts/reenroll_backend.py`.
        """
        import warnings
        with self._conn() as c:
            rows = c.execute(
                "SELECT u.user_id, u.name, e.embedding "
                "FROM users u JOIN embeddings e ON u.user_id=e.user_id "
                "WHERE e.backend_id = ?",
                (backend_id,)
            ).fetchall()
        result = {}
        for uid, name, blob in rows:
            emb = np.frombuffer(blob, dtype=np.float32)
            if emb.size != expected_dim:
                warnings.warn(
                    f"User {uid}: embedding shape={emb.size} ≠ expected "
                    f"{expected_dim} (backend={backend_id}). Bỏ qua.",
                    RuntimeWarning, stacklevel=2,
                )
                continue
            result[uid] = (name, emb)
        return result

    def users_with_backend(self, backend_id: str) -> list:
        """List user_id đã enroll cho backend này. Dùng cho re-enroll script
        biết user nào còn thiếu embedding."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT user_id FROM embeddings WHERE backend_id = ?",
                (backend_id,)
            ).fetchall()
        return [r[0] for r in rows]

    # ----------------------------------------------------------------
    # Model Feedback
    # ----------------------------------------------------------------
    def add_feedback(self, request_id: str, user_id_hash: str, intent: str,
                     rating: int, comment: str = "") -> int:
        """Lưu feedback: rating (1-5) + comment tuỳ chọn cho 1 turn.

        Returns row id.
        """
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).isoformat(timespec='seconds')
        with self._conn() as c:
            cursor = c.execute(
                """INSERT INTO model_feedback
                   (request_id, user_id_hash, intent, rating, comment, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (request_id, user_id_hash or "", intent or "", rating, comment or "", ts)
            )
            return cursor.lastrowid

    def get_feedback_summary(self, intent: str = None) -> dict:
        """Get average rating + count feedback per intent (hoặc toàn bộ nếu intent=None).

        Returns: {intent: {avg_rating: float, count: int}, ...}
        """
        with self._conn() as c:
            if intent:
                rows = c.execute(
                    """SELECT intent, AVG(rating) as avg_rating, COUNT(*) as count
                       FROM model_feedback WHERE intent = ? GROUP BY intent""",
                    (intent,)
                ).fetchall()
            else:
                rows = c.execute(
                    """SELECT intent, AVG(rating) as avg_rating, COUNT(*) as count
                       FROM model_feedback GROUP BY intent"""
                ).fetchall()
        return {r[0]: {"avg_rating": round(r[1], 2), "count": r[2]} for r in rows}


# ==========================================================================
# Speaker manager: enroll, identify, verify
# ==========================================================================
class SpeakerManager:
    """High-level API cho enrollment, SID, SV. Cache embeddings trong RAM
    để identify nhanh mà không phải query DB mỗi lần."""

    def __init__(self, db: UserDB = None):
        self.db = db or UserDB()
        self.encoder = speaker_encoder.get_encoder()
        self._cache = None  # {user_id: (name, embedding)}

    def _refresh_cache(self):
        # Query encoder cho dim + backend hiện tại → cache chỉ chứa embedding
        # khớp model đang dùng. Đổi SPEAKER_BACKEND giữa các phiên → DB không
        # cần xóa, chỉ filter ở đây.
        self._cache = self.db.load_all_embeddings(
            expected_dim=getattr(self.encoder, "embedding_dim", 192),
            backend_id=getattr(self.encoder, "backend_id", "ecapa"),
        )

    # ----- Audio preparation: preprocess + encode (multi-window nếu bật) -----
    def _prepare(self, audio: np.ndarray) -> np.ndarray:
        """Pipeline: preprocess → encode (single hoặc multi-window).

        Chỉ preprocess khi flag bật (xem config). Embedding luôn L2-normalized.
        """
        prepped = audio_io.preprocess_audio(audio)
        if config.SPEAKER_MULTIWINDOW:
            return self.encoder.encode_multiwindow(prepped)
        return self.encoder.encode(prepped)

    # ----- AS-Norm-lite: z-normalize raw cosine theo cohort -----
    def _asnorm_zscore(self, emb: np.ndarray, target_uid: str,
                       raw_score: float) -> float | None:
        """Z-normalize raw cosine theo top-K cohort scores của các user khác.

        Returns z-score, hoặc None nếu cohort < SPEAKER_ASNORM_MIN_COHORT
        (statistics không đủ tin cậy với cohort nhỏ).
        """
        if not config.SPEAKER_ASNORM or self._cache is None:
            return None

        cohort_scores = [
            speaker_encoder.cosine(emb, ref)
            for uid, (_, ref) in self._cache.items()
            if uid != target_uid
        ]
        if len(cohort_scores) < config.SPEAKER_ASNORM_MIN_COHORT:
            return None

        cohort_scores.sort(reverse=True)
        top = cohort_scores[:config.SPEAKER_ASNORM_TOP_K]
        mean = float(np.mean(top))
        std  = float(np.std(top)) + 1e-6
        return (raw_score - mean) / std

    def enroll(self, user_id: str, name: str, audios: list,
               preferences: dict = None, password: str = "") -> np.ndarray:
        """Đăng ký user mới với list các audio mẫu. Trả về centroid.

        Preprocessing được apply trước encode_centroid để khớp với pipeline
        khi verify/identify (nếu SPEAKER_PREPROCESS=true ở runtime).
        Embedding được tag bằng `encoder.backend_id` → đổi backend không
        nhầm lẫn.
        """
        if self.db.get_user(user_id):
            raise ValueError(f"User '{user_id}' đã tồn tại")
        prepped = [audio_io.preprocess_audio(a) for a in audios]
        centroid = self.encoder.encode_centroid(prepped)
        self.db.add_user(user_id, name, centroid, preferences, password,
                         backend_id=self.encoder.backend_id)
        self._cache = None  # invalidate
        return centroid

    def enroll_additional_backend(self, user_id: str, audios: list) -> np.ndarray:
        """Encode lại audio đã có và lưu embedding cho backend hiện tại.

        Dùng khi user đã enroll cho backend cũ (vd: ecapa), muốn enroll thêm
        cho backend mới (vd: wavlm) — KHÔNG tạo user mới, chỉ thêm embedding.
        Idempotent: gọi lại sẽ replace embedding cho cùng backend.
        """
        if not self.db.get_user(user_id):
            raise ValueError(f"User '{user_id}' chưa tồn tại — dùng enroll() thay")
        prepped = [audio_io.preprocess_audio(a) for a in audios]
        centroid = self.encoder.encode_centroid(prepped)
        self.db.upsert_embedding(user_id, centroid,
                                 backend_id=self.encoder.backend_id)
        self._cache = None
        return centroid

    def identify(self, audio: np.ndarray,
                 min_threshold: float | None = None):
        """SID: tìm user gần nhất. Return (user_id, name, score) hoặc
        (None, "Guest", best_score) nếu < threshold.

        Internally encodes audio rồi gọi _identify_from_embedding để không phải
        encode 2 lần khi caller cần cả SID + SV trên cùng audio."""
        if min_threshold is None:
            min_threshold = self.encoder.sid_min_threshold
        emb = self._prepare(audio)
        return self._identify_from_embedding(emb, min_threshold)

    def _identify_from_embedding(self, emb: np.ndarray,
                                 min_threshold: float | None = None):
        if min_threshold is None:
            min_threshold = self.encoder.sid_min_threshold
        if self._cache is None:
            self._refresh_cache()
        if not self._cache:
            return (None, "Guest", 0.0)

        best_uid, best_name, best_score = None, "Guest", -1.0
        for uid, (name, ref) in self._cache.items():
            score = speaker_encoder.cosine(emb, ref)
            if score > best_score:
                best_uid, best_name, best_score = uid, name, score

        if best_score < min_threshold:
            return (None, "Guest", best_score)
        return (best_uid, best_name, best_score)

    def identify_with_margin(self, audio: np.ndarray,
                             min_threshold: float | None = None,
                             min_margin: float = config.SID_MIN_MARGIN):
        if min_threshold is None:
            min_threshold = self.encoder.sid_min_threshold
        """SID + margin check.

        Trả về (user_id, name, top1_score, margin):
        - Nếu top1_score < min_threshold → (None, "Guest", top1, 0.0)
        - Nếu margin (top1 − top2) < min_margin → (None, "Ambiguous", top1, margin)
          (Block intent IMPORTANT khi 2 user cạnh tranh — chống mạo danh nhẹ)
        - Else → (uid, name, top1, margin)
        """
        if self._cache is None:
            self._refresh_cache()
        if not self._cache:
            return (None, "Guest", 0.0, 0.0)

        emb = self._prepare(audio)
        scored = sorted(
            ((uid, name, speaker_encoder.cosine(emb, ref))
             for uid, (name, ref) in self._cache.items()),
            key=lambda x: -x[2],
        )
        top1_uid, top1_name, top1_score = scored[0]
        top2_score = scored[1][2] if len(scored) >= 2 else -1.0
        margin = top1_score - top2_score if len(scored) >= 2 else 1.0

        if top1_score < min_threshold:
            return (None, "Guest", top1_score, 0.0)
        if margin < min_margin:
            return (None, "Ambiguous", top1_score, margin)
        return (top1_uid, top1_name, top1_score, margin)

    def verify(self, audio: np.ndarray, claimed_user_id: str,
               threshold: float | None = None):
        """SV: kiểm tra audio có khớp với claimed user không.

        Quy trình:
          1. Raw cosine ≥ threshold (giữ ngưỡng calibrated cũ)
          2. AS-Norm (nếu cohort đủ lớn): z-score ≥ SPEAKER_ASNORM_Z_THRESHOLD
             — chống case impostor giọng giống user thật, vẫn vượt được raw
             threshold nhưng cũng giống nhiều user khác → z-score thấp.

        Return (is_match: bool, score: float). score là raw cosine để tương
        thích với UI/log; AS-Norm chỉ thêm guard, không thay thế.
        """
        if threshold is None:
            threshold = self.encoder.sv_threshold
        if self._cache is None:
            self._refresh_cache()
        if claimed_user_id not in self._cache:
            return (False, 0.0)
        emb = self._prepare(audio)
        ref = self._cache[claimed_user_id][1]
        score = speaker_encoder.cosine(emb, ref)
        if score < threshold:
            return (False, score)

        # Layer 2: AS-Norm guard. None = cohort không đủ → skip layer này.
        z = self._asnorm_zscore(emb, claimed_user_id, score)
        if z is not None and z < config.SPEAKER_ASNORM_Z_THRESHOLD:
            return (False, score)
        return (True, score)

    def invalidate_cache(self):
        """Public method để invalidate cache thay vì truy cập _cache trực tiếp."""
        self._cache = None

    def incremental_update_centroid(self, audio: np.ndarray, user_id: str,
                                    alpha: float = 0.1) -> tuple[bool, float]:
        """Cập nhật incremental centroid của user sau khi SV pass — chống
        speaker drift theo thời gian (giọng user thay đổi do tuổi/mic/nhiễu).

        Cập nhật theo công thức:
            new_centroid = normalize((1 - alpha) * old + alpha * emb_new)

        alpha nhỏ (default 0.1) để mỗi lần update chỉ "kéo" centroid 10% về
        hướng audio mới — tránh poisoning nếu 1 audio bị nhiễu.

        Returns (updated, new_centroid_norm). updated=False nếu user chưa enroll
        cho backend hiện tại.
        """
        if self._cache is None:
            self._refresh_cache()
        if user_id not in self._cache:
            return (False, 0.0)

        emb_new = self._prepare(audio)
        name, old_centroid = self._cache[user_id]

        a = max(0.0, min(1.0, float(alpha)))
        merged = (1.0 - a) * old_centroid + a * emb_new
        # Re-normalize để giữ unit-length (cosine luôn = dot product).
        norm = float(np.linalg.norm(merged)) + 1e-9
        merged = (merged / norm).astype(np.float32)

        # Persist + update cache in-place để turn tiếp theo dùng centroid mới.
        self.db.update_embedding(user_id, merged,
                                 backend_id=self.encoder.backend_id)
        self._cache[user_id] = (name, merged)
        return (True, float(np.linalg.norm(merged)))
