"""User database: SQLite cho metadata, embedding lưu dưới dạng BLOB (numpy bytes).

Schema:
    users(user_id PK, name, created_at, preferences_json, password_hash)
    embeddings(user_id FK, embedding BLOB)  -- centroid 192-d
"""
import base64
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import numpy as np
from datetime import datetime, timezone
from hashlib import pbkdf2_hmac
from pathlib import Path
from typing import Optional

from . import config
from . import speaker_encoder


# ==========================================================================
# OAuth token encryption (Fernet AES-128-CBC + HMAC-SHA256)
# ==========================================================================
# Key dẫn xuất từ FLASK_SECRET. Nếu thiếu → throw runtime để admin set env.
# DB leak alone = không đủ để chiếm Gmail (cần thêm FLASK_SECRET ở env server).
_FERNET = None


def _fernet():
    """Lazy init Fernet từ FLASK_SECRET."""
    global _FERNET
    if _FERNET is None:
        from cryptography.fernet import Fernet
        secret = os.environ.get("FLASK_SECRET", "").strip()
        if not secret:
            raise RuntimeError(
                "Cần set FLASK_SECRET để mã hoá OAuth tokens. "
                "Sinh bằng: python -c \"import secrets; print(secrets.token_hex(32))\""
            )
        key = hashlib.sha256(secret.encode("utf-8")).digest()
        _FERNET = Fernet(base64.urlsafe_b64encode(key))
    return _FERNET


def _encrypt(plaintext: str) -> str:
    if not plaintext:
        return ""
    return _fernet().encrypt(plaintext.encode("utf-8")).decode("ascii")


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
            # Có thể là plaintext bắt đầu bằng gAAAA → trả thẳng
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
            c.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    user_id TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(user_id)
                )
            """)
            try:
                c.execute("ALTER TABLE users ADD COLUMN password_hash TEXT DEFAULT ''")
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

    # ----------------------------------------------------------------
    # User CRUD
    # ----------------------------------------------------------------
    def add_user(self, user_id: str, name: str,
                 embedding: np.ndarray, preferences: dict = None,
                 password: str = ""):
        prefs = json.dumps(preferences or {})
        pw_hash = _hash_pw(password) if password else ""
        with self._conn() as c:
            c.execute(
                "INSERT INTO users(user_id, name, created_at, preferences, password_hash) "
                "VALUES (?, ?, ?, ?, ?)",
                (user_id, name, datetime.now(timezone.utc).isoformat(), prefs, pw_hash),
            )
            c.execute(
                "INSERT INTO embeddings(user_id, embedding) VALUES (?, ?)",
                (user_id, embedding.astype(np.float32).tobytes()),
            )

    def update_embedding(self, user_id: str, embedding: np.ndarray):
        with self._conn() as c:
            c.execute("UPDATE embeddings SET embedding=? WHERE user_id=?",
                      (embedding.astype(np.float32).tobytes(), user_id))

    def get_user(self, user_id: str) -> Optional[dict]:
        with self._conn() as c:
            row = c.execute(
                "SELECT name, created_at, preferences FROM users WHERE user_id=?",
                (user_id,)
            ).fetchone()
        if not row:
            return None
        return {"user_id": user_id, "name": row[0],
                "created_at": row[1], "preferences": json.loads(row[2])}

    def update_preferences(self, user_id: str, preferences: dict):
        with self._conn() as c:
            c.execute("UPDATE users SET preferences=? WHERE user_id=?",
                      (json.dumps(preferences), user_id))

    def list_users(self) -> list:
        with self._conn() as c:
            rows = c.execute("SELECT user_id, name, created_at FROM users").fetchall()
        return [{"user_id": r[0], "name": r[1], "created_at": r[2]} for r in rows]

    def find_user_by_name_substring(self, query: str) -> Optional[dict]:
        """Tìm 1 user theo substring tên (case-insensitive). Trả về None nếu
        không thấy hoặc match nhiều hơn 1 user (ambiguous).
        Thay thế cho pattern N+1 query list_users() + get_user() trong handlers."""
        q = (query or "").lower().strip()
        if not q:
            return None
        with self._conn() as c:
            rows = c.execute(
                "SELECT user_id, name, preferences FROM users "
                "WHERE LOWER(name) LIKE ? LIMIT 2",
                (f"%{q}%",)
            ).fetchall()
        if len(rows) != 1:
            return None
        return {
            "user_id": rows[0][0],
            "name": rows[0][1],
            "preferences": json.loads(rows[0][2] or "{}"),
        }

    def delete_user(self, user_id: str):
        with self._conn() as c:
            c.execute("DELETE FROM embeddings WHERE user_id=?", (user_id,))
            c.execute("DELETE FROM users WHERE user_id=?", (user_id,))

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
    # Load tất cả embeddings (cho identification)
    # ----------------------------------------------------------------
    def load_all_embeddings(self) -> dict:
        """Trả về {user_id: (name, embedding)} cho tất cả users.

        Validate shape: ECAPA-TDNN 192-d float32. Embedding sai shape (vd: schema
        cũ 256-d, hoặc corrupt) → skip + log warning thay vì silent corrupt centroid.
        """
        import warnings
        EXPECTED_DIM = 192
        with self._conn() as c:
            rows = c.execute(
                "SELECT u.user_id, u.name, e.embedding "
                "FROM users u JOIN embeddings e ON u.user_id=e.user_id"
            ).fetchall()
        result = {}
        for uid, name, blob in rows:
            emb = np.frombuffer(blob, dtype=np.float32)
            if emb.size != EXPECTED_DIM:
                warnings.warn(
                    f"User {uid}: embedding shape={emb.size} ≠ expected {EXPECTED_DIM}. "
                    "Bỏ qua user này. Hãy enroll lại nếu schema model thay đổi.",
                    RuntimeWarning, stacklevel=2,
                )
                continue
            result[uid] = (name, emb)
        return result


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
        self._cache = self.db.load_all_embeddings()

    def enroll(self, user_id: str, name: str, audios: list,
               preferences: dict = None, password: str = "") -> np.ndarray:
        """Đăng ký user mới với list các audio mẫu. Trả về centroid."""
        if self.db.get_user(user_id):
            raise ValueError(f"User '{user_id}' đã tồn tại")
        centroid = self.encoder.encode_centroid(audios)
        self.db.add_user(user_id, name, centroid, preferences, password)
        self._cache = None  # invalidate
        return centroid

    def identify(self, audio: np.ndarray,
                 min_threshold: float | None = None):
        """SID: tìm user gần nhất. Return (user_id, name, score) hoặc
        (None, "Guest", best_score) nếu < threshold.

        Internally encodes audio rồi gọi _identify_from_embedding để không phải
        encode 2 lần khi caller cần cả SID + SV trên cùng audio."""
        if min_threshold is None:
            min_threshold = self.encoder.sid_min_threshold
        emb = self.encoder.encode(audio)
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

        emb = self.encoder.encode(audio)
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
        Return (is_match: bool, score: float)."""
        if threshold is None:
            threshold = self.encoder.sv_threshold
        if self._cache is None:
            self._refresh_cache()
        if claimed_user_id not in self._cache:
            return (False, 0.0)
        emb = self.encoder.encode(audio)
        ref = self._cache[claimed_user_id][1]
        score = speaker_encoder.cosine(emb, ref)
        return (score >= threshold, score)

    def invalidate_cache(self):
        """Public method để invalidate cache thay vì truy cập _cache trực tiếp."""
        self._cache = None
