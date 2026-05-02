"""User database: SQLite cho metadata, embedding lưu dưới dạng BLOB (numpy bytes).

Schema:
    users(user_id PK, name, created_at, preferences_json, password_hash)
    embeddings(user_id FK, embedding BLOB)  -- centroid 192-d
"""
import hashlib
import json
import sqlite3
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import config
from . import speaker_encoder


def _hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode("utf-8")).hexdigest()


class UserDB:
    def __init__(self, db_path: Path = config.DB_PATH):
        self.db_path = Path(db_path)
        self._init()

    def _conn(self):
        return sqlite3.connect(str(self.db_path))

    def _init(self):
        with self._conn() as c:
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
                (user_id, name, datetime.utcnow().isoformat(), prefs, pw_hash),
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

    def delete_user(self, user_id: str):
        with self._conn() as c:
            c.execute("DELETE FROM embeddings WHERE user_id=?", (user_id,))
            c.execute("DELETE FROM users WHERE user_id=?", (user_id,))

    def check_password(self, user_id: str, password: str) -> bool:
        """Verify password for user_id.
        Returns False if user not found.
        Returns True if no password is set (backward compat).
        Otherwise compares stored hash to hash of supplied password."""
        with self._conn() as c:
            row = c.execute(
                "SELECT password_hash FROM users WHERE user_id=?",
                (user_id,)
            ).fetchone()
        if row is None:
            return False
        stored_hash = row[0]
        if stored_hash == "":
            return True
        return stored_hash == _hash_pw(password)

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
        """Lưu hoặc cập nhật token OAuth cho user."""
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
                 token_data.get("access_token", ""),
                 token_data.get("refresh_token", ""),
                 token_data.get("gmail_address", ""),
                 token_data.get("expiry", 0.0)),
            )

    def get_oauth_token(self, user_id: str) -> Optional[dict]:
        """Trả về dict token hoặc None nếu chưa có."""
        with self._conn() as c:
            row = c.execute(
                "SELECT access_token, refresh_token, gmail_address, expiry "
                "FROM oauth_tokens WHERE user_id=?",
                (user_id,)
            ).fetchone()
        if not row:
            return None
        return {
            "access_token":  row[0],
            "refresh_token": row[1],
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
        """Trả về {user_id: (name, embedding)} cho tất cả users."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT u.user_id, u.name, e.embedding "
                "FROM users u JOIN embeddings e ON u.user_id=e.user_id"
            ).fetchall()
        result = {}
        for uid, name, blob in rows:
            emb = np.frombuffer(blob, dtype=np.float32)
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
                 min_threshold: float = config.SID_MIN_THRESHOLD):
        """SID: tìm user gần nhất. Return (user_id, name, score) hoặc
        (None, "Guest", best_score) nếu < threshold."""
        if self._cache is None:
            self._refresh_cache()
        if not self._cache:
            return (None, "Guest", 0.0)

        emb = self.encoder.encode(audio)
        best_uid, best_name, best_score = None, "Guest", -1.0
        for uid, (name, ref) in self._cache.items():
            score = speaker_encoder.cosine(emb, ref)
            if score > best_score:
                best_uid, best_name, best_score = uid, name, score

        if best_score < min_threshold:
            return (None, "Guest", best_score)
        return (best_uid, best_name, best_score)

    def verify(self, audio: np.ndarray, claimed_user_id: str,
               threshold: float = config.SV_THRESHOLD):
        """SV: kiểm tra audio có khớp với claimed user không.
        Return (is_match: bool, score: float)."""
        if self._cache is None:
            self._refresh_cache()
        if claimed_user_id not in self._cache:
            return (False, 0.0)
        emb = self.encoder.encode(audio)
        ref = self._cache[claimed_user_id][1]
        score = speaker_encoder.cosine(emb, ref)
        return (score >= threshold, score)
