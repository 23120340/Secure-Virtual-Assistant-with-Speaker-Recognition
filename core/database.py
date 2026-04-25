"""User database: SQLite cho metadata, embedding lưu dưới dạng BLOB (numpy bytes).

Schema:
    users(user_id PK, name, created_at, preferences_json)
    embeddings(user_id FK, embedding BLOB)  -- centroid 192-d
"""
import json
import sqlite3
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import config
from . import speaker_encoder


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

    # ----------------------------------------------------------------
    # User CRUD
    # ----------------------------------------------------------------
    def add_user(self, user_id: str, name: str,
                 embedding: np.ndarray, preferences: dict = None):
        prefs = json.dumps(preferences or {})
        with self._conn() as c:
            c.execute(
                "INSERT INTO users(user_id, name, created_at, preferences) "
                "VALUES (?, ?, ?, ?)",
                (user_id, name, datetime.utcnow().isoformat(), prefs),
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
            rows = c.execute("SELECT user_id, name FROM users").fetchall()
        return [{"user_id": r[0], "name": r[1]} for r in rows]

    def delete_user(self, user_id: str):
        with self._conn() as c:
            c.execute("DELETE FROM embeddings WHERE user_id=?", (user_id,))
            c.execute("DELETE FROM users WHERE user_id=?", (user_id,))

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
               preferences: dict = None) -> np.ndarray:
        """Đăng ký user mới với list các audio mẫu. Trả về centroid."""
        if self.db.get_user(user_id):
            raise ValueError(f"User '{user_id}' đã tồn tại")
        centroid = self.encoder.encode_centroid(audios)
        self.db.add_user(user_id, name, centroid, preferences)
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
