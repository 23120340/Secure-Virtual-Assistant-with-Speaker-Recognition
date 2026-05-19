"""Test OAuth token encryption ở rest (Fernet) — không leak plaintext qua DB.

Regression cho MASTER-REVIEW DB4: bug cũ là access_token/refresh_token lưu
plaintext trong SQLite. DB backup leak → attacker có vĩnh viễn Gmail access.

Cover thêm cho P2-J: HKDF key derivation + TOKEN_ENCRYPTION_KEY dedicated.
"""
import sqlite3

import numpy as np
import pytest

from core.database import _encrypt, _decrypt, _derive_fernet_key_from_secret


def test_encrypt_roundtrip():
    plain = "ya29.a0AeXRPp7..."
    enc = _encrypt(plain)
    assert enc != plain, "phải khác plaintext"
    assert enc.startswith("gAAAA"), "Fernet ciphertext luôn bắt đầu gAAAA"
    assert _decrypt(enc) == plain


def test_decrypt_legacy_plaintext_returns_as_is():
    """Backward compat: row cũ không có prefix Fernet → trả thẳng plaintext.
    Document hành vi này để có thể tighten sau khi migration."""
    legacy = "old_plaintext_token"
    assert _decrypt(legacy) == legacy


def test_encrypt_empty_string():
    assert _encrypt("") == ""
    assert _decrypt("") == ""


def test_two_encrypts_differ_due_to_iv():
    """Fernet IV random → 2 ciphertext của cùng plaintext phải khác."""
    plain = "secret"
    a = _encrypt(plain)
    b = _encrypt(plain)
    assert a != b


def test_hkdf_derives_different_key_than_legacy_sha256():
    """HKDF + salt phải ra key KHÁC với single SHA-256 (legacy).

    Đảm bảo migration đúng — không tự nhiên collision với key cũ.
    """
    import hashlib
    secret = "the-quick-brown-fox-jumps-over-the-lazy-dog"
    legacy_key = hashlib.sha256(secret.encode()).digest()
    new_key    = _derive_fernet_key_from_secret(secret)
    assert legacy_key != new_key
    assert len(new_key) == 32


def test_hkdf_deterministic_for_same_secret():
    """Cùng secret → cùng HKDF output. Không có randomness lẫn lộn."""
    s = "fixed-secret-for-test"
    a = _derive_fernet_key_from_secret(s)
    b = _derive_fernet_key_from_secret(s)
    assert a == b


def test_dedicated_token_encryption_key(monkeypatch, tmp_path):
    """Khi TOKEN_ENCRYPTION_KEY set → dùng key đó, không qua HKDF.

    Reset module-level cache để pickup env mới.
    """
    import base64, secrets
    import core.database as db_mod
    raw = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", raw)
    # Reset cache để Fernet rebuild
    db_mod._FERNET = None
    enc = db_mod._encrypt("hello world")
    assert enc.startswith("gAAAA")
    assert db_mod._decrypt(enc) == "hello world"
    db_mod._FERNET = None  # cleanup cho test khác


def test_invalid_token_encryption_key_raises(monkeypatch):
    """TOKEN_ENCRYPTION_KEY không đúng 32 byte → RuntimeError rõ ràng."""
    import core.database as db_mod
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", "too-short-not-base64")
    db_mod._FERNET = None
    with pytest.raises(RuntimeError, match="TOKEN_ENCRYPTION_KEY"):
        db_mod._fernet()
    db_mod._FERNET = None


def test_db_stores_ciphertext_not_plaintext(tmp_db):
    """Roundtrip qua UserDB: token lưu trong SQLite KHÔNG được là plaintext."""
    # Tạo user trước (foreign key)
    fake_emb = np.zeros(192, dtype=np.float32)
    tmp_db.add_user("uid1", "Test User", fake_emb)
    tmp_db.save_oauth_token("uid1", {
        "access_token":  "secret_access_xyz",
        "refresh_token": "secret_refresh_abc",
        "gmail_address": "test@example.com",
        "expiry":        12345.0,
    })
    # Đọc raw SQLite (bypass _decrypt) để confirm ciphertext on disk.
    conn = sqlite3.connect(str(tmp_db.db_path))
    row = conn.execute(
        "SELECT access_token, refresh_token FROM oauth_tokens WHERE user_id=?",
        ("uid1",),
    ).fetchone()
    conn.close()
    assert row is not None
    assert "secret_access_xyz" not in row[0]
    assert "secret_refresh_abc" not in row[1]
    assert row[0].startswith("gAAAA")
    # Đọc qua API → plaintext khôi phục được
    tok = tmp_db.get_oauth_token("uid1")
    assert tok["access_token"] == "secret_access_xyz"
    assert tok["refresh_token"] == "secret_refresh_abc"
