"""Test password hashing: PBKDF2 + salt, hằng-thời gian compare.

Regression chống lại bug cũ ở SecureVA-MASTER-REVIEW.md DB1/DB2/DB3:
- SHA-256 unsalted (cùng pw → cùng hash → rainbow table được).
- `stored_hash == ""` → return True (backdoor mặc định).
- Không dùng hmac.compare_digest (timing attack).
"""
import pytest

from core.database import _hash_pw, _verify_pw


def test_hash_format_has_salt_separator():
    """Hash phải có dạng `<salt_hex>$<derived_hex>` (PBKDF2 layout)."""
    h = _hash_pw("hunter2")
    assert "$" in h, "thiếu separator giữa salt và hash"
    salt_hex, derived_hex = h.split("$", 1)
    assert len(salt_hex) == 32, f"salt phải 16 byte = 32 hex char, got {len(salt_hex)}"
    assert len(derived_hex) == 64, f"sha256 hex = 64 char, got {len(derived_hex)}"


def test_hash_is_salted_so_same_password_yields_different_hash():
    """Salt random → 2 hash của cùng password phải khác nhau."""
    h1 = _hash_pw("hunter2")
    h2 = _hash_pw("hunter2")
    assert h1 != h2, "salt không random — 2 hash giống nhau là vulnerability rainbow-table"


def test_verify_pw_correct_password():
    h = _hash_pw("hunter2")
    assert _verify_pw("hunter2", h) is True


def test_verify_pw_wrong_password():
    h = _hash_pw("hunter2")
    assert _verify_pw("wrong", h) is False


def test_verify_pw_empty_stored_returns_false():
    """Bug cũ: stored_hash == "" → return True. Phải False."""
    assert _verify_pw("anything", "") is False
    assert _verify_pw("", "") is False


def test_verify_pw_invalid_stored_no_separator():
    """Stored không có '$' (corrupt / legacy SHA-256 unsalted) → False."""
    assert _verify_pw("hunter2", "abc123") is False


def test_verify_pw_invalid_hex_in_stored():
    """Stored có ký tự không phải hex → fromhex raises → False."""
    assert _verify_pw("hunter2", "zzz$abc") is False


def test_hash_unicode_password():
    """Password Unicode (tiếng Việt) hoạt động đúng."""
    pw = "mật_khẩu_tiếng_việt_2025!"
    h = _hash_pw(pw)
    assert _verify_pw(pw, h) is True
    assert _verify_pw(pw + "x", h) is False
