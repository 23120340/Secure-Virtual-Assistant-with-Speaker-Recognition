"""Test `_safe_filename`: Unicode-aware sanitizer cho file upload.

Quan tâm:
- Path separator (/, \\), shell special, control char, null byte → bị strip.
- Path traversal `..` → bị neutralize.
- Tiếng Việt NFC giữ nguyên (không bị tách diacritics).
- max_stem_len enforce (Windows path limit ~260).
- allowed_exts whitelist returns False khi mismatch.
"""
from web.app import _safe_filename


def test_strips_path_separator():
    n, ok = _safe_filename("../../etc/passwd")
    assert "/" not in n and "\\" not in n
    assert ".." not in n
    assert ok is True


def test_strips_null_byte_and_control():
    n, _ = _safe_filename("file\x00with\x01null.txt")
    assert "\x00" not in n
    assert "\x01" not in n


def test_strips_shell_special():
    n, _ = _safe_filename("a|b*c?d.txt")
    for ch in "|*?":
        assert ch not in n


def test_preserves_vietnamese_unicode():
    """Tiếng Việt diacritics giữ nguyên; whitespace bị collapse thành '_'
    (theo _safe_filename — chống tên file đa whitespace gây confusion)."""
    n, ok = _safe_filename("Báo cáo cuối kỳ.docx", allowed_exts={".docx"})
    # Diacritics nguyên vẹn (NFC).
    assert "Báo" in n and "cuối" in n and "kỳ" in n
    # Space → underscore.
    assert n == "Báo_cáo_cuối_kỳ.docx"
    assert ok is True


def test_extension_whitelist():
    _, ok = _safe_filename("evil.exe", allowed_exts={".pdf", ".txt"})
    assert ok is False
    _, ok2 = _safe_filename("doc.pdf", allowed_exts={".pdf", ".txt"})
    assert ok2 is True


def test_empty_string_yields_default():
    n, _ = _safe_filename("")
    assert n  # không rỗng — fallback "file"


def test_long_stem_truncated():
    long_stem = "x" * 200 + ".txt"
    n, _ = _safe_filename(long_stem, max_stem_len=100)
    assert len(n.rsplit(".", 1)[0]) <= 100


def test_dot_collapse():
    """Many dots → not used as path traversal."""
    n, _ = _safe_filename("a....b....c.txt")
    assert ".." not in n
