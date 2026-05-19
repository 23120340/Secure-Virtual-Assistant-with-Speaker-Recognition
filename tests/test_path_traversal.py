"""Test path containment helper `_resolve_child_path`.

Regression cho bug `str(path).startswith(str(base))` ở MASTER-REVIEW Sprint 1
item path-traversal — Windows edge case: prefix path so sánh string có thể
escape ra (e.g. base="data/user", child="../user_evil/" → resolved path
`data/user_evil` startswith `data/user` = True).
"""
from pathlib import Path

from web.app import _resolve_child_path


def test_normal_child_allowed(tmp_path):
    base = tmp_path / "user_files"
    base.mkdir()
    (base / "doc.pdf").write_text("x")
    p = _resolve_child_path(base, "doc.pdf")
    assert p is not None
    assert p.name == "doc.pdf"


def test_dotdot_escape_rejected(tmp_path):
    base = tmp_path / "user_files" / "uid1"
    base.mkdir(parents=True)
    p = _resolve_child_path(base, "../uid2/secret.pdf")
    assert p is None, "phải reject child có '..' escape ra ngoài base"


def test_absolute_path_rejected(tmp_path):
    base = tmp_path / "user_files"
    base.mkdir()
    # On Windows absolute path; on POSIX cũng absolute.
    p = _resolve_child_path(base, str(tmp_path / "other" / "evil"))
    assert p is None, "absolute path không phải child của base → reject"


def test_prefix_overlap_not_confused(tmp_path):
    """Bug: base='data/user', child='../user_evil/doc' resolves to
    'data/user_evil/doc' — string startswith('data/user') True nhưng path
    không phải child. `Path.relative_to()` reject đúng."""
    parent = tmp_path / "data"
    parent.mkdir()
    base_user = parent / "user"
    base_user.mkdir()
    (parent / "user_evil").mkdir()
    p = _resolve_child_path(base_user, "../user_evil/doc")
    assert p is None


def test_nested_subdir_allowed(tmp_path):
    base = tmp_path / "user_files"
    sub = base / "subdir"
    sub.mkdir(parents=True)
    p = _resolve_child_path(base, "subdir/file.txt")
    assert p is not None
    assert p == (base / "subdir" / "file.txt").resolve()


def test_unicode_filename_allowed(tmp_path):
    """Tiếng Việt unicode trong tên file phải pass."""
    base = tmp_path / "user_files"
    base.mkdir()
    p = _resolve_child_path(base, "Ghi_chú_quan_trọng.pdf")
    assert p is not None
    assert "Ghi" in p.name and "trọng" in p.name
