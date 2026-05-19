"""Test `delete_user` cascade: xoá oauth_tokens + embeddings, không orphan row.

Regression cho P0-7: bug cũ chỉ DELETE FROM users, để oauth_tokens orphan.
Re-enrollment cùng user_id sau đó kế thừa token cũ → security issue.
"""
import sqlite3
import numpy as np


def test_delete_user_cascades_embeddings(tmp_db):
    fake_emb = np.zeros(192, dtype=np.float32)
    tmp_db.add_user("uid1", "Test", fake_emb)
    # Verify trước
    conn = sqlite3.connect(str(tmp_db.db_path))
    assert conn.execute("SELECT COUNT(*) FROM embeddings WHERE user_id=?", ("uid1",)).fetchone()[0] == 1
    conn.close()

    tmp_db.delete_user("uid1")

    conn = sqlite3.connect(str(tmp_db.db_path))
    assert conn.execute("SELECT COUNT(*) FROM embeddings WHERE user_id=?", ("uid1",)).fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM users WHERE user_id=?", ("uid1",)).fetchone()[0] == 0
    conn.close()


def test_delete_user_cascades_oauth_tokens(tmp_db):
    fake_emb = np.zeros(192, dtype=np.float32)
    tmp_db.add_user("uid1", "Test", fake_emb)
    tmp_db.save_oauth_token("uid1", {
        "access_token": "a", "refresh_token": "r",
        "gmail_address": "x@y", "expiry": 0,
    })

    tmp_db.delete_user("uid1")

    # Token phải biến mất — không orphan.
    assert tmp_db.get_oauth_token("uid1") is None


def test_delete_then_reenroll_does_not_inherit_token(tmp_db):
    """Bug cũ: token tồn tại cho user_id mới sau khi delete + re-enroll same id."""
    fake_emb = np.zeros(192, dtype=np.float32)
    tmp_db.add_user("uid1", "Old", fake_emb)
    tmp_db.save_oauth_token("uid1", {
        "access_token": "old", "refresh_token": "old_r",
        "gmail_address": "old@y", "expiry": 0,
    })
    tmp_db.delete_user("uid1")
    # Re-enroll cùng id
    tmp_db.add_user("uid1", "New", fake_emb)
    # New user phải KHÔNG kế thừa token cũ.
    assert tmp_db.get_oauth_token("uid1") is None
