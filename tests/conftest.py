"""Pytest fixtures dùng chung cho test suite.

Mục tiêu: test regression SECURITY (password hashing, OAuth state replay,
path traversal, session auth method). KHÔNG test ML quality — model load
chậm và phụ thuộc dữ liệu enroll thật.

Trick: nếu test cần Flask client mà không muốn nuốt 30s loading model,
import `web.app` với env `SECVA_TEST_NO_MODELS=1` để skip eager load.
Hiện chưa support — test app phải chấp nhận load ASR/encoder lần đầu chậm,
hoặc test trực tiếp các module core không qua HTTP.
"""
import os
import sys
from pathlib import Path

import pytest

# Đảm bảo root repo trong sys.path để `import core.*` từ test file hoạt động.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Đặt FLASK_SECRET trước khi import bất cứ thứ gì dùng Fernet/secret.
os.environ.setdefault("FLASK_SECRET", "test-secret-" + "0" * 50)


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """UserDB với SQLite file riêng cho test — không đụng db production."""
    from core import config, database
    db_path = tmp_path / "users.db"
    monkeypatch.setattr(config, "DB_PATH", db_path)
    return database.UserDB(db_path)
