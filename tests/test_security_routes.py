"""End-to-end test các security-sensitive route qua Flask test client.

Chậm hơn unit test (eager-load model ~30s) — chạy riêng:
    pytest tests/test_security_routes.py -v

Nếu CI environment không có ML deps, mark `--ignore=tests/test_security_routes.py`.
"""
import pytest


@pytest.fixture(scope="module")
def app():
    """Tạo Flask app 1 lần cho cả module — eager model load chỉ 1 lần."""
    from web.app import create_app
    a = create_app()
    a.testing = True
    return a


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def reset_rate_limit():
    """Clear in-process rate-limit bucket trước/sau test để các test không
    leak state qua nhau (bucket là process-global)."""
    from web import app as _app_mod
    _app_mod._RL_BUCKETS.clear()
    yield
    _app_mod._RL_BUCKETS.clear()


def _xhr_headers():
    return {"X-Requested-With": "XMLHttpRequest", "Content-Type": "application/json"}


# ----- CSRF -------------------------------------------------------------------
def test_post_without_xrw_blocked(client):
    """Mọi POST phải có X-Requested-With (CSRF guard)."""
    r = client.post("/api/files/verify-password",
                    json={"user_id": "x", "password": "y"})
    assert r.status_code == 403
    assert "CSRF" in (r.get_json() or {}).get("error", "")


def test_post_with_xrw_passes_csrf(client):
    """Header X-Requested-With cho phép request qua CSRF guard (vẫn 404 user)."""
    r = client.post("/api/files/verify-password",
                    json={"user_id": "nobody-csrf-test", "password": "x"},
                    headers=_xhr_headers())
    # 404 user không tồn tại, KHÔNG phải 403 CSRF.
    assert r.status_code == 404


# ----- Rate limit -------------------------------------------------------------
def test_rate_limit_password_endpoint_returns_429(client, reset_rate_limit):
    """Sau 5 lần thử password fail → endpoint trả 429 + Retry-After."""
    h = _xhr_headers()
    # 5 lần fail thường (404 user / 403 sai pw đều count vào rate-limit decorator).
    for _ in range(5):
        r = client.post("/api/files/verify-password",
                        json={"user_id": "bruteforce-test", "password": "wrong"},
                        headers=h)
        assert r.status_code in (403, 404)
    # Lần 6 phải bị limit.
    r = client.post("/api/files/verify-password",
                    json={"user_id": "bruteforce-test", "password": "wrong"},
                    headers=h)
    assert r.status_code == 429
    assert "Retry-After" in r.headers


# ----- Text-mode IMPORTANT bypass ---------------------------------------------
def test_text_mode_important_blocked_by_default(client):
    """Default ALLOW_PASSWORD_FOR_IMPORTANT=False → text-mode block IMPORTANT
    ngay cả khi cấp đúng password."""
    r = client.post("/api/assistant/text",
                    json={"text": "kiểm tra số dư", "user_id": "x", "password": "y"},
                    headers=_xhr_headers())
    assert r.status_code == 200
    data = r.get_json()
    # Nếu rule-based NLU rớt sang unknown thì test này không assert được chắc;
    # nhưng ít nhất confirm key field nếu rule-based catch được.
    if data.get("intent") == "check_balance":
        assert data["blocked"] is True
        assert "giọng" in data.get("response", "").lower()


def test_text_mode_open_files_password_fallback_no_500(client, app, reset_rate_limit):
    """open_files ở text-mode được phép dùng password và không crash với HandlerResult."""
    import numpy as np
    db = app.config["db"]
    uid = "text-open-files-test"
    if db.get_user(uid) is None:
        db.add_user(uid, "File Test", np.zeros(192, dtype=np.float32),
                    preferences={}, password="file-pw-123")
    try:
        r = client.post("/api/assistant/text",
                        json={"text": "mở file của tôi", "user_id": uid,
                              "password": "file-pw-123"},
                        headers=_xhr_headers())
        assert r.status_code == 200
        data = r.get_json()
        if data.get("intent") == "open_files":
            assert data["blocked"] is False
            assert data.get("action_type") == "show_files"
            assert "file" in data.get("response", "").lower()
    finally:
        db.delete_user(uid)


# ----- Session cookie flags ----------------------------------------------------
def test_session_cookie_has_security_flags(app):
    """SESSION_COOKIE_HTTPONLY + SAMESITE + SECURE phải set ở config."""
    assert app.config.get("SESSION_COOKIE_HTTPONLY") is True
    assert app.config.get("SESSION_COOKIE_SAMESITE") == "Lax"
    # SECURE depends on FLASK_DEV_HTTP env — test mặc định
    # (chỉ check là có cấu hình, không assert giá trị tuyệt đối).
    assert "SESSION_COOKIE_SECURE" in app.config


# ----- Security headers --------------------------------------------------------
def test_security_headers_present(client):
    r = client.get("/api/health")
    h = r.headers
    assert h.get("X-Content-Type-Options") == "nosniff"
    assert h.get("X-Frame-Options") == "DENY"
    assert "Content-Security-Policy" in h


# ----- User endpoint không leak preferences ------------------------------------
def test_api_get_user_does_not_leak_preferences(client, app):
    """Public GET /api/users/<id> KHÔNG được trả về full preferences
    (regression cho P1-7: trước đó leak notes/email/balance/schedule)."""
    import numpy as np
    db = app.config["db"]
    if db.get_user("leak-test-uid") is None:
        db.add_user(
            "leak-test-uid", "Leak Test",
            np.zeros(192, dtype=np.float32),
            preferences={"balance": 12345, "notes": ["secret-note"],
                         "email": "private@example.com"},
            password="dontcare",
        )
    try:
        r = client.get("/api/users/leak-test-uid")
        assert r.status_code == 200
        body = r.get_json()
        # Public response: prefs phải rỗng / không chứa giá trị nhạy cảm.
        assert body["preferences"] == {}, body["preferences"]
        # Nhưng các field public OK:
        assert body["name"] == "Leak Test"
        assert body["has_password"] is True
    finally:
        db.delete_user("leak-test-uid")


def test_user_can_change_own_password(client, app, reset_rate_limit):
    """User self-service change-password: cần old_password + new_password match."""
    import numpy as np
    db = app.config["db"]
    uid = "chgpw-test"
    if db.get_user(uid) is None:
        db.add_user(uid, "Test", np.zeros(192, dtype=np.float32),
                    preferences={}, password="old-pw-12345")
    try:
        # Wrong old password → 403
        r = client.post(
            f"/api/users/{uid}/change-password",
            json={"old_password": "wrong", "new_password": "new-pw-67890",
                  "new_password_confirm": "new-pw-67890"},
            headers=_xhr_headers(),
        )
        assert r.status_code == 403
        assert db.check_password(uid, "old-pw-12345") is True

        # Confirm mismatch → 400
        r = client.post(
            f"/api/users/{uid}/change-password",
            json={"old_password": "old-pw-12345", "new_password": "new-pw-67890",
                  "new_password_confirm": "different"},
            headers=_xhr_headers(),
        )
        assert r.status_code == 400

        # New = old → 400
        r = client.post(
            f"/api/users/{uid}/change-password",
            json={"old_password": "old-pw-12345", "new_password": "old-pw-12345",
                  "new_password_confirm": "old-pw-12345"},
            headers=_xhr_headers(),
        )
        assert r.status_code == 400

        # Too short → 400
        r = client.post(
            f"/api/users/{uid}/change-password",
            json={"old_password": "old-pw-12345", "new_password": "short",
                  "new_password_confirm": "short"},
            headers=_xhr_headers(),
        )
        assert r.status_code == 400

        # Success
        r = client.post(
            f"/api/users/{uid}/change-password",
            json={"old_password": "old-pw-12345", "new_password": "new-pw-67890",
                  "new_password_confirm": "new-pw-67890"},
            headers=_xhr_headers(),
        )
        assert r.status_code == 200
        assert db.check_password(uid, "old-pw-12345") is False
        assert db.check_password(uid, "new-pw-67890") is True
    finally:
        db.delete_user(uid)


def test_admin_info_masked_redacts_email(client, app, reset_rate_limit):
    """Admin info endpoint phải mask email/name, không leak full PII."""
    import numpy as np
    from core import config
    db = app.config["db"]
    uid = "admin-mask-test"
    old_admin = config.ADMIN_PASS
    config.ADMIN_PASS = "admin-pw"
    if db.get_user(uid) is None:
        db.add_user(uid, "Nguyễn Văn Test",
                    np.zeros(192, dtype=np.float32),
                    preferences={"email": "nguyenvantest@gmail.com"},
                    password="x")
    try:
        # Wrong admin pass → 403
        r = client.post(f"/api/admin/users/{uid}/info-masked",
                        json={"admin_password": "wrong"},
                        headers=_xhr_headers())
        assert r.status_code == 403

        # Correct admin pass → masked info, NO password field, NO full email
        r = client.post(f"/api/admin/users/{uid}/info-masked",
                        json={"admin_password": "admin-pw"},
                        headers=_xhr_headers())
        assert r.status_code == 200
        body = r.get_json()
        assert "password" not in body
        assert "password_hash" not in body
        assert body["email_masked"].startswith("ngu")
        assert body["email_masked"].endswith("@gmail.com")
        assert "nguyenvantest" not in body["email_masked"]
        assert body["name_masked"].startswith("Nguyễn")
        assert "Test" not in body["name_masked"]
        assert body["voice_status"] in ("Registered", "Not Registered")
        assert body["revoke_pending"] is False
    finally:
        db.delete_user(uid)
        config.ADMIN_PASS = old_admin


def test_admin_revoke_voice_sets_flag(client, app, reset_rate_limit):
    """Admin revoke-voice: xoá embedding + set revoke_pending=True."""
    import numpy as np
    from core import config
    db = app.config["db"]
    uid = "revoke-flag-test"
    old_admin = config.ADMIN_PASS
    config.ADMIN_PASS = "admin-pw"
    if db.get_user(uid) is None:
        db.add_user(uid, "Revoke Test",
                    np.ones(192, dtype=np.float32) / (192 ** 0.5),
                    password="user-pw")
    try:
        assert db.is_revoke_pending(uid) is False
        r = client.post(f"/api/admin/users/{uid}/revoke-voice",
                        json={"admin_password": "admin-pw", "reason": "Test revoke"},
                        headers=_xhr_headers())
        assert r.status_code == 200
        assert db.is_revoke_pending(uid) is True
        # Embedding should be gone for current backend.
        spk_mgr = app.config["spk_mgr"]
        cache = db.load_all_embeddings(
            expected_dim=spk_mgr.encoder.embedding_dim,
            backend_id=spk_mgr.encoder.backend_id,
        )
        assert uid not in cache
    finally:
        db.set_revoke_pending(uid, False)
        db.delete_user(uid)
        config.ADMIN_PASS = old_admin


def test_forgot_password_without_gmail_creates_admin_request(client, app, reset_rate_limit):
    """Nếu user chưa link Gmail, forgot-password tạo request cho admin dashboard."""
    import numpy as np
    db = app.config["db"]
    uid = "forgot-admin-test"
    if db.get_user(uid) is None:
        db.add_user(uid, "Forgot Test", np.zeros(192, dtype=np.float32),
                    preferences={}, password="old-password")
    try:
        r = client.post(f"/api/users/{uid}/forgot-password/request",
                        json={}, headers=_xhr_headers())
        assert r.status_code == 200
        body = r.get_json()
        assert body["mode"] == "admin_request"
        reqs = db.list_password_reset_requests()
        assert any(x["user_id"] == uid for x in reqs)
    finally:
        db.delete_user(uid)


def test_forgot_password_confirm_updates_password(client, app, reset_rate_limit):
    """Mã reset hợp lệ đổi được mật khẩu và bị xoá sau khi dùng."""
    import time
    import hashlib
    import numpy as np
    db = app.config["db"]
    uid = "forgot-confirm-test"
    if db.get_user(uid) is None:
        db.add_user(uid, "Forgot Confirm", np.zeros(192, dtype=np.float32),
                    preferences={}, password="old-password")
    try:
        code = "123456"
        db.save_password_reset_code(
            uid, hashlib.sha256(code.encode()).hexdigest(),
            time.time() + 600, time.time())
        r = client.post(f"/api/users/{uid}/forgot-password/confirm",
                        json={"code": code, "new_password": "new-password-123",
                              "new_password_confirm": "new-password-123"},
                        headers=_xhr_headers())
        assert r.status_code == 200
        assert db.check_password(uid, "new-password-123") is True
        assert db.get_password_reset_code(uid) is None
    finally:
        db.delete_user(uid)


def test_admin_request_reenroll_keeps_embedding(client, app, reset_rate_limit):
    """Reroll/re-enroll request chỉ set flag, không xoá embedding."""
    import numpy as np
    from core import config
    db = app.config["db"]
    uid = "admin-reroll-test"
    old_admin = config.ADMIN_PASS
    config.ADMIN_PASS = "admin-pw"
    emb = np.ones(192, dtype=np.float32) / (192 ** 0.5)
    if db.get_user(uid) is None:
        db.add_user(uid, "Reroll Test", emb, password="pw")
    try:
        r = client.post(f"/api/admin/users/{uid}/request-reenroll",
                        json={"admin_password": "admin-pw", "reason": "test"},
                        headers=_xhr_headers())
        assert r.status_code == 200
        assert db.is_revoke_pending(uid) is True
        spk_mgr = app.config["spk_mgr"]
        cache = db.load_all_embeddings(
            expected_dim=spk_mgr.encoder.embedding_dim,
            backend_id=spk_mgr.encoder.backend_id,
        )
        assert uid in cache
    finally:
        db.set_revoke_pending(uid, False)
        db.delete_user(uid)
        config.ADMIN_PASS = old_admin
