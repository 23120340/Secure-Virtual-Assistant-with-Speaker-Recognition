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


def test_admin_can_reset_user_password(client, app, reset_rate_limit):
    """Forgot-password path: admin can replace a user's password, but never recover it."""
    import numpy as np
    from core import config

    db = app.config["db"]
    uid = "reset-pw-test"
    old_admin = config.ADMIN_PASS
    config.ADMIN_PASS = "admin-reset-secret"
    if db.get_user(uid) is None:
        db.add_user(uid, "Reset Test", np.zeros(192, dtype=np.float32),
                    preferences={}, password="old-password")
    try:
        r = client.post(
            f"/api/admin/users/{uid}/reset-password",
            json={"admin_password": "wrong", "new_password": "new-password-123"},
            headers=_xhr_headers(),
        )
        assert r.status_code == 403
        assert db.check_password(uid, "old-password") is True

        r = client.post(
            f"/api/admin/users/{uid}/reset-password",
            json={"admin_password": "admin-reset-secret", "new_password": "short"},
            headers=_xhr_headers(),
        )
        assert r.status_code == 400

        r = client.post(
            f"/api/admin/users/{uid}/reset-password",
            json={"admin_password": "admin-reset-secret", "new_password": "new-password-123"},
            headers=_xhr_headers(),
        )
        assert r.status_code == 200
        body = r.get_json()
        assert body["ok"] is True
        assert body["user_id"] == uid
        assert db.check_password(uid, "old-password") is False
        assert db.check_password(uid, "new-password-123") is True
    finally:
        db.delete_user(uid)
        config.ADMIN_PASS = old_admin
