"""Google OAuth 2.0 helpers cho Gmail API + PKCE.

Flow:
    1. build_auth_url(state, code_challenge)  → redirect user to Google consent screen
    2. exchange_code(code, code_verifier)     → backend gets access_token + refresh_token
    3. get_user_email(token)                  → lấy Gmail address của user
    4. refresh_access_token()                 → làm mới khi access_token hết hạn (1h)
    5. revoke_token()                         → thu hồi token khi user đăng xuất

State parameter = nonce (random, lưu session) để chống CSRF.
code_verifier/challenge (PKCE) chống interception authorization code.
"""
import base64
import hashlib
import secrets
import time
import urllib.parse

import requests as _req

from . import config
from .resilience import CircuitBreaker, retry

# Circuit breaker chung cho Google OAuth endpoints. 5 fail consecutive → open
# 60s. Tránh hammer Google khi backend đang down + giảm tail latency cho user.
_GOOGLE_OAUTH_BREAKER = CircuitBreaker("google-oauth", fail_threshold=5, reset_after=60)


def gen_pkce_pair() -> tuple[str, str]:
    """Generate PKCE code_verifier + S256 challenge.

    Returns (verifier, challenge). Verifier lưu session, challenge gửi qua URL.
    Khi exchange code, gửi verifier để server check khớp với challenge → chống
    attacker intercept code và đổi token (vì không có verifier).
    """
    verifier = secrets.token_urlsafe(64)[:128]  # RFC 7636: 43-128 chars
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return verifier, challenge

_AUTH_URL     = "https://accounts.google.com/o/oauth2/v2/auth"
_TOKEN_URL    = "https://oauth2.googleapis.com/token"
_REVOKE_URL   = "https://oauth2.googleapis.com/revoke"
_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"

# Minimal scopes: chỉ gửi mail + lấy địa chỉ email, không đọc mail
_SCOPES = " ".join([
    "https://www.googleapis.com/auth/gmail.send",
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
])

# Marker đặt trước phản hồi từ handler khi cần OAuth — app.py nhận diện và
# tách thành action_type="oauth_required" + auth URL riêng.
OAUTH_RESP_PREFIX = "__OAUTH__"


def build_auth_url(state: str, code_challenge: str | None = None) -> str:
    """Tạo URL consent screen Google.

    state: random nonce (lưu session, validate ở callback) — chống CSRF.
    code_challenge: PKCE S256 challenge (optional, khuyến nghị) — chống code interception.
    """
    if not config.GOOGLE_CLIENT_ID:
        raise RuntimeError("GOOGLE_CLIENT_ID chưa được cấu hình trong .env")
    params = {
        "client_id":     config.GOOGLE_CLIENT_ID,
        "redirect_uri":  config.GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope":         _SCOPES,
        "access_type":   "offline",   # yêu cầu refresh_token
        "prompt":        "consent",   # luôn hỏi consent để đảm bảo có refresh_token
        "state":         state,
    }
    if code_challenge:
        params["code_challenge"]        = code_challenge
        params["code_challenge_method"] = "S256"
    return f"{_AUTH_URL}?{urllib.parse.urlencode(params)}"


def _extract_oauth_error(r) -> str:
    """Lấy thông báo lỗi rõ ràng từ response Google.

    raise_for_status mặc định raise HTTPError generic; Google trả
    {error, error_description} chi tiết hơn nên parse trước.
    """
    try:
        err = r.json()
        msg = err.get("error_description") or err.get("error", "")
        if msg:
            return str(msg)
    except Exception:
        pass
    return (r.text or "")[:200]


@_GOOGLE_OAUTH_BREAKER
@retry(max_attempts=3, backoff_base=0.5, retriable=(_req.RequestException, RuntimeError))
def exchange_code(code: str, code_verifier: str | None = None) -> dict:
    """Trao đổi authorization code lấy access_token + refresh_token.

    Trả về dict: {access_token, refresh_token, expires_in, expiry (absolute timestamp)}.
    code_verifier: nếu authorization request đã gửi code_challenge → phải gửi kèm.

    Retry 3 lần với exponential backoff cho network hiccup. Circuit breaker
    open sau 5 fail consecutive (60s reset).
    """
    payload = {
        "code":          code,
        "client_id":     config.GOOGLE_CLIENT_ID,
        "client_secret": config.GOOGLE_CLIENT_SECRET,
        "redirect_uri":  config.GOOGLE_REDIRECT_URI,
        "grant_type":    "authorization_code",
    }
    if code_verifier:
        payload["code_verifier"] = code_verifier
    r = _req.post(_TOKEN_URL, data=payload, timeout=10)
    if not r.ok:
        raise RuntimeError(f"OAuth token exchange failed: {_extract_oauth_error(r)}")
    data = r.json()
    # Chuyển expires_in (giây tương đối) → expiry (Unix timestamp tuyệt đối)
    # Trừ 60s để buffer tránh edge case
    data["expiry"] = time.time() + data.get("expires_in", 3600) - 60
    return data


@_GOOGLE_OAUTH_BREAKER
@retry(max_attempts=3, backoff_base=0.5, retriable=(_req.RequestException, RuntimeError))
def refresh_access_token(refresh_token: str) -> dict:
    """Làm mới access_token bằng refresh_token (không cần user interaction).

    Trả về dict với access_token mới và expiry mới. Retry + circuit breaker
    cùng config với exchange_code.
    """
    r = _req.post(_TOKEN_URL, data={
        "refresh_token": refresh_token,
        "client_id":     config.GOOGLE_CLIENT_ID,
        "client_secret": config.GOOGLE_CLIENT_SECRET,
        "grant_type":    "refresh_token",
    }, timeout=10)
    if not r.ok:
        raise RuntimeError(f"OAuth token refresh failed: {_extract_oauth_error(r)}")
    data = r.json()
    data["expiry"] = time.time() + data.get("expires_in", 3600) - 60
    return data


@_GOOGLE_OAUTH_BREAKER
@retry(max_attempts=3, backoff_base=0.5, retriable=(_req.RequestException, RuntimeError))
def get_user_email(access_token: str) -> str:
    """Lấy địa chỉ Gmail của user đã xác thực."""
    r = _req.get(_USERINFO_URL,
                 headers={"Authorization": f"Bearer {access_token}"},
                 timeout=10)
    if not r.ok:
        raise RuntimeError(f"Userinfo lookup failed: {_extract_oauth_error(r)}")
    return r.json().get("email", "")


def revoke_token(token: str) -> bool:
    """Thu hồi token trên phía Google.

    Returns True nếu Google xác nhận revoke (200) hoặc token đã invalid trước đó (400).
    Returns False nếu network/auth lỗi (caller có thể quyết định xóa local hay không).
    """
    if not token:
        return True
    try:
        r = _req.post(_REVOKE_URL, params={"token": token}, timeout=10)
        return r.status_code in (200, 400)
    except _req.RequestException:
        return False
