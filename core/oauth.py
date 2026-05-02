"""Google OAuth 2.0 helpers cho Gmail API.

Flow:
    1. build_auth_url(state)  → redirect user to Google consent screen
    2. exchange_code(code)    → backend gets access_token + refresh_token
    3. get_user_email(token)  → lấy Gmail address của user
    4. refresh_access_token() → làm mới khi access_token hết hạn (1h)
    5. revoke_token()         → thu hồi token khi user đăng xuất

State parameter = user_id (bind OAuth response về đúng speaker profile).
"""
import time
import urllib.parse

import requests as _req

from . import config

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


def build_auth_url(state: str) -> str:
    """Tạo URL consent screen Google. state = user_id để bind callback."""
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
    return f"{_AUTH_URL}?{urllib.parse.urlencode(params)}"


def exchange_code(code: str) -> dict:
    """Trao đổi authorization code lấy access_token + refresh_token.

    Trả về dict: {access_token, refresh_token, expires_in, expiry (absolute timestamp)}.
    """
    r = _req.post(_TOKEN_URL, data={
        "code":          code,
        "client_id":     config.GOOGLE_CLIENT_ID,
        "client_secret": config.GOOGLE_CLIENT_SECRET,
        "redirect_uri":  config.GOOGLE_REDIRECT_URI,
        "grant_type":    "authorization_code",
    }, timeout=10)
    r.raise_for_status()
    data = r.json()
    # Chuyển expires_in (giây tương đối) → expiry (Unix timestamp tuyệt đối)
    # Trừ 60s để buffer tránh edge case
    data["expiry"] = time.time() + data.get("expires_in", 3600) - 60
    return data


def refresh_access_token(refresh_token: str) -> dict:
    """Làm mới access_token bằng refresh_token (không cần user interaction).

    Trả về dict với access_token mới và expiry mới.
    """
    r = _req.post(_TOKEN_URL, data={
        "refresh_token": refresh_token,
        "client_id":     config.GOOGLE_CLIENT_ID,
        "client_secret": config.GOOGLE_CLIENT_SECRET,
        "grant_type":    "refresh_token",
    }, timeout=10)
    r.raise_for_status()
    data = r.json()
    data["expiry"] = time.time() + data.get("expires_in", 3600) - 60
    return data


def get_user_email(access_token: str) -> str:
    """Lấy địa chỉ Gmail của user đã xác thực."""
    r = _req.get(_USERINFO_URL,
                 headers={"Authorization": f"Bearer {access_token}"},
                 timeout=10)
    r.raise_for_status()
    return r.json().get("email", "")


def revoke_token(token: str) -> None:
    """Thu hồi token (access hoặc refresh) trên phía Google."""
    _req.post(_REVOKE_URL, params={"token": token}, timeout=10)
