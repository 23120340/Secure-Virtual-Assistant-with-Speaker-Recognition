"""Gửi email qua Gmail REST API bằng OAuth access token.

Không dùng SMTP — sử dụng endpoint:
  POST https://gmail.googleapis.com/gmail/v1/users/me/messages/send
  Authorization: Bearer <access_token>

Ưu điểm so với SMTP + App Password:
  - Gửi từ đúng tài khoản Gmail của user (không qua email hệ thống)
  - Không cần lưu mật khẩu — chỉ cần OAuth token
  - Tuân thủ chính sách bảo mật Google (App Password đang bị deprecated)
  - Hỗ trợ 2FA tự nhiên qua OAuth consent flow
"""
import base64
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import requests


def send_email(access_token: str, to: str, subject: str, body: str,
               from_name: str = "") -> dict:
    """Gửi email qua Gmail API.

    Args:
        access_token: OAuth2 access token của người gửi.
        to:           Địa chỉ email người nhận.
        subject:      Tiêu đề email.
        body:         Nội dung email (plain text).
        from_name:    Tên hiển thị của người gửi (tùy chọn).

    Returns:
        dict: Response JSON từ Gmail API (chứa message id).

    Raises:
        requests.HTTPError: Khi API trả về lỗi (401, 403, 429...).
    """
    msg = MIMEText(body or " ", "plain", "utf-8")
    msg["To"]      = to
    msg["Subject"] = subject
    # Gmail API dùng "From" để hiển thị tên, địa chỉ thật lấy từ token
    if from_name:
        msg["From"] = from_name

    raw_bytes = base64.urlsafe_b64encode(msg.as_bytes()).decode("ascii")

    r = requests.post(
        "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type":  "application/json",
        },
        json={"raw": raw_bytes},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()
