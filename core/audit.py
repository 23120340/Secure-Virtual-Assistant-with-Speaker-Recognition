"""Structured audit log cho các tác vụ nhạy cảm.

JSON-line sink ghi vào `data/audit.log`. Mỗi dòng là 1 event độc lập
(parse bằng `jq` hoặc `pandas.read_json(lines=True)`).

Không log nội dung nhạy cảm:
- KHÔNG ghi raw transcript / email body / note body.
- KHÔNG ghi password / token / Fernet ciphertext.
- Chỉ ghi METADATA: user_id, intent, auth_method, scores, outcome, ip, ua hash.

Sự kiện được audit:
- `auth.password_check` — success/fail, scope (files/admin/user-update/...).
- `auth.voice_verify`   — kết quả SV cho IMPORTANT intent.
- `auth.sid`            — kết quả SID khi level=IMPORTANT hoặc PERSONAL.
- `oauth.granted`       — user link Gmail thành công.
- `oauth.revoked`       — user/admin revoke token.
- `oauth.exchange_error`— callback fail.
- `data.delete_user`    — xoá user + cascade.

Cách dùng:
    from core.audit import audit
    audit("auth.password_check", user_id=uid, scope="files",
          outcome="success", ip=request.remote_addr)
"""
import hashlib
import json
import logging
import logging.handlers
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config

# ==========================================================================
# Logger setup — lazy, idempotent
# ==========================================================================
_LOCK = threading.Lock()
_LOGGER: logging.Logger | None = None
_AUDIT_PATH = config.DATA_DIR / "audit.log"


class _JsonLineFormatter(logging.Formatter):
    """Format mỗi record thành 1 dòng JSON. `record.msg` chứa dict event."""

    def format(self, record: logging.LogRecord) -> str:
        if isinstance(record.msg, dict):
            payload = dict(record.msg)
        else:
            payload = {"message": record.getMessage()}
        payload.setdefault("ts", datetime.now(timezone.utc).isoformat())
        payload.setdefault("level", record.levelname)
        return json.dumps(payload, ensure_ascii=False, default=str)


def _get_logger() -> logging.Logger:
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER
    with _LOCK:
        if _LOGGER is not None:
            return _LOGGER
        logger = logging.getLogger("secva.audit")
        logger.setLevel(logging.INFO)
        # Tách handler riêng — không inherit root handler để tránh dirty
        # stdout console với JSON lines.
        logger.propagate = False
        if not logger.handlers:
            _AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
            # Rotate 5MB × 5 file → tổng tối đa 25MB audit cũ trên disk.
            handler = logging.handlers.RotatingFileHandler(
                str(_AUDIT_PATH),
                maxBytes=5 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",
            )
            handler.setFormatter(_JsonLineFormatter())
            logger.addHandler(handler)
        _LOGGER = logger
        return logger


# ==========================================================================
# Public API
# ==========================================================================
def _hash_short(value: str) -> str:
    """SHA-256 prefix 12 hex — đủ unique để correlate, không reverse được."""
    return hashlib.sha256(value.encode("utf-8", "replace")).hexdigest()[:12]


def audit(event: str, **fields: Any) -> None:
    """Ghi 1 audit event vào log file.

    `event` theo dạng `<area>.<action>` (vd `auth.password_check`).
    Các field nhạy cảm được auto-hash:
      - `ua` (user-agent) → `ua_hash`
      - `ip` giữ nguyên (cần cho rate-limit / IR investigation).

    Auto-attach `request_id` từ Flask `g.request_id` nếu trong request context
    — cho phép correlate audit event với trace + response header X-Request-ID.

    Truncation: bất kỳ field nào dài > 200 ký tự bị cắt để tránh file phình.
    """
    payload: dict = {"event": event}
    # Best-effort attach request_id từ Flask g (silent fail nếu không trong
    # request context — vd: audit từ background task).
    try:
        from flask import g, has_request_context
        if has_request_context():
            rid = getattr(g, "request_id", None)
            if rid:
                payload["request_id"] = rid
    except Exception:
        pass
    for k, v in fields.items():
        if v is None:
            continue
        if k == "ua":
            payload["ua_hash"] = _hash_short(str(v))
            continue
        if isinstance(v, (str, bytes)) and len(v) > 200:
            payload[k] = f"{str(v)[:200]}...(truncated)"
        else:
            payload[k] = v
    _get_logger().info(payload)
