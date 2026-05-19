"""Structured turn logs for audit, benchmark, and NLU dataset collection.

These logs are meant to be machine-readable JSONL.  They keep enough signal for
model training/evaluation while redacting secrets and high-risk free-form data.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config


SCHEMA_VERSION = "secva.turn.v1"
NLU_SCHEMA_VERSION = "secva.nlu_candidate.v1"

TURN_LOG_PATH = config.DATA_DIR / "turn_log.jsonl"
NLU_CANDIDATES_PATH = config.DATA_DIR / "nlu_training_candidates.jsonl"

_LOCK = threading.Lock()
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)")
_SECRET_KEYS = {
    "password",
    "admin_password",
    "new_password",
    "token",
    "access_token",
    "refresh_token",
    "secret",
    "api_key",
    "authorization",
}
_FREE_TEXT_KEYS = {"body", "content", "message", "note", "notes"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def short_hash(value: Any) -> str:
    raw = str(value or "").encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()[:12]


def redact_text(text: Any, *, max_len: int = 300) -> str:
    """Redact obvious PII and keep bounded text for trainable command logs."""
    if text is None:
        return ""
    s = str(text)
    s = _EMAIL_RE.sub("<email>", s)
    s = _PHONE_RE.sub("<phone>", s)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def sanitize_value(key: str, value: Any) -> Any:
    lk = str(key).lower()
    if any(secret in lk for secret in _SECRET_KEYS):
        return "<redacted>"
    if lk in _FREE_TEXT_KEYS:
        return "<redacted_text>"
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return sanitize_dict(value)
    if isinstance(value, list):
        return [sanitize_value(key, item) for item in value[:20]]
    return value


def sanitize_dict(data: dict | None) -> dict:
    if not isinstance(data, dict):
        return {}
    return {str(k): sanitize_value(str(k), v) for k, v in data.items()}


def _as_dict(result: Any) -> dict:
    if result is None:
        return {}
    if is_dataclass(result):
        return asdict(result)
    if isinstance(result, dict):
        return dict(result)
    raise TypeError(f"Unsupported turn result type: {type(result)!r}")


def build_turn_event(result: Any, *, source: str,
                     request_id: str | None = None,
                     auth_method: str | None = None) -> dict:
    data = _as_dict(result)
    uid = data.get("identified_user_id")
    return {
        "schema_version": SCHEMA_VERSION,
        "ts": utc_now_iso(),
        "source": source,
        "request_id": request_id or "",
        "transcript": redact_text(data.get("transcript", "")),
        "intent": data.get("intent") or "unknown",
        "auth_level": data.get("auth_level") or "normal",
        "entities": sanitize_dict(data.get("entities") or {}),
        "identified_user_id_hash": short_hash(uid) if uid else "",
        "identified_user_name": redact_text(data.get("identified_user_name", "")),
        "sid_score": data.get("sid_score"),
        "sv_required": bool(data.get("sv_required", False)),
        "sv_passed": data.get("sv_passed"),
        "sv_score": data.get("sv_score"),
        "auth_method": auth_method or "",
        "blocked": bool(data.get("blocked", False)),
        "action_type": data.get("action_type") or "",
        "flow_active": bool(data.get("flow_active", False)),
        "label_source": "model_prediction",
    }


def build_nlu_candidate(turn_event: dict) -> dict:
    return {
        "schema_version": NLU_SCHEMA_VERSION,
        "ts": turn_event.get("ts", utc_now_iso()),
        "source": turn_event.get("source", ""),
        "request_id": turn_event.get("request_id", ""),
        "utterance": turn_event.get("transcript", ""),
        "predicted_intent": turn_event.get("intent", "unknown"),
        "predicted_entities": turn_event.get("entities", {}),
        "expected_intent": "",
        "expected_entities": {},
        "label_status": "unlabeled",
        "blocked": bool(turn_event.get("blocked", False)),
    }


def append_jsonl(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
    with _LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def log_turn(result: Any, *, source: str, request_id: str | None = None,
             auth_method: str | None = None,
             turn_path: Path = TURN_LOG_PATH,
             nlu_path: Path = NLU_CANDIDATES_PATH) -> dict:
    """Append a structured turn event and its NLU training candidate."""
    event = build_turn_event(result, source=source,
                             request_id=request_id,
                             auth_method=auth_method)
    if (os.environ.get("PYTEST_CURRENT_TEST")
            and turn_path == TURN_LOG_PATH
            and nlu_path == NLU_CANDIDATES_PATH):
        return event
    append_jsonl(turn_path, event)
    append_jsonl(nlu_path, build_nlu_candidate(event))
    return event
