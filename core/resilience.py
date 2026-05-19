"""Retry + Circuit breaker cho external dependency (Gmail, Gemini, OAuth...).

Không pull dependency mới (tenacity, pybreaker) — tự implement ngắn gọn vì:
- Logic đơn giản, ít state.
- Tránh thêm dep cho thesis project.
- Có thể migrate sang `tenacity` sau nếu cần.

Pattern dùng:
    @retry(max_attempts=3, backoff_base=0.5)
    def exchange_code(...): ...

    breaker = CircuitBreaker("gmail-send", fail_threshold=5, reset_after=60)
    @breaker
    @retry(max_attempts=3, backoff_base=0.5)
    def send_email(...): ...

Khi circuit open → call ngay lập tức raise `CircuitOpen`; tránh hammer dep
đang down + hold thread của app.
"""
from __future__ import annotations

import functools
import logging
import random
import threading
import time
from typing import Any, Callable, Type

_log = logging.getLogger("secva.resilience")


class CircuitOpen(Exception):
    """Raised khi circuit breaker đang OPEN — caller nên fail-fast."""


def retry(max_attempts: int = 3,
          backoff_base: float = 0.5,
          backoff_cap: float = 8.0,
          retriable: tuple[Type[BaseException], ...] = (Exception,),
          on_giveup: Callable[[BaseException], None] | None = None):
    """Decorator retry với exponential backoff + jitter.

    backoff_base * 2^attempt thêm jitter ±25%.
    Chỉ retry exception thuộc `retriable`; raise ngay nếu không match.
    `on_giveup` callback khi đã hết retry — log/audit.
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc: BaseException | None = None
            for attempt in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except CircuitOpen:
                    raise  # Không retry khi circuit đang open.
                except retriable as e:
                    last_exc = e
                    if attempt == max_attempts - 1:
                        break
                    sleep = min(backoff_base * (2 ** attempt), backoff_cap)
                    sleep *= 0.75 + random.random() * 0.5  # ±25% jitter
                    _log.info("retry %s attempt %d/%d after %.2fs: %s",
                              fn.__name__, attempt + 1, max_attempts, sleep, e)
                    time.sleep(sleep)
            if on_giveup and last_exc is not None:
                try:
                    on_giveup(last_exc)
                except Exception:
                    pass
            raise last_exc  # type: ignore[misc]
        return wrapper
    return decorator


class CircuitBreaker:
    """Circuit breaker đơn giản với 3 trạng thái: closed / open / half-open.

    - closed: normal — count failures.
    - open: trip khi >= `fail_threshold` failures consecutive; reject mọi call
      tới khi `reset_after` giây trôi qua.
    - half-open: cho 1 call thử nghiệm; pass → close, fail → open lại.
    """
    def __init__(self, name: str, fail_threshold: int = 5,
                 reset_after: float = 60.0):
        self.name = name
        self.fail_threshold = fail_threshold
        self.reset_after = reset_after
        self._lock = threading.Lock()
        self._fail_count = 0
        self._opened_at: float | None = None
        self._half_open_trial = False

    def _is_open(self) -> bool:
        if self._opened_at is None:
            return False
        if time.time() - self._opened_at >= self.reset_after:
            # Move to half-open: cho phép 1 trial.
            self._opened_at = None
            self._half_open_trial = True
            _log.info("circuit %s → half-open (allowing trial)", self.name)
            return False
        return True

    def _on_success(self):
        with self._lock:
            if self._half_open_trial:
                _log.info("circuit %s → closed (trial succeeded)", self.name)
                self._half_open_trial = False
            self._fail_count = 0
            self._opened_at = None

    def _on_failure(self):
        with self._lock:
            self._fail_count += 1
            if self._half_open_trial:
                _log.warning("circuit %s → open (trial failed)", self.name)
                self._opened_at = time.time()
                self._half_open_trial = False
                self._fail_count = self.fail_threshold
            elif self._fail_count >= self.fail_threshold:
                if self._opened_at is None:
                    _log.warning("circuit %s → open (%d failures)",
                                 self.name, self._fail_count)
                self._opened_at = time.time()

    def __call__(self, fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with self._lock:
                if self._is_open():
                    raise CircuitOpen(
                        f"circuit '{self.name}' is open — service is degraded")
            try:
                result = fn(*args, **kwargs)
            except Exception:
                self._on_failure()
                raise
            else:
                self._on_success()
                return result
        return wrapper
