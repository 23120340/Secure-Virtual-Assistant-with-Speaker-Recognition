"""Test retry + circuit breaker module."""
import pytest

from core.resilience import CircuitBreaker, CircuitOpen, retry


def test_retry_succeeds_eventually():
    """Function fail 2 lần đầu, success lần 3 → retry phải trả về kết quả."""
    counter = {"n": 0}

    @retry(max_attempts=3, backoff_base=0.001)
    def flaky():
        counter["n"] += 1
        if counter["n"] < 3:
            raise ValueError("transient")
        return "ok"

    assert flaky() == "ok"
    assert counter["n"] == 3


def test_retry_gives_up_after_max_attempts():
    counter = {"n": 0}

    @retry(max_attempts=3, backoff_base=0.001)
    def always_fail():
        counter["n"] += 1
        raise ValueError("permanent")

    with pytest.raises(ValueError):
        always_fail()
    assert counter["n"] == 3


def test_retry_skips_non_retriable():
    """Exception không thuộc `retriable` → fail-fast, không retry."""
    counter = {"n": 0}

    @retry(max_attempts=5, backoff_base=0.001, retriable=(ValueError,))
    def fail_with_type():
        counter["n"] += 1
        raise TypeError("not retriable")

    with pytest.raises(TypeError):
        fail_with_type()
    assert counter["n"] == 1


def test_circuit_breaker_opens_after_threshold():
    cb = CircuitBreaker("test", fail_threshold=3, reset_after=10.0)

    @cb
    def fail():
        raise RuntimeError("boom")

    # 3 failures liên tiếp
    for _ in range(3):
        with pytest.raises(RuntimeError):
            fail()
    # Lần 4: circuit open, fail-fast
    with pytest.raises(CircuitOpen):
        fail()


def test_circuit_breaker_half_open_recovers():
    import time
    cb = CircuitBreaker("test", fail_threshold=2, reset_after=0.05)

    @cb
    def maybe_fail(should_fail):
        if should_fail:
            raise RuntimeError("boom")
        return "ok"

    # Trip circuit
    for _ in range(2):
        with pytest.raises(RuntimeError):
            maybe_fail(True)
    with pytest.raises(CircuitOpen):
        maybe_fail(True)

    # Wait for reset_after → half-open
    time.sleep(0.06)
    # Trial call: success → close
    assert maybe_fail(False) == "ok"
    # Sau khi close, calls bình thường
    assert maybe_fail(False) == "ok"


def test_circuit_breaker_reopens_if_trial_fails():
    import time
    cb = CircuitBreaker("test", fail_threshold=2, reset_after=0.05)

    @cb
    def fail():
        raise RuntimeError("boom")

    for _ in range(2):
        with pytest.raises(RuntimeError):
            fail()
    with pytest.raises(CircuitOpen):
        fail()
    time.sleep(0.06)
    # Trial fail → re-open
    with pytest.raises(RuntimeError):
        fail()
    with pytest.raises(CircuitOpen):
        fail()
