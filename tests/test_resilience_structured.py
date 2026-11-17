import random
import time

import pytest
from pydantic import BaseModel, Field

from ailearn.llm import ScriptedLLM
from ailearn.resilience import (BadRequestError, CircuitBreaker, CircuitOpenError, LLMTimeoutError,
                                RateLimitError, ResilientLLM, ServerError, TokenBucket, backoff_delays,
                                call_with_timeout, retry)
from ailearn.structured import StructuredOutputError, extract_json, generate_structured


class FakeClock:
    def __init__(self):
        self.t = 0.0
        self.sleeps = []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


class Flaky:
    """Raises the given errors in order, then returns 'ok'."""

    def __init__(self, *errors):
        self.errors, self.calls = list(errors), 0

    def complete(self, prompt):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return f"ok:{prompt}"


# ---------------- backoff & retry ----------------
def test_backoff_grows_exponentially_and_is_capped():
    assert backoff_delays(6, base=1, cap=10, jitter=False) == [1, 2, 4, 8, 10, 10]


def test_full_jitter_stays_within_ceiling():
    d = backoff_delays(5, base=1, cap=10, jitter=True, rng=random.Random(0))
    assert all(0 <= x <= min(10, 2 ** i) for i, x in enumerate(d))


def test_retry_recovers_from_transient_errors():
    clock, svc = FakeClock(), Flaky(ServerError(), RateLimitError())
    call = retry(max_retries=3, jitter=False, sleep=clock.sleep)(svc.complete)
    assert call("x") == "ok:x" and svc.calls == 3 and clock.sleeps == [1.0, 2.0]


def test_retry_honours_retry_after_header():
    clock, svc = FakeClock(), Flaky(RateLimitError(retry_after=7))
    retry(max_retries=2, jitter=False, sleep=clock.sleep)(svc.complete)("x")
    assert clock.sleeps == [7]


def test_bad_request_is_never_retried():
    clock, svc = FakeClock(), Flaky(BadRequestError("context length exceeded"))
    with pytest.raises(BadRequestError):
        retry(max_retries=5, sleep=clock.sleep)(svc.complete)("x")
    assert svc.calls == 1 and clock.sleeps == []


def test_retry_gives_up_after_max_retries():
    svc = Flaky(*[ServerError()] * 10)
    with pytest.raises(ServerError):
        retry(max_retries=2, sleep=lambda s: None)(svc.complete)("x")
    assert svc.calls == 3


# ---------------- rate limiting ----------------
def test_token_bucket_blocks_until_refilled():
    clock = FakeClock()
    bucket = TokenBucket(rate=2, capacity=2, clock=clock, sleep=clock.sleep)  # 2 req/s
    assert bucket.acquire() == 0 and bucket.acquire() == 0         # burst of 2
    waited = bucket.acquire()                                       # 3rd must wait 0.5s
    assert waited == pytest.approx(0.5)


def test_token_bucket_tpm_style_cost():
    clock = FakeClock()
    tpm = TokenBucket(rate=1000 / 60, capacity=1000, clock=clock, sleep=clock.sleep)  # 1000 tokens/min
    tpm.acquire(800)
    assert not tpm.try_acquire(800)
    tpm.acquire(800)
    assert clock.t == pytest.approx(600 / (1000 / 60))  # waited for 600 tokens to refill (36 s)
    with pytest.raises(ValueError):
        tpm.acquire(5000)


# ---------------- timeout & circuit breaker ----------------
def test_call_with_timeout():
    assert call_with_timeout(lambda: 42, 1.0) == 42
    with pytest.raises(LLMTimeoutError):
        call_with_timeout(lambda: time.sleep(0.5), 0.05)


def test_circuit_breaker_opens_then_half_opens():
    clock = FakeClock()
    cb = CircuitBreaker(failure_threshold=2, cooldown=10, clock=clock)
    boom = Flaky(*[ServerError()] * 3)
    for _ in range(2):
        with pytest.raises(ServerError):
            cb.call(boom.complete, "x")
    assert cb.state == "open"
    with pytest.raises(CircuitOpenError):          # fails fast, provider not called
        cb.call(boom.complete, "x")
    assert boom.calls == 2
    clock.t += 11                                  # cooldown passes -> half-open trial
    with pytest.raises(ServerError):
        cb.call(boom.complete, "x")
    assert cb.state == "open"
    clock.t += 11
    assert cb.call(boom.complete, "x") == "ok:x" and cb.state == "closed"


# ---------------- ResilientLLM ----------------
def test_resilient_llm_falls_back_and_caches():
    primary = Flaky(*[ServerError()] * 10)
    backup = Flaky()
    llm = ResilientLLM([primary, backup], max_retries=2, timeout=None, sleep=lambda s: None)
    assert llm.complete("hi") == "ok:hi"
    assert primary.calls == 3 and backup.calls == 1
    assert llm.complete("hi") == "ok:hi" and backup.calls == 1      # served from cache
    assert "cache_hit" in llm.events and any(e.startswith("fallback from p0") for e in llm.events)


def test_resilient_llm_does_not_fall_back_on_bad_request():
    backup = Flaky()
    llm = ResilientLLM([Flaky(BadRequestError()), backup], timeout=None, sleep=lambda s: None)
    with pytest.raises(BadRequestError):
        llm.complete("x")
    assert backup.calls == 0


# ---------------- structured output ----------------
class DQIssue(BaseModel):
    column: str
    issue_type: str = Field(pattern="^(NULLS|DUPLICATES|SCHEMA_DRIFT|OUTLIER|FRESHNESS)$")
    severity: int = Field(ge=1, le=5)


@pytest.mark.parametrize("raw", [
    '{"a": 1}', 'Sure! Here you go:\n```json\n{"a": 1}\n```', 'Answer: {"a": 1} hope that helps'])
def test_extract_json(raw):
    assert extract_json(raw) == '{"a": 1}'


def test_structured_output_first_try():
    llm = ScriptedLLM(['{"column": "customer_id", "issue_type": "NULLS", "severity": 3}'])
    out = generate_structured(llm, "customer_id is empty in 4% rows", DQIssue)
    assert out == DQIssue(column="customer_id", issue_type="NULLS", severity=3)
    assert '"severity"' in llm.calls[0]                         # schema was included in the prompt


def test_structured_output_retries_with_validation_error():
    llm = ScriptedLLM(['{"column": "amount", "issue_type": "WEIRD", "severity": 9}',
                       '```json\n{"column": "amount", "issue_type": "OUTLIER", "severity": 4}\n```'])
    out = generate_structured(llm, "amount has 9,999,999 values", DQIssue)
    assert out.issue_type == "OUTLIER"
    assert "Validation errors" in llm.calls[1] and "severity" in llm.calls[1]


def test_structured_output_gives_up():
    llm = ScriptedLLM(["not json"] * 3)
    with pytest.raises(StructuredOutputError) as e:
        generate_structured(llm, "x", DQIssue, max_retries=2)
    assert len(e.value.attempts) == 3
