"""Production resilience for LLM calls: retries with backoff, rate limiting, timeouts,
circuit breaker, fallbacks and caching.

Real LLM APIs fail in boring, predictable ways:
  * 429 Too Many Requests (rate limit, often with a Retry-After header)
  * 500/502/503/529 overloaded or transient server errors
  * timeouts / dropped connections
  * 400 bad request, 401 auth, context-length exceeded  -> NOT retryable (fix the request)

Everything here takes injectable `sleep` / `clock` functions so tests run instantly.
In production you may prefer `tenacity` (retries) and your SDK's built-in `max_retries`,
but you should understand what they do.
"""
from __future__ import annotations

import functools
import hashlib
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Callable, Iterable


# ----------------------------------------------------------------- error taxonomy
class LLMError(Exception):
    """Base class for provider errors (map your SDK's exceptions onto these)."""


class RateLimitError(LLMError):
    def __init__(self, msg: str = "429 rate limited", retry_after: float | None = None):
        super().__init__(msg)
        self.retry_after = retry_after


class ServerError(LLMError):
    """5xx / overloaded — transient."""


class LLMTimeoutError(LLMError):
    """The call took too long."""


class BadRequestError(LLMError):
    """4xx other than 429 (invalid params, context too long, auth). Never retry."""


class CircuitOpenError(LLMError):
    """Fail fast: the provider has been failing, we stopped calling it for a while."""


RETRYABLE: tuple[type[Exception], ...] = (RateLimitError, ServerError, LLMTimeoutError, ConnectionError)


# ----------------------------------------------------------------- backoff
def backoff_delays(max_retries: int, base: float = 1.0, cap: float = 30.0, jitter: bool = True,
                   rng: random.Random | None = None) -> list[float]:
    """Exponential backoff with *full jitter*: delay_i = uniform(0, min(cap, base * 2**i)).

    Jitter spreads retries out so 1,000 clients that failed together don't all retry together
    (the "thundering herd").
    """
    rng = rng or random.Random()
    out = []
    for i in range(max_retries):
        ceiling = min(cap, base * 2 ** i)
        out.append(rng.uniform(0, ceiling) if jitter else ceiling)
    return out


def retry(max_retries: int = 4, base: float = 1.0, cap: float = 30.0, jitter: bool = True,
          retry_on: tuple[type[Exception], ...] = RETRYABLE, sleep: Callable[[float], None] = time.sleep,
          on_retry: Callable[[int, Exception, float], None] | None = None, rng: random.Random | None = None):
    """Decorator: retry transient errors, honour Retry-After, never retry bad requests."""

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            delays = backoff_delays(max_retries, base, cap, jitter, rng)
            for attempt in range(max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except retry_on as e:
                    if attempt == max_retries:
                        raise
                    delay = delays[attempt]
                    if isinstance(e, RateLimitError) and e.retry_after is not None:
                        delay = max(delay, e.retry_after)  # the server told us how long to wait
                    if on_retry:
                        on_retry(attempt + 1, e, delay)
                    sleep(delay)
        return wrapper
    return deco


# ----------------------------------------------------------------- rate limiting
class TokenBucket:
    """Client-side rate limiter. `rate` tokens are added per second up to `capacity`.

    Use one bucket for requests/minute and another for tokens/minute (TPM) — providers
    enforce both. `acquire(cost)` blocks until enough tokens are available.
    """

    def __init__(self, rate: float, capacity: float, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.rate, self.capacity = rate, capacity
        self.tokens, self.clock, self.sleep = capacity, clock, sleep
        self.updated = clock()
        self.lock = threading.Lock()

    def _refill(self):
        now = self.clock()
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
        self.updated = now

    def try_acquire(self, cost: float = 1.0) -> bool:
        with self.lock:
            self._refill()
            if self.tokens >= cost:
                self.tokens -= cost
                return True
            return False

    def acquire(self, cost: float = 1.0) -> float:
        """Block until `cost` tokens are available. Returns total seconds waited."""
        if cost > self.capacity:
            raise ValueError("cost exceeds bucket capacity")
        waited = 0.0
        while not self.try_acquire(cost):
            with self.lock:
                need = (cost - self.tokens) / self.rate
            self.sleep(need)
            waited += need
        return waited


# ----------------------------------------------------------------- timeout
def call_with_timeout(fn: Callable, timeout: float, *args, **kwargs):
    """Run fn in a worker thread; raise LLMTimeoutError if it exceeds `timeout` seconds.

    (Prefer the SDK's own `timeout=` parameter when available — it actually cancels the HTTP
    request. This generic version only stops *waiting*.)
    """
    pool = ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(fn, *args, **kwargs)
    try:
        return fut.result(timeout=timeout)
    except FutureTimeout as e:
        raise LLMTimeoutError(f"call exceeded {timeout}s") from e
    finally:
        pool.shutdown(wait=False)


# ----------------------------------------------------------------- circuit breaker
class CircuitBreaker:
    """CLOSED -> (N consecutive failures) -> OPEN -> (cooldown) -> HALF_OPEN -> success: CLOSED / fail: OPEN."""

    def __init__(self, failure_threshold: int = 5, cooldown: float = 30.0, clock: Callable[[], float] = time.monotonic):
        self.failure_threshold, self.cooldown, self.clock = failure_threshold, cooldown, clock
        self.failures, self.opened_at, self.state = 0, 0.0, "closed"

    def call(self, fn: Callable, *args, **kwargs):
        if self.state == "open":
            if self.clock() - self.opened_at < self.cooldown:
                raise CircuitOpenError("circuit open — failing fast")
            self.state = "half_open"
        try:
            result = fn(*args, **kwargs)
        except Exception:
            self.failures += 1
            if self.state == "half_open" or self.failures >= self.failure_threshold:
                self.state, self.opened_at = "open", self.clock()
            raise
        self.failures, self.state = 0, "closed"
        return result


# ----------------------------------------------------------------- putting it together
class ResilientLLM:
    """Wrap one or more LLMs (primary first, then fallbacks) with cache, rate limit, timeout,
    retries and a circuit breaker per provider. Implements the same `complete(prompt)` interface.
    """

    def __init__(self, providers: Iterable, max_retries: int = 3, timeout: float | None = 60.0,
                 limiter: TokenBucket | None = None, cache: bool = True,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic,
                 breaker_threshold: int = 5, breaker_cooldown: float = 30.0):
        self.providers = list(providers)
        self.breakers = [CircuitBreaker(breaker_threshold, breaker_cooldown, clock) for _ in self.providers]
        self.max_retries, self.timeout, self.limiter, self.sleep = max_retries, timeout, limiter, sleep
        self.cache: dict[str, str] | None = {} if cache else None
        self.events: list[str] = []

    def _key(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()

    def complete(self, prompt: str) -> str:
        key = self._key(prompt)
        if self.cache is not None and key in self.cache:
            self.events.append("cache_hit")
            return self.cache[key]
        last_error: Exception | None = None
        for i, (llm, breaker) in enumerate(zip(self.providers, self.breakers)):
            @retry(self.max_retries, sleep=self.sleep, base=0.5,
                   on_retry=lambda n, e, d, i=i: self.events.append(f"retry p{i} #{n} {type(e).__name__}"))
            def attempt():
                if self.limiter:
                    self.limiter.acquire()
                call = (lambda: llm.complete(prompt))
                if self.timeout:
                    return breaker.call(call_with_timeout, call, self.timeout)
                return breaker.call(call)
            try:
                out = attempt()
                self.events.append(f"ok p{i}")
                if self.cache is not None:
                    self.cache[key] = out
                return out
            except BadRequestError:
                raise  # a fallback model won't fix a bad request
            except Exception as e:  # exhausted retries or circuit open -> next provider
                last_error = e
                self.events.append(f"fallback from p{i}: {type(e).__name__}")
        raise last_error or LLMError("no providers configured")
