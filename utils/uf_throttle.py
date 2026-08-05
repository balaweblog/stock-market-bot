"""
Centralized yfinance rate-limiting helpers.

Every module in this codebase that talks to Yahoo Finance via yfinance --
controllers.stock_controller.fetch_data, services.stock_fetcher's
fetch_fundamentals / fetch_advanced_fundamentals / fetch_ownership_activity,
and models.market_context's fetch_index_context / build_market_context --
was previously throttling and retrying independently, each with its own
requests.Session and its own (or no) backoff. Because Yahoo Finance's rate
limiter is keyed off request *rate from this process/IP as a whole*, several
uncoordinated streams of requests still trip it even when each stream looks
well-behaved in isolation -- which is exactly what the logs showed: one
module's calls succeeding while another's failed in the same second.

This module gives every caller, in every thread, one shared clock and one
shared HTTP session:

  1. throttle() enforces a minimum spacing between *any* two yfinance HTTP
     calls anywhere in the process, regardless of which module/thread
     issues them.
  2. get_shared_session() returns a single, lazily-created, reused
     requests.Session so yfinance negotiates its cookie/crumb once instead
     of every fresh session forcing a renegotiation (itself an extra
     request that can get rate-limited).
  3. call_with_retries() gives a rate-limit-aware backoff: a 429 means the
     whole IP is currently blocked, not just this one call, so it backs off
     much harder than a generic transient error and pushes every other
     in-flight caller's next allowed call time out too.
"""
import os
import random
import threading
import time

import requests

try:
    from yfinance.exceptions import YFRateLimitError
except ImportError:
    # Older/newer yfinance versions may not expose this exact class -- fall
    # back to a sentinel that never matches isinstance(), so the rate-limit
    # handling below just degrades to string-matching the error message.
    class YFRateLimitError(Exception):
        pass

# Minimum seconds between any two yfinance HTTP calls, enforced across every
# thread and every calling module. Override via env var if a given
# environment (e.g. CI runners sharing a noisier IP pool) needs more room.
YF_MIN_INTERVAL_SECONDS = float(os.environ.get("YF_MIN_INTERVAL_SECONDS", "2.5"))

_throttle_lock = threading.Lock()
_last_call_time = [0.0]

_session_lock = threading.Lock()
_session = None


def get_shared_session():
    """Return a single, process-wide requests.Session for all yfinance
    calls. Created once and reused so yfinance doesn't have to renegotiate
    a cookie/crumb on every call -- previously each module (and, in
    market_context's case, each individual call) created a brand new
    Session with no cookie continuity at all."""
    global _session
    with _session_lock:
        if _session is None:
            _session = requests.Session()
            _session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;"
                          "q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Connection": "keep-alive",
            })
        return _session


def throttle():
    """Block until at least YF_MIN_INTERVAL_SECONDS have passed since the
    last yfinance call anywhere in the process, then reserve this slot.
    Call this immediately before every direct yfinance HTTP-triggering
    call (yf.download(...), ticker.info, ticker.institutional_holders,
    etc.) -- or just use call_with_retries(), which does it for you."""
    with _throttle_lock:
        now = time.monotonic()
        wait = _last_call_time[0] + YF_MIN_INTERVAL_SECONDS - now
        if wait > 0:
            time.sleep(wait)
        _last_call_time[0] = time.monotonic()


def push_throttle(seconds):
    """Force the next call (from any thread, any module) to wait at least
    `seconds` from now. Used after hitting a 429 so every other in-flight
    caller backs off too, instead of only the one that happened to hit the
    error while the others keep hammering the same blocked IP."""
    with _throttle_lock:
        candidate = time.monotonic() + seconds
        if candidate > _last_call_time[0]:
            _last_call_time[0] = candidate


def is_rate_limit_error(exc):
    return (
        isinstance(exc, YFRateLimitError)
        or "429" in str(exc)
        or "Too Many Requests" in str(exc)
        or "Rate limited" in str(exc)
    )


def call_with_retries(func, *args, max_attempts=5, **kwargs):
    """Call func(*args, **kwargs), applying the shared throttle before
    every attempt and a rate-limit-aware backoff between retries. Raises
    the last exception if every attempt fails."""
    last_exc = None
    for attempt in range(max_attempts):
        throttle()
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt == max_attempts - 1:
                raise
            if is_rate_limit_error(exc):
                # The whole IP is currently rate-limited -- a short backoff
                # just re-trips it. Back off hard and hold every other
                # in-flight caller back too.
                sleep_for = (15 * (attempt + 1)) + random.uniform(0, 5)
                push_throttle(sleep_for)
            else:
                sleep_for = (2 ** attempt) + random.uniform(0, 1)
            print(f"yfinance call failed (attempt {attempt + 1}/{max_attempts}): "
                  f"{last_exc!r}, retrying in {sleep_for:.1f}s")
            time.sleep(sleep_for)
    raise last_exc