"""Shared retry and circuit-breaker utilities for the stock market bot."""

import time
import logging
from functools import wraps

log = logging.getLogger("stock_analyzer")


def retry(max_attempts=3, backoff_factor=2, retryable_exceptions=(Exception,)):
    """Decorator for retrying a function with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exc = e
                    if attempt == max_attempts - 1:
                        raise
                    wait = backoff_factor ** attempt
                    log.warning(
                        f"{func.__name__} attempt {attempt + 1}/{max_attempts} failed: {e}. "
                        f"Retrying in {wait}s..."
                    )
                    time.sleep(wait)
            raise last_exc  # Should not reach here
        return wrapper
    return decorator


class CircuitBreaker:
    """Simple circuit breaker to avoid hammering failing APIs.
    
    Usage:
        breaker = CircuitBreaker(failure_threshold=5, reset_timeout=300)
        for ticker in tickers:
            if breaker.is_open:
                log.error("Circuit breaker tripped, skipping remaining calls")
                break
            try:
                data = fetch_data(ticker)
                breaker.record_success()
            except RequestException:
                breaker.record_failure()
    """
    
    def __init__(self, failure_threshold=5, reset_timeout=300):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self._consecutive_failures = 0
        self._last_failure_time = 0
    
    @property
    def is_open(self):
        if self._consecutive_failures >= self.failure_threshold:
            if time.time() - self._last_failure_time > self.reset_timeout:
                self._consecutive_failures = 0
                return False
            return True
        return False
    
    def record_success(self):
        self._consecutive_failures = 0
    
    def record_failure(self):
        self._consecutive_failures += 1
        self._last_failure_time = time.time()
