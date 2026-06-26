import requests_cache
from ratelimit import limits, sleep_and_retry
from logger import log

# Use a specific CachedSession for this client, not a global patch
session = requests_cache.CachedSession('api_cache', expire_after=3600)

class APIClient:
    """
    A resilient API client with caching, rate limiting, and retries.
    """
    # Rate limit: 30 calls per minute
    CALLS = 30
    RATE_LIMIT = 60

    @staticmethod
    @sleep_and_retry
    @limits(calls=CALLS, period=RATE_LIMIT)
    def get(url, params=None, headers=None):
        """
        Perform a GET request with caching, rate limiting, and retries.
        """
        try:
            response = session.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Check if the response was from cache
            was_cached = getattr(response, 'from_cache', False)
            log.info(f"API request to {url} was {'CACHED' if was_cached else 'LIVE'}")
            
            return response
        except requests.exceptions.RequestException as e:
            log.error(f"API request to {url} failed: {e}")
            return None

client = APIClient()
