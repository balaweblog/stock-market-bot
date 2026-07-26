"""
test_llm_backend.py

Standalone smoke test for every tier in llm_backend.py's fallback chain.
Run this from the same directory as llm_backend.py / main.py (so it can
import them and reuse your real env vars / config).

    python test_llm_backend.py

Each check is deliberately tiny (max_tokens=20-50, one short prompt) so
this costs almost nothing against any free-tier quota. It does NOT go
through generate_analysis()'s chain logic -- it calls each tier directly
so a failure in tier 2 doesn't stop tier 3 from being tested.
"""

import os
import sys

import llm_backend as lb


def _status(ok, label, detail=""):
    mark = "PASS" if ok else "FAIL"
    line = f"[{mark}] {label}"
    if detail:
        line += f" -- {detail}"
    print(line)


def check_env():
    print("\n--- Env vars detected ---")
    for key in ("GROQ_API_KEY", "GOOGLE_API_KEY", "MISTRAL_API_KEY", "TAVILY_API_KEY"):
        present = bool(os.getenv(key))
        _status(present, key, "set" if present else "not set -- that tier will be skipped")


def check_groq_compound():
    if not os.getenv("GROQ_API_KEY"):
        _status(False, "groq/compound", "skipped, no GROQ_API_KEY")
        return
    lb.init_llm_generator()
    result = lb._try_groq_compound_model(
        "Reply with exactly the word: PONG", "groq/compound",
        max_attempts=1, max_tokens=20, log_label="test",
    )
    if result is None:
        _status(False, "groq/compound", "call failed -- see log line above for the real error")
    else:
        text, sources, _live = result
        _status(True, "groq/compound", f"replied: {text!r}")


def check_groq_compound_mini():
    if not os.getenv("GROQ_API_KEY"):
        _status(False, "groq/compound-mini", "skipped, no GROQ_API_KEY")
        return
    result = lb._try_groq_compound_model(
        "Reply with exactly the word: PONG", "groq/compound-mini",
        max_attempts=1, max_tokens=20, log_label="test",
    )
    if result is None:
        _status(False, "groq/compound-mini", "call failed -- see log line above")
    else:
        text, sources, _live = result
        _status(True, "groq/compound-mini", f"replied: {text!r}")


def check_synthesis_models():
    if not os.getenv("GROQ_API_KEY"):
        _status(False, "synthesis models (plain Groq)", "skipped, no GROQ_API_KEY")
        return
    # Tests each model in SYNTHESIS_MODELS individually so a dead model
    # (like the old qwen3-32b) shows up on its own line instead of being
    # silently swallowed by the fallback loop.
    for model_name in lb.SYNTHESIS_MODELS:
        try:
            response = lb.groq_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "Reply with exactly the word: PONG"}],
                temperature=0.4,
                max_tokens=20,
            )
            text = response.choices[0].message.content.strip()
            _status(True, f"synthesis model: {model_name}", f"replied: {text!r}")
        except Exception as e:
            _status(False, f"synthesis model: {model_name}", str(e))


def check_gemini():
    if not os.getenv("GOOGLE_API_KEY"):
        _status(False, "gemini grounded", "skipped, no GOOGLE_API_KEY")
        return
    result = lb._try_gemini_grounded("Reply with exactly the word: PONG", log_label="test")
    if result is None:
        _status(False, "gemini grounded", "call failed -- see log line above")
    else:
        text, sources, _live = result
        _status(True, "gemini grounded", f"replied: {text!r}")


def check_mistral():
    if not os.getenv("MISTRAL_API_KEY"):
        _status(False, "mistral web_search", "skipped, no MISTRAL_API_KEY")
        return
    result = lb._try_mistral_web_search("Reply with exactly the word: PONG", max_tokens=50, log_label="test")
    if result is None:
        _status(False, "mistral web_search", "call failed -- see log line above (also check `pip install mistralai`)")
    else:
        text, sources, _live = result
        _status(True, "mistral web_search", f"replied: {text!r}")


def check_tavily():
    if not os.getenv("TAVILY_API_KEY"):
        _status(False, "tavily search", "skipped, no TAVILY_API_KEY")
        return
    import requests
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": os.getenv("TAVILY_API_KEY"), "query": "test", "max_results": 1},
            timeout=10,
        )
        resp.raise_for_status()
        _status(True, "tavily search", f"HTTP {resp.status_code}")
    except Exception as e:
        _status(False, "tavily search", str(e))


def _print_ratelimit_headers(headers, label):
    """Print any header whose name contains 'ratelimit' or 'rate-limit'."""
    found = False
    for k, v in headers.items():
        if "ratelimit" in k.lower().replace("-", ""):
            print(f"    {k}: {v}")
            found = True
    if not found:
        print(f"    (no rate-limit headers returned by {label})")


def check_groq_quota():
    """Groq puts remaining TPM/RPD directly on every response's headers --
    no separate quota endpoint exists, so we make one tiny call and read
    the headers off it instead of consuming a whole request just to ask."""
    if not os.getenv("GROQ_API_KEY"):
        _status(False, "groq quota", "skipped, no GROQ_API_KEY")
        return
    lb.init_llm_generator()
    try:
        model = lb.SYNTHESIS_MODELS[0] if getattr(lb, "SYNTHESIS_MODELS", None) else "llama-3.1-8b-instant"
        raw = lb.groq_client.chat.completions.with_raw_response.create(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )
        headers = raw.headers
        limit_req = headers.get("x-ratelimit-limit-requests")
        rem_req = headers.get("x-ratelimit-remaining-requests")
        limit_tok = headers.get("x-ratelimit-limit-tokens")
        rem_tok = headers.get("x-ratelimit-remaining-tokens")
        reset_req = headers.get("x-ratelimit-reset-requests")
        reset_tok = headers.get("x-ratelimit-reset-tokens")
        _status(True, "groq quota", f"model={model}")
        print(f"    requests/day  -- limit: {limit_req}  remaining: {rem_req}  resets in: {reset_req}")
        print(f"    tokens/minute -- limit: {limit_tok}  remaining: {rem_tok}  resets in: {reset_tok}")
        print("    note: Groq does not expose the daily token cap (TPD) in headers for all plans/models;")
        print("    check console.groq.com/settings/limits for the full picture.")
    except Exception as e:
        _status(False, "groq quota", str(e))


def check_mistral_quota():
    """Mistral's documented signal is the X-RateLimit-Remaining header on a
    normal request -- there's no dedicated usage endpoint on the free tier.
    Header presence has been inconsistent across accounts, so we print
    whatever we get back and fall back to pointing at the dashboard."""
    if not os.getenv("MISTRAL_API_KEY"):
        _status(False, "mistral quota", "skipped, no MISTRAL_API_KEY")
        return
    import requests
    try:
        resp = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('MISTRAL_API_KEY')}",
                "Content-Type": "application/json",
            },
            json={
                "model": "mistral-small-latest",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
            },
            timeout=15,
        )
        _status(True, "mistral quota", f"HTTP {resp.status_code}")
        _print_ratelimit_headers(resp.headers, "Mistral")
        print("    if nothing printed above: Mistral doesn't reliably expose remaining")
        print("    tokens/requests via headers on all accounts -- check console.mistral.ai/usage")
    except Exception as e:
        _status(False, "mistral quota", str(e))


def check_tavily_quota():
    """Tavily has an actual GET /usage endpoint that returns key + account
    credit usage/limits directly -- this is the one provider where we don't
    have to infer anything from headers."""
    if not os.getenv("TAVILY_API_KEY"):
        _status(False, "tavily quota", "skipped, no TAVILY_API_KEY")
        return
    import requests
    try:
        resp = requests.get(
            "https://api.tavily.com/usage",
            headers={"Authorization": f"Bearer {os.getenv('TAVILY_API_KEY')}"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        _status(True, "tavily quota", f"HTTP {resp.status_code}")
        print(f"    {data}")
    except Exception as e:
        _status(False, "tavily quota", str(e))


def check_gemini_quota():
    """Google doesn't return remaining-quota headers on Gemini API responses
    and there's no equivalent GET /usage call -- free-tier quota (RPD/RPM/TPM
    per model) is only visible in the AI Studio / Cloud Console dashboards."""
    if not os.getenv("GOOGLE_API_KEY"):
        _status(False, "gemini quota", "skipped, no GOOGLE_API_KEY")
        return
    _status(False, "gemini quota", "not exposed via API")
    print("    check remaining Gemini free-tier quota at https://aistudio.google.com/usage")
    print("    or https://console.cloud.google.com/apis/api/generativelanguage.googleapis.com/quotas")


def check_local_model():
    backend = lb.init_llm_generator(force_local=True)
    if backend != "local" or lb.llm_pipeline is None:
        _status(False, "local Qwen2.5-1.5B", "not available (transformers not installed, or model failed to load)")
        return
    text = lb._generate_local("Reply with exactly the word: PONG", max_new_tokens=20, log_label="test")
    _status(bool(text), "local Qwen2.5-1.5B", f"replied: {text!r}" if text else "generation returned nothing")


if __name__ == "__main__":
    check_env()
    print("\n--- Live tier checks ---")
    check_groq_compound()
    check_groq_compound_mini()
    check_synthesis_models()
    check_gemini()
    check_mistral()
    check_tavily()
    print("\n--- Quota / usage remaining ---")
    check_groq_quota()
    check_mistral_quota()
    check_tavily_quota()
    check_gemini_quota()
    print("\n--- Non-live fallback ---")
    if os.getenv("REQUIRE_LIVE_DATA", "true").lower() != "false":
        print("[SKIP] local model -- set REQUIRE_LIVE_DATA=false to test this tier too")
    else:
        check_local_model()
    print("\nDone.")