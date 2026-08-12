"""
llm_backend.py

Single shared LLM backend + fallback chain for the whole project. Used by
main.py (AI Stocks Story), swing_trade_advisor.py (swing-trade generation),
and optionstrategy.py (indirectly, via swing_trade_advisor.generate_analysis).

WHY THIS EXISTS
----------------
Before this module, main.py and swing_trade_advisor.py each carried their
own copy of: model init, retry/backoff, error classification, and the
live-search fallback chain. They drifted:
  - main.py's AI Stocks Story chain hardcoded a single synthesis model
    (llama-3.3-70b-versatile) with no retry list and had NO Mistral tier.
  - swing_trade_advisor.py's chain tried a 3-model SYNTHESIS_MODELS list
    that included `qwen/qwen3-32b`, which 404s on standard Groq accounts
    (confirmed in production logs) -- a dead tier hit on every single run.
  - Retry-attempt counts for groq/compound vs compound-mini differed
    between the two scripts for no functional reason.
Consolidating here means one fix applies everywhere, one model list, one
retry policy, and no more silently-dead tiers.

FALLBACK CHAIN (all live-search paths tried before any non-live generation)
----------------------------------------------------------------------------
  1. groq/compound        -- Groq's tool-using model, autonomous web search.
  2. groq/compound-mini    -- lighter variant, tried if #1 is unavailable.
  3. Tavily direct search + plain Groq synthesis (own quota, no compound
     orchestration overhead) -- only if the caller supplies a
     `gather_context_fn`. Synthesis tries SYNTHESIS_MODELS in order, since
     Groq's daily token quota (TPD) is tracked per model, not per account.
  4. Gemini + Google Search grounding (own free quota).
  5. Mistral's web_search agent (own free quota; requires
     `pip install mistralai`) -- tried regardless of which primary backend
     was selected, since it shares no quota with anything above.
Only if every one of those fails, and only if REQUIRE_LIVE_DATA=false (the
project default is "true"), does this fall through to non-live plain-Groq
generation (no local-model tier -- see "Removed tiers" below).

The list above is the order used when nothing is known about current
quota yet (e.g. the first stock of a batch run). From the 2nd call
onward in the same process, steps 1-2 and the SYNTHESIS_MODELS list
inside step 3 are re-ordered (and thin/exhausted models skipped
outright) based on real remaining-tokens/remaining-requests headers
captured off every actual Groq call, and step 3 as a whole is skipped
if a cached Tavily /usage check shows no credits left. This is quota-
adaptive routing, not a fixed sequence -- see "Live quota tracking"
below _try_groq_compound_model's helpers for how it works. Gemini has
no equivalent signal (no headers, no usage endpoint) so it stays
purely reactive: tried in place, falls through only on an actual error.

Callers own their own prompt construction, response parsing, and
domain-specific context gathering (e.g. which Tavily queries to run) --
this module only owns "which model, in which order, with which
retry/backoff policy, and which error is worth retrying at all."

REMOVED TIERS (financial-analysis fit / dead-code cleanup)
------------------------------------------------------------
  - Local Qwen2.5-1.5B-Instruct fallback: dropped. A 1.5B local model has
    no web-search access and is unreliable at numeric/financial reasoning
    (stock prices, mutual-fund NAVs, ratios) -- not fit for this project's
    domain even as a last resort. It was also unreachable in practice:
    REQUIRE_LIVE_DATA defaults to "true", and generate_analysis() returns
    before ever reaching a local-model call when that's set (a stale,
    non-live answer for stock/fund analysis is worse than no answer).
    Removing it also drops the transformers/torch dependency for a path
    that never fired.
  - llama-3.1-8b-instant dropped from SYNTHESIS_MODELS: an 8B model is
    meaningfully weaker than the 70B tier at numeric/financial reasoning
    and more prone to inventing figures. Unlike the compound/Gemini/
    Mistral tiers, it isn't a quota-independent path (same Groq account,
    just a smaller model), so it added hallucination risk without adding
    real resilience.
  - openai / anthropic imports removed: never referenced anywhere in this
    module -- dead imports.
"""

import os
import re
import time
import json
import threading

try:
    from groq import Groq
except ImportError:
    Groq = None
try:
    from google import genai
except ImportError:
    genai = None

from utils.logger import log


# -----------------------------------------------------------------------
# Small shared config helper (previously duplicated in swing_trade_advisor.py)
# -----------------------------------------------------------------------
def _env_int(name, default):
    """
    Parses an integer env var, falling back to `default` (and logging a
    warning) if it's unset, empty, or not a valid integer -- so a typo'd
    workflow-yaml value can't crash a script at import time before logging
    is even configured.
    """
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.strip())
    except ValueError:
        print(f"WARNING: env var {name}='{raw}' is not a valid integer -- using default {default}.")
        return default


# -----------------------------------------------------------------------
# Model tiers -- override via env var without touching code
# -----------------------------------------------------------------------
GROQ_COMPOUND_MODELS = ["groq/compound", "groq/compound-mini"]

# Attempts per compound tier before moving on. Standardized to 2/2 across
# every caller (previously 3/2 in swing_trade_advisor.py and 2/2 in
# main.py) -- a 3rd attempt inside the same minute almost never clears a
# TPM limit (see the TPM math in the retry-delay log lines) and just burns
# more wall-clock time before falling through to a tier that can actually
# succeed.
GROQ_COMPOUND_ATTEMPTS = _env_int("GROQ_COMPOUND_ATTEMPTS", 2)
GROQ_COMPOUND_MINI_ATTEMPTS = _env_int("GROQ_COMPOUND_MINI_ATTEMPTS", 2)

# Models tried (in order) for the Tavily-context synthesis tier and for the
# final non-live Groq fallback. Deliberately excludes:
#   - qwen/qwen3-32b: 404s ("does not exist or you do not have access to
#     it") on standard Groq accounts -- confirmed dead in production logs,
#     so it's dropped rather than burning a retry slot on every run.
#   - openai/gpt-oss-120b: ships with a built-in "browser" tool that can
#     fire on its own (tool_use_failed) even with no tools passed, so it
#     isn't reliable for plain synthesis calls. Add it back explicitly via
#     SYNTHESIS_MODELS if you want to try it anyway.
#   - llama-3.1-8b-instant: dropped as of this revision -- an 8B model is
#     too weak at numeric/financial reasoning for stock/mutual-fund
#     analysis and shares the 70B tier's Groq account rather than adding
#     an independent quota path. Add it back via SYNTHESIS_MODELS if a
#     future caller has a lower-stakes, non-financial use for it.
DEFAULT_SYNTHESIS_MODELS = "llama-3.3-70b-versatile"
SYNTHESIS_MODELS = [
    m.strip() for m in os.getenv("SYNTHESIS_MODELS", DEFAULT_SYNTHESIS_MODELS).split(",")
    if m.strip()
]

GEMINI_MODEL = "gemini-flash-latest"
MISTRAL_MODEL = "mistral-medium-latest"

# Hard ceiling on any single call's max_tokens, regardless of what a caller
# requests -- keeps a bug in one caller's token-budget math from turning
# into an oversized (and wastefully expensive) request. Callers should
# still pass their own tighter budget; this is just a backstop.
MAX_TOKENS_CEILING = 4200


# -----------------------------------------------------------------------
# Shared client state (mirrors what main.py used to hold module-globally)
# -----------------------------------------------------------------------
model_lock = threading.Lock()
use_gemini_flash = False
gemini_client = None
use_groq = False
groq_client = None


# -----------------------------------------------------------------------
# Live quota tracking -- lets the chain route to whichever tier actually
# has headroom left *right now* instead of always trying tiers in the
# same hardcoded order and discovering mid-batch that #1 is out of tokens.
#
# HOW IT WORKS
#   Groq and Mistral both return remaining-tokens/requests headers on
#   every real response, for free -- so instead of a separate "probe"
#   call (which would itself burn quota just to ask about quota), every
#   real call captures its own headers and updates this module-global
#   state for the *next* call to consult. That means:
#     - The 1st call to a given model in a process still goes in the
#       documented default order (no data yet -- treated as "unknown",
#       not "exhausted").
#     - From the 2nd call onward in the same run (e.g. stock #2 of a
#       50-stock batch), a tier that's already thin gets skipped or
#       deprioritized based on what the *previous* call actually saw,
#       instead of being retried and failing again.
#     - Data older than its own reported reset window is treated as
#       unknown again (the bucket has almost certainly refilled),
#       rather than trusted indefinitely.
#   Gemini has no equivalent signal (no headers, no usage endpoint) so
#   it stays purely reactive -- try it, and only fall through on a real
#   error. Tavily has a real GET /usage endpoint but no per-response
#   headers, so it's checked at most once every few minutes rather than
#   on every call, and the local estimate is decremented between checks.
# -----------------------------------------------------------------------
_quota_lock = threading.Lock()
_groq_quota = {}   # model_name -> {"remaining_tokens", "remaining_requests", "updated_at"}
_tavily_quota = {"remaining_credits": None, "updated_at": None}

# A tier reporting fewer tokens than this is treated as having no real
# headroom even if technically nonzero -- avoids picking a tier that has
# just enough left for *this* call but nothing for the next stock in the
# same batch, which would just move the 429 one call later.
_QUOTA_SAFETY_MARGIN_TOKENS = 500

# Tavily's /usage costs an HTTP round-trip with no telemetry benefit
# beyond the check itself, so it's cached rather than called every time
# generate_analysis() considers the Tavily tier.
_TAVILY_QUOTA_REFRESH_SECONDS = 300
_QUOTA_CACHE_FILE = "quota_cache.json"


def _load_quota_cache():
    try:
        if os.path.exists(_QUOTA_CACHE_FILE):
            with open(_QUOTA_CACHE_FILE, "r") as f:
                data = json.load(f)
                with _quota_lock:
                    _groq_quota.update(data.get("groq", {}))
                    _tavily_quota.update(data.get("tavily", {}))
    except Exception as exc:
        log.debug(f"Could not load quota cache: {exc}")


def _save_quota_cache():
    try:
        with _quota_lock:
            payload = {"groq": _groq_quota, "tavily": _tavily_quota}
        with open(_QUOTA_CACHE_FILE, "w") as f:
            json.dump(payload, f)
    except Exception as exc:
        log.debug(f"Could not save quota cache: {exc}")


# Load cached quota state from disk on import so process restarts inherit headroom knowledge
_load_quota_cache()


def _record_groq_headers(headers, model_name):
    """Pulls remaining-tokens/requests off a raw Groq response and stashes
    them so the *next* call routing decision can see them."""
    if not headers:
        return
    try:
        rem_tok = headers.get("x-ratelimit-remaining-tokens")
        rem_req = headers.get("x-ratelimit-remaining-requests")
        with _quota_lock:
            _groq_quota[model_name] = {
                "remaining_tokens": int(rem_tok) if rem_tok is not None else None,
                "remaining_requests": int(rem_req) if rem_req is not None else None,
                "updated_at": time.time(),
            }
        _save_quota_cache()
    except Exception as e:
        log.warning(f"Could not record Groq quota headers for {model_name}: {e}")


def _groq_headroom(model_name, needed_tokens):
    """
    True/False/None for whether `model_name` looks able to serve a call
    needing ~needed_tokens right now, based on the last real response
    from that model this run:
      True  -- last-known remaining tokens/requests comfortably cover it
      False -- last-known remaining tokens/requests do NOT cover it
      None  -- no (fresh) data -- caller should still try it
    TPM windows are under a minute, so data older than 65s is treated as
    unknown rather than trusted -- the bucket has likely already reset.
    """
    with _quota_lock:
        info = _groq_quota.get(model_name)
    if not info:
        return None
    if time.time() - info["updated_at"] > 65:
        return None
    rem_tok, rem_req = info.get("remaining_tokens"), info.get("remaining_requests")
    if rem_req is not None and rem_req <= 0:
        return False
    if rem_tok is not None and rem_tok < needed_tokens + _QUOTA_SAFETY_MARGIN_TOKENS:
        return False
    return True


def _order_by_headroom(model_names, needed_tokens):
    """
    Sorts model_names so ones with known headroom go first, ones with
    known exhaustion go last, and ones with no data yet keep their
    original relative order in the middle. This is what makes the chain
    "route based on pending tokens" rather than always trying the same
    model first: if compound-mini currently has more headroom than
    compound, it gets tried first.
    """
    def _key(name):
        headroom = _groq_headroom(name, needed_tokens)
        return {True: 0, None: 1, False: 2}[headroom]
    return sorted(model_names, key=_key)


def _tavily_remaining_credits():
    """
    Best-effort remaining-credit count for Tavily's free tier. Cached for
    _TAVILY_QUOTA_REFRESH_SECONDS so checking quota doesn't itself become
    an extra network call on every single stock in a batch. Returns None
    (treated as "assume available") if there's no key, the check fails,
    or nothing's been checked yet this run.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return None
    with _quota_lock:
        cached = _tavily_quota["remaining_credits"]
        fresh = _tavily_quota["updated_at"] and (time.time() - _tavily_quota["updated_at"] < _TAVILY_QUOTA_REFRESH_SECONDS)
        if fresh:
            return cached
    try:
        import requests
        resp = requests.get(
            "https://api.tavily.com/usage",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        resp.raise_for_status()
        account = resp.json().get("account", {})
        plan_limit, plan_usage = account.get("plan_limit"), account.get("plan_usage")
        remaining = (plan_limit - plan_usage) if (plan_limit is not None and plan_usage is not None) else None
        with _quota_lock:
            _tavily_quota["remaining_credits"] = remaining
            _tavily_quota["updated_at"] = time.time()
        _save_quota_cache()
        return remaining
    except Exception as e:
        log.warning(f"Could not refresh Tavily quota: {e}")
        return None


def init_llm_generator():
    """
    Initializes whichever LLM backend is available.
    Priority: Groq (free tier) if GROQ_API_KEY is present, else Gemini
    Flash if GOOGLE_API_KEY is present. Returns None if neither key is
    configured -- there is no local-model fallback (see module docstring,
    "Removed tiers": a local 1.5B model isn't fit for stock/mutual-fund
    analysis and was unreachable in practice anyway).
    """
    global use_gemini_flash, gemini_client, use_groq, groq_client

    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key and Groq is not None:
        try:
            log.info("Groq API key detected. Initializing Groq (Free Tier)...")
            groq_client = Groq(api_key=groq_key)
            use_groq = True
            log.info("Groq initialized successfully.")
            return "groq"
        except Exception as exc:
            log.warning(f"Failed to initialize Groq, falling back: {exc}")
            use_groq = False
            groq_client = None

    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key and genai is not None:
        try:
            log.info("Google API key detected. Initializing Gemini Flash (Free Cloud Tier)...")
            gemini_client = genai.Client(api_key=api_key)
            use_gemini_flash = True
            log.info("Gemini Flash initialized successfully.")
            return "gemini"
        except Exception as exc:
            log.warning(f"Failed to initialize Gemini: {exc}")
            use_gemini_flash = False
            gemini_client = None

    log.warning("Neither GROQ_API_KEY nor GOOGLE_API_KEY is configured. LLM reasoning will be disabled.")
    return None

# -----------------------------------------------------------------------
# Error classification (single copy -- previously duplicated verbatim)
# -----------------------------------------------------------------------
def _is_request_too_large(exc):
    """True for Groq's 413 'Request Entity Too Large' -- a payload-size
    failure, not a rate limit, so retrying the same request can't help."""
    msg = str(exc)
    return "413" in msg or "request_too_large" in msg or "Request Entity Too Large" in msg


def _is_daily_quota_exceeded(exc):
    """True for a Groq 429 that's specifically a daily (TPD) limit, as
    opposed to the much shorter per-minute (TPM) limit -- TPD only resets
    after potentially over an hour, so retrying it wastes time."""
    msg = str(exc)
    return "tokens per day" in msg or "TPD" in msg


def _is_auth_error(exc):
    """
    True for errors caused by the API key itself (invalid/missing/revoked),
    which will fail identically for every model on the same key -- as
    opposed to a model-specific quirk where trying a different model is
    still worth it.
    """
    msg = str(exc)
    return "401" in msg or "invalid_api_key" in msg or "Incorrect API key" in msg


def _parse_groq_retry_seconds(exc):
    """Groq's 429 body includes a 'Please try again in 7.342s' hint."""
    match = re.search(r"try again in ([\d.]+)s", str(exc))
    if match:
        try:
            return float(match.group(1)) + 0.5
        except ValueError:
            return None
    return None


def _extract_groq_sources(response):
    """Pulls (title, url) pairs out of groq/compound's executed_tools field
    so callers can show what was actually searched. Defensive about
    attribute-vs-dict access since SDK response objects vary."""
    def _get(obj, key):
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    sources = []
    try:
        message = response.choices[0].message
        for tool in (_get(message, "executed_tools") or []):
            search_results = _get(tool, "search_results")
            for r in (_get(search_results, "results") or []):
                url = _get(r, "url")
                title = _get(r, "title") or url
                if url and (title, url) not in sources:
                    sources.append((title, url))
    except Exception as e:
        log.warning(f"Could not extract Groq search sources: {e}")
    return sources


# -----------------------------------------------------------------------
# Per-tier call helpers
# -----------------------------------------------------------------------
def _try_groq_compound_model(prompt, model_name, max_attempts, max_tokens, log_label="analysis"):
    """
    Runs the prompt against a Groq compound (tool-using, live-search-capable)
    model and returns (text, sources, True) on success, or None if it
    fails after retries -- callers should fall through to their next tier.

    max_tokens caps the model's TOTAL output for this call, which for a
    compound model includes its internal tool-call/search reasoning as
    well as the final answer -- too low a budget silently truncates how
    much searching it actually does before it has to wrap up.

    A 413 ("Request Entity Too Large") or a daily-quota 429 both mean
    retrying this exact call is pointless, so those stop immediately
    instead of burning the remaining attempts.
    """
    max_tokens = min(max_tokens, MAX_TOKENS_CEILING)

    if _groq_headroom(model_name, max_tokens) is False:
        log.info(
            f"Skipping {model_name} for {log_label} -- last-known quota (from an "
            f"earlier call this run) shows no headroom for ~{max_tokens} tokens."
        )
        return None

    for attempt in range(max_attempts):
        try:
            raw = groq_client.chat.completions.with_raw_response.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=max_tokens,
            )
            _record_groq_headers(raw.headers, model_name)
            response = raw.parse()
            text = response.choices[0].message.content.strip()
            sources = _extract_groq_sources(response)
            return text, sources, True  # True = had live web search available
        except Exception as e:
            if _is_request_too_large(e):
                log.warning(f"Request too large for {model_name} -- skipping further retries of this payload.")
                return None
            if _is_daily_quota_exceeded(e):
                log.warning(f"Groq daily token quota (TPD) exhausted for {model_name} -- skipping remaining retries.")
                return None

            log.warning(
                f"Groq ({model_name}) {log_label} generation failed "
                f"(attempt {attempt + 1}/{max_attempts}): {e}"
            )
            if attempt < max_attempts - 1:
                wait_s = _parse_groq_retry_seconds(e) or 10
                log.info(f"Retrying {model_name} in {wait_s:.1f}s...")
                time.sleep(wait_s)
    return None


def _try_synthesis_models(prompt, max_tokens, log_label="analysis"):
    """
    Runs `prompt` (already grounded with pre-gathered context, e.g. from
    Tavily) against SYNTHESIS_MODELS in order. These are plain (non-tool)
    Groq models, so unlike groq/compound they don't burn a shared
    orchestration token budget -- and Groq's daily quota (TPD) is tracked
    per model, not per account, so one model running dry doesn't mean the
    others are unavailable too.

    Returns (text, True) on success, or None if every model fails.
    """
    if groq_client is None:
        return None
    max_tokens = min(max_tokens, MAX_TOKENS_CEILING)
    ordered_models = _order_by_headroom(SYNTHESIS_MODELS, max_tokens)
    if ordered_models != SYNTHESIS_MODELS:
        log.info(
            f"Reordered synthesis models by current quota headroom for {log_label}: {ordered_models}"
        )
    for model_name in ordered_models:
        if _groq_headroom(model_name, max_tokens) is False:
            log.info(
                f"Skipping synthesis model {model_name} for {log_label} -- last-known "
                f"quota shows no headroom for ~{max_tokens} tokens."
            )
            continue
        try:
            raw = groq_client.chat.completions.with_raw_response.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=max_tokens,
            )
            _record_groq_headers(raw.headers, model_name)
            response = raw.parse()
            text = response.choices[0].message.content.strip()
            return text, True  # True: grounded in real pre-gathered context
        except Exception as e:
            log.warning(f"Groq synthesis over gathered context failed with {model_name} ({log_label}): {e}")
            if _is_auth_error(e):
                # Same key, so every other model would fail identically --
                # no point burning more requests trying them.
                break
            # Otherwise (quota, request-too-large, or a model-specific
            # quirk) keep trying the next model -- none of those reasons
            # generalize to a different model.
    return None


def _try_gemini_grounded(prompt, log_label="analysis"):
    """
    Genuine live-search fallback: Gemini's free tier supports real Google
    Search grounding (a quota separate from both Groq and Tavily).
    Returns (text, sources, used_live) on success, or None.
    """
    global gemini_client
    if gemini_client is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key or genai is None:
            return None
        try:
            gemini_client = genai.Client(api_key=api_key)
        except Exception as e:
            log.error(f"Could not lazily initialize Gemini client for {log_label}: {e}")
            return None
    try:
        from google.genai import types
        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(tools=[grounding_tool]),
        )
        sources = []
        try:
            for candidate in response.candidates:
                gm = getattr(candidate, "grounding_metadata", None)
                for chunk in (getattr(gm, "grounding_chunks", None) or []):
                    web = getattr(chunk, "web", None)
                    if web and web.uri and (web.title, web.uri) not in sources:
                        sources.append((web.title or web.uri, web.uri))
        except Exception as e:
            log.warning(f"Could not extract Gemini grounding sources: {e}")
        used_live = bool(sources)
        return response.text.strip(), sources, used_live
    except Exception as e:
        log.error(f"Gemini grounded (live search) generation failed for {log_label}: {e}")
        return None


def _try_mistral_web_search(prompt, max_tokens=1500, log_label="analysis"):
    """
    Live-search fallback tier: Mistral's free-tier Agents API, which has a
    first-party `web_search` tool -- an entirely separate quota from
    Groq's (all models), Tavily's, and Gemini's, so it's still available
    even if all of those are exhausted on a given day.

    Requires MISTRAL_API_KEY (console.mistral.ai, free tier, no card) and
    the `mistralai` package (`pip install mistralai`). Skips cleanly (no
    exception) if either is missing.

    IMPORTANT: web_search only works via the Agents/Conversations API
    (client.beta.agents / client.beta.conversations), not the plain Chat
    Completions API -- Mistral's docs are explicit that Chat Completions
    responses can't carry the search-result references the tool returns.
    That's why this creates a throwaway agent per call rather than using a
    simple chat completion.

    Returns (text, sources, True) on success, or None on any failure so
    callers can fall through to their next option.
    """
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        log.info(f"Mistral fallback skipped for {log_label}: MISTRAL_API_KEY is not configured.")
        return None
    try:
        # mistralai>=2.7 has no top-level `mistralai/__init__.py` -- the
        # package now ships `mistralai.client` (plus `mistralai.azure.client`
        # / `mistralai.gcp.client` variants), and `Mistral` lives there. The
        # old `from mistralai import Mistral` therefore raises ImportError
        # unconditionally on 2.7.x, even when the package is installed fine
        # -- it's not a missing-package signal on this version, just a moved
        # symbol. Try the new location first, fall back to the old one in
        # case an older mistralai version is pinned.
        try:
            from mistralai.client import Mistral
        except ImportError:
            from mistralai import Mistral
    except ImportError:
        log.info(f"Mistral fallback skipped for {log_label}: the `mistralai` package isn't installed (pip install mistralai).")
        return None
    try:
        client = Mistral(api_key=api_key)
        agent = client.beta.agents.create(
            model=MISTRAL_MODEL,
            name="swing-trade-live-search",
            description="One-off agent for live-search-grounded generation.",
            tools=[{"type": "web_search"}],
        )
        response = client.beta.conversations.start(
            agent_id=agent.id,
            inputs=prompt,
        )
        text_parts, sources = [], []
        for item in getattr(response, "outputs", None) or []:
            content = getattr(item, "content", None)
            if isinstance(content, str):
                text_parts.append(content)
                continue
            for chunk in content or []:
                chunk_type = getattr(chunk, "type", None)
                if chunk_type == "text":
                    text_parts.append(getattr(chunk, "text", "") or "")
                elif chunk_type in ("tool_reference", "url_citation"):
                    url = getattr(chunk, "url", None)
                    title = getattr(chunk, "title", None) or url
                    if url and (title, url) not in sources:
                        sources.append((title, url))
        text = "".join(text_parts).strip()
        if not text:
            log.warning(f"Mistral web-search agent returned no text content for {log_label}.")
            return None
        return text, sources, True  # True: grounded via Mistral's web_search tool
    except Exception as e:
        log.error(f"Mistral web-search fallback failed for {log_label}: {e}")
        return None


# -----------------------------------------------------------------------
# Non-live synthesis tier -- for reasoning over data that's ALREADY been
# gathered (no new facts needed), e.g. reformatting a prior model reply
# into strict JSON, repairing a rejected subset of an earlier analysis,
# or a final synthesis stage over earlier stages' already-live-sourced
# output. Deliberately skips groq/compound, Tavily, Gemini-grounding, and
# Mistral entirely -- none of those add anything when the prompt already
# contains everything the model needs, and running them anyway just burns
# live-search quota that a genuinely search-dependent call elsewhere in
# the same batch/run may need. Use generate_analysis() instead whenever
# the call needs to find or verify current facts (prices, news, NAVs).
# -----------------------------------------------------------------------
def generate_synthesis(prompt, max_tokens=1200, validate_fn=None, log_label="synthesis"):
    """
    Lightweight two-tier fallback (plain Groq -> plain Gemini) for
    reasoning-only calls. No search, no sources, no live-data flag --
    callers that need any of those want generate_analysis() instead.

    validate_fn: optional text -> bool, same idea as generate_analysis()'s
    validate_fn. If Groq's reply fails this check (e.g. doesn't parse as
    the expected JSON shape), it's treated as a failure and Gemini is
    tried next, instead of returning content the caller can't use.
    Defaults to "non-empty".

    Returns the generated text, or "" (falsy) on total failure -- same
    falsy-on-failure contract as generate_analysis()'s text slot, so
    existing `if not text:` checks in callers work unchanged.
    """
    global gemini_client
    if validate_fn is None:
        validate_fn = lambda t: bool(t and t.strip())
    max_tokens = min(max_tokens, MAX_TOKENS_CEILING)
    backend = init_llm_generator()
    log.info(f"{log_label} using LLM backend: {backend}")

    if backend == "groq" and groq_client is not None and SYNTHESIS_MODELS:
        try:
            response = groq_client.chat.completions.create(
                model=SYNTHESIS_MODELS[0],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=max_tokens,
            )
            text = response.choices[0].message.content.strip()
            if validate_fn(text):
                return text
            log.warning(f"Plain Groq synthesis for {log_label} didn't match the expected format -- trying Gemini.")
        except Exception as e:
            log.warning(f"Plain Groq synthesis failed for {log_label}: {e} -- trying Gemini.")

    have_gemini = gemini_client is not None or (os.getenv("GOOGLE_API_KEY") and genai is not None)
    if have_gemini:
        try:
            if gemini_client is None:
                gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
            response = gemini_client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            text = response.text.strip()
            if validate_fn(text):
                return text
            log.warning(f"Plain Gemini synthesis for {log_label} didn't match the expected format.")
        except Exception as e:
            log.error(f"Plain Gemini synthesis failed for {log_label}: {e}")

    log.error(f"Both plain-Groq and plain-Gemini synthesis failed for {log_label}.")
    return ""


# -----------------------------------------------------------------------
# The unified chain
# -----------------------------------------------------------------------
def generate_analysis(
    prompt,
    max_tokens=1200,
    gather_context_fn=None,
    build_grounded_prompt=None,
    validate_fn=None,
    log_label="analysis",
):
    """
    Unified live-search-first LLM fallback chain (see module docstring for
    the full tier order). Every script in this project should call this
    instead of hand-rolling its own chain.

    prompt: the base (non‑grounded) prompt.
    max_tokens: output token budget, clamped to MAX_TOKENS_CEILING.
    gather_context_fn: optional () -> (context_text, sources) used ONLY for
      tier 3 (Tavily + plain Groq synthesis). If omitted, or if it returns
      falsy context_text, tier 3 is skipped and the chain moves straight
      from groq/compound-mini to Gemini.
    build_grounded_prompt: optional context_text -> str, used to build the
      tier‑3 prompt. Defaults to f"{context_text}\n\n{prompt}".
    validate_fn: optional text -> bool. If a tier's raw output fails this
      check (e.g. "didn't parse as the expected JSON shape"), that tier is
      treated as a failure and the chain moves to the next one, instead of
      returning content the caller can't use. Defaults to "non‑empty".
    log_label: short human‑readable label used in log lines (e.g. "AI
      Stocks Story", "swing‑trade generation").

    Returns (text, sources, used_live_search). On total failure, text is
    "" (falsy) so callers can check `if not text:` uniformly.
    """
    # -------------------------------------------------------------------
    # Prompt size safety net – Groq returns 413 if the request payload is
    # too large (typically when the prompt exceeds ~4\u202fKB). We proactively
    # truncate overly long prompts to avoid hitting that error and then
    # let the normal fallback chain handle the shortened request.
    #
    # IMPORTANT: callers in this project build prompts as
    #   <intro/context> + <live data block> + <analysis instructions> +
    #   <OUTPUT FORMAT -- respond with ONLY raw JSON ...> + <schema>
    # -- i.e. the live-data block (long, and least critical to keep in
    # full) comes BEFORE the format contract, not after. Blindly keeping
    # only the first MAX_PROMPT_CHARS chars therefore risks silently
    # slicing off the "respond with ONLY raw JSON" instruction and the
    # schema itself -- which doesn't make the call fail, it just makes
    # every tier respond with normal prose instead of JSON, and that
    # failure is easy to miss since nothing errors. To avoid that, when a
    # format-contract marker is present we always keep everything from
    # that marker to the end intact, and truncate only the (earlier,
    # data-heavy) portion before it.
    # -------------------------------------------------------------------
    MAX_PROMPT_CHARS = 3000  # empirical safe ceiling; can be tuned via env
    if len(prompt) > MAX_PROMPT_CHARS:
        format_marker = None
        for marker in ("OUTPUT FORMAT", "Respond with ONLY raw JSON"):
            marker_idx = prompt.find(marker)
            if marker_idx != -1:
                format_marker = marker_idx
                break

        if format_marker is not None:
            tail = prompt[format_marker:]
            head_budget = MAX_PROMPT_CHARS - len(tail)
            if head_budget < 200:
                # The format contract alone is close to (or over) the
                # ceiling -- keep it whole anyway (a working JSON
                # response is useless without it) and log that the
                # ceiling had to be exceeded, rather than truncate the
                # instructions themselves.
                log.warning(
                    f"Prompt length ({len(prompt)}) exceeds safe limit ({MAX_PROMPT_CHARS}), and "
                    f"the OUTPUT FORMAT/schema block alone is {len(tail)} chars -- keeping it whole "
                    "and truncating only the data/context ahead of it instead of risking a "
                    "prose-instead-of-JSON response."
                )
                head_budget = 200
            head = prompt[:head_budget]
            prompt = f"{head}\n\n[...truncated for length...]\n\n{tail}"
            log.warning(
                f"Prompt length ({len(prompt)}) exceeds safe limit ({MAX_PROMPT_CHARS}); "
                "truncating the data/context section only, and keeping the OUTPUT FORMAT "
                "instructions and schema intact so tiers still respond with JSON."
            )
        else:
            log.warning(
                f"Prompt length ({len(prompt)}) exceeds safe limit ({MAX_PROMPT_CHARS}); "
                "truncating to avoid Groq 413 errors."
            )
            prompt = prompt[:MAX_PROMPT_CHARS]

    if validate_fn is None:
        validate_fn = lambda t: bool(t and t.strip())

    def _validate(text, tier_name):
        ok = validate_fn(text)
        if not ok:
            preview = (text or "").strip().replace("\n", " ")[:200]
            log.warning(
                f"{tier_name} returned a reply for {log_label} but it didn't match the "
                f"expected format -- rejecting and trying the next tier. Reply preview: {preview!r}"
            )
        return ok

    max_tokens = min(max_tokens, MAX_TOKENS_CEILING)
    backend = init_llm_generator()
    log.info(f"{log_label} using LLM backend: {backend}")
    require_live = os.getenv("REQUIRE_LIVE_DATA", "true").lower() == "true"

    if backend == "groq":
        compound_attempts = {
            "groq/compound": GROQ_COMPOUND_ATTEMPTS,
            "groq/compound-mini": GROQ_COMPOUND_MINI_ATTEMPTS,
        }
        compound_order = _order_by_headroom(GROQ_COMPOUND_MODELS, max_tokens)
        if compound_order != GROQ_COMPOUND_MODELS:
            log.info(
                f"Trying compound tiers in quota‑adjusted order for {log_label}: {compound_order} "
                "(reordered from the default groq/compound -> groq/compound‑mini based on what "
                "an earlier call this run saw)."
            )
        result = None
        for model_name in compound_order:
            result = _try_groq_compound_model(
                prompt, model_name, max_attempts=compound_attempts[model_name],
                max_tokens=max_tokens, log_label=log_label,
            )
            if result is not None and _validate(result[0], model_name):
                return result
            if model_name != compound_order[-1]:
                log.info(f"{model_name} unavailable for {log_label} -- trying {compound_order[compound_order.index(model_name) + 1]}...")

        if gather_context_fn is not None:
            tavily_remaining = _tavily_remaining_credits()
            if tavily_remaining is not None and tavily_remaining <= 0:
                log.info(
                    f"Skipping Tavily‑context tier for {log_label} -- last‑known Tavily "
                    f"quota shows {tavily_remaining} credits remaining."
                )
                context_text = None
            else:
                context_text, gathered_sources = gather_context_fn()
            if context_text:
                grounded_prompt = (
                    build_grounded_prompt(context_text) if build_grounded_prompt
                    else f"{context_text}\n\n{prompt}"
                )
                synth_result = _try_synthesis_models(grounded_prompt, max_tokens=max_tokens, log_label=log_label)
                if synth_result is not None:
                    text, live = synth_result
                    if _validate(text, "Gathered-context synthesis"):
                        return text, gathered_sources, live
            else:
                log.warning(f"Gathered‑context tier returned no usable results for {log_label} -- skipping.")

        if gemini_client is not None or (os.getenv("GOOGLE_API_KEY") and genai is not None):
            grounded = _try_gemini_grounded(prompt, log_label=log_label)
            if grounded is not None and _validate(grounded[0], "Gemini grounded"):
                return grounded

        mistral_result = _try_mistral_web_search(prompt, max_tokens=max_tokens, log_label=log_label)
        if mistral_result is not None and _validate(mistral_result[0], "Mistral web-search"):
            return mistral_result

        if not require_live:
            plain_result = _try_synthesis_models(prompt, max_tokens=max_tokens, log_label=f"{log_label} (no search)")
            if plain_result is not None:
                text, _live = plain_result
                if _validate(text, "no-search synthesis"):
                    return text, [], False

    elif backend == "gemini":
        grounded = _try_gemini_grounded(prompt, log_label=log_label)
        if grounded is not None and _validate(grounded[0], "Gemini grounded"):
            return grounded

        mistral_result = _try_mistral_web_search(prompt, max_tokens=max_tokens, log_label=log_label)
        if mistral_result is not None and _validate(mistral_result[0], "Mistral web-search"):
            return mistral_result

        if not require_live and gemini_client is not None:
            try:
                response = gemini_client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
                text = response.text.strip()
                if _validate(text, "Gemini (no search)"):
                    return text, [], False
            except Exception as e:
                log.error(f"Gemini (no search) generation failed for {log_label}: {e}")

    if require_live:
        log.info(
            f"Every live-search path failed for {log_label}, and REQUIRE_LIVE_DATA=true "
            "means a non-search fallback's output would be discarded anyway -- skipping "
            "the no-search Groq/Gemini call entirely rather than spending remaining quota "
            "on a result that can't be used. Set REQUIRE_LIVE_DATA=false to allow a "
            "clearly-labeled stale-data run instead."
        )

    return "", [], False