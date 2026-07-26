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
project default is "true"), does this fall through to non-live generation:
plain Groq, then the local Qwen2.5-1.5B model.

Callers own their own prompt construction, response parsing, and
domain-specific context gathering (e.g. which Tavily queries to run) --
this module only owns "which model, in which order, with which
retry/backoff policy, and which error is worth retrying at all."
"""

import os
import re
import time
import threading

try:
    from transformers import pipeline
except ImportError:
    pipeline = None
try:
    from groq import Groq
except ImportError:
    Groq = None
try:
    from google import genai
except ImportError:
    genai = None

from logger import log


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
DEFAULT_SYNTHESIS_MODELS = "llama-3.3-70b-versatile,llama-3.1-8b-instant"
SYNTHESIS_MODELS = [
    m.strip() for m in os.getenv("SYNTHESIS_MODELS", DEFAULT_SYNTHESIS_MODELS).split(",")
    if m.strip()
]

GEMINI_MODEL = "gemini-flash-latest"
MISTRAL_MODEL = "mistral-medium-latest"
LOCAL_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Hard ceiling on any single call's max_tokens, regardless of what a caller
# requests -- keeps a bug in one caller's token-budget math from turning
# into an oversized (and wastefully expensive) request. Callers should
# still pass their own tighter budget; this is just a backstop.
MAX_TOKENS_CEILING = 4200


# -----------------------------------------------------------------------
# Shared client state (mirrors what main.py used to hold module-globally)
# -----------------------------------------------------------------------
model_lock = threading.Lock()
llm_pipeline = None
use_gemini_flash = False
gemini_client = None
use_groq = False
groq_client = None


def init_llm_generator(force_local=False):
    """
    Initializes whichever LLM backend is available.
    Priority: Groq (free tier) if GROQ_API_KEY is present, then Gemini
    2.5 Flash if GOOGLE_API_KEY is present, then the local Qwen2.5-1.5B
    model.

    force_local=True skips the Groq/Gemini checks entirely and goes
    straight to the local model. Use this when Groq/Gemini were already
    tried by an earlier call in the same fallback chain and exhausted
    their quota -- otherwise this function always re-picks Groq first
    because it only checks whether GROQ_API_KEY is set, not whether it
    still has quota left.
    """
    global llm_pipeline, use_gemini_flash, gemini_client, use_groq, groq_client

    if not force_local:
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
                log.info("Google API key detected. Initializing Gemini 2.5 Flash (Free Cloud Tier)...")
                gemini_client = genai.Client(api_key=api_key)
                use_gemini_flash = True
                log.info("Gemini 2.5 Flash initialized successfully.")
                return "gemini"
            except Exception as exc:
                log.warning(f"Failed to initialize Gemini, falling back to local model: {exc}")
                use_gemini_flash = False
                gemini_client = None

    if pipeline is None:
        log.warning("The 'transformers' library is not installed. LLM reasoning will be disabled.")
        return None

    if llm_pipeline is None:
        try:
            import torch
            device = -1
            torch_dtype = torch.float32

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
                torch_dtype = torch.float16
                log.info("Apple Silicon GPU (MPS) detected. Enabling hardware acceleration.")
            elif torch.cuda.is_available():
                device = 0
                torch_dtype = torch.float16
                log.info("Nvidia GPU (CUDA) detected. Enabling hardware acceleration.")
            else:
                log.info("No compatible GPU detected. Running model on CPU.")

            log.info(f"Initializing local AI model ({LOCAL_MODEL})...")
            llm_pipeline = pipeline(
                "text-generation",
                model=LOCAL_MODEL,
                device=device,
                torch_dtype=torch_dtype,
            )
            log.info("Local AI model initialized successfully.")
        except Exception as e:
            log.error(f"Failed to initialize local AI model: {e}")
            llm_pipeline = None

    return "local" if llm_pipeline is not None else None


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
    for attempt in range(max_attempts):
        try:
            response = groq_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=max_tokens,
            )
            text = response.choices[0].message.content.strip()
            sources = _extract_groq_sources(response)
            return text, sources, True  # True = had live web search available
        except Exception as e:
            log.error(
                f"Groq ({model_name}) {log_label} generation failed "
                f"(attempt {attempt + 1}/{max_attempts}): {e}"
            )
            if _is_request_too_large(e):
                log.error(f"Request too large for {model_name} -- skipping further retries of this payload.")
                return None
            if _is_daily_quota_exceeded(e):
                log.error(f"Groq daily token quota (TPD) exhausted for {model_name} -- skipping remaining retries.")
                return None
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
    for model_name in SYNTHESIS_MODELS:
        try:
            response = groq_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=max_tokens,
            )
            text = response.choices[0].message.content.strip()
            return text, True  # True: grounded in real pre-gathered context
        except Exception as e:
            log.error(f"Groq synthesis over gathered context failed with {model_name} ({log_label}): {e}")
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


def _generate_local(prompt, max_new_tokens=1200, log_label="analysis"):
    """Runs `prompt` through the local Qwen2.5-1.5B pipeline. Returns the
    generated text, or None if unavailable/failed."""
    if llm_pipeline is None:
        return None
    try:
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = llm_pipeline.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        with model_lock:
            outputs = llm_pipeline(
                formatted_prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.4,
                top_k=50,
                top_p=0.95,
            )
        generated_text = outputs[0]["generated_text"]
        return generated_text.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
    except Exception as e:
        log.error(f"Local {log_label} generation failed: {e}")
        return None


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

    prompt: the base (non-grounded) prompt.
    max_tokens: output token budget, clamped to MAX_TOKENS_CEILING.
    gather_context_fn: optional () -> (context_text, sources) used ONLY for
      tier 3 (Tavily + plain Groq synthesis). If omitted, or if it returns
      falsy context_text, tier 3 is skipped and the chain moves straight
      from groq/compound-mini to Gemini.
    build_grounded_prompt: optional context_text -> str, used to build the
      tier-3 prompt. Defaults to f"{context_text}\\n\\n{prompt}".
    validate_fn: optional text -> bool. If a tier's raw output fails this
      check (e.g. "didn't parse as the expected JSON shape"), that tier is
      treated as a failure and the chain moves to the next one, instead of
      returning content the caller can't use. Defaults to "non-empty".
    log_label: short human-readable label used in log lines (e.g. "AI
      Stocks Story", "swing-trade generation").

    Returns (text, sources, used_live_search). On total failure, text is
    "" (falsy) so callers can check `if not text:` uniformly.
    """
    if validate_fn is None:
        validate_fn = lambda t: bool(t and t.strip())

    max_tokens = min(max_tokens, MAX_TOKENS_CEILING)
    backend = init_llm_generator()
    log.info(f"{log_label} using LLM backend: {backend}")
    require_live = os.getenv("REQUIRE_LIVE_DATA", "true").lower() == "true"

    if backend == "groq":
        result = _try_groq_compound_model(
            prompt, "groq/compound", max_attempts=GROQ_COMPOUND_ATTEMPTS,
            max_tokens=max_tokens, log_label=log_label,
        )
        if result is not None and validate_fn(result[0]):
            return result

        log.info(f"groq/compound unavailable for {log_label} -- trying groq/compound-mini...")
        result = _try_groq_compound_model(
            prompt, "groq/compound-mini", max_attempts=GROQ_COMPOUND_MINI_ATTEMPTS,
            max_tokens=max_tokens, log_label=log_label,
        )
        if result is not None and validate_fn(result[0]):
            return result

        if gather_context_fn is not None:
            context_text, gathered_sources = gather_context_fn()
            if context_text:
                grounded_prompt = (
                    build_grounded_prompt(context_text) if build_grounded_prompt
                    else f"{context_text}\n\n{prompt}"
                )
                synth_result = _try_synthesis_models(grounded_prompt, max_tokens=max_tokens, log_label=log_label)
                if synth_result is not None:
                    text, live = synth_result
                    if validate_fn(text):
                        return text, gathered_sources, live
            else:
                log.warning(f"Gathered-context tier returned no usable results for {log_label} -- skipping.")

        if gemini_client is not None or (os.getenv("GOOGLE_API_KEY") and genai is not None):
            grounded = _try_gemini_grounded(prompt, log_label=log_label)
            if grounded is not None and validate_fn(grounded[0]):
                return grounded

        mistral_result = _try_mistral_web_search(prompt, max_tokens=max_tokens, log_label=log_label)
        if mistral_result is not None and validate_fn(mistral_result[0]):
            return mistral_result

        if not require_live:
            plain_result = _try_synthesis_models(prompt, max_tokens=max_tokens, log_label=f"{log_label} (no search)")
            if plain_result is not None:
                text, _live = plain_result
                if validate_fn(text):
                    return text, [], False

    elif backend == "gemini":
        grounded = _try_gemini_grounded(prompt, log_label=log_label)
        if grounded is not None and validate_fn(grounded[0]):
            return grounded

        mistral_result = _try_mistral_web_search(prompt, max_tokens=max_tokens, log_label=log_label)
        if mistral_result is not None and validate_fn(mistral_result[0]):
            return mistral_result

        if not require_live and gemini_client is not None:
            try:
                response = gemini_client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
                text = response.text.strip()
                if validate_fn(text):
                    return text, [], False
            except Exception as e:
                log.error(f"Gemini (no search) generation failed for {log_label}: {e}")

    if require_live:
        log.info(
            f"Every live-search path failed for {log_label}, and REQUIRE_LIVE_DATA=true "
            "means a non-search fallback's output would be discarded anyway -- skipping "
            "the no-search Groq/Gemini call and the local model entirely rather than "
            "spending remaining quota/CPU time on a result that can't be used. Set "
            "REQUIRE_LIVE_DATA=false to allow a clearly-labeled stale-data run instead."
        )
        return "", [], False

    local_backend = init_llm_generator(force_local=True)
    if local_backend == "local" and llm_pipeline is not None:
        text = _generate_local(prompt, max_new_tokens=min(max_tokens, 1200), log_label=log_label)
        if text and validate_fn(text):
            return text, [], False

    return "", [], False