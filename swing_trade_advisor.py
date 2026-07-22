"""
swing_trade_advisor.py

Standalone companion to main.py. Runs a single, open-ended "find me a
high-conviction 3-5 month swing trade" prompt against whichever free LLM
backend main.py already knows how to set up (Groq free tier -> Gemini
2.5 Flash free tier -> local Qwen2.5-1.5B fallback), then emails the
result to the same recipients configured for the stock report
(EMAIL_TO / EMAIL_CC in config.py / the workflow yaml's env vars).

This intentionally reuses main.py's LLM-selection and email-credential
plumbing instead of duplicating it, so both scripts stay in sync with
whatever provider is configured.

LIVE DATA: tries several live-search paths in order before ever falling back
to a plain (non-live) model:
  1. groq/compound -- Groq's free tool-using system, autonomous web search.
  2. groq/compound-mini -- lighter variant, tried if #1 is unavailable.
  3. Tavily direct search (TAVILY_API_KEY, free tier, 1,000 searches/month,
     no card) + a plain Groq model for synthesis -- a real live-search path
     that doesn't share compound's expensive internal orchestration budget.
  4. Gemini + Google Search grounding (GOOGLE_API_KEY, separate free quota).
Only if all four fail does it drop to non-live generation (plain Groq, then
local Qwen2.5-1.5B) -- and even then, REQUIRE_LIVE_DATA=true (the default)
aborts the run instead of emailing unverified, training-data-only output.
The email lists whichever sources were actually used under "Sources
checked" so you can spot-check the claims.

CAVEAT: this is still not a verified real-time trade signal. Web search
results can be a few hours stale, incomplete, or misread by the model.
Treat every price level, %, and "recent" news item as a starting hypothesis
to verify against a live quote/news source yourself -- not investment advice.
"""

import os
import re
import sys
import json
import html
import time
import requests
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo

import smtplib
from email.mime.text import MIMEText

import pandas as pd

import main  # reuses LLM init, email config/credentials, and helpers
from compliance import build_compliance_block_html

# -----------------------------
# Prompt
# -----------------------------
def build_prompt():
    today_str = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%d %B %Y")
    return f"""Core Objective: Using the most current market data as of {today_str}, identify a high-conviction swing trade suitable for a 3-5 month hold, based on a rigorous blend of fundamental, technical, and sentiment analysis. Only recommend a stock you have genuine conviction in; weigh relevant global cues, Indian market news, and risks. Do not fabricate a price, financial figure, or news item -- if you cannot find or verify a real current number for a required field, say so explicitly in "rationale" rather than inventing one.

Stock Selection Strategy (choose one per stock):
- Momentum Breakout: a large-cap or quality mid-cap breaking out of a multi-month consolidation (Cup and Handle, Multi-Year Base, Symmetrical Triangle on a weekly chart) on significantly above-average volume, signaling a new sustained intermediate-term uptrend.
- Event-Driven: positioned to gain from a major near-term catalyst (regulatory approval, large contract win, demerger/spinoff, M&A arbitrage) with a clearly quantifiable price impact inside the 3-5 month window.
- Technical Swing Trade: a stock in a confirmed strong secular uptrend (above 50-WMA and 200-WMA) that has pulled back to a key support level (20-WMA, 38.2%/50% Fibonacci retracement, or horizontal support) with a clear reversal candlestick pattern.
- Fundamental Short-Term Bet: compelling valuation plus an exceptionally strong recent quarter (YoY and QoQ growth, clear beat on analyst consensus) and strongly positive guidance -- a re-rating play.

Mandatory analysis parameters:
- Fundamentals: low debt-to-equity (or strong asset quality for financials), high/improving ROCE/ROE, and check for promoter/institutional buying last quarter.
- Latest quarter: net profit and revenue growth both above 20% YoY, with margin expansion.
- Technicals (3-5M view): price above 20-week and 50-week SMA; weekly RSI trending up but below 70; bullish MACD crossover.
- Sentiment: recent positive catalysts (analyst upgrades, sector tailwinds, large orders) and supportive FII/DII activity.
- Risk/reward: minimum 1:2.5 based on your own proposed stop-loss and target -- before answering, verify the arithmetic yourself: risk_reward_ratio must equal (target1_pct / stop_loss_pct) to one decimal place; if it doesn't, adjust the target or stop-loss until it does rather than reporting a mismatched ratio.

Provide TWO stocks: the highest-conviction one from any strategy above, and a second from a different strategy if a genuinely qualifying one exists.

OUTPUT FORMAT -- respond with ONLY raw JSON matching the schema below, and nothing else (no markdown, no code fences, no commentary before or after). Plain text/numbers only (no HTML):

{{
  "stocks": [
    {{
      "name": "Stock name",
      "ticker": "Exact, currently-listed Yahoo Finance ticker (e.g. 'RELIANCE.NS' for NSE-listed, 'AAPL' for US-listed) -- must be a real symbol you are confident is correct, since it is used to fetch a live quote; a wrong or invented ticker will silently break that lookup",
      "allocation_pct": "e.g. 5-10%",
      "entry_date": "Targeted entry date",
      "exit_date": "Expected exit date, 3-5 months from entry",
      "strategy_type": "Strategy name used",
      "confidence_score": "Conviction out of 10 (e.g. 8.8) -- weigh fundamental + technical + sentiment strength together",
      "risk_level": "One word: 'Medium' or 'High'",
      "key_catalysts": "2-4 near-term catalysts, comma-separated, e.g. 'Earnings, AI Chip Launch, Fed Meeting'",
      "risk_reward_ratio": "e.g. '1 : 2.5' -- must arithmetically match stop_loss_pct and target1_pct below",
      "upside_target_pct": "Favourable % for 3-5 months",
      "stop_loss_pct": "Risk % (Stop-Loss)",
      "target1_pct": "Expected Profit % (T1)",
      "target2_pct": "Expected Profit % (T2), optional",
      "top_buyers": "Recent FII/DII activity, if known",
      "broker_recommendations": "e.g. 'Buy' with target X from a named brokerage, if known",
      "rationale": "Two to three sentences covering fundamental + technical + sentiment rationale and the key risk to watch"
    }},
    {{ ... second stock, same fields ... }}
  ]
}}
"""


# -----------------------------
# LLM call (larger token budget than main.py's per-stock reasoning calls,
# since this is one long-form response rather than a short per-stock blurb)
# -----------------------------
def _tavily_search(query, max_results=4):
    """
    Runs one query against Tavily's search API directly (the same backend
    groq/compound uses internally) and returns a list of
    {"title", "url", "content"} dicts, or [] on any failure. Uses Tavily's
    own free-tier quota (1,000 searches/month, no credit card) -- entirely
    separate from Groq's and Gemini's budgets, so it isn't affected by
    either being exhausted.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return []
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "search_depth": "basic",
                "max_results": max_results,
                "include_answer": False,
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for r in data.get("results", []):
            results.append({
                "title": r.get("title") or r.get("url", ""),
                "url": r.get("url", ""),
                # Trimmed to keep the combined context compact -- this is
                # meant to ground the model in real facts/URLs, not to hand
                # it full articles.
                "content": (r.get("content") or "")[:280],
            })
        return results
    except Exception as e:
        main.log.warning(f"Tavily search failed for query '{query}': {e}")
        return []


def _gather_tavily_context(today_str):
    """
    Runs a small, fixed set of targeted queries covering the areas the
    prompt actually needs (momentum/breakout setups, FII/DII activity,
    broker calls) and assembles the results into a compact context block
    plus a deduplicated (title, url) source list. Returns (context_text,
    sources) -- context_text is "" if Tavily isn't configured or every
    query failed.
    """
    queries = [
        f"NSE BSE India stock momentum breakout {today_str}",
        f"India stock market FII DII buying activity {today_str}",
        f"Indian stock brokerage buy rating target price upgrade {today_str}",
    ]
    sources = []
    blocks = []
    for q in queries:
        for r in _tavily_search(q):
            if not r["url"]:
                continue
            if (r["title"], r["url"]) not in sources:
                sources.append((r["title"], r["url"]))
            blocks.append(f"- {r['title']} ({r['url']}): {r['content']}")
    if not blocks:
        return "", []
    context_text = (
        "LIVE SEARCH RESULTS (use these real, freshly-fetched facts as your "
        "data source -- do not treat this as training data):\n"
        + "\n".join(blocks)
        + "\n\n"
    )
    return context_text, sources


def _try_tavily_plus_groq(prompt, today_str):
    """
    Fallback tier that decouples search from generation: fetch real search
    results via Tavily directly (its own separate free quota), prepend them
    to the prompt as grounding context, then run that through Groq's plain
    (non-compound) model -- which has no tool-orchestration overhead and so
    doesn't burn through a shared TPM budget the way groq/compound does.
    Returns (text, sources, True) on success, or None if Tavily isn't
    configured / returned nothing, or the Groq call fails.
    """
    if not os.getenv("TAVILY_API_KEY"):
        main.log.info(
            "Tavily fallback skipped: TAVILY_API_KEY is not configured."
        )
        return None
    context_text, sources = _gather_tavily_context(today_str)
    if not context_text:
        main.log.warning("Tavily returned no usable results -- skipping this fallback tier.")
        return None
    try:
        response = main.groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": context_text + prompt}],
            temperature=0.4,
            max_tokens=1500,
        )
        text = response.choices[0].message.content.strip()
        return text, sources, True  # True: grounded in real Tavily results
    except Exception as e:
        main.log.error(f"Groq synthesis over Tavily context failed: {e}")
        return None


def _try_groq_compound_model(prompt, model_name, max_attempts=3):
    """
    Runs the prompt against a Groq compound (tool-using, live-search-capable)
    model and returns (text, sources, True) on success, or None if it
    fails after retries -- callers should fall through to their next option.

    A 413 ("Request Entity Too Large") or a daily-quota 429 both mean
    retrying this exact call is pointless, so those stop immediately
    instead of burning the remaining attempts.
    """
    for attempt in range(max_attempts):
        try:
            response = main.groq_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=1200,
            )
            text = response.choices[0].message.content.strip()
            sources = _extract_groq_sources(response)
            return text, sources, True  # True = had live web search available
        except Exception as e:
            main.log.error(
                f"Groq ({model_name}) swing-trade generation failed "
                f"(attempt {attempt + 1}/{max_attempts}): {e}"
            )
            if _is_request_too_large(e):
                main.log.error(
                    f"Request too large for {model_name} -- skipping "
                    "further retries of this payload."
                )
                return None
            if _is_daily_quota_exceeded(e):
                main.log.error(
                    f"Groq daily token quota (TPD) exhausted for "
                    f"{model_name} -- retrying within seconds cannot help. "
                    "Skipping remaining retries."
                )
                return None
            if attempt < max_attempts - 1:
                wait_s = _parse_groq_retry_seconds(e) or 10
                main.log.info(f"Retrying {model_name} in {wait_s:.1f}s...")
                time.sleep(wait_s)
    return None


def generate_analysis(prompt):
    backend = main.init_llm_generator()
    main.log.info(f"Swing trade advisor using LLM backend: {backend}")

    if backend == "groq":
        # groq/compound is Groq's free, tool-using system: same API key/client
        # as the plain llama model, but it can autonomously trigger a live web
        # search (via Tavily under the hood) when the query needs current
        # info -- which is exactly what this "use latest data" prompt needs.
        # It's slower and uses more of the free-tier quota than a plain chat
        # completion (it may make several web searches per request under the
        # hood), but for a once-a-run email that's a good trade.
        #
        # groq/compound orchestrates via meta-llama/llama-4-scout, whose free
        # tier is capped at 30,000 tokens/minute org-wide. A 429 here is
        # usually just that minute's shared quota being briefly exhausted --
        # not a real outage -- so retry a couple of times with the wait time
        # Groq's own error message reports before giving up on it.
        result = _try_groq_compound_model(prompt, "groq/compound", max_attempts=3)
        if result is not None:
            return result

        # groq/compound is unavailable/overloaded. Community reports (Groq's
        # own forum) show the free tier throwing bare "Request too large"
        # 413s on `compound` even for tiny prompts -- this isn't reliably
        # about payload size, it's compound's heavier tool-orchestration
        # overhead hitting the free tier's ceiling. groq/compound-mini uses
        # a lighter underlying model with less orchestration overhead, so
        # it's worth one more attempt on the same free-tier key before
        # spending a separate service's (Gemini's) quota.
        main.log.info("groq/compound unavailable -- trying groq/compound-mini...")
        result = _try_groq_compound_model(prompt, "groq/compound-mini", max_attempts=2)
        if result is not None:
            return result

        # Both compound variants are still rate-limited/unavailable -- and
        # per observed logs, that's not a transient blip: groq/compound's
        # own internal search/planning loop can burn most of its shared
        # 30,000 TPM budget on a single logical request, so retrying within
        # the compound family doesn't reliably help. Try Tavily directly
        # (the same search backend compound uses, but on its own separate
        # free quota) paired with a plain, non-orchestrating Groq call --
        # genuine live grounding without compound's overhead.
        today_str = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%d %B %Y")
        result = _try_tavily_plus_groq(prompt, today_str)
        if result is not None:
            return result

        # Tavily/plain-Groq wasn't available either. Try Gemini + Google
        # Search grounding next, since that's a genuinely separate free-tier
        # quota from both Groq's and Tavily's.
        if main.gemini_client is None and not (os.getenv("GOOGLE_API_KEY") and main.genai is not None):
            main.log.info(
                "Gemini live-search fallback skipped: GOOGLE_API_KEY is not "
                "configured (or the google-genai package isn't installed), "
                "so there is no second live-data path available for this run."
            )
        else:
            grounded = _try_gemini_grounded(prompt)
            if grounded is not None:
                return grounded

        # Last resort (still "live-ish"): plain (non-search) Groq model, so
        # the run still produces *something* rather than giving up entirely.
        try:
            response = main.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=1500,
            )
            return response.choices[0].message.content.strip(), [], False
        except Exception as e2:
            main.log.error(f"Groq fallback (no search) generation also failed: {e2}")
            if _is_daily_quota_exceeded(e2):
                main.log.error(
                    "Groq's daily token quota is exhausted for this org -- "
                    "this is shared with main.py's per-stock reasoning calls "
                    "(same GROQ_API_KEY, same default model). If main.py ran "
                    "earlier today against many stocks, it may have used "
                    "most of the 100k/day budget before this script ran. "
                    "Falling back to the local model instead of retrying Groq."
                )

    elif backend == "gemini" or main.gemini_client is not None:
        grounded = _try_gemini_grounded(prompt)
        if grounded is not None:
            return grounded
        try:
            response = main.gemini_client.models.generate_content(
                # See note in _try_gemini_grounded below: pinned dated model
                # IDs can get gated/retired without warning, so use Google's
                # auto-updated alias instead.
                model="gemini-flash-latest",
                contents=prompt,
            )
            return response.text.strip(), [], False
        except Exception as e:
            main.log.error(f"Gemini swing-trade generation failed: {e}")

    # Absolute last resort: the local Qwen2.5-1.5B model. Previously this
    # branch only ran when `backend == "local"` -- i.e. only when neither
    # GROQ_API_KEY nor GOOGLE_API_KEY was set at all. If Groq *was*
    # configured (the common case) but every Groq/Gemini path above failed
    # (rate limits, exhausted daily quota, no GOOGLE_API_KEY), the function
    # fell straight through to `return None, [], False` without ever trying
    # the local model -- even though it may already be loaded in memory.
    # force_local=True here re-enters init_llm_generator() and skips
    # straight to the local-model section regardless of which backend was
    # originally selected, since we already know Groq/Gemini can't serve
    # this request right now.
    local_backend = main.init_llm_generator(force_local=True)
    if local_backend == "local" and main.llm_pipeline is not None:
        text = _generate_local(prompt)
        if text:
            return text, [], False

    return None, [], False


def _generate_local(prompt):
    """Runs the prompt through the local Qwen2.5-1.5B pipeline. Returns the
    generated text, or None if the local model isn't available or fails."""
    if main.llm_pipeline is None:
        return None
    try:
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = main.llm_pipeline.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        with main.model_lock:
            outputs = main.llm_pipeline(
                formatted_prompt,
                max_new_tokens=1200,
                do_sample=True,
                temperature=0.4,
                top_k=50,
                top_p=0.95,
            )
        generated_text = outputs[0]["generated_text"]
        text = generated_text.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
        return text
    except Exception as e:
        main.log.error(f"Local swing-trade generation failed: {e}")
        return None

    return None, [], False


def _is_request_too_large(exc):
    """True for Groq's 413 'Request Entity Too Large' -- a payload-size
    failure, not a rate limit, so unlike a 429 it can't be fixed by waiting
    and retrying the same request."""
    msg = str(exc)
    return "413" in msg or "request_too_large" in msg or "Request Entity Too Large" in msg


def _is_daily_quota_exceeded(exc):
    """True for a Groq 429 that's specifically a 'tokens per day' (TPD)
    limit, as opposed to the much shorter 'tokens per minute' (TPM) limit.
    TPM resets within seconds and is worth retrying; TPD is an org-wide
    daily budget that only resets after (per Groq's own error message)
    potentially over an hour -- retrying it with a short backoff is
    guaranteed to fail again and just wastes the run's time budget.
    """
    msg = str(exc)
    return "tokens per day" in msg or "TPD" in msg


def _parse_groq_retry_seconds(exc):
    """Groq's 429 body includes a 'Please try again in 7.342s' hint -- pull
    that out so retries wait the right amount instead of guessing."""
    match = re.search(r"try again in ([\d.]+)s", str(exc))
    if match:
        try:
            return float(match.group(1)) + 0.5  # small buffer
        except ValueError:
            return None
    return None


def _try_gemini_grounded(prompt):
    """
    Genuine live-search fallback: Gemini's free tier supports real Google
    Search grounding (separate free-tier quota from Groq entirely), so this
    is a real second live-data path, not just another training-data model.
    Returns (text, sources, True) on success, or None if Gemini isn't
    configured or the call fails -- callers should fall through to their
    next option in that case.
    """
    if main.gemini_client is None:
        # main.init_llm_generator() only builds gemini_client when Gemini is
        # chosen as the *primary* backend -- if GROQ_API_KEY was set, Groq
        # wins that selection and the function returns before ever touching
        # GOOGLE_API_KEY, so gemini_client stays None even when a valid
        # GOOGLE_API_KEY exists. Build a client here on demand instead of
        # silently giving up, so this fallback path actually gets used.
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key or main.genai is None:
            return None
        try:
            main.gemini_client = main.genai.Client(api_key=api_key)
        except Exception as e:
            main.log.error(f"Could not lazily initialize Gemini client for grounded fallback: {e}")
            return None
    try:
        from google.genai import types
        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        response = main.gemini_client.models.generate_content(
            # See note by the other generate_content calls: use Google's
            # auto-updated alias instead of a pinned dated model ID that can
            # get gated/retired for some accounts without warning.
            model="gemini-flash-latest",
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
            main.log.warning(f"Could not extract Gemini grounding sources: {e}")
        # Only report this as "live" if grounding actually fired -- Gemini
        # may decide a query doesn't need a search and skip it.
        used_live = bool(sources)
        return response.text.strip(), sources, used_live
    except Exception as e:
        main.log.error(f"Gemini grounded (live search) generation failed: {e}")
        return None


def _extract_groq_sources(response):
    """
    Pulls (title, url) pairs out of groq/compound's executed_tools field so
    the email can show what was actually searched -- lets the reader spot-
    check the model's claims instead of taking them on faith. Defensive
    about attribute-vs-dict access since SDK response objects vary.
    """
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
        main.log.warning(f"Could not extract Groq search sources: {e}")
    return sources


def _parse_analysis_json(text):
    """Parses the compact JSON the LLM now returns (see build_prompt's
    OUTPUT FORMAT). Models sometimes still wrap it in ```json fences or add
    stray text around it despite instructions -- handle both. Returns a
    list of stock dicts, or None if nothing usable could be parsed."""
    cleaned = _strip_code_fences(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: grab the first {...} span in case the model added any
        # commentary before/after the JSON despite instructions not to.
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    stocks = data.get("stocks") if isinstance(data, dict) else None
    if not stocks or not isinstance(stocks, list):
        return None
    return stocks


def _fetch_current_price(ticker):
    """
    Fetches the latest live close price for a ticker via the same
    yfinance path main.py uses for its own daily report (main.fetch_data),
    so the Swing Trade Research Note shows a real, current quote rather
    than a price the LLM may have guessed or pulled from stale training
    data. Returns (price_float, currency_symbol) or (None, None) on any
    failure -- callers should treat that as "unavailable", not an error.
    """
    ticker = (ticker or "").strip()
    if not ticker:
        return None, None
    try:
        df = main.fetch_data(ticker)
        latest_close = main._safe_float(df.iloc[-1].get("close"))
        if latest_close is None:
            return None, None
        market = main.classify_market(ticker)
        currency_symbol = "₹" if market == "India" else "$"
        return latest_close, currency_symbol
    except Exception as e:
        main.log.warning(f"Could not fetch live price for '{ticker}': {e}")
        return None, None


def _attach_live_prices(stocks):
    """
    Mutates each stock dict in place, adding a 'current_price_display'
    string (e.g. '₹2,845.30' or '$212.40') built from a live yfinance
    quote keyed off the 'ticker' field the LLM was asked to supply. Falls
    back to '—' (rendered as such in the table) if the ticker is missing
    or the live fetch fails, rather than showing an unverified figure.
    """
    for stock in stocks:
        price, currency_symbol = _fetch_current_price(stock.get("ticker"))
        if price is not None:
            stock["current_price_display"] = f"{currency_symbol}{price:,.2f}"
        else:
            stock["current_price_display"] = None
    return stocks


# -----------------------------
# Independent verification layer
#
# Everything above this point either comes straight from the LLM's JSON or
# is a live quote used to *display* a price. Nothing so far actually checks
# whether the LLM's claims are true. The functions below re-derive the
# things that can be re-derived from real data (arithmetic, price history)
# and flag -- never silently fix or hide -- anything that doesn't hold up.
# This cannot make the recommendation itself "correct" (that depends on
# unknowable future price action), it can only stop unverified or
# self-contradictory claims from being presented with unearned confidence.
# -----------------------------

def _parse_first_number(value):
    """Extracts the first numeric value found in a string like '8%', '5-10%',
    '1 : 2.5', or a bare float/int. Returns None if nothing numeric is found."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    # No sign-matching: these fields (%, ratios, scores) are never
    # legitimately negative, and a leading '-' is far more likely to be a
    # range separator (e.g. '5-10%') than a minus sign -- matching it as a
    # sign would silently misread '5-10%' as the two numbers 5 and -10.
    match = re.search(r"\d*\.?\d+", str(value))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _parse_max_number(value):
    """Like _parse_first_number, but for ranges like '5-10%' returns the
    upper bound -- used for concentration-risk checks where the worst case
    (not the first number encountered) is what matters."""
    if value is None:
        return None
    numbers = re.findall(r"\d*\.?\d+", str(value))
    if not numbers:
        return None
    try:
        return max(float(n) for n in numbers)
    except ValueError:
        return None


def _verify_risk_reward(stock):
    """
    Recomputes risk_reward_ratio from stop_loss_pct/target1_pct instead of
    trusting the LLM's self-reported arithmetic -- the prompt asks the model
    to verify this itself, but LLMs are unreliable at exact arithmetic even
    when explicitly instructed to check it, so it needs a real check on the
    Python side. Sets stock['risk_reward_ratio_verified'] to the recomputed
    '1 : X' string regardless of outcome, and returns a list of warning
    strings (empty if the model's reported ratio matches).
    """
    notes = []
    stop = _parse_first_number(stock.get("stop_loss_pct"))
    target = _parse_first_number(stock.get("target1_pct"))
    stated_ratio = stock.get("risk_reward_ratio")

    if stop is None or target is None or stop == 0:
        notes.append("Could not verify risk:reward -- stop-loss or target1 missing/unparseable.")
        stock["risk_reward_ratio_verified"] = None
        return notes

    true_ratio = round(target / stop, 1)
    stock["risk_reward_ratio_verified"] = f"1 : {true_ratio}"

    stated_match = re.search(r":\s*([-+]?\d*\.?\d+)", str(stated_ratio) if stated_ratio else "")
    stated_val = float(stated_match.group(1)) if stated_match else None

    if stated_val is None:
        notes.append(
            f"Reported risk:reward '{stated_ratio}' could not be parsed -- "
            f"recomputed value is 1 : {true_ratio}."
        )
    elif abs(stated_val - true_ratio) > 0.15:
        notes.append(
            f"Risk:reward mismatch -- model reported 1 : {stated_val}, but "
            f"target1 ({target}%) / stop-loss ({stop}%) actually gives 1 : {true_ratio}."
        )
    if true_ratio < 2.0:
        notes.append(f"Risk:reward of 1 : {true_ratio} is below the 1:2.5 minimum the prompt requires.")
    return notes


def _fetch_weekly_technicals(ticker):
    """
    Independently computes weekly SMA20/SMA50, 14-period weekly RSI, and
    MACD(12,26,9) from real OHLC history (via the same main.fetch_data path
    already used for the live price), so the email can confirm -- rather
    than just trust -- the LLM's implicit claim that the stock satisfies
    the prompt's technical filters (price above 20W/50W SMA, RSI below 70
    and rising, bullish MACD).

    Returns a dict of computed values, {'insufficient_history': True, ...}
    if there isn't enough price history for a reliable 50-week SMA (~350+
    trading days), or None if the fetch/resample fails outright -- callers
    should treat both of those as "unverifiable", not "fails the check".
    """
    try:
        df = main.fetch_data(ticker)
        if df is None or len(df) < 30 or "close" not in df.columns:
            return None

        if not isinstance(df.index, pd.DatetimeIndex):
            date_col = next((c for c in df.columns if c.lower() == "date"), None)
            if date_col is None:
                return None
            df = df.set_index(pd.to_datetime(df[date_col]))

        close = df["close"].dropna()
        if len(close) < 30:
            return None

        weekly_close = close.resample("W").last().dropna()
        if len(weekly_close) < 55:
            result = {"insufficient_history": True, "weeks_available": len(weekly_close)}
            if len(weekly_close) >= 20:
                result["sma20w"] = round(float(weekly_close.rolling(20).mean().iloc[-1]), 2)
            return result

        sma20w = weekly_close.rolling(20).mean()
        sma50w = weekly_close.rolling(50).mean()

        delta = weekly_close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, float("nan"))
        rsi = 100 - (100 / (1 + rs))

        ema12 = weekly_close.ewm(span=12, adjust=False).mean()
        ema26 = weekly_close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()

        rsi_now = rsi.iloc[-1]
        rsi_prev = rsi.iloc[-3] if len(rsi) > 3 else None

        return {
            "insufficient_history": False,
            "latest_close": round(float(weekly_close.iloc[-1]), 2),
            "sma20w": round(float(sma20w.iloc[-1]), 2),
            "sma50w": round(float(sma50w.iloc[-1]), 2),
            "rsi14w": round(float(rsi_now), 1) if pd.notna(rsi_now) else None,
            "rsi14w_prev": round(float(rsi_prev), 1) if rsi_prev is not None and pd.notna(rsi_prev) else None,
            "macd": round(float(macd_line.iloc[-1]), 3),
            "macd_signal": round(float(signal_line.iloc[-1]), 3),
        }
    except Exception as e:
        main.log.warning(f"Could not compute weekly technicals for '{ticker}': {e}")
        return None


def _verify_technicals(stock):
    """
    Compares the prompt's mandatory technical filters (price above 20W/50W
    SMA, RSI below 70 and rising, bullish MACD) against indicators computed
    from real price history, instead of trusting the LLM's implicit claim
    that the stock satisfies them. Returns a list of warning strings (empty
    if everything checks out or there wasn't enough data to check at all).
    """
    ticker = (stock.get("ticker") or "").strip()
    if not ticker:
        return ["No ticker provided -- technicals could not be verified."]

    tech = _fetch_weekly_technicals(ticker)
    if tech is None:
        return ["Technicals could not be independently verified (price history fetch failed)."]
    if tech.get("insufficient_history"):
        return [
            f"Only {tech.get('weeks_available')} weeks of price history available -- "
            "not enough to verify the 50-week SMA; technical claims are unverified."
        ]

    notes = []
    price = tech["latest_close"]
    if price < tech["sma20w"]:
        notes.append(f"Price ({price}) is BELOW the 20-week SMA ({tech['sma20w']}) -- contradicts the required uptrend filter.")
    if price < tech["sma50w"]:
        notes.append(f"Price ({price}) is BELOW the 50-week SMA ({tech['sma50w']}) -- contradicts the required uptrend filter.")

    if tech["rsi14w"] is not None:
        if tech["rsi14w"] >= 70:
            notes.append(f"Weekly RSI is {tech['rsi14w']} (>=70, overbought) -- contradicts the 'RSI below 70' requirement.")
        if tech.get("rsi14w_prev") is not None and tech["rsi14w"] < tech["rsi14w_prev"]:
            notes.append(f"Weekly RSI is falling ({tech['rsi14w_prev']} to {tech['rsi14w']}), not rising as the strategy requires.")

    if tech["macd"] < tech["macd_signal"]:
        notes.append("MACD line is currently below its signal line -- no bullish crossover in effect right now.")

    return notes


def _verify_sanity_bounds(stock):
    """
    Flags LLM outputs that are structurally valid JSON but numerically
    implausible for a 3-5 month swing trade. These aren't things live data
    can confirm or deny -- just guardrails against an obviously runaway or
    malformed number slipping through unnoticed.
    """
    notes = []

    conf = _parse_first_number(stock.get("confidence_score"))
    if conf is None:
        notes.append("Confidence score missing or unparseable.")
    elif not (0 <= conf <= 10):
        notes.append(f"Confidence score {conf} is outside the expected 0-10 range.")

    alloc = _parse_max_number(stock.get("allocation_pct"))
    if alloc is not None and alloc > 15:
        notes.append(f"Allocation up to {alloc}% in a single stock is a large concentration for one position -- double-check position sizing.")

    upside = _parse_first_number(stock.get("upside_target_pct"))
    if upside is not None and upside > 60:
        notes.append(f"Upside target of {upside}% in a 3-5 month window is unusually aggressive -- treat as a stretch case, not a base case.")

    risk_level = (stock.get("risk_level") or "").strip().lower()
    if risk_level not in ("medium", "high"):
        notes.append(f"Risk level '{stock.get('risk_level')}' is not one of the expected 'Medium'/'High' values.")

    return notes


def _verify_stock_claims(stocks):
    """
    Runs every independent check above against each stock the LLM returned
    and attaches the results as stock['_verification_notes'] (a list of
    strings), so the email shows exactly which claims are data-confirmed
    and which are unverified or contradicted -- instead of presenting
    everything with equal, unearned confidence.

    This does NOT reject or filter stocks: a contradiction here is
    information for the reader, not proof the trade idea is wrong (e.g. the
    LLM's own live search may be more current than today's yfinance pull).
    Mutates each stock dict in place and also returns the list.
    """
    for stock in stocks:
        notes = []
        notes += _verify_risk_reward(stock)
        notes += _verify_technicals(stock)
        notes += _verify_sanity_bounds(stock)
        if not stock.get("current_price_display"):
            notes.append("Live quote lookup failed for this ticker -- confirm it's a real, currently-listed symbol before trading it.")
        stock["_verification_notes"] = notes
    return stocks


def _verification_display(stock):
    """Renders stock['_verification_notes'] as HTML for the email table --
    a clean green check if nothing was flagged, or an itemized warning list
    if something was."""
    notes = stock.get("_verification_notes") or []
    if not notes:
        return '<span style="color:#2F5233;font-weight:700;">Verified -- no contradictions found</span>'
    items = "".join(f'<div style="margin-top:3px;color:#8B2E2E;">- {html.escape(n)}</div>' for n in notes)
    return f'<span style="color:#8B2E2E;font-weight:700;">{len(notes)} item(s) to check:</span>{items}'


def _confidence_display(score):
    """Turns a 0-10 confidence_score into '⭐⭐⭐⭐⭐ (8.8/10)' -- 5 stars max,
    filled proportionally (score/2, rounded), score shown to 1 decimal."""
    try:
        s = float(str(score).strip().rstrip("%"))
    except (TypeError, ValueError):
        return None
    s = max(0.0, min(10.0, s))
    filled = max(0, min(5, round(s / 2)))
    stars = "⭐" * filled + "☆" * (5 - filled)
    return f"{stars} ({s:.1f}/10)"


def _risk_level_badge(level):
    """Colored pill for 'Medium'/'High' (falls back to a plain dash for
    anything else so an unexpected LLM value never renders broken HTML)."""
    text = (level or "").strip()
    low = text.lower()
    if "high" in low:
        color, bg = "#8B2E2E", "#FBEAEA"
    elif "med" in low:
        color, bg = "#A6812F", "#FDF3D9"
    else:
        return "—"
    return (
        f'<span style="display:inline-block;padding:3px 10px;border-radius:3px;'
        f'font-size:11px;font-weight:700;color:{color};background:{bg};">{html.escape(text)}</span>'
    )


def render_stock_table_html(stocks):
    """Builds the styled HTML table locally from parsed stock data, instead
    of asking the LLM to reproduce the ~7KB inline-styled template in every
    request. This is what previously made the groq/compound prompt large
    enough to trip Groq's per-request token budget (413 Request Too Large)
    -- the template itself, not the actual analysis content, was the bulk
    of the payload."""
    sans = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
    a = stocks[0] if len(stocks) > 0 else {}
    b = stocks[1] if len(stocks) > 1 else {}

    def esc(v):
        v = "" if v is None else str(v).strip()
        return html.escape(v) if v else "—"

    def row(label, key, value_color="#14213D", bold=False):
        weight = "font-weight:700;" if bold else ""
        cells = "".join(
            f'<td style="padding:6px 10px;font-size:12px;{weight}font-family:{sans};'
            f'color:{value_color};border-top:1px solid #EDEAE2;">{esc(stock.get(key))}</td>'
            for stock in (a, b)
        )
        return (
            f'<tr><td style="padding:6px 10px;font-size:12px;font-family:{sans};'
            f'color:#4A5063;border-top:1px solid #EDEAE2;">{label}</td>{cells}</tr>'
        )

    def raw_row(label, cell_html_fn):
        """Like row(), but cell_html_fn(stock) returns ready-made HTML for
        each cell instead of plain text -- used for the confidence stars
        and the risk-level badge, which aren't safe to run through esc()."""
        cells = "".join(
            f'<td style="padding:6px 10px;font-size:12px;font-family:{sans};'
            f'color:#14213D;border-top:1px solid #EDEAE2;">{cell_html_fn(stock) or "—"}</td>'
            for stock in (a, b)
        )
        return (
            f'<tr><td style="padding:6px 10px;font-size:12px;font-family:{sans};'
            f'color:#4A5063;border-top:1px solid #EDEAE2;">{label}</td>{cells}</tr>'
        )

    rows = "".join([
        row("Name of the Stock", "name", bold=True),
        row("Current Market Price", "current_price_display", value_color="#14213D", bold=True),
        raw_row("Confidence Score", lambda s: _confidence_display(s.get("confidence_score"))),
        raw_row("Risk Level", lambda s: _risk_level_badge(s.get("risk_level"))),
        row("Key Catalysts", "key_catalysts"),
        row("Risk : Reward", "risk_reward_ratio", value_color="#14213D", bold=True),
        row("Allocation (% of capital)", "allocation_pct"),
        row("Entry Date (Targeted)", "entry_date"),
        row("Exit Date (Expected)", "exit_date"),
        row("Strategy Type", "strategy_type"),
        row("Upside Target %", "upside_target_pct", value_color="#2F5233", bold=True),
        row("Stop-Loss %", "stop_loss_pct", value_color="#8B2E2E", bold=True),
        row("Target 1 (T1) %", "target1_pct"),
        row("Target 2 (T2) %", "target2_pct"),
        row("Recent Top Buyers (FII/DII)", "top_buyers"),
        row("Broker Recommendations", "broker_recommendations"),
        raw_row("Data Verification", _verification_display),
    ])

    rationale_items = "".join(
        f'<div style="margin-top:8px;"><strong style="color:#14213D;">{esc(stock.get("name"))}:</strong> {esc(stock.get("rationale"))}</div>'
        for stock in (a, b) if stock
    )

    execution_plans = "".join(
        _trade_execution_plan_html(stock, sans) for stock in (a, b) if stock
    )

    return f"""<table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="border:1px solid #E7E4DC;border-radius:4px;overflow:hidden;border-collapse:collapse;">
<tr style="background:#14213D;"><td style="padding:9px 10px;font-family:{sans};font-size:11px;font-weight:700;color:#B08D57;text-transform:uppercase;letter-spacing:0.05em;">Parameter</td><td style="padding:9px 10px;font-family:{sans};font-size:11px;font-weight:700;color:#ffffff;text-transform:uppercase;letter-spacing:0.05em;">Stock A</td><td style="padding:9px 10px;font-family:{sans};font-size:11px;font-weight:700;color:#ffffff;text-transform:uppercase;letter-spacing:0.05em;">Stock B</td></tr>
{rows}
</table>
<div style="margin-top:14px;font-family:{sans};font-size:12px;color:#4A5063;line-height:1.65;"><strong style="color:#14213D;">Investment Rationale:</strong>{rationale_items}</div>
{execution_plans}
"""


def _trade_execution_plan_html(stock, sans):
    """Per-stock Action/Rule execution playbook. The rules themselves are a
    fixed, generic scaling plan (not something the LLM is asked to invent),
    with that stock's own T1/T2 substituted in wherever the rule refers to
    a specific target."""
    name = html.escape(str(stock.get("name") or "").strip() or "This stock")
    t1 = str(stock.get("target1_pct") or "").strip() or "Target 1"
    t2 = str(stock.get("target2_pct") or "").strip() or "Target 2"

    plan_rows = [
        ("Initial Buy", "50% at entry"),
        ("Add Position", "25% on 3–5% pullback"),
        ("Profit Booking", f"Sell 50% at Target 1 ({html.escape(t1)})"),
        ("Final Exit", f"Sell remaining at Target 2 ({html.escape(t2)}) or trailing stop"),
        ("Stop Loss", "Exit immediately if SL is hit"),
    ]
    rows_html = "".join(
        f'<tr><td style="padding:6px 10px;font-size:12px;font-weight:700;font-family:{sans};'
        f'color:#14213D;border-top:1px solid #EDEAE2;">{action}</td>'
        f'<td style="padding:6px 10px;font-size:12px;font-family:{sans};'
        f'color:#4A5063;border-top:1px solid #EDEAE2;">{rule}</td></tr>'
        for action, rule in plan_rows
    )

    return f"""
<div style="margin-top:18px;">
  <div style="font-family:{sans};font-size:12px;font-weight:700;color:#14213D;margin-bottom:6px;">Trade Execution Plan &mdash; {name}</div>
  <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="border:1px solid #E7E4DC;border-radius:4px;overflow:hidden;border-collapse:collapse;">
    <tr style="background:#F4F2ED;">
      <td style="padding:7px 10px;font-family:{sans};font-size:11px;font-weight:700;color:#8A8F9C;text-transform:uppercase;letter-spacing:0.05em;">Action</td>
      <td style="padding:7px 10px;font-family:{sans};font-size:11px;font-weight:700;color:#8A8F9C;text-transform:uppercase;letter-spacing:0.05em;">Rule</td>
    </tr>
    {rows_html}
  </table>
</div>
"""


def _strip_code_fences(text):
    """Models occasionally wrap the requested HTML in ```html ... ``` even
    when told not to -- strip that off so it renders as HTML, not literal text."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


# -----------------------------
# Email (reuses main.py's credentials/config, custom subject + wrapper
# since this isn't the daily stock report)
# -----------------------------
def _build_sources_html(sources):
    if not sources:
        return ""
    sans = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
    items = "".join(
        f'<div style="margin:5px 0 0;font-family:{sans};font-size:11px;">'
        f'<a href="{url}" style="color:#14213D;text-decoration:none;border-bottom:1px solid #B08D57;">{title}</a></div>'
        for title, url in sources[:12]
    )
    return f"""
        <div style="margin-top:14px;padding-top:12px;border-top:1px solid #EDEAE2;">
          <div style="font-family:{sans};font-size:11px;font-weight:700;color:#14213D;text-transform:uppercase;letter-spacing:0.06em;">Sources Consulted &nbsp;&middot;&nbsp; Live Web Search</div>
          {items}
        </div>
    """


def build_email_html(analysis_html, today_str, sources, used_live_search):
    if used_live_search:
        disclaimer = (
            "Generated using Groq's compound model, which can run live web searches -- see "
            "\"Sources checked\" above for what it actually looked at. Search results can still be "
            "incomplete, out of date by a few hours, or misread by the model, so confirm the key "
            "prices/dates/news against a live source before acting. Not investment advice."
        )
    else:
        disclaimer = (
            "Generated by an LLM with no live market/internet access for this run -- prices, dates, "
            "\"recent\" news and broker calls above are model output and are NOT verified against a "
            "live feed. Confirm every figure against a real-time quote/news source before acting. "
            "Not investment advice."
        )

    sources_html = _build_sources_html(sources)
    sans = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
    serif = "Georgia,'Times New Roman',serif"
    live_tag = (
        '<span style="color:#B08D57;">&nbsp;&middot;&nbsp; Live web search used</span>'
        if used_live_search else ""
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Swing Trade Research Note</title>
<style>
  body {{ margin:0; padding:0; background:#F2F0EC; }}
  table {{ border-collapse:collapse !important; }}
  @media screen and (max-width:600px) {{
    .email-container {{ width:100% !important; max-width:100% !important; border-radius:0 !important; }}
    .email-padding {{ padding-left:16px !important; padding-right:16px !important; }}
  }}
</style>
</head>
<body style="margin:0;padding:0;background:#F2F0EC;font-family:{serif};color:#1B2233;">
  <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="background:#F2F0EC;width:100%;">
    <tr>
      <td align="center" style="padding:20px 16px;" class="email-padding">
        <table width="100%" cellpadding="0" cellspacing="0" role="presentation" class="email-container" style="max-width:680px;min-width:280px;background:#ffffff;border:1px solid #DAD5CB;border-radius:4px;overflow:hidden;">
          <tr>
            <td style="background:#14213D;padding:26px 28px 22px;" class="email-padding">
              <div style="font-family:{sans};font-size:10px;font-weight:700;letter-spacing:0.16em;text-transform:uppercase;color:#B08D57;">Market Intelligence &nbsp;&bull;&nbsp; Idea Generation</div>
              <h1 style="margin:8px 0 0;font-family:{serif};font-weight:400;font-size:23px;line-height:1.3;color:#ffffff;letter-spacing:0.01em;">Swing Trade Research Note</h1>
              <p style="margin:6px 0 0;font-family:{sans};font-size:12px;color:#B7BEC9;">3&ndash;5 Month Positioning Horizon</p>
            </td>
          </tr>
          <tr>
            <td style="height:3px;line-height:3px;font-size:0;background:linear-gradient(90deg,#B08D57,#D9C393 45%,#B08D57);">&nbsp;</td>
          </tr>
          <tr>
            <td style="padding:16px 28px 4px;" class="email-padding">
              <p style="margin:0;font-family:{sans};font-size:12px;color:#8A8F9C;">Prepared {today_str} at {datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%I:%M %p IST")}{live_tag}</p>
            </td>
          </tr>
          <tr>
            <td style="padding:14px 28px 18px;" class="email-padding">
              {analysis_html}
              {sources_html}
            </td>
          </tr>
{build_compliance_block_html(report_kind="swing", run_note=disclaimer)}
        </table>
      </td>
    </tr>
  </table>
</body>
</html>
"""


def send_swing_trade_email(html_body):
    if not all([main.EMAIL_FROM, main.EMAIL_PASSWORD, main.EMAIL_TO]):
        main.log.error(
            "Email credentials not found. Please set EMAIL_FROM, EMAIL_PASSWORD, "
            "and EMAIL_TO (the same env vars main.py uses)."
        )
        return False

    to_recipients = main.parse_email_list(main.EMAIL_TO)
    cc_recipients = main.parse_email_list(getattr(main, "EMAIL_CC", "") or "")

    if not to_recipients:
        main.log.error("No valid TO recipients found in EMAIL_TO.")
        return False

    now_ist = datetime.now(ZoneInfo("UTC")).astimezone(ZoneInfo("Asia/Kolkata"))
    time_str = now_ist.strftime("%I:%M %p IST")
    subject = f"Swing Trade Research Note — {main.get_date_with_suffix(now_ist)} · {time_str}"

    msg = MIMEText(html_body, "html")
    msg["Subject"] = subject
    msg["From"] = main.EMAIL_FROM
    msg["To"] = ", ".join(to_recipients)
    if cc_recipients:
        msg["Cc"] = ", ".join(cc_recipients)

    all_recipients = to_recipients + cc_recipients

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(main.EMAIL_FROM, main.EMAIL_PASSWORD)
        server.sendmail(main.EMAIL_FROM, all_recipients, msg.as_string())
        server.quit()
        main.log.info("Swing trade email sent successfully.")
        return True
    except smtplib.SMTPAuthenticationError:
        main.log.error(
            "SMTP Authentication Error: check EMAIL_FROM/EMAIL_PASSWORD "
            "(use a Gmail App Password, not the account password)."
        )
    except Exception as e:
        main.log.error(f"Failed to send swing trade email: {e}")
        traceback.print_exc()
    return False


def run():
    today_str = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%d %B %Y")
    prompt = build_prompt()

    analysis, sources, used_live_search = generate_analysis(prompt)
    if not analysis:
        main.log.error(
            "No LLM backend produced output (no GROQ_API_KEY/GOOGLE_API_KEY set "
            "and local model unavailable/failed). Aborting without sending an email."
        )
        sys.exit(1)

    if not used_live_search and os.getenv("REQUIRE_LIVE_DATA", "true").lower() == "true":
        # Only groq/compound actually hits the live web. Every other path
        # (Groq's plain llama fallback, Gemini, or the local model) is pure
        # training-data reasoning with no real-time prices/news -- exactly
        # the stale output this run is meant to avoid. Refuse to email it
        # rather than silently sending non-verified figures. Set
        # REQUIRE_LIVE_DATA=false to explicitly allow a stale-data fallback
        # email instead of aborting.
        main.log.error(
            "Live web search was not used for this run (Groq's live-search "
            "model was unavailable or the backend fell back to Gemini/local), "
            "so the output would only reflect stale training-data prices/news. "
            "Aborting without sending an email. Set REQUIRE_LIVE_DATA=false to "
            "override and allow a clearly-labeled stale-data email instead."
        )
        sys.exit(1)

    stocks = _parse_analysis_json(analysis)
    if stocks:
        stocks = _attach_live_prices(stocks)
        stocks = _verify_stock_claims(stocks)
        analysis_html = render_stock_table_html(stocks)
    else:
        # Model didn't return parseable JSON despite instructions -- don't
        # silently drop the content. Show it as plain text so the run still
        # produces something reviewable instead of a blank/broken email.
        main.log.error(
            "Could not parse JSON from LLM output; falling back to raw text display."
        )
        sans = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
        analysis_html = (
            f'<div style="font-family:{sans};font-size:12px;color:#8B2E2E;margin-bottom:8px;">'
            f"Note: the model's response could not be parsed as structured data; showing raw output below.</div>"
            f'<pre style="white-space:pre-wrap;font-family:{sans};font-size:12px;color:#14213D;">{html.escape(_strip_code_fences(analysis))}</pre>'
        )
    email_html = build_email_html(analysis_html, today_str, sources, used_live_search)

    if os.getenv("DRY_RUN", "false").lower() == "true":
        with open("swing_trade_report.html", "w") as f:
            f.write(email_html)
        main.log.info("DRY_RUN enabled -- wrote swing_trade_report.html instead of emailing.")
        return

    send_swing_trade_email(email_html)


if __name__ == "__main__":
    run()

# -----------------------------
# Real-time grounding status
# -----------------------------
# groq/compound / groq/compound-mini: autonomous live web search
# (Tavily-backed under the hood) via the same GROQ_API_KEY as main.py's own
# calls. executed_tools[].search_results gives back the actual URLs used --
# surfaced in the email under "Sources checked". This is the first path
# tried since main.init_llm_generator() picks Groq first when configured.
# In practice, compound's internal search/planning loop can burn through
# most of its shared 30,000 TPM budget on a single request on the free
# tier, so treat both compound variants as opportunistic, not reliable.
#
# Tavily direct + plain Groq synthesis (_try_tavily_plus_groq): fetches
# real search results via Tavily's own API directly (TAVILY_API_KEY, free
# tier, 1,000 searches/month, no card -- entirely separate from Groq's and
# Gemini's quotas), then hands them to a plain (non-orchestrating) Groq
# model call. This is genuinely live-grounded but doesn't carry compound's
# expensive internal tool-orchestration overhead, so it's the more reliable
# of the two Groq-adjacent paths on the free tier.
#
# Gemini grounded (_try_gemini_grounded): Google Search grounding via
# GOOGLE_API_KEY, a separate free-tier quota from both Groq and Tavily.
#
# Plain Groq / local fallback: non-grounded chat completions with no live
# internet access -- pure training-data reasoning. Only reached if every
# live-search path above fails, and even then REQUIRE_LIVE_DATA=true (the
# default) aborts the run rather than emailing that unverified output.