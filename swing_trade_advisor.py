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
# Qualifying-stock gate
# -----------------------------
# The model's own JSON output is not trusted at face value: _verify_stock_claims
# independently checks every mandatory filter from the prompt (uptrend, RSI/MACD,
# growth thresholds, risk:reward minimum, debt/ROE) against real data. A stock
# with any "hard" contradiction -- i.e. one where the independent check actively
# disagrees with the model's claim, not just "couldn't be verified" -- fails its
# own strategy's entry criteria and must not be recommended. REQUIRE_QUALIFYING_STOCK
# (default true) enforces that; set to "false" to restore the old behavior of
# reporting every candidate regardless of contradictions.
REQUIRE_QUALIFYING_STOCK = os.getenv("REQUIRE_QUALIFYING_STOCK", "true").lower() == "true"
# How many times to re-prompt the model (with the specific rejection reasons fed
# back in) before giving up and reporting "no qualifying trade found" instead of
# emailing a pick that fails its own criteria.
MAX_GENERATION_ATTEMPTS = int(os.getenv("MAX_GENERATION_ATTEMPTS", "3"))

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

Provide a ranked LIST of every stock you can genuinely find, via real current search, that satisfies every mandatory filter above -- up to 10 stocks, highest-conviction first, favoring diversity across the four strategies where multiple genuinely qualify. Do NOT pad the list to hit a target count or to always show two: only include a stock if it truly satisfies every mandatory parameter with real, verifiable current data. It is normal and expected for very few stocks (even zero) to qualify on a given day -- return exactly as many as genuinely qualify, no more.

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
    {{ ... repeat for each additional genuinely qualifying stock, same fields, up to 10 total ... }}
  ]
}}
"""


def build_retry_prompt(rejected):
    """
    Re-issues the original prompt but tells the model exactly which candidates
    were independently verified and rejected last attempt, and why -- so a
    retry doesn't just propose the same failing stocks again with a more
    confident-sounding rationale.
    """
    reject_lines = "\n".join(
        f"- {s.get('name')} ({s.get('ticker')}): " + "; ".join(_hard_contradictions(s))
        for s in rejected
    ) or "- (the previous attempt returned zero candidates)"
    return (
        f"{build_prompt()}\n\n"
        "IMPORTANT: a previous attempt at this exact request proposed stocks that "
        "FAILED independent verification against real market data on the mandatory "
        "filters above:\n"
        f"{reject_lines}\n\n"
        "Do not propose these same stocks again, and do not describe technicals "
        "(SMA position, RSI direction, MACD crossover) or growth figures in your "
        "rationale unless you are confident they are genuinely true right now -- "
        "your rationale will be checked against real data again. Find different "
        "candidates that actually satisfy every mandatory filter as of real current "
        "data. If, after genuinely searching, no real stock satisfies all mandatory "
        "filters right now, return fewer than two stocks (even an empty \"stocks\": "
        "[] array) rather than forcing a pick that doesn't qualify."
    )


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
        result = _try_groq_compound_model(prompt, "groq/compound", max_attempts=3)
        if result is not None:
            return result

        main.log.info("groq/compound unavailable -- trying groq/compound-mini...")
        result = _try_groq_compound_model(prompt, "groq/compound-mini", max_attempts=2)
        if result is not None:
            return result

        today_str = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%d %B %Y")
        result = _try_tavily_plus_groq(prompt, today_str)
        if result is not None:
            return result

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
                model="gemini-flash-latest",
                contents=prompt,
            )
            return response.text.strip(), [], False
        except Exception as e:
            main.log.error(f"Gemini swing-trade generation failed: {e}")

    local_backend = main.init_llm_generator(force_local=True)
    if local_backend == "local" and main.llm_pipeline is not None:
        text = _generate_local(prompt)
        if text:
            return text, [], False

    return None, [], False


def _generate_local(prompt):
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


def _is_request_too_large(exc):
    msg = str(exc)
    return "413" in msg or "request_too_large" in msg or "Request Entity Too Large" in msg


def _is_daily_quota_exceeded(exc):
    msg = str(exc)
    return "tokens per day" in msg or "TPD" in msg


def _parse_groq_retry_seconds(exc):
    match = re.search(r"try again in ([\d.]+)s", str(exc))
    if match:
        try:
            return float(match.group(1)) + 0.5
        except ValueError:
            return None
    return None


def _try_gemini_grounded(prompt):
    if main.gemini_client is None:
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
        used_live = bool(sources)
        return response.text.strip(), sources, used_live
    except Exception as e:
        main.log.error(f"Gemini grounded (live search) generation failed: {e}")
        return None


def _extract_groq_sources(response):
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
    cleaned = _strip_code_fences(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    stocks = data.get("stocks") if isinstance(data, dict) else None
    # An empty list is a deliberate, valid "no qualifying candidate" response
    # (the retry prompt explicitly asks for it) -- only a missing/non-list
    # "stocks" field is an actual parse failure.
    if not isinstance(stocks, list):
        return None
    return stocks


def _fetch_current_price(ticker):
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
    for stock in stocks:
        price, currency_symbol = _fetch_current_price(stock.get("ticker"))
        if price is not None:
            stock["current_price_display"] = f"{currency_symbol}{price:,.2f}"
        else:
            stock["current_price_display"] = None
    return stocks


def _parse_first_number(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"\d*\.?\d+", str(value))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _parse_max_number(value):
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
    Returns a list of (note_text, severity) tuples. severity is "hard" when
    the independently recomputed number actually contradicts the model's
    claim, "soft" when the claim simply couldn't be checked either way.
    """
    notes = []
    stop = _parse_first_number(stock.get("stop_loss_pct"))
    target = _parse_first_number(stock.get("target1_pct"))
    stated_ratio = stock.get("risk_reward_ratio")

    if stop is None or target is None or stop == 0:
        notes.append(("Could not verify risk:reward -- stop-loss or target1 missing/unparseable.", "soft"))
        stock["risk_reward_ratio_verified"] = None
        return notes

    true_ratio = round(target / stop, 1)
    stock["risk_reward_ratio_verified"] = f"1 : {true_ratio}"

    stated_match = re.search(r":\s*([-+]?\d*\.?\d+)", str(stated_ratio) if stated_ratio else "")
    stated_val = float(stated_match.group(1)) if stated_match else None

    if stated_val is None:
        notes.append((
            f"Reported risk:reward '{stated_ratio}' could not be parsed -- "
            f"recomputed value is 1 : {true_ratio}.", "soft"
        ))
    elif abs(stated_val - true_ratio) > 0.15:
        notes.append((
            f"Risk:reward mismatch -- model reported 1 : {stated_val}, but "
            f"target1 ({target}%) / stop-loss ({stop}%) actually gives 1 : {true_ratio}.", "hard"
        ))
    if true_ratio < 2.0:
        notes.append((f"Risk:reward of 1 : {true_ratio} is below the 1:2.5 minimum the prompt requires.", "hard"))
    return notes


def _fetch_weekly_technicals(ticker):
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
    """Returns a list of (note_text, severity) tuples -- see _verify_risk_reward."""
    ticker = (stock.get("ticker") or "").strip()
    if not ticker:
        return [("No ticker provided -- technicals could not be verified.", "soft")]

    tech = _fetch_weekly_technicals(ticker)
    if tech is None:
        return [("Technicals could not be independently verified (price history fetch failed).", "soft")]
    if tech.get("insufficient_history"):
        return [(
            f"Only {tech.get('weeks_available')} weeks of price history available -- "
            "not enough to verify the 50-week SMA; technical claims are unverified.", "soft"
        )]

    notes = []
    price = tech["latest_close"]
    if price < tech["sma20w"]:
        notes.append((f"Price ({price}) is BELOW the 20-week SMA ({tech['sma20w']}) -- contradicts the required uptrend filter.", "hard"))
    if price < tech["sma50w"]:
        notes.append((f"Price ({price}) is BELOW the 50-week SMA ({tech['sma50w']}) -- contradicts the required uptrend filter.", "hard"))

    if tech["rsi14w"] is not None:
        if tech["rsi14w"] >= 70:
            notes.append((f"Weekly RSI is {tech['rsi14w']} (>=70, overbought) -- contradicts the 'RSI below 70' requirement.", "hard"))
        if tech.get("rsi14w_prev") is not None and tech["rsi14w"] < tech["rsi14w_prev"]:
            notes.append((f"Weekly RSI is falling ({tech['rsi14w_prev']} to {tech['rsi14w']}), not rising as the strategy requires.", "soft"))

    if tech["macd"] < tech["macd_signal"]:
        notes.append(("MACD line is currently below its signal line -- no bullish crossover in effect right now.", "hard"))

    return notes


def _fetch_fundamentals(ticker):
    """
    Fetches point-in-time fundamentals (debt/equity, ROE) plus a quarterly
    revenue/net-income series directly via yfinance -- independent of
    whatever the LLM claimed about the company's financials. Returns a
    dict (fields may be None if unavailable) or None if the fetch fails
    entirely / yfinance isn't installed.
    """
    ticker = (ticker or "").strip()
    if not ticker:
        return None
    try:
        import yfinance as yf
    except ImportError:
        main.log.warning("yfinance not installed -- fundamentals verification skipped.")
        return None
    try:
        yt = yf.Ticker(ticker)
        info = yt.info or {}
        result = {
            "debt_to_equity": info.get("debtToEquity"),
            "roe": info.get("returnOnEquity"),
            "revenue_growth_yoy": None,
            "profit_growth_yoy": None,
        }
        try:
            qf = yt.quarterly_financials  # rows = line items, cols = quarter-end dates, most-recent first
            if qf is not None and not qf.empty and qf.shape[1] >= 5:
                revenue_row = next((r for r in qf.index if "total revenue" in r.lower()), None)
                income_row = next((r for r in qf.index if r.lower() == "net income"), None)
                if revenue_row is not None:
                    latest, year_ago = qf.loc[revenue_row].iloc[0], qf.loc[revenue_row].iloc[4]
                    if year_ago and year_ago != 0 and pd.notna(latest) and pd.notna(year_ago):
                        result["revenue_growth_yoy"] = round(((latest - year_ago) / abs(year_ago)) * 100, 1)
                if income_row is not None:
                    latest, year_ago = qf.loc[income_row].iloc[0], qf.loc[income_row].iloc[4]
                    if year_ago and year_ago != 0 and pd.notna(latest) and pd.notna(year_ago):
                        result["profit_growth_yoy"] = round(((latest - year_ago) / abs(year_ago)) * 100, 1)
        except Exception as e:
            main.log.warning(f"Could not compute quarterly growth for '{ticker}': {e}")
        return result
    except Exception as e:
        main.log.warning(f"Could not fetch fundamentals for '{ticker}': {e}")
        return None


def _verify_fundamentals(stock):
    """
    Independently checks the prompt's mandatory fundamentals filters
    (low debt-to-equity, high/improving ROE, >=20% YoY revenue and
    profit growth) against real data instead of trusting the model's
    fundamental narrative. Returns a list of (note_text, severity) tuples.
    """
    ticker = (stock.get("ticker") or "").strip()
    if not ticker:
        return [("No ticker provided -- fundamentals could not be verified.", "soft")]

    data = _fetch_fundamentals(ticker)
    if data is None:
        return [("Fundamentals could not be independently verified (data fetch failed).", "soft")]

    notes = []

    dte = data.get("debt_to_equity")
    if dte is not None:
        if dte > 100:
            notes.append((f"Debt-to-equity is {dte:.0f}% -- elevated, contradicts the 'low debt-to-equity' requirement.", "hard"))
    else:
        notes.append(("Debt-to-equity not available from data provider -- unverified.", "soft"))

    roe = data.get("roe")
    if roe is not None:
        roe_pct = roe * 100 if abs(roe) <= 1 else roe
        if roe_pct < 10:
            notes.append((f"ROE is {roe_pct:.1f}% -- weak, contradicts the 'high/improving ROCE/ROE' requirement.", "hard"))
    else:
        notes.append(("ROE not available from data provider -- unverified.", "soft"))

    rev_g = data.get("revenue_growth_yoy")
    if rev_g is not None:
        if rev_g < 20:
            notes.append((f"Revenue growth YoY is {rev_g}% -- below the 20% threshold the prompt requires.", "hard"))
    else:
        notes.append(("Revenue YoY growth could not be computed (insufficient quarterly history from data provider).", "soft"))

    profit_g = data.get("profit_growth_yoy")
    if profit_g is not None:
        if profit_g < 20:
            notes.append((f"Net profit growth YoY is {profit_g}% -- below the 20% threshold the prompt requires.", "hard"))
    else:
        notes.append(("Net profit YoY growth could not be computed (insufficient quarterly history from data provider).", "soft"))

    return notes


def _verify_sanity_bounds(stock):
    """Returns a list of (note_text, severity) tuples -- see _verify_risk_reward."""
    notes = []

    conf = _parse_first_number(stock.get("confidence_score"))
    if conf is None:
        notes.append(("Confidence score missing or unparseable.", "soft"))
    elif not (0 <= conf <= 10):
        notes.append((f"Confidence score {conf} is outside the expected 0-10 range.", "soft"))

    alloc = _parse_max_number(stock.get("allocation_pct"))
    if alloc is not None and alloc > 15:
        notes.append((f"Allocation up to {alloc}% in a single stock is a large concentration for one position -- double-check position sizing.", "soft"))

    upside = _parse_first_number(stock.get("upside_target_pct"))
    if upside is not None and upside > 60:
        notes.append((f"Upside target of {upside}% in a 3-5 month window is unusually aggressive -- treat as a stretch case, not a base case.", "soft"))

    risk_level = (stock.get("risk_level") or "").strip().lower()
    if risk_level not in ("medium", "high"):
        notes.append((f"Risk level '{stock.get('risk_level')}' is not one of the expected 'Medium'/'High' values.", "soft"))

    return notes


def _verify_stock_claims(stocks):
    for stock in stocks:
        notes = []
        notes += _verify_risk_reward(stock)
        notes += _verify_technicals(stock)
        notes += _verify_fundamentals(stock)
        notes += _verify_sanity_bounds(stock)
        if not stock.get("current_price_display"):
            notes.append(("Live quote lookup failed for this ticker -- confirm it's a real, currently-listed symbol before trading it.", "soft"))
        stock["_verification_notes"] = notes
        _adjust_confidence(stock, notes)
    return stocks


def _adjust_confidence(stock, notes):
    """
    Derives an adjusted confidence score from the model's self-reported
    one, penalizing it for what verification actually found: "hard"
    contradictions (price below a required SMA, RSI failing the
    threshold, risk:reward under the stated minimum, weak fundamentals)
    cost more than "soft" ones (a claim that simply couldn't be checked
    either way). This stops a high self-reported score from being shown
    at face value when the independent checks disagree with it -- the
    email displays the adjusted number as the headline figure, with the
    model's original score and the reason for the gap shown alongside it.
    """
    original = _parse_first_number(stock.get("confidence_score"))
    stock["confidence_score_original"] = original
    if original is None:
        stock["confidence_score_adjusted"] = None
        return

    hard_count = sum(1 for _, sev in notes if sev == "hard")
    soft_count = sum(1 for _, sev in notes if sev == "soft")
    penalty = hard_count * 0.8 + soft_count * 0.2
    stock["confidence_score_adjusted"] = max(0.0, round(original - penalty, 1))
    stock["_confidence_penalty_detail"] = f"{hard_count} contradiction(s), {soft_count} unverifiable item(s)"


def _hard_contradictions(stock):
    """Text of every 'hard' verification note -- i.e. an independent check that
    actively disagrees with the model's claim, as opposed to a 'soft' note where
    something simply couldn't be verified either way."""
    return [n for n, sev in (stock.get("_verification_notes") or []) if sev == "hard"]


def _split_qualifying(stocks):
    """
    Splits verified stocks into (qualifying, rejected). A stock qualifies only if
    it has zero 'hard' contradictions -- i.e. nothing in its own strategy's
    mandatory filters (uptrend, RSI/MACD, growth thresholds, risk:reward minimum,
    debt/ROE) was independently found to be false. 'Soft' notes (couldn't be
    verified) are still disclosed in the report but don't block a recommendation.
    """
    qualifying, rejected = [], []
    for s in stocks:
        (rejected if _hard_contradictions(s) else qualifying).append(s)
    return qualifying, rejected


def _verification_display(stock):
    notes = stock.get("_verification_notes") or []
    if not notes:
        return '<span style="color:#2F5233;font-weight:700;">Verified -- no contradictions found</span>'
    hard = [n for n, sev in notes if sev == "hard"]
    soft = [n for n, sev in notes if sev == "soft"]
    header_bits = []
    if hard:
        header_bits.append(f"{len(hard)} contradiction(s)")
    if soft:
        header_bits.append(f"{len(soft)} unverifiable")
    header = f'<span style="color:#8B2E2E;font-weight:700;">{", ".join(header_bits)}:</span>'
    items = "".join(
        f'<div style="margin-top:3px;color:{"#8B2E2E" if sev == "hard" else "#A6812F"};">- {html.escape(n)}</div>'
        for n, sev in notes
    )
    return header + items


def _stars(s):
    s = max(0.0, min(10.0, s))
    filled = max(0, min(5, round(s / 2)))
    return "⭐" * filled + "☆" * (5 - filled)


def _confidence_display(stock):
    original = stock.get("confidence_score_original")
    if original is None:
        # Fallback for callers that haven't run verification (shouldn't happen
        # in the normal render_stock_table_html flow, but keeps this safe).
        original = _parse_first_number(stock.get("confidence_score"))
        if original is None:
            return None
    adjusted = stock.get("confidence_score_adjusted")
    if adjusted is None or adjusted == original:
        return f"{_stars(original)} ({original:.1f}/10)"
    detail = stock.get("_confidence_penalty_detail", "")
    return (
        f'{_stars(adjusted)} <strong>({adjusted:.1f}/10 adjusted)</strong>'
        f'<div style="margin-top:2px;font-size:11px;color:#8A8F9C;">'
        f'Model self-reported {original:.1f}/10 &middot; lowered for {html.escape(detail)}</div>'
    )


def _risk_level_badge(level):
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


def _render_one_stock_card(stock, idx, sans):
    def esc(v):
        v = "" if v is None else str(v).strip()
        return html.escape(v) if v else "—"

    def row(label, key, value_color="#14213D", bold=False):
        weight = "font-weight:700;" if bold else ""
        return (
            f'<tr><td style="padding:6px 10px;font-size:12px;font-family:{sans};'
            f'color:#4A5063;border-top:1px solid #EDEAE2;width:38%;">{label}</td>'
            f'<td style="padding:6px 10px;font-size:12px;{weight}font-family:{sans};'
            f'color:{value_color};border-top:1px solid #EDEAE2;">{esc(stock.get(key))}</td></tr>'
        )

    def raw_row(label, cell_html_fn):
        return (
            f'<tr><td style="padding:6px 10px;font-size:12px;font-family:{sans};'
            f'color:#4A5063;border-top:1px solid #EDEAE2;width:38%;">{label}</td>'
            f'<td style="padding:6px 10px;font-size:12px;font-family:{sans};'
            f'color:#14213D;border-top:1px solid #EDEAE2;">{cell_html_fn(stock) or "—"}</td></tr>'
        )

    rows = "".join([
        row("Current Market Price", "current_price_display", bold=True),
        raw_row("Confidence Score", _confidence_display),
        raw_row("Risk Level", lambda s: _risk_level_badge(s.get("risk_level"))),
        row("Key Catalysts", "key_catalysts"),
        row("Risk : Reward", "risk_reward_ratio", bold=True),
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

    name = esc(stock.get("name"))
    ticker = esc(stock.get("ticker"))
    rationale = esc(stock.get("rationale"))

    return f"""<div style="margin-top:{0 if idx == 0 else 22}px;">
<table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="border:1px solid #E7E4DC;border-radius:4px;overflow:hidden;border-collapse:collapse;">
<tr style="background:#14213D;"><td colspan="2" style="padding:9px 10px;font-family:{sans};font-size:11px;font-weight:700;color:#ffffff;text-transform:uppercase;letter-spacing:0.05em;">{idx + 1}. {name} <span style="color:#B08D57;">({ticker})</span></td></tr>
{rows}
</table>
<div style="margin-top:10px;font-family:{sans};font-size:12px;color:#4A5063;line-height:1.65;"><strong style="color:#14213D;">Investment Rationale:</strong> {rationale}</div>
{_trade_execution_plan_html(stock, sans)}
</div>"""


def render_stock_table_html(stocks):
    sans = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
    if not stocks:
        return _no_qualifying_stock_html([])
    return "".join(
        _render_one_stock_card(stock, idx, sans) for idx, stock in enumerate(stocks)
    )



def _no_qualifying_stock_html(rejected):
    """
    Rendered instead of a recommendation table when every candidate this run
    failed independent verification against its own strategy's mandatory
    filters (even after retries). Being honest that nothing qualified today is
    the correct output here -- forcing a pick that fails its own criteria is
    the bug this replaces.
    """
    sans = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
    out = (
        f'<div style="font-family:{sans};font-size:13px;color:#14213D;line-height:1.65;'
        f'padding:14px 16px;background:#F4F2ED;border-radius:4px;border:1px solid #E7E4DC;">'
        "<strong>No qualifying trade found for this run.</strong> Every candidate considered "
        "failed at least one of the strategy's mandatory filters (uptrend, rising RSI below 70, "
        "bullish MACD crossover, &ge;20% YoY revenue/profit growth, or &ge;1:2.5 risk:reward) once "
        "checked against independently-verified data, even after retrying with feedback. No pick is "
        "being reported rather than recommending one that fails its own entry criteria."
        "</div>"
    )
    if rejected:
        items = "".join(
            f'<div style="margin-top:8px;font-family:{sans};font-size:12px;color:#4A5063;">'
            f'<strong style="color:#14213D;">{html.escape(str(s.get("name") or s.get("ticker") or "Unnamed"))}</strong>'
            f' &mdash; rejected: {html.escape("; ".join(_hard_contradictions(s)) or "unspecified")}'
            "</div>"
            for s in rejected[-6:]
        )
        out += (
            f'<div style="margin-top:14px;padding-top:10px;border-top:1px solid #EDEAE2;'
            f'font-family:{sans};font-size:11px;font-weight:700;color:#8A8F9C;text-transform:uppercase;'
            f'letter-spacing:0.05em;">Candidates considered and rejected this run</div>{items}'
        )
    return out


def _trade_execution_plan_html(stock, sans):
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
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


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

    analysis_html = None
    sources = []
    used_live_search = False
    all_rejected = []
    qualifying = []
    analysis = None

    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        analysis, sources, used_live_search = generate_analysis(prompt)
        if not analysis:
            main.log.error(
                "No LLM backend produced output (no GROQ_API_KEY/GOOGLE_API_KEY set "
                "and local model unavailable/failed). Aborting without sending an email."
            )
            sys.exit(1)

        if not used_live_search and os.getenv("REQUIRE_LIVE_DATA", "true").lower() == "true":
            main.log.error(
                "Live web search was not used for this run (Groq's live-search "
                "model was unavailable or the backend fell back to Gemini/local), "
                "so the output would only reflect stale training-data prices/news. "
                "Aborting without sending an email. Set REQUIRE_LIVE_DATA=false to "
                "override and allow a clearly-labeled stale-data email instead."
            )
            sys.exit(1)

        stocks = _parse_analysis_json(analysis)
        if stocks is None:
            main.log.error(
                "Could not parse JSON from LLM output; falling back to raw text display."
            )
            sans = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
            analysis_html = (
                f'<div style="font-family:{sans};font-size:12px;color:#8B2E2E;margin-bottom:8px;">'
                f"Note: the model's response could not be parsed as structured data; showing raw output below.</div>"
                f'<pre style="white-space:pre-wrap;font-family:{sans};font-size:12px;color:#14213D;">{html.escape(_strip_code_fences(analysis))}</pre>'
            )
            break

        stocks = _attach_live_prices(stocks)
        stocks = _verify_stock_claims(stocks)
        qualifying, rejected = _split_qualifying(stocks)
        all_rejected.extend(rejected)

        if qualifying or not REQUIRE_QUALIFYING_STOCK:
            analysis_html = render_stock_table_html(qualifying or stocks)
            break

        if attempt < MAX_GENERATION_ATTEMPTS:
            main.log.info(
                f"Attempt {attempt}/{MAX_GENERATION_ATTEMPTS}: no candidate passed "
                f"independent verification of the mandatory filters ({len(rejected)} "
                "rejected) -- retrying with the specific rejection reasons fed back in."
            )
            prompt = build_retry_prompt(rejected)
        else:
            main.log.warning(
                f"All {MAX_GENERATION_ATTEMPTS} attempts failed to produce a stock that "
                "passes its own strategy's mandatory filters against real data. "
                "Reporting 'no qualifying trade' instead of a contradicted pick."
            )
            analysis_html = _no_qualifying_stock_html(all_rejected)

    email_html = build_email_html(analysis_html, today_str, sources, used_live_search)

    if os.getenv("DRY_RUN", "false").lower() == "true":
        with open("swing_trade_report.html", "w") as f:
            f.write(email_html)
        main.log.info("DRY_RUN enabled -- wrote swing_trade_report.html instead of emailing.")
        return

    send_swing_trade_email(email_html)


if __name__ == "__main__":
    run()