import os
import re
import math
import json
import time
import html
import requests
import yfinance as yf
import pandas as pd
import ta
import smtplib
import traceback
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from datetime import datetime
from zoneinfo import ZoneInfo
try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None
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
from config import *
from stock_fetcher import fetch_fundamentals
from fundamentals import score_fundamentals
from advanced_fundamentals import fetch_advanced_fundamentals, score_advanced_fundamentals
from market_context import build_market_context, get_resilient_session
from news_engine import get_news
from sentiment_score import score_headlines
from scorer import final_score, decision
from position_sizing import apply_risk_management
from recommendation_logic import choose_stock_entry
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from logger import log
from commodity_tracker import CommodityTracker
from compliance import build_compliance_block_html
from track_record import update_track_record, build_track_record_html
from support_resistance import compute_pivot_levels, compute_swing_zones, build_support_resistance_html

import threading

model_lock = threading.Lock()

llm_pipeline = None
use_gemini_flash = False
gemini_client = None
use_groq = False
groq_client = None

# -----------------------------
# Run-history persistence (enables signal-change tracking, stop/target
# breach alerts, and commodity buy-signal streaks across runs)
# -----------------------------
RUN_HISTORY_PATH = os.getenv("RUN_HISTORY_PATH", "run_history.json")

# How much (in INR) a hypothetical monthly silver savings scheme puts in,
# used purely to illustrate "how much silver would this buy today".
SILVER_MONTHLY_BUDGET = float(os.getenv("SILVER_MONTHLY_BUDGET", "5000"))

# If a single sector accounts for more than this % of a market's current
# Buy list, a concentration alert card is shown.
SECTOR_CONCENTRATION_THRESHOLD_PCT = float(os.getenv("SECTOR_CONCENTRATION_THRESHOLD_PCT", "40"))


def load_run_history():
    """
    Loads the previous run's state (per-stock signal/price/target/stop-loss,
    per-commodity buy-streak info) so the current run can diff against it.
    Returns a safe default structure if the file is missing or unreadable.
    """
    try:
        with open(RUN_HISTORY_PATH, "r") as f:
            data = json.load(f)
            data.setdefault("stocks", {})
            data.setdefault("commodities", {})
            data.setdefault("track_record", {"open": {}, "closed": []})
            return data
    except Exception:
        return {"stocks": {}, "commodities": {}, "track_record": {"open": {}, "closed": []}}


def save_run_history(history):
    """
    Persists the current run's state to disk so the next run can diff
    against it. NOTE: if this script runs on an ephemeral CI runner
    (e.g. GitHub Actions) without a persistent volume or cache step, this
    file will not survive between runs and change-tracking features will
    silently reset each time. Point RUN_HISTORY_PATH at a mounted/cached
    location in that case.
    """
    try:
        with open(RUN_HISTORY_PATH, "w") as f:
            json.dump(history, f, indent=2, default=str)
    except Exception as e:
        log.error(f"Failed to save run history to {RUN_HISTORY_PATH}: {e}")

def init_llm_generator(force_local=False):
    """
    Initializes the AI model.
    Priority: Groq (free tier, fast hosted inference on Llama/DeepSeek) if
    GROQ_API_KEY is present, then Gemini 2.5 Flash if GOOGLE_API_KEY is
    present, then falls back to the local Qwen2.5-1.5B model.

    force_local=True skips the Groq/Gemini checks entirely and goes
    straight to the local model. Use this when Groq/Gemini were already
    tried (e.g. by a prior call to this function) and exhausted their
    quota for the current request -- otherwise this function always
    re-picks Groq first because it only checks whether GROQ_API_KEY is
    set, not whether it still has quota left.
    """
    global llm_pipeline, use_gemini_flash, gemini_client, use_groq, groq_client

    if not force_local:
        # 1. Check for Groq API key -- fastest option, no local compute needed,
        # and (unlike the local model) safe to call concurrently from every
        # worker thread since there's no shared model/GPU to serialize on.
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

        # 2. Check for Gemini API key
        api_key = os.getenv("GOOGLE_API_KEY") or globals().get("GOOGLE_API_KEY")
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

    # 3. Fallback to local model
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

            log.info("Initializing high-quality local AI model (Qwen2.5-1.5B-Instruct)...")
            # Using Qwen2.5-1.5B-Instruct: extremely smart, fast on MPS/CPU, vastly superior reasoning
            llm_pipeline = pipeline(
                "text-generation", 
                model="Qwen/Qwen2.5-1.5B-Instruct",
                device=device,
                torch_dtype=torch_dtype
            )
            log.info("Local AI model initialized successfully.")
        except Exception as e:
            log.error(f"Failed to initialize local AI model: {e}")
            llm_pipeline = None
            
    return "local" if llm_pipeline is not None else None


# -----------------------------
# AI Stocks Story -- one combined, live-grounded call for ALL stocks
# -----------------------------
# Previously process_stock() made its own individual AI call per stock (a
# now-removed generate_llm_reasoning() that hit Groq/Gemini/local once per
# stock) -- N stocks meant N separate AI requests, each with its own prompt
# overhead. This section replaces every one of those individual calls with
# ONE combined call that returns a short, bulleted "AI Stock Story" for
# every stock in STOCKS at once, following the same live-search tiering used
# in swing_trade_advisor.py (groq/compound -> groq/compound-mini -> Tavily
# search + plain Groq -> Gemini + Google Search grounding -> plain Groq ->
# local model), just aimed at many short per-stock outputs instead of one
# long open-ended one. Far fewer AI round-trips and far fewer tokens spent
# on repeated prompt/instruction overhead for a portfolio of any size.
AI_STORY_BULLETS_PER_STOCK = 3

# Number of newspaper-digest-style bullet points the combined AI call writes
# for the portfolio-wide "AI Portfolio Story" summary (see
# _build_ai_stocks_story_prompt / build_ai_portfolio_story_html).
AI_PORTFOLIO_SUMMARY_POINTS = 5


def _tavily_search(query, max_results=2):
    """
    Runs one query against Tavily's search API directly and returns a list
    of {"title", "url", "content"} dicts, or [] on any failure / if
    TAVILY_API_KEY isn't set. Same free tier (1,000 searches/month, no
    card) swing_trade_advisor.py uses, entirely separate from Groq's and
    Gemini's own quotas.
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
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for r in data.get("results", []):
            results.append({
                "title": r.get("title") or r.get("url", ""),
                "url": r.get("url", ""),
                # Trimmed hard -- this only needs to ground the model in a
                # fact or two per stock, not hand it full articles. Keeping
                # this short is what keeps the *combined* prompt (every
                # stock's worth of context at once) inside a sane token
                # budget.
                "content": (r.get("content") or "")[:140],
            })
        return results
    except Exception as e:
        log.warning(f"Tavily search failed for query '{query}': {e}")
        return []


def _gather_ai_stocks_story_context(stock_names, today_str):
    """
    Runs one targeted Tavily query per stock (in parallel, reusing the
    same worker pool style as process_stock) and assembles a compact,
    per-stock-tagged context block plus a deduplicated source list.
    Returns ("", []) if Tavily isn't configured or nothing came back.
    """
    if not os.getenv("TAVILY_API_KEY"):
        return "", []

    sources = []
    blocks_by_stock = {}

    def _fetch_one(name):
        query = f"{name} share price news {today_str}"
        return name, _tavily_search(query, max_results=2)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_fetch_one, name) for name in stock_names]
        for future in as_completed(futures):
            try:
                name, results = future.result()
            except Exception as e:
                log.warning(f"Tavily lookup failed for a stock: {e}")
                continue
            for r in results:
                if not r["url"]:
                    continue
                if (r["title"], r["url"]) not in sources:
                    sources.append((r["title"], r["url"]))
                blocks_by_stock.setdefault(name, []).append(f"{r['title']}: {r['content']}")

    if not blocks_by_stock:
        return "", []

    lines = []
    for name in stock_names:
        snippets = blocks_by_stock.get(name)
        if snippets:
            lines.append(f"[{name}] " + " | ".join(snippets))
    context_text = (
        "LIVE NEWS SNIPPETS (real, freshly-fetched -- treat as your factual "
        "source for each stock; a stock with no snippet below has no fresh "
        "news, so keep its story brief and generic rather than inventing "
        "specifics):\n" + "\n".join(lines) + "\n\n"
    )
    return context_text, sources


def _build_ai_stocks_story_prompt(stock_names, context_text, today_str):
    names_block = "\n".join(f"- {n}" for n in stock_names)
    return (
        f"{context_text}"
        f"Today is {today_str}. For EACH stock listed below, write an \"AI Stock Story\": "
        f"exactly {AI_STORY_BULLETS_PER_STOCK} short bullet points (max 15 words each, plain "
        f"text, no sub-bullets, no markdown) covering, in order: "
        f"1) the single most important recent driver, catalyst, or news item, "
        f"2) the current momentum/sentiment read, "
        f"3) the key risk or watch-item. "
        f"Base each story on the live news snippets above where one exists for that stock; "
        f"otherwise give a brief, generic-but-accurate read rather than inventing specifics.\n\n"
        f"Then also write ONE \"Portfolio Summary\" laid out like the front-page digest of a "
        f"financial newspaper (think a Bloomberg/WSJ \"markets at a glance\" box), synthesizing "
        f"the picture ACROSS all the stocks together:\n"
        f"1) \"headline\": one punchy news-style headline (max 12 words, title case, no ending "
        f"period) capturing today's overall portfolio tone (risk-on/risk-off/mixed).\n"
        f"2) \"points\": exactly {AI_PORTFOLIO_SUMMARY_POINTS} short, scannable news-brief bullet "
        f"points (max 18 words each, plain text, no sub-bullets, no markdown, each reading like a "
        f"newspaper digest item), covering across all points: the overall tone, the one or two "
        f"most notable opportunities, and the one or two biggest risks or things to watch this "
        f"week. No generic filler, every point must say something specific.\n\n"
        f"Stocks:\n{names_block}\n\n"
        "OUTPUT FORMAT -- respond with ONLY raw JSON, nothing else (no markdown, no code "
        "fences, no commentary before or after):\n"
        "{\n"
        '  "stories": {\n'
        '    "<exact stock name as listed above>": ["bullet 1", "bullet 2", "bullet 3"],\n'
        "    ...\n"
        "  },\n"
        '  "portfolio_summary": {\n'
        '    "headline": "punchy news-style headline here",\n'
        '    "points": ["news-brief point 1", "news-brief point 2", "..."]\n'
        "  }\n"
        "}\n"
        "Every stock listed above must appear as a key, using the exact name given."
    )


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


def _parse_groq_retry_seconds(exc):
    """Groq's 429 body includes a 'Please try again in 7.342s' hint."""
    match = re.search(r"try again in ([\d.]+)s", str(exc))
    if match:
        try:
            return float(match.group(1)) + 0.5
        except ValueError:
            return None
    return None


def _strip_code_fences(text):
    """Models occasionally wrap the requested JSON in ```json ... ``` even
    when told not to -- strip that off so it parses cleanly."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def _normalize_portfolio_summary(raw_summary):
    """
    Coerces whatever the model returned for "portfolio_summary" into a
    consistent {"headline": str|None, "points": [str, ...]} dict, or None
    if there's nothing usable. Accepts:
      - the intended {"headline": ..., "points": [...]} shape
      - a bare list of strings (points, no headline)
      - a legacy plain-string paragraph (older prompt format / stale
        cache) -- split on sentence boundaries so it still renders as
        newspaper-style points instead of disappearing.
    """
    if isinstance(raw_summary, dict):
        headline = raw_summary.get("headline")
        headline = headline.strip() if isinstance(headline, str) and headline.strip() else None
        points = raw_summary.get("points")
        if isinstance(points, list):
            points = [str(p).strip() for p in points if str(p).strip()][:AI_PORTFOLIO_SUMMARY_POINTS]
        else:
            points = []
        if not headline and not points:
            return None
        return {"headline": headline, "points": points}

    if isinstance(raw_summary, list):
        points = [str(p).strip() for p in raw_summary if str(p).strip()][:AI_PORTFOLIO_SUMMARY_POINTS]
        return {"headline": None, "points": points} if points else None

    if isinstance(raw_summary, str) and raw_summary.strip():
        # Legacy prose paragraph -- split into sentence-ish points so old
        # cached/off-spec responses still render in the new points format.
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw_summary.strip()) if s.strip()]
        return {"headline": None, "points": sentences[:AI_PORTFOLIO_SUMMARY_POINTS]}

    return None


def _parse_ai_stocks_story_json(text, stock_names):
    """
    Parses the compact per-stock JSON the LLM returns (see
    _build_ai_stocks_story_prompt's OUTPUT FORMAT). Returns
    (stories, portfolio_summary):
      stories: {stock_name: [bullet, ...]} for whichever names it
        recognizes, matched back to the exact configured stock name
        case/whitespace-insensitively so lookups in process_stock are a
        plain dict hit. {} if nothing usable could be parsed.
      portfolio_summary: {"headline": str, "points": [str, ...]} -- the
        newspaper-digest-style summary synthesizing all stocks together,
        or None if missing/unparseable. Tolerates an older-style plain
        string (treated as a single point with no headline) so a stale
        cached response or an off-spec model reply doesn't just vanish.
    """
    cleaned = _strip_code_fences(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return {}, None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}, None

    stories = data.get("stories") if isinstance(data, dict) else None
    raw_summary = data.get("portfolio_summary") if isinstance(data, dict) else None
    portfolio_summary = _normalize_portfolio_summary(raw_summary)
    if not isinstance(stories, dict):
        return {}, portfolio_summary

    result = {}
    name_lookup = {n.strip().lower(): n for n in stock_names}
    for key, bullets in stories.items():
        if not isinstance(bullets, list):
            continue
        clean_bullets = [str(b).strip() for b in bullets if str(b).strip()][:AI_STORY_BULLETS_PER_STOCK]
        if not clean_bullets:
            continue
        matched_name = name_lookup.get(str(key).strip().lower(), key)
        result[matched_name] = clean_bullets
    return result, portfolio_summary


def _try_groq_compound_model_for_stories(prompt, model_name, stock_count, max_attempts=2):
    max_tokens = min(4200, max(850, stock_count * 70 + 150))
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
                f"Groq ({model_name}) AI Stocks Story generation failed "
                f"(attempt {attempt + 1}/{max_attempts}): {e}"
            )
            if _is_request_too_large(e):
                log.error(f"Request too large for {model_name} -- skipping further retries.")
                return None
            if _is_daily_quota_exceeded(e):
                log.error(f"Groq daily token quota exhausted for {model_name} -- skipping remaining retries.")
                return None
            if attempt < max_attempts - 1:
                wait_s = _parse_groq_retry_seconds(e) or 10
                log.info(f"Retrying {model_name} in {wait_s:.1f}s...")
                time.sleep(wait_s)
    return None


def _try_tavily_plus_groq_for_stories(prompt, stock_count):
    if not os.getenv("TAVILY_API_KEY") or groq_client is None:
        return None
    max_tokens = min(4200, max(850, stock_count * 70 + 150))
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip(), True
    except Exception as e:
        log.error(f"Groq synthesis over Tavily context failed for AI Stocks Story: {e}")
        return None


def _try_gemini_grounded_for_stories(prompt):
    """
    Genuine live-search fallback: Gemini's free tier supports real Google
    Search grounding (a separate free-tier quota from both Groq and
    Tavily). Returns (text, sources, True) on success, or None.
    """
    global gemini_client
    if gemini_client is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key or genai is None:
            return None
        try:
            gemini_client = genai.Client(api_key=api_key)
        except Exception as e:
            log.error(f"Could not lazily initialize Gemini client for AI Stocks Story: {e}")
            return None
    try:
        from google.genai import types
        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        response = gemini_client.models.generate_content(
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
            log.warning(f"Could not extract Gemini grounding sources: {e}")
        used_live = bool(sources)
        return response.text.strip(), sources, used_live
    except Exception as e:
        log.error(f"Gemini grounded AI Stocks Story generation failed: {e}")
        return None


def _generate_local_story(prompt):
    """Runs the combined-stories prompt through the local Qwen2.5-1.5B
    pipeline. Returns the generated text, or None if unavailable/failed."""
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
                max_new_tokens=1200,
                do_sample=True,
                temperature=0.4,
                top_k=50,
                top_p=0.95,
            )
        generated_text = outputs[0]["generated_text"]
        return generated_text.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
    except Exception as e:
        log.error(f"Local AI Stocks Story generation failed: {e}")
        return None


def generate_ai_stocks_story(stock_names):
    """
    Single, combined, live-grounded call that produces a short, bulleted
    "AI Stock Story" for every stock in `stock_names`, PLUS one
    professional-prose "Portfolio Summary" synthesizing all of them
    together -- all from the same one call. Tiering mirrors
    swing_trade_advisor.py: groq/compound -> groq/compound-mini -> Tavily
    search + plain Groq -> Gemini + Google Search grounding -> plain
    (non-live) Groq -> local model.

    Returns (stories, sources, used_live_search, portfolio_summary):
      stories: {stock_name: [bullet, bullet, bullet]} -- {} on total failure
      sources: [(title, url), ...] actually used, for optional display
      used_live_search: True if the tier that produced the result was
        genuinely grounded in live search results
      portfolio_summary: {"headline": str|None, "points": [str, ...]} --
        a newspaper-digest-style summary covering all stocks together, or
        a locally-built generic fallback in the same shape if every AI
        tier failed or omitted it
    """
    if not stock_names:
        return {}, [], False, None

    today_str = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%d %B %Y")
    stock_count = len(stock_names)

    backend = init_llm_generator()
    log.info(f"AI Stocks Story using LLM backend: {backend}")

    bare_prompt = _build_ai_stocks_story_prompt(stock_names, "", today_str)

    if backend == "groq":
        # Tier 1/2: groq/compound (and the lighter compound-mini) can
        # search the live web on their own -- no separate context needed.
        result = _try_groq_compound_model_for_stories(bare_prompt, "groq/compound", stock_count, max_attempts=2)
        if result is not None:
            text, sources, live = result
            stories, summary = _parse_ai_stocks_story_json(text, stock_names)
            if stories:
                return stories, sources, live, summary or _fallback_portfolio_summary(stock_names)

        log.info("groq/compound unavailable for AI Stocks Story -- trying groq/compound-mini...")
        result = _try_groq_compound_model_for_stories(bare_prompt, "groq/compound-mini", stock_count, max_attempts=2)
        if result is not None:
            text, sources, live = result
            stories, summary = _parse_ai_stocks_story_json(text, stock_names)
            if stories:
                return stories, sources, live, summary or _fallback_portfolio_summary(stock_names)

        # Tier 3: fetch live snippets via Tavily directly (own free quota,
        # one query per stock, run in parallel), then hand them to a
        # plain, non-orchestrating Groq call.
        context_text, tavily_sources = _gather_ai_stocks_story_context(stock_names, today_str)
        if context_text:
            grounded_prompt = _build_ai_stocks_story_prompt(stock_names, context_text, today_str)
            result = _try_tavily_plus_groq_for_stories(grounded_prompt, stock_count)
            if result is not None:
                text, live = result
                stories, summary = _parse_ai_stocks_story_json(text, stock_names)
                if stories:
                    return stories, tavily_sources, live, summary or _fallback_portfolio_summary(stock_names)

        # Tier 4: Gemini + Google Search grounding, a separate free quota.
        if gemini_client is not None or (os.getenv("GOOGLE_API_KEY") and genai is not None):
            grounded = _try_gemini_grounded_for_stories(bare_prompt)
            if grounded is not None:
                text, sources, live = grounded
                stories, summary = _parse_ai_stocks_story_json(text, stock_names)
                if stories:
                    return stories, sources, live, summary or _fallback_portfolio_summary(stock_names)

        # Tier 5: plain (non-search) Groq -- still one call for every
        # stock, just without live grounding.
        try:
            max_tokens = min(4200, max(850, stock_count * 70 + 150))
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": bare_prompt}],
                temperature=0.4,
                max_tokens=max_tokens,
            )
            text = response.choices[0].message.content.strip()
            stories, summary = _parse_ai_stocks_story_json(text, stock_names)
            if stories:
                return stories, [], False, summary or _fallback_portfolio_summary(stock_names)
        except Exception as e:
            log.error(f"Groq fallback (no search) AI Stocks Story generation failed: {e}")
            if _is_daily_quota_exceeded(e):
                log.error(
                    "Groq's daily token quota is exhausted for this org. "
                    "Falling back to the local model instead of retrying Groq."
                )

    elif backend == "gemini" or gemini_client is not None:
        grounded = _try_gemini_grounded_for_stories(bare_prompt)
        if grounded is not None:
            text, sources, live = grounded
            stories, summary = _parse_ai_stocks_story_json(text, stock_names)
            if stories:
                return stories, sources, live, summary or _fallback_portfolio_summary(stock_names)
        try:
            response = gemini_client.models.generate_content(
                model="gemini-flash-latest",
                contents=bare_prompt,
            )
            stories, summary = _parse_ai_stocks_story_json(response.text.strip(), stock_names)
            if stories:
                return stories, [], False, summary or _fallback_portfolio_summary(stock_names)
        except Exception as e:
            log.error(f"Gemini AI Stocks Story generation failed: {e}")

    # Last resort: local model -- still one combined call, not one per stock.
    local_backend = init_llm_generator(force_local=True)
    if local_backend == "local" and llm_pipeline is not None:
        text = _generate_local_story(bare_prompt)
        if text:
            stories, summary = _parse_ai_stocks_story_json(text, stock_names)
            if stories:
                return stories, [], False, summary or _fallback_portfolio_summary(stock_names)

    return {}, [], False, None


def _fallback_stock_story(stock_name, signal, tech_score, fund_score, sentiment_score, sentiment_label, headlines):
    """
    Cheap, non-AI 3-bullet fallback used when the batched AI Stocks Story
    call didn't produce an entry for this stock (e.g. every AI backend
    failed, or the model skipped a name). Computed instantly from data
    already on hand for this stock -- no extra AI call.
    """
    headline = headlines[0] if headlines else "No recent headlines available."
    if sentiment_label == "Data Unavailable":
        sentiment_line = "Sentiment: data unavailable -- news fetch failed this run."
    else:
        sentiment_line = f"Sentiment reads {sentiment_label.lower()} ({round(sentiment_score, 1)})."
    return [
        f"Signal: {signal}, technical score {tech_score}, fundamentals {fund_score}.",
        sentiment_line,
        f"Latest headline: {headline}",
    ]


def _fallback_portfolio_summary(stock_names):
    """
    Cheap, non-AI fallback used when the AI backend produced per-stock
    stories but the model omitted (or mis-formatted) the portfolio_summary
    field -- keeps the email section populated with an honest, generic
    line instead of silently disappearing. No extra AI call. Returned in
    the same {"headline", "points"} shape as the AI-generated summary.
    """
    return {
        "headline": "Portfolio Commentary Unavailable This Run",
        "points": [
            f"Automated portfolio-wide commentary was not returned for the "
            f"{len(stock_names)} tracked stocks this run.",
            "See the per-stock AI Stock Story notes and signal/score "
            "breakdown below for the underlying read on each name.",
        ],
    }


def build_ai_story_bullets_html(bullets):
    """Renders a stock's AI Stock Story as a labeled section with a compact
    <ul> underneath, HTML-escaping each bullet since the text may come from
    an LLM. Returns "" (nothing rendered) if there are no bullets."""
    if not bullets:
        return ""
    items = "".join(f"<li style=\"margin:0 0 4px 0;\">{html.escape(b)}</li>" for b in bullets)
    label = (
        '<div style="font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',Helvetica,Arial,'
        'sans-serif;font-size:10px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;'
        'color:#B08D57;margin:0 0 4px;">🤖 AI Stock Story</div>'
    )
    return f'{label}<ul style="margin:0;padding-left:16px;">{items}</ul>'


def build_ai_portfolio_story_html(portfolio_summary, stock_count, used_live_search):
    """
    Small, email-safe section combining every stock's AI Stock Story into
    a newspaper-digest-style briefing (kicker, headline, ruled bullet
    points) built from generate_ai_stocks_story's portfolio_summary --
    from the same single combined AI call, not a separate AI request.
    Shown near the top of BOTH the email body and the PDF, unlike the
    full per-stock cards which are PDF-only.

    portfolio_summary is {"headline": str|None, "points": [str, ...]}
    (see _normalize_portfolio_summary). Returns "" if there's nothing to
    show (no headline and no points).
    """
    if not portfolio_summary:
        return ""
    headline = portfolio_summary.get("headline") if isinstance(portfolio_summary, dict) else None
    points = portfolio_summary.get("points") if isinstance(portfolio_summary, dict) else None
    points = [p for p in (points or []) if p]
    if not headline and not points:
        return ""

    sans = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
    serif = "Georgia,'Times New Roman',Times,serif"
    today_str = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%d %b %Y").upper()
    live_tag = " &nbsp;&middot;&nbsp; LIVE-GROUNDED" if used_live_search else ""

    headline_html = ""
    if headline:
        headline_html = (
            f'<div style="margin:6px 0 8px;font-family:{serif};font-size:18px;'
            f'font-weight:700;line-height:1.3;color:#1F2430;">{html.escape(headline)}</div>'
        )

    points_html = ""
    if points:
        items = "".join(
            f'<tr><td style="padding:0 8px 6px 0;font-family:{serif};font-size:14px;'
            f'line-height:1;color:#B08D57;vertical-align:top;">&#9642;</td>'
            f'<td style="padding:0 0 6px;font-family:{sans};font-size:13px;line-height:1.55;'
            f'color:#3C4256;">{html.escape(p)}</td></tr>'
            for p in points
        )
        points_html = f'<table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="margin-top:4px;">{items}</table>'

    return f"""
        <tr>
          <td style="padding:0 28px 18px;" class="email-padding">
            <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="border-radius:6px;background:#FAF8F1;border:1px solid #E7DFC9;">
              <tr>
                <td style="padding:14px 16px 16px;">
                  <div style="font-family:{sans};font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#B08D57;">🤖 AI Portfolio Story &nbsp;&middot;&nbsp; {today_str}{live_tag}</div>
                  {headline_html}
                  <div style="border-top:1px solid #E7DFC9;margin:2px 0 8px;"></div>
                  {points_html}
                  <p style="margin:10px 0 0;font-family:{sans};font-size:10px;color:#8A8F9C;">Synthesized in a single combined AI pass across all {stock_count} tracked stocks &mdash; see each stock's card for its individual AI Stock Story.</p>
                </td>
              </tr>
            </table>
          </td>
        </tr>
    """


def fetch_data(symbol):
    df = yf.download(
        symbol,
        period="300d",
        interval="1d",
        auto_adjust=True,
        progress=False,
        session=get_resilient_session()
    )

    if df.empty:
        raise Exception(f"No data for {symbol}")

    df.reset_index(inplace=True)

    # flatten columns if MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        cols = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    else:
        cols = list(df.columns)

    normalized = []
    for col in cols:
        name = str(col).lower().strip()
        name = name.replace(" ", "").replace("-", "").replace("_", "")
        normalized.append(name)

    df.columns = normalized

    if "close" not in df.columns:
        source_close = None
        for candidate in df.columns:
            if "close" in candidate:
                source_close = candidate
                break
        if source_close is not None:
            df["close"] = df[source_close]

    if "close" not in df.columns:
        raise Exception(
            f"Missing required 'close' column after normalization for {symbol}. "
            f"Available columns: {', '.join(df.columns)}"
        )

    return df


# -----------------------------
# Indicator calculations
# -----------------------------
def calculate_indicators(df):
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema100"] = df["close"].ewm(span=100, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    rsi_indicator = ta.momentum.RSIIndicator(
        close=df["close"],
        window=14
    )
    df["rsi"] = rsi_indicator.rsi()

    df["vol_avg"] = df["volume"].rolling(20).mean()

    df["macd"] = df["close"].ewm(span=12, adjust=False).mean() - df["close"].ewm(span=26, adjust=False).mean()
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    df["adx"] = ta.trend.ADXIndicator(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=14
    ).adx()

    df["atr"] = ta.volatility.AverageTrueRange(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=14
    ).average_true_range()

    df["high20"] = df["high"].rolling(20).max()
    df["low20"] = df["low"].rolling(20).min()

    return df


# -----------------------------
# Scoring engine
# -----------------------------
def calculate_score(df):
    latest = df.iloc[-1]
    score = 0
    reason = []

    price = latest["close"]
    ema20 = latest["ema20"]
    ema50 = latest["ema50"]
    ema100 = latest["ema100"]
    ema200 = latest["ema200"]
    rsi = latest["rsi"]
    macd = latest["macd"]
    macd_signal = latest["macd_signal"]
    macd_hist = latest["macd_hist"]
    adx = latest["adx"]

    if price > ema200:
        score += 20
        reason.append("Above EMA200")
    if price > ema100:
        score += 15
        reason.append("Above EMA100")
    if price > ema50:
        score += 10
        reason.append("Above EMA50")
    if price > ema20:
        score += 10
        reason.append("Above EMA20")

    if ema20 > ema50 > ema100 > ema200:
        score += 20
        reason.append("Strong trend structure")
    elif ema50 > ema100 > ema200:
        score += 10
        reason.append("Uptrend confirmed")
    elif ema20 > ema50 and ema50 > ema200:
        score += 5
        reason.append("Bullish alignment")

    if rsi <= 60:
        score += 20
        if rsi < 40:
            reason.append("RSI low (swing entry zone)")
        else:
            reason.append("RSI healthy")
    elif rsi <= 70:
        score += 10
        reason.append("RSI moderate")
    else:
        score -= 5
        reason.append("RSI high")

    if macd > macd_signal:
        score += 10
        reason.append("MACD bullish")
        if macd_hist > 0:
            score += 5
            reason.append("MACD momentum rising")
    else:
        reason.append("MACD bearish")

    if adx >= 25:
        score += 10
        reason.append("ADX strong trend")
    elif adx >= 20:
        score += 5
        reason.append("ADX trend developing")
    else:
        reason.append("ADX weak")

    if latest["volume"] > latest["vol_avg"]:
        score += 10
        reason.append("Volume strong")

    if latest["high20"] and price >= latest["high20"] * 0.98:
        score += 5
        reason.append("Near 20-day breakout")
    elif latest["low20"] and price < latest["low20"] * 1.02 and price > ema20:
        score += 3
        reason.append("Healthy pullback")

    return score, reason


    # -----------------------------
    # Signal

def get_signal(score):
    if score >= 80:
        return "GREEN -> BUY / ADD"
    elif score >= 50:
        return "YELLOW -> HOLD"
    else:
        return "RED -> SELL / REDUCE"


def get_conviction_rating(score):
    try:
        numeric_score = float(score)
    except (TypeError, ValueError):
        numeric_score = 0.0

    numeric_score = max(0.0, min(100.0, numeric_score))

    if numeric_score >= 90:
        filled_icons = 5
        label = "Elite"
        fill_color = "#2F5233"
    elif numeric_score >= 75:
        filled_icons = 4
        label = "High"
        fill_color = "#3D6B44"
    elif numeric_score >= 60:
        filled_icons = 3
        label = "Strong"
        fill_color = "#4A7A6B"
    elif numeric_score >= 45:
        filled_icons = 2
        label = "Mixed"
        fill_color = "#A6812F"
    elif numeric_score >= 30:
        filled_icons = 1
        label = "Low"
        fill_color = "#9C4A2E"
    else:
        filled_icons = 0
        label = "Very Low"
        fill_color = "#8B2E2E"

    icons = []
    icons_text = []
    for index in range(5):
        active = index < filled_icons
        icon_background = fill_color if active else "#f8fafc"
        icon_border = fill_color if active else "#dbe3ea"
        icon_color = "#ffffff" if active else "#cbd5e1"
        glyph = "★" if active else "☆"
        icons_text.append(glyph)
        icons.append(
            f'<span style="display:inline-block;width:26px;height:26px;'
            f'line-height:26px;text-align:center;border-radius:999px;'
            f'margin-left:2px;margin-right:2px;font-size:15px;font-weight:900;'
            f'background:{icon_background};border:1px solid {icon_border};'
            f'color:{icon_color};box-shadow:0 1px 2px rgba(15,23,42,0.08);">'
            f'{glyph}</span>'
        )

    return {
        "label": label,
        "fill_color": fill_color,
        "icons_html": "".join(icons),
        "icons_text": "".join(icons_text),
    }


def _safe_float(value):
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None

    if pd.isna(numeric_value):
        return None

    return numeric_value


def _risk_level_meta(score):
    if score <= 1.5:
        return {
            "label": "Low",
            "emoji": "🟢",
            "color": "#047857",
            "background": "#dcfce7",
            "border": "#86efac",
        }
    if score <= 2.25:
        return {
            "label": "Medium",
            "emoji": "🟡",
            "color": "#a16207",
            "background": "#fef3c7",
            "border": "#fcd34d",
        }
    return {
        "label": "High",
        "emoji": "🔴",
        "color": "#b91c1c",
        "background": "#fee2e2",
        "border": "#fca5a5",
    }


def _risk_component_score(value, low_threshold, medium_threshold, reverse=False):
    if value is None:
        return 2

    if reverse:
        if value >= low_threshold:
            return 1
        if value >= medium_threshold:
            return 2
        return 3

    if value <= low_threshold:
        return 1
    if value <= medium_threshold:
        return 2
    return 3


def calculate_risk_meter(df, latest, beta=None):
    close = _safe_float(latest.get("close"))
    atr = _safe_float(latest.get("atr"))
    adx = _safe_float(latest.get("adx"))
    beta_value = _safe_float(beta)

    atr_pct = round((atr / close) * 100, 2) if atr is not None and close not in (None, 0) else None

    volatility_pct = None
    recent_returns = df["close"].pct_change().dropna().tail(20)
    if len(recent_returns) >= 2:
        recent_volatility = recent_returns.std()
        if pd.notna(recent_volatility):
            volatility_pct = round(float(recent_volatility) * math.sqrt(252) * 100, 2)

    atr_score = _risk_component_score(atr_pct, 2.0, 4.0)
    volatility_score = _risk_component_score(volatility_pct, 20.0, 35.0)
    beta_score = _risk_component_score(beta_value, 0.9, 1.2)
    adx_score = _risk_component_score(adx, 25.0, 18.0, reverse=True)

    overall_score = round((atr_score + volatility_score + beta_score + adx_score) / 4, 2)
    overall_meta = _risk_level_meta(overall_score)

    factor_rows = [
        {
            "label": "ATR",
            "value": f"{atr_pct:.2f}%" if atr_pct is not None else "n/a",
            "meta": _risk_level_meta(atr_score),
        },
        {
            "label": "Volatility",
            "value": f"{volatility_pct:.2f}%" if volatility_pct is not None else "n/a",
            "meta": _risk_level_meta(volatility_score),
        },
        {
            "label": "Beta",
            "value": f"{beta_value:.2f}" if beta_value is not None else "n/a",
            "meta": _risk_level_meta(beta_score),
        },
        {
            "label": "ADX",
            "value": f"{adx:.2f}" if adx is not None else "n/a",
            "meta": _risk_level_meta(adx_score),
        },
    ]

    return {
        "label": overall_meta["label"],
        "emoji": overall_meta["emoji"],
        "color": overall_meta["color"],
        "background": overall_meta["background"],
        "border": overall_meta["border"],
        "score": overall_score,
        "factors": factor_rows,
    }


def build_risk_meter_html(risk_meter):
    def _factor_cell(factor, right=False):
        pad = "padding:8px 0 0 10px;" if right else "padding:8px 10px 0 0;"
        return (
            f'<td style="{pad}width:50%;vertical-align:top;">'
            f'<div style="font-size:11px;color:#64748b;font-weight:700;text-transform:uppercase;letter-spacing:0.03em;">'
            f'{factor["label"]}</div>'
            f'<div style="margin-top:3px;color:#0f172a;font-size:13px;font-weight:700;">'
            f'{factor["value"]}</div>'
            f'<div style="margin-top:2px;font-size:11px;color:{factor["meta"]["color"]};font-weight:700;">'
            f'{factor["meta"]["emoji"]} {factor["meta"]["label"]}</div></td>'
        )

    factors = risk_meter["factors"]

    return f"""
                            <tr>
                                <td colspan="2" style="padding-top:10px;border-top:1px solid #eef2f7;">
                                    <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="border-collapse:collapse;">
                                        <tr>
                                            <td style="vertical-align:middle;">
                                                <div style="font-size:13px;color:#475569;font-weight:700;">Risk Meter</div>
                                                <div style="margin-top:2px;font-size:11px;color:#94a3b8;">ATR &bull; Volatility &bull; Beta &bull; ADX</div>
                                            </td>
                                            <td style="text-align:right;vertical-align:middle;">
                                                <span style="display:inline-block;padding:5px 12px;border-radius:999px;background:{risk_meter['background']};border:1px solid {risk_meter['border']};color:{risk_meter['color']};font-size:12px;font-weight:800;white-space:nowrap;">{risk_meter['emoji']} {risk_meter['label']} Risk</span>
                                            </td>
                                        </tr>
                                    </table>
                                    <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="margin-top:8px;border-collapse:collapse;">
                                        <tr>
                                            {_factor_cell(factors[0])}
                                            {_factor_cell(factors[1], right=True)}
                                        </tr>
                                        <tr>
                                            {_factor_cell(factors[2])}
                                            {_factor_cell(factors[3], right=True)}
                                        </tr>
                                    </table>
                                </td>
                            </tr>
    """


def calculate_52_week_range(df, latest):
    current_price = _safe_float(latest.get("close"))
    recent_history = df.tail(252)

    high_52w = _safe_float(recent_history["high"].max()) if "high" in recent_history else None
    low_52w = _safe_float(recent_history["low"].min()) if "low" in recent_history else None

    below_high_pct = None
    if current_price is not None and high_52w not in (None, 0):
        below_high_pct = round(((current_price - high_52w) / high_52w) * 100, 2)

    above_low_pct = None
    if current_price is not None and low_52w not in (None, 0):
        above_low_pct = round(((current_price - low_52w) / low_52w) * 100, 2)

    if below_high_pct is None:
        high_distance_text = "n/a"
        high_distance_color = "#64748b"
    elif below_high_pct <= 0:
        high_distance_text = f"↓ {abs(below_high_pct):.1f}% below high"
        high_distance_color = "#dc2626" if below_high_pct <= -15 else "#d97706"
    else:
        high_distance_text = f"↑ {below_high_pct:.1f}% above high"
        high_distance_color = "#047857"

    if above_low_pct is None:
        low_distance_text = "n/a"
        low_distance_color = "#64748b"
    elif above_low_pct >= 0:
        low_distance_text = f"↑ {above_low_pct:.1f}% above low"
        low_distance_color = "#047857" if above_low_pct >= 20 else "#d97706"
    else:
        low_distance_text = f"↓ {abs(above_low_pct):.1f}% below low"
        low_distance_color = "#dc2626"

    return {
        "current_price": current_price,
        "high_52w": high_52w,
        "low_52w": low_52w,
        "high_distance_text": high_distance_text,
        "high_distance_color": high_distance_color,
        "low_distance_text": low_distance_text,
        "low_distance_color": low_distance_color,
    }


def _format_rupee(value):
    if value is None:
        return "n/a"
    return f"₹{value:.2f}"


def build_52_week_range_html(range_data):
    return f"""
                            <tr>
                                <td colspan="2" style="padding-top:10px;border-top:1px solid #eef2f7;">
                                    <!-- Section heading row -->
                                    <div style="font-size:13px;color:#475569;font-weight:700;">Distance from 52-Week High / Low</div>
                                    <!-- High | Current | Low three-column price display -->
                                    <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="margin-top:10px;border-collapse:collapse;">
                                        <tr>
                                            <!-- 52W High -->
                                            <td style="width:33.33%;vertical-align:top;padding-right:6px;">
                                                <div style="padding:10px;border-radius:10px;background:#f0fdf4;border:1px solid #bbf7d0;text-align:center;">
                                                    <div style="font-size:10px;color:#16a34a;font-weight:700;text-transform:uppercase;letter-spacing:0.04em;">52W High</div>
                                                    <div style="margin-top:5px;color:#0f172a;font-size:14px;font-weight:800;">{_format_rupee(range_data['high_52w'])}</div>
                                                </div>
                                            </td>
                                            <!-- Current Price -->
                                            <td style="width:33.33%;vertical-align:top;padding:0 3px;">
                                                <div style="padding:10px;border-radius:10px;background:#f8fafc;border:1px solid #e2e8f0;text-align:center;">
                                                    <div style="font-size:10px;color:#475569;font-weight:700;text-transform:uppercase;letter-spacing:0.04em;">Current</div>
                                                    <div style="margin-top:5px;color:#0f172a;font-size:14px;font-weight:800;">{_format_rupee(range_data['current_price'])}</div>
                                                </div>
                                            </td>
                                            <!-- 52W Low -->
                                            <td style="width:33.33%;vertical-align:top;padding-left:6px;">
                                                <div style="padding:10px;border-radius:10px;background:#fef2f2;border:1px solid #fecaca;text-align:center;">
                                                    <div style="font-size:10px;color:#dc2626;font-weight:700;text-transform:uppercase;letter-spacing:0.04em;">52W Low</div>
                                                    <div style="margin-top:5px;color:#0f172a;font-size:14px;font-weight:800;">{_format_rupee(range_data['low_52w'])}</div>
                                                </div>
                                            </td>
                                        </tr>
                                    </table>
                                    <!-- Distance badges -->
                                    <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="margin-top:8px;border-collapse:collapse;">
                                        <tr>
                                            <td style="width:50%;padding-right:4px;vertical-align:top;">
                                                <div style="padding:7px 10px;border-radius:8px;background:#f8fafc;border:1px solid #e2e8f0;">
                                                    <div style="font-size:10px;color:#64748b;font-weight:700;text-transform:uppercase;letter-spacing:0.03em;">vs 52W High</div>
                                                    <div style="margin-top:4px;font-size:13px;font-weight:800;color:{range_data['high_distance_color']};">{range_data['high_distance_text']}</div>
                                                </div>
                                            </td>
                                            <td style="width:50%;padding-left:4px;vertical-align:top;">
                                                <div style="padding:7px 10px;border-radius:8px;background:#f8fafc;border:1px solid #e2e8f0;">
                                                    <div style="font-size:10px;color:#64748b;font-weight:700;text-transform:uppercase;letter-spacing:0.03em;">vs 52W Low</div>
                                                    <div style="margin-top:4px;font-size:13px;font-weight:800;color:{range_data['low_distance_color']};">{range_data['low_distance_text']}</div>
                                                </div>
                                            </td>
                                        </tr>
                                    </table>
                                </td>
                            </tr>
    """


def _format_ratio(value, decimals=1):
    numeric_value = _safe_float(value)
    if numeric_value is None:
        return "n/a"
    return f"{numeric_value:.{decimals}f}"


def _format_percent(value, decimals=1):
    numeric_value = _safe_float(value)
    if numeric_value is None:
        return "n/a"
    if abs(numeric_value) <= 1:
        numeric_value *= 100
    return f"{numeric_value:.{decimals}f}%"


def _fund_color(label, raw_value):
    """Return (value_color, hint_text, hint_color) based on metric health."""
    v = _safe_float(raw_value)
    if v is None:
        return "#64748b", "No data", "#94a3b8"

    if label == "PE":
        if v < 15:
            return "#047857", "Undervalued", "#047857"
        if v < 25:
            return "#0f172a", "Fair", "#64748b"
        if v < 40:
            return "#d97706", "Stretched", "#d97706"
        return "#dc2626", "Expensive", "#dc2626"

    if label == "PB":
        if v < 1:
            return "#047857", "Below book", "#047857"
        if v < 3:
            return "#0f172a", "Reasonable", "#64748b"
        if v < 6:
            return "#d97706", "Premium", "#d97706"
        return "#dc2626", "Very high", "#dc2626"

    if label == "Dividend Yield":
        # raw_value comes in as a fraction (e.g. 0.023) or percent
        pct = v * 100 if v <= 1 else v
        if pct >= 3:
            return "#047857", "High yield", "#047857"
        if pct >= 1:
            return "#0f172a", "Moderate", "#64748b"
        if pct > 0:
            return "#d97706", "Low yield", "#d97706"
        return "#94a3b8", "No dividend", "#94a3b8"

    if label == "ROE":
        pct = v * 100 if v <= 1 else v
        if pct >= 20:
            return "#047857", "Strong", "#047857"
        if pct >= 12:
            return "#0f172a", "Decent", "#64748b"
        if pct >= 0:
            return "#d97706", "Weak", "#d97706"
        return "#dc2626", "Negative", "#dc2626"

    if label == "Debt / Equity":
        if v < 30:
            return "#047857", "Low debt", "#047857"
        if v < 100:
            return "#0f172a", "Manageable", "#64748b"
        if v < 200:
            return "#d97706", "High debt", "#d97706"
        return "#dc2626", "Very high", "#dc2626"

    return "#0f172a", "", "#64748b"


def build_fundamentals_html(fundamentals, fund_score):
    # Score pill color based on score value
    if fund_score >= 70:
        score_bg, score_border, score_color = "#f0fdf4", "#bbf7d0", "#15803d"
    elif fund_score >= 40:
        score_bg, score_border, score_color = "#fffbeb", "#fde68a", "#b45309"
    else:
        score_bg, score_border, score_color = "#fef2f2", "#fecaca", "#b91c1c"

    def _tile(label, raw_value, formatted_value, right=False):
        val_color, hint, hint_color = _fund_color(label, raw_value)
        pad = "padding:0 0 6px 6px;" if right else "padding:0 6px 6px 0;"
        hint_html = (
            f'<div style="margin-top:3px;font-size:10px;font-weight:700;color:{hint_color};">{hint}</div>'
            if hint else ""
        )
        return (
            f'<td style="{pad}width:50%;vertical-align:top;">'
            f'<div style="padding:9px 10px;border-radius:8px;background:#f8fafc;border:1px solid #e2e8f0;">'
            f'<div style="font-size:10px;color:#64748b;font-weight:700;text-transform:uppercase;letter-spacing:0.04em;">{label}</div>'
            f'<div style="margin-top:5px;font-size:15px;font-weight:800;color:{val_color};">{formatted_value}</div>'
            f'{hint_html}'
            f'</div></td>'
        )

    pe_raw   = fundamentals.get("pe")
    pb_raw   = fundamentals.get("pb")
    div_raw  = fundamentals.get("dividendYield")
    roe_raw  = fundamentals.get("roe")
    debt_raw = fundamentals.get("debtToEquity")

    pe_fmt   = _format_ratio(pe_raw, 1)
    pb_fmt   = _format_ratio(pb_raw, 1)
    div_fmt  = _format_percent(div_raw, 1)
    roe_fmt  = _format_percent(roe_raw, 1)
    debt_fmt = _format_ratio(debt_raw, 1)

    return f"""
                            <tr>
                                <td colspan="2" style="padding-top:10px;border-top:1px solid #eef2f7;">
                                    <!-- Header: title + score pill -->
                                    <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="border-collapse:collapse;">
                                        <tr>
                                            <td style="vertical-align:middle;">
                                                <div style="font-size:13px;color:#475569;font-weight:700;">Fundamentals</div>
                                            </td>
                                            <td style="text-align:right;vertical-align:middle;">
                                                <span style="display:inline-block;padding:4px 10px;border-radius:999px;background:{score_bg};border:1px solid {score_border};color:{score_color};font-size:12px;font-weight:800;white-space:nowrap;">Score {fund_score}</span>
                                            </td>
                                        </tr>
                                    </table>
                                    <!-- Metric tiles grid -->
                                    <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="margin-top:8px;border-collapse:collapse;">
                                        <tr>
                                            {_tile("PE",   pe_raw,   pe_fmt)}
                                            {_tile("PB",   pb_raw,   pb_fmt,  right=True)}
                                        </tr>
                                        <tr>
                                            {_tile("ROE",            roe_raw,  roe_fmt)}
                                            {_tile("Dividend Yield", div_raw,  div_fmt, right=True)}
                                        </tr>
                                        <tr>
                                            {_tile("Debt / Equity",  debt_raw, debt_fmt)}
                                            <td style="width:50%;padding:0 0 6px 6px;vertical-align:top;"></td>
                                        </tr>
                                    </table>
                                </td>
                            </tr>
    """


def calculate_combined_score(technical, fundamentals, sentiment, adv_fundamentals, market_context):
    # Use the existing final_score weighting and fold advanced fundamentals into total score.
    combined_fund = fundamentals + (adv_fundamentals * 0.4)
    total = final_score(technical, combined_fund, sentiment)

    trend_text = str(market_context.get("trend", "")).lower()
    if "bullish" in trend_text or "up" in trend_text or "positive" in trend_text:
        total += 2
    elif "bearish" in trend_text or "down" in trend_text or "negative" in trend_text:
        total -= 2

    return round(max(0, min(100, total)), 2)


def get_recommended_entry(signal, total_score, latest, market_context, entry_context):
    return choose_stock_entry(signal, total_score, latest, market_context, entry_context)


def truncate_text(text, limit=350):
    if not text:
        return text
    if len(text) <= limit:
        return text
    # Prefer to cut at sentence boundary
    cut = text[:limit]
    last_period = cut.rfind('. ')
    if last_period != -1 and last_period > limit - 100:
        return cut[:last_period + 1] + '...'
    return cut.rstrip() + '...'


def get_date_with_suffix(d):
    day = d.day
    if 4 <= day <= 20 or 24 <= day <= 30:
        suffix = "th"
    else:
        suffix = ["st", "nd", "rd"][day % 10 - 1]
    return d.strftime(f"%#d{suffix} %B %Y" if os.name == 'nt' else f"%-d{suffix} %B %Y")


# -----------------------------
# Email
# -----------------------------
def send_email(report_html, mode, pdf_attachment=None, pdf_filename="stock_report.pdf"):
    if not all([EMAIL_FROM, EMAIL_PASSWORD, EMAIL_TO]):
        print(
            "Email credentials not found. "
            "Please set EMAIL_FROM, EMAIL_PASSWORD, and EMAIL_TO environment variables."
        )
        return

    to_recipients = parse_email_list(EMAIL_TO)
    cc_recipients = parse_email_list(EMAIL_CC)

    if not to_recipients:
        print("No valid TO recipients found. Please set EMAIL_TO with a comma-separated list of emails.")
        return

    subject = "Equity & Commodity Research Briefing"
    if mode == "real_time":
        subject = f"Intraday Update — {subject}"
    elif mode == "eod":
        subject = f"Daily {subject}"

    # Append the formatted date and current time in IST
    now_utc = datetime.now(ZoneInfo("UTC"))
    now_ist = now_utc.astimezone(ZoneInfo("Asia/Kolkata"))
    formatted_date = get_date_with_suffix(now_ist)
    formatted_time = now_ist.strftime("%I:%M %p")
    subject += f" - {formatted_date} .{formatted_time} IST"

    if pdf_attachment:
        msg = MIMEMultipart("mixed")
        msg.attach(MIMEText(report_html, "html"))
        attachment_part = MIMEApplication(pdf_attachment, _subtype="pdf", Name=pdf_filename)
        attachment_part["Content-Disposition"] = f'attachment; filename="{pdf_filename}"'
        msg.attach(attachment_part)
    else:
        msg = MIMEText(report_html, "html")

    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = ", ".join(to_recipients)
    if cc_recipients:
        msg["Cc"] = ", ".join(cc_recipients)

    all_recipients = to_recipients + cc_recipients

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL_FROM, EMAIL_PASSWORD)
        server.sendmail(EMAIL_FROM, all_recipients, msg.as_string())
        server.quit()
    except smtplib.SMTPAuthenticationError:
        print(
            "SMTP Authentication Error: The username or password you entered is not correct. "
            "Please check your credentials and App Password if using Gmail."
        )
    except Exception as e:
        print(f"An error occurred while sending the email: {e}")
        traceback.print_exc()


def process_stock(stock_name, ticker, use_llm=True, detailed_llm=False, ai_stories=None):
    try:
        # fetch data and compute scores
        df = fetch_data(ticker)
        df = calculate_indicators(df)

        # consolidated technical score calculation
        tech_score, reasons = calculate_score(df)
        tech_signal = get_signal(tech_score)

        fund_raw = fetch_fundamentals(ticker)
        fund_score = score_fundamentals(fund_raw)
        upcoming_events = fund_raw.get("upcomingEvents", {})

        adv_raw = fetch_advanced_fundamentals(ticker)
        adv_fund_score = score_advanced_fundamentals(adv_raw)

        news_result = get_news(stock_name)
        headlines = news_result["headlines"]
        news_available = news_result["available"]
        if not news_available:
            print(f"News fetch failed for {stock_name}: sources_failed={news_result.get('sources_failed')}")
        sentiment = score_headlines(headlines, available=news_available)
        sentiment_score = sentiment.get("score", 50.0)
        sentiment_label = sentiment.get("label", "Neutral")
        # Grey out the sentiment display when we couldn't fetch news at all,
        # so a failed fetch doesn't read the same as a real neutral score.
        sentiment_color = "#8A8F9C" if sentiment_label == "Data Unavailable" else "#0f172a"

        try:
            market_context = build_market_context(ticker)
        except Exception as exc:
            print(f"Market context failed for {ticker}: {exc}")
            market_context = {
                "trend": "unknown",
                "return_20d": 0.0,
                "return_50d": 0.0,
                "sector": "unknown",
                "industry": "unknown",
            }
        total_score = calculate_combined_score(
            tech_score,
            fund_score,
            sentiment_score,
            adv_fund_score,
            market_context,
        )

        signal = decision(total_score)

        latest = df.iloc[-1]
        prev_close = df.iloc[-2]["close"] if len(df) >= 2 else None
        prev_close_change_pct = None
        if prev_close is not None and prev_close != 0 and pd.notna(prev_close) and pd.notna(latest["close"]):
            computed_pct = ((latest["close"] - prev_close) / prev_close) * 100
            if pd.notna(computed_pct):
                prev_close_change_pct = round(computed_pct, 2)
        prev_close_change_color = "#16a34a" if prev_close_change_pct is not None and prev_close_change_pct >= 0 else "#dc2626"
        # Blank (not "n/a"/"+nan%") when the value is missing or NaN -- a
        # blank cell reads as "no data" without the ugly literal "nan" text.
        prev_close_change_text = f"{prev_close_change_pct:+.2f}%" if prev_close_change_pct is not None else ""
        news_text = ", ".join(headlines[:3])
        
        # Get risk management data
        # Build the EMA/volume portion of entry_context first so it can be
        # passed into apply_risk_management -- previously this was built
        # afterward, so apply_risk_management always ran with entry_context
        # defaulted to 0/None, while choose_stock_entry (below) used the real
        # values. That meant the buy-level numbers and the chosen entry-style
        # label were computed from inconsistent inputs.
        entry_context = {
            "current_price": round(latest["close"], 2),
            "price_vs_ema20_pct": round(((latest["close"] - latest["ema20"]) / latest["ema20"]) * 100, 2) if pd.notna(latest["ema20"]) and latest["ema20"] else None,
            "price_vs_ema50_pct": round(((latest["close"] - latest["ema50"]) / latest["ema50"]) * 100, 2) if pd.notna(latest["ema50"]) and latest["ema50"] else None,
            "volume_vs_avg_pct": round(((latest["volume"] - latest["vol_avg"]) / latest["vol_avg"]) * 100, 2) if pd.notna(latest["vol_avg"]) and latest["vol_avg"] else None,
        }

        risk_data = apply_risk_management(signal, total_score, cash=100000, price=latest["close"], entry_context=entry_context)
        risk_meter = calculate_risk_meter(df, latest, fund_raw.get("beta"))
        range_52w = calculate_52_week_range(df, latest)
        pivot_levels = compute_pivot_levels(df)
        swing_zones = compute_swing_zones(df)

        entry_context["risk_reward_ratio"] = round((risk_data["target"] - latest["close"]) / max(latest["close"] - risk_data["stop_loss"], 1e-6), 2) if latest["close"] > risk_data["stop_loss"] else None

        recommended_entry = get_recommended_entry(signal, total_score, latest, market_context, entry_context)
        risk_data["recommended_entry"] = recommended_entry
        risk_data["recommended_entry_label"] = recommended_entry.replace("_", " ").title()
        risk_data["recommended_buy_level"] = risk_data["buy_levels"][recommended_entry]
        conviction_rating = get_conviction_rating(total_score)

        # "Swing setup" tag: near a 20-day breakout with a confirmed strong
        # trend (ADX >= 25) mirrors the recurring high-conviction setups
        # already used for stocks like BEL / ICICI Bank.
        swing_setup = ("Near 20-day breakout" in reasons) and ("ADX strong trend" in reasons)

        # AI Stock Story: looked up from the single combined call made once
        # for every stock in generate_ai_stocks_story() (see main()) rather
        # than making a fresh AI call per stock here. Falls back to a cheap,
        # non-AI 3-bullet summary (computed instantly from data already on
        # hand) if the batch call didn't produce an entry for this stock.
        story_bullets = (ai_stories or {}).get(stock_name) if use_llm else None
        if use_llm and not story_bullets:
            story_bullets = _fallback_stock_story(
                stock_name, signal, tech_score, fund_score, sentiment_score, sentiment_label, headlines
            )
        llm_display = build_ai_story_bullets_html(story_bullets) if story_bullets else ""

        if "sell" in signal.lower():
            priority = 3
        elif "hold" in signal.lower() or "buy / hold" in signal.lower():
            priority = 2
        else:
            priority = 1

        events = upcoming_events

        if events.get("has_event"):

            next_label = events.get("next_upcoming_event_label") or ""
            next_date = events.get("next_upcoming_event_date")

            # Only list an event here if it isn't the same one already
            # featured above as "Next Upcoming Event" -- previously both
            # were shown, duplicating the same date/label twice.
            details = []
            dividend_is_featured = "dividend" in next_label.lower() and events.get("dividend_record_date") == next_date
            if events["dividend_record_date"] != "NA" and not dividend_is_featured:
                details.append(
                    f"<div><strong>Dividend Record:</strong> {events['dividend_record_date']}</div>"
                )

            results_is_featured = "results" in next_label.lower() and events.get("results_announcement_date") == next_date
            if events["results_announcement_date"] != "NA" and not results_is_featured:
                details.append(
                    f"<div><strong>Results Announcement:</strong> {events['results_announcement_date']}</div>"
                )

            extra_details_html = ""
            if details:
                extra_details_html = (
                    '<hr style="margin:8px 0;border:none;border-top:1px solid #FCD34D;">'
                    + "".join(details)
                )

            events_html = f"""
            <div style="margin-top:6px;padding:10px 12px;
                        border-radius:8px;
                        background:#FEF3C7;
                        border-left:4px solid #F59E0B;">

                <div style="font-size:12px;
                            color:#92400E;
                            font-weight:bold;
                            text-transform:uppercase;">
                    Next Upcoming Event
                </div>

                <div style="font-size:16px;
                            color:#92400E;
                            font-weight:bold;
                            margin-top:3px;">
                    {events['next_upcoming_event_label']}
                </div>

                <div style="font-size:14px;
                            color:#B45309;
                            margin-top:3px;">
                    {events['next_upcoming_event_date']}
                </div>

                {extra_details_html}

            </div>
            """

        else:

            events_html = """
            <div style="margin-top:6px;
                        padding:10px;
                        border-radius:8px;
                        background:#F8FAFC;
                        border:1px solid #CBD5E1;
                        color:#475569;
                        font-size:13px;">
                No dividend or earnings announcements are scheduled in the next 60 days.
            </div>
            """
            # card-style HTML for each stock (email-safe)
        row_html = f"""
            <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="margin:14px 0;border-radius:6px;background:#ffffff;border:1px solid #E7E4DC;">
                <tr>
                    <td style="height:3px;line-height:3px;font-size:0;background:{'#8B2E2E' if 'sell' in signal.lower() else '#A6812F' if 'hold' in signal.lower() else '#2F5233'};border-radius:6px 6px 0 0;">&nbsp;</td>
                </tr>
                <tr>
                    <td style="padding:16px 16px 0;">
                        <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="border-collapse:collapse;">
                            <tr>
                                <td style="vertical-align:top;">
                                    <h3 style="margin:0;font-family:Georgia,'Times New Roman',serif;font-weight:400;font-size:17px;color:#14213D;line-height:1.25;">{stock_name} <span style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;font-size:12px;color:#8A8F9C;">{ticker}</span></h3>
                                    <div class="llm-thesis" style="margin:8px 0 0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;font-size:13px;color:#3C4256;line-height:1.55;max-height:140px;overflow:hidden;">{llm_display}</div>
                                </td>
                                <td style="width:150px;text-align:right;vertical-align:top;">
                                    <div style="display:inline-block;padding:5px 11px;border-radius:3px;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;font-size:11px;font-weight:700;letter-spacing:0.04em;text-transform:uppercase;color:#fff;background:{'#8B2E2E' if 'sell' in signal.lower() else '#A6812F' if 'hold' in signal.lower() else '#2F5233'};">{signal}</div>
                                    <div style="margin-top:12px;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;font-size:10px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#8A8F9C;">Conviction</div>
                                    <div style="margin-top:6px;white-space:nowrap;">{conviction_rating['icons_html']}</div>
                                    <div style="margin-top:6px;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;font-size:12px;font-weight:700;color:{conviction_rating['fill_color']};">{conviction_rating['label']}</div>
                                </td>
                            </tr>
                        </table>
                    </td>
                </tr>
                <tr>
                    <td style="padding:0 16px;">
                        <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="border-collapse:collapse;margin-top:10px;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;font-size:13px;color:#4A5063;">
                            <tr>
                                <td style="padding:6px 0;width:50%;"><strong>Current Price</strong><div style="color:#0f172a;margin-top:4px;">{round(latest['close'],2)}</div></td>
                                <td style="padding:6px 0;width:50%;"><strong>EMA20 / EMA50</strong><div style="color:#0f172a;margin-top:4px;">{round(latest['ema20'],2)} / {round(latest['ema50'],2)}</div></td>
                            </tr>
                            <tr>
                                <td style="padding:6px 0;"><strong>EMA100 / EMA200</strong><div style="color:#0f172a;margin-top:4px;">{round(latest['ema100'],2)} / {round(latest['ema200'],2)}</div></td>
                                <td style="padding:6px 0;"><strong>RSI / ADX</strong><div style="color:#0f172a;margin-top:4px;">{round(latest['rsi'],2)} / {round(latest['adx'],2)}</div></td>
                            </tr>
                            <tr>
                                <td style="padding:6px 0;"><strong>Prev Close</strong><div style="color:#0f172a;margin-top:4px;">{prev_close if prev_close is not None else 'n/a'}</div></td>
                                <td style="padding:6px 0;"><strong>Change vs Prev</strong><div style="color:{prev_close_change_color};margin-top:4px;">{prev_close_change_text}</div></td>
                            </tr>
                            <tr>
                                <td style="padding:6px 0;"><strong>Entry Edge</strong><div style="color:#0f172a;margin-top:4px;">{entry_context['price_vs_ema20_pct']:+.1f}% vs EMA20 / {entry_context['price_vs_ema50_pct']:+.1f}% vs EMA50</div></td>
                                <td style="padding:6px 0;"><strong>Volume / RR</strong><div style="color:#0f172a;margin-top:4px;">{entry_context['volume_vs_avg_pct']:+.1f}% vol / {entry_context['risk_reward_ratio']}:1 RR</div></td>
                            </tr>
                            <tr>
                                <td style="padding:6px 0;"><strong>Technical Score</strong><div style="color:#0f172a;margin-top:4px;">{tech_score}</div></td>
                                <td style="padding:6px 0;"><strong>Sentiment</strong><div style="color:{sentiment_color};margin-top:4px;">{"Data Unavailable" if sentiment_label == "Data Unavailable" else f"{sentiment_score} ({sentiment_label})"}</div></td>
                            </tr>
                            <tr>
                                <td style="padding:6px 0;"><strong>Target / Stop</strong><div style="color:#0f172a;margin-top:4px;">{risk_data['target']} / {risk_data['stop_loss']}</div></td>
                                <td style="padding:6px 0;"><strong>Trend</strong><div style="color:#0f172a;margin-top:4px;">{market_context['trend']}</div></td>
                            </tr>
                            {build_fundamentals_html(fund_raw, fund_score)}
                            {build_support_resistance_html(pivot_levels, swing_zones, round(latest['close'], 2))}
                            {build_52_week_range_html(range_52w)}
                            {build_risk_meter_html(risk_meter)}
                        </table>
                    </td>
                </tr>
                <tr>
                    <td style="padding:0 16px;">
                        <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="border-collapse:collapse;">
                            <tr>
                                <td style="padding-top:10px;border-top:1px solid #eef2f7;">
                                    <div style="font-size:13px;color:#475569;"><strong>Buy Levels:</strong></div>
                                    <div style="font-size:12px;color:#0f172a;margin-top:4px;">
                                        <span style="color:#047857;font-weight:700;">Recommended {risk_data['recommended_entry_label']}: <strong>{risk_data['recommended_buy_level']}</strong></span>
                                        <div style="margin-top:4px;color:#64748b;">
                                            Patient: <strong>{risk_data['buy_levels']['patient_entry']}</strong> &bull;
                                            Optimal: <strong>{risk_data['buy_levels']['optimal_entry']}</strong> &bull;
                                            Aggressive: <strong>{risk_data['buy_levels']['aggressive_entry']}</strong>
                                        </div>
                                    </div>
                                </td>
                            </tr>
                        </table>
                    </td>
                </tr>
                <tr>
                    <td style="padding:0 16px;">
                        <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="border-collapse:collapse;">
                            <tr>
                                <td style="padding-top:10px;border-top:1px solid #eef2f7;">
                            <div style="font-size:13px;color:#475569;">
                            <strong>Upcoming Events</strong>
                        </div>

                        {events_html}
                                   
                                </td>
                            </tr>
                        </table>
                    </td>
                </tr>
                <tr>
                    <td style="padding:0 16px 16px;">
                        <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="border-collapse:collapse;">
                            <tr>
                                <td style="padding-top:10px;border-top:1px solid #eef2f7;font-size:13px;color:#475569;"><strong>News:</strong> {news_text or 'No recent headlines.'}</td>
                            </tr>
                        </table>
                    </td>
                </tr>
            </table>
            """
        summary_entry = {
            "stock_name": stock_name,
            "ticker": ticker,
            "signal": signal,
            "current_price": round(latest["close"], 2),
            "recommended_buy_level": risk_data.get("recommended_buy_level"),
            "ema20": latest.get("ema20"),
            "upcoming_events": events,
            "target": risk_data.get("target"),
            "stop_loss": risk_data.get("stop_loss"),
            "sector": market_context.get("sector"),
            "priority": priority,
            "total_score": total_score,
            "swing_setup": swing_setup,
            "day_change_pct": prev_close_change_pct,
            "trend": market_context.get("trend"),
        }

        print(f"{stock_name} ({ticker}) -> {signal} | Conviction: {conviction_rating['label']} {conviction_rating['icons_text']} | Risk: {risk_meter['label']}")
        return (priority, total_score, stock_name, row_html, summary_entry) 
    
    except Exception as e:
        error_text = f"Error processing {ticker}: {str(e)}"
        print(error_text)
        traceback.print_exc()
        
        err_html = f"""
            <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="margin:12px 20px;border-radius:8px;background:#fff7f7;border:1px solid #f5c2c7;">
                <tr>
                    <td style="padding:12px;color:#721c24;font-size:13px;"><strong>Error:</strong> {error_text}</td>
                </tr>
            </table>
            """
    return (4, 0, ticker, err_html, {
        "stock_name": ticker,
        "ticker": ticker,
        "signal": "ERROR",
        "current_price": None,
        "recommended_buy_level": None,
        "ema20": None,
        "upcoming_events": {},
        "target": None,
        "stop_loss": None,
        "sector": None,
        "priority": 4,
        "total_score": 0,
        "swing_setup": False,
        "day_change_pct": None,
        "trend": None,
    })


def _format_short_date(value):
    if not value or str(value).upper() == "NA":
        return None

    text = str(value).strip()
    for fmt in ("%d %b %Y", "%Y-%m-%d", "%d-%b-%Y"):
        try:
            return datetime.strptime(text, fmt).strftime("%d %b")
        except ValueError:
            continue
    return text


def _relative_day_label(value):
    if not value or str(value).upper() == "NA":
        return None

    text = str(value).strip()
    for fmt in ("%d %b %Y", "%Y-%m-%d", "%d-%b-%Y"):
        try:
            event_date = datetime.strptime(text, fmt).date()
            break
        except ValueError:
            event_date = None

    if event_date is None:
        return None

    today = datetime.now(ZoneInfo("Asia/Kolkata")).date()
    delta_days = (event_date - today).days
    if delta_days == 0:
        return "today"
    if delta_days == 1:
        return "in 1 day"
    if delta_days > 1:
        return f"in {delta_days} days"
    if delta_days == -1:
        return "1 day ago"
    return f"{abs(delta_days)} days ago"


def build_quick_summary(rows):
    """
    Builds compact, human-readable summary bullets for the top of the
    report, grouped first by market (US / India) and then by recommendation
    (Buy / Hold / Sell) within each market -- mirroring the Portfolio
    Heatmap's US / India / Commodities grouping so the Executive Summary
    reads the same way at a glance.
    Returns a dict: {"US": {"Buy": [...], "Hold": [...], "Sell": [...]},
                     "India": {"Buy": [...], "Hold": [...], "Sell": [...]}}.
    """
    groups = {
        "US": {"Buy": [], "Hold": [], "Sell": []},
        "India": {"Buy": [], "Hold": [], "Sell": []},
    }
    if not rows:
        return groups

    priority_to_group = {1: "Buy", 2: "Hold", 3: "Sell"}

    for row in rows:
        group_key = priority_to_group.get(row.get("priority"))
        if group_key is None:
            continue  # errors aren't actionable in the quick summary

        market_key = classify_market(row.get("ticker"))
        if market_key not in groups:
            market_key = "US"
        market_bucket = groups[market_key]

        stock_name = row.get("stock_name") or ""
        signal = str(row.get("signal") or "").upper()
        current_price = row.get("current_price")
        recommended_buy_level = row.get("recommended_buy_level")
        ema20 = row.get("ema20")
        upcoming_events = row.get("upcoming_events") or {}

        if "BUY" in signal and current_price is not None and recommended_buy_level is not None and current_price < recommended_buy_level:
            market_bucket[group_key].append(f"✅ {stock_name} below ₹{recommended_buy_level}")
            continue

        results_date = upcoming_events.get("results_announcement_date")
        if results_date and str(results_date).upper() != "NA":
            short_date = _format_short_date(results_date)
            relative_label = _relative_day_label(results_date)
            if relative_label == "today":
                market_bucket[group_key].append(f"📊 {stock_name} results today")
            else:
                market_bucket[group_key].append(f"📊 {stock_name} results on {short_date or results_date}")
            continue

        dividend_date = upcoming_events.get("dividend_record_date")
        if dividend_date and str(dividend_date).upper() != "NA":
            relative_label = _relative_day_label(dividend_date)
            if relative_label == "today":
                market_bucket[group_key].append(f"📅 {stock_name} dividend record today")
            else:
                market_bucket[group_key].append(f"📅 {stock_name} dividend record {relative_label or 'soon'}")
            continue

        if current_price is not None and ema20 is not None and current_price < ema20:
            market_bucket[group_key].append(f"⚠ {stock_name} below EMA20—avoid adding")

    return groups


# Thresholds for the Action Plan table's buy-zone classification. Tunable
# via env vars without touching code. ADD_ALLOCATION_PCT is the generic
# allocation size suggested when a stock is in its buy zone -- main.py
# doesn't currently size positions per-stock (see position_sizing.py for
# the risk-based stop/target sizing, which is separate from allocation %).
ADD_ALLOCATION_PCT = int(os.getenv("ADD_ALLOCATION_PCT", "25"))
SLIGHTLY_ABOVE_BUY_ZONE_PCT = float(os.getenv("SLIGHTLY_ABOVE_BUY_ZONE_PCT", "6"))
IN_BUY_ZONE_TOLERANCE_PCT = float(os.getenv("IN_BUY_ZONE_TOLERANCE_PCT", "1.5"))

_ACTION_PLAN_STATUS_DISPLAY = {
    "in_zone":        ("In buy zone",            "#2F5233", "#E7EEE4"),
    "slightly_above": ("Slightly above buy zone", "#A6812F", "#FDF3D9"),
    "overextended":   ("Overextended",            "#8B2E2E", "#FBEAEA"),
    "sell_signal":    ("Sell signal active",      "#8B2E2E", "#FBEAEA"),
    "unknown":        ("—",                       "#8A8F9C", "#F4F2ED"),
}

_ACTION_PLAN_NEXT_ACTION = {
    "in_zone":        f"Add {ADD_ALLOCATION_PCT}% allocation",
    "slightly_above": "Wait for dip",
    "overextended":   "Don't add",
    "sell_signal":    "Exit / Reduce",
    "unknown":        "—",
}


def _classify_buy_zone(current_price, buy_level, signal=""):
    """
    Classifies where the current price sits relative to the recommended
    buy level, for the Action Plan table. Returns one of:
    "in_zone" / "slightly_above" / "overextended" / "sell_signal" / "unknown".

    - A SELL signal always takes priority over the price/buy-level math.
    - "in_zone": at or within IN_BUY_ZONE_TOLERANCE_PCT of the buy level
      (covers prices at or slightly below it too).
    - "slightly_above": up to SLIGHTLY_ABOVE_BUY_ZONE_PCT above the buy
      level -- still a fair entry, but no longer the ideal one.
    - "overextended": further above than that -- chasing the price here
      has a worse risk/reward than waiting for a pullback.
    """
    if "sell" in str(signal or "").lower():
        return "sell_signal"
    if current_price is None or buy_level is None or buy_level == 0:
        return "unknown"
    pct_above = (current_price - buy_level) / buy_level * 100
    if pct_above <= IN_BUY_ZONE_TOLERANCE_PCT:
        return "in_zone"
    elif pct_above <= SLIGHTLY_ABOVE_BUY_ZONE_PCT:
        return "slightly_above"
    return "overextended"


def _action_plan_profit_booking(status_key, buy_level, target):
    """
    "Partial at +X%" when the stock is in the 'slightly_above' zone and a
    target is known (X = the planned gain from buy level to target) --
    this is the state where someone who already holds a position from
    around the buy level is sitting on a gain worth considering booking,
    even though it's not a great level to add fresh money at. Every other
    state (still accumulating, overextended, or unknown) just says "Hold":
    either it's too early to think about booking, or there's nothing new
    to act on here.
    """
    if status_key == "sell_signal":
        return "Book profit"
    if status_key == "slightly_above" and buy_level and target and target > buy_level:
        gain_pct = round((target - buy_level) / buy_level * 100)
        return f"Partial at +{gain_pct}%"
    return "Hold"


def _action_plan_row_html(name, currency_symbol, buy_level, target, status_key, ticker_label=None, sans=None):
    sans = sans or "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
    label, color, bg = _ACTION_PLAN_STATUS_DISPLAY[status_key]
    next_action = _ACTION_PLAN_NEXT_ACTION[status_key]
    profit_booking = _action_plan_profit_booking(status_key, buy_level, target)
    add_below = f"{currency_symbol}{buy_level:,.2f}" if buy_level is not None else "—"
    name_display = f"{name} <span style=\"color:#8A8F9C;font-size:11px;\">{ticker_label}</span>" if ticker_label else name
    return f"""
        <tr>
            <td style="padding:7px 10px;font-size:12px;font-weight:700;font-family:{sans};color:#14213D;border-top:1px solid #EDEAE2;">{name_display}</td>
            <td style="padding:7px 10px;font-size:12px;font-family:{sans};color:#14213D;border-top:1px solid #EDEAE2;">{add_below}</td>
            <td style="padding:7px 10px;font-size:12px;font-family:{sans};border-top:1px solid #EDEAE2;">
                <span style="display:inline-block;padding:2px 8px;border-radius:999px;background:{bg};color:{color};font-size:11px;font-weight:700;">{label}</span>
            </td>
            <td style="padding:7px 10px;font-size:12px;font-family:{sans};color:#14213D;border-top:1px solid #EDEAE2;">{next_action}</td>
            <td style="padding:7px 10px;font-size:12px;font-family:{sans};color:#14213D;border-top:1px solid #EDEAE2;">{profit_booking}</td>
        </tr>
    """


def build_action_plan_table_html(summary_rows, commodity_data=None, gold_levels=None, silver_levels=None, gold_plan=None, silver_plan=None):
    """
    Builds the "Action Plan" table shown right after the Executive Summary:
    one row per stock/commodity with Add Below / Current Status / Next
    Action / Profit Booking, grouped into US Stocks, India Stocks, Gold,
    and Silver -- the same four groups used elsewhere in the report.

    Only covers stocks with a usable current price and recommended buy
    level (skips Errors, where neither is available). Gold/Silver rows
    are best-effort: this reads a few plausible key names off whatever
    CommodityTracker.derive_buy_levels()/build_trade_plan() return, and
    falls back to "—" for any field it can't find rather than crashing --
    that module wasn't available when this was written, so if the metals
    rows come back mostly blank, the key names below need adjusting to
    match your actual commodity_tracker.py.
    """
    sans = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"

    def group_header_row(label):
        return (
            f'<tr><td colspan="5" style="padding:10px 10px 6px;font-family:{sans};'
            f'font-size:11px;font-weight:700;color:#14213D;text-transform:uppercase;'
            f'letter-spacing:0.05em;background:#F4F2ED;">{label}</td></tr>'
        )

    stock_rows_by_market = {"US": [], "India": []}
    for entry in (summary_rows or []):
        signal = entry.get("signal") or ""
        if str(signal).upper() == "ERROR":
            continue
        current_price = entry.get("current_price")
        buy_level = entry.get("recommended_buy_level")
        target = entry.get("target")
        status_key = _classify_buy_zone(current_price, buy_level, signal)
        if status_key == "unknown":
            continue
        currency_symbol = "₹" if classify_market(entry.get("ticker")) == "India" else "$"
        row_html_str = _action_plan_row_html(
            entry.get("stock_name") or entry.get("ticker") or "—",
            currency_symbol, buy_level, target, status_key,
            ticker_label=entry.get("ticker"), sans=sans,
        )
        market_key = classify_market(entry.get("ticker"))
        stock_rows_by_market.setdefault(market_key, []).append((entry.get("total_score") or 0, row_html_str))

    def sorted_rows(market_key):
        return [r for _, r in sorted(stock_rows_by_market.get(market_key, []), key=lambda x: x[0], reverse=True)]

    us_rows = sorted_rows("US")
    india_rows = sorted_rows("India")

    def commodity_row(name, data, levels, plan):
        if not data:
            return ""
        current_price = data.get("current")
        # Best-effort lookup across a few plausible key names -- see the
        # docstring above about commodity_tracker.py not being available.
        buy_level = None
        for src in (levels or {}, plan or {}):
            for key in ("recommended_buy_level", "optimal_entry", "buy_level", "entry"):
                if isinstance(src, dict) and src.get(key) is not None:
                    buy_level = src.get(key)
                    break
            if buy_level is not None:
                break
        target = None
        for src in (plan or {}, levels or {}):
            for key in ("target", "target1", "recommended_target"):
                if isinstance(src, dict) and src.get(key) is not None:
                    target = src.get(key)
                    break
            if target is not None:
                break
        status_key = _classify_buy_zone(current_price, buy_level)
        if status_key == "unknown":
            return ""
        return _action_plan_row_html(name, "₹", buy_level, target, status_key, sans=sans)

    gold_row = commodity_row("Gold (22K)", (commodity_data or {}).get("gold"), gold_levels, gold_plan)
    silver_row = commodity_row("Silver", (commodity_data or {}).get("silver"), silver_levels, silver_plan)

    body = ""
    if us_rows:
        body += group_header_row("🇺🇸 US Stocks") + "".join(us_rows)
    if india_rows:
        body += group_header_row("🇮🇳 India Stocks") + "".join(india_rows)
    if gold_row:
        body += group_header_row("🥇 Gold") + gold_row
    if silver_row:
        body += group_header_row("🥈 Silver") + silver_row

    if not body:
        return ""

    return f"""
        <tr>
          <td style="padding:0 28px 18px;" class="email-padding">
            <h2 style="margin:0 0 8px;font-family:Georgia,'Times New Roman',serif;font-weight:400;font-size:16px;color:#14213D;">Action Plan</h2>
            <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="border:1px solid #E7E4DC;border-radius:4px;overflow:hidden;border-collapse:collapse;">
              <tr style="background:#14213D;">
                <td style="padding:8px 10px;font-family:{sans};font-size:11px;font-weight:700;color:#B08D57;text-transform:uppercase;letter-spacing:0.05em;">Stock</td>
                <td style="padding:8px 10px;font-family:{sans};font-size:11px;font-weight:700;color:#ffffff;text-transform:uppercase;letter-spacing:0.05em;">Add Below</td>
                <td style="padding:8px 10px;font-family:{sans};font-size:11px;font-weight:700;color:#ffffff;text-transform:uppercase;letter-spacing:0.05em;">Current Status</td>
                <td style="padding:8px 10px;font-family:{sans};font-size:11px;font-weight:700;color:#ffffff;text-transform:uppercase;letter-spacing:0.05em;">Next Action</td>
                <td style="padding:8px 10px;font-family:{sans};font-size:11px;font-weight:700;color:#ffffff;text-transform:uppercase;letter-spacing:0.05em;">Profit Booking</td>
              </tr>
              {body}
            </table>
          </td>
        </tr>
    """


def classify_market(ticker):
    """
    Classifies a ticker as India or US based on its exchange suffix.
    Indian tickers pulled via yfinance carry a '.NS' (NSE) or '.BO' (BSE) suffix;
    US tickers (e.g. AAPL, GOOG, AMZN, QQQ) have no suffix.
    """
    if not ticker:
        return "US"
    upper_ticker = ticker.upper().strip()
    if upper_ticker.endswith(".NS") or upper_ticker.endswith(".BO"):
        return "India"
    return "US"


def get_section_html(title, count, items):
    """
    Generates the HTML for a section of the report.
    """
    if not items:
        return ""

    header = (
        f'<tr><td style="padding:14px 28px 0;" class="email-padding">'
        f'<h2 style="margin:0;font-family:Georgia,&quot;Times New Roman&quot;,serif;font-weight:400;font-size:16px;color:#14213D;">'
        f'{title} ({count})</h2></td></tr>'
    )
    rows_html = "".join([f'<tr><td style="padding:0 28px;page-break-inside:avoid;break-inside:avoid;" class="email-padding">{html}</td></tr>' for _, _, html in items])
    return header + rows_html


def get_market_section_html(market_label, market_groups):
    """
    Generates the HTML for an entire market block (e.g. US Stocks / India Stocks),
    with its own Buy / Hold / Sell / Errors sub-sections nested inside.
    """
    total = sum(len(items) for items in market_groups.values())
    if total == 0:
        return ""

    header = f"""
        <tr>
          <td style="padding:20px 28px 6px;" class="email-padding">
            <h2 style="margin:0;font-family:Georgia,'Times New Roman',serif;font-weight:400;font-size:18px;color:#14213D;border-bottom:1px solid #B08D57;padding-bottom:8px;letter-spacing:0.01em;">{market_label} <span style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;font-size:12px;color:#8A8F9C;font-weight:600;">({total})</span></h2>
          </td>
        </tr>
    """
    body = ""
    for title in ["Buy", "Hold", "Sell", "Errors"]:
        body += get_section_html(title, len(market_groups[title]), market_groups[title])

    return header + body


# -----------------------------
# Report enhancements: change tracking, breach alerts, swing tags,
# quick-jump table, concentration alerts, error banner, CSV export
# -----------------------------
def build_stock_enrichment_html(summary_entry, prior_entry):
    """
    Builds a small badge row shown above a stock's card:
    - signal-change badge (vs the previous run)
    - stop-loss / target breach badge (current price vs the previous run's levels)
    - swing-setup tag
    """
    badges = []

    signal = summary_entry.get("signal") or ""
    if prior_entry:
        prior_signal = prior_entry.get("signal")
        if prior_signal and prior_signal != signal and prior_signal != "ERROR":
            prior_priority = prior_entry.get("priority")
            current_priority = summary_entry.get("priority")
            if current_priority is not None and prior_priority is not None and current_priority < prior_priority:
                arrow, color, bg = "⬆", "#047857", "#dcfce7"
            elif current_priority is not None and prior_priority is not None and current_priority > prior_priority:
                arrow, color, bg = "⬇", "#dc2626", "#fee2e2"
            else:
                arrow, color, bg = "↔", "#a16207", "#fef3c7"
            badges.append(
                f'<span style="display:inline-block;margin:0 6px 6px 0;padding:4px 10px;border-radius:999px;'
                f'font-size:11px;font-weight:700;color:{color};background:{bg};border:1px solid {color}33;">'
                f'{arrow} {prior_signal} &rarr; {signal}</span>'
            )

        current_price = summary_entry.get("current_price")
        prior_stop = prior_entry.get("stop_loss")
        prior_target = prior_entry.get("target")
        if current_price is not None and prior_stop is not None and current_price <= prior_stop:
            badges.append(
                '<span style="display:inline-block;margin:0 6px 6px 0;padding:4px 10px;border-radius:999px;'
                'font-size:11px;font-weight:700;color:#dc2626;background:#fee2e2;border:1px solid #dc262633;">'
                f'🛑 Below prior stop-loss ({prior_stop})</span>'
            )
        elif current_price is not None and prior_target is not None and current_price >= prior_target:
            badges.append(
                '<span style="display:inline-block;margin:0 6px 6px 0;padding:4px 10px;border-radius:999px;'
                'font-size:11px;font-weight:700;color:#047857;background:#dcfce7;border:1px solid #04785733;">'
                f'🎯 Target reached ({prior_target})</span>'
            )

    if summary_entry.get("swing_setup"):
        badges.append(
            '<span style="display:inline-block;margin:0 6px 6px 0;padding:4px 10px;border-radius:999px;'
            'font-size:11px;font-weight:700;color:#14213D;background:#EFEAE0;border:1px solid #B08D5766;">'
            '⚡ Swing Setup</span>'
        )

    if not badges:
        return ""

    return f'<div style="margin:12px 0 -6px;">{"".join(badges)}</div>'


def _heatmap_signal_style(pr):
    """Maps a priority tier to (dot emoji, text color, action label)."""
    if pr == 1:
        return "🟢", "#2F5233", "Add"
    if pr == 2:
        return "🟡", "#A6812F", "Wait"
    if pr == 3:
        return "🔴", "#8B2E2E", "Exit"
    return "⚪", "#8A8F9C", "—"


def _heatmap_trend_label(raw_trend):
    """Normalizes whatever string market_context returns into a short,
    consistent label + icon (source strings aren't guaranteed to be
    exactly 'Bullish'/'Bearish', so this matches loosely on keywords)."""
    text = str(raw_trend or "").strip().lower()
    if not text or text == "unknown":
        return "➖ Unknown"
    if "up" in text or "bull" in text or "positive" in text:
        return "📈 Bullish"
    if "down" in text or "bear" in text or "negative" in text:
        return "📉 Bearish"
    return "➖ Weak"


def _heatmap_move_html(day_change_pct):
    """Formats the day's % move with color; blank (not 'n/a') when missing."""
    if day_change_pct is None:
        return ""
    color = "#2F5233" if day_change_pct >= 0 else "#8B2E2E"
    return f'<span style="color:{color};font-weight:700;">{day_change_pct:+.2f}%</span>'


def build_quick_jump_table_html(rows, commodity_data=None, commodity_buy_signals=None):
    """
    Portfolio Heatmap: a single scannable table (Symbol / Signal / Today's
    Move / Trend / Action) covering every ticker across both markets,
    placed above the full stock cards so the report can be read at a
    glance before scrolling through the detailed sections. Grouped into
    explicit US / India sub-sections (rather than just sorted together)
    so each market's tickers are visually separated. When commodity_data
    is supplied, a final "Commodities" group (Gold & Silver) is appended
    after US/India, mirroring those two groups, so the metals are
    scannable in the same at-a-glance table -- not just in the detailed
    cards further down (or, for the email, only in the attached PDF).
    """
    if not rows and not commodity_data:
        return ""

    by_market = {"US": [], "India": []}
    for pr, score, name, _html, summary_entry, market in rows:
        by_market.setdefault(market, []).append((pr, score, name, summary_entry))

    market_sections = [("US", "🇺🇸 US Stocks"), ("India", "🇮🇳 India Stocks")]

    sans = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
    col_header = (
        '<tr>'
        f'<td style="padding:6px 8px 6px 14px;font-family:{sans};font-size:10px;font-weight:700;color:#8A8F9C;text-transform:uppercase;letter-spacing:0.06em;">Symbol</td>'
        f'<td style="padding:6px 8px;font-family:{sans};font-size:10px;font-weight:700;color:#8A8F9C;text-transform:uppercase;letter-spacing:0.06em;">Signal</td>'
        f'<td style="padding:6px 8px;font-family:{sans};font-size:10px;font-weight:700;color:#8A8F9C;text-transform:uppercase;letter-spacing:0.06em;text-align:right;">Today\'s Move</td>'
        f'<td style="padding:6px 8px;font-family:{sans};font-size:10px;font-weight:700;color:#8A8F9C;text-transform:uppercase;letter-spacing:0.06em;">Trend</td>'
        f'<td style="padding:6px 8px 6px 8px;font-family:{sans};font-size:10px;font-weight:700;color:#8A8F9C;text-transform:uppercase;letter-spacing:0.06em;text-align:right;">Action</td>'
        '</tr>'
    )

    body = ""
    for market_key, market_label in market_sections:
        entries = by_market.get(market_key) or []
        if not entries:
            continue
        entries.sort(key=lambda e: (e[0], -e[1]))  # priority, score desc
        body += (
            f'<tr><td colspan="5" style="padding:7px 10px;background:#F4F2ED;'
            f'font-family:{sans};font-size:11px;font-weight:700;color:#14213D;text-transform:uppercase;'
            f'letter-spacing:0.05em;">{market_label} ({len(entries)})</td></tr>'
        )
        body += col_header
        for pr, score, name, summary_entry in entries:
            signal = summary_entry.get("signal", "n/a")
            dot, color, action = _heatmap_signal_style(pr)
            move_html = _heatmap_move_html(summary_entry.get("day_change_pct"))
            trend_html = _heatmap_trend_label(summary_entry.get("trend"))
            body += (
                f'<tr>'
                f'<td style="padding:7px 8px 7px 14px;font-family:{sans};font-size:12px;font-weight:600;color:#14213D;border-bottom:1px solid #EFEDE7;">{name}</td>'
                f'<td style="padding:7px 8px;font-family:{sans};font-size:12px;font-weight:700;color:{color};border-bottom:1px solid #EFEDE7;white-space:nowrap;">{dot} {signal}</td>'
                f'<td style="padding:7px 8px;font-family:{sans};font-size:12px;border-bottom:1px solid #EFEDE7;text-align:right;">{move_html}</td>'
                f'<td style="padding:7px 8px;font-family:{sans};font-size:12px;color:#4A5063;border-bottom:1px solid #EFEDE7;white-space:nowrap;">{trend_html}</td>'
                f'<td style="padding:7px 8px;font-family:{sans};font-size:12px;font-weight:700;color:{color};border-bottom:1px solid #EFEDE7;text-align:right;">{action}</td>'
                f'</tr>'
            )

    # Commodities group (Gold & Silver) -- appended last, mirroring the
    # US/India market groups above. Reuses the same buy-trigger signal
    # already computed for the Quick Summary bullets (commodity_buy_signals)
    # so this doesn't re-derive or drift from that logic.
    if commodity_data:
        commodity_buy_signals = commodity_buy_signals or {}
        commodity_entries = []
        for label, key in [("🥇 Gold (22K)", "gold"), ("🥈 Silver", "silver")]:
            metal_data = commodity_data.get(key) or {}
            if not metal_data:
                continue
            change = metal_data.get("change", 0) or 0
            buy_triggered = commodity_buy_signals.get(key, {}).get("last_buy_triggered", False)
            if buy_triggered:
                pr, signal = 1, "Buy"
            elif change <= -1.5:
                pr, signal = 2, "Watch"
            elif change >= 1.5:
                pr, signal = 2, "Momentum"
            else:
                pr, signal = 2, "Stable"
            trend_raw = "up" if change > 0 else ("down" if change < 0 else "")
            commodity_entries.append((label, pr, signal, change, trend_raw))

        if commodity_entries:
            body += (
                f'<tr><td colspan="5" style="padding:7px 10px;background:#F4F2ED;'
                f'font-family:{sans};font-size:11px;font-weight:700;color:#14213D;text-transform:uppercase;'
                f'letter-spacing:0.05em;">🪙 Commodities ({len(commodity_entries)})</td></tr>'
            )
            body += col_header
            for name, pr, signal, change, trend_raw in commodity_entries:
                dot, color, action = _heatmap_signal_style(pr)
                move_html = _heatmap_move_html(change)
                trend_html = _heatmap_trend_label(trend_raw)
                body += (
                    f'<tr>'
                    f'<td style="padding:7px 8px 7px 14px;font-family:{sans};font-size:12px;font-weight:600;color:#14213D;border-bottom:1px solid #EFEDE7;">{name}</td>'
                    f'<td style="padding:7px 8px;font-family:{sans};font-size:12px;font-weight:700;color:{color};border-bottom:1px solid #EFEDE7;white-space:nowrap;">{dot} {signal}</td>'
                    f'<td style="padding:7px 8px;font-family:{sans};font-size:12px;border-bottom:1px solid #EFEDE7;text-align:right;">{move_html}</td>'
                    f'<td style="padding:7px 8px;font-family:{sans};font-size:12px;color:#4A5063;border-bottom:1px solid #EFEDE7;white-space:nowrap;">{trend_html}</td>'
                    f'<td style="padding:7px 8px;font-family:{sans};font-size:12px;font-weight:700;color:{color};border-bottom:1px solid #EFEDE7;text-align:right;">{action}</td>'
                    f'</tr>'
                )

    if not body:
        return ""

    return f"""
        <tr>
          <td style="padding:0 28px 14px;" class="email-padding">
            <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="border:1px solid #E7E4DC;border-radius:4px;overflow:hidden;">
              <tr><td colspan="5" style="padding:9px 10px;background:#14213D;font-family:{sans};font-size:11px;font-weight:700;color:#B08D57;text-transform:uppercase;letter-spacing:0.08em;">Portfolio Heatmap</td></tr>
              {body}
            </table>
          </td>
        </tr>
    """


def build_concentration_alert_html(rows):
    """
    Flags when a single sector dominates the current Buy list within a
    market (US or India), to surface concentration risk early.
    """
    by_market = {"US": {}, "India": {}}
    for pr, _score, _name, _html, summary_entry, market in rows:
        if pr != 1:  # only look at current Buy signals
            continue
        raw_sector = (summary_entry.get("sector") or "").strip()
        # Normalize so "unknown" / "Unknown" / "" / None are all treated the
        # same -- market_context's fallback returns lowercase "unknown",
        # which previously slipped past the "!= 'Unknown'" check below and
        # showed up in the email as a literal "unknown is 100%..." alert.
        sector = raw_sector if raw_sector and raw_sector.lower() != "unknown" else "Unknown"
        by_market.setdefault(market, {})
        by_market[market][sector] = by_market[market].get(sector, 0) + 1

    alerts = []
    for market, sector_counts in by_market.items():
        total = sum(sector_counts.values())
        # A single Buy signal being "100% of the list" isn't a meaningful
        # concentration risk -- require at least a couple of positions
        # before flagging it.
        if total < 2:
            continue
        top_sector, top_count = max(sector_counts.items(), key=lambda kv: kv[1])
        pct = round((top_count / total) * 100)
        if pct >= SECTOR_CONCENTRATION_THRESHOLD_PCT and top_sector != "Unknown":
            flag = "🇮🇳" if market == "India" else "🇺🇸"
            alerts.append(
                f'<div style="margin:4px 0 0;font-size:13px;color:#92400e;">'
                f'{flag} <strong>{top_sector}</strong> is {pct}% of your {market} Buy list ({top_count}/{total})</div>'
            )

    if not alerts:
        return ""

    return f"""
        <tr>
          <td style="padding:0 28px 14px;" class="email-padding">
            <div style="border:1px solid #EAD9B8;border-left:3px solid #A6812F;border-radius:4px;background:#FBF6EB;padding:12px 14px;">
              <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;font-size:11px;font-weight:700;color:#8A6D3B;text-transform:uppercase;letter-spacing:0.06em;">Concentration Watch</div>
              {''.join(alerts)}
            </div>
          </td>
        </tr>
    """


def build_error_summary_html(groups):
    """
    One-line banner near the top of the report summarizing fetch failures,
    instead of only showing them buried in the Errors sections.
    """
    failed_names = []
    for market in groups.values():
        failed_names.extend(name for _score, name, _html in market["Errors"])

    if not failed_names:
        return ""

    names_text = ", ".join(failed_names)
    return f"""
        <tr>
          <td style="padding:0 28px 14px;" class="email-padding">
            <div style="border:1px solid #E3C5BF;border-left:3px solid #8B2E2E;border-radius:4px;background:#FBF2F0;padding:12px 14px;">
              <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;font-size:11px;font-weight:700;color:#8B2E2E;text-transform:uppercase;letter-spacing:0.06em;">
                {len(failed_names)} stock{'s' if len(failed_names) != 1 else ''} failed to fetch
              </div>
              <div style="margin:5px 0 0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;font-size:13px;color:#6B3A38;">{names_text}</div>
            </div>
          </td>
        </tr>
    """


def build_pdf_attachment(report_html):
    """
    Renders the exact same HTML that goes into the email body to a PDF
    file, for attaching to the email (replaces the old CSV attachment).

    Uses headless Chromium via Playwright rather than WeasyPrint: Playwright
    bundles its own browser binary, so there's no dependency on system-level
    Pango/Cairo/GTK libraries (the source of the libpango dlopen errors seen
    with WeasyPrint on some Mac/Anaconda setups). It also renders the report
    exactly as a real browser would -- gradients, border-radius, and the
    emoji in the report headers all render correctly, which WeasyPrint
    doesn't guarantee.

    Returns None (and logs why) if rendering isn't possible, so a failure
    here never blocks the email itself from sending.
    """
    if sync_playwright is None:
        log.error(
            "playwright is not installed; skipping PDF attachment. "
            "Install it with: pip install playwright && playwright install chromium"
        )
        return None

    # Fixed viewport/page width. The @media print rule sizes .email-container
    # to 820px, so we give it 60px of horizontal margin (30px each side) to
    # sit inside without ever being clipped on the right edge.
    PAGE_WIDTH_PX = 880
    MARGIN_PX = 30

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            try:
                page = browser.new_page(viewport={"width": PAGE_WIDTH_PX, "height": 1080})
                # Force "print" media explicitly (rather than relying on
                # page.pdf()'s implicit default) so the @media print rules
                # above apply.
                page.emulate_media(media="print")
                page.set_content(report_html, wait_until="networkidle")

                # Render as normal paginated A4 sheets rather than one giant
                # continuous page. A single page sized to the full report
                # height (previous approach) produced a PDF several thousand
                # px tall for reports with many stock cards. That renders
                # fine in a dedicated PDF app (Acrobat, desktop Chrome) but
                # mobile/webmail inline previewers (iOS Mail QuickLook,
                # Gmail's built-in PDF viewer, etc.) choke on rasterizing an
                # extremely tall single page and show a blank/broken/failed
                # preview -- even though the attachment downloads fine. The
                # @media print rules above already set page-break-inside:
                # avoid on every row and page-break-after:avoid on headings,
                # so standard pagination no longer slices cards across page
                # boundaries the way it did before those rules existed.
                pdf_bytes = page.pdf(
                    format="A4",
                    print_background=True,
                    prefer_css_page_size=False,
                    margin={
                        "top": f"{MARGIN_PX}px",
                        "bottom": f"{MARGIN_PX}px",
                        "left": f"{MARGIN_PX}px",
                        "right": f"{MARGIN_PX}px",
                    },
                )
            finally:
                browser.close()
        return pdf_bytes
    except Exception as e:
        log.error(f"Failed to render PDF attachment: {e}")
        traceback.print_exc()
        return None


# -----------------------------
# Main
# -----------------------------
def main(mode, use_llm, detailed_llm=False):
    log.info(f"Starting stock analysis run. Mode: {mode}, LLM Enabled: {use_llm}, Detailed LLM: {detailed_llm}")

    # Diagnostic breakdown so a missing market (e.g. India) shows up in the
    # logs immediately instead of only being noticed once the email arrives.
    # classify_market only recognizes a ticker as "India" if it carries a
    # .NS (NSE) or .BO (BSE) suffix -- anything else silently falls back to
    # "US". A ticker like "RELIANCE" (no suffix) will both fail to fetch
    # from yfinance AND get bucketed under US Stocks > Errors, not India.
    _us_tickers = [f"{n} ({t})" for n, t in STOCKS.items() if classify_market(t) == "US"]
    _in_tickers = [f"{n} ({t})" for n, t in STOCKS.items() if classify_market(t) == "India"]
    log.info(
        f"Loaded {len(STOCKS)} stocks from config -> "
        f"US: {len(_us_tickers)} {_us_tickers} | India: {len(_in_tickers)} {_in_tickers}"
    )
    if STOCKS and not _in_tickers:
        log.warning(
            "No India tickers detected. Indian stock tickers must end in "
            "'.NS' (NSE) or '.BO' (BSE), e.g. 'RELIANCE.NS', or they will "
            "be treated as US tickers and likely fail to fetch."
        )

    report_html = """<!DOCTYPE html>
<html lang="en" xmlns="http://www.w3.org/1999/xhtml" xmlns:v="urn:schemas-microsoft-com:vml" xmlns:o="urn:schemas-microsoft-com:office:office">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta http-equiv="X-UA-Compatible" content="IE=edge">
<meta name="x-apple-disable-message-reformatting">
<meta name="format-detection" content="telephone=no, date=no, address=no, email=no">
<meta name="color-scheme" content="light">
<meta name="supported-color-schemes" content="light">
<title>Stock &amp; Commodity Report</title>
<!--[if mso]>
<noscript>
<xml>
<o:OfficeDocumentSettings>
<o:PixelsPerInch>96</o:PixelsPerInch>
</o:OfficeDocumentSettings>
</xml>
</noscript>
<style>table {border-collapse:collapse;} td, h1, h2, h3, p {font-family:Arial, sans-serif;}</style>
<![endif]-->
<style>
  body, table, td, a { -webkit-text-size-adjust:100%; -ms-text-size-adjust:100%; }
  table, td { mso-table-lspace:0pt; mso-table-rspace:0pt; }
  img { border:0; line-height:100%; outline:none; text-decoration:none; -ms-interpolation-mode:bicubic; }
  table { border-collapse:collapse !important; }
  body { height:100% !important; margin:0 !important; padding:0 !important; width:100% !important; background:#f4f6f8; }
  @media screen and (max-width:600px) {
    .email-container { width:100% !important; max-width:100% !important; border-radius:0 !important; border-left:0 !important; border-right:0 !important; }
    .email-padding { padding-left:14px !important; padding-right:14px !important; }
    h1 { font-size:20px !important; }
    h2 { font-size:15px !important; }
  }
  @media print {
    /* The PDF is rendered as standard paginated A4 sheets (see
       build_pdf_attachment) so it previews correctly in mobile/webmail
       PDF viewers. These rules stop a card or heading from being sliced
       across a page boundary. */
    tr { page-break-inside: avoid; break-inside: avoid; }
    h1, h2, h3 { page-break-after: avoid; break-after: avoid-page; }
    /* If a very long AI thesis still can't fit inside one avoided <tr>
       and the browser is forced to split it, don't strand a single
       line at the top/bottom of a page. */
    p, div { orphans: 3; widows: 3; }
    body { background:#ffffff !important; }
    .email-container {
      max-width:820px !important;
      width:820px !important;
      border:none !important;
      border-radius:0 !important;
      box-shadow:none !important;
    }
    /* The email view clips each stock's AI thesis to 140px so the inbox
       message stays short; the PDF is a standalone report and should show
       the full text instead of cutting sentences off mid-word. */
    .llm-thesis { max-height:none !important; overflow:visible !important; }
  }
</style>
</head>
<body style="margin:0;padding:0;background:#F2F0EC;font-family:Georgia,'Times New Roman',serif;color:#1B2233;">
  <div style="display:none;max-height:0;overflow:hidden;opacity:0;mso-hide:all;">Signal changes, price levels and precious-metals positioning for today's session.&nbsp;&zwnj;&nbsp;&zwnj;&nbsp;&zwnj;&nbsp;&zwnj;</div>
  <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="background:#F2F0EC;width:100%;min-width:100%;">
    <tr>
      <td align="center" style="padding:20px 16px;" class="email-padding">
        <table width="100%" cellpadding="0" cellspacing="0" role="presentation" class="email-container" style="max-width:680px;min-width:280px;background:#ffffff;border:1px solid #DAD5CB;border-radius:4px;overflow:hidden;">
          <tr>
            <td style="background:#14213D;padding:26px 28px 22px;" class="email-padding">
              <table width="100%" cellpadding="0" cellspacing="0" role="presentation">
                <tr>
                  <td>
                    <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;font-size:10px;font-weight:700;letter-spacing:0.16em;text-transform:uppercase;color:#B08D57;">Market Intelligence &nbsp;&bull;&nbsp; Daily Briefing</div>
                    <h1 style="margin:8px 0 0;font-family:Georgia,'Times New Roman',serif;font-size:24px;font-weight:400;line-height:1.3;color:#ffffff;letter-spacing:0.01em;">Equity &amp; Commodity Research Note</h1>
                  </td>
                </tr>
              </table>
            </td>
          </tr>
          <tr>
            <td style="height:3px;line-height:3px;font-size:0;background:linear-gradient(90deg,#B08D57,#D9C393 45%,#B08D57);">&nbsp;</td>
          </tr>
            <tr>
              <td style="padding:18px 28px 4px;" class="email-padding">
                <p style="margin:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;font-size:13px;line-height:1.65;color:#4A5063;">A concise portfolio briefing covering signal changes, price levels and precious-metals positioning, prepared for review across desktop and mobile.</p>
              </td>
            </tr>
            <tr>
              <td style="padding:12px 28px 18px;border-bottom:1px solid #EDEAE2;" class="email-padding">
                <p style="margin:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;font-size:11px;color:#8A8F9C;letter-spacing:0.02em;">Prepared for internal use &nbsp;&middot;&nbsp; Not investment advice</p>
              </td>
            </tr>
"""

    rows = []
    summary_rows = []
    ai_stocks_story_map = {}
    ai_portfolio_story_html = ""
    if use_llm:
        init_llm_generator()
        try:
            ai_stocks_story_map, ai_story_sources, ai_story_live, ai_portfolio_summary = generate_ai_stocks_story(list(STOCKS.keys()))
            log.info(
                f"AI Stocks Story: {len(ai_stocks_story_map)}/{len(STOCKS)} stocks covered by a "
                f"single combined AI call (live_search={ai_story_live}, sources_used={len(ai_story_sources)}, "
                f"portfolio_summary={'yes' if ai_portfolio_summary else 'no'})."
            )
            ai_portfolio_story_html = build_ai_portfolio_story_html(ai_portfolio_summary, len(STOCKS), ai_story_live)
        except Exception as e:
            log.error(f"AI Stocks Story batch generation failed, per-stock fallback text will be used instead: {e}")
            traceback.print_exc()
            ai_stocks_story_map = {}

    run_history = load_run_history()
    prev_stock_history = run_history.get("stocks", {})
    new_stock_history = {}
    track_record_state = run_history.get("track_record", {"open": {}, "closed": []})

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_stock = {
            executor.submit(process_stock, name, ticker, use_llm, detailed_llm, ai_stocks_story_map): (name, ticker)
            for name, ticker in STOCKS.items()
        }
        for future in as_completed(future_to_stock):
            stock_name, ticker = future_to_stock[future]
            market = classify_market(ticker)
            try:
                result = future.result()
                if result:
                    pr, score, name, row_html, summary_entry = result
                    prior_entry = prev_stock_history.get(ticker)
                    enrichment_html = build_stock_enrichment_html(summary_entry, prior_entry)
                    enriched_html = enrichment_html + row_html
                    rows.append((pr, score, name, enriched_html, summary_entry, market))
                    summary_rows.append(summary_entry)
                    if summary_entry.get("signal") != "ERROR":
                        new_stock_history[ticker] = {
                            "signal": summary_entry.get("signal"),
                            "priority": pr,
                            "price": summary_entry.get("current_price"),
                            "target": summary_entry.get("target"),
                            "stop_loss": summary_entry.get("stop_loss"),
                            "last_run_at": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),
                        }
                        update_track_record(
                            track_record_state,
                            ticker,
                            stock_name,
                            summary_entry.get("signal") or "",
                            summary_entry.get("current_price"),
                            summary_entry.get("target"),
                            summary_entry.get("stop_loss"),
                        )
                    elif prior_entry:
                        # Keep the last known-good entry so a single failed
                        # run doesn't wipe out change-tracking history.
                        new_stock_history[ticker] = prior_entry
            except Exception as exc:
                log.error(f"Error processing {stock_name} ({ticker}) in executor: {exc}")
                err_html = f"""
                <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="margin:12px 20px;border-radius:8px;background:#fff7f7;border:1px solid #f5c2c7;">
                    <tr>
                        <td style="padding:12px;color:#721c24;font-size:13px;"><strong>Error processing {ticker}:</strong> {exc}</td>
                    </tr>
                </table>
                """
                error_summary_entry = {
                    "stock_name": stock_name,
                    "ticker": ticker,
                    "signal": "ERROR",
                    "current_price": None,
                    "recommended_buy_level": None,
                    "ema20": None,
                    "upcoming_events": {},
                    "target": None,
                    "stop_loss": None,
                    "sector": None,
                    "priority": 4,
                    "total_score": 0,
                    "swing_setup": False,
                }
                rows.append((4, 0, ticker, err_html, error_summary_entry, market))
                summary_rows.append(error_summary_entry)
                prior_entry = prev_stock_history.get(ticker)
                if prior_entry:
                    new_stock_history[ticker] = prior_entry

    # This block is now correctly dedented and will run only once
    # Stocks are grouped first by market (US / India), then by signal (Buy/Hold/Sell/Errors)
    groups = {
        "US": {"Buy": [], "Hold": [], "Sell": [], "Errors": []},
        "India": {"Buy": [], "Hold": [], "Sell": [], "Errors": []},
    }
    for pr, score, name, html, _, market in rows:
        market_key = market if market in groups else "US"
        if pr == 1:
            groups[market_key]["Buy"].append((score, name, html))
        elif pr == 2:
            groups[market_key]["Hold"].append((score, name, html))
        elif pr == 3:
            groups[market_key]["Sell"].append((score, name, html))
        else:
            groups[market_key]["Errors"].append((score, name, html))

    for market_key in groups:
        for key in groups[market_key]:
            groups[market_key][key].sort(key=lambda item: item[0], reverse=True)

    buy_count = len(groups["US"]["Buy"]) + len(groups["India"]["Buy"])
    hold_count = len(groups["US"]["Hold"]) + len(groups["India"]["Hold"])
    sell_count = len(groups["US"]["Sell"]) + len(groups["India"]["Sell"])
    err_count = len(groups["US"]["Errors"]) + len(groups["India"]["Errors"])

    # Format date for the report header
    now_utc = datetime.now(ZoneInfo("UTC"))
    now_ist = now_utc.astimezone(ZoneInfo("Asia/Kolkata"))
    formatted_date_ist = now_ist.strftime('%A, %d %B %Y, %I:%M %p %Z')

    summary_html = f"""
        <tr>
          <td style="padding:18px 28px;border-top:1px solid #EDEAE2;" class="email-padding">
            <table width="100%" cellpadding="0" cellspacing="0" role="presentation">
              <tr>
                <td>
                  <h2 style="margin:0;font-family:Georgia,'Times New Roman',serif;font-weight:400;font-size:17px;color:#14213D;">Portfolio Snapshot</h2>
                </td>
                <td style="text-align:right;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;font-size:12px;color:#8A8F9C;">{formatted_date_ist}</td>
              </tr>
            </table>
            <p style="margin:8px 0 0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;font-size:13px;color:#4A5063;">Coverage: <strong style="color:#14213D;">{len(rows)}</strong> symbols &nbsp;&middot;&nbsp; Buy: <strong style="color:#2F5233;">{buy_count}</strong> &nbsp;&middot;&nbsp; Hold: <strong style="color:#8A6D3B;">{hold_count}</strong> &nbsp;&middot;&nbsp; Sell: <strong style="color:#8B2E2E;">{sell_count}</strong></p>
          </td>
        </tr>
    """

    quick_summary_groups = build_quick_summary(summary_rows)

    # Fetch commodity data early so we can include it in the quick summary
    # and reuse the same data object when rendering the full section below.
    commodity_data = None
    commodity_fetch_error = None
    try:
        tracker = CommodityTracker()
        commodity_data = tracker.get_commodity_data()
    except Exception as e:
        commodity_fetch_error = e
        log.error(f"Commodity tracker failed during data fetch: {e}")
        traceback.print_exc()

    # Build commodity summary bullets, kept separate per metal (gold/silver)
    # so each can render as its own group in the Executive Summary, mirroring
    # the Portfolio Heatmap's separate metal rows.
    prev_commodity_history = run_history.get("commodities", {})
    new_commodity_history = {}
    commodity_bullets_by_metal = {"gold": [], "silver": []}
    if commodity_data:
        for metal, key in [("Gold (22K)", "gold"), ("Silver", "silver")]:
            metal_data = commodity_data[key]
            change = metal_data.get("change", 0)
            current = metal_data.get("current", 0)
            history = metal_data.get("history", [])

            change_sign = "+" if change > 0 else ""
            direction = "↑" if change > 0 else ("↓" if change < 0 else "→")

            # Check if a buy signal is triggered
            buy_triggered = False
            if len(history) >= 3:
                recent = [r["change"] for r in history[-3:]]
                latest_c, prev_c, older_c = recent[-1], recent[-2], recent[-3]
                score = 0
                if latest_c <= -1.5: score += 4
                if latest_c <= -2.5: score += 4
                if prev_c <= -1.0:   score += 2
                if older_c <= -1.0:  score += 1
                if latest_c < prev_c: score += 2
                if latest_c < 0:     score += 1
                buy_triggered = score >= 8

            # Buy-signal streak: how many consecutive runs this metal has
            # shown a buy signal, so a single-day dip doesn't look the
            # same as a sustained one.
            prior_metal_history = prev_commodity_history.get(key, {})
            prior_streak = prior_metal_history.get("buy_streak", 0)
            current_streak = (prior_streak + 1) if buy_triggered else 0
            new_commodity_history[key] = {
                "buy_streak": current_streak,
                "last_buy_triggered": buy_triggered,
                "last_run_at": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),
            }
            streak_suffix = f" — {current_streak} runs in a row" if buy_triggered and current_streak > 1 else ""

            metal_key = key  # "gold" or "silver"
            if buy_triggered:
                commodity_bullets_by_metal[metal_key].append(
                    f"✅ Buy Signal: {metal} at &#8377;{current:.2f} ({change_sign}{change}%){streak_suffix}"
                )
            elif change <= -1.5:
                commodity_bullets_by_metal[metal_key].append(
                    f"📉 {metal} {direction} {change_sign}{change}% — watching for entry"
                )
            elif change >= 1.5:
                commodity_bullets_by_metal[metal_key].append(
                    f"📈 {metal} {direction} {change_sign}{change}% — momentum up"
                )
            else:
                commodity_bullets_by_metal[metal_key].append(
                    f"⬛ {metal} {direction} {change_sign}{change}% — stable"
                )

    # Build the Executive Summary grouped by US Stocks / India Stocks / Gold /
    # Silver -- the same grouping and visual language as the Portfolio
    # Heatmap above it, so both sections read consistently. Each market's
    # Buy/Hold/Sell recommendations are nested inside its own group, and each
    # commodity gets its own group with its signal as the "recommendation".
    sans_es = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
    group_styles = {
        "Buy":  ("#2F5233", "#E7EEE4"),
        "Hold": ("#8A6D3B", "#F3ECDD"),
        "Sell": ("#8B2E2E", "#F3E4E0"),
    }

    def _es_group_header(label, count):
        return (
            f'<div style="margin:14px 0 0;padding:6px 10px;background:#F4F2ED;border-radius:3px;'
            f'font-family:{sans_es};font-size:11px;font-weight:700;color:#14213D;text-transform:uppercase;'
            f'letter-spacing:0.05em;">{label} ({count})</div>'
        )

    def _es_market_block(label, market_bucket):
        total = sum(len(market_bucket.get(k) or []) for k in ("Buy", "Hold", "Sell"))
        if total == 0:
            return ""
        block_html = _es_group_header(label, total)
        for group_key in ("Buy", "Hold", "Sell"):
            bullets = market_bucket.get(group_key) or []
            if not bullets:
                continue
            color, bg = group_styles[group_key]
            block_html += (
                '<div style="margin-top:8px;">'
                f'<span style="display:inline-block;padding:2px 8px;border-radius:999px;'
                f'background:{bg};color:{color};font-size:11px;font-weight:700;'
                f'text-transform:uppercase;letter-spacing:0.03em;">{group_key}</span>'
                + "".join(
                    f'<div style="margin:4px 0 0;font-size:13px;color:#0f172a;">{item}</div>'
                    for item in bullets
                )
                + '</div>'
            )
        return block_html

    def _es_commodity_block(label, bullets):
        if not bullets:
            return ""
        block_html = _es_group_header(label, len(bullets))
        block_html += (
            '<div style="margin-top:8px;">'
            + "".join(
                f'<div style="margin:4px 0 0;font-size:13px;color:#0f172a;">{item}</div>'
                for item in bullets
            )
            + '</div>'
        )
        return block_html

    us_block_html = _es_market_block("🇺🇸 US Stocks", quick_summary_groups.get("US") or {})
    india_block_html = _es_market_block("🇮🇳 India Stocks", quick_summary_groups.get("India") or {})
    gold_block_html = _es_commodity_block("🥇 Gold (22K)", commodity_bullets_by_metal.get("gold") or [])
    silver_block_html = _es_commodity_block("🥈 Silver", commodity_bullets_by_metal.get("silver") or [])

    quick_summary_html = ""
    if us_block_html or india_block_html or gold_block_html or silver_block_html:
        quick_summary_html = f"""
            <tr>
              <td style="padding:0 28px 14px;" class="email-padding">
                <div style="border:1px solid #EDEAE2;border-left:3px solid #B08D57;border-radius:4px;background:#FAF9F6;padding:14px 16px;">
                  <div style="font-family:{sans_es};font-size:11px;font-weight:700;color:#14213D;text-transform:uppercase;letter-spacing:0.08em;">Executive Summary</div>
                  {us_block_html}
                  {india_block_html}
                  {gold_block_html}
                  {silver_block_html}
                </div>
              </td>
            </tr>
        """

    # India rendered before US: on a large report (many stocks / detailed
    # LLM mode) Gmail clips the message around ~102 KB and everything past
    # that point is cut off entirely. India was previously appended last
    # and could be silently truncated out of the visible email even though
    # it was fully computed (see "At a Glance" / Quick Summary above, which
    # are compact and always render near the top regardless of this order).
    section_html = get_market_section_html("🇮🇳 India Stocks", groups["India"])
    section_html += get_market_section_html("🇺🇸 US Stocks", groups["US"])

    # Commodity section (Gold & Silver) — built BEFORE stock sections so it appears
    # near the top of the email and is never clipped by Gmail's 102 KB limit.
    commodity_row_html = ""
    gold_levels = silver_levels = gold_plan = silver_plan = None
    if commodity_data is not None:
        try:
            gold_levels   = tracker.derive_buy_levels(commodity_data["gold"]["current"],   commodity_data["gold"]["history"])
            silver_levels = tracker.derive_buy_levels(commodity_data["silver"]["current"], commodity_data["silver"]["history"])
            gold_plan   = tracker.build_trade_plan(commodity_data["gold"]["current"],   commodity_data["gold"]["history"],   gold_levels)
            silver_plan = tracker.build_trade_plan(commodity_data["silver"]["current"], commodity_data["silver"]["history"], silver_levels)

            gold_card = tracker._commodity_card_html(
                name="Gold (22K)", ticker_label="XAU/INR",
                current_price=commodity_data["gold"]["current"],
                change=commodity_data["gold"]["change"],
                history=commodity_data["gold"]["history"],
                levels=gold_levels, plan=gold_plan,
                sparkline_history=commodity_data["gold"].get("sparkline_history"),
            )
            silver_card = tracker._commodity_card_html(
                name="Silver", ticker_label="XAG/INR",
                current_price=commodity_data["silver"]["current"],
                change=commodity_data["silver"]["change"],
                history=commodity_data["silver"]["history"],
                levels=silver_levels, plan=silver_plan,
                sparkline_history=commodity_data["silver"].get("sparkline_history"),
            )
            commodity_section_html = f"""
                <table width="100%" cellpadding="0" cellspacing="0" role="presentation">
                    <tr>
                        <td style="padding:12px 0 0;">
                            <h2 style="margin:0;font-size:15px;color:#111827;">Commodities (2)</h2>
                        </td>
                    </tr>
                </table>
                {gold_card}
                {silver_card}"""
            commodity_row_html = f"""
                <tr>
                  <td style="padding:0 28px 20px;" class="email-padding">
                    {commodity_section_html}
                  </td>
                </tr>
            """
        except Exception as e:
            log.error(f"Commodity card render failed: {e}")
            traceback.print_exc()
            commodity_row_html = """
                <tr>
                  <td style="padding:0 28px 20px;" class="email-padding">
                    <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="margin:12px 0;border-radius:12px;background:#fff7f7;border:1px solid #f5c2c7;">
                      <tr>
                        <td style="padding:14px;color:#721c24;font-size:13px;">
                          <strong>Commodities Unavailable:</strong> Could not render Gold &amp; Silver cards.
                        </td>
                      </tr>
                    </table>
                  </td>
                </tr>
            """
    else:
        # commodity_fetch_error was previously captured and logged but never
        # surfaced in the report itself -- the reader just saw a generic
        # "unavailable" line with no indication of why. Show the actual
        # reason (str(exception), not a full traceback) so this reads the
        # same as a real data-quality note rather than a silent gap.
        error_detail = (
            f" ({html.escape(str(commodity_fetch_error))})" if commodity_fetch_error else ""
        )
        commodity_row_html = f"""
            <tr>
              <td style="padding:0 28px 20px;" class="email-padding">
                <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="margin:12px 0;border-radius:12px;background:#fff7f7;border:1px solid #f5c2c7;">
                  <tr>
                    <td style="padding:14px;color:#721c24;font-size:13px;">
                      <strong>Commodities Unavailable:</strong> Could not fetch Gold &amp; Silver prices at this time{error_detail}.
                    </td>
                  </tr>
                </table>
              </td>
            </tr>
        """

    # Snapshot of the header/banner HTML built so far, so the email body can
    # reuse it without the full per-stock "portfolio" sections that follow.
    report_html_header = report_html

    # Action Plan table -- Stock/commodity | Add Below | Current Status |
    # Next Action | Profit Booking, grouped like everything else (US /
    # India / Gold / Silver). Placed right after the Executive Summary in
    # both the PDF and the email body.
    action_plan_html = build_action_plan_table_html(
        summary_rows, commodity_data=commodity_data,
        gold_levels=gold_levels, silver_levels=silver_levels,
        gold_plan=gold_plan, silver_plan=silver_plan,
    )

    # Build new report-enhancement blocks
    error_summary_html = build_error_summary_html(groups)
    quick_jump_html = build_quick_jump_table_html(
        rows, commodity_data=commodity_data, commodity_buy_signals=new_commodity_history
    )
    concentration_alert_html = build_concentration_alert_html(rows)
    track_record_html = build_track_record_html(track_record_state)

    footer_html = (
        build_compliance_block_html(
            report_kind="equity",
            run_note=(
                "This briefing is generated from automated technical, fundamental and "
                "sentiment models and is provided for informational purposes only. It does "
                "not constitute investment advice or a recommendation to buy or sell any "
                "security. Verify all prices, levels and news against a live source before acting."
            ),
        )
        + """
          </table>
        </td>
      </tr>
    </table>
  </body>
</html>
    """
    )

    # PDF gets the full report: heatmap, quick summary, portfolio snapshot,
    # commodities, concentration alerts, and every stock's detailed card.
    # Commodities first, then stock sections — ensures commodities are never clipped
    report_html += (
        error_summary_html
        + quick_jump_html
        + ai_portfolio_story_html
        + track_record_html
        + quick_summary_html
        + action_plan_html
        + summary_html
        + commodity_row_html
        + concentration_alert_html
        + section_html
    )
    report_html += footer_html

    # Email gets the Portfolio Heatmap + Quick Summary, PLUS the small AI
    # Portfolio Story paragraph (one professional-level summary combining
    # every stock's AI Stock Story, from the same single combined AI call
    # -- no extra AI request). The full per-stock "portfolio" breakdown
    # (summary/commodity cards/concentration alerts/detailed stock cards
    # with their individual AI Stock Story bullets) is dropped from the
    # inline message and lives only in the attached PDF -- keeps the email
    # short and avoids Gmail's ~102 KB body-clipping entirely rather than
    # just working around it.
    email_html = report_html_header + quick_jump_html + ai_portfolio_story_html + track_record_html + quick_summary_html + action_plan_html + footer_html

    # Persist this run's state so the next run can diff signals/prices
    # against it (change badges, breach alerts, commodity streaks).
    save_run_history({
        "stocks": new_stock_history,
        "commodities": new_commodity_history,
        "track_record": track_record_state,
    })

    pdf_bytes = build_pdf_attachment(report_html)

    if os.getenv("DRY_RUN", "false").lower() == "true":
        with open("report.html", "w") as f:
            f.write(report_html)
        with open("email.html", "w") as f:
            f.write(email_html)
        if pdf_bytes:
            with open("report.pdf", "wb") as f:
                f.write(pdf_bytes)
            log.info("Report saved to report.html, email.html and report.pdf (DRY_RUN enabled)")
        else:
            log.info("Report saved to report.html and email.html only -- PDF rendering failed or unavailable (DRY_RUN enabled)")
    else:
        send_email(email_html, mode, pdf_attachment=pdf_bytes, pdf_filename="stock_report.pdf")
        log.info("Email report sent successfully.")
    log.info("Stock analysis run finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run stock analysis.")
    parser.add_argument(
        "--mode",
        type=str,
        default="eod",
        choices=["real_time", "eod"],
        help="Specify the mode: real_time or eod"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable AI-powered reasoning to speed up execution."
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Include full detailed LLM analysis in the report (can increase size)."
    )
    args = parser.parse_args()
    main(args.mode, use_llm=not args.no_llm, detailed_llm=args.detailed)