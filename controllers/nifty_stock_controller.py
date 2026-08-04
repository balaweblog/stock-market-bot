"""
stock_market_advisor.py

Standalone companion to main.py / swing_trade_advisor.py / mutual_fund_advisor.py.
Runs a "last-7-days" stock market news review -- broad market/macro
context, watchlist-stock news, and sector performance -- against
whichever free LLM backend main.py already knows how to set up, then
emails the result to the same recipients configured for the other
reports (EMAIL_TO / EMAIL_CC in config.py / the workflow yaml's env
vars).

This deliberately reuses main.py's LLM-selection and email-credential
plumbing, AND swing_trade_advisor.py's live-search cascade
(generate_analysis) and JSON/source helpers, instead of duplicating any
of it -- so all four scripts stay in sync with whatever provider/quota
situation is configured. See swing_trade_advisor.py's own docstring for
the full list of live-search fallback tiers (Groq compound ->
compound-mini -> Tavily+Groq -> Gemini grounding -> Mistral web search ->
non-live fallback, gated by REQUIRE_LIVE_DATA).

WHY A MULTI-STAGE PIPELINE (not one giant prompt): same rationale as
mutual_fund_advisor.py -- asking one model call to search, verify, and
structure macro news + watchlist-stock news + sector news all at once
reliably truncates or hallucinates. Instead:
  Stage 1 -- Market & macro (one live-search call, ~17 topics).
  Stage 2 -- Watchlist-stock news, in rotating batches of
             STOCK_PER_BATCH tickers per live-search call.
  Stage 3 -- Sector performance, in rotating batches of
             SECTORS_PER_BATCH sectors per live-search call.
  Stage 4 -- Synthesis: a single NON-live call that reasons over the
             already-gathered, already-cited output of Stages 1-3 to
             produce the top-developments list for the executive
             summary. This stage doesn't fetch new facts, so it isn't
             gated by REQUIRE_LIVE_DATA.

CAVEAT: this is not investment advice, and not a substitute for your
own research or a SEBI-registered adviser. Web search results can be
stale, incomplete, or misread by the model. Treat every price move,
news item, and "recent" claim as a starting point to verify against an
exchange filing, company announcement, or live quote -- not investment
advice. See compliance.py for the full disclosure block attached to
every email.
"""

import os
import re
import sys
import json
import html
import traceback
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import smtplib
from email.mime.text import MIMEText

import yfinance as yf

from utils import config
from utils.logger import log
from utils.compliance import build_compliance_block_html
from utils.constants import WATCHLIST
from controllers.swing_controller import (
    _env_int,
    generate_analysis,
    _strip_code_fences,
    _build_sources_html,
    _generate_local,
    _require_live_or_abort,
)
from llm import llm_backend

SANS = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
SERIF = "Georgia,'Times New Roman',serif"

# -----------------------------
# Watchlist & sector universe
# -----------------------------
# WATCHLIST is imported from constants.py (previously loaded at runtime via
# the STOCK_WATCHLIST_JSON environment variable).

MARKET_TOPICS = [
    "Nifty 50", "Sensex", "Bank Nifty", "Midcap Index", "Smallcap Index",
    "RBI", "Inflation", "Interest Rates", "Rupee vs Dollar",
    "FII Activity", "DII Activity", "US Markets", "Federal Reserve",
    "China", "Crude Oil", "Gold", "IPO Market",
]

SECTORS_STOCK = [
    "Banking", "IT", "Pharma", "Auto", "FMCG",
    "Metals", "Energy", "Realty", "Infrastructure",
    "Telecom", "PSU", "Defence", "US Technology",
]

STOCK_PER_BATCH = _env_int("STOCK_PER_BATCH", 3)
SECTORS_PER_BATCH = _env_int("STOCK_SECTORS_PER_BATCH", 5)

SOURCE_QUALITY_NOTE = (
    "Source quality: prioritize exchange filings/announcements (NSE, "
    "BSE), SEBI, RBI circulars, company press releases, and reputable "
    "financial media (ET Markets, Mint, Moneycontrol, Business "
    "Standard, Reuters, Bloomberg). Do NOT rely on or cite social "
    "media posts (X/Twitter, Telegram, Instagram, Facebook) -- they are "
    "unverifiable and frequently wrong."
)

NO_FABRICATION_NOTE = (
    "Do not fabricate a date, price level, percentage move, or "
    "corporate-action detail. If you cannot verify a specific figure "
    "from a real source published in the window below, say so "
    "explicitly (e.g. \"not independently verifiable this run\") "
    "instead of inventing one."
)


# -----------------------------
# Weekly return -- computed from real price data, not the LLM
# -----------------------------
# The LLM's "weekly_return_pct" field is only ever a search-based guess and
# frequently comes back null (e.g. when its search pass found no news for a
# stock that week). We already know exactly which stocks are on the
# watchlist, so compute the actual trailing-week % move directly via
# yfinance and use that instead, falling back to the LLM's figure only if a
# ticker has no mapping.
DEFAULT_WATCHLIST_TICKERS = {
    "Reliance Industries": "RELIANCE.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Infosys": "INFY.NS",
    "Tata Consultancy Services": "TCS.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Larsen & Toubro": "LT.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Shriram Finance": "SHRIRAMFIN.NS",
}


def _load_ticker_map():
    merged = dict(DEFAULT_WATCHLIST_TICKERS)
    raw = os.getenv("STOCK_TICKER_MAP_JSON")
    if raw:
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                merged.update({
                    k.strip(): v.strip()
                    for k, v in data.items()
                    if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip()
                })
            else:
                log.warning("STOCK_TICKER_MAP_JSON is not a JSON object -- ignoring.")
        except json.JSONDecodeError as e:
            log.warning(f"STOCK_TICKER_MAP_JSON is not valid JSON ({e}) -- ignoring.")
    return merged


TICKER_MAP = _load_ticker_map()


def fetch_weekly_returns(watchlist):
    """
    Real trailing ~1-week % price move per watchlist stock, computed from
    actual close prices (yfinance) rather than an LLM guess.

    Returns {stock_name: pct_change_float_or_None}. None means either the
    stock has no ticker mapping (see STOCK_TICKER_MAP_JSON / add it to
    DEFAULT_WATCHLIST_TICKERS) or the price fetch failed for it.
    """
    results = {name: None for name in watchlist}

    unmapped = [name for name in watchlist if not TICKER_MAP.get(name)]
    if unmapped:
        log.warning(
            "No ticker mapping for: %s -- weekly return will show n/a for "
            "these until added to DEFAULT_WATCHLIST_TICKERS or "
            "STOCK_TICKER_MAP_JSON." % ", ".join(unmapped)
        )

    for name in watchlist:
        ticker = TICKER_MAP.get(name)
        if not ticker:
            continue
        try:
            hist = yf.Ticker(ticker).history(period="12d", interval="1d", auto_adjust=True)
            closes = hist["Close"].dropna()
            if len(closes) < 2:
                log.warning(f"Not enough price history to compute weekly return for {name} ({ticker}).")
                continue
            recent = float(closes.iloc[-1])
            # ~5 trading days back approximates "last week"; clamp for short history.
            week_ago = float(closes.iloc[max(0, len(closes) - 6)])
            if week_ago:
                results[name] = round((recent - week_ago) / week_ago * 100, 2)
        except Exception as e:
            log.warning(f"Could not compute weekly return for {name} ({ticker}): {e}")

    return results


def _chunks(items, size):
    size = max(1, size)
    return [items[i:i + size] for i in range(0, len(items), size)]


# -----------------------------
# Schedule / lookback window
# -----------------------------
# Unlike mutual_fund_advisor.py's monthly cadence, this script is
# intended to run WEEKLY (stock news moves too fast for a 30-day
# window to stay useful). Suggested cron for the workflow yaml (03:00
# UTC every Monday = 08:30 IST, ahead of market open, giving a clean
# "past week" wrap covering the prior trading week):
#   cron: '0 3 * * 1'
# It can also be triggered manually (workflow_dispatch) at any time --
# the 7-day window below is always relative to "today", not to a fixed
# calendar week, so an on-demand run mid-week still makes sense.
def _run_context():
    now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
    today_str = now_ist.strftime("%d %B %Y")
    window_start_str = (now_ist - timedelta(days=7)).strftime("%d %B %Y")
    lookback_note = (
        f"STRICTLY covering the window from {window_start_str} to {today_str} "
        "(the past 7 days) only -- exclude anything older, and exclude "
        "anything you cannot confidently date within this window."
    )
    return today_str, now_ist, lookback_note


# -----------------------------
# Stage 1 -- Market & macro
# -----------------------------
def build_market_prompt(today_str, lookback_note):
    topic_list = ", ".join(MARKET_TOPICS)
    return f"""Act as a CFA charterholder and equity market analyst covering the Indian market. Using the most current data as of {today_str}, {lookback_note}

Find and summarize the most important developments across these topics: {topic_list}.

{SOURCE_QUALITY_NOTE}

{NO_FABRICATION_NOTE}

For each topic, search for what actually happened in the window above (index levels/moves, RBI actions, inflation prints, FII/DII net flows, Fed decisions, crude/gold moves, notable IPOs, etc.) -- do not pad with generic commentary that isn't tied to a dated event.

OUTPUT FORMAT -- respond with ONLY raw JSON matching this schema, nothing else (no markdown, no code fences, no commentary before or after):

{{
  "market_sentiment": "Bullish | Neutral | Bearish",
  "sentiment_reason": "One or two sentences on why",
  "developments": [
    {{
      "date": "DD Month YYYY",
      "topic": "one of: {topic_list}",
      "headline": "Short headline",
      "summary": "2-3 sentence summary",
      "investor_impact": "How this specifically affects an active equity investor holding a diversified Indian large/mid-cap stock portfolio -- not just a restatement of the headline",
      "confidence": "High | Medium | Low"
    }}
  ]
}}
List 15-25 genuine, dated developments across the topics above, in roughly chronological order. It is fine for some topics to have fewer items than others if less happened.
"""


# -----------------------------
# Stage 2 -- Watchlist-stock news (batched)
# -----------------------------
def build_stock_prompt(stocks_batch, today_str, lookback_note):
    listing = "\n".join(f"- {s}" for s in stocks_batch)
    return f"""Act as an equity research analyst. Using the most current data as of {today_str}, {lookback_note}

For EACH of the following stocks, research recent news, price action, and corporate actions:
{listing}

{SOURCE_QUALITY_NOTE}

{NO_FABRICATION_NOTE}

For each stock, look for: earnings/results, management commentary, analyst rating or target-price changes, M&A or capex announcements, regulatory action, order wins/losses, and any other news specifically naming this company.

OUTPUT FORMAT -- respond with ONLY raw JSON matching this schema, nothing else (no markdown, no code fences, no commentary before or after):

{{
  "stocks": [
    {{
      "stock_name": "Exact name as given above",
      "news_timeline": [
        {{
          "date": "DD Month YYYY",
          "headline": "Short headline",
          "summary": "2-3 sentence summary",
          "why_it_matters": "One sentence",
          "impact_on_stock": "Positive | Neutral | Negative",
          "confidence": "High | Medium | Low"
        }}
      ],
      "corporate_actions": "Earnings, buybacks, splits, bonus issues, board changes, M&A, or capex announcements this week -- or 'No material disclosed changes found this window' if none verifiable",
      "weekly_return_pct": "Approximate % price move this window as a plain number string (e.g. '2.4'), or null if not verifiable. Return the value under the exact key 'weekly_return_pct' only.",
      "analyst_view": "Any analyst rating/target-price change mentioned, or 'No update found this window'",
      "current_price": "Latest price as a plain number string, or 'Not disclosed'",
      "market_cap_cr": "Market cap in ₹ crore as a plain number string, or 'Not disclosed'",
      "pe_ratio": "PE ratio as a plain number string, or 'Not disclosed'",
      "sector": "Sector name or 'Not disclosed'",
      "beta": "Beta as a plain number string, or 'Not disclosed'",
      "risk_level": "Low | Medium | High or 'Not disclosed'",
      "decision_note": "One short sentence on why this stock is attractive, neutral, or unattractive for a short-term or swing-trading investor",
      "assessment": "Positive | Neutral | Negative",
      "short_term_outlook": "1-4 week outlook, 1-2 sentences",
      "recommendation": "Strong Buy | Buy | Hold | Review | Reduce | Exit"
    }}
  ]
}}
Include one object per stock listed above, even if you found little news for it (in that case say so plainly rather than inventing content).
"""


# -----------------------------
# Stage 3 -- Sector performance (batched)
# -----------------------------
def build_sector_prompt(sectors_batch, today_str, lookback_note):
    listing = ", ".join(sectors_batch)
    return f"""Act as an equity sector analyst covering the Indian and relevant global (US tech) markets. Using the most current data as of {today_str}, {lookback_note}

For EACH of these sectors, summarize the past week's performance and outlook: {listing}.

{SOURCE_QUALITY_NOTE}

{NO_FABRICATION_NOTE}

OUTPUT FORMAT -- respond with ONLY raw JSON matching this schema, nothing else (no markdown, no code fences, no commentary before or after):

{{
  "sectors": [
    {{
      "sector": "Exact sector name as given above",
      "weekly_performance": "1-2 sentences on how the sector index/theme moved this window",
      "key_news": "1-2 sentences on the most important dated news item(s)",
      "outlook": "1-2 sentences, next 2-4 weeks",
      "rating": "Integer 1-5 (5 = most favorable outlook)"
    }}
  ]
}}
Include one object per sector listed above.
"""


# -----------------------------
# Stage 4 -- Synthesis (non-live)
# -----------------------------
def build_synthesis_prompt(market_data, stocks_data, sectors_data, watchlist, today_str):
    context = json.dumps(
        {"market": market_data, "stocks": stocks_data, "sectors": sectors_data},
        ensure_ascii=False,
    )
    stock_list = ", ".join(watchlist)
    return f"""You are a CFA charterholder and equity research analyst. You have ALREADY researched the material below (already dated/sourced -- do not re-search or add new facts, just reason over what's here):

{context}

The investor's watchlist is exactly these {len(watchlist)} stocks: {stock_list}.

Using ONLY the material above, produce a synthesis for an active equity investor. Respond with ONLY raw JSON matching this schema, nothing else (no markdown, no code fences, no commentary before or after):

{{
  "top_developments": ["Up to 10 of the single most important developments from the material above, one sentence each, most important first"]
}}
As of {today_str}.
"""


# -----------------------------
# JSON parsing helpers
# -----------------------------
def _parse_json_object(text):
    cleaned = _strip_code_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


# -----------------------------
# Stage runners
# -----------------------------
def run_market_stage(today_str, lookback_note):
    prompt = build_market_prompt(today_str, lookback_note)
    text, sources, live = generate_analysis(prompt, max_tokens=3000)
    if not text:
        log.error("No LLM backend produced Stage 1 (market/macro) output. Aborting without sending an email.")
        sys.exit(1)
    _require_live_or_abort(live, "Stage 1 (market/macro)")
    data = _parse_json_object(text)
    if not isinstance(data, dict):
        log.warning("Stage 1 output could not be parsed as JSON -- proceeding with an empty market section.")
        data = {}
    data.setdefault("developments", [])
    data.setdefault("market_sentiment", "Neutral")
    return data, sources, live


def run_stock_stage(today_str, lookback_note):
    all_stocks, sources, used_live = [], [], False
    for batch in _chunks(WATCHLIST, STOCK_PER_BATCH):
        log.info(f"Stage 2 -- stock batch: {', '.join(batch)}")
        prompt = build_stock_prompt(batch, today_str, lookback_note)
        stock_queries = [f"{name} share price target news {today_str}" for name in batch]
        text, s, live = generate_analysis(prompt, max_tokens=3200, extra_context_queries=stock_queries)
        if not text:
            log.error(f"No LLM output for stock batch ({', '.join(batch)}) -- skipping this batch.")
            continue
        _require_live_or_abort(live, f"Stage 2 (stock batch: {', '.join(batch)})")
        data = _parse_json_object(text)
        stocks = data.get("stocks") if isinstance(data, dict) else None
        if isinstance(stocks, list):
            all_stocks.extend(stocks)
        else:
            log.warning(f"Could not parse stock JSON for batch: {', '.join(batch)}")
        for src in s:
            if src not in sources:
                sources.append(src)
        used_live = used_live or live

    real_returns = fetch_weekly_returns(WATCHLIST)
    by_name = {s.get("stock_name"): s for s in all_stocks if isinstance(s, dict)}
    for name, pct in real_returns.items():
        if pct is None:
            continue  # no ticker mapping or fetch failed -- leave the LLM's field (likely still null) alone
        stock = by_name.get(name)
        if stock is not None:
            stock["weekly_return_pct"] = str(pct)
        else:
            # LLM produced no card at all for this stock -- still surface the real number.
            all_stocks.append({"stock_name": name, "weekly_return_pct": str(pct)})

    return all_stocks, sources, used_live


def run_sector_stage(today_str, lookback_note):
    all_sectors, sources, used_live = [], [], False
    for batch in _chunks(SECTORS_STOCK, SECTORS_PER_BATCH):
        log.info(f"Stage 3 -- sector batch: {', '.join(batch)}")
        prompt = build_sector_prompt(batch, today_str, lookback_note)
        text, s, live = generate_analysis(prompt, max_tokens=2000)
        if not text:
            log.error(f"No LLM output for sector batch ({', '.join(batch)}) -- skipping this batch.")
            continue
        _require_live_or_abort(live, f"Stage 3 (sector batch: {', '.join(batch)})")
        data = _parse_json_object(text)
        sectors = data.get("sectors") if isinstance(data, dict) else None
        if isinstance(sectors, list):
            all_sectors.extend(sectors)
        else:
            log.warning(f"Could not parse sector JSON for batch: {', '.join(batch)}")
        for src in s:
            if src not in sources:
                sources.append(src)
        used_live = used_live or live
    return all_sectors, sources, used_live


def _plain_generate(prompt, max_tokens=3800):
    """
    Stage 4 is reasoning over already-gathered, already-cited output from
    Stages 1-3 -- it isn't fetching new facts, so it doesn't need (and
    isn't gated by) live web search. Tries Groq (plain) -> Gemini (plain)
    -> local model, mirroring generate_analysis's backend order without
    any of the search-cascade machinery.
    """
    backend = llm_backend.init_llm_generator()
    log.info(f"Stage 4 (synthesis) using LLM backend: {backend}")

    if backend == "groq" and getattr(llm_backend, "groq_client", None) is not None:
        try:
            response = llm_backend.groq_client.chat.completions.create(
                model=llm_backend.SYNTHESIS_MODELS[0],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=3800,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            log.error(f"Groq synthesis call failed: {e}")

    have_gemini = getattr(llm_backend, "gemini_client", None) is not None or (
        os.getenv("GOOGLE_API_KEY") and getattr(llm_backend, "genai", None) is not None
    )
    if backend == "gemini" or have_gemini:
        try:
            if getattr(llm_backend, "gemini_client", None) is None:
                llm_backend.gemini_client = llm_backend.genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
            response = llm_backend.gemini_client.models.generate_content(
                model=llm_backend.GEMINI_MODEL, contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            log.error(f"Gemini synthesis call failed: {e}")

    local_backend = llm_backend.init_llm_generator(force_local=True)
    if local_backend == "local" and llm_backend.llm_pipeline is not None:
        text = _generate_local(prompt)
        if text:
            return text

    return None


def run_synthesis_stage(market_data, stocks_data, sectors_data, today_str):
    prompt = build_synthesis_prompt(market_data, stocks_data, sectors_data, WATCHLIST, today_str)
    text = _plain_generate(prompt)
    if not text:
        log.error("No LLM backend produced Stage 4 (synthesis) output. Aborting without sending an email.")
        sys.exit(1)
    data = _parse_json_object(text)
    if not isinstance(data, dict):
        log.warning("Stage 4 output could not be parsed as JSON -- proceeding with an empty synthesis section.")
        data = {}
    data.setdefault("top_developments", [])
    return data


# -----------------------------
# HTML rendering
# -----------------------------
def _esc(v):
    return html.escape(str(v)) if v is not None else ""


def _sentiment_color(label):
    label = (label or "").strip().lower()
    if label == "bullish":
        return "#1E7A46"
    if label == "bearish":
        return "#B0473F"
    return "#8A6D1D"


def _reco_color(reco):
    reco = (reco or "").strip().lower()
    if reco in ("strong buy", "buy"):
        return "#1E7A46"
    if reco in ("hold", "keep", "monitor", "no action"):
        return "#8A6D1D"
    if reco in ("review", "within 7 days"):
        return "#C8792A"
    if reco in ("reduce", "exit", "sell", "immediate"):
        return "#B0473F"
    return "#4A5063"


def _section_title(text):
    return (
        f'<h2 style="margin:22px 0 10px;font-family:{SERIF};font-weight:400;'
        f'font-size:18px;color:#14213D;border-bottom:2px solid #B08D57;padding-bottom:6px;">{_esc(text)}</h2>'
    )


def render_executive_summary(market_data, synthesis_data):
    sentiment = market_data.get("market_sentiment", "Neutral")
    color = _sentiment_color(sentiment)
    reason = market_data.get("sentiment_reason", "")
    top_dev = synthesis_data.get("top_developments") or []
    items = "".join(f'<li style="margin:0 0 6px;">{_esc(d)}</li>' for d in top_dev[:10])
    return f"""
    {_section_title("1. Executive Summary")}
    <div style="display:inline-block;padding:4px 12px;border-radius:20px;background:{color}1A;
                font-family:{SANS};font-size:12px;font-weight:700;color:{color};margin-bottom:8px;">
      Overall Market Sentiment (7 Days): {_esc(sentiment)}
    </div>
    <p style="margin:6px 0 12px;font-family:{SANS};font-size:13px;color:#4A5063;">{_esc(reason)}</p>
    <div style="font-family:{SANS};font-size:12.5px;font-weight:700;color:#14213D;margin-bottom:4px;">Top developments this week</div>
    <ol style="margin:0;padding-left:18px;font-family:{SANS};font-size:12.5px;line-height:1.6;color:#1B2233;">{items}</ol>
    """


def _format_weekly_return_display(stock_data):
    for key in ("weekly_return_pct", "weekly_return", "weekly_return_percent", "return_pct"):
        value = stock_data.get(key)
        if value in (None, "", "null", "None"):
            continue
        if isinstance(value, (int, float)):
            return f"{value}%"
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned in ("", "null", "None"):
                continue
            if cleaned.endswith("%"):
                return cleaned
            return f"{cleaned}%"
    return "n/a"


def _decision_signal_text(stock_data):
    explicit_signal = stock_data.get("decision_signal") or stock_data.get("decision_signal_label")
    if explicit_signal:
        return str(explicit_signal), _reco_color(stock_data.get("recommendation"))

    reco = (stock_data.get("recommendation") or "").strip()
    assess = (stock_data.get("assessment") or "").strip()
    pieces = []
    if reco and reco.lower() not in ("-", "none"):
        pieces.append(reco)
    if assess and assess.lower() not in ("-", "none"):
        pieces.append(assess)
    if not pieces:
        pieces.append("Review")
    return " / ".join(pieces), _reco_color(reco) if reco else _sentiment_color(assess)


def _quality_status(stock_data):
    decision_note = str(stock_data.get("decision_note") or "").strip()
    recommendation = str(stock_data.get("recommendation") or "").strip()
    assessment = str(stock_data.get("assessment") or "").strip()
    current_price = stock_data.get("current_price")
    if decision_note and recommendation and assessment:
        return "Verified", "#1E7A46"
    if decision_note and current_price not in (None, "", "null", "None"):
        return "Verified", "#1E7A46"
    if decision_note or current_price not in (None, "", "null", "None"):
        return "Partial", "#B08D57"
    return "Needs review", "#B0473F"


def _action_now_text(stock_data):
    reco = (stock_data.get("recommendation") or "").strip().lower()
    if reco in {"strong buy", "buy"}:
        return "Add on pullbacks or near support; keep position size disciplined."
    if reco in {"hold", "review"}:
        return "Wait for confirmation and a clearer catalyst before adding."
    if reco in {"reduce", "exit", "sell"}:
        return "Reduce exposure or avoid new entry until the setup improves."
    return "Monitor the setup and wait for a clearer confirmation signal."


def render_stock_cards(stocks_data):
    if not stocks_data:
        return _section_title("2. Watchlist Stock Analysis") + (
            f'<p style="font-family:{SANS};font-size:12.5px;color:#B0473F;">No stock data could be generated this run.</p>'
        )
    cards = []
    for st in stocks_data:
        name = st.get("stock_name", "Unknown Stock")
        news = st.get("news_timeline") or []
        news_html = ""
        if news:
            for n in news:
                headline = n.get("headline") or "No major news found"
                summary = n.get("summary") or "No significant news or announcements were found for this company in the given time window."
                confidence = n.get("confidence") or "Low"
                date_text = n.get("date") or "Not specified"
                why_it_matters = n.get("why_it_matters") or "The evidence is limited or not independently verifiable this run."
                impact = n.get("impact_on_stock") or "Neutral"
                badge_color = "#B0473F" if str(confidence).lower() == "low" else "#B08D57"
                news_html += (
                    f'<div style="margin:0 0 10px;padding:8px 10px;background:#F8F7F3;border-left:3px solid {badge_color};border-radius:2px;">'
                    f'<div style="font-family:{SANS};font-size:11px;color:#8A8F9C;">{_esc(date_text)} &middot; Confidence: {_esc(confidence)}</div>'
                    f'<div style="font-family:{SANS};font-size:12.5px;font-weight:700;color:#14213D;margin:2px 0;">{_esc(headline)}</div>'
                    f'<div style="font-family:{SANS};font-size:12px;color:#4A5063;">{_esc(summary)}</div>'
                    f'<div style="font-family:{SANS};font-size:11.5px;color:#4A5063;margin-top:3px;"><em>Why it matters:</em> {_esc(why_it_matters)} &middot; Impact: {_esc(impact)}</div>'
                    f'</div>'
                )
        else:
            news_html = f'<p style="font-family:{SANS};font-size:12px;color:#8A8F9C;">No dated news items found this window.</p>'

        reco = st.get("recommendation", "-")
        assess = st.get("assessment", "-")
        ret_str = _format_weekly_return_display(st)
        signal_text, signal_color = _decision_signal_text(st)
        quality_status, quality_color = _quality_status(st)
        action_now = _action_now_text(st)

        snapshot_items = [
            ("Price", st.get("current_price") or "Not disclosed"),
            ("MCap (₹ Cr)", st.get("market_cap_cr") or "Not disclosed"),
            ("PE", st.get("pe_ratio") or "Not disclosed"),
            ("Sector", st.get("sector") or "Not disclosed"),
            ("Beta", st.get("beta") or "Not disclosed"),
            ("Risk", st.get("risk_level") or "Not disclosed"),
        ]
        snapshot_html = "".join(
            f'<tr><td style="width:50%;padding:6px 0;vertical-align:top;font-family:{SANS};font-size:11px;color:#8A8F9C;">'
            f'<div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.04em;">{_esc(label)}</div>'
            f'<div style="margin-top:3px;font-size:12px;font-weight:700;color:#14213D;">{_esc(value)}</div></td>'
            f'<td style="width:50%;padding:6px 0;vertical-align:top;font-family:{SANS};font-size:11px;color:#8A8F9C;">'
            f'<div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.04em;">{_esc(next_label)}</div>'
            f'<div style="margin-top:3px;font-size:12px;font-weight:700;color:#14213D;">{_esc(next_value)}</div></td></tr>'
            for (label, value), (next_label, next_value) in zip(snapshot_items[::2], snapshot_items[1::2])
        )
        if len(snapshot_items) % 2 == 1:
            last_label, last_value = snapshot_items[-1]
            snapshot_html += (
                f'<tr><td style="width:50%;padding:6px 0;vertical-align:top;font-family:{SANS};font-size:11px;color:#8A8F9C;">'
                f'<div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.04em;">{_esc(last_label)}</div>'
                f'<div style="margin-top:3px;font-size:12px;font-weight:700;color:#14213D;">{_esc(last_value)}</div></td>'
                f'<td style="width:50%;padding:6px 0;vertical-align:top;font-family:{SANS};font-size:11px;color:#8A8F9C;"></td></tr>'
            )

        cards.append(f"""
        <table width="100%" cellpadding="0" cellspacing="0" role="presentation"
               style="margin:14px 0;border:1px solid #E7E4DC;border-radius:6px;overflow:hidden;border-collapse:collapse;">
          <tr>
            <td style="padding:10px 14px;background:#14213D;">
              <span style="font-family:{SERIF};font-size:14.5px;color:#ffffff;">{_esc(name)}</span>
              <span style="float:right;font-family:{SANS};font-size:11px;font-weight:700;color:{_reco_color(reco)};
                    background:#ffffff;padding:2px 10px;border-radius:12px;">{_esc(reco)}</span>
            </td>
          </tr>
          <tr>
            <td style="padding:12px 14px;">
              <div style="font-family:{SANS};font-size:11px;font-weight:700;color:#8A8F9C;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px;">News Timeline</div>
              {news_html}
              <div style="font-family:{SANS};font-size:11px;font-weight:700;color:#8A8F9C;text-transform:uppercase;letter-spacing:0.05em;margin:10px 0 4px;">Corporate Actions</div>
              <p style="margin:0;font-family:{SANS};font-size:12px;color:#4A5063;">{_esc(st.get("corporate_actions","-"))}</p>
              <div style="font-family:{SANS};font-size:11px;font-weight:700;color:#8A8F9C;text-transform:uppercase;letter-spacing:0.05em;margin:10px 0 4px;">Snapshot</div>
              <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="margin-top:4px;border-collapse:collapse;">
                <tbody>
                  {snapshot_html}
                </tbody>
              </table>
              <p style="margin:8px 0 0;font-family:{SANS};font-size:12px;color:#4A5063;line-height:1.6;white-space:pre-wrap;word-break:break-word;"><strong>Decision note:</strong> {_esc(st.get("decision_note","-"))}</p>
              <div style="margin-top:10px;padding:8px 10px;background:#F8F7F3;border:1px solid #E7E4DC;border-radius:4px;">
                <div style="font-family:{SANS};font-size:11px;font-weight:700;color:#8A8F9C;text-transform:uppercase;letter-spacing:0.04em;">Decision Signal</div>
                <div style="margin-top:4px;display:inline-block;padding:3px 10px;border-radius:999px;background:{signal_color}1A;color:{signal_color};font-family:{SANS};font-size:12px;font-weight:700;">{_esc(signal_text)}</div>
                <div style="margin-top:6px;font-family:{SANS};font-size:11px;color:#4A5063;">
                  <span style="display:inline-block;padding:2px 8px;border-radius:999px;background:{quality_color}1A;color:{quality_color};font-weight:700;">{_esc(quality_status)}</span>
                </div>
              </div>
              <div style="margin-top:10px;padding:8px 10px;background:#fffdf8;border:1px solid #F2E2BF;border-radius:4px;">
                <div style="font-family:{SANS};font-size:11px;font-weight:700;color:#8A8F9C;text-transform:uppercase;letter-spacing:0.04em;">Action now</div>
                <div style="margin-top:4px;font-family:{SANS};font-size:12px;color:#14213D;line-height:1.5;">{_esc(action_now)}</div>
              </div>
              <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="margin-top:10px;">
                <tr>
                  <td style="width:33%;padding:6px 0;font-family:{SANS};font-size:11px;color:#8A8F9C;">Approx. Weekly Return<br>
                    <span style="font-size:13px;font-weight:700;color:#14213D;">{_esc(ret_str)}</span></td>
                  <td style="width:34%;padding:6px 0;font-family:{SANS};font-size:11px;color:#8A8F9C;">Analyst View<br>
                    <span style="font-size:12px;color:#4A5063;">{_esc(st.get("analyst_view","-"))}</span></td>
                  <td style="width:33%;padding:6px 0;font-family:{SANS};font-size:11px;color:#8A8F9C;">Assessment<br>
                    <span style="font-size:12px;font-weight:700;color:{_sentiment_color(assess)};">{_esc(assess)}</span></td>
                </tr>
              </table>
              <div style="font-family:{SANS};font-size:11px;font-weight:700;color:#8A8F9C;text-transform:uppercase;letter-spacing:0.05em;margin:10px 0 4px;">Outlook</div>
              <p style="margin:0;font-family:{SANS};font-size:12px;color:#4A5063;"><strong>Short-term (1-4 weeks):</strong> {_esc(st.get("short_term_outlook","-"))}</p>
            </td>
          </tr>
        </table>
        """)
    return _section_title("2. Watchlist Stock Analysis (Last 7 Days)") + "".join(cards)


def render_market_news(market_data):
    devs = market_data.get("developments") or []
    rows = "".join(
        f'<tr><td style="padding:7px 10px;font-family:{SANS};font-size:11.5px;color:#8A8F9C;border-top:1px solid #EDEAE2;white-space:nowrap;">{_esc(d.get("date",""))}</td>'
        f'<td style="padding:7px 10px;font-family:{SANS};font-size:11.5px;font-weight:700;color:#14213D;border-top:1px solid #EDEAE2;">{_esc(d.get("topic",""))}</td>'
        f'<td style="padding:7px 10px;font-family:{SANS};font-size:12px;color:#1B2233;border-top:1px solid #EDEAE2;">'
        f'<strong>{_esc(d.get("headline",""))}</strong><br>{_esc(d.get("summary",""))}'
        f'<br><span style="color:#4A5063;"><em>Investor impact:</em> {_esc(d.get("investor_impact",""))}</span></td></tr>'
        for d in devs
    )
    if not rows:
        rows = f'<tr><td colspan="3" style="padding:10px;font-family:{SANS};font-size:12px;color:#8A8F9C;">No market developments could be generated this run.</td></tr>'
    return _section_title("3. Market News (Past 7 Days)") + f"""
    <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="border:1px solid #E7E4DC;border-radius:4px;border-collapse:collapse;">
      <tr style="background:#F4F2ED;">
        <td style="padding:7px 10px;font-family:{SANS};font-size:11px;font-weight:700;color:#8A8F9C;text-transform:uppercase;">Date</td>
        <td style="padding:7px 10px;font-family:{SANS};font-size:11px;font-weight:700;color:#8A8F9C;text-transform:uppercase;">Topic</td>
        <td style="padding:7px 10px;font-family:{SANS};font-size:11px;font-weight:700;color:#8A8F9C;text-transform:uppercase;">Development &amp; Impact</td>
      </tr>
      {rows}
    </table>
    """


def render_sector_table(sectors_data):
    def stars(rating):
        try:
            n = max(0, min(5, round(float(rating))))
        except (TypeError, ValueError):
            n = 0
        return "&#9733;" * n + "&#9734;" * (5 - n)

    rows = "".join(
        f'<tr><td style="padding:7px 10px;font-family:{SANS};font-size:12px;font-weight:700;color:#14213D;border-top:1px solid #EDEAE2;">{_esc(s.get("sector",""))}</td>'
        f'<td style="padding:7px 10px;font-family:{SANS};font-size:11.5px;color:#4A5063;border-top:1px solid #EDEAE2;">{_esc(s.get("weekly_performance",""))}</td>'
        f'<td style="padding:7px 10px;font-family:{SANS};font-size:11.5px;color:#4A5063;border-top:1px solid #EDEAE2;">{_esc(s.get("key_news",""))}</td>'
        f'<td style="padding:7px 10px;font-family:{SANS};font-size:11.5px;color:#4A5063;border-top:1px solid #EDEAE2;">{_esc(s.get("outlook",""))}</td>'
        f'<td style="padding:7px 10px;font-family:{SANS};font-size:14px;color:#B08D57;border-top:1px solid #EDEAE2;white-space:nowrap;">{stars(s.get("rating"))}</td></tr>'
        for s in sectors_data
    )
    if not rows:
        rows = f'<tr><td colspan="5" style="padding:10px;font-family:{SANS};font-size:12px;color:#8A8F9C;">No sector data could be generated this run.</td></tr>'
    return _section_title("4. Sector Performance") + f"""
    <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="border:1px solid #E7E4DC;border-radius:4px;border-collapse:collapse;">
      <tr style="background:#F4F2ED;">
        <td style="padding:7px 10px;font-family:{SANS};font-size:11px;font-weight:700;color:#8A8F9C;text-transform:uppercase;">Sector</td>
        <td style="padding:7px 10px;font-family:{SANS};font-size:11px;font-weight:700;color:#8A8F9C;text-transform:uppercase;">Weekly Performance</td>
        <td style="padding:7px 10px;font-family:{SANS};font-size:11px;font-weight:700;color:#8A8F9C;text-transform:uppercase;">Key News</td>
        <td style="padding:7px 10px;font-family:{SANS};font-size:11px;font-weight:700;color:#8A8F9C;text-transform:uppercase;">Outlook</td>
        <td style="padding:7px 10px;font-family:{SANS};font-size:11px;font-weight:700;color:#8A8F9C;text-transform:uppercase;">Rating</td>
      </tr>
      {rows}
    </table>
    """


def build_email_html(market_data, stocks_data, sectors_data, synthesis_data, sources, used_live_search, today_str):
    sections = (
        render_executive_summary(market_data, synthesis_data)
        + render_stock_cards(stocks_data)
        + render_market_news(market_data)
        + render_sector_table(sectors_data)
    )
    sources_html = _build_sources_html(sources)

    if used_live_search:
        run_note = (
            "This review is generated using live web search across several model calls (see "
            "\"Sources Consulted\" below for what was actually looked at) covering the trailing "
            "7 days. Search results can still be incomplete, out of date by a few hours, or "
            "misread by the model -- verify every price move, corporate-action claim, and "
            "news item against a live quote or exchange filing before acting."
        )
    else:
        run_note = (
            "Live web search was not used for part or all of this run -- prices, dates, "
            "\"recent\" news, and corporate-action claims above may reflect model training data "
            "rather than the actual trailing 7 days. Verify everything against a live quote or "
            "exchange filing before acting."
        )

    live_tag = (
        '<span style="color:#B08D57;">&nbsp;&middot;&nbsp; Live web search used</span>'
        if used_live_search else ""
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Stock Market News Review</title>
<style>
  body {{ margin:0; padding:0; background:#F2F0EC; }}
  table {{ border-collapse:collapse !important; }}
  @media screen and (max-width:600px) {{
    .email-container {{ width:100% !important; max-width:100% !important; border-radius:0 !important; }}
    .email-padding {{ padding-left:16px !important; padding-right:16px !important; }}
  }}
</style>
</head>
<body style="margin:0;padding:0;background:#F2F0EC;font-family:{SERIF};color:#1B2233;">
  <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="background:#F2F0EC;width:100%;">
    <tr>
      <td align="center" style="padding:20px 16px;" class="email-padding">
        <table width="100%" cellpadding="0" cellspacing="0" role="presentation" class="email-container" style="max-width:720px;min-width:280px;background:#ffffff;border:1px solid #DAD5CB;border-radius:4px;overflow:hidden;">
          <tr>
            <td style="background:#14213D;padding:26px 28px 22px;" class="email-padding">
              <div style="font-family:{SANS};font-size:10px;font-weight:700;letter-spacing:0.16em;text-transform:uppercase;color:#B08D57;">Market Research &nbsp;&bull;&nbsp; Weekly Review</div>
              <h1 style="margin:8px 0 0;font-family:{SERIF};font-weight:400;font-size:23px;line-height:1.3;color:#ffffff;letter-spacing:0.01em;">Stock Market News Summary</h1>
              <p style="margin:6px 0 0;font-family:{SANS};font-size:12px;color:#B7BEC9;">Past 7 Days &mdash; {len(WATCHLIST)} Stock Watchlist</p>
            </td>
          </tr>
          <tr>
            <td style="height:3px;line-height:3px;font-size:0;background:linear-gradient(90deg,#B08D57,#D9C393 45%,#B08D57);">&nbsp;</td>
          </tr>
          <tr>
            <td style="padding:16px 28px 4px;" class="email-padding">
              <p style="margin:0;font-family:{SANS};font-size:12px;color:#8A8F9C;">Prepared {today_str} at {datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%I:%M %p IST")}{live_tag}</p>
            </td>
          </tr>
          <tr>
            <td style="padding:14px 28px 18px;" class="email-padding">
              {sections}
              {sources_html}
            </td>
          </tr>
{build_compliance_block_html(report_kind="stock_market", run_note=run_note)}
        </table>
      </td>
    </tr>
  </table>
</body>
</html>
"""


def send_stock_email(html_body, today_str):
    if not all([config.EMAIL_FROM, config.EMAIL_PASSWORD, config.EMAIL_TO]):
        log.error(
            "Email credentials not found. Please set EMAIL_FROM, EMAIL_PASSWORD, "
            "and EMAIL_TO (the same env vars main.py uses)."
        )
        return False

    to_recipients = config.parse_email_list(config.EMAIL_TO)
    cc_recipients = config.parse_email_list(getattr(config, "EMAIL_CC", "") or "")

    if not to_recipients:
        log.error("No valid TO recipients found in EMAIL_TO.")
        return False

    now_ist = datetime.now(ZoneInfo("UTC")).astimezone(ZoneInfo("Asia/Kolkata"))
    time_str = now_ist.strftime("%I:%M %p IST")
    subject = f"Stock Market News Review — {config.get_date_with_suffix(now_ist)} · {time_str}"

    msg = MIMEText(html_body, "html")
    msg["Subject"] = subject
    msg["From"] = config.EMAIL_FROM
    msg["To"] = ", ".join(to_recipients)
    if cc_recipients:
        msg["Cc"] = ", ".join(cc_recipients)

    all_recipients = to_recipients + cc_recipients

    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
            server.starttls()
            server.login(config.EMAIL_FROM, config.EMAIL_PASSWORD)
            server.sendmail(config.EMAIL_FROM, all_recipients, msg.as_string())
        log.info("Stock market news email sent successfully.")
        return True
    except smtplib.SMTPAuthenticationError:
        log.error(
            "SMTP Authentication Error: check EMAIL_FROM/EMAIL_PASSWORD "
            "(use a Gmail App Password, not the account password)."
        )
    except Exception as e:
        log.error(f"Failed to send stock market news email: {e}")
        traceback.print_exc()
    return False


def run():
    today_str, now_ist, lookback_note = _run_context()
    log.info(f"Stock market news review starting for {today_str} (watchlist: {len(WATCHLIST)} stocks).")

    market_data, market_sources, market_live = run_market_stage(today_str, lookback_note)
    stocks_data, stock_sources, stocks_live = run_stock_stage(today_str, lookback_note)
    sectors_data, sector_sources, sectors_live = run_sector_stage(today_str, lookback_note)

    sources = []
    for src_list in (market_sources, stock_sources, sector_sources):
        for s in src_list:
            if s not in sources:
                sources.append(s)
    used_live_search = market_live or stocks_live or sectors_live

    synthesis_data = run_synthesis_stage(market_data, stocks_data, sectors_data, today_str)

    email_html = build_email_html(
        market_data, stocks_data, sectors_data, synthesis_data,
        sources, used_live_search, today_str,
    )

    if os.getenv("DRY_RUN", "false").lower() == "true":
        with open("stock_market_report.html", "w") as f:
            f.write(email_html)
        stockpredictor.log.info("DRY_RUN enabled -- wrote stock_market_report.html instead of emailing.")
        return

    send_stock_email(email_html, today_str)


if __name__ == "__main__":
    run()