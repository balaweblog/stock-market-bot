"""
mutual_fund_advisor.py

Standalone companion to main.py / swing_trade_advisor.py. Runs a
"last-30-days" mutual fund portfolio review -- market/macro context,
fund-wise news, and sector performance -- against whichever free LLM
backend main.py already knows how to set up, then emails the result to
the same recipients configured for the other reports (EMAIL_TO / EMAIL_CC
in config.py / the workflow yaml's env vars).

This deliberately reuses main.py's LLM-selection and email-credential
plumbing, AND swing_trade_advisor.py's live-search cascade
(generate_analysis) and JSON/source helpers, instead of duplicating any
of it -- so all three scripts stay in sync with whatever provider/quota
situation is configured. See swing_trade_advisor.py's own docstring for
the full list of live-search fallback tiers (Groq compound ->
compound-mini -> Tavily+Groq -> Gemini grounding -> Mistral web search ->
non-live fallback, gated by REQUIRE_LIVE_DATA).

WHY A MULTI-STAGE PIPELINE (not one giant prompt):
The uploaded prompt this script automates asks for several sections
covering 7 funds, 17 macro topics, and 13 sectors -- all "last 30 days
only, cite dates, no fabrication." Asking one model call to search,
verify, and structure all of that at once reliably truncates or
hallucinates. Instead:
  Stage 1 -- Market & macro (one live-search call, ~17 topics).
  Stage 2 -- Fund-wise news, in rotating batches of MF_FUNDS_PER_BATCH
             funds per live-search call (so each call can search a
             handful of funds deeply rather than seven thinly).
  Stage 3 -- Sector performance, in rotating batches of
             MF_SECTORS_PER_BATCH sectors per live-search call.
  Stage 4 -- Synthesis: a single NON-live call that reasons over the
             already-gathered, already-cited output of Stages 1-3 to
             produce the top-developments list for the executive
             summary. This stage doesn't fetch new facts, so it isn't
             gated by REQUIRE_LIVE_DATA.

CAVEAT: this is not a substitute for a SEBI-registered adviser or your
own factsheet review. Web search results can be stale, incomplete, or
misread by the model. Treat every return figure, portfolio-change claim,
and "recent" news item as a starting point to verify against the AMC's
own factsheet/press release -- not investment advice. See compliance.py
for the full disclosure block attached to every email.
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

import main  # reuses LLM init, email config/credentials, and helpers
from compliance import build_compliance_block_html
from swing_trade_advisor import (
    _env_int,
    generate_analysis,
    _strip_code_fences,
    _build_sources_html,
    _generate_local,
    _require_live_or_abort,
)

SANS = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
SERIF = "Georgia,'Times New Roman',serif"

# -----------------------------
# Portfolio & sector universe
# -----------------------------
DEFAULT_PORTFOLIO = [
    "Mirae Asset Large & Midcap Fund - Direct Growth",
    "Parag Parikh Flexi Cap Fund - Direct Growth",
    "SBI Small Cap Fund - Direct Growth",
    "DSP Multi Asset Fund - Direct Growth",
    "ICICI Prudential US Bluechip Fund",
    "ICICI Prudential Manufacturing Fund",
    "DSP Natural Resources & New Energy Fund",
]

MARKET_TOPICS = [
    "Nifty 50", "Sensex", "Midcap Index", "Smallcap Index", "RBI",
    "Inflation", "Interest Rates", "GDP", "Rupee vs Dollar",
    "FII Activity", "DII Activity", "US Markets", "Federal Reserve",
    "China", "Crude Oil", "Gold", "Dollar Index",
]

SECTORS_MF = [
    "Banking", "Financial Services", "IT", "Pharma", "Manufacturing",
    "Capital Goods", "Infrastructure", "Defence", "Energy",
    "Commodities", "Metals", "Renewable Energy", "US Technology",
]

FUNDS_PER_BATCH = _env_int("MF_FUNDS_PER_BATCH", 3)
SECTORS_PER_BATCH = _env_int("MF_SECTORS_PER_BATCH", 5)

SOURCE_QUALITY_NOTE = (
    "Source quality: prioritize official AMC factsheets/press releases, "
    "SEBI, RBI, NSE/BSE circulars, AMFI data, and reputable financial "
    "media (Value Research, Morningstar India, ET Markets, Mint, "
    "Moneycontrol, Business Standard). Do NOT rely on or cite social "
    "media posts (X/Twitter, Telegram, Instagram, Facebook) -- they are "
    "unverifiable and frequently wrong."
)

NO_FABRICATION_NOTE = (
    "Do not fabricate a date, return figure, NAV, portfolio-holding "
    "change, or AUM number. If you cannot verify a specific figure from "
    "a real source published in the window below, say so explicitly "
    "(e.g. \"not independently verifiable this run\") instead of "
    "inventing one."
)


def _load_portfolio():
    raw = os.getenv("MF_PORTFOLIO_JSON")
    if raw:
        try:
            data = json.loads(raw)
            if isinstance(data, list) and data and all(isinstance(x, str) and x.strip() for x in data):
                return [x.strip() for x in data]
            main.log.warning(
                "MF_PORTFOLIO_JSON parsed but is not a non-empty list of "
                "strings -- using the default portfolio instead."
            )
        except json.JSONDecodeError as e:
            main.log.warning(f"MF_PORTFOLIO_JSON is not valid JSON ({e}) -- using the default portfolio.")
    return DEFAULT_PORTFOLIO


PORTFOLIO = _load_portfolio()


def _chunks(items, size):
    size = max(1, size)
    return [items[i:i + size] for i in range(0, len(items), size)]


# -----------------------------
# Schedule / lookback window
# -----------------------------
# Unlike swing_trade_advisor.py's weekly cadence, this script is intended
# to run MONTHLY (the source prompt is explicitly framed as a "past 30
# days only" review). Suggested cron for the workflow yaml (03:00 UTC on
# the 1st = 08:30 IST, well before market open):
#   cron: '0 3 1 * *'
# It can also be triggered manually (workflow_dispatch) at any time --
# the 30-day window below is always relative to "today", not to a fixed
# calendar month, so an on-demand run mid-month still makes sense.
def _run_context():
    now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
    today_str = now_ist.strftime("%d %B %Y")
    window_start_str = (now_ist - timedelta(days=30)).strftime("%d %B %Y")
    lookback_note = (
        f"STRICTLY covering the window from {window_start_str} to {today_str} "
        "(the past 30 days) only -- exclude anything older, and exclude "
        "anything you cannot confidently date within this window."
    )
    return today_str, now_ist, lookback_note


# -----------------------------
# Stage 1 -- Market & macro
# -----------------------------
def build_market_prompt(today_str, lookback_note):
    topic_list = ", ".join(MARKET_TOPICS)
    return f"""Act as a CFA charterholder and mutual fund research analyst covering the Indian market. Using the most current data as of {today_str}, {lookback_note}

Find and summarize the most important developments across these topics: {topic_list}.

{SOURCE_QUALITY_NOTE}

{NO_FABRICATION_NOTE}

For each topic, search for what actually happened in the window above (index levels/moves, RBI policy actions, inflation prints, GDP releases, FII/DII net flows, Fed decisions, crude/gold/DXY moves, etc.) -- do not pad with generic commentary that isn't tied to a dated event.

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
      "sip_investor_impact": "How this specifically affects a long-term (5-20yr) SIP investor holding diversified Indian equity, US equity, manufacturing, and multi-asset/commodity mutual funds -- not just a restatement of the headline",
      "confidence": "High | Medium | Low"
    }}
  ]
}}
List 15-25 genuine, dated developments across the topics above, in roughly chronological order. It is fine for some topics to have fewer items than others if less happened.
"""


# -----------------------------
# Stage 2 -- Fund-wise news (batched)
# -----------------------------
def build_fund_prompt(funds_batch, today_str, lookback_note):
    listing = "\n".join(f"- {f}" for f in funds_batch)
    return f"""Act as a SEBI-aware mutual fund research analyst. Using the most current data as of {today_str}, {lookback_note}

For EACH of the following Indian mutual funds, research recent news, portfolio/AUM changes, and performance:
{listing}

{SOURCE_QUALITY_NOTE}

{NO_FABRICATION_NOTE}

For each fund, look for: AMC announcements, fund manager changes, portfolio additions/exits or sector-weight shifts disclosed in the latest factsheet, AUM changes, category-average/benchmark comparison, and any news specifically naming this fund or its major holdings.

OUTPUT FORMAT -- respond with ONLY raw JSON matching this schema, nothing else (no markdown, no code fences, no commentary before or after):

{{
  "funds": [
    {{
      "fund_name": "Exact fund name as given above",
      "news_timeline": [
        {{
          "date": "DD Month YYYY",
          "headline": "Short headline",
          "summary": "2-3 sentence summary",
          "why_it_matters": "One sentence",
          "impact_on_fund": "Positive | Neutral | Negative",
          "confidence": "High | Medium | Low"
        }}
      ],
      "portfolio_changes": "New additions, exits, increased/reduced holdings, sector-allocation shifts, cash allocation, AUM change, or fund-manager updates this month -- or 'No material disclosed changes found this window' if none verifiable",
      "monthly_return_pct": "Approximate % return this window as a plain number string (e.g. '2.4'), or null if not verifiable",
      "benchmark_comparison": "How the fund did vs its benchmark/category this window",
      "assessment": "Positive | Neutral | Negative",
      "short_term_outlook": "3-6 month outlook, 1-2 sentences",
      "long_term_outlook": "5-20 year outlook, 1-2 sentences",
      "recommendation": "Strong Buy | Buy | Continue SIP | Hold | Review | Reduce | Exit"
    }}
  ]
}}
Include one object per fund listed above, even if you found little news for it (in that case say so plainly rather than inventing content).
"""


# -----------------------------
# Stage 3 -- Sector performance (batched)
# -----------------------------
def build_sector_prompt(sectors_batch, today_str, lookback_note):
    listing = ", ".join(sectors_batch)
    return f"""Act as an equity sector analyst covering the Indian and relevant global (US tech) markets. Using the most current data as of {today_str}, {lookback_note}

For EACH of these sectors, summarize the past month's performance and outlook: {listing}.

{SOURCE_QUALITY_NOTE}

{NO_FABRICATION_NOTE}

OUTPUT FORMAT -- respond with ONLY raw JSON matching this schema, nothing else (no markdown, no code fences, no commentary before or after):

{{
  "sectors": [
    {{
      "sector": "Exact sector name as given above",
      "monthly_performance": "1-2 sentences on how the sector index/theme moved this window",
      "key_news": "1-2 sentences on the most important dated news item(s)",
      "outlook": "1-2 sentences, next 3-6 months",
      "rating": "Integer 1-5 (5 = most favorable outlook)"
    }}
  ]
}}
Include one object per sector listed above.
"""


# -----------------------------
# Stage 4 -- Synthesis (non-live)
# -----------------------------
def build_synthesis_prompt(market_data, funds_data, sectors_data, portfolio, today_str):
    context = json.dumps(
        {"market": market_data, "funds": funds_data, "sectors": sectors_data},
        ensure_ascii=False,
    )
    fund_list = ", ".join(portfolio)
    return f"""You are a CFA charterholder and SEBI-aware mutual fund research analyst. You have ALREADY researched the material below (already dated/sourced -- do not re-search or add new facts, just reason over what's here):

{context}

The investor's portfolio is exactly these {len(portfolio)} funds: {fund_list}.

Using ONLY the material above, produce a synthesis for a long-term (5-20 year) SIP investor. Respond with ONLY raw JSON matching this schema, nothing else (no markdown, no code fences, no commentary before or after):

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
        main.log.error("No LLM backend produced Stage 1 (market/macro) output. Aborting without sending an email.")
        sys.exit(1)
    _require_live_or_abort(live, "Stage 1 (market/macro)")
    data = _parse_json_object(text)
    if not isinstance(data, dict):
        main.log.warning("Stage 1 output could not be parsed as JSON -- proceeding with an empty market section.")
        data = {}
    data.setdefault("developments", [])
    data.setdefault("market_sentiment", "Neutral")
    return data, sources, live


def run_fund_stage(today_str, lookback_note):
    all_funds, sources, used_live = [], [], False
    for batch in _chunks(PORTFOLIO, FUNDS_PER_BATCH):
        main.log.info(f"Stage 2 -- fund batch: {', '.join(batch)}")
        prompt = build_fund_prompt(batch, today_str, lookback_note)
        text, s, live = generate_analysis(prompt, max_tokens=3200)
        if not text:
            main.log.error(f"No LLM output for fund batch ({', '.join(batch)}) -- skipping this batch.")
            continue
        _require_live_or_abort(live, f"Stage 2 (fund batch: {', '.join(batch)})")
        data = _parse_json_object(text)
        funds = data.get("funds") if isinstance(data, dict) else None
        if isinstance(funds, list):
            all_funds.extend(funds)
        else:
            main.log.warning(f"Could not parse fund JSON for batch: {', '.join(batch)}")
        for src in s:
            if src not in sources:
                sources.append(src)
        used_live = used_live or live
    return all_funds, sources, used_live


def run_sector_stage(today_str, lookback_note):
    all_sectors, sources, used_live = [], [], False
    for batch in _chunks(SECTORS_MF, SECTORS_PER_BATCH):
        main.log.info(f"Stage 3 -- sector batch: {', '.join(batch)}")
        prompt = build_sector_prompt(batch, today_str, lookback_note)
        text, s, live = generate_analysis(prompt, max_tokens=2000)
        if not text:
            main.log.error(f"No LLM output for sector batch ({', '.join(batch)}) -- skipping this batch.")
            continue
        _require_live_or_abort(live, f"Stage 3 (sector batch: {', '.join(batch)})")
        data = _parse_json_object(text)
        sectors = data.get("sectors") if isinstance(data, dict) else None
        if isinstance(sectors, list):
            all_sectors.extend(sectors)
        else:
            main.log.warning(f"Could not parse sector JSON for batch: {', '.join(batch)}")
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
    backend = main.init_llm_generator()
    main.log.info(f"Stage 4 (synthesis) using LLM backend: {backend}")

    if backend == "groq" and getattr(main, "groq_client", None) is not None:
        try:
            response = main.groq_client.chat.completions.create(
                model=os.getenv("MF_SYNTHESIS_MODEL", "llama-3.3-70b-versatile"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            main.log.error(f"Groq synthesis call failed: {e}")

    have_gemini = getattr(main, "gemini_client", None) is not None or (
        os.getenv("GOOGLE_API_KEY") and getattr(main, "genai", None) is not None
    )
    if have_gemini:
        try:
            if getattr(main, "gemini_client", None) is None:
                main.gemini_client = main.genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
            response = main.gemini_client.models.generate_content(
                model="gemini-flash-latest",
                contents=prompt,
            )
            return response.text.strip()
        except Exception as e:
            main.log.error(f"Gemini synthesis call failed: {e}")

    local_backend = main.init_llm_generator(force_local=True)
    if local_backend == "local" and main.llm_pipeline is not None:
        text = _generate_local(prompt)
        if text:
            return text

    return None


def run_synthesis_stage(market_data, funds_data, sectors_data, today_str):
    prompt = build_synthesis_prompt(market_data, funds_data, sectors_data, PORTFOLIO, today_str)
    text = _plain_generate(prompt)
    if not text:
        main.log.error("No LLM backend produced Stage 4 (synthesis) output. Aborting without sending an email.")
        sys.exit(1)
    data = _parse_json_object(text)
    if not isinstance(data, dict):
        main.log.warning("Stage 4 output could not be parsed as JSON -- proceeding with an empty synthesis section.")
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
    if reco in ("strong buy", "buy", "increase sip"):
        return "#1E7A46"
    if reco in ("continue sip", "hold", "keep", "monitor", "no action"):
        return "#8A6D1D"
    if reco in ("review", "within 30 days"):
        return "#C8792A"
    if reco in ("reduce", "exit", "replace", "immediate"):
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
      Overall Market Sentiment (30 Days): {_esc(sentiment)}
    </div>
    <p style="margin:6px 0 12px;font-family:{SANS};font-size:13px;color:#4A5063;">{_esc(reason)}</p>
    <div style="font-family:{SANS};font-size:12.5px;font-weight:700;color:#14213D;margin-bottom:4px;">Top developments this month</div>
    <ol style="margin:0;padding-left:18px;font-family:{SANS};font-size:12.5px;line-height:1.6;color:#1B2233;">{items}</ol>
    """


def render_fund_cards(funds_data):
    if not funds_data:
        return _section_title("2. Fund-wise Analysis") + (
            f'<p style="font-family:{SANS};font-size:12.5px;color:#B0473F;">No fund data could be generated this run.</p>'
        )
    cards = []
    for f in funds_data:
        name = f.get("fund_name", "Unknown Fund")
        news = f.get("news_timeline") or []
        news_html = "".join(
            f'<div style="margin:0 0 10px;padding:8px 10px;background:#F8F7F3;border-left:3px solid #B08D57;border-radius:2px;">'
            f'<div style="font-family:{SANS};font-size:11px;color:#8A8F9C;">{_esc(n.get("date",""))} &middot; Confidence: {_esc(n.get("confidence","-"))}</div>'
            f'<div style="font-family:{SANS};font-size:12.5px;font-weight:700;color:#14213D;margin:2px 0;">{_esc(n.get("headline",""))}</div>'
            f'<div style="font-family:{SANS};font-size:12px;color:#4A5063;">{_esc(n.get("summary",""))}</div>'
            f'<div style="font-family:{SANS};font-size:11.5px;color:#4A5063;margin-top:3px;"><em>Why it matters:</em> {_esc(n.get("why_it_matters",""))} &middot; Impact: {_esc(n.get("impact_on_fund","-"))}</div>'
            f'</div>'
            for n in news
        ) or f'<p style="font-family:{SANS};font-size:12px;color:#8A8F9C;">No dated news items found this window.</p>'

        reco = f.get("recommendation", "-")
        assess = f.get("assessment", "-")
        ret = f.get("monthly_return_pct")
        ret_str = f"{ret}%" if ret not in (None, "", "null") else "n/a"

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
              <div style="font-family:{SANS};font-size:11px;font-weight:700;color:#8A8F9C;text-transform:uppercase;letter-spacing:0.05em;margin:10px 0 4px;">Portfolio Changes</div>
              <p style="margin:0;font-family:{SANS};font-size:12px;color:#4A5063;">{_esc(f.get("portfolio_changes","-"))}</p>
              <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="margin-top:10px;">
                <tr>
                  <td style="width:33%;padding:6px 0;font-family:{SANS};font-size:11px;color:#8A8F9C;">Approx. Monthly Return<br>
                    <span style="font-size:13px;font-weight:700;color:#14213D;">{_esc(ret_str)}</span></td>
                  <td style="width:34%;padding:6px 0;font-family:{SANS};font-size:11px;color:#8A8F9C;">Benchmark/Peer Comparison<br>
                    <span style="font-size:12px;color:#4A5063;">{_esc(f.get("benchmark_comparison","-"))}</span></td>
                  <td style="width:33%;padding:6px 0;font-family:{SANS};font-size:11px;color:#8A8F9C;">Assessment<br>
                    <span style="font-size:12px;font-weight:700;color:{_sentiment_color(assess)};">{_esc(assess)}</span></td>
                </tr>
              </table>
              <div style="font-family:{SANS};font-size:11px;font-weight:700;color:#8A8F9C;text-transform:uppercase;letter-spacing:0.05em;margin:10px 0 4px;">Outlook</div>
              <p style="margin:0;font-family:{SANS};font-size:12px;color:#4A5063;"><strong>Short-term (3-6m):</strong> {_esc(f.get("short_term_outlook","-"))}</p>
              <p style="margin:4px 0 0;font-family:{SANS};font-size:12px;color:#4A5063;"><strong>Long-term (5-20y):</strong> {_esc(f.get("long_term_outlook","-"))}</p>
            </td>
          </tr>
        </table>
        """)
    return _section_title("2. Fund-wise Analysis (Last 30 Days)") + "".join(cards)


def render_market_news(market_data):
    devs = market_data.get("developments") or []
    rows = "".join(
        f'<tr><td style="padding:7px 10px;font-family:{SANS};font-size:11.5px;color:#8A8F9C;border-top:1px solid #EDEAE2;white-space:nowrap;">{_esc(d.get("date",""))}</td>'
        f'<td style="padding:7px 10px;font-family:{SANS};font-size:11.5px;font-weight:700;color:#14213D;border-top:1px solid #EDEAE2;">{_esc(d.get("topic",""))}</td>'
        f'<td style="padding:7px 10px;font-family:{SANS};font-size:12px;color:#1B2233;border-top:1px solid #EDEAE2;">'
        f'<strong>{_esc(d.get("headline",""))}</strong><br>{_esc(d.get("summary",""))}'
        f'<br><span style="color:#4A5063;"><em>SIP impact:</em> {_esc(d.get("sip_investor_impact",""))}</span></td></tr>'
        for d in devs
    )
    if not rows:
        rows = f'<tr><td colspan="3" style="padding:10px;font-family:{SANS};font-size:12px;color:#8A8F9C;">No market developments could be generated this run.</td></tr>'
    return _section_title("3. Market News (Past 30 Days)") + f"""
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
        f'<td style="padding:7px 10px;font-family:{SANS};font-size:11.5px;color:#4A5063;border-top:1px solid #EDEAE2;">{_esc(s.get("monthly_performance",""))}</td>'
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
        <td style="padding:7px 10px;font-family:{SANS};font-size:11px;font-weight:700;color:#8A8F9C;text-transform:uppercase;">Monthly Performance</td>
        <td style="padding:7px 10px;font-family:{SANS};font-size:11px;font-weight:700;color:#8A8F9C;text-transform:uppercase;">Key News</td>
        <td style="padding:7px 10px;font-family:{SANS};font-size:11px;font-weight:700;color:#8A8F9C;text-transform:uppercase;">Outlook</td>
        <td style="padding:7px 10px;font-family:{SANS};font-size:11px;font-weight:700;color:#8A8F9C;text-transform:uppercase;">Rating</td>
      </tr>
      {rows}
    </table>
    """


def build_email_html(market_data, funds_data, sectors_data, synthesis_data, sources, used_live_search, today_str):
    sections = (
        render_executive_summary(market_data, synthesis_data)
        + render_fund_cards(funds_data)
        + render_market_news(market_data)
        + render_sector_table(sectors_data)
    )
    sources_html = _build_sources_html(sources)

    if used_live_search:
        run_note = (
            "This review is generated using live web search across several model calls (see "
            "\"Sources Consulted\" below for what was actually looked at) covering the trailing "
            "30 days. Search results can still be incomplete, out of date by a few hours, or "
            "misread by the model -- verify every return figure, portfolio-change claim, and "
            "news item against the AMC's own factsheet or a live source before acting."
        )
    else:
        run_note = (
            "Live web search was not used for part or all of this run -- prices, dates, "
            "\"recent\" news, and portfolio-change claims above may reflect model training data "
            "rather than the actual trailing 30 days. Verify everything against the AMC's own "
            "factsheet or a live source before acting."
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
<title>Mutual Fund Portfolio Review</title>
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
              <div style="font-family:{SANS};font-size:10px;font-weight:700;letter-spacing:0.16em;text-transform:uppercase;color:#B08D57;">Portfolio Research &nbsp;&bull;&nbsp; Monthly Review</div>
              <h1 style="margin:8px 0 0;font-family:{SERIF};font-weight:400;font-size:23px;line-height:1.3;color:#ffffff;letter-spacing:0.01em;">Mutual Fund Portfolio Market Summary</h1>
              <p style="margin:6px 0 0;font-family:{SANS};font-size:12px;color:#B7BEC9;">Past 30 Days &mdash; {len(PORTFOLIO)} Fund Portfolio</p>
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
{build_compliance_block_html(report_kind="mutual_fund", run_note=run_note)}
        </table>
      </td>
    </tr>
  </table>
</body>
</html>
"""


def send_portfolio_email(html_body, today_str):
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
    subject = f"Mutual Fund Portfolio Review — {main.get_date_with_suffix(now_ist)} · {time_str}"

    msg = MIMEText(html_body, "html")
    msg["Subject"] = subject
    msg["From"] = main.EMAIL_FROM
    msg["To"] = ", ".join(to_recipients)
    if cc_recipients:
        msg["Cc"] = ", ".join(cc_recipients)

    all_recipients = to_recipients + cc_recipients

    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
            server.starttls()
            server.login(main.EMAIL_FROM, main.EMAIL_PASSWORD)
            server.sendmail(main.EMAIL_FROM, all_recipients, msg.as_string())
        main.log.info("Mutual fund portfolio email sent successfully.")
        return True
    except smtplib.SMTPAuthenticationError:
        main.log.error(
            "SMTP Authentication Error: check EMAIL_FROM/EMAIL_PASSWORD "
            "(use a Gmail App Password, not the account password)."
        )
    except Exception as e:
        main.log.error(f"Failed to send mutual fund portfolio email: {e}")
        traceback.print_exc()
    return False


def run():
    today_str, now_ist, lookback_note = _run_context()
    main.log.info(f"Mutual fund portfolio review starting for {today_str} (portfolio: {len(PORTFOLIO)} funds).")

    market_data, market_sources, market_live = run_market_stage(today_str, lookback_note)
    funds_data, fund_sources, funds_live = run_fund_stage(today_str, lookback_note)
    sectors_data, sector_sources, sectors_live = run_sector_stage(today_str, lookback_note)

    sources = []
    for src_list in (market_sources, fund_sources, sector_sources):
        for s in src_list:
            if s not in sources:
                sources.append(s)
    used_live_search = market_live or funds_live or sectors_live

    synthesis_data = run_synthesis_stage(market_data, funds_data, sectors_data, today_str)

    email_html = build_email_html(
        market_data, funds_data, sectors_data, synthesis_data,
        sources, used_live_search, today_str,
    )

    if os.getenv("DRY_RUN", "false").lower() == "true":
        with open("mutual_fund_report.html", "w") as f:
            f.write(email_html)
        main.log.info("DRY_RUN enabled -- wrote mutual_fund_report.html instead of emailing.")
        return

    send_portfolio_email(email_html, today_str)


if __name__ == "__main__":
    run()