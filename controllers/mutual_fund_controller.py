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
import ssl
import traceback
import urllib.request
import urllib.parse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import smtplib
from email.mime.text import MIMEText

from utils import config
from utils.logger import log
from utils.prompt_loader import load_prompt
from llm import llm_backend
from utils.compliance import build_compliance_block_html
from controllers.swing_controller import (
    _env_int,
    generate_analysis,
    _strip_code_fences,
    _build_sources_html,
    generate_synthesis,
    _require_live_or_abort,
)

SANS = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
SERIF = "Georgia,'Times New Roman',serif"

# -----------------------------
# Portfolio & sector universe
# -----------------------------
# Portfolio now lives in constants.py instead of the MF_PORTFOLIO_JSON
# environment variable.
from utils.constants import MF_PORTFOLIO

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

# -----------------------------
# AMFI / mfapi.in data layer
# -----------------------------
# mfapi.in is a free, community-maintained wrapper over AMFI's official NAV
# data. It provides daily NAV, scheme metadata, and 1Y/3Y/5Y returns via a
# simple REST API -- no authentication required. We use it to pre-populate
# the deterministic fields (NAV, AUM, returns, category, expense ratio) so
# the LLM stage only needs to find recent news and analysis, not guess facts
# that are publicly available from AMFI.

_AMFI_SEARCH_URL = "https://api.mfapi.in/mf/search?q="
_AMFI_NAV_URL    = "https://api.mfapi.in/mf/"
_AMFI_TIMEOUT_S  = 8

# Map of known AMFI scheme codes for the funds in MF_PORTFOLIO.
# Seed known codes here for maximum speed and reliability; the search fallback
# handles everything else.
_KNOWN_SCHEME_CODES = {
    "Mirae Asset Large & Midcap Fund - Direct Growth":       "118834",
    "Parag Parikh Flexi Cap Fund - Direct Growth":           "122639",
    "SBI Small Cap Fund - Direct Growth":                    "125497",
    "DSP Multi Asset Fund - Direct Growth":                  "152056",
    "ICICI Prudential Manufacturing Fund":                   "145075",
    "DSP Natural Resources & New Energy Fund":               "119775",
    "Nippon India Gold Savings Fund":                        "118663",
}

_SSL_CONTEXT = (ssl._create_unverified_context() 
                if os.getenv("ALLOW_UNVERIFIED_SSL", "false").lower() == "true" 
                else ssl.create_default_context())


def _amfi_get(url):
    """GET url, return parsed JSON or None on any error."""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 StockBot/1.0"},
        )
        kwargs = {"timeout": _AMFI_TIMEOUT_S}
        if _SSL_CONTEXT is not None:
            kwargs["context"] = _SSL_CONTEXT
        with urllib.request.urlopen(req, **kwargs) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        log.warning(f"AMFI fetch failed ({url!r}): {exc}")
        return None


def _amfi_search_scheme_code(fund_name):
    """Search mfapi.in for a scheme code by fund name. Returns code str or None."""
    # Strip common suffixes to improve search hit rate
    query = re.sub(r"\s*-\s*Direct\s*Growth\s*$", "", fund_name, flags=re.IGNORECASE)
    query = re.sub(r"\s*Fund\s*$", "", query, flags=re.IGNORECASE).strip()
    encoded = urllib.parse.quote(query)
    results = _amfi_get(f"{_AMFI_SEARCH_URL}{encoded}")
    if not isinstance(results, list) or not results:
        return None
    # Prefer direct-plan results when available
    for r in results:
        if "direct" in str(r.get("schemeName", "")).lower() and "growth" in str(r.get("schemeName", "")).lower():
            return str(r.get("schemeCode", ""))
    return str(results[0].get("schemeCode", "")) if results else None


def _annualised_return(nav_series, years):
    """
    Compute CAGR from a list of NAV dicts [{"date": "DD-MM-YYYY", "nav": "123.45"}, ...].
    The series is newest-first (mfapi.in order).
    Returns a rounded float or None.
    """
    if not nav_series or len(nav_series) < 2:
        return None
    try:
        latest_nav = float(nav_series[0]["nav"])
        target_days = int(years * 365)
        # Find the NAV closest to `years` ago
        for entry in nav_series:
            try:
                entry_date = datetime.strptime(entry["date"], "%d-%m-%Y").date()
            except ValueError:
                continue
            latest_date = datetime.strptime(nav_series[0]["date"], "%d-%m-%Y").date()
            delta = (latest_date - entry_date).days
            if delta >= target_days - 10:   # within 10 days of the target
                past_nav = float(entry["nav"])
                if past_nav <= 0:
                    return None
                cagr = (latest_nav / past_nav) ** (1 / years) - 1
                return round(cagr * 100, 2)
    except Exception as exc:
        log.warning(f"Error computing annualised return: {exc}")
    return None


def _fetch_amfi_fund_snapshot(fund_name):
    """
    Returns a dict with deterministically fetched fields for one fund, or {}
    if AMFI data is unavailable. Fields match the LLM JSON schema keys.
    """
    code = _KNOWN_SCHEME_CODES.get(fund_name) or _amfi_search_scheme_code(fund_name)
    if not code:
        log.debug(f"AMFI: no scheme code found for {fund_name!r}")
        return {}

    data = _amfi_get(f"{_AMFI_NAV_URL}{code}")
    if not isinstance(data, dict):
        return {}

    meta      = data.get("meta", {})
    nav_list  = data.get("data", [])   # newest-first
    latest    = nav_list[0] if nav_list else {}

    nav_str = latest.get("nav")
    try:
        nav_f = round(float(nav_str), 2) if nav_str else None
    except (ValueError, TypeError):
        nav_f = None

    ret_1y = _annualised_return(nav_list, 1)
    ret_3y = _annualised_return(nav_list, 3)
    ret_5y = _annualised_return(nav_list, 5)

    # Monthly return: compare today's NAV vs ~30 days ago
    ret_1m = None
    if nav_list and len(nav_list) > 20:
        try:
            nav_now  = float(nav_list[0]["nav"])
            nav_30d  = float(nav_list[min(22, len(nav_list)-1)]["nav"])  # ~22 trading days
            if nav_30d > 0:
                ret_1m = round((nav_now / nav_30d - 1) * 100, 2)
        except Exception as exc:
            log.warning(f"Error computing monthly return: {exc}")

    scheme_type = str(meta.get("scheme_type", "")).lower()
    scheme_cat  = meta.get("scheme_category", "") or ""
    fund_house  = meta.get("fund_house", "") or ""

    # Derive a readable category label from the scheme category string
    cat_label = scheme_cat.strip()
    if not cat_label:
        cat_label = "Not disclosed"

    return {
        "_amfi_code":          code,
        "_amfi_fund_house":    fund_house,
        "_amfi_scheme_type":   scheme_type,
        "_amfi_nav_date":      latest.get("date", ""),
        "nav_latest":          str(nav_f) if nav_f else None,
        "fund_category":       cat_label if cat_label != "Not disclosed" else None,
        "one_year_return_pct": str(ret_1y) if ret_1y is not None else None,
        "three_year_return_pct": str(ret_3y) if ret_3y is not None else None,
        "five_year_return_pct": str(ret_5y) if ret_5y is not None else None,
        "monthly_return_pct":  str(ret_1m) if ret_1m is not None else None,
    }


_amfi_snapshots = {}   # fund_name -> snapshot dict, populated once per run


def prefetch_amfi_snapshots():
    """Fetch AMFI data for all portfolio funds before the LLM stage runs."""
    for fund_name in PORTFOLIO:
        log.info(f"AMFI prefetch: {fund_name}")
        snap = _fetch_amfi_fund_snapshot(fund_name)
        _amfi_snapshots[fund_name] = snap
        if snap:
            nav_date = snap.get("_amfi_nav_date", "?")
            log.info(
                f"  → NAV={snap.get('nav_latest')} ({nav_date}), "
                f"1Y={snap.get('one_year_return_pct')}%, "
                f"3Y={snap.get('three_year_return_pct')}%"
            )
        else:
            log.warning(f"  → No AMFI data found")


def _enrich_funds_with_amfi(funds_data):
    """
    Merge the AMFI-sourced deterministic fields into each fund dict returned
    by the LLM. LLM fields take precedence only when they are non-empty and
    non-'Not disclosed' -- otherwise the AMFI fact overwrites the LLM's guess.
    This ensures fields like NAV, 1Y/3Y returns, and category are always
    populated when AMFI data was available, even if the LLM left them blank.
    """
    _EMPTY_VALUES = (None, "", "null", "None", "Not disclosed", "not disclosed", "-", "n/a")

    for f in funds_data:
        name = f.get("fund_name", "")
        snap = _amfi_snapshots.get(name, {})
        if not snap:
            continue
        # Fields to overwrite when LLM returned nothing useful
        for key in ("nav_latest", "fund_category", "one_year_return_pct",
                    "three_year_return_pct", "monthly_return_pct"):
            amfi_val = snap.get(key)
            if amfi_val is None:
                continue
            
            # Authoritative keys: always use AMFI data when available
            if key in ("nav_latest", "one_year_return_pct", "three_year_return_pct", "monthly_return_pct"):
                f[key] = amfi_val
            elif str(f.get(key, "")).strip() in _EMPTY_VALUES:
                f[key] = amfi_val
        # Always attach source metadata for the Data Inputs section
        f["_amfi_code"]     = snap.get("_amfi_code")
        f["_amfi_nav_date"] = snap.get("_amfi_nav_date")
        f["_amfi_fund_house"] = snap.get("_amfi_fund_house")
        if snap.get("five_year_return_pct"):
            f["five_year_return_pct"] = snap["five_year_return_pct"]
    return funds_data


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


PORTFOLIO = MF_PORTFOLIO


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
    return load_prompt(
        "mutual_fund/market",
        today_str=today_str,
        lookback_note=lookback_note,
        topic_list=topic_list,
        SOURCE_QUALITY_NOTE=SOURCE_QUALITY_NOTE,
        NO_FABRICATION_NOTE=NO_FABRICATION_NOTE,
    )


# -----------------------------
# Stage 2 -- Fund-wise news (batched)
# -----------------------------
def build_fund_prompt(funds_batch, today_str, lookback_note):
    # Build per-fund context blocks with any AMFI-fetched facts injected.
    # This means the LLM doesn't need to search for NAV / returns -- it
    # already has them and can focus its search budget on news, portfolio
    # changes, and benchmark comparisons.
    fund_blocks = []
    for name in funds_batch:
        snap = _amfi_snapshots.get(name, {})
        facts = []
        if snap.get("nav_latest"):        facts.append(f"NAV: {snap['nav_latest']} (as of {snap.get('_amfi_nav_date','?')}, source: AMFI/mfapi.in)")
        if snap.get("fund_category"):     facts.append(f"Category: {snap['fund_category']}")
        if snap.get("one_year_return_pct"): facts.append(f"1Y CAGR: {snap['one_year_return_pct']}%")
        if snap.get("three_year_return_pct"): facts.append(f"3Y CAGR: {snap['three_year_return_pct']}%")
        if snap.get("five_year_return_pct"): facts.append(f"5Y CAGR: {snap['five_year_return_pct']}%")
        if snap.get("monthly_return_pct"): facts.append(f"Approx 30-day return: {snap['monthly_return_pct']}%")
        fact_str = ("\n  Pre-fetched facts (use as-is, verified from AMFI): " + "; ".join(facts)) if facts else ""
        fund_blocks.append(f"- {name}{fact_str}")

    listing = "\n".join(fund_blocks)
    return load_prompt(
        "mutual_fund/fund",
        today_str=today_str,
        lookback_note=lookback_note,
        listing=listing,
        SOURCE_QUALITY_NOTE=SOURCE_QUALITY_NOTE,
        NO_FABRICATION_NOTE=NO_FABRICATION_NOTE,
    )


# -----------------------------
# Stage 3 -- Sector performance (batched)
# -----------------------------
def build_sector_prompt(sectors_batch, today_str, lookback_note):
    listing = ", ".join(sectors_batch)
    return load_prompt(
        "mutual_fund/sector",
        today_str=today_str,
        lookback_note=lookback_note,
        listing=listing,
        SOURCE_QUALITY_NOTE=SOURCE_QUALITY_NOTE,
        NO_FABRICATION_NOTE=NO_FABRICATION_NOTE,
    )


# -----------------------------
# Stage 4 -- Synthesis (non-live)
# -----------------------------
def build_synthesis_prompt(market_data, funds_data, sectors_data, portfolio, today_str):
    context = json.dumps(
        {"market": market_data, "funds": funds_data, "sectors": sectors_data},
        ensure_ascii=False,
    )
    fund_list = ", ".join(portfolio)
    return load_prompt(
        "mutual_fund/synthesis",
        context=context,
        portfolio_count=len(portfolio),
        fund_list=fund_list,
        today_str=today_str,
    )


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
def _valid_market_json(text):
    data = _parse_json_object(text)
    return isinstance(data, dict) and isinstance(data.get("developments"), list)


def run_market_stage(today_str, lookback_note):
    prompt = build_market_prompt(today_str, lookback_note)
    text, sources, live = generate_analysis(prompt, max_tokens=3000, validate_fn=_valid_market_json)
    if not text:
        err_msg = "No LLM backend produced Stage 1 (market/macro) output. Aborting without sending an email."
        log.error(err_msg)
        raise RuntimeError(err_msg)
    _require_live_or_abort(live, "Stage 1 (market/macro)")
    data = _parse_json_object(text)
    if not isinstance(data, dict):
        log.warning("Stage 1 output could not be parsed as JSON -- proceeding with an empty market section.")
        data = {}
    data.setdefault("developments", [])
    data.setdefault("market_sentiment", "Neutral")
    return data, sources, live


def _valid_funds_json(text):
    data = _parse_json_object(text)
    if not isinstance(data, dict):
        return False
    funds = data.get("funds")
    return isinstance(funds, list) or isinstance(funds, dict)


def _filter_to_requested_funds(funds, batch):
    """
    Keeps only the fund objects whose fund_name matches one of the funds
    actually requested in this batch (normalized, case/whitespace-insensitive
    match against MF_PORTFOLIO's names) -- and drops everything else.

    Without this, a live-search-grounded LLM call occasionally returns
    extra fund objects beyond what build_fund_prompt() asked for: a
    similarly-named fund that turned up in its search results, a peer/
    comparison fund mentioned in a news article, or an outright
    hallucinated name -- even though the prompt explicitly says "Exact
    fund name as given above" and "one object per fund listed above".
    Previously run_fund_stage() appended whatever list came back with no
    check at all, so any such extra entry rode straight through
    _enrich_funds_with_amfi() and into render_fund_cards(), showing up in
    the email as if it were part of MF_PORTFOLIO. Fund identity here is
    the whole point (this report is a review of a fixed personal
    portfolio, not fund discovery), so an unrequested name is always
    dropped rather than shown -- logged so it's visible, not just silently
    discarded. Also dedupes to the first entry per requested fund, in case
    the model repeats one.
    """
    wanted = {name.strip().casefold(): name for name in batch}
    kept, seen = [], set()
    for f in funds if isinstance(funds, list) else []:
        raw_name = str((f or {}).get("fund_name") or "").strip()
        key = raw_name.casefold()
        canonical = wanted.get(key)
        if canonical is None:
            log.warning(
                f"Dropping unrequested fund '{raw_name}' returned for batch "
                f"({', '.join(batch)}) -- not one of MF_PORTFOLIO's exact names."
            )
            continue
        if canonical in seen:
            log.warning(f"Duplicate entry for '{canonical}' in this batch's response -- keeping the first.")
            continue
        seen.add(canonical)
        f["fund_name"] = canonical  # normalize to the exact PORTFOLIO spelling
        kept.append(f)
    return kept


def run_fund_stage(today_str, lookback_note):
    all_funds, sources, used_live = [], [], False
    for batch in _chunks(PORTFOLIO, FUNDS_PER_BATCH):
        log.info(f"Stage 2 -- fund batch: {', '.join(batch)}")
        prompt = build_fund_prompt(batch, today_str, lookback_note)
        # Build targeted search queries per fund:
        # - Short name (strip "- Direct Growth") + AMFI + factsheet → finds official pages
        # - Short name + "news" + month/year → finds recent news
        # This is much more effective than the full long name in a search string.
        fund_queries = []
        now_ym = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%B %Y")
        for name in batch:
            short = re.sub(r"\s*-\s*Direct\s*Growth\s*$", "", name, flags=re.IGNORECASE).strip()
            fund_queries.append(f"{short} AMFI factsheet AUM expense ratio {now_ym}")
            fund_queries.append(f"{short} news portfolio changes {now_ym}")
        text, s, live = generate_analysis(
            prompt, max_tokens=4500, extra_context_queries=fund_queries,
            validate_fn=_valid_funds_json,
        )
        if not text:
            log.error(f"No LLM output for fund batch ({', '.join(batch)}) -- skipping this batch.")
            continue
        _require_live_or_abort(live, f"Stage 2 (fund batch: {', '.join(batch)})")
        data = _parse_json_object(text)
        funds = data.get("funds") if isinstance(data, dict) else None
        if isinstance(funds, dict):
            # Some models return a bare object instead of a list when the
            # batch has only one fund -- normalize instead of dropping it.
            funds = [funds]
        if isinstance(funds, list):
            all_funds.extend(_filter_to_requested_funds(funds, batch))
        else:
            snippet_len = 400
            head = text[:snippet_len]
            tail = text[-snippet_len:] if len(text) > snippet_len else ""
            log.warning(
                f"Could not parse fund JSON for batch: {', '.join(batch)} "
                f"(raw length={len(text)} chars). "
                f"Start of output: {head!r}"
                + (f" ... End of output: {tail!r}" if tail else "")
            )
        for src in s:
            if src not in sources:
                sources.append(src)
        used_live = used_live or live
    return all_funds, sources, used_live


def _valid_sectors_json(text):
    data = _parse_json_object(text)
    return isinstance(data, dict) and isinstance(data.get("sectors"), list)


def run_sector_stage(today_str, lookback_note):
    all_sectors, sources, used_live = [], [], False
    for batch in _chunks(SECTORS_MF, SECTORS_PER_BATCH):
        log.info(f"Stage 3 -- sector batch: {', '.join(batch)}")
        prompt = build_sector_prompt(batch, today_str, lookback_note)
        text, s, live = generate_analysis(prompt, max_tokens=2000, validate_fn=_valid_sectors_json)
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
    isn't gated by) live web search. Thin wrapper around
    llm_backend.generate_synthesis() -- the shared non-live reasoning
    tier (plain Groq -> plain Gemini) -- so this stage doesn't spend
    groq/compound, Tavily, Gemini-grounding, or Mistral quota that
    Stages 1-3's genuinely search-dependent calls may still need this run.
    Returns "" (falsy) rather than None on failure, matching
    generate_synthesis()'s contract -- update callers accordingly.
    """
    log.info("Stage 4 (synthesis) using non-live generate_synthesis().")
    return generate_synthesis(prompt, max_tokens=max_tokens, log_label="Stage 4 (synthesis)")


def run_synthesis_stage(market_data, funds_data, sectors_data, today_str):
    prompt = build_synthesis_prompt(market_data, funds_data, sectors_data, PORTFOLIO, today_str)
    text = _plain_generate(prompt)
    if not text:
        err_msg = "No LLM backend produced Stage 4 (synthesis) output. Aborting without sending an email."
        log.error(err_msg)
        raise RuntimeError(err_msg)
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


def _fund_section_title(text, count):
    """
    Same section-title treatment as _section_title(), but bolded, in the
    gold accent color instead of navy, and with the total number of funds
    analyzed this run appended -- so the Fund-wise Analysis header stands
    out from the other (default-styled) section headers and states its
    count up front rather than requiring a scroll-and-count.
    """
    fund_word = "fund" if count == 1 else "funds"
    return (
        f'<h2 style="margin:22px 0 10px;font-family:{SERIF};font-weight:700;'
        f'font-size:18px;color:#B08D57;border-bottom:2px solid #B08D57;padding-bottom:6px;">'
        f'{_esc(text)} '
        f'<span style="font-size:13px;font-weight:700;color:#4A5063;">'
        f'({count} {fund_word} analyzed)</span></h2>'
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


def _format_monthly_return_display(fund_data):
    for key in ("monthly_return_pct", "monthly_return", "monthly_return_percent", "return_pct"):
        value = fund_data.get(key)
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


def _quality_status(fund_data):
    decision_note = str(fund_data.get("decision_note") or "").strip()
    nav = fund_data.get("nav_latest")
    if decision_note and nav not in (None, "", "null", "None"):
        return "Verified", "#1E7A46"
    if decision_note or nav not in (None, "", "null", "None"):
        return "Partial", "#B08D57"
    return "Needs review", "#B0473F"


def _action_now_text(fund_data):
    reco = (fund_data.get("recommendation") or "").strip().lower()
    if reco in {"strong buy", "buy", "continue sip"}:
        return "Consider adding gradually and monitor the fund’s positioning versus its benchmark."
    if reco in {"hold", "review"}:
        return "Wait for a clearer trend shift before changing SIP or allocation."
    if reco in {"reduce", "exit", "sell"}:
        return "Trim exposure and reassess if the fund’s risk profile or flows weaken."
    return "Monitor the fund with a long-term lens and reassess on a new data point."


def render_fund_cards(funds_data):
    if not funds_data:
        return _fund_section_title("2. Fund-wise Analysis (Last 30 Days)", 0) + (
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
        ret_str = _format_monthly_return_display(f)
        quality_status, quality_color = _quality_status(f)
        action_now = _action_now_text(f)

        # Build snapshot using a flexible structure -- only show a row if at
        # least one cell in that row has real data.
        def _snap_val(key, fallback="Not disclosed"):
            v = f.get(key)
            return str(v).strip() if v and str(v).strip() not in ("", "null", "None", "Not disclosed") else fallback

        snapshot_items = [
            ("Category", _snap_val("fund_category")),
            ("Benchmark", _snap_val("benchmark")),
            ("NAV", _snap_val("nav_latest")),
            ("AUM (₹ Cr)", _snap_val("aum_cr")),
            ("Expense Ratio", _snap_val("expense_ratio_pct")),
            ("Risk", _snap_val("risk_level")),
            ("1Y CAGR", _snap_val("one_year_return_pct")),
            ("3Y CAGR", _snap_val("three_year_return_pct")),
            ("5Y CAGR", _snap_val("five_year_return_pct")),
            ("~30d Return", ret_str if ret_str != "n/a" else "Not disclosed"),
        ]
        snapshot_html = ""
        for idx in range(0, len(snapshot_items), 2):
            row_items = snapshot_items[idx:idx+2]
            row_cells = "".join(
                f'<td style="width:50%;padding:6px 0;vertical-align:top;font-family:{SANS};font-size:11px;color:#8A8F9C;">'
                f'<div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.04em;">{_esc(label)}</div>'
                f'<div style="margin-top:3px;font-size:12px;font-weight:700;color:{"#8A8F9C" if value == "Not disclosed" else "#14213D"};">'
                f'{_esc(value)}</div></td>'
                for label, value in row_items
            )
            snapshot_html += f'<tr>{row_cells}</tr>'

        # Attribution note when AMFI data was used
        amfi_code = f.get("_amfi_code")
        amfi_nav_date = f.get("_amfi_nav_date", "")
        amfi_note = (
            f'<div style="font-family:{SANS};font-size:10px;color:#8A8F9C;margin-top:4px;">'
            f'NAV &amp; returns: AMFI/mfapi.in (scheme code {_esc(amfi_code)}, NAV date {_esc(amfi_nav_date)})</div>'
            if amfi_code else ""
        )

        cards.append(f"""
        <table width="100%" cellpadding="0" cellspacing="0" role="presentation"
               style="margin:14px 0;border:1px solid #E7E4DC;border-radius:6px;overflow:hidden;border-collapse:collapse;">
          <tr>
            <td style="padding:10px 14px;background:#14213D;">
              <table width="100%" cellpadding="0" cellspacing="0" role="presentation">
                <tr>
                  <td style="font-family:{SERIF};font-size:14.5px;color:#ffffff;vertical-align:middle;">{_esc(name)}</td>
                  <td style="text-align:right;vertical-align:middle;">
                    <span style="font-family:{SANS};font-size:11px;font-weight:700;color:{_reco_color(reco)};
                          background:#ffffff;padding:2px 10px;border-radius:12px;display:inline-block;">{_esc(reco)}</span>
                  </td>
                </tr>
              </table>
            </td>
          </tr>
          <tr>
            <td style="padding:12px 14px;">
              <div style="font-family:{SANS};font-size:11px;font-weight:700;color:#8A8F9C;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px;">News Timeline</div>
              {news_html}
              <div style="font-family:{SANS};font-size:11px;font-weight:700;color:#8A8F9C;text-transform:uppercase;letter-spacing:0.05em;margin:10px 0 4px;">Portfolio Changes</div>
              <p style="margin:0;font-family:{SANS};font-size:12px;color:#4A5063;line-height:1.6;white-space:pre-wrap;word-break:break-word;">{_esc(f.get("portfolio_changes","-"))}</p>
              <div style="font-family:{SANS};font-size:11px;font-weight:700;color:#8A8F9C;text-transform:uppercase;letter-spacing:0.05em;margin:10px 0 4px;">Snapshot</div>
              <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="margin-top:4px;border-collapse:collapse;">
                <tbody>
                  {snapshot_html}
                </tbody>
              </table>
              {amfi_note}
              <p style="margin:8px 0 0;font-family:{SANS};font-size:12px;color:#4A5063;line-height:1.6;white-space:pre-wrap;word-break:break-word;"><strong>Decision note:</strong> {_esc(f.get("decision_note","-"))}</p>
              <div style="margin-top:10px;padding:8px 10px;background:#fffdf8;border:1px solid #F2E2BF;border-radius:4px;">
                <div style="font-family:{SANS};font-size:11px;font-weight:700;color:#8A8F9C;text-transform:uppercase;letter-spacing:0.04em;">Action now</div>
                <div style="margin-top:4px;font-family:{SANS};font-size:12px;color:#14213D;line-height:1.5;">{_esc(action_now)}</div>
              </div>
              <div style="margin-top:8px;font-family:{SANS};font-size:11px;color:#4A5063;">
                <span style="display:inline-block;padding:2px 8px;border-radius:999px;background:{quality_color}1A;color:{quality_color};font-weight:700;">{_esc(quality_status)}</span>
              </div>
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
    return _fund_section_title("2. Fund-wise Analysis (Last 30 Days)", len(funds_data)) + "".join(cards)


def build_email_html(market_data, funds_data, sectors_data, synthesis_data, sources, used_live_search, today_str):
    # NOTE: "3. Market News (Past 30 Days)" and "4. Sector Performance" were
    # removed from the email on request -- the underlying market_data /
    # sectors_data are still fetched (Stage 1 / Stage 3 below) because
    # market_data still feeds the Executive Summary's sentiment badge/reason
    # and sectors_data still feeds Stage 4's synthesized "top developments"
    # list -- only the two standalone tables are gone.
    sections = (
        render_executive_summary(market_data, synthesis_data)
        + render_fund_cards(funds_data)
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
<meta http-equiv="X-UA-Compatible" content="IE=edge">
<meta name="x-apple-disable-message-reformatting">
<meta name="format-detection" content="telephone=no, date=no, address=no, email=no">
<meta name="color-scheme" content="light">
<meta name="supported-color-schemes" content="light">
<title>Mutual Fund Portfolio Review</title>
<style>
  body, table, td, a {{ -webkit-text-size-adjust:100%; -ms-text-size-adjust:100%; }}
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
    subject = f"Mutual Fund Portfolio Review — {config.get_date_with_suffix(now_ist)} · {time_str}"

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
        log.info("Mutual fund portfolio email sent successfully.")
        return True
    except smtplib.SMTPAuthenticationError:
        log.error(
            "SMTP Authentication Error: check EMAIL_FROM/EMAIL_PASSWORD "
            "(use a Gmail App Password, not the account password)."
        )
    except Exception as e:
        log.error(f"Failed to send mutual fund portfolio email: {e}")
        traceback.print_exc()
    return False


def run():
    today_str, now_ist, lookback_note = _run_context()
    log.info(f"Mutual fund portfolio review starting for {today_str} (portfolio: {len(PORTFOLIO)} funds).")

    # Prefetch deterministic AMFI data for all funds before the LLM stage.
    # This runs once, populates _amfi_snapshots, and gives the LLM prompt
    # pre-verified facts (NAV, 1Y/3Y returns, category) so it doesn't need
    # to search for -- or guess -- values that are available from AMFI.
    prefetch_amfi_snapshots()

    market_data, market_sources, market_live = run_market_stage(today_str, lookback_note)
    funds_data, fund_sources, funds_live = run_fund_stage(today_str, lookback_note)

    # Merge AMFI-sourced facts into LLM output -- fills any fields the LLM
    # left as "Not disclosed" with the verified values fetched earlier.
    funds_data = _enrich_funds_with_amfi(funds_data)

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
        with open("mutual_fund_report.html", "w", encoding="utf-8") as f:
            f.write(email_html)
        log.info("DRY_RUN enabled -- wrote mutual_fund_report.html instead of emailing.")
        return

    if not send_portfolio_email(email_html, today_str):
        log.critical("Failed to send mutual fund email report.")


if __name__ == "__main__":
    run()