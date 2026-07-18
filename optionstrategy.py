"""
optionstrategy.py

Standalone companion to main.py, structured the same way as
swing_trade_advisor.py. Runs a single "recommend the best risk-defined
Nifty options strategy across Weekly / Monthly / Quarterly / Annual
horizons" prompt against whichever free LLM backend main.py already knows
how to set up (Groq free tier -> Gemini 2.5 Flash free tier -> local
Qwen2.5-1.5B fallback), then emails the result to the same recipients
configured for the stock report (EMAIL_TO / EMAIL_CC in config.py / the
workflow yaml's env vars).

This deliberately does NOT re-implement the LLM-selection / live-search
fallback chain -- it reuses swing_trade_advisor.generate_analysis(), which
is already generic (it just takes a prompt string and returns
(text, sources, used_live_search)), so both scripts stay in sync with
whatever provider/search path is configured. Email credential plumbing is
reused from main.py the same way swing_trade_advisor.py does it.

LIVE DATA: this script fetches Nifty spot, India VIX, and a full per-horizon
options-chain snapshot (PCR by OI, max pain, top-OI strikes on both sides)
directly from NSE India's public option-chain JSON endpoint and Yahoo
Finance BEFORE calling the LLM, and embeds those real numbers straight into
the prompt as ground truth (see fetch_live_market_data() / build_prompt()).
NSE has no official public API for this and can rate-limit or block
non-browser traffic -- especially likely from a CI runner's IP (e.g. GitHub
Actions) -- so the fetch degrades gracefully: on partial/total failure it
falls through to swing_trade_advisor.generate_analysis()'s own live-search
path (groq/compound's built-in web search, then Tavily+Groq, then
Gemini+Google Search grounding) to fill the gaps at generation time.
REQUIRE_LIVE_DATA=true (the default) refuses to email a run where NEITHER
the direct fetch NOR the LLM's own search produced anything live, exactly
like swing_trade_advisor.py.

CAVEAT: this is not a verified real-time trading signal. Both the direct
feed and web search results can be a few minutes to hours stale, and the
model can still misread or mis-combine what it's given. Every strike, level,
and Greek-adjacent figure below is a starting hypothesis to verify against
your broker's live options chain (e.g. Kite Connect / Upstox API, or NSE's
site directly) before placing any order. Not investment advice.
"""

import os
import re
import io
import csv
import sys
import json
import html
import zipfile
import traceback
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo

import requests
try:
    import yfinance as yf
except ImportError:
    yf = None  # VIX/spot cross-check degrades gracefully if not installed

import smtplib
from email.mime.text import MIMEText

import main                 # LLM init + email config/credentials (via config.py)
import swing_trade_advisor as swing  # reuses generate_analysis() + helpers, no duplication

# -----------------------------
# Config (env-overridable capital caps, per the prompt's constraint #3)
# -----------------------------
PER_HORIZON_CAP_PCT = float(os.getenv("OPTIONS_PER_HORIZON_CAP_PCT", "5"))
AGGREGATE_CAP_PCT = float(os.getenv("OPTIONS_AGGREGATE_CAP_PCT", "15"))

HORIZON_ORDER = ["Weekly", "Monthly", "Quarterly", "Annual"]

# -----------------------------
# Live data fetch (NSE India option-chain API + Yahoo Finance)
# -----------------------------
# NSE has no official public API for this, but nseindia.com's own
# option-chain page calls this exact JSON endpoint client-side, so it's the
# same "standard website" data a human would read there. It requires a
# browser-like User-Agent and a warm-up hit to set session cookies -- calling
# it cold returns 401/403. It also actively rate-limits/blocks non-browser
# traffic, which is especially likely from a CI runner's IP (e.g. GitHub
# Actions) -- so every step below degrades gracefully rather than crashing
# the run. Needs `pip install requests yfinance`.
_NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/option-chain",
}


def fetch_nse_option_chain(symbol="NIFTY", timeout=12):
    """Pulls the live, full option chain (all listed expiries, every
    strike's OI/changeInOI/IV/LTP for CE and PE) straight from NSE's
    option-chain JSON endpoint. Returns the parsed dict, or None on any
    failure (network, block, non-200, bad JSON)."""
    try:
        session = requests.Session()
        session.headers.update(_NSE_HEADERS)
        session.get("https://www.nseindia.com/option-chain", timeout=timeout)  # sets cookies
        resp = session.get(
            f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}",
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        main.log.warning(f"NSE option-chain fetch failed for {symbol}: {e}")
        return None


def _parse_nse_date(d):
    return datetime.strptime(d, "%d-%b-%Y")


def _pick_horizon_expiry_dates(dt_list):
    """Shared horizon-selection logic, operating on plain datetime objects
    so both the live option-chain JSON path (DD-Mon-YYYY strings) and the
    Bhavcopy fallback path (YYYY-MM-DD strings) pick horizons identically:
    nearest weekly / current monthly / nearest Mar-Jun-Sep-Dec quarterly /
    farthest available."""
    dts = sorted(set(dt_list))
    if not dts:
        return {}
    weekly_dt = dts[0]
    same_month = [dt for dt in dts if (dt.year, dt.month) == (weekly_dt.year, weekly_dt.month)]
    monthly_dt = same_month[-1] if same_month else weekly_dt
    quarter_candidates = [dt for dt in dts if dt.month in (3, 6, 9, 12) and dt >= monthly_dt]
    quarterly_dt = quarter_candidates[0] if quarter_candidates else dts[-1]
    annual_dt = dts[-1]
    return {"Weekly": weekly_dt, "Monthly": monthly_dt, "Quarterly": quarterly_dt, "Annual": annual_dt}


def _pick_horizon_expiries(expiry_dates):
    """Maps NSE's raw list of listed expiry-date STRINGS (live option-chain
    JSON, DD-Mon-YYYY format) onto the four horizon definitions used in the
    prompt, via the shared datetime-based picker above."""
    by_dt = {_parse_nse_date(d): d for d in expiry_dates}
    picked = _pick_horizon_expiry_dates(by_dt.keys())
    return {h: by_dt[dt] for h, dt in picked.items()}


def _extract_expiry_snapshot(rows, expiry_str):
    """From NSE's raw per-strike rows, compute the figures the prompt needs
    for one expiry: PCR by OI, max pain, and the top-5 OI strikes on each
    side (a direct, computed liquidity signal for constraint #7)."""
    calls, puts = {}, {}
    for r in rows:
        if r.get("expiryDate") != expiry_str:
            continue
        strike = r.get("strikePrice")
        if r.get("CE"):
            calls[strike] = r["CE"]
        if r.get("PE"):
            puts[strike] = r["PE"]

    total_call_oi = sum(c.get("openInterest", 0) for c in calls.values())
    total_put_oi = sum(p.get("openInterest", 0) for p in puts.values())
    pcr_oi = round(total_put_oi / total_call_oi, 2) if total_call_oi else None

    top_calls = sorted(calls.items(), key=lambda kv: kv[1].get("openInterest", 0), reverse=True)[:5]
    top_puts = sorted(puts.items(), key=lambda kv: kv[1].get("openInterest", 0), reverse=True)[:5]

    max_pain = None
    all_strikes = sorted(set(calls) | set(puts))
    if all_strikes:
        pain = {
            k: (
                sum(c.get("openInterest", 0) * max(0, k - s) for s, c in calls.items())
                + sum(p.get("openInterest", 0) * max(0, s - k) for s, p in puts.items())
            )
            for k in all_strikes
        }
        max_pain = min(pain, key=pain.get)

    return {
        "expiry": expiry_str,
        "pcr_oi": pcr_oi,
        "max_pain": max_pain,
        "total_call_oi": total_call_oi,
        "total_put_oi": total_put_oi,
        "top_call_oi": [(s, c.get("openInterest", 0)) for s, c in top_calls],
        "top_put_oi": [(s, p.get("openInterest", 0)) for s, p in top_puts],
    }


# -----------------------------
# EOD fallback: NSE's official Bhavcopy (UDiFF format)
# -----------------------------
# When the live option-chain JSON endpoint above is blocked/rate-limited
# (common from CI/cloud IPs), this pulls NSE's own official end-of-day F&O
# settlement file instead. It's a static daily ZIP, not a live quote feed,
# so figures reflect the last trading day's CLOSE -- not intraday moves --
# but it's still real official exchange data (OI, settlement price, PCR,
# max pain), which is strictly better than the LLM guessing from memory.
# Format confirmed via NSE Circular No. 62424 (12 Jun 2024): the old
# `foDDMONbhav.csv.zip` path was retired 08 Jul 2024 in favour of this
# UDiFF path. Published only after each trading day's close (typically
# from ~7-8 PM IST onward).
_BHAVCOPY_FO_URL = "https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{yyyymmdd}_F_0000.csv.zip"


def _parse_bhavcopy_date(s):
    s = (s or "").strip()
    for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unrecognized Bhavcopy date format: {s!r}")


def fetch_nse_bhavcopy_fo(trade_date, timeout=15):
    """Downloads and parses one day's official NSE F&O UDiFF Bhavcopy (a
    zipped EOD settlement CSV covering every listed F&O instrument) for the
    given date. Returns a list of CSV row dicts, or None on failure (file
    not yet published, weekend/holiday, network block, format change)."""
    url = _BHAVCOPY_FO_URL.format(yyyymmdd=trade_date.strftime("%Y%m%d"))
    try:
        session = requests.Session()
        session.headers.update(_NSE_HEADERS)
        resp = session.get(url, timeout=timeout)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
            with zf.open(csv_name) as f:
                return list(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")))
    except Exception as e:
        main.log.warning(f"NSE Bhavcopy fetch failed for {trade_date.date()}: {e}")
        return None


def fetch_latest_nse_bhavcopy_fo(max_days_back=6):
    """Walks backward from today (skipping the file NSE hasn't published
    yet for "today" before ~7 PM IST) to find the most recent trading
    day's F&O Bhavcopy. Returns (rows, trade_date) or (None, None)."""
    now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
    for i in range(max_days_back + 1):
        candidate = now_ist - timedelta(days=i)
        if i == 0 and now_ist.time() < dtime(19, 0):
            continue  # today's file isn't published yet -- skip straight to yesterday
        rows = fetch_nse_bhavcopy_fo(candidate)
        if rows:
            return rows, candidate.date()
    return None, None


def _extract_bhavcopy_snapshot(rows, symbol, expiry_dt):
    """Same output shape as _extract_expiry_snapshot (PCR/max-pain/top-OI),
    computed instead from one day's official EOD Bhavcopy rows for a single
    NIFTY index-options (FinInstrmTp='IDO') expiry."""
    calls, puts = {}, {}
    for r in rows:
        if r.get("TckrSymb", "").strip() != symbol or r.get("FinInstrmTp", "").strip() != "IDO":
            continue
        try:
            if _parse_bhavcopy_date(r.get("XpryDt", "")) != expiry_dt:
                continue
            strike = float(r.get("StrkPric") or 0)
            strike = int(strike) if strike.is_integer() else strike
            oi = int(float(r.get("OpnIntrst") or 0))
        except (ValueError, TypeError):
            continue
        opt_type = r.get("OptnTp", "").strip()
        if opt_type == "CE":
            calls[strike] = oi
        elif opt_type == "PE":
            puts[strike] = oi

    total_call_oi = sum(calls.values())
    total_put_oi = sum(puts.values())
    pcr_oi = round(total_put_oi / total_call_oi, 2) if total_call_oi else None

    top_calls = sorted(calls.items(), key=lambda kv: kv[1], reverse=True)[:5]
    top_puts = sorted(puts.items(), key=lambda kv: kv[1], reverse=True)[:5]

    max_pain = None
    all_strikes = sorted(set(calls) | set(puts))
    if all_strikes:
        pain = {
            k: (
                sum(oi * max(0, k - s) for s, oi in calls.items())
                + sum(oi * max(0, s - k) for s, oi in puts.items())
            )
            for k in all_strikes
        }
        max_pain = min(pain, key=pain.get)

    return {
        "expiry": expiry_dt.strftime("%d-%b-%Y"),
        "pcr_oi": pcr_oi,
        "max_pain": max_pain,
        "total_call_oi": total_call_oi,
        "total_put_oi": total_put_oi,
        "top_call_oi": top_calls,
        "top_put_oi": top_puts,
    }


def _fill_horizons_from_bhavcopy(data, notes, symbol="NIFTY"):
    """Fallback path used when the live option-chain JSON fetch failed
    entirely: pulls the latest available EOD Bhavcopy and fills data["horizons"]
    from it, clearly tagging every filled horizon as EOD (not live)."""
    bhav_rows, bhav_date = fetch_latest_nse_bhavcopy_fo()
    if not bhav_rows:
        notes.append("EOD Bhavcopy fallback also failed or is unavailable (no file found in the last 6 days).")
        return False

    try:
        expiry_dts = {
            _parse_bhavcopy_date(r.get("XpryDt", ""))
            for r in bhav_rows
            if r.get("TckrSymb", "").strip() == symbol and r.get("FinInstrmTp", "").strip() == "IDO"
        }
        horizon_dts = _pick_horizon_expiry_dates(expiry_dts)
        for horizon, dt in horizon_dts.items():
            snap = _extract_bhavcopy_snapshot(bhav_rows, symbol, dt)
            snap["source"] = f"EOD Bhavcopy ({bhav_date.strftime('%d-%b-%Y')})"
            data["horizons"][horizon] = snap

        if data["spot"] is None:
            fut_closes = sorted(
                (
                    (_parse_bhavcopy_date(r.get("XpryDt", "")), float(r.get("ClsPric") or r.get("SttlmPric") or 0))
                    for r in bhav_rows
                    if r.get("TckrSymb", "").strip() == symbol and r.get("FinInstrmTp", "").strip() == "IDF"
                ),
                key=lambda t: t[0],
            )
            if fut_closes:
                data["spot"] = fut_closes[0][1]
                notes.append("Spot figure is a Bhavcopy near-month futures CLOSE proxy, not true cash spot.")

        notes.append(
            f"Live NSE option-chain feed was unavailable this run -- per-horizon OI/PCR/max-pain "
            f"were filled from NSE's official EOD Bhavcopy dated {bhav_date.strftime('%d %b %Y')} "
            f"instead (last trading day's CLOSE, not live/intraday)."
        )
        return True
    except Exception as e:
        notes.append(f"Bhavcopy was fetched but could not be parsed: {e}")
        return False


def fetch_live_market_data():
    """Top-level orchestrator: pulls Nifty spot, India VIX, and a full
    per-horizon option-chain snapshot (PCR/max-pain/top-OI strikes) BEFORE
    the LLM is ever called, so the model is reasoning over real fetched
    numbers instead of whatever it recalls or half-finds via its own
    search. Tries the live option-chain JSON first; if that's blocked, falls
    back to NSE's official EOD Bhavcopy before giving up on direct data
    entirely. Returns a dict; always check "status" ("ok"/"eod_fallback"/
    "partial"/"failed") and read "notes" for what, if anything, happened."""
    notes = []
    data = {
        "status": "failed",
        "fetched_at": datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%d %b %Y, %I:%M %p IST"),
        "spot": None,
        "vix": None,
        "vix_change_pct": None,
        "horizons": {},
        "notes": notes,
    }

    if yf is not None:
        try:
            vix_hist = yf.Ticker("^INDIAVIX").history(period="5d")
            if not vix_hist.empty:
                data["vix"] = round(float(vix_hist["Close"].iloc[-1]), 2)
                if len(vix_hist) >= 2:
                    prev = float(vix_hist["Close"].iloc[-2])
                    data["vix_change_pct"] = round((data["vix"] - prev) / prev * 100, 2)
        except Exception as e:
            notes.append(f"Yahoo Finance VIX fetch failed: {e}")
        try:
            spot_hist = yf.Ticker("^NSEI").history(period="1d")
            if not spot_hist.empty:
                data["spot"] = round(float(spot_hist["Close"].iloc[-1]), 2)
        except Exception as e:
            notes.append(f"Yahoo Finance Nifty spot fetch failed: {e}")
    else:
        notes.append("yfinance not installed -- spot/VIX cross-check skipped (pip install yfinance).")

    chain = fetch_nse_option_chain("NIFTY")
    if chain is None:
        notes.append(
            "NSE live option-chain fetch failed (blocked, rate-limited, or "
            "endpoint changed) -- trying the EOD Bhavcopy fallback instead."
        )
        if _fill_horizons_from_bhavcopy(data, notes):
            data["status"] = "eod_fallback"
        else:
            data["status"] = "partial" if (data["spot"] or data["vix"]) else "failed"
        return data

    try:
        records = chain.get("records", {})
        if data["spot"] is None:
            data["spot"] = records.get("underlyingValue")
        horizon_expiries = _pick_horizon_expiries(records.get("expiryDates", []))
        rows = records.get("data", [])
        for horizon, expiry in horizon_expiries.items():
            data["horizons"][horizon] = _extract_expiry_snapshot(rows, expiry)
        if "Annual" in data["horizons"]:
            notes.append(
                "NSE's feed rarely carries a genuinely liquid 12-month-out "
                "expiry -- the 'Annual' snapshot is just the farthest expiry "
                "NSE returned; verify its real liquidity (constraint #7) and "
                "lean on live web search if it's not actually far/liquid enough."
            )
        data["status"] = "ok"
    except Exception as e:
        notes.append(f"NSE option-chain data was fetched but could not be parsed -- trying the EOD Bhavcopy fallback: {e}")
        if _fill_horizons_from_bhavcopy(data, notes):
            data["status"] = "eod_fallback"
        else:
            data["status"] = "partial" if (data["spot"] or data["vix"]) else "failed"

    return data


def _fmt_oi_pairs(pairs):
    if not pairs:
        return "n/a"
    return "; ".join(f"{strike}: {oi:,} OI" for strike, oi in pairs)


def format_live_data_block(data):
    """Renders the fetched dict as a plain-text block to embed directly in
    the LLM prompt, framed explicitly as ground truth rather than something
    to re-derive from web search."""
    if not data or data.get("status") == "failed":
        note_text = "; ".join(data.get("notes", [])) if data else "fetch not attempted"
        return (
            "LIVE DATA FEED: unavailable this run (direct NSE/Yahoo fetch, and "
            f"the EOD Bhavcopy fallback, both failed). Notes: {note_text}. You "
            "must rely entirely on your own live web search for every figure in "
            "the ADDITIONAL LIVE DATA section below, and be honest about this "
            "in each horizon's data_status field."
        )

    lines = [
        f"LIVE DATA FEED (fetched directly from NSE India and Yahoo Finance at "
        f"{data['fetched_at']}, status={data['status']}). Treat every figure "
        f"below as verified ground truth -- do NOT re-derive or second-guess "
        f"these via web search. Each horizon is tagged with its data source: "
        f"'live (NSE option-chain API)' means current intraday data -- use "
        f"data_status='live' for that horizon if nothing else is stale. "
        f"'EOD Bhavcopy (<date>)' means NSE's official end-of-day settlement "
        f"file, i.e. the last trading day's CLOSE, not intraday -- for those "
        f"horizons use data_status='partial' at best and say so in bias_reason. "
        f"Only web-search for whatever is marked unavailable below, plus "
        f"qualitative context (FII/DII flows, GIFT Nifty pre-market, US/Asian "
        f"markets overnight, event risk):",
        f"- Nifty 50 spot: {data.get('spot', 'n/a')}",
        "- India VIX: "
        + str(data.get("vix", "n/a"))
        + (f" ({data['vix_change_pct']:+.2f}% vs prior close)" if data.get("vix_change_pct") is not None else ""),
    ]
    for horizon in HORIZON_ORDER:
        snap = data.get("horizons", {}).get(horizon)
        if not snap:
            lines.append(f"- {horizon} expiry: not available from the direct feed -- find via web search.")
            continue
        source = snap.get("source", "live (NSE option-chain API)")
        lines.append(
            f"- {horizon} expiry ({snap['expiry']}, source: {source}): PCR(OI)={snap.get('pcr_oi', 'n/a')}, "
            f"Max Pain={snap.get('max_pain', 'n/a')}, Total Call OI="
            f"{snap.get('total_call_oi', 0):,}, Total Put OI={snap.get('total_put_oi', 0):,}"
        )
        lines.append(f"    Top Call OI strikes: {_fmt_oi_pairs(snap.get('top_call_oi'))}")
        lines.append(f"    Top Put OI strikes: {_fmt_oi_pairs(snap.get('top_put_oi'))}")

    if data.get("notes"):
        lines.append("Fetch notes: " + "; ".join(data["notes"]))

    return "\n".join(lines)


def _market_session_label():
    """
    NSE regular trading session is 09:15-15:30 IST. This run is allowed to
    fire at any point (constraint #3 in the prompt), but the email should
    say plainly whether it was generated during a live session (options
    chain likely fresh) or outside one (options chain will reflect the
    last close, not live quotes) -- that materially changes how much to
    trust the strike levels below.
    """
    now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
    is_weekday = now_ist.weekday() < 5  # Mon-Fri
    in_session = is_weekday and dtime(9, 15) <= now_ist.time() <= dtime(15, 30)
    if in_session:
        return "Live NSE trading session", True
    return "Outside NSE trading hours (data reflects last close, not live quotes)", False


# -----------------------------
# Prompt
# -----------------------------
def build_prompt(live_data=None):
    today_str = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%d %B %Y")
    session_label, in_session = _market_session_label()
    live_data_block = format_live_data_block(live_data)

    return f"""Core Objective: You are an expert at analysing Nifty options-related data. A LIVE DATA FEED has already been fetched for you and is provided below -- use those figures as ground truth for this run rather than searching for or guessing them yourself. Using that feed (plus web search only for what it doesn't cover), recommend the best risk-defined strategy -- or combination of strategies -- for the current moment, independently across FOUR time horizons:

{live_data_block}

1. Nifty Weekly -- Buy, Sell, or a Combination of both (current/nearest weekly expiry)
2. Nifty Monthly -- Buy, Sell, or a Combination of both (current monthly expiry)
3. Nifty Quarterly -- Buy, Sell, or a Combination of both (nearest quarterly expiry on the Mar/Jun/Sep/Dec cycle)
4. Nifty Annual -- Buy, Sell, or a Combination of both (farthest available annual/LEAPS-style expiry)

Horizons are independent from Monthly/Quarterly/Annual onward -- they do not need to agree with each other. For example Monthly can be a bullish debit spread while Quarterly is a range-bound Iron Condor, if the data supports it. The Weekly horizon, however, follows the bias-sequencing rule below (constraint #5), which explicitly couples the current week to the next week.

NON-NEGOTIABLE CONSTRAINTS

1. RISK-DEFINED ONLY. Every recommended strategy must have a mathematically capped maximum loss known before entry (e.g. Bull/Bear Call/Put Spreads, Iron Condors, Iron Butterflies, Calendar Spreads, Ratio Spreads with a protective wing, Debit/Credit Spreads). NEVER recommend a naked/undefined-risk position (naked short call, naked short put, uncovered short straddle/strangle) under any circumstance, for any horizon.

2. GAP PROTECTION. Every recommendation must explicitly address overnight/weekend gap risk:
   - State the max loss if Nifty gaps up or down beyond the current day's expected move (derive the expected move from India VIX for that horizon's time-to-expiry).
   - Prefer structures whose short strikes sit outside the current expected-move band (VIX-implied 1 standard deviation) for that horizon.
   - If global cues (GIFT Nifty pre-market, US markets overnight, Asian markets) suggest elevated gap risk right now, explicitly flag it and bias the recommendation toward tighter risk-defined structures or reduced size -- never toward removing a hedge leg.

3. CAPITAL PROTECTION RULES.
   - Max capital at risk per horizon must not exceed {PER_HORIZON_CAP_PCT:.0f}% of a hypothetical total options-trading capital pool.
   - Aggregate capital at risk across all four horizons combined must not exceed {AGGREGATE_CAP_PCT:.0f}%.
   - State the stop-loss / adjustment trigger for each recommendation (e.g. "exit if Nifty closes beyond X level" or "adjust if short strike is breached").

4. This is a stateless, any-time request -- today is {today_str}, and this run is being generated during: {session_label}. Produce a full, self-contained recommendation using only live data you can find right now; do not assume access to any prior recommendation.

5. WEEKLY BIAS SEQUENCING.
   - Determine the Current Week's bias (Bullish or Bearish) strictly from the live market data you find (spot/futures trend, OI buildup, PCR, FII/DII flow, global cues) -- never assume a default direction.
   - Apply a mean-reversion assumption for the immediate next weekly expiry: if the Current Week is Bullish, treat the Next Week as Bearish; if the Current Week is Bearish, treat the Next Week as Bullish. Reflect this explicitly in the Weekly horizon's strategy legs and in a dedicated "next_week_bias" field (see schema below) -- e.g. a current-week bull put spread paired with a next-week-aware bear call spread or calendar structure that benefits from the expected reversal.
   - If live data shows a clear reason to override this assumption (e.g. a major event landing exactly in the next-week window), say so explicitly in bias_reason rather than silently ignoring the rule.

6. MONTHLY/QUARTERLY BIAS TOWARD SIDEWAYS/RANGE-BOUND. Unless live data shows a strong, well-supported directional catalyst inside that horizon's window, default the Monthly and Quarterly recommendations to range-bound/theta-positive structures (Iron Condor, Iron Butterfly, Calendar/Diagonal Spread) rather than pure directional debit/credit spreads -- these horizons are for harvesting range and time decay, not chasing the weekly directional call.

7. NIFTY LIQUIDITY FILTER. Only select strikes with adequate live liquidity for Nifty options -- meaningful open interest, tight bid-ask spread, and strikes close to standard NSE strike intervals. If a theoretically ideal strike is illiquid, state that explicitly and move to the nearest liquid strike instead. Note the liquidity basis (OI/spread) briefly for each leg selected.

8. RISK-CONTROLLED SYNTHESIS. Every horizon's final recommendation must simultaneously satisfy constraints #1-#7 above -- risk-defined, gap-protected, capital-capped, correctly bias-sequenced (Weekly), correctly range-biased (Monthly/Quarterly), and liquidity-filtered. If any of these pull in conflicting directions for a given horizon (e.g. the liquid strike sits inside the expected-move band), state the trade-off explicitly in that horizon's bias_reason and resolve it in favor of the tighter risk-defined structure, never in favor of higher theoretical reward.

ADDITIONAL LIVE DATA TO SEARCH FOR (only for what the LIVE DATA FEED above does not already give you -- do not re-search or contradict anything already provided there; if you cannot find a real current figure, say so for that field rather than inventing one):
- Nifty futures price (weekly/monthly, for basis) -- the feed above gives spot, not futures
- Confirmation/context on options-chain levels for any horizon the feed marked unavailable
- FII/DII cash + index-options net positioning (most recent session) -- not covered by the feed
- GIFT Nifty pre-market level (if available) and prior-session US market close (S&P 500, Nasdaq), for gap risk
- Any major near-term event risk (RBI policy, US Fed meeting, major earnings cluster, budget/election dates) that falls within each horizon's window

OUTPUT FORMAT -- respond with ONLY raw JSON matching the schema below, and nothing else (no markdown, no code fences, no commentary before or after). Keep every field to plain text/numbers only (no HTML):

{{
  "horizons": [
    {{
      "horizon": "Weekly",
      "expiry_date": "The actual expiry date used, e.g. '24 Jul 2026'",
      "bias": "One of: Bullish / Bearish / Neutral / Range-bound",
      "next_week_bias": "ONLY for the Weekly horizon object: 'Bullish' or 'Bearish', the opposite of 'bias' per constraint #5's mean-reversion rule (or a brief override note if constraint #5 was overridden). Omit or leave empty for Monthly/Quarterly/Annual.",
      "bias_reason": "One sentence grounded in the live data you found",
      "strategy_name": "e.g. 'Bear Call Spread' or 'Iron Condor'",
      "legs": "Full leg structure, e.g. 'Sell 25000 CE, Buy 25200 CE (1 lot)'",
      "max_loss": "Rupee figure, e.g. '₹9,500 per lot'",
      "max_loss_pct_capital": "% of allocated capital for this horizon, must be <= {PER_HORIZON_CAP_PCT:.0f}",
      "max_profit": "Rupee figure",
      "max_profit_pct_capital": "% of allocated capital for this horizon",
      "breakeven": "Breakeven level(s)",
      "gap_risk": "Explicit statement of loss if Nifty gaps beyond the expected move before next session, per constraint #2",
      "adjustment_trigger": "The specific exit/adjust rule, per constraint #3",
      "confidence": "One of: High / Medium / Low",
      "data_status": "One of: 'live' (found real current data), 'partial' (some fields estimated), 'stale' (data may be several hours old or unavailable) -- be honest here"
    }},
    {{ "horizon": "Monthly", ... same fields ... }},
    {{ "horizon": "Quarterly", ... same fields ... }},
    {{ "horizon": "Annual", ... same fields ... }}
  ],
  "aggregate_capital_at_risk_pct": "Sum of all four horizons' max_loss_pct_capital, must be <= {AGGREGATE_CAP_PCT:.0f}",
  "portfolio_view": "One paragraph: are you net long or net short gamma across all four horizons combined, does the combined structure over- or under-hedge overall market gap risk, and does the aggregate capital at risk stay within the {AGGREGATE_CAP_PCT:.0f}% cap"
}}
"""


# -----------------------------
# Parsing (mirrors swing_trade_advisor._parse_analysis_json, different key)
# -----------------------------
def _parse_analysis_json(text):
    """Parses the JSON the LLM was asked to return. Models sometimes still
    wrap it in ```json fences or add stray text despite instructions --
    handle both. Returns (horizons_list, aggregate_pct, portfolio_view), or
    (None, None, None) if nothing usable could be parsed."""
    cleaned = swing._strip_code_fences(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return None, None, None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None, None, None
    if not isinstance(data, dict):
        return None, None, None
    horizons = data.get("horizons")
    if not horizons or not isinstance(horizons, list):
        return None, None, None
    return horizons, data.get("aggregate_capital_at_risk_pct"), data.get("portfolio_view")


# -----------------------------
# HTML rendering
# -----------------------------
_BIAS_STYLE = {
    "bullish": ("#2F5233", "#E7EEE4"),
    "bearish": ("#8B2E2E", "#FBEAEA"),
    "neutral": ("#8A6D3B", "#F3ECDD"),
    "range-bound": ("#8A6D3B", "#F3ECDD"),
    "range bound": ("#8A6D3B", "#F3ECDD"),
}

_CONFIDENCE_STYLE = {
    "high": ("#2F5233", "#E7EEE4"),
    "medium": ("#A6812F", "#FDF3D9"),
    "low": ("#8B2E2E", "#FBEAEA"),
}

_DATA_STATUS_STYLE = {
    "live": ("#2F5233", "#E7EEE4", "Live"),
    "partial": ("#A6812F", "#FDF3D9", "Partial"),
    "stale": ("#8B2E2E", "#FBEAEA", "Stale"),
}


def _esc(v):
    v = "" if v is None else str(v).strip()
    return html.escape(v) if v else "—"


def _badge(text, color, bg):
    return (
        f'<span style="display:inline-block;padding:3px 10px;border-radius:3px;'
        f'font-size:11px;font-weight:700;color:{color};background:{bg};">{html.escape(str(text))}</span>'
    )


def _bias_badge(bias):
    key = str(bias or "").strip().lower()
    color, bg = _BIAS_STYLE.get(key, ("#8A8F9C", "#F4F2ED"))
    return _badge(bias or "—", color, bg)


def _confidence_badge(conf):
    key = str(conf or "").strip().lower()
    color, bg = _CONFIDENCE_STYLE.get(key, ("#8A8F9C", "#F4F2ED"))
    return _badge(conf or "—", color, bg)


def _data_status_badge(status):
    key = str(status or "").strip().lower()
    color, bg, label = _DATA_STATUS_STYLE.get(key, ("#8A8F9C", "#F4F2ED", status or "—"))
    return _badge(label, color, bg)


def _horizon_card_html(h, sans, serif):
    """One risk-defined-strategy card per horizon (Weekly/Monthly/
    Quarterly/Annual), styled consistently with the swing-trade note's
    per-stock parameter tables."""
    name = _esc(h.get("horizon"))
    expiry = _esc(h.get("expiry_date"))

    def row(label, key, value_color="#14213D", bold=False):
        weight = "font-weight:700;" if bold else ""
        return (
            f'<tr><td style="padding:6px 10px;font-size:12px;font-family:{sans};'
            f'color:#4A5063;border-top:1px solid #EDEAE2;width:38%;">{label}</td>'
            f'<td style="padding:6px 10px;font-size:12px;{weight}font-family:{sans};'
            f'color:{value_color};border-top:1px solid #EDEAE2;">{_esc(h.get(key))}</td></tr>'
        )

    def raw_row(label, cell_html):
        return (
            f'<tr><td style="padding:6px 10px;font-size:12px;font-family:{sans};'
            f'color:#4A5063;border-top:1px solid #EDEAE2;width:38%;">{label}</td>'
            f'<td style="padding:6px 10px;font-size:12px;font-family:{sans};'
            f'color:#14213D;border-top:1px solid #EDEAE2;">{cell_html or "—"}</td></tr>'
        )

    rows = "".join([
        row("Strategy", "strategy_name", bold=True),
        row("Legs", "legs"),
        raw_row("Directional Bias (Current Week)" if h.get("next_week_bias") else "Directional Bias", _bias_badge(h.get("bias"))),
        *([raw_row("Directional Bias (Next Week)", _bias_badge(h.get("next_week_bias")))] if h.get("next_week_bias") else []),
        row("Bias Rationale", "bias_reason"),
        row("Max Loss", "max_loss", value_color="#8B2E2E", bold=True),
        row("Max Loss (% of horizon capital)", "max_loss_pct_capital", value_color="#8B2E2E"),
        row("Max Profit", "max_profit", value_color="#2F5233", bold=True),
        row("Max Profit (% of horizon capital)", "max_profit_pct_capital", value_color="#2F5233"),
        row("Breakeven", "breakeven"),
        row("Gap Risk", "gap_risk"),
        row("Adjustment / Exit Trigger", "adjustment_trigger"),
        raw_row("Confidence", _confidence_badge(h.get("confidence"))),
        raw_row("Data Freshness", _data_status_badge(h.get("data_status"))),
    ])

    return f"""
<div style="margin-top:18px;border:1px solid #E7E4DC;border-radius:4px;overflow:hidden;">
  <div style="background:#14213D;padding:9px 12px;">
    <span style="font-family:{sans};font-size:12px;font-weight:700;color:#ffffff;">{name}</span>
    <span style="font-family:{sans};font-size:11px;color:#B08D57;margin-left:8px;">Expiry: {expiry}</span>
  </div>
  <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="border-collapse:collapse;">
    {rows}
  </table>
</div>
"""


def render_horizons_html(horizons, aggregate_pct, portfolio_view):
    sans = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
    serif = "Georgia,'Times New Roman',serif"

    # Render in the fixed Weekly -> Monthly -> Quarterly -> Annual order
    # regardless of what order the model returned them in.
    by_name = {}
    for h in horizons:
        key = str(h.get("horizon") or "").strip()
        if key:
            by_name[key] = h

    cards = ""
    for label in HORIZON_ORDER:
        h = by_name.get(label)
        if h:
            cards += _horizon_card_html(h, sans, serif)
        else:
            cards += (
                f'<div style="margin-top:18px;padding:12px;border:1px solid #EDEAE2;'
                f'border-radius:4px;background:#F4F2ED;font-family:{sans};font-size:12px;'
                f'color:#8A8F9C;">{html.escape(label)}: not returned by the model this run.</div>'
            )

    agg_display = _esc(aggregate_pct)
    agg_color = "#14213D"
    try:
        if float(str(aggregate_pct).replace("%", "").strip()) > AGGREGATE_CAP_PCT:
            agg_color = "#8B2E2E"  # flag if the model's own math breached the cap
    except (TypeError, ValueError):
        pass

    portfolio_html = (
        f'<div style="margin-top:16px;padding:12px 14px;background:#FAF9F6;'
        f'border:1px solid #EDEAE2;border-left:3px solid #B08D57;border-radius:4px;">'
        f'<div style="font-family:{sans};font-size:11px;font-weight:700;color:#14213D;'
        f'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">Portfolio View '
        f'&nbsp;&middot;&nbsp; Aggregate Capital at Risk: '
        f'<span style="color:{agg_color};">{agg_display}%</span> '
        f'(cap {AGGREGATE_CAP_PCT:.0f}%)</div>'
        f'<div style="font-family:{sans};font-size:12px;color:#4A5063;line-height:1.65;">{_esc(portfolio_view)}</div>'
        f'</div>'
    )

    return cards + portfolio_html


# -----------------------------
# Email
# -----------------------------
def _live_feed_html(data, sans):
    if not data:
        return ""
    status = data.get("status", "failed")
    style = {
        "ok": ("#2F5233", "#E7EEE4", "Live feed OK"),
        "eod_fallback": ("#8A6D3B", "#F3ECDD", "Live feed down — EOD Bhavcopy used"),
        "partial": ("#A6812F", "#FDF3D9", "Live feed partial"),
        "failed": ("#8B2E2E", "#FBEAEA", "Live feed failed"),
    }.get(status, ("#8A8F9C", "#F4F2ED", status))
    color, bg, label = style

    bits = [f"Spot: {_esc(data.get('spot'))}", f"VIX: {_esc(data.get('vix'))}"]
    for horizon in HORIZON_ORDER:
        snap = data.get("horizons", {}).get(horizon)
        if snap:
            src = " [EOD]" if snap.get("source") else ""
            bits.append(f"{horizon} ({snap['expiry']}){src}: PCR {snap.get('pcr_oi', 'n/a')}, Max Pain {snap.get('max_pain', 'n/a')}")

    notes_html = ""
    if data.get("notes"):
        notes_html = (
            f'<p style="margin:6px 0 0;font-family:{sans};font-size:11px;color:#9AA0AC;">'
            f'{_esc("; ".join(data["notes"]))}</p>'
        )

    return f"""
<div style="margin-bottom:14px;padding:10px 12px;border:1px solid #E7E4DC;border-radius:4px;background:#FAFAF7;">
  <span style="display:inline-block;padding:3px 10px;border-radius:3px;font-size:11px;font-weight:700;color:{color};background:{bg};">{label}</span>
  <span style="font-family:{sans};font-size:10px;color:#8A8F9C;margin-left:8px;">NSE India option-chain API + Yahoo Finance &middot; fetched {_esc(data.get("fetched_at"))}</span>

  <p style="margin:6px 0 0;font-family:{sans};font-size:11px;line-height:1.6;color:#4A5063;">{_esc(" | ".join(bits))}</p>
  {notes_html}
</div>
"""


def build_email_html(horizons_html, today_str, sources, used_live_search, session_label, live_data=None):
    if used_live_search:
        disclaimer = (
            "Generated using a live-web-search-capable model plus a direct NSE/Yahoo data fetch -- see "
            "\"Sources checked\" and the live-feed summary above for what was actually used. Options-chain "
            "levels, IV, and OI figures can still be a few minutes to hours stale, incomplete, or misread by "
            "the model. Verify every strike, premium, and Greek against your broker's live options chain "
            "before placing any order. Not investment advice."
        )
    else:
        disclaimer = (
            "Generated by an LLM with no live web search this run -- see the live-feed summary above for "
            "what the direct NSE/Yahoo fetch did or didn't supply. Any field not covered by that feed is "
            "model output and is NOT verified against a live source. Do not trade off this run without "
            "confirming every figure against a real-time options chain first. Not investment advice."
        )

    sources_html = swing._build_sources_html(sources)
    sans = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
    serif = "Georgia,'Times New Roman',serif"
    live_tag = (
        '<span style="color:#B08D57;">&nbsp;&middot;&nbsp; Live web search used</span>'
        if used_live_search else ""
    )
    live_feed_html = _live_feed_html(live_data, sans)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Nifty Options Strategy Note</title>
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
              <div style="font-family:{sans};font-size:10px;font-weight:700;letter-spacing:0.16em;text-transform:uppercase;color:#B08D57;">Market Intelligence &nbsp;&bull;&nbsp; Derivatives Desk</div>
              <h1 style="margin:8px 0 0;font-family:{serif};font-weight:400;font-size:23px;line-height:1.3;color:#ffffff;letter-spacing:0.01em;">Nifty Options Strategy Note</h1>
              <p style="margin:6px 0 0;font-family:{sans};font-size:12px;color:#B7BEC9;">Weekly &middot; Monthly &middot; Quarterly &middot; Annual &mdash; Risk-Defined Only</p>
            </td>
          </tr>
          <tr>
            <td style="height:3px;line-height:3px;font-size:0;background:linear-gradient(90deg,#B08D57,#D9C393 45%,#B08D57);">&nbsp;</td>
          </tr>
          <tr>
            <td style="padding:16px 28px 4px;" class="email-padding">
              <p style="margin:0;font-family:{sans};font-size:12px;color:#8A8F9C;">Prepared {today_str} at {datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%I:%M %p IST")} &nbsp;&middot;&nbsp; {session_label}{live_tag}</p>
            </td>
          </tr>
          <tr>
            <td style="padding:14px 28px 18px;" class="email-padding">
              {live_feed_html}
              {horizons_html}
              {sources_html}
            </td>
          </tr>
          <tr>
            <td style="padding:16px 28px 22px;border-top:1px solid #EDEAE2;" class="email-padding">
              <p style="margin:0;font-family:{sans};font-size:11px;line-height:1.6;color:#9AA0AC;">{disclaimer}</p>
              <p style="margin:10px 0 0;font-family:{sans};font-size:10px;letter-spacing:0.04em;color:#B9BEC7;">&copy; Portfolio Research Desk</p>
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>
"""


def send_option_strategy_email(html_body):
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
    subject = f"Nifty Options Strategy Note — {main.get_date_with_suffix(now_ist)} · {time_str}"

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
        main.log.info("Nifty options strategy email sent successfully.")
        return True
    except smtplib.SMTPAuthenticationError:
        main.log.error(
            "SMTP Authentication Error: check EMAIL_FROM/EMAIL_PASSWORD "
            "(use a Gmail App Password, not the account password)."
        )
    except Exception as e:
        main.log.error(f"Failed to send Nifty options strategy email: {e}")
        traceback.print_exc()
    return False


# -----------------------------
# Entry point
# -----------------------------
def run():
    today_str = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%d %B %Y")
    session_label, _in_session = _market_session_label()

    live_data = fetch_live_market_data()
    main.log.info(
        f"Live data feed status: {live_data['status']}"
        + (f" -- {'; '.join(live_data['notes'])}" if live_data["notes"] else "")
    )
    prompt = build_prompt(live_data)

    # Reuses swing_trade_advisor's Groq(compound/compound-mini) -> Tavily+Groq
    # -> Gemini+Search -> plain Groq -> local model chain unmodified -- this
    # function is generic over the prompt text, so nothing options-specific
    # needs duplicating here.
    analysis, sources, used_live_search = swing.generate_analysis(prompt)
    if not analysis:
        main.log.error(
            "No LLM backend produced output (no GROQ_API_KEY/GOOGLE_API_KEY set "
            "and local model unavailable/failed). Aborting without sending an email."
        )
        sys.exit(1)

    live_feed_ok = live_data.get("status") in ("ok", "eod_fallback", "partial") and (
        live_data.get("spot") or live_data.get("horizons")
    )
    if not used_live_search and not live_feed_ok and os.getenv("REQUIRE_LIVE_DATA", "true").lower() == "true":
        # Neither our own direct NSE/Yahoo fetch nor the LLM's own live search
        # produced anything -- without one of those this is pure training-data
        # reasoning about option strikes/IV/OI, which is exactly the stale,
        # unverified output this run exists to avoid.
        main.log.error(
            "Neither the direct NSE/Yahoo live data fetch nor the LLM's own live "
            "web search succeeded this run, so option-chain levels, IV, VIX, and "
            "OI figures would only reflect stale training-data -- not current "
            "strikes. Aborting without sending an email. Set REQUIRE_LIVE_DATA="
            "false to override and allow a clearly-labeled stale-data email instead."
        )
        sys.exit(1)

    horizons, aggregate_pct, portfolio_view = _parse_analysis_json(analysis)
    if horizons:
        horizons_html = render_horizons_html(horizons, aggregate_pct, portfolio_view)
    else:
        main.log.error(
            "Could not parse JSON from LLM output; falling back to raw text display."
        )
        sans = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
        horizons_html = (
            f'<div style="font-family:{sans};font-size:12px;color:#8B2E2E;margin-bottom:8px;">'
            f"Note: the model's response could not be parsed as structured data; showing raw output below.</div>"
            f'<pre style="white-space:pre-wrap;font-family:{sans};font-size:12px;color:#14213D;">{html.escape(swing._strip_code_fences(analysis))}</pre>'
        )

    email_html = build_email_html(horizons_html, today_str, sources, used_live_search, session_label, live_data)

    if os.getenv("DRY_RUN", "false").lower() == "true":
        with open("option_strategy_report.html", "w") as f:
            f.write(email_html)
        main.log.info("DRY_RUN enabled -- wrote option_strategy_report.html instead of emailing.")
        return

    send_option_strategy_email(email_html)


if __name__ == "__main__":
    run()

# -----------------------------
# Real-time grounding status -- see swing_trade_advisor.py's own footer
# comment block for the full detail on each fallback tier (groq/compound,
# groq/compound-mini, Tavily+plain-Groq, Gemini+Google Search grounding,
# plain Groq, local Qwen2.5-1.5B). This module reuses that chain as-is via
# swing.generate_analysis(prompt) rather than re-implementing it.
# -----------------------------