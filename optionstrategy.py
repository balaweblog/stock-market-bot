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
import math
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


# A NIFTY index-options "Annual" horizon is only meaningful if the farthest
# listed expiry is genuinely long-dated AND has real open interest -- NSE
# rarely lists a liquid 12-month-out index option, so naively using
# dts[-1] regardless of how far out (or how thin) it is produces a
# strategy recommendation for an expiry nobody can actually trade at size.
ANNUAL_MIN_DAYS_OUT = int(os.getenv("ANNUAL_MIN_DAYS_OUT", "300"))
ANNUAL_MIN_TOTAL_OI = int(os.getenv("ANNUAL_MIN_TOTAL_OI", "500"))


def _annual_usability(weekly_dt, annual_dt, total_oi=None):
    """Returns (usable: bool, reason: str). Checked independently of
    whatever the LLM says -- if this comes back False, the Annual horizon
    is rendered as omitted regardless of model output (see run())."""
    if not weekly_dt or not annual_dt:
        return False, "No expiries available to evaluate."
    days_out = (annual_dt - weekly_dt).days
    if days_out < ANNUAL_MIN_DAYS_OUT:
        return False, (
            f"Farthest listed expiry is only {days_out} days out (< {ANNUAL_MIN_DAYS_OUT} "
            f"required for a genuine annual/LEAPS-style horizon) -- NSE does not currently "
            f"list a sufficiently long-dated Nifty options expiry."
        )
    if total_oi is not None and total_oi < ANNUAL_MIN_TOTAL_OI:
        return False, (
            f"Farthest listed expiry ({days_out} days out) has only {total_oi:,} combined "
            f"OI -- too illiquid to trade at any meaningful size."
        )
    return True, f"Farthest listed expiry is {days_out} days out with adequate OI."


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

    # Full strike->LTP maps (not just the top-5-by-OI subset above). This is
    # what makes it possible to CALCULATE max profit/loss/breakeven for a
    # chosen spread in code from real premiums, instead of asking the LLM to
    # invent those numbers -- see compute_spread_payoff() below.
    call_ltp = {s: c.get("lastPrice") for s, c in calls.items() if c.get("lastPrice") is not None}
    put_ltp = {s: p.get("lastPrice") for s, p in puts.items() if p.get("lastPrice") is not None}
    # Per-strike IV, straight from NSE's own chain -- this is what makes
    # Greeks/POP/expected-move computable in code instead of asked of the
    # LLM. Only present on the live JSON path; EOD Bhavcopy has no IV field.
    call_iv = {s: c.get("impliedVolatility") for s, c in calls.items() if c.get("impliedVolatility")}
    put_iv = {s: p.get("impliedVolatility") for s, p in puts.items() if p.get("impliedVolatility")}

    return {
        "expiry": expiry_str,
        "pcr_oi": pcr_oi,
        "max_pain": max_pain,
        "total_call_oi": total_call_oi,
        "total_put_oi": total_put_oi,
        "top_call_oi": [(s, c.get("openInterest", 0)) for s, c in top_calls],
        "top_put_oi": [(s, p.get("openInterest", 0)) for s, p in top_puts],
        "call_ltp": call_ltp,
        "put_ltp": put_ltp,
        "call_iv": call_iv,
        "put_iv": put_iv,
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
    call_ltp, put_ltp = {}, {}
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
        try:
            settle_px = float(r.get("ClsPric") or r.get("SttlmPric") or 0) or None
        except (ValueError, TypeError):
            settle_px = None
        if opt_type == "CE":
            calls[strike] = oi
            if settle_px:
                call_ltp[strike] = settle_px
        elif opt_type == "PE":
            puts[strike] = oi
            if settle_px:
                put_ltp[strike] = settle_px

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
        # EOD close/settlement price used as an LTP proxy -- less reliable
        # than a live quote, so callers should treat these premiums as
        # data_status='partial' at best (mirrors the OI figures' own caveat).
        "call_ltp": call_ltp,
        "put_ltp": put_ltp,
        # NSE's Bhavcopy has no implied-vol column -- Greeks/POP/expected-move
        # are simply unavailable ("n/a") for any horizon filled from this
        # fallback path, and are labeled as such rather than estimated.
        "call_iv": {},
        "put_iv": {},
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

        expiry_dts = {
            _parse_bhavcopy_date(r.get("XpryDt", ""))
            for r in bhav_rows
            if r.get("TckrSymb", "").strip() == symbol and r.get("FinInstrmTp", "").strip() == "IDO"
        }
        horizon_dts = _pick_horizon_expiry_dates(expiry_dts)
        for horizon, dt in horizon_dts.items():
            snap = _extract_bhavcopy_snapshot(bhav_rows, symbol, dt)
            snap["source"] = f"EOD Bhavcopy ({bhav_date.strftime('%d-%b-%Y')})"
            snap["expected_move"] = compute_expected_move(snap["call_ltp"], snap["put_ltp"], data["spot"])
            data["horizons"][horizon] = snap

        annual_snap = data["horizons"].get("Annual")
        annual_total_oi = (
            (annual_snap.get("total_call_oi", 0) + annual_snap.get("total_put_oi", 0))
            if annual_snap else None
        )
        usable, reason = _annual_usability(
            horizon_dts.get("Weekly"), horizon_dts.get("Annual"), annual_total_oi
        )
        data["annual_usable"], data["annual_reason"] = usable, reason
        if not usable:
            notes.append(f"Annual horizon not usable: {reason}")

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
        "annual_usable": None,   # None = not yet evaluated (e.g. total fetch failure)
        "annual_reason": None,
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
            snap = _extract_expiry_snapshot(rows, expiry)
            snap["expected_move"] = compute_expected_move(snap["call_ltp"], snap["put_ltp"], data["spot"])
            data["horizons"][horizon] = snap

        by_dt = {_parse_nse_date(d): d for d in records.get("expiryDates", [])}
        horizon_dts = _pick_horizon_expiry_dates(by_dt.keys())
        annual_snap = data["horizons"].get("Annual")
        annual_total_oi = (
            (annual_snap.get("total_call_oi", 0) + annual_snap.get("total_put_oi", 0))
            if annual_snap else None
        )
        usable, reason = _annual_usability(
            horizon_dts.get("Weekly"), horizon_dts.get("Annual"), annual_total_oi
        )
        data["annual_usable"], data["annual_reason"] = usable, reason
        if not usable:
            notes.append(f"Annual horizon not usable: {reason}")
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
        exp_move = snap.get("expected_move")
        if exp_move:
            lines.append(
                f"    Expected move (ATM straddle at {exp_move['atm_strike']:g}): "
                f"±{exp_move['expected_move_pts']:g} pts (~{exp_move.get('expected_move_pct', 'n/a')}% of spot) "
                f"by expiry -- use this as the 1-SD band constraint #2 asks you to keep short strikes outside of."
            )
        call_ltp, put_ltp = snap.get("call_ltp") or {}, snap.get("put_ltp") or {}
        call_iv, put_iv = snap.get("call_iv") or {}, snap.get("put_iv") or {}
        if call_ltp or put_ltp:
            top_call_strikes = [s for s, _ in snap.get("top_call_oi", [])]
            top_put_strikes = [s for s, _ in snap.get("top_put_oi", [])]
            ce_px = "; ".join(f"{s}: ₹{call_ltp[s]:.2f}" for s in top_call_strikes if s in call_ltp) or "n/a"
            pe_px = "; ".join(f"{s}: ₹{put_ltp[s]:.2f}" for s in top_put_strikes if s in put_ltp) or "n/a"
            lines.append(f"    Live CE premiums (LTP) at top-OI strikes: {ce_px}")
            lines.append(f"    Live PE premiums (LTP) at top-OI strikes: {pe_px}")
            if call_iv or put_iv:
                ce_iv = "; ".join(f"{s}: {call_iv[s]:.1f}%" for s in top_call_strikes if s in call_iv) or "n/a"
                pe_iv = "; ".join(f"{s}: {put_iv[s]:.1f}%" for s in top_put_strikes if s in put_iv) or "n/a"
                lines.append(f"    Live IV at top-OI strikes -- CE: {ce_iv} | PE: {pe_iv}")
            lines.append(
                "    NOTE: only pick strikes for this horizon's legs from strikes that appear "
                "above (they have both real OI and a real fetched premium) -- max profit/loss/"
                "breakeven, Greeks (delta/theta/vega/gamma), probability of profit, and margin/ROM "
                "will all be CALCULATED FROM THESE REAL PREMIUMS/IVs by code after you respond, not "
                "from any number you write, so do not bother computing them precisely yourself. In "
                "'strike_rationale' (schema below), just describe qualitatively why you picked each "
                "short strike (e.g. 'outside the expected-move band above' or 'near the far edge of "
                "top OI concentration') -- do not state a delta value or POP number, since those are "
                "verified figures the code will compute and display separately."
            )

    if data.get("annual_usable") is False:
        lines.append(
            f"- ANNUAL HORIZON: NOT USABLE this run ({data.get('annual_reason')}). Do not invent an "
            f"annual strategy -- set strategy_name to 'N/A' and bias_reason to explain why, for the "
            f"Annual horizon object only."
        )

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
# Strategy payoff verification (real math, not LLM arithmetic)
# -----------------------------
# The model is asked (see build_prompt below) to pick a strategy NAME and
# LEGS (strikes/CE-PE/buy-sell) grounded in the live liquidity data it's
# given. It is deliberately NOT trusted to compute max profit, max loss,
# breakeven, or gap risk itself -- LLM arithmetic on option payoffs is
# exactly what produced the impossible numbers (max profit > max loss on a
# credit spread, gap risk exceeding the defined max loss, etc.) this
# verification step exists to eliminate. Instead, every leg's real fetched
# premium is looked up and the payoff is computed directly.
#
# NIFTY's lot size is periodically revised by NSE/SEBI (e.g. 50 -> 75 in
# 2025) -- keep this current, or override per-run via env var.
NIFTY_LOT_SIZE = int(os.getenv("NIFTY_LOT_SIZE", "75"))

# Assumed total options-trading capital pool used to convert rupee max loss
# into "% of capital" -- override to match your actual account size.
TOTAL_CAPITAL_INR = float(os.getenv("OPTIONS_TOTAL_CAPITAL_INR", "1000000"))

# -----------------------------
# Quant add-ons: expected move, Greeks, POP, margin/ROM (all computed in
# code from real fetched IV/premiums -- never asked of or trusted from the
# LLM, for the same reason max_loss/max_profit above aren't).
# -----------------------------
# 10Y G-Sec-ish proxy for the risk-free rate used in Black-Scholes. Nifty
# index options are European-style and cash-settled, so plain BS (no
# early-exercise adjustment) is the standard practitioner approximation.
RISK_FREE_RATE = float(os.getenv("OPTIONS_RISK_FREE_RATE", "0.065"))


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _iv_to_frac(iv):
    """NSE reports IV as a percentage (e.g. 14.5 meaning 14.5%); this
    normalizes either a percent or already-fractional input to a fraction."""
    if iv is None:
        return None
    iv = float(iv)
    return iv / 100.0 if iv > 3 else iv


def time_to_expiry_years(expiry_dt, now=None):
    """Years-to-expiry using calendar days/365 to the 15:30 IST close on
    expiry day -- the standard simplification for a desk-level Greeks
    estimate (not a trading-day/252 model). Floored just above zero so
    same-day expiry doesn't blow up the Black-Scholes formulas below."""
    now = now or datetime.now(ZoneInfo("Asia/Kolkata"))
    expiry_close = datetime.combine(expiry_dt.date(), dtime(15, 30), tzinfo=ZoneInfo("Asia/Kolkata"))
    return max((expiry_close - now).total_seconds() / (365.0 * 86400), 1e-6)


def bs_greeks(spot, strike, t_years, iv, opt_type, r=RISK_FREE_RATE):
    """Black-Scholes delta/gamma/theta/vega for one leg, from a real fetched
    per-strike IV -- used only to characterize the STRUCTURE's risk (net
    delta/theta/vega/gamma), never to re-derive a premium the live feed
    already gave us. Returns None if inputs are unusable (missing/zero IV,
    zero time, etc.) rather than a fabricated number.
    theta is per calendar day; vega is per 1 implied-vol point (e.g. IV
    14.5 -> 15.5)."""
    try:
        sigma = _iv_to_frac(iv)
        if spot is None or strike is None or sigma is None or t_years is None:
            return None
        if t_years <= 0 or sigma <= 0 or spot <= 0 or strike <= 0:
            return None
        sqrt_t = math.sqrt(t_years)
        d1 = (math.log(spot / strike) + (r + 0.5 * sigma ** 2) * t_years) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t
        pdf_d1 = _norm_pdf(d1)
        if opt_type == "CE":
            delta = _norm_cdf(d1)
            theta = (
                -(spot * pdf_d1 * sigma) / (2 * sqrt_t)
                - r * strike * math.exp(-r * t_years) * _norm_cdf(d2)
            ) / 365.0
        else:
            delta = _norm_cdf(d1) - 1.0
            theta = (
                -(spot * pdf_d1 * sigma) / (2 * sqrt_t)
                + r * strike * math.exp(-r * t_years) * _norm_cdf(-d2)
            ) / 365.0
        gamma = pdf_d1 / (spot * sigma * sqrt_t)
        vega = spot * pdf_d1 * sqrt_t / 100.0
        return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def compute_expected_move(call_ltp, put_ltp, spot):
    """Expected move to expiry from the ATM straddle -- the standard
    options-desk shorthand: ATM call premium + ATM put premium approximates
    the market-implied ~1-standard-deviation move by expiry. Returns the
    raw straddle-derived figure (no extra multiplier applied) plus the ATM
    strike used, or None if there's no strike with both a live CE and PE
    premium to build a straddle from."""
    strikes = sorted(set(call_ltp or {}) & set(put_ltp or {}))
    if not strikes or spot is None:
        return None
    atm = min(strikes, key=lambda s: abs(s - spot))
    straddle = call_ltp.get(atm, 0) + put_ltp.get(atm, 0)
    return {
        "atm_strike": atm,
        "straddle_premium": round(straddle, 2),
        "expected_move_pts": round(straddle, 2),
        "expected_move_pct": round(straddle / spot * 100, 2) if spot else None,
    }


def compute_pop(spot, t_years, iv, payoff_fn, breakevens, r=RISK_FREE_RATE):
    """Approximate risk-neutral probability of profit at expiry: integrates
    a flat-IV lognormal distribution (the ATM implied vol for this expiry --
    not a full strike-by-strike vol surface) over whichever price regions
    the structure's own piecewise-linear payoff is actually positive in.
    This is a standard desk-level approximation, not a guaranteed outcome --
    realized price paths are not exactly lognormal and IV itself will drift
    before expiry. Returns a percentage, or None if inputs are unusable."""
    sigma = _iv_to_frac(iv)
    if spot is None or sigma is None or t_years is None or t_years <= 0 or sigma <= 0:
        return None

    def cdf_le(K):
        if K <= 0:
            return 0.0
        sqrt_t = math.sqrt(t_years)
        d1 = (math.log(spot / K) + (r + 0.5 * sigma ** 2) * t_years) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t
        return 1.0 - _norm_cdf(d2)  # P(S_T <= K) under the risk-neutral lognormal

    lo_bound, hi_bound = 0.01, spot * 5.0
    bounds = sorted(set([lo_bound] + [round(b, 2) for b in breakevens if b > 0] + [hi_bound]))
    prob_profit = 0.0
    try:
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            mid = (lo + hi) / 2.0
            if payoff_fn(mid) > 0:
                prob_profit += cdf_le(hi) - cdf_le(lo)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    return round(max(0.0, min(1.0, prob_profit)) * 100, 1)


def estimate_margin_and_rom(max_loss_inr, max_profit_inr, priced_legs, lot_size):
    """Approximates margin required for a risk-defined spread and the
    resulting Return on Margin.

    IMPORTANT CAVEAT: this is NOT a real SPAN+exposure margin figure -- NSE
    does not expose a public margin-calculator API, so the true number
    depends on same-day volatility scans only your broker's margin
    calculator has. What IS reliably true for a genuinely risk-defined
    spread is that no broker should require MARGIN GREATER than the
    position's own worst-case loss (that's the entire point of a defined-
    risk structure), so max_loss is used as the floor/ceiling proxy here,
    with a small exposure-style buffer on the short legs so the estimate
    isn't naively low for a spread with a very cheap max_loss. Always
    confirm the exact figure in your broker's margin calculator before
    sizing a position."""
    short_legs = [l for l in priced_legs if l["action"] == "Sell"]
    exposure_buffer = 0.03 * sum(l["strike"] for l in short_legs) * lot_size if short_legs else 0.0
    margin = max(max_loss_inr, exposure_buffer, 1.0)
    rom_pct = round(max_profit_inr / margin * 100, 2) if margin else None
    return round(margin, 0), rom_pct


_LEG_RE = re.compile(r"\b(Buy|Sell)\s+(\d+(?:\.\d+)?)\s*(CE|PE)\b", re.IGNORECASE)


def parse_legs(legs_text):
    """Extracts (action, strike, type) triples from the model's free-text
    'legs' field, e.g. 'Sell 24800 CE, Buy 25000 CE (1 lot)' ->
    [{'action':'Sell','strike':24800.0,'type':'CE'}, {'action':'Buy','strike':25000.0,'type':'CE'}].
    Returns [] if nothing recognizable is found."""
    out = []
    for action, strike, opt_type in _LEG_RE.findall(legs_text or ""):
        out.append({"action": action.capitalize(), "strike": float(strike), "type": opt_type.upper()})
    return out


def _leg_premium(horizon_snap, strike, opt_type):
    table = (horizon_snap or {}).get("call_ltp" if opt_type == "CE" else "put_ltp", {})
    # Strikes may be int or float depending on source; try both.
    return table.get(strike, table.get(int(strike) if float(strike).is_integer() else strike))


def _leg_iv(horizon_snap, strike, opt_type):
    """Same int/float-key lookup as _leg_premium, but against the per-strike
    IV maps -- empty for the EOD Bhavcopy path, since it carries no IV."""
    table = (horizon_snap or {}).get("call_iv" if opt_type == "CE" else "put_iv", {})
    return table.get(strike, table.get(int(strike) if float(strike).is_integer() else strike))


def compute_strategy_payoff(legs_text, horizon_snap, lot_size=NIFTY_LOT_SIZE):
    """Computes max profit, max loss, breakeven(s), and whether the
    structure is genuinely risk-defined -- purely from real fetched
    premiums and standard option-payoff-at-expiry math (no LLM arithmetic).
    Works generically for any combination of same-expiry legs (vertical
    spreads, iron condors, iron butterflies, etc.) because it evaluates the
    piecewise-linear combined payoff directly rather than special-casing
    strategy names.

    Returns a dict with keys: ok (bool), and on success: max_profit,
    max_loss, breakevens (list), net_premium, unbounded_risk (bool -- True
    means constraint #1 was violated, e.g. a naked leg slipped through). On
    failure, 'ok' is False and 'reason' explains why (unparseable legs,
    missing live premium for a leg, etc.) -- callers must show that reason
    to the user rather than fabricating a number.
    """
    legs = parse_legs(legs_text)
    if not legs:
        return {"ok": False, "reason": "Could not parse strikes/legs from the model's output."}

    priced = []
    for leg in legs:
        premium = _leg_premium(horizon_snap, leg["strike"], leg["type"])
        if premium is None:
            return {
                "ok": False,
                "reason": (
                    f"No live premium available for {leg['action']} {leg['strike']:g} {leg['type']} "
                    f"-- cannot verify this leg's payoff against real market prices."
                ),
            }
        priced.append({**leg, "premium": float(premium)})

    strikes = sorted(set(l["strike"] for l in priced))

    def payoff_points(S):
        total = 0.0
        for l in priced:
            intrinsic = max(S - l["strike"], 0) if l["type"] == "CE" else max(l["strike"] - S, 0)
            total += (l["premium"] - intrinsic) if l["action"] == "Sell" else (intrinsic - l["premium"])
        return total

    far_below, far_above = strikes[0] - 5000, strikes[-1] + 5000
    sample_xs = [far_below] + strikes + [far_above]
    sample_ys = [payoff_points(x) for x in sample_xs]

    # Slope beyond the outermost strikes tells us if risk is actually capped.
    slope_left = payoff_points(far_below + 1) - payoff_points(far_below)
    slope_right = payoff_points(far_above) - payoff_points(far_above - 1)
    unbounded_risk = slope_left < -0.01 or slope_right < -0.01

    max_profit_pts = max(sample_ys)
    max_loss_pts = -min(sample_ys)

    breakevens = []
    for i in range(len(sample_xs) - 1):
        x0, x1, y0, y1 = sample_xs[i], sample_xs[i + 1], sample_ys[i], sample_ys[i + 1]
        if y0 == 0:
            breakevens.append(round(x0, 2))
        elif (y0 < 0) != (y1 < 0):
            breakevens.append(round(x0 + (-y0 / (y1 - y0)) * (x1 - x0), 2))

    net_premium = sum((l["premium"] if l["action"] == "Sell" else -l["premium"]) for l in priced)

    return {
        "ok": True,
        "max_profit": round(max(max_profit_pts, 0) * lot_size, 2),
        "max_loss": round(max(max_loss_pts, 0) * lot_size, 2),
        "breakevens": breakevens,
        "net_premium": round(net_premium, 2),
        "unbounded_risk": unbounded_risk,
        "priced_legs": priced,
        "payoff_fn": payoff_points,
        "strikes": strikes,
    }


def classify_structure(priced_legs):
    """Determines the ACTUAL structure from the priced legs' strikes --
    independent of whatever name the model gave it. This is what catches a
    same-strike short call + short put being mislabeled as an 'Iron Condor'
    (that's an Iron Butterfly, or a wide-winged variant of one) instead of
    just trusting the model's strategy_name."""
    calls = [l for l in priced_legs if l["type"] == "CE"]
    puts = [l for l in priced_legs if l["type"] == "PE"]
    short_calls = [l for l in calls if l["action"] == "Sell"]
    long_calls = [l for l in calls if l["action"] == "Buy"]
    short_puts = [l for l in puts if l["action"] == "Sell"]
    long_puts = [l for l in puts if l["action"] == "Buy"]

    if len(priced_legs) == 4 and len(short_calls) == 1 and len(long_calls) == 1 and len(short_puts) == 1 and len(long_puts) == 1:
        sc, lc, sp, lp = short_calls[0], long_calls[0], short_puts[0], long_puts[0]
        if sc["strike"] == sp["strike"]:
            return "Iron Butterfly"
        elif sc["strike"] > sp["strike"]:
            return "Iron Condor"
        else:
            return "Non-standard 4-leg structure (short call strike below short put strike -- verify manually)"

    if len(priced_legs) == 2 and len(calls) == 2 and not puts:
        sc, lc = (short_calls[0], long_calls[0]) if short_calls and long_calls else (None, None)
        if sc and lc:
            return "Bear Call Spread (credit)" if sc["strike"] < lc["strike"] else "Bull Call Spread (debit)"

    if len(priced_legs) == 2 and len(puts) == 2 and not calls:
        sp, lp = (short_puts[0], long_puts[0]) if short_puts and long_puts else (None, None)
        if sp and lp:
            return "Bull Put Spread (credit)" if sp["strike"] > lp["strike"] else "Bear Put Spread (debit)"

    return f"Custom {len(priced_legs)}-leg risk-defined structure"


_STRUCTURE_KEYWORDS = ("iron condor", "iron butterfly", "bull call", "bear call", "bull put", "bear put")


def _label_conflicts(model_name, classified_name):
    """Loose check for whether the model's own strategy_name contradicts
    what the strikes actually form (e.g. model said 'Iron Condor' but the
    strikes form an Iron Butterfly)."""
    m = (model_name or "").lower()
    found_in_model = [kw for kw in _STRUCTURE_KEYWORDS if kw in m]
    if not found_in_model:
        return False
    return not any(kw in classified_name.lower() for kw in found_in_model)


def generate_adjustment_trigger(priced_legs, net_premium):
    """Replaces the model's own adjustment-trigger text with a
    deterministic, standard rule derived from the actual short strikes and
    net premium -- the model's free-text version produced both a
    non-standard metric ('spread width exceeds 10% of credit', which isn't
    a recognized rule) and, separately, a directionally illogical trigger
    (a level already breached by the current spot for a bullish spread).
    Grounding this in the real strikes avoids both failure modes."""
    short_call_strikes = sorted(l["strike"] for l in priced_legs if l["type"] == "CE" and l["action"] == "Sell")
    short_put_strikes = sorted(l["strike"] for l in priced_legs if l["type"] == "PE" and l["action"] == "Sell")

    breach_parts = []
    if short_put_strikes:
        breach_parts.append(f"closes below {min(short_put_strikes):g} (short put strike)")
    if short_call_strikes:
        breach_parts.append(f"closes above {max(short_call_strikes):g} (short call strike)")
    breach_clause = " or ".join(breach_parts) if breach_parts else "either short strike is breached"

    if net_premium >= 0:
        loss_trigger_pts = round(1.5 * abs(net_premium), 1)
        return (
            f"Exit/adjust if Nifty {breach_clause}, or if running loss reaches roughly 1.5x the net "
            f"credit collected (~{loss_trigger_pts:g} pts), whichever comes first -- standard "
            f"rules-of-thumb for a credit structure, not an arbitrary threshold."
        )
    else:
        return (
            f"Exit if running loss reaches roughly 50% of the net premium paid (~{abs(net_premium) * 0.5:.1f} pts), "
            f"or if the original directional thesis hasn't played out by the final fifth of the time "
            f"remaining to expiry -- standard rules for a debit structure."
        )


def apply_verified_payoff(horizon_dict, horizon_snap, spot=None):
    """Mutates horizon_dict in place: recomputes max_loss / max_profit /
    max_loss_pct_capital / max_profit_pct_capital / breakeven / gap_risk /
    adjustment_trigger / net Greeks / probability of profit / margin+ROM
    from real premiums, real IVs, and actual strike structure, discarding
    whatever free-text figures the model wrote for those fields. Always
    adds a 'verification' field describing what happened, so the email
    never silently presents an unverifiable number as if it were checked --
    and never claims 'live' premiums (or live Greeks/POP) when the
    underlying data was actually an EOD Bhavcopy close."""
    horizon_dict["bias_reason"] = _strip_cap_claims(horizon_dict.get("bias_reason"))
    legs_text = horizon_dict.get("legs", "")
    result = compute_strategy_payoff(legs_text, horizon_snap)
    is_eod = bool((horizon_snap or {}).get("source"))  # only set on the Bhavcopy fallback path
    premium_label = "EOD Bhavcopy closing prices (not live)" if is_eod else "live NSE premiums"

    if not result["ok"]:
        horizon_dict["max_loss"] = "Unverified -- do not trade"
        horizon_dict["max_profit"] = "Unverified -- do not trade"
        horizon_dict["max_loss_pct_capital"] = "n/a"
        horizon_dict["max_profit_pct_capital"] = "n/a"
        horizon_dict["breakeven"] = "n/a"
        horizon_dict["gap_risk"] = "n/a"
        horizon_dict["adjustment_trigger"] = "n/a"
        horizon_dict["expected_move"] = "n/a"
        horizon_dict["net_greeks"] = "n/a"
        horizon_dict["probability_of_profit"] = "n/a"
        horizon_dict["margin_required"] = "n/a"
        horizon_dict["return_on_margin"] = "n/a"
        horizon_dict["verification"] = f"⚠ Not verified: {result['reason']}"
        horizon_dict["_verified_max_loss_inr"] = 0.0
        return horizon_dict

    if result["unbounded_risk"]:
        # This should never happen given constraint #1, but the payoff math
        # itself is the backstop if a naked leg slips through anyway.
        horizon_dict["max_loss"] = "UNDEFINED RISK -- reject this trade"
        horizon_dict["max_profit"] = "n/a"
        horizon_dict["max_loss_pct_capital"] = "n/a"
        horizon_dict["max_profit_pct_capital"] = "n/a"
        horizon_dict["breakeven"] = "n/a"
        horizon_dict["gap_risk"] = "Loss is theoretically unbounded -- this violates the risk-defined-only rule."
        horizon_dict["adjustment_trigger"] = "n/a"
        horizon_dict["expected_move"] = "n/a"
        horizon_dict["net_greeks"] = "n/a"
        horizon_dict["probability_of_profit"] = "n/a"
        horizon_dict["margin_required"] = "n/a"
        horizon_dict["return_on_margin"] = "n/a"
        horizon_dict["verification"] = "🛑 Rejected: legs do not form a capped-risk structure (naked exposure detected)."
        horizon_dict["_verified_max_loss_inr"] = float("inf")
        return horizon_dict

    priced_legs = result["priced_legs"]
    classified = classify_structure(priced_legs)
    label_note = ""
    if _label_conflicts(horizon_dict.get("strategy_name"), classified):
        label_note = f" Re-labeled from model's '{horizon_dict.get('strategy_name')}' to match the actual strikes."
        horizon_dict["strategy_name"] = classified
    elif not (horizon_dict.get("strategy_name") or "").strip():
        horizon_dict["strategy_name"] = classified

    max_loss = result["max_loss"]
    max_profit = result["max_profit"]
    net_premium = result["net_premium"]
    be = ", ".join(f"{b:,.2f}" for b in result["breakevens"]) if result["breakevens"] else "n/a"

    # Sanity check: a credit that consumes most of the spread's width isn't
    # impossible (it happens for ATM short straddle-style structures), but
    # it's unusual enough to warrant flagging for a manual premium check
    # rather than presenting it silently as fully verified.
    strikes = sorted(set(l["strike"] for l in priced_legs))
    width = strikes[-1] - strikes[0] if len(strikes) > 1 else 0
    rich_credit_flag = ""
    if width > 0 and net_premium > 0 and (net_premium / width) > 0.5:
        pct_of_width = net_premium / width * 100
        rich_credit_flag = (
            f" ⚠ Net credit is {pct_of_width:.0f}% of the {width:g}-point spread width -- unusually rich; "
            f"double-check these premiums against a live broker terminal before trusting this figure."
        )

    horizon_dict["max_loss"] = f"₹{max_loss:,.0f} per lot ({NIFTY_LOT_SIZE} qty)"
    horizon_dict["max_profit"] = f"₹{max_profit:,.0f} per lot ({NIFTY_LOT_SIZE} qty)"
    horizon_dict["max_loss_pct_capital"] = round(max_loss / TOTAL_CAPITAL_INR * 100, 2)
    horizon_dict["max_profit_pct_capital"] = round(max_profit / TOTAL_CAPITAL_INR * 100, 2)
    horizon_dict["breakeven"] = be
    # Correct, code-derived statement of gap risk: for any genuinely
    # risk-defined structure, loss can never exceed max_loss regardless of
    # how large the gap is -- that's the entire point of a defined-risk
    # spread. No separate, larger "gap risk" figure is possible.
    horizon_dict["gap_risk"] = (
        f"Capped at max loss (₹{max_loss:,.0f}) even on a gap beyond the strikes -- "
        f"this is a defined-risk structure, so gap risk cannot exceed max loss."
    )
    horizon_dict["adjustment_trigger"] = generate_adjustment_trigger(priced_legs, net_premium)

    # --- Expected move (pass-through from the live-data snapshot) ---
    exp_move = (horizon_snap or {}).get("expected_move")
    if exp_move:
        horizon_dict["expected_move"] = (
            f"±{exp_move['expected_move_pts']:g} pts (~{exp_move.get('expected_move_pct', 'n/a')}% of spot) "
            f"by expiry, from the {exp_move['atm_strike']:g} ATM straddle"
        )
    else:
        horizon_dict["expected_move"] = "n/a (no ATM straddle premium available this run)"

    # --- Net Greeks (Black-Scholes, from real fetched per-strike IV) ---
    expiry_dt = None
    try:
        expiry_dt = _parse_nse_date((horizon_snap or {}).get("expiry", ""))
    except (ValueError, TypeError):
        expiry_dt = None
    t_years = time_to_expiry_years(expiry_dt) if expiry_dt else None

    greeks_ok = spot is not None and t_years is not None and not is_eod  # Bhavcopy carries no IV
    net_delta = net_gamma = net_theta = net_vega = 0.0
    if greeks_ok:
        for leg in priced_legs:
            iv = _leg_iv(horizon_snap, leg["strike"], leg["type"])
            g = bs_greeks(spot, leg["strike"], t_years, iv, leg["type"])
            if g is None:
                greeks_ok = False
                break
            sign = 1 if leg["action"] == "Buy" else -1
            net_delta += sign * g["delta"]
            net_gamma += sign * g["gamma"]
            net_theta += sign * g["theta"]
            net_vega += sign * g["vega"]

    if greeks_ok:
        horizon_dict["net_greeks"] = (
            f"Δ {net_delta * NIFTY_LOT_SIZE:+.1f} · Γ {net_gamma * NIFTY_LOT_SIZE:+.3f} · "
            f"Θ ₹{net_theta * NIFTY_LOT_SIZE:+,.0f}/day · Vega ₹{net_vega * NIFTY_LOT_SIZE:+,.0f}/vol pt "
            f"(per lot, from live IV; positive Θ = time decay working in your favor)"
        )
    else:
        greeks_reason = "EOD Bhavcopy has no IV column" if is_eod else "live IV unavailable for one or more legs this run"
        horizon_dict["net_greeks"] = f"n/a ({greeks_reason})"

    # --- Probability of profit (flat-IV lognormal, ATM IV for this expiry) ---
    pop = None
    if not is_eod and spot is not None and t_years is not None and exp_move:
        atm_iv = _leg_iv(horizon_snap, exp_move["atm_strike"], "CE") or _leg_iv(horizon_snap, exp_move["atm_strike"], "PE")
        pop = compute_pop(spot, t_years, atm_iv, result["payoff_fn"], result["breakevens"])
    horizon_dict["probability_of_profit"] = (
        f"~{pop:.0f}% (model: flat ATM IV, lognormal price at expiry -- not a guarantee)"
        if pop is not None else "n/a (requires live IV, unavailable this run)"
    )

    # --- Margin (approximate) and Return on Margin ---
    margin, rom = estimate_margin_and_rom(max_loss, max_profit, priced_legs, NIFTY_LOT_SIZE)
    horizon_dict["margin_required"] = f"~₹{margin:,.0f} per lot (approx. -- confirm exact figure in your broker's margin calculator)"
    horizon_dict["return_on_margin"] = f"~{rom:.1f}%" if rom is not None else "n/a"

    horizon_dict["verification"] = (
        f"✅ Verified from {premium_label}: net {'credit' if net_premium >= 0 else 'debit'} "
        f"of {abs(net_premium):.2f} pts/leg-set.{label_note}{rich_credit_flag}"
    )
    horizon_dict["_verified_max_loss_inr"] = max_loss
    return horizon_dict


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
   - Prefer structures whose short strikes sit outside the current expected-move band (VIX-implied 1 standard deviation) for that horizon.
   - If global cues (GIFT Nifty pre-market, US markets overnight, Asian markets) suggest elevated gap risk right now, explicitly flag it and bias the recommendation toward tighter risk-defined structures or reduced size -- never toward removing a hedge leg.
   - Do NOT compute or state a rupee gap-risk figure yourself -- for a genuinely risk-defined structure, loss on ANY gap size is capped at max loss, and that figure is calculated from real fetched premiums after you respond (see OUTPUT FORMAT).

3. CAPITAL PROTECTION RULES.
   - Max capital at risk per horizon must not exceed {PER_HORIZON_CAP_PCT:.0f}% of a hypothetical total options-trading capital pool.
   - Aggregate capital at risk across all four horizons combined must not exceed {AGGREGATE_CAP_PCT:.0f}%.
   - Do NOT write a stop-loss/adjustment trigger yourself -- a standard one (short-strike breach, or loss reaching a set multiple of credit/premium) is generated in code from your chosen strikes after you respond.
   - Do NOT compute the rupee max loss/profit or the % of capital yourself -- those are calculated from real fetched premiums after you respond (see OUTPUT FORMAT). Focus only on choosing strikes/legs that keep risk within the caps once priced.

4. This is a stateless, any-time request -- today is {today_str}, and this run is being generated during: {session_label}. Produce a full, self-contained recommendation using only live data you can find right now; do not assume access to any prior recommendation.

5. WEEKLY BIAS SEQUENCING.
   - Determine the Current Week's bias (Bullish or Bearish) strictly from the live market data you find (spot/futures trend, OI buildup, PCR, FII/DII flow, global cues) -- never assume a default direction.
   - CORRECT PCR READING: PCR(OI) > 1 means more puts are written than calls, which is generally read as bullish-to-neutral (put writers expect the market to stay above their strike, building support below spot) -- NOT bearish. PCR < 1 leans bearish-to-neutral. Do not state a bias that contradicts the PCR figure you were given without explaining, in bias_reason, exactly why other data overrides it.
   - CORRECT VIX READING: India VIX measures the MAGNITUDE of expected volatility, not direction. A high VIX means bigger expected moves either way (and richer option premiums, favorable for premium-selling structures) -- it does NOT by itself mean bearish. A low VIX (e.g. under ~15) signals calm, range-bound conditions, not bullishness or bearishness. Never write "high/low VIX indicates bullish/bearish trend" -- derive direction only from price/OI/PCR/flow data, and use VIX only for expected-move sizing and premium-selling suitability.
   - Apply a mean-reversion ASSUMPTION (not a factual prediction) for the immediate next weekly expiry: if the Current Week is Bullish, treat the Next Week as Bearish; if the Current Week is Bearish, treat the Next Week as Bullish. Reflect this explicitly in the Weekly horizon's strategy legs and in a dedicated "next_week_bias" field (see schema below) -- e.g. a current-week bull put spread paired with a next-week-aware bear call spread or calendar structure that benefits from the expected reversal. Say plainly in bias_reason that this reversal is a modeling assumption, not a confirmed forecast.
   - If live data shows a clear reason to override this assumption (e.g. a major event landing exactly in the next-week window), say so explicitly in bias_reason rather than silently ignoring the rule.

6. MONTHLY/QUARTERLY BIAS TOWARD SIDEWAYS/RANGE-BOUND. Unless live data shows a strong, well-supported directional catalyst inside that horizon's window, default the Monthly and Quarterly recommendations to range-bound/theta-positive structures (Iron Condor, Iron Butterfly, Calendar/Diagonal Spread) rather than pure directional debit/credit spreads -- these horizons are for harvesting range and time decay, not chasing the weekly directional call.
   - IRON BUTTERFLY DEFINITION: an Iron Butterfly sells a call AND a put at the SAME at-the-money strike (e.g. Sell 24500 CE + Sell 24500 PE), then buys further OTM wings on each side for protection. If the short call and short put strikes differ, that is an Iron Condor, not an Iron Butterfly -- do not mislabel one as the other.

7. NIFTY LIQUIDITY FILTER. Only select strikes with adequate live liquidity for Nifty options -- meaningful open interest, tight bid-ask spread, and strikes close to standard NSE strike intervals. Choose legs ONLY from strikes shown with a live premium in the LIVE DATA FEED above for that horizon; if a theoretically ideal strike isn't listed there, state that explicitly and move to the nearest listed liquid strike instead.

8. RISK-CONTROLLED SYNTHESIS. Every horizon's final recommendation must simultaneously satisfy constraints #1-#7 above -- risk-defined, gap-protected, capital-capped, correctly bias-sequenced (Weekly), correctly range-biased (Monthly/Quarterly), and liquidity-filtered. If any of these pull in conflicting directions for a given horizon (e.g. the liquid strike sits inside the expected-move band), state the trade-off explicitly in that horizon's bias_reason and resolve it in favor of the tighter risk-defined structure, never in favor of higher theoretical reward.

9. ANNUAL HORIZON MAY BE UNAVAILABLE. If the LIVE DATA FEED above marks the Annual horizon as not usable, do not invent a strategy for it -- set its strategy_name to "N/A" and explain why in bias_reason. Never fabricate a long-dated strategy just to fill the field.

ADDITIONAL LIVE DATA TO SEARCH FOR (only for what the LIVE DATA FEED above does not already give you -- do not re-search or contradict anything already provided there; if you cannot find a real current figure, say so for that field rather than inventing one):
- Nifty futures price (weekly/monthly, for basis) -- the feed above gives spot, not futures
- Confirmation/context on options-chain levels for any horizon the feed marked unavailable
- FII/DII cash + index-options net positioning (most recent session) -- not covered by the feed
- GIFT Nifty pre-market level (if available) and prior-session US market close (S&P 500, Nasdaq), for gap risk
- Any major near-term event risk (RBI policy, US Fed meeting, major earnings cluster, budget/election dates) that falls within each horizon's window
- Use ONLY primary/official sources: NSE option chain and Bhavcopy, NSE market statistics, India VIX, NSE FII/DII statistics, GIFT Nifty/SGX derivatives data, RBI's policy calendar, company earnings calendars, and official exchange circulars. Never cite social media (Instagram, YouTube, X/Twitter), influencer "prediction" videos, or unsourced broker marketing content as a basis for any figure or bias.

OUTPUT FORMAT -- respond with ONLY raw JSON matching the schema below, and nothing else (no markdown, no code fences, no commentary before or after). Keep every field to plain text/numbers only (no HTML). NOTE: max_loss, max_profit, the two pct_capital fields, breakeven, gap_risk, net Greeks (delta/theta/vega/gamma), probability of profit, margin required, and return on margin are NOT requested below -- they are all calculated from real fetched premiums/IVs in code after parsing your "legs" field, specifically to avoid the arithmetic errors that come from an LLM inventing option payoff numbers. Your job is only to choose the strikes; be precise and unambiguous in "legs" (format: "Sell 25000 CE, Buy 25200 CE" -- always "Buy"/"Sell", a strike from the live data, and "CE"/"PE"):

{{
  "horizons": [
    {{
      "horizon": "Weekly",
      "expiry_date": "The actual expiry date used, e.g. '24 Jul 2026'",
      "bias": "One of: Bullish / Bearish / Neutral / Range-bound",
      "next_week_bias": "ONLY for the Weekly horizon object: 'Bullish' or 'Bearish', the opposite of 'bias' per constraint #5's mean-reversion rule (or a brief override note if constraint #5 was overridden). Omit or leave empty for Monthly/Quarterly/Annual.",
      "bias_reason": "One or two sentences grounded in the live data you found, internally consistent with the PCR/VIX reading rules in constraint #5",
      "strategy_name": "e.g. 'Bear Call Spread' or 'Iron Condor' (or 'N/A' for Annual if marked not usable)",
      "legs": "Full leg structure using ONLY strikes shown in the live data, e.g. 'Sell 25000 CE, Buy 25200 CE (1 lot)'",
      "strike_rationale": "One sentence, qualitative only (no delta/POP numbers -- those are computed and shown separately): why THIS short strike, e.g. 'outside the expected-move band and beyond the nearest major OI wall'",
      "confidence": "One of: High / Medium / Low",
      "data_status": "One of: 'live' (found real current data), 'partial' (some fields estimated), 'stale' (data may be several hours old or unavailable) -- be honest here"
    }},
    {{ "horizon": "Monthly", ... same fields ... }},
    {{ "horizon": "Quarterly", ... same fields ... }},
    {{ "horizon": "Annual", ... same fields ... }}
  ],
  "portfolio_view": "One paragraph: are you net long or net short gamma across all four horizons combined, and does the combined structure over- or under-hedge overall market gap risk. Do NOT state or estimate the aggregate capital-at-risk percentage or whether it stays within the {AGGREGATE_CAP_PCT:.0f}% cap -- that figure is summed from verified per-horizon numbers in code after you respond, and any claim you make about it here will be discarded."
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
        row("Strike Selection Rationale", "strike_rationale"),
        row("Expected Move (ATM Straddle)", "expected_move"),
        row("Max Loss", "max_loss", value_color="#8B2E2E", bold=True),
        row("Max Loss (% of horizon capital)", "max_loss_pct_capital", value_color="#8B2E2E"),
        row("Max Profit", "max_profit", value_color="#2F5233", bold=True),
        row("Max Profit (% of horizon capital)", "max_profit_pct_capital", value_color="#2F5233"),
        row("Breakeven", "breakeven"),
        row("Probability of Profit", "probability_of_profit"),
        row("Net Greeks (per lot)", "net_greeks"),
        row("Margin Required", "margin_required"),
        row("Return on Margin", "return_on_margin"),
        row("Gap Risk", "gap_risk"),
        row("Adjustment / Exit Trigger", "adjustment_trigger"),
        raw_row("Confidence", _confidence_badge(h.get("confidence"))),
        raw_row("Data Freshness", _data_status_badge(h.get("data_status"))),
        row("Payoff Verification", "verification"),
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


def _strip_cap_claims(text):
    """Removes any sentence where the model asserted or implied cap
    compliance (e.g. 'stays within the 15% aggregate limit') -- the model
    is never in a position to know this correctly since the real aggregate
    is only summed from verified per-horizon numbers AFTER it responds, and
    letting its guess stand next to the real figure is exactly what
    produced the self-contradiction (issue #1)."""
    if not text:
        return text
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    kept = [
        s for s in sentences
        if not (re.search(r"%", s) and re.search(r"\b(cap|limit)\b", s, re.IGNORECASE))
    ]
    return " ".join(kept).strip()


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
    over_cap = False
    try:
        over_cap = float(str(aggregate_pct).replace("%", "").strip()) > AGGREGATE_CAP_PCT
        if over_cap:
            agg_color = "#8B2E2E"
    except (TypeError, ValueError):
        pass

    # This is the single authoritative statement of cap compliance -- always
    # computed from the same verified figure shown in the badge above it, so
    # it can never contradict what's displayed next to it (issue #1).
    verdict = (
        f"⚠ EXCEEDS the {AGGREGATE_CAP_PCT:.0f}% aggregate cap -- reduce position size before entering."
        if over_cap else
        f"✅ Stays within the {AGGREGATE_CAP_PCT:.0f}% aggregate cap."
    )
    portfolio_text = _strip_cap_claims(portfolio_view)
    portfolio_html = (
        f'<div style="margin-top:16px;padding:12px 14px;background:#FAF9F6;'
        f'border:1px solid #EDEAE2;border-left:3px solid #B08D57;border-radius:4px;">'
        f'<div style="font-family:{sans};font-size:11px;font-weight:700;color:#14213D;'
        f'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">Portfolio View '
        f'&nbsp;&middot;&nbsp; Aggregate Capital at Risk: '
        f'<span style="color:{agg_color};">{agg_display}%</span> '
        f'(cap {AGGREGATE_CAP_PCT:.0f}%)</div>'
        f'<div style="font-family:{sans};font-size:12px;color:#4A5063;line-height:1.65;">{_esc(portfolio_text)}</div>'
        f'<div style="font-family:{sans};font-size:12px;font-weight:700;color:{agg_color};margin-top:8px;">{verdict}</div>'
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
# -----------------------------
# Post-processing: enforce Annual omission + real aggregate-risk arithmetic
# -----------------------------
def finalize_horizons(horizons, live_data):
    """Runs every horizon's legs through apply_verified_payoff() (real
    premium-based math, not LLM arithmetic), forcibly overrides the Annual
    horizon with an omitted placeholder if it was flagged unusable
    (regardless of what the model returned -- the model's compliance with
    constraint #9 is not trusted on its own), and recomputes the aggregate
    capital-at-risk as a plain sum of the per-horizon figures instead of
    trusting a self-reported total (this is what issue #8's arithmetic
    mismatch traces back to). Returns (horizons, aggregate_pct, over_cap)."""
    by_name = {str(h.get("horizon") or "").strip(): h for h in horizons}

    if live_data.get("annual_usable") is False and "Annual" in by_name:
        by_name["Annual"] = {
            "horizon": "Annual",
            "expiry_date": "n/a",
            "bias": "N/A",
            "bias_reason": live_data.get("annual_reason") or "Long-dated option chain unavailable.",
            "strategy_name": "N/A -- omitted",
            "legs": "n/a",
            "max_loss": "n/a",
            "max_profit": "n/a",
            "max_loss_pct_capital": "n/a",
            "max_profit_pct_capital": "n/a",
            "breakeven": "n/a",
            "gap_risk": "n/a",
            "adjustment_trigger": "n/a",
            "strike_rationale": "n/a",
            "expected_move": "n/a",
            "net_greeks": "n/a",
            "probability_of_profit": "n/a",
            "margin_required": "n/a",
            "return_on_margin": "n/a",
            "confidence": "n/a",
            "data_status": "unavailable",
            "verification": "Omitted: long-dated option chain unavailable or insufficiently liquid.",
            "_verified_max_loss_inr": 0.0,
        }

    total_at_risk = 0.0
    for name, h in by_name.items():
        if h.get("_verified_max_loss_inr") is not None:
            continue  # Annual placeholder already finalized above
        snap = (live_data.get("horizons") or {}).get(name, {})
        apply_verified_payoff(h, snap, live_data.get("spot"))

    for h in by_name.values():
        v = h.get("_verified_max_loss_inr", 0.0)
        if v not in (None, float("inf")):
            total_at_risk += v

    aggregate_pct = round(total_at_risk / TOTAL_CAPITAL_INR * 100, 2) if TOTAL_CAPITAL_INR else 0.0
    over_cap = aggregate_pct > AGGREGATE_CAP_PCT

    ordered = [by_name[h] for h in HORIZON_ORDER if h in by_name]
    return ordered, aggregate_pct, over_cap


# Only these look like credible primary/official sources for a Nifty
# derivatives report (issue #10) -- everything else (social media,
# influencer "prediction" videos, unsourced broker marketing blogs) is
# dropped before the report is built, regardless of what the model's own
# search turned up.
_OFFICIAL_SOURCE_DOMAINS = (
    "nseindia.com", "nsearchives.nseindia.com", "bseindia.com",
    "rbi.org.in", "sebi.gov.in", "sgx.com", "moneycontrol.com",
    "yahoo.com", "finance.yahoo.com", "reuters.com", "bloomberg.com",
    "livemint.com", "economictimes.indiatimes.com", "business-standard.com",
)
_BLOCKED_SOURCE_HINTS = ("instagram.com", "youtube.com", "youtu.be", "twitter.com", "x.com", "facebook.com", "tiktok.com")


def _filter_sources(sources):
    """Defensive filter over whatever shape swing.generate_analysis()
    returns for 'sources' (list of dicts with a url/link key, or plain
    strings) -- drops social-media and unsourced-video links outright, and
    otherwise passes through anything not explicitly on the block list
    (better to under-filter than to silently hide a legitimate source the
    allowlist above doesn't happen to name)."""
    if not sources:
        return sources
    out = []
    for s in sources:
        url = ""
        if isinstance(s, dict):
            url = str(s.get("url") or s.get("link") or "")
        else:
            url = str(s)
        if any(bad in url.lower() for bad in _BLOCKED_SOURCE_HINTS):
            continue
        out.append(s)
    return out


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

    horizons, _model_aggregate_pct, portfolio_view = _parse_analysis_json(analysis)
    sources = _filter_sources(sources)
    if horizons:
        # The model only picked strikes/bias -- real payoff numbers and the
        # aggregate capital-at-risk are computed here from fetched premiums,
        # never trusted from the model's own arithmetic (see finalize_horizons).
        horizons, aggregate_pct, over_cap = finalize_horizons(horizons, live_data)
        if over_cap:
            main.log.warning(
                f"Computed aggregate capital at risk ({aggregate_pct}%) exceeds the "
                f"{AGGREGATE_CAP_PCT:.0f}% cap -- flagging in the report rather than silently sending."
            )
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