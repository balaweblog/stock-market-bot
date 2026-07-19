"""
optionstrategy.py

Standalone companion to main.py, structured the same way as
swing_trade_advisor.py. Runs a single "recommend the best risk-defined
Nifty options strategy across Weekly / Monthly / Quarterly
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
import time
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

# Absolute lot ceiling per horizon, independent of the PER_HORIZON_CAP_PCT
# math above -- issue #6. A strategy with a small per-lot max loss (tight
# spread, far-OTM strikes) can pass the percentage-of-capital check with a
# large lot count (e.g. floor(5% of 1L / Rs.229) = 21 lots) even though
# that many lots is still impractical: NSE index-option strikes away from
# the money often can't absorb 20+ lots without meaningful slippage, and
# more lots means more margin, more exposure to a gap move, and a bigger
# single order to actually get filled. PER_HORIZON_CAP_PCT stays as the
# primary risk-based cap; this is a separate, independent ceiling on top
# of it, not a replacement.
MAX_LOTS_PER_HORIZON = int(os.getenv("OPTIONS_MAX_LOTS_PER_HORIZON", "5"))

# Quality filters applied in code AFTER pricing (apply_verified_payoff), on
# top of the risk-defined-only rule -- a capped-risk spread can still be a
# bad trade if the reward on offer doesn't justify the capital locked up as
# max loss. Both are checked independently; either failing rejects the
# horizon and triggers the same repair-and-retry pass used for structurally
# invalid legs (see _horizon_rejected / repair_rejected_legs).
#   - MIN_REWARD_RISK_RATIO: max_profit / max_loss must be at least this
#     (0.5 = risking at most 2x what you stand to make).
#   - MIN_CREDIT_WIDTH_PCT: for a credit spread, net credit collected must
#     be at least this % of the strike width -- e.g. a 300-point-wide Bear
#     Call Spread collecting only a few rupees of credit per point of width
#     is technically "risk-defined" but not worth the capital at risk.
#   - MAX_PLAUSIBLE_REWARD_RISK_RATIO: the mirror-image problem. A genuine
#     defined-risk vertical spread's max_profit and max_loss are two pieces
#     of the SAME strike width (max_profit + max_loss == width * lot_size),
#     so a ratio far above ~3-5:1 almost never comes from real, correctly
#     priced legs -- it's a much stronger signal of a premium-extraction,
#     strike-mapping, or buy/sell-direction bug than of a genuinely great
#     trade. This does NOT reject the horizon (the payoff math may well be
#     internally consistent given the premiums it was handed -- e.g. the
#     numbers can sum exactly to width*lot_size and still be implausible),
#     it downgrades the verdict to Caution and asks for a manual premium
#     check rather than letting an unaudited 60:1 print as "Consider".
MIN_REWARD_RISK_RATIO = float(os.getenv("OPTIONS_MIN_REWARD_RISK_RATIO", "0.5"))
MIN_CREDIT_WIDTH_PCT = float(os.getenv("OPTIONS_MIN_CREDIT_WIDTH_PCT", "15"))
MAX_PLAUSIBLE_REWARD_RISK_RATIO = float(os.getenv("OPTIONS_MAX_PLAUSIBLE_REWARD_RISK_RATIO", "5"))
# Below this modeled POP, a reward:risk ratio above the ceiling above is
# read as a genuine (if easily misread) long-shot/"lottery-like" payoff
# rather than a data bug -- see the lottery-vs-bug split in
# apply_verified_payoff, right after POP and EV are both computed.
LOTTERY_POP_THRESHOLD_PCT = float(os.getenv("OPTIONS_LOTTERY_POP_THRESHOLD_PCT", "15"))
# Final tiebreaker once a horizon has cleared every hard Skip/Caution gate
# below: compute_trade_quality_score (0-100, built from EV, R:R, POP,
# and code-scored confidence -- never the model's own self-rating) decides
# Consider vs Neutral. This replaces what used to be a separate, narrower
# "confidence == Low" -> Caution check with one explicit, documented
# threshold, so every verdict traces to a single number a reader can look
# up in the Trade Quality Score row instead of several scattered heuristics.
CONSIDER_QUALITY_THRESHOLD = float(os.getenv("OPTIONS_CONSIDER_QUALITY_THRESHOLD", "75"))

# Hard reject an Iron Condor whose short strike(s) sit INSIDE the horizon's
# own 1-sigma expected move (the ATM-straddle-derived band), rather than
# only flagging it as a confidence/quality ding the way compute_confidence's
# short-strike-inside-expected-move check already did. A short strike inside
# the 1-sigma band is priced richer precisely because it is proportionately
# more likely to be breached before expiry -- for a defined-risk vertical
# that can be a legitimate, deliberate premium-for-risk trade-off the trader
# chose strike-by-strike, but a 4-leg Iron Condor is explicitly sold AS a
# range-bound, "price stays inside this band" bet, so a short strike already
# inside that band is the structure contradicting its own thesis, not just
# a spicier version of it. Scoped to Iron Condor only (an Iron Butterfly's
# shorts are ATM by definition and are excluded from this check, along with
# verticals/other structures) so this doesn't silently re-purpose a
# condor-specific rule into a blanket ban on close-to-the-money premium
# selling. Defaults to on; set OPTIONS_REJECT_IC_SHORT_INSIDE_EM=false to
# fall back to the old flag-only (non-rejecting) behavior.
REJECT_IC_SHORT_INSIDE_EM = os.getenv("OPTIONS_REJECT_IC_SHORT_INSIDE_EM", "true").lower() == "true"

HORIZON_ORDER = ["Weekly", "Monthly", "Quarterly"]

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
    # NSE's edge/WAF checks these Fetch-Metadata headers on top of UA/Referer;
    # a request missing them looks scripted even with a browser UA set.
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Dest": "empty",
    "Connection": "keep-alive",
    "DNT": "1",
}


def _describe_http_error(e):
    """Pulls out the bits that actually explain an NSE fetch failure --
    HTTP status code and a short body snippet for HTTPError, or the
    exception class name for everything else (timeout, connection reset,
    DNS failure, etc.) -- so logs say *why* it failed instead of just
    that it did. A bare `str(e)` on a requests.HTTPError often just says
    "403 Client Error: Forbidden for url: ..." with no body context."""
    resp = getattr(e, "response", None)
    if resp is not None:
        snippet = ""
        try:
            snippet = resp.text[:200].replace("\n", " ")
        except Exception:
            pass
        return f"HTTP {resp.status_code} ({type(e).__name__}){' -- ' + snippet if snippet else ''}"
    return f"{type(e).__name__}: {e}"


def _is_retryable_nse_error(e):
    """Distinguishes a plausibly-transient failure (worth a quick retry)
    from one that means "this IP/session is blocked" (retrying is wasted
    time, since it'll fail the same way every time). requests.Timeout
    (ReadTimeout/ConnectTimeout) is the signature of a WAF silently
    stalling a flagged connection rather than answering -- retrying that
    from the same IP just repeats the same stall. A 5xx HTTPError or a
    raw ConnectionError (reset, refused) is more consistent with a
    genuine transient blip and is worth one retry."""
    if isinstance(e, requests.exceptions.Timeout):
        return False
    resp = getattr(e, "response", None)
    if resp is not None:
        return 500 <= resp.status_code < 600
    return isinstance(e, requests.exceptions.ConnectionError)


def _nse_warm_session(session, timeout, referer_path="/option-chain"):
    """Sets session cookies the way a real browser would: land on the NSE
    homepage first (this is where the anti-bot cookie is actually issued),
    then the specific page whose Referer the API call will send. Hitting
    only the sub-page (as a single warm-up GET) skips the cookie NSE's edge
    sets on the root domain, which is a common cause of a 401/403 on the
    very next API call even with a correct-looking Referer header."""
    session.get("https://www.nseindia.com/", timeout=timeout)
    time.sleep(0.6)  # brief pause -- an instant homepage->API hit itself looks scripted
    session.get(f"https://www.nseindia.com{referer_path}", timeout=timeout)
    time.sleep(0.6)


def fetch_nse_option_chain(symbol="NIFTY", timeout=12, max_attempts=3):
    """Pulls the live, full option chain (all listed expiries, every
    strike's OI/changeInOI/IV/LTP for CE and PE) straight from NSE's
    option-chain JSON endpoint. Returns the parsed dict, or None on any
    failure (network, block, non-200, bad JSON). Retries plausibly-transient
    failures (5xx, connection reset) with backoff, but fails fast on a
    Timeout -- that's the signature of a blocked IP being silently
    stalled by NSE's WAF, and repeating it just burns the run's time
    budget for no chance of a different outcome."""
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            session = requests.Session()
            session.headers.update(_NSE_HEADERS)
            _nse_warm_session(session, timeout, "/option-chain")
            resp = session.get(
                f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}",
                timeout=timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_err = e
            main.log.warning(
                f"NSE option-chain fetch attempt {attempt}/{max_attempts} failed for "
                f"{symbol}: {_describe_http_error(e)}"
            )
            if attempt < max_attempts and _is_retryable_nse_error(e):
                time.sleep(1.5 * attempt)  # linear backoff between attempts
            elif attempt < max_attempts:
                main.log.warning(
                    f"NSE option-chain fetch for {symbol}: error looks like a block "
                    f"rather than a transient blip -- skipping remaining retries."
                )
                break
    main.log.warning(
        f"NSE option-chain fetch failed for {symbol}: {_describe_http_error(last_err)}"
    )
    return None


def _parse_nse_date(d):
    return datetime.strptime(d, "%d-%b-%Y")


def fetch_nse_fii_dii(timeout=12, max_attempts=2):
    """Pulls the latest FII/DII net cash-market trading activity straight
    from NSE's own JSON endpoint -- the same one https://www.nseindia.com
    /reports/fii-dii calls client-side (and the endpoint the widely-used
    nsepython library wraps as nse_fiidii()). Returns
    {"fii_net_cr": float, "dii_net_cr": float, "fii_dii_date": str,
    "fii_dii_source": str}, or None on any failure/unexpected shape.

    This is what lets the Market Data Inputs table show a real ₹ Cr net
    figure with a clean source label, instead of dumping the raw list of
    (title, url) search-result tuples the FII/DII row previously fell back
    to whenever this category came from the LLM's own web search rather
    than a direct fetch."""
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            session = requests.Session()
            session.headers.update(_NSE_HEADERS)
            _nse_warm_session(session, timeout, "/reports/fii-dii")
            resp = session.get("https://www.nseindia.com/api/fiidiiTradeReact", timeout=timeout)
            resp.raise_for_status()
            rows = resp.json()

            fii_net = dii_net = fii_dii_date = None
            for row in rows or []:
                cat = str(row.get("category", "")).upper()
                net = row.get("netValue")
                if net is None:
                    continue
                try:
                    net = float(net)
                except (TypeError, ValueError):
                    continue
                # NSE labels these e.g. "FII/FPI *" and "DII **" -- match on
                # substring rather than an exact string so a trailing
                # asterisk/spacing change upstream doesn't silently break this.
                if "FII" in cat or "FPI" in cat:
                    fii_net = net
                    fii_dii_date = row.get("date") or fii_dii_date
                elif "DII" in cat:
                    dii_net = net
                    fii_dii_date = row.get("date") or fii_dii_date

            if fii_net is None and dii_net is None:
                main.log.warning(
                    "NSE FII/DII fetch succeeded but returned no recognizable "
                    "FII/DII rows -- endpoint shape may have changed."
                )
                return None
            return {
                "fii_net_cr": fii_net,
                "dii_net_cr": dii_net,
                "fii_dii_date": fii_dii_date,
                "fii_dii_source": "NSE FII/DII Trading Activity report (Cash Market, net ₹ Cr)",
            }
        except Exception as e:
            last_err = e
            main.log.warning(
                f"NSE FII/DII fetch attempt {attempt}/{max_attempts} failed: {_describe_http_error(e)}"
            )
            if attempt < max_attempts and _is_retryable_nse_error(e):
                time.sleep(1.5 * attempt)
            elif attempt < max_attempts:
                main.log.warning("NSE FII/DII fetch: error looks like a block -- skipping remaining retries.")
                break
    main.log.warning(f"NSE FII/DII fetch failed: {_describe_http_error(last_err)}")
    return None


def _pick_horizon_expiry_dates(dt_list):
    """Shared horizon-selection logic, operating on plain datetime objects
    so both the live option-chain JSON path (DD-Mon-YYYY strings) and the
    Bhavcopy fallback path (YYYY-MM-DD strings) pick horizons identically:
    nearest weekly / current monthly / nearest Mar-Jun-Sep-Dec quarterly."""
    dts = sorted(set(dt_list))
    if not dts:
        return {}
    weekly_dt = dts[0]
    same_month = [dt for dt in dts if (dt.year, dt.month) == (weekly_dt.year, weekly_dt.month)]
    monthly_dt = same_month[-1] if same_month else weekly_dt
    quarter_candidates = [dt for dt in dts if dt.month in (3, 6, 9, 12) and dt >= monthly_dt]
    quarterly_dt = quarter_candidates[0] if quarter_candidates else dts[-1]
    return {"Weekly": weekly_dt, "Monthly": monthly_dt, "Quarterly": quarterly_dt}


def _pick_horizon_expiries(expiry_dates):
    """Maps NSE's raw list of listed expiry-date STRINGS (live option-chain
    JSON, DD-Mon-YYYY format) onto the three horizon definitions used in the
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
    # % change in OI per strike (NSE's own field), used to give the strike
    # rationale a real figure like "PE OI +18%" instead of a vague "near top
    # OI" (issue #3).
    call_oi_chg_pct = {s: c.get("pchangeinOpenInterest") for s, c in calls.items() if c.get("pchangeinOpenInterest") is not None}
    put_oi_chg_pct = {s: p.get("pchangeinOpenInterest") for s, p in puts.items() if p.get("pchangeinOpenInterest") is not None}
    # Absolute OI change (contracts, vs previous session) and absolute LTP
    # change, both straight from NSE's own chain -- together these are what
    # let compute_oi_trend() below classify Call Writing / Put Writing /
    # Unwinding instead of just showing a bare OI-change percentage (issue #8).
    call_oi_chg_abs = {s: c.get("changeinOpenInterest") for s, c in calls.items() if c.get("changeinOpenInterest") is not None}
    put_oi_chg_abs = {s: p.get("changeinOpenInterest") for s, p in puts.items() if p.get("changeinOpenInterest") is not None}
    call_price_chg = {s: c.get("change") for s, c in calls.items() if c.get("change") is not None}
    put_price_chg = {s: p.get("change") for s, p in puts.items() if p.get("change") is not None}

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
        "call_oi_chg_pct": call_oi_chg_pct,
        "put_oi_chg_pct": put_oi_chg_pct,
        "call_oi_chg_abs": call_oi_chg_abs,
        "put_oi_chg_abs": put_oi_chg_abs,
        "call_price_chg": call_price_chg,
        "put_price_chg": put_price_chg,
    }


def compute_oi_trend(horizon_snap):
    """Classifies each side of the chain (Calls / Puts) into the standard
    OI-buildup categories traders actually look for -- Writing (OI up while
    premium falls: fresh short positions), Buying (OI up while premium
    rises: fresh long positions), or Unwinding (OI down: positions
    closing) -- instead of just showing a bare aggregate OI-change number.

    Both signals are aggregated across every strike in the chain (not just
    the top-5-by-OI subset shown elsewhere), OI-weighted so a large move at
    a high-OI strike counts more than the same move at a thin one:
      - total OI change = sum of each strike's absolute change in OI
      - premium direction = OI-weighted average of each strike's absolute
        LTP change (sign only matters here, not the magnitude)

    Needs NSE's live-JSON 'changeinOpenInterest' and 'change' fields, which
    only exist on the live option-chain path -- the EOD Bhavcopy fallback
    has no previous-session comparison baked into a single day's file, so
    this returns None for that path (honest 'n/a' rather than a guess)."""
    def _classify(oi_chg_map, price_chg_map, oi_map):
        if not oi_chg_map:
            return None
        total_oi_chg = sum(oi_chg_map.values())
        weighted_price_chg = sum(
            price_chg_map.get(s, 0) * oi_map.get(s, 0) for s in price_chg_map
        )
        if total_oi_chg > 0:
            label = "Writing" if weighted_price_chg <= 0 else "Buying"
        elif total_oi_chg < 0:
            label = "Unwinding"
        else:
            label = "Flat"
        return {"total_oi_chg": total_oi_chg, "label": label}

    call_oi_chg_abs = horizon_snap.get("call_oi_chg_abs") or {}
    put_oi_chg_abs = horizon_snap.get("put_oi_chg_abs") or {}
    if not call_oi_chg_abs and not put_oi_chg_abs:
        return None

    calls = _classify(call_oi_chg_abs, horizon_snap.get("call_price_chg") or {}, horizon_snap.get("call_ltp") or {})
    puts = _classify(put_oi_chg_abs, horizon_snap.get("put_price_chg") or {}, horizon_snap.get("put_ltp") or {})
    if not calls and not puts:
        return None

    parts = []
    if calls:
        parts.append(f"Call {calls['label']} ({calls['total_oi_chg']:+,} OI)")
    if puts:
        parts.append(f"Put {puts['label']} ({puts['total_oi_chg']:+,} OI)")

    # A plain-English read of the two sides together, using the same
    # framework NSE terminals use: put writing = bullish support building,
    # call writing = bearish resistance building, unwinding on either side
    # = conviction fading rather than a fresh directional bet.
    read = None
    if calls and puts:
        if calls["label"] == "Writing" and puts["label"] == "Writing":
            read = "both sides writing premium -- range-bound expectation building"
        elif puts["label"] == "Writing" and calls["label"] != "Writing":
            read = "put writing dominant -- bullish support building"
        elif calls["label"] == "Writing" and puts["label"] != "Writing":
            read = "call writing dominant -- bearish resistance building"
        elif calls["label"] == "Unwinding" and puts["label"] == "Unwinding":
            read = "unwinding on both sides -- conviction fading, not a fresh directional bet"

    return " · ".join(parts) + (f" -- {read}" if read else "")


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
#
# NSE has served this same UDiFF filename from more than one archive
# subdomain over time (nsearchives.nseindia.com is the one referenced in
# NSE's own circular/all-reports page; archives.nseindia.com is an older
# host some third-party downloaders still use successfully for the same
# path). Trying both -- in this order -- means a block, DNS hiccup, or
# silent host retirement on one doesn't take down the whole fallback.
_BHAVCOPY_FO_HOSTS = ["nsearchives.nseindia.com", "archives.nseindia.com"]
_BHAVCOPY_FO_PATH = "/content/fo/BhavCopy_NSE_FO_0_0_0_{yyyymmdd}_F_0000.csv.zip"


def _parse_bhavcopy_date(s):
    s = (s or "").strip()
    for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unrecognized Bhavcopy date format: {s!r}")


def fetch_nse_bhavcopy_fo(trade_date, timeout=15, max_attempts_per_host=2):
    """Downloads and parses one day's official NSE F&O UDiFF Bhavcopy (a
    zipped EOD settlement CSV covering every listed F&O instrument) for the
    given date. Returns a list of CSV row dicts, or None on failure (file
    not yet published, weekend/holiday, network block, format change).
    Tries each candidate archive host in turn. Only retries a host when the
    failure looks transient (5xx, connection reset); a Timeout moves
    straight to the next host instead of repeating the same stall, since
    that's what a WAF silently blocking this IP looks like."""
    date_path = _BHAVCOPY_FO_PATH.format(yyyymmdd=trade_date.strftime("%Y%m%d"))
    last_err = None
    for host in _BHAVCOPY_FO_HOSTS:
        url = f"https://{host}{date_path}"
        for attempt in range(1, max_attempts_per_host + 1):
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
                last_err = e
                main.log.warning(
                    f"NSE Bhavcopy fetch failed for {trade_date.date()} via {host} "
                    f"(attempt {attempt}/{max_attempts_per_host}): {_describe_http_error(e)}"
                )
                if attempt < max_attempts_per_host and _is_retryable_nse_error(e):
                    time.sleep(1.0 * attempt)
                else:
                    break  # move on to the next host rather than repeating a non-transient failure
    main.log.warning(
        f"NSE Bhavcopy fetch failed for {trade_date.date()} across all hosts "
        f"({', '.join(_BHAVCOPY_FO_HOSTS)}): {_describe_http_error(last_err)}"
    )
    return None


def fetch_latest_nse_bhavcopy_fo(max_days_back=6):
    """Walks backward from today (skipping the file NSE hasn't published
    yet for "today" before ~7 PM IST) to find the most recent trading
    day's F&O Bhavcopy. Returns (rows, trade_date, skipped_weekdays):
    - rows/trade_date: the parsed CSV rows and date they came from, or
      (None, None) if nothing was found in max_days_back days.
    - skipped_weekdays: dates strictly between "today" and trade_date that
      were Mon-Fri (so a Bhavcopy should plausibly exist) but whose fetch
      failed anyway -- e.g. not yet published, transient block, or a
      market holiday. Weekends are never included here since no Bhavcopy
      is expected for them. Reported in the email so "Bhavcopy dated 16
      Jul" next to "fetched 19 Jul" reads as an explained gap rather than
      an unrelated data inconsistency."""
    now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
    skipped_weekdays = []
    for i in range(max_days_back + 1):
        candidate = now_ist - timedelta(days=i)
        if i == 0 and now_ist.time() < dtime(19, 0):
            continue  # today's file isn't published yet -- skip straight to yesterday
        rows = fetch_nse_bhavcopy_fo(candidate)
        if rows:
            return rows, candidate.date(), skipped_weekdays
        if candidate.weekday() < 5:  # Mon-Fri: a file should plausibly exist
            skipped_weekdays.append(candidate.date())
    return None, None, skipped_weekdays


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
    bhav_rows, bhav_date, skipped_weekdays = fetch_latest_nse_bhavcopy_fo()
    if not bhav_rows:
        notes.append("EOD Bhavcopy fallback also failed or is unavailable (no file found in the last 6 days).")
        return False

    staleness_note = ""
    if skipped_weekdays:
        skipped_str = ", ".join(d.strftime("%d %b %Y") for d in skipped_weekdays)
        staleness_note = (
            f" (more recent weekday Bhavcopy file(s) for {skipped_str} could not be "
            f"fetched this run -- not yet published, a market holiday, or a transient block)"
        )

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
                data["spot_source"] = f"EOD Bhavcopy near-month futures close proxy ({bhav_date.strftime('%d-%b-%Y')})"
                notes.append("Spot figure is a Bhavcopy near-month futures CLOSE proxy, not true cash spot.")

        data["option_chain_source"] = (
            f"EOD Bhavcopy ({bhav_date.strftime('%d-%b-%Y')}) — last trading day's close, not live"
            f"{staleness_note}"
        )

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

        notes.append(
            f"Live NSE option-chain feed was unavailable this run -- per-horizon OI/PCR/max-pain "
            f"were filled from NSE's official EOD Bhavcopy dated {bhav_date.strftime('%d %b %Y')} "
            f"instead (last trading day's CLOSE, not live/intraday){staleness_note}."
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
        "spot_source": None,
        "vix": None,
        "vix_change_pct": None,
        "vix_source": None,
        "iv_rank": None,
        "iv_percentile": None,
        "iv_rank_days": 0,
        "option_chain_source": None,
        "fii_net_cr": None,
        "dii_net_cr": None,
        "fii_dii_date": None,
        "fii_dii_source": None,
        "horizons": {},
        "notes": notes,
    }

    if yf is not None:
        try:
            vix_hist = yf.Ticker("^INDIAVIX").history(period="1y")
            if not vix_hist.empty:
                data["vix"] = round(float(vix_hist["Close"].iloc[-1]), 2)
                data["vix_source"] = "Yahoo Finance (^INDIAVIX)"
                if len(vix_hist) >= 2:
                    prev = float(vix_hist["Close"].iloc[-2])
                    data["vix_change_pct"] = round((data["vix"] - prev) / prev * 100, 2)
                # IV Rank / IV Percentile -- estimated from this same
                # trailing-year VIX series already fetched above, rather
                # than a second API call. See compute_iv_rank_percentile
                # for why this is a VIX-based estimate, not a per-expiry
                # historical-IV rank.
                rank, pct, days_used = compute_iv_rank_percentile(vix_hist["Close"], data["vix"])
                data["iv_rank"] = rank
                data["iv_percentile"] = pct
                data["iv_rank_days"] = days_used
        except Exception as e:
            notes.append(f"Yahoo Finance VIX fetch failed: {e}")
        try:
            spot_hist = yf.Ticker("^NSEI").history(period="1d")
            if not spot_hist.empty:
                data["spot"] = round(float(spot_hist["Close"].iloc[-1]), 2)
                data["spot_source"] = "Yahoo Finance (^NSEI)"
        except Exception as e:
            notes.append(f"Yahoo Finance Nifty spot fetch failed: {e}")
    else:
        notes.append("yfinance not installed -- spot/VIX cross-check skipped (pip install yfinance).")

    try:
        fii_dii = fetch_nse_fii_dii()
        if fii_dii:
            data.update(fii_dii)
        else:
            notes.append(
                "NSE FII/DII trading-activity fetch failed (blocked, rate-limited, or "
                "endpoint shape changed) -- Market Data Inputs will fall back to "
                "search-link attribution for this category, if any turned up."
            )
    except Exception as e:
        notes.append(f"NSE FII/DII trading-activity fetch failed: {_describe_http_error(e)}")

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
            if data["spot"] is not None:
                data["spot_source"] = "NSE option-chain API (underlyingValue)"
        data["option_chain_source"] = f"Live NSE option-chain API (fetched {data['fetched_at']})"
        horizon_expiries = _pick_horizon_expiries(records.get("expiryDates", []))
        rows = records.get("data", [])
        for horizon, expiry in horizon_expiries.items():
            snap = _extract_expiry_snapshot(rows, expiry)
            snap["expected_move"] = compute_expected_move(snap["call_ltp"], snap["put_ltp"], data["spot"])
            data["horizons"][horizon] = snap

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
        f"markets overnight, event risk). Wherever premiums/IVs are listed below, "
        f"pick legs ONLY from those strikes (real OI + real fetched premium) -- "
        f"max profit/loss/breakeven, Greeks, POP, and margin/ROM are all CALCULATED "
        f"FROM THOSE REAL PREMIUMS by code after you respond, not from any number "
        f"you write, so don't compute them yourself. In 'strike_rationale', "
        f"describe qualitatively why you picked each short strike (e.g. 'outside "
        f"the expected-move band' or 'near the top OI concentration') -- no delta "
        f"or POP numbers, since those are verified separately.",
        f"- Nifty 50 spot: {data.get('spot', 'n/a')}",
        "- India VIX: "
        + str(data.get("vix", "n/a"))
        + (f" ({data['vix_change_pct']:+.2f}% vs prior close)" if data.get("vix_change_pct") is not None else ""),
    ]
    if data.get("iv_rank") is not None and data.get("iv_percentile") is not None:
        lines.append(
            f"- IV Rank: {data['iv_rank']:g} / IV Percentile: {data['iv_percentile']:g} "
            f"(estimated from India VIX's own trailing {data.get('iv_rank_days', 0)}-trading-day range -- "
            f"a market-wide proxy, not this specific expiry's historical IV. Higher values mean richer "
            f"premium relative to the last year and favor premium-selling structures (Iron Condor/Butterfly, "
            f"credit spreads); lower values favor debit structures or standing aside on theta-selling. Use "
            f"qualitatively only -- do not restate this figure as if it were a per-strike IV.)"
        )
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


def compute_iv_rank_percentile(vix_series, current_vix):
    """IV Rank and IV Percentile, estimated from India VIX's own trailing
    history -- the standard desk proxy for where implied vol sits when
    real historical per-expiry/per-strike IV isn't available (this file
    only has this run's live option-chain snapshot, not a stored year of
    daily chains to rank the actual expiry's IV against). VIX is a
    single NIFTY-wide 30-day figure rather than any one expiry's specific
    IV, so this is explicitly an ESTIMATE -- labelled as such wherever
    it's shown or passed to the model, the same way the VIX-proxy IV
    already used as a POP fallback elsewhere in this file is labelled.

      IV Rank = where today's VIX sits between the trailing-window low
                and high, as a percentage (100 = today is the highest
                close in the window, 0 = today is the lowest).
      IV Percentile = the percentage of trailing-window trading days
                whose closing VIX was BELOW today's -- more robust to a
                single outlier spike/trough than Rank alone, since Rank
                only looks at the two endpoints.

    Returns (rank, percentile, days_used), all None/0 if there isn't
    enough history (fewer than 20 trading days) to compute either."""
    if vix_series is None or current_vix is None or len(vix_series) < 20:
        return None, None, 0
    lo, hi = float(vix_series.min()), float(vix_series.max())
    days = len(vix_series)
    rank = round((current_vix - lo) / (hi - lo) * 100, 1) if hi > lo else 50.0
    below = int((vix_series < current_vix).sum())
    percentile = round(below / days * 100, 1)
    return rank, percentile, days


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


def _pop_diagnostics(spot, t_years, iv, breakevens):
    """Surfaces the raw inputs behind compute_pop's number, in the units a
    person can actually sanity-check by eye: the flat sigma used (as a %),
    days to expiry, the model-implied 1-sigma move in points, and how many
    sigmas away the NEAREST breakeven sits. This exists because a POP can
    look surprisingly low for a structure that 'looks wide' in absolute
    strike-point terms -- e.g. a several-hundred-point-wide Monthly Iron
    Condor is not actually wide relative to a full month's sigma*sqrt(T)
    move the way the same point-width would be for a Weekly one. Showing
    the sigma multiple directly lets the reader confirm the number is a
    property of the model's inputs (time + IV), not a bug. Returns None if
    inputs are unusable."""
    sigma = _iv_to_frac(iv)
    if spot is None or sigma is None or t_years is None or t_years <= 0 or sigma <= 0 or not breakevens:
        return None
    one_sigma_move_pts = spot * sigma * math.sqrt(t_years)
    if one_sigma_move_pts <= 0:
        return None
    nearest_gap_pts = min(abs(b - spot) for b in breakevens)
    return {
        "iv_pct": round(sigma * 100, 1),
        "days_to_expiry": round(t_years * 365, 1),
        "one_sigma_move_pts": round(one_sigma_move_pts, 0),
        "nearest_breakeven_gap_pts": round(nearest_gap_pts, 0),
        "nearest_breakeven_sigma_mult": round(nearest_gap_pts / one_sigma_move_pts, 2),
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


def compute_touch_probability(spot, t_years, iv, barrier, r=RISK_FREE_RATE):
    """Exact reflection-principle probability that price touches `barrier`
    at ANY point before expiry (not just at expiry), under the same flat-IV
    lognormal assumption as compute_pop. This is the closed-form
    first-passage-time probability for GBM -- not the common desk shortcut
    of "~2x the expiration probability" (that shortcut is itself just an
    approximation of this same formula, historically used because it's
    easier to eyeball off a delta than to compute the exact integral by
    hand). Since we already have a clean flat sigma here, we use the exact
    formula rather than the approximation.

    Returns a percentage, or None if inputs are unusable."""
    sigma = _iv_to_frac(iv)
    if (
        spot is None or sigma is None or t_years is None or t_years <= 0
        or sigma <= 0 or barrier is None or barrier <= 0
    ):
        return None
    try:
        mu = r - 0.5 * sigma ** 2  # drift of log-price under risk-neutral GBM
        a = math.log(barrier / spot)  # log-distance from spot to the barrier
        sqrt_t = math.sqrt(t_years)
        if a > 0:  # barrier above spot -- P(running max reaches it)
            d1 = (mu * t_years - a) / (sigma * sqrt_t)
            d2 = (-mu * t_years - a) / (sigma * sqrt_t)
        elif a < 0:  # barrier below spot -- P(running min reaches it)
            d1 = (a - mu * t_years) / (sigma * sqrt_t)
            d2 = (a + mu * t_years) / (sigma * sqrt_t)
        else:
            return 100.0  # spot is already sitting at the barrier
        prob = _norm_cdf(d1) + math.exp(2 * mu * a / sigma ** 2) * _norm_cdf(d2)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    return round(max(0.0, min(1.0, prob)) * 100, 1)


def compute_expectancy_metrics(pop_pct, max_profit_inr, max_loss_inr):
    """Turns POP + verified max profit/loss into expectancy-style stats that
    catch the classic high-win-rate/negative-expectancy trap credit spreads
    are prone to: a defined-risk spread can show a POP of 75%+ while still
    being a net loser in expectation if the loss leg dwarfs the credit
    collected. A high POP alone doesn't tell you that; these three do.

    - Expected Value (EV): p*max_profit - (1-p)*max_loss, in INR per lot --
      the average per-trade outcome if this exact setup were repeated many
      times at the modeled POP. Negative EV means the trade is a net loser
      in expectation despite a possibly-high win rate.
    - Kelly %: the classic Kelly-criterion optimal-bet fraction for a binary
      win/lose payoff, f* = p - (1-p)/b where b = max_profit/max_loss is the
      payoff odds (win amount per unit risked). Clamped to [0, 100]; a
      negative raw Kelly (shown as 0%) means the modeled edge is negative
      and the criterion says risk nothing. This is a sizing heuristic that
      assumes the binary win/lose model is correct -- not a margin or
      capital-allocation instruction on its own.
    - Sharpe-like expectancy: EV divided by the standard deviation of the
      binary payoff (max_profit w.p. p, -max_loss w.p. 1-p) -- a crude
      reward-per-unit-of-payoff-volatility ratio, analogous to a single-
      trade Sharpe ratio. Higher is better; it penalizes a high-POP trade
      whose rare loss is disproportionately large relative to its typical win.

    All three are only as good as the flat-IV lognormal POP feeding them --
    they inherit the same approximation caveats as compute_pop, not a
    verified statistical edge. Returns a dict with all values None if the
    inputs are unusable, so callers render an 'n/a'/hidden row instead of
    crashing the run."""
    if (
        pop_pct is None or max_profit_inr is None or max_loss_inr is None
        or max_loss_inr <= 0 or not (0 <= pop_pct <= 100)
    ):
        return {"ev_inr": None, "kelly_pct": None, "sharpe_like": None}

    p = pop_pct / 100.0
    q = 1.0 - p
    ev = p * max_profit_inr - q * max_loss_inr

    b = max_profit_inr / max_loss_inr  # payoff odds: win amount per unit risked
    kelly_pct = None
    if b > 0:
        kelly_raw = p - (q / b)
        kelly_pct = round(max(0.0, min(1.0, kelly_raw)) * 100, 1)

    variance = p * (max_profit_inr - ev) ** 2 + q * (-max_loss_inr - ev) ** 2
    std = math.sqrt(variance) if variance > 0 else 0.0
    sharpe_like = round(ev / std, 2) if std > 0 else None

    return {
        "ev_inr": round(ev, 0),
        "kelly_pct": kelly_pct,
        "sharpe_like": sharpe_like,
    }


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


_OUTSIDE_BAND_CLAIM_RE = re.compile(
    r"[.,]?\s*(?:sits?|is|lies?|falls?|positioned|remains?)?\s*"
    r"outside\s+(?:the\s+)?(?:current\s+)?(?:1[\s-]?sd\s+)?expected[\s-]?move(?:\s+band)?",
    re.IGNORECASE,
)


def _scrub_false_band_claim(rationale, any_leg_inside_band):
    """The prompt's own strike_rationale example ('outside the
    expected-move band and beyond the nearest major OI wall') encourages
    the model to make a band-position claim, but constraint #7/#8
    (liquidity filter, trade-offs) can legitimately force a strike that's
    actually inside the band -- and the model sometimes still writes
    'outside the expected move' out of habit. That directly contradicts
    the code-computed '-- inside the 1-SD band (elevated risk)' note that
    gets appended right after it in the same sentence (issue #2/#3). The
    band position is deterministic and computed from real fetched
    premiums, so code wins here. Earlier versions tried to surgically
    delete just the offending clause, but that leaves a dangling
    ungrammatical fragment (e.g. 'Chosen for being, near strong
    resistance') -- arguably as unpolished as the original contradiction.
    So instead the ENTIRE model sentence is discarded and replaced with a
    fixed, grammatical fallback; the real OI/liquidity/band facts are
    still shown in full via the computed addendum appended right after."""
    if not rationale or not any_leg_inside_band:
        return rationale
    if _OUTSIDE_BAND_CLAIM_RE.search(rationale):
        return "Selected primarily on OI/liquidity positioning -- see the computed band status below"
    return rationale


def build_strike_rationale_addendum(priced_legs, horizon_snap, spot):
    """Appends real fetched/computed numbers to the model's qualitative
    strike rationale for EVERY short strike actually chosen: its own OI,
    the OI % change, the expiry's expected move, and the strike's
    Black-Scholes delta (from live IV) -- turning a vague 'near top OI'
    into something like 'Short strike 24000 (PE): OI 6,000,000 (+18%); Δ
    -0.32; expected move ±246; outside the 1-SD band' (issue #2/#3).
    Returns (addendum_text, any_leg_inside_band): addendum_text is "" if
    there isn't enough live data to say anything concrete; any_leg_inside_band
    is True if at least one short leg actually falls inside the computed
    1-SD band, so the caller can scrub a contradictory "outside the
    expected move" claim the model may have written into strike_rationale
    before concatenating the two."""
    if not horizon_snap:
        return "", False

    parts = []
    top_call_oi = horizon_snap.get("top_call_oi") or []
    top_put_oi = horizon_snap.get("top_put_oi") or []
    if top_call_oi:
        parts.append(f"Highest CE OI: {top_call_oi[0][0]:g} ({top_call_oi[0][1]:,} OI)")
    if top_put_oi:
        parts.append(f"Highest PE OI: {top_put_oi[0][0]:g} ({top_put_oi[0][1]:,} OI)")

    oi_by_strike = {"CE": dict(top_call_oi), "PE": dict(top_put_oi)}
    call_chg = horizon_snap.get("call_oi_chg_pct") or {}
    put_chg = horizon_snap.get("put_oi_chg_pct") or {}
    exp_move = horizon_snap.get("expected_move")
    band_lo = band_hi = None
    if exp_move and spot is not None:
        band_lo, band_hi = spot - exp_move["expected_move_pts"], spot + exp_move["expected_move_pts"]
        parts.append(f"Expected move: ±{exp_move['expected_move_pts']:g}")

    is_eod = bool(horizon_snap.get("source"))  # Bhavcopy has no IV -- delta unavailable
    t_years = None
    if not is_eod:
        try:
            t_years = time_to_expiry_years(_parse_nse_date(horizon_snap.get("expiry", "")))
        except (ValueError, TypeError):
            t_years = None

    short_legs = [l for l in (priced_legs or []) if l["action"] == "Sell"]
    any_leg_inside_band = False
    for leg in short_legs:
        strike, opt_type = leg["strike"], leg["type"]
        oi_val = oi_by_strike[opt_type].get(strike, oi_by_strike[opt_type].get(int(strike) if float(strike).is_integer() else strike))
        chg_map = call_chg if opt_type == "CE" else put_chg
        chg = chg_map.get(strike, chg_map.get(int(strike) if float(strike).is_integer() else strike))
        oi_note = f"OI {oi_val:,}" if oi_val is not None else "OI unavailable"
        if chg is not None:
            chg_note = f"{chg:+.0f}% chg"
        elif is_eod:
            chg_note = "OI change unavailable from EOD Bhavcopy"
        else:
            chg_note = "OI change unavailable this run"

        delta_note = (
            "Greeks unavailable (EOD Bhavcopy has no implied volatility)"
            if is_eod else "Greeks unavailable (live IV missing for this strike)"
        )
        if t_years is not None and spot is not None:
            iv = _leg_iv(horizon_snap, strike, opt_type)
            g = bs_greeks(spot, strike, t_years, iv, opt_type)
            if g is not None:
                delta_note = f"Δ {g['delta']:+.2f}"

        band_note = ""
        if band_lo is not None:
            outside = strike < band_lo or strike > band_hi
            if outside:
                band_note = " -- outside the 1-SD band"
            else:
                band_note = " -- inside the 1-SD band (elevated risk)"
                any_leg_inside_band = True

        parts.append(f"Short strike {strike:g} ({opt_type}): {oi_note} ({chg_note}); {delta_note}{band_note}")

    return " | ".join(parts), any_leg_inside_band


_STRIKE_DISTANCE_FACTOR = float(os.getenv("OPTIONS_STRIKE_DISTANCE_FACTOR", "3"))


def _clamp01_100(x):
    return max(0, min(100, x))


def compute_trade_quality_score(priced_legs, horizon_snap, ev_inr, max_loss, pop_pct, reward_risk_ratio, conf_pct, is_eod):
    """Composite 0-100 Trade Quality Score blending six independently-
    interpretable signals, all derived from already-verified fields (real
    premiums/IV, code-computed EV/POP/R:R, code-scored confidence) -- never
    from the model's own self-assessment. Returns (score:int, breakdown,
    penalty_notes) where breakdown is an ordered dict of {component:
    (sub_score 0-100, weight_pct)}, so the report can show both the
    headline number and exactly what drove it (the same "show your work"
    pattern used elsewhere in this file, e.g. compute_confidence's checks
    list), and penalty_notes is a list of strings describing any hard cap
    applied below.

    Weights (sum to 100%):
      Expected Value  30% -- the single most decision-relevant number: a
                              structure with negative EV is a bad trade
                              regardless of how the other five look, so it
                              gets the largest single weight.
      Reward:Risk     20% -- upside on offer per rupee risked.
      Probability of
        Profit        15% -- win-rate alone, without EV context, so it's
                              weighted below EV/R:R rather than driving the
                              score by itself (a 90% POP credit spread with
                              negative EV should NOT score well here).
      Confidence      15% -- reuses compute_confidence's existing
                              code-verified checklist (OI wall alignment,
                              Max Pain, VIX regime, strike-distance sanity)
                              rather than re-deriving it.
      Liquidity       10% -- whether this run had live OI/chain data to
                              price and vet the trade against, vs. a stale
                              EOD Bhavcopy fallback with no OI/IV at all.
      OI Alignment    10% -- specifically whether the SHORT strike(s) sit
                              at a real, current open-interest wall
                              (support/resistance) -- scored independently
                              since it's specific and decision-relevant
                              enough to warrant its own weight rather than
                              being folded silently into Confidence.

    Hard caps (applied AFTER the weighted blend above, not folded into it):
    a linear blend across six components means a single objectively bad
    signal can still be outvoted by three or four merely-decent ones (e.g.
    negative EV dragging its own 30%-weighted sub-score to 0 can still
    leave R:R/POP/Confidence/Liquidity/OI-Alignment averaging high enough
    to clear 50 on the other 70% of the weight). Each condition below
    describes a trade that should never read as "roughly a coin flip or
    better" regardless of how the other components look, so each caps the
    final score at a ceiling rather than just contributing its own share:
      - Negative EV            -> capped at 20.
      - Reward:Risk < 1         -> capped at 40.
      - Probability of Profit
        < 20%                   -> capped at 40.
    These caps are diagnostic, not the primary enforcement mechanism --
    negative EV and a below-floor Reward:Risk already reject the horizon
    outright elsewhere (compute_horizon_recommendation rule 3; the
    MIN_REWARD_RISK_RATIO gate in apply_verified_payoff), and a low-POP
    trade paired with an implausibly high R:R already gets its own
    "Lottery-Like Payoff" Caution verdict. The caps here exist so the raw
    Trade Quality Score number itself stays honest in every case that
    reaches this function, rather than only being right when nothing else
    already caught the problem.
    """
    components = {}

    # EV, normalized as EV / max_loss (edge per rupee risked) -- a
    # scale-free ratio so a Weekly and a Quarterly trade compare on the
    # same footing regardless of their absolute rupee size.
    if ev_inr is not None and max_loss:
        ev_score = _clamp01_100(50 + (ev_inr / max_loss) * 100)
    else:
        ev_score = 0
    components["Expected Value"] = (round(ev_score), 30)

    # Reward:Risk -- MIN_REWARD_RISK_RATIO (0.5 by default) is the bare
    # pass/fail floor already enforced elsewhere in apply_verified_payoff;
    # 1.5 is treated as a strong R:R for a defined-risk Nifty spread and
    # maps to a full 100.
    if reward_risk_ratio is not None and reward_risk_ratio != float("inf"):
        rr_score = _clamp01_100((reward_risk_ratio / 1.5) * 100)
    else:
        rr_score = 0
    components["Reward:Risk"] = (round(rr_score), 20)

    # POP -- already a 0-100 percentage from compute_pop.
    pop_score = _clamp01_100(pop_pct) if pop_pct is not None else 0
    components["Probability of Profit"] = (round(pop_score), 15)

    # Confidence -- reuse the existing verifiable-checks score as-is.
    conf_score = _clamp01_100(conf_pct) if conf_pct is not None else 0
    components["Confidence"] = (round(conf_score), 15)

    # Liquidity -- proxy for real, current tradeable liquidity: live chain
    # with OI data beats live chain without OI beats a stale EOD Bhavcopy
    # close (no OI/IV at all, since that's the degraded-data path).
    top_call_oi = (horizon_snap or {}).get("top_call_oi") or []
    top_put_oi = (horizon_snap or {}).get("top_put_oi") or []
    has_oi_data = bool(top_call_oi or top_put_oi)
    if is_eod:
        liq_score = 30
    elif has_oi_data:
        liq_score = 90
    else:
        liq_score = 55
    components["Liquidity"] = (round(liq_score), 10)

    # OI Alignment -- do the SHORT strikes sit at a real current OI wall
    # (support/resistance)? Same signal compute_confidence checks, but
    # scored on its own here rather than only being one of several inputs
    # folded into the blended Confidence percentage above.
    top_by_type = {
        "CE": top_call_oi[0][0] if top_call_oi else None,
        "PE": top_put_oi[0][0] if top_put_oi else None,
    }
    short_legs = [l for l in (priced_legs or []) if l["action"] == "Sell"]
    if not has_oi_data or not short_legs:
        oi_score = 50  # unknown -- neither rewarded nor penalized
    else:
        aligned = any(top_by_type.get(l["type"]) == l["strike"] for l in short_legs)
        oi_score = 100 if aligned else 35
    components["OI Alignment"] = (round(oi_score), 10)

    total = sum(sub_score * weight / 100 for sub_score, weight in components.values())
    score = round(total)

    # Hard caps -- see docstring. Applied after the weighted blend, on the
    # same already-verified ev_inr/reward_risk_ratio/pop_pct inputs (never
    # re-derived), so each cap traces to exactly one concrete number a
    # reader can check against the report's own EV/R:R/POP fields.
    penalty_notes = []
    if ev_inr is not None and ev_inr < 0 and score > 20:
        score = 20
        penalty_notes.append("Negative EV caps score at 20")
    if (
        reward_risk_ratio is not None
        and reward_risk_ratio != float("inf")
        and reward_risk_ratio < 1
        and score > 40
    ):
        score = 40
        penalty_notes.append("Reward:Risk < 1 caps score at 40")
    if pop_pct is not None and pop_pct < 20 and score > 40:
        score = 40
        penalty_notes.append("POP < 20% caps score at 40")

    return score, components, penalty_notes


def compute_confidence(priced_legs, horizon_snap, breakevens, is_eod, vix, spot=None):
    """Replaces the model's unexplained High/Medium/Low self-rating with a
    percentage score built from checks that are actually verifiable from
    real fetched data -- the same "code wins over the model's own claim"
    principle used for max_loss/Greeks/POP elsewhere in this file
    (issue #7). Returns (label, pct, checks) where checks is a list of
    (passed: bool, reason: str) tuples suitable for a checklist display.

    The label and percentage BAND are driven primarily by the data-quality
    tier this horizon actually had available, per the desk's own scale
    (issue #4 -- a flat "count of 4 unweighted checks passed" score could
    not distinguish a fully-verified live trade from one built on a stale
    EOD Bhavcopy close, and a rejected/unverified trade was showing a
    literal 0%, which read as "scored zero" rather than "never scored"):
      - Live chain with both OI data and IV/VIX available -> 80-95%, "High"
      - EOD Bhavcopy fallback (no live premiums/IV)         -> 65-80%, "Medium"
      - Live chain but missing OI or VIX data (partial)     -> 40-60%, "Low"
    (A rejected/unverified trade never reaches this function at all --
    apply_verified_payoff sets confidence to "Not generated" with no
    percentage before compute_confidence would otherwise be called.)

    Within that band, the secondary checks below (short strike at the top
    OI wall, Max Pain inside the profit zone, calm VIX regime, and strike
    distance vs. the horizon's own expected move -- issue #7) place the
    exact percentage -- they fine-tune confidence, they don't determine
    its ceiling."""
    checks = []

    top_call_oi = horizon_snap.get("top_call_oi") or []
    top_put_oi = horizon_snap.get("top_put_oi") or []
    has_oi_data = bool(top_call_oi or top_put_oi)
    top_by_type = {
        "CE": top_call_oi[0][0] if top_call_oi else None,
        "PE": top_put_oi[0][0] if top_put_oi else None,
    }
    short_legs = [l for l in (priced_legs or []) if l["action"] == "Sell"]
    if has_oi_data:
        oi_aligned = any(top_by_type.get(l["type"]) == l["strike"] for l in short_legs) if short_legs else False
        checks.append((
            oi_aligned,
            "Short strike aligned with the top open-interest wall" if oi_aligned
            else "Short strike is not at the top open-interest wall",
        ))
    else:
        checks.append((False, "Open-interest data unavailable this run"))

    max_pain = horizon_snap.get("max_pain")
    if max_pain is not None and breakevens and len(breakevens) >= 2:
        lo, hi = min(breakevens), max(breakevens)
        max_pain_aligned = lo <= max_pain <= hi
        checks.append((
            max_pain_aligned,
            "Max Pain aligned with the structure's profit zone" if max_pain_aligned
            else "Max Pain sits outside the structure's profit zone",
        ))
    # else: not enough info to call this one either way (e.g. a single-
    # breakeven vertical spread) -- omitted rather than guessed.

    if vix is not None:
        calm = vix < 20
        checks.append((
            calm,
            f"Calm volatility regime (VIX {vix:g} < 20)" if calm
            else f"Elevated volatility (VIX {vix:g} \u2265 20) -- wider realistic price swings",
        ))
    else:
        checks.append((False, "VIX unavailable this run"))

    # Strike-distance sanity check (issue #7): a strike sitting many
    # multiples of the horizon's OWN expected move away from spot (e.g.
    # Quarterly strikes ~1700 pts from a ~24,300 spot) only makes sense if
    # there's a real OI wall out there -- otherwise it's not a defensible
    # pick, just an arbitrary far-OTM strike. Compares against each
    # horizon's own ATM-straddle-derived expected move, so a Quarterly
    # horizon (naturally wider) isn't held to a Weekly-sized band.
    exp_move = horizon_snap.get("expected_move") or {}
    exp_move_pts = exp_move.get("expected_move_pts")
    if spot is not None and exp_move_pts and priced_legs:
        farthest = max(priced_legs, key=lambda l: abs(l["strike"] - spot))
        distance = abs(farthest["strike"] - spot)
        max_allowed = _STRIKE_DISTANCE_FACTOR * exp_move_pts
        oi_wall_strikes = {s for s, _ in top_call_oi} | {s for s, _ in top_put_oi}
        oi_backed = farthest["strike"] in oi_wall_strikes
        multiple = distance / exp_move_pts if exp_move_pts else None
        if distance > max_allowed and not oi_backed:
            checks.append((
                False,
                f"Farthest strike {farthest['strike']:g} is {distance:.0f} pts from spot "
                f"({multiple:.1f}x expected move) with no supporting OI wall found at that strike",
            ))
        else:
            checks.append((
                True,
                f"Farthest strike {farthest['strike']:g} is {distance:.0f} pts from spot "
                f"({multiple:.1f}x expected move)"
                + (" -- backed by a real OI wall" if oi_backed else " -- within a reasonable multiple of expected move"),
            ))

    # Short-strike-inside-expected-move check (issue #7): the block above
    # catches strikes placed too FAR from spot without OI backing, but
    # nothing was catching the opposite problem -- a short strike sitting
    # INSIDE the horizon's own expected-move band, which is what
    # constraint #2 actually asks short strikes to sit outside of. A
    # strike inside the band collects richer premium but is
    # proportionately more likely to be breached before expiry -- that
    # can be a deliberate, legitimate choice (harvesting elevated
    # premium), but it should surface as a flag for review rather than
    # pass silently. This matters most for Monthly/Quarterly, where a
    # strike a few hundred points from spot can look "wide" by strike
    # count alone while still sitting well inside a four-figure expected
    # move -- e.g. a Quarterly ±1115 expected move with short strikes at
    # 24200/24500 against a ~24,650 spot.
    if spot is not None and exp_move_pts and short_legs:
        nearest_short = min(short_legs, key=lambda l: abs(l["strike"] - spot))
        near_distance = abs(nearest_short["strike"] - spot)
        near_multiple = near_distance / exp_move_pts
        inside_band = near_distance < exp_move_pts
        if inside_band:
            checks.append((
                False,
                f"Nearest short strike {nearest_short['strike']:g} is only {near_distance:.0f} pts from "
                f"spot ({near_multiple:.2f}x expected move) -- inside the expected-move band, so this "
                f"trades richer premium for a higher probability of breach before expiry",
            ))
        else:
            checks.append((
                True,
                f"Nearest short strike {nearest_short['strike']:g} is {near_distance:.0f} pts from spot "
                f"({near_multiple:.2f}x expected move) -- outside the expected-move band",
            ))

    if is_eod:
        label, lo_band, hi_band = "Medium", 65, 80
        tier_desc = "EOD Bhavcopy fallback used (no live premiums/IV)"
    elif has_oi_data and vix is not None:
        label, lo_band, hi_band = "High", 80, 95
        tier_desc = "Live option-chain data with OI and VIX available"
    else:
        label, lo_band, hi_band = "Low", 40, 60
        tier_desc = "Live chain used, but missing OI and/or VIX data"

    total = len(checks)
    passed = sum(1 for ok, _ in checks if ok)
    frac = (passed / total) if total else 0.5
    pct = round(lo_band + frac * (hi_band - lo_band))

    checks = [(True, f"Data tier: {tier_desc}")] + checks
    return label, pct, checks


def apply_verified_payoff(horizon_dict, horizon_snap, spot=None, vix=None):
    """Mutates horizon_dict in place: recomputes max_loss / max_profit /
    max_loss_pct_capital / max_profit_pct_capital / breakeven / gap_risk /
    adjustment_trigger / net Greeks / probability of profit / margin+ROM
    from real premiums, real IVs, and actual strike structure, discarding
    whatever free-text figures the model wrote for those fields. Always
    adds a 'verification' field describing what happened, so the email
    never silently presents an unverifiable number as if it were checked --
    and never claims 'live' premiums (or live Greeks/POP) when the
    underlying data was actually an EOD Bhavcopy close."""
    horizon_dict["bias_reason"] = _scrub_pcr_mischaracterization(
        _strip_cap_claims(horizon_dict.get("bias_reason")),
        (horizon_snap or {}).get("pcr_oi"),
    )
    # Also cross-check the 'bias' badge field itself against PCR -- fixes
    # cases like PCR 1.61 rendered as a "Balanced"/"Neutral" badge even
    # when bias_reason's prose was fine (issue #5: the two fields were
    # being validated inconsistently).
    horizon_dict["bias"] = _scrub_bias_pcr_conflict(
        horizon_dict.get("bias"),
        (horizon_snap or {}).get("pcr_oi"),
    )
    # OI trend (Call/Put Writing vs Buying vs Unwinding) -- computed from
    # the chain snapshot alone, so it's set unconditionally here rather
    # than inside the payoff-verification branches below: it doesn't
    # depend on whether the chosen legs price/verify successfully, only on
    # whether the live NSE chain (not EOD Bhavcopy) was available this run.
    horizon_dict["oi_trend"] = compute_oi_trend(horizon_snap or {})
    legs_text = horizon_dict.get("legs", "")
    result = compute_strategy_payoff(legs_text, horizon_snap)
    is_eod = bool((horizon_snap or {}).get("source"))  # only set on the Bhavcopy fallback path
    premium_label = "EOD Bhavcopy closing prices (not live)" if is_eod else "live NSE premiums"
    # Override whatever the model wrote for data_status -- it self-reported
    # "Live" in a run where the direct feed was down and every horizon
    # actually priced off the EOD Bhavcopy fallback, directly contradicting
    # the live-feed-status banner elsewhere in the same email (issue #5).
    # Freshness must come from the same real signal is_eod already uses,
    # never from the model's own claim.
    horizon_dict["data_status"] = "eod" if is_eod else "live"

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
        horizon_dict["probability_of_profit"] = None
        horizon_dict["probability_of_touch"] = None
        horizon_dict["expected_win_rate"] = None
        horizon_dict["expected_value"] = None
        horizon_dict["kelly_pct"] = None
        horizon_dict["expectancy_ratio"] = None
        horizon_dict["reward_risk_ratio"] = None
        horizon_dict["margin_required"] = "n/a"
        horizon_dict["return_on_margin"] = "n/a"
        horizon_dict["capital_efficiency"] = "n/a"
        horizon_dict["confidence"] = "Not generated"
        horizon_dict["confidence_pct"] = None
        horizon_dict["confidence_reasons"] = [(False, f"Trade could not be verified: {result['reason']}")]
        horizon_dict["verification"] = f"⚠ Not verified: {result['reason']}"
        horizon_dict["_verified_max_loss_inr"] = 0.0
        horizon_dict["_net_gamma"] = None
        horizon_dict["_negative_ev"] = False
        horizon_dict["_trade_quality_score"] = None
        horizon_dict["trade_quality_score"] = "n/a"
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
        horizon_dict["probability_of_profit"] = None
        horizon_dict["probability_of_touch"] = None
        horizon_dict["expected_win_rate"] = None
        horizon_dict["expected_value"] = None
        horizon_dict["kelly_pct"] = None
        horizon_dict["expectancy_ratio"] = None
        horizon_dict["reward_risk_ratio"] = None
        horizon_dict["margin_required"] = "n/a"
        horizon_dict["return_on_margin"] = "n/a"
        horizon_dict["capital_efficiency"] = "n/a"
        horizon_dict["confidence"] = "Not generated"
        horizon_dict["confidence_pct"] = None
        horizon_dict["confidence_reasons"] = [(False, "Rejected: legs do not form a capped-risk structure")]
        horizon_dict["verification"] = "🛑 Rejected: legs do not form a capped-risk structure (naked exposure detected)."
        horizon_dict["_verified_max_loss_inr"] = float("inf")
        horizon_dict["_net_gamma"] = None
        horizon_dict["_negative_ev"] = False
        horizon_dict["_trade_quality_score"] = None
        horizon_dict["trade_quality_score"] = "n/a"
        return horizon_dict

    priced_legs = result["priced_legs"]

    rationale_addendum, any_leg_inside_band = build_strike_rationale_addendum(priced_legs, horizon_snap, spot)
    # apply_verified_payoff runs more than once per horizon in the real
    # pipeline (once in finalize_horizons, again in reverify_horizons after
    # the repair pass) and must be idempotent. Reading strike_rationale
    # directly here would read back whatever the PREVIOUS call already
    # appended to it, so a second call re-appends the same addendum onto
    # its own prior output -- "Highest CE OI... Highest PE OI..." showing
    # up twice. Instead, cache the model's original raw text the first
    # time this runs, and always rebuild from that untouched original on
    # every subsequent call.
    if "_raw_strike_rationale" not in horizon_dict:
        horizon_dict["_raw_strike_rationale"] = (horizon_dict.get("strike_rationale") or "").strip()
    existing = _scrub_false_band_claim(horizon_dict["_raw_strike_rationale"], any_leg_inside_band)
    if rationale_addendum:
        horizon_dict["strike_rationale"] = (
            f"{existing} ({rationale_addendum})" if existing else rationale_addendum
        )
    else:
        horizon_dict["strike_rationale"] = existing

    classified = classify_structure(priced_legs)
    label_note = ""
    if _label_conflicts(horizon_dict.get("strategy_name"), classified):
        label_note = f" Re-labeled from model's '{horizon_dict.get('strategy_name')}' to match the actual strikes."
        horizon_dict["strategy_name"] = classified
    elif not (horizon_dict.get("strategy_name") or "").strip():
        horizon_dict["strategy_name"] = classified

    # Hard filter (see REJECT_IC_SHORT_INSIDE_EM): an Iron Condor whose
    # nearer short strike sits inside the horizon's own 1-sigma expected
    # move is rejected outright, the same way an undefined-risk or
    # poor-reward:risk structure is -- not just downgraded to a confidence
    # flag. Checked BEFORE the reward:risk/credit-width gate below so the
    # rejection reason a reader sees is the actual root cause (thesis
    # contradiction) rather than whatever generic reward:risk number falls
    # out of an inside-the-band short strike's inflated credit. Only
    # applies to a genuine Iron Condor (classify_structure's strike-derived
    # label, not the model's own claimed name) -- an Iron Butterfly's
    # ATM shorts are excluded by construction.
    if REJECT_IC_SHORT_INSIDE_EM and classified == "Iron Condor":
        exp_move_for_filter = (horizon_snap or {}).get("expected_move") or {}
        em_pts = exp_move_for_filter.get("expected_move_pts")
        short_legs_for_filter = [l for l in priced_legs if l["action"] == "Sell"]
        if spot is not None and em_pts and short_legs_for_filter:
            nearest_short_for_filter = min(short_legs_for_filter, key=lambda l: abs(l["strike"] - spot))
            near_distance_for_filter = abs(nearest_short_for_filter["strike"] - spot)
            if near_distance_for_filter < em_pts:
                near_multiple_for_filter = near_distance_for_filter / em_pts
                reason_text = (
                    f"Short strike {nearest_short_for_filter['strike']:g} is only "
                    f"{near_distance_for_filter:.0f} pts from spot ({near_multiple_for_filter:.2f}x the "
                    f"±{em_pts:g}-pt 1-sigma expected move) -- an Iron Condor's short strikes must sit "
                    f"outside the expected-move band by construction; this one contradicts its own "
                    f"range-bound thesis."
                )
                horizon_dict["max_loss"] = (
                    f"SHORT STRIKE INSIDE EXPECTED MOVE -- reject this trade "
                    f"({nearest_short_for_filter['strike']:g} is {near_distance_for_filter:.0f} pts from spot, "
                    f"{near_multiple_for_filter:.2f}x expected move)"
                )
                horizon_dict["max_profit"] = "n/a"
                horizon_dict["max_loss_pct_capital"] = "n/a"
                horizon_dict["max_profit_pct_capital"] = "n/a"
                horizon_dict["breakeven"] = ", ".join(f"{b:,.2f}" for b in result["breakevens"]) if result["breakevens"] else "n/a"
                horizon_dict["gap_risk"] = "n/a"
                horizon_dict["adjustment_trigger"] = "n/a"
                horizon_dict["expected_move"] = (
                    f"±{em_pts:g} pts (~{exp_move_for_filter.get('expected_move_pct', 'n/a')}% of spot) "
                    f"by expiry -- short strike sits inside this band"
                )
                horizon_dict["net_greeks"] = "n/a"
                horizon_dict["probability_of_profit"] = None
                horizon_dict["probability_of_touch"] = None
                horizon_dict["expected_win_rate"] = None
                horizon_dict["expected_value"] = None
                horizon_dict["kelly_pct"] = None
                horizon_dict["expectancy_ratio"] = None
                horizon_dict["reward_risk_ratio"] = None
                horizon_dict["margin_required"] = "n/a"
                horizon_dict["return_on_margin"] = "n/a"
                horizon_dict["capital_efficiency"] = "n/a"
                horizon_dict["confidence"] = "Not generated"
                horizon_dict["confidence_pct"] = None
                horizon_dict["confidence_reasons"] = [(False, reason_text)]
                horizon_dict["verification"] = f"🛑 Rejected: {reason_text}"
                horizon_dict["_verified_max_loss_inr"] = 0.0
                horizon_dict["_net_gamma"] = None
                horizon_dict["_negative_ev"] = False
                horizon_dict["_trade_quality_score"] = None
                horizon_dict["trade_quality_score"] = "n/a"
                return horizon_dict

    max_loss = result["max_loss"]
    max_profit = result["max_profit"]
    net_premium = result["net_premium"]
    be = ", ".join(f"{b:,.2f}" for b in result["breakevens"]) if result["breakevens"] else "n/a"

    # Sanity check: a credit that consumes most of the spread's width isn't
    # impossible (it happens for ATM short straddle-style structures), but
    # it's unusual enough to warrant flagging for a manual premium check
    # rather than presenting it silently as fully verified.
    #
    # "Width" here must be the width of whichever side actually determines
    # max_loss -- for a 2-leg vertical, that's simply the distance between
    # its two strikes, so strikes[-1]-strikes[0] happened to work. But for a
    # 4-leg Iron Condor/Butterfly, strikes[-1]-strikes[0] is the OUTER
    # wing-to-wing span (e.g. 1000 pts for wings at 24000/25000), not the
    # width that caps risk (e.g. 300 pts for shorts at 24300/24700) -- using
    # the outer span understated credit_width_pct by 3x+ and could wrongly
    # reject a perfectly good condor. Deriving width from the already-
    # verified max_loss instead (max_loss = width_of_breached_side -
    # net_credit, so width = max_loss + net_credit) is exact for any leg
    # count because it comes from the real payoff evaluation, not strike
    # geometry.
    max_loss_pts = result["max_loss"] / NIFTY_LOT_SIZE if NIFTY_LOT_SIZE else 0
    width = (max_loss_pts + net_premium) if net_premium > 0 else 0
    rich_credit_flag = ""
    if width > 0 and net_premium > 0 and (net_premium / width) > 0.5:
        pct_of_width = net_premium / width * 100
        rich_credit_flag = (
            f" ⚠ Net credit is {pct_of_width:.0f}% of the {width:g}-point spread width -- unusually rich; "
            f"double-check these premiums against a live broker terminal before trusting this figure."
        )

    # Reward-quality filter: a structure can be genuinely risk-defined (max
    # loss is capped) and still be a bad trade if the reward on offer barely
    # justifies the capital locked up -- e.g. a wide credit spread collecting
    # only a sliver of its own width as premium. Checked independently of
    # the naked/unbounded check above; either failing here REJECTS the
    # horizon the same way (feeds into _horizon_rejected -> one repair-and-
    # retry pass -> degrades to a clearly-labeled rejected trade if the
    # model can't fix it), rather than silently presenting a 25:1
    # risk-to-reward spread as if it were a normal recommendation.
    reward_risk_ratio = (max_profit / max_loss) if max_loss > 0 else float("inf")
    credit_width_pct = (net_premium / width * 100) if width > 0 and net_premium > 0 else None
    poor_reward_risk = max_loss > 0 and reward_risk_ratio < MIN_REWARD_RISK_RATIO
    poor_credit_width = credit_width_pct is not None and credit_width_pct < MIN_CREDIT_WIDTH_PCT

    if poor_reward_risk or poor_credit_width:
        reasons = []
        if poor_reward_risk:
            reasons.append(
                f"Reward:Risk is only {reward_risk_ratio:.2f} (₹{max_profit:,.0f} max profit vs "
                f"₹{max_loss:,.0f} max loss per lot) -- below the {MIN_REWARD_RISK_RATIO:g} minimum."
            )
        if poor_credit_width:
            reasons.append(
                f"Net credit is only {credit_width_pct:.1f}% of the {width:g}-point spread width -- "
                f"below the {MIN_CREDIT_WIDTH_PCT:g}% minimum for a credit spread to be worth the capped risk."
            )
        reason_text = " ".join(reasons)
        horizon_dict["max_loss"] = (
            f"POOR REWARD/RISK -- reject this trade (₹{max_loss:,.0f} at risk for only "
            f"₹{max_profit:,.0f} potential, per lot)"
        )
        horizon_dict["max_profit"] = f"₹{max_profit:,.0f} per lot ({NIFTY_LOT_SIZE} qty)"
        horizon_dict["max_loss_pct_capital"] = "n/a"
        horizon_dict["max_profit_pct_capital"] = "n/a"
        horizon_dict["breakeven"] = be
        horizon_dict["gap_risk"] = "n/a"
        horizon_dict["adjustment_trigger"] = "n/a"
        horizon_dict["expected_move"] = "n/a"
        horizon_dict["net_greeks"] = "n/a"
        horizon_dict["probability_of_profit"] = None
        horizon_dict["probability_of_touch"] = None
        horizon_dict["expected_win_rate"] = None
        horizon_dict["expected_value"] = None
        horizon_dict["kelly_pct"] = None
        horizon_dict["expectancy_ratio"] = None
        horizon_dict["reward_risk_ratio"] = (
            f"{reward_risk_ratio:.2f} -- below the {MIN_REWARD_RISK_RATIO:g} minimum (rejected)"
            if max_loss > 0 else None
        )
        horizon_dict["margin_required"] = "n/a"
        horizon_dict["return_on_margin"] = "n/a"
        horizon_dict["capital_efficiency"] = "n/a"
        horizon_dict["confidence"] = "Not generated"
        horizon_dict["confidence_pct"] = None
        horizon_dict["confidence_reasons"] = [(False, reason_text)]
        horizon_dict["verification"] = f"🛑 Rejected: {reason_text}"
        horizon_dict["_verified_max_loss_inr"] = 0.0
        horizon_dict["_net_gamma"] = None
        horizon_dict["_negative_ev"] = False
        horizon_dict["_trade_quality_score"] = None
        horizon_dict["trade_quality_score"] = "n/a"
        return horizon_dict

    horizon_dict["max_loss"] = f"₹{max_loss:,.0f} per lot ({NIFTY_LOT_SIZE} qty)"
    horizon_dict["max_profit"] = f"₹{max_profit:,.0f} per lot ({NIFTY_LOT_SIZE} qty)"
    horizon_dict["max_loss_pct_capital"] = round(max_loss / TOTAL_CAPITAL_INR * 100, 2)
    horizon_dict["max_profit_pct_capital"] = round(max_profit / TOTAL_CAPITAL_INR * 100, 2)
    # Flip side of the MIN_REWARD_RISK_RATIO floor: a ratio far above what a
    # real defined-risk vertical can produce needs a flag, but WHICH flag
    # depends on POP, computed further down this function -- a high ratio
    # with normal/unknown POP is a data-integrity smell (wrong premium,
    # wrong strike/leg mapping, buy/sell flipped); the same high ratio with
    # a genuinely low POP is instead a legitimate but easily-misread
    # long-shot ("lottery-like") payoff profile. Only the raw over-ceiling
    # boolean is recorded here; the actual warning text and verdict
    # framing are decided once POP is known, below.
    horizon_dict["_over_ratio_ceiling"] = (
        max_loss > 0 and reward_risk_ratio > MAX_PLAUSIBLE_REWARD_RISK_RATIO
    )
    # Reward:Risk = max_profit / max_loss, the same ratio already computed
    # above to enforce MIN_REWARD_RISK_RATIO -- surfaced here as its own
    # field rather than left buried in the rejection-only code path, since
    # it's a standard number traders expect to see even for an ACCEPTED
    # trade (not just as a reason a trade got rejected).
    horizon_dict["reward_risk_ratio"] = (
        f"{reward_risk_ratio:.2f} (₹{max_profit:,.0f} potential vs ₹{max_loss:,.0f} at risk, per lot)"
        if max_loss > 0 else "n/a (no capital at risk)"
    )
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
        band = ""
        if spot is not None:
            lo, hi = spot - exp_move["expected_move_pts"], spot + exp_move["expected_move_pts"]
            band = f" -- 68% probability band: {lo:,.0f}–{hi:,.0f}"
        horizon_dict["expected_move"] = (
            f"±{exp_move['expected_move_pts']:g} pts (~{exp_move.get('expected_move_pct', 'n/a')}% of spot) "
            f"by expiry, from the {exp_move['atm_strike']:g} ATM straddle{band}"
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
        horizon_dict["_net_gamma"] = net_gamma * NIFTY_LOT_SIZE
    else:
        greeks_reason = (
            "implied volatility is unavailable in the selected data source (EOD Bhavcopy has no IV column)"
            if is_eod else "implied volatility is unavailable for one or more legs this run"
        )
        horizon_dict["net_greeks"] = f"Greeks unavailable -- {greeks_reason}"
        horizon_dict["_net_gamma"] = None

    # --- Probability of profit (flat-IV lognormal) ---
    # Prefers live per-strike ATM IV. When only the EOD Bhavcopy fallback is
    # available (no IV column on that feed), falls back to India VIX as a
    # flat-IV proxy instead of giving up -- less precise than a real
    # per-expiry IV (VIX is a 30-day NIFTY-wide figure, not this specific
    # expiry's own smile), but it's the same proxy the whole options
    # industry reaches for when a clean per-expiry IV isn't available, and
    # it's clearly labeled as an estimate either way so nobody mistakes it
    # for a live-IV number. This is what actually eliminates the "n/a"
    # that showed on every EOD-fallback run.
    pop = pop_source = None
    effective_iv = None
    if spot is not None and t_years is not None and exp_move:
        atm_iv = _leg_iv(horizon_snap, exp_move["atm_strike"], "CE") or _leg_iv(horizon_snap, exp_move["atm_strike"], "PE")
        if atm_iv and not is_eod:
            effective_iv, pop_source = atm_iv, "live ATM IV"
        elif vix:
            effective_iv, pop_source = vix, "India VIX proxy IV -- approximate"
    if effective_iv:
        pop = compute_pop(spot, t_years, effective_iv, result["payoff_fn"], result["breakevens"])
    pop_diag = _pop_diagnostics(spot, t_years, effective_iv, result["breakevens"]) if pop is not None else None
    diag_note = (
        f"; IV≈{pop_diag['iv_pct']:g}%, T≈{pop_diag['days_to_expiry']:g}d implies a 1σ move of "
        f"±{pop_diag['one_sigma_move_pts']:,.0f}pts, nearest breakeven is only "
        f"{pop_diag['nearest_breakeven_gap_pts']:,.0f}pts away (~{pop_diag['nearest_breakeven_sigma_mult']:g}σ) "
        f"-- a longer horizon needs a proportionally wider structure for the same POP"
        if pop_diag else ""
    )
    horizon_dict["probability_of_profit"] = (
        f"~{pop:.0f}% ({pop_source}, lognormal price at expiry -- not a guarantee{diag_note})"
        if pop is not None else None  # sentinel: render step hides this row rather than printing n/a
    )

    # --- Probability of touch ---
    # Reports the WORST (highest) touch probability across the structure's
    # breakeven level(s) -- i.e. whichever one the underlying is most
    # likely to reach first -- since that's the level that would actually
    # trigger an early adjustment/exit decision, not just the expiry
    # outcome POP already covers.
    pot = None
    if effective_iv and result["breakevens"]:
        touch_probs = [
            p for p in (
                compute_touch_probability(spot, t_years, effective_iv, b)
                for b in result["breakevens"]
            ) if p is not None
        ]
        if touch_probs:
            pot = max(touch_probs)
    horizon_dict["probability_of_touch"] = (
        f"~{pot:.0f}% ({pop_source}) -- chance price touches breakeven at some point "
        f"before expiry, not just at expiry; commonly used as an early-management trigger"
        if pot is not None else None
    )

    # --- Expected win rate ---
    # Industry convention: for a position held to expiry, "win rate" and
    # "probability of profit" are the same figure by definition (share of
    # terminal-price outcomes where payoff > 0). Shown as its own row since
    # reports conventionally list it next to POP, but it intentionally
    # reuses the same computed number rather than inventing a second,
    # different-looking figure with no extra real signal behind it. If the
    # position is actively managed (e.g. closed at 50% of max profit) the
    # realized win rate is typically higher than this -- noted explicitly
    # rather than estimated, since that requires a management-rule backtest
    # this file doesn't have live data to run.
    horizon_dict["expected_win_rate"] = (
        f"~{pop:.0f}% if held to expiry ({pop_source}) -- typically higher in practice "
        f"if the position is actively managed/closed early rather than held to expiry"
        if pop is not None else None
    )

    # --- Expected Value / Kelly % / Sharpe-like expectancy ---
    # A high POP is common with credit spreads and can coexist with a net
    # negative expectancy if the loss leg dwarfs the credit collected --
    # these three numbers make that trade-off explicit rather than leaving
    # a bare POP figure to imply "good trade" on its own.
    expectancy = compute_expectancy_metrics(pop, max_profit, max_loss)
    ev_inr = expectancy["ev_inr"]
    horizon_dict["_negative_ev"] = ev_inr is not None and ev_inr < 0
    horizon_dict["expected_value"] = (
        (
            "❌ Avoid — negative expected value under current assumptions. "
            if ev_inr < 0 else ""
        ) + f"₹{ev_inr:,.0f} per lot ({'positive' if ev_inr >= 0 else 'negative'} expectancy at the "
        f"modeled POP -- avg outcome if this exact setup repeated many times, not a guarantee)"
        if ev_inr is not None else None
    )
    horizon_dict["kelly_pct"] = (
        f"~{expectancy['kelly_pct']:.1f}% of capital (Kelly criterion, floored at 0% -- a sizing "
        f"heuristic assuming the binary win/lose model holds, not a margin recommendation)"
        if expectancy["kelly_pct"] is not None else None
    )
    horizon_dict["expectancy_ratio"] = (
        f"{expectancy['sharpe_like']:.2f} (EV / payoff std-dev -- single-trade, Sharpe-like reward "
        f"per unit of payoff volatility; penalizes a high-POP trade with a disproportionately large "
        f"rare loss)"
        if expectancy["sharpe_like"] is not None else None
    )

    # --- Resolve the deferred ratio-over-ceiling flag now that POP is known ---
    # A reward:risk far above MAX_PLAUSIBLE_REWARD_RISK_RATIO means one of
    # two very different things, and readers need to be told which:
    #   - POP is low (< LOTTERY_POP_THRESHOLD_PCT): the structure is a
    #     genuine long-shot -- most likely outcome is max loss, and the
    #     "positive EV" comes entirely from a rare large payoff. Real, not
    #     a bug, but easily misread as an ordinary high-confidence spread
    #     if the report doesn't say so explicitly.
    #   - POP is normal/high/unknown: a defined-risk vertical's max_profit
    #     and max_loss are two pieces of the same strike width, so this
    #     combination essentially never comes from correctly priced legs --
    #     treated as a premium/strike/direction bug to verify, not a trade.
    horizon_dict["_lottery_like"] = False
    horizon_dict["_implausible_reward_risk"] = False
    if horizon_dict.get("_over_ratio_ceiling"):
        if pop is not None and pop < LOTTERY_POP_THRESHOLD_PCT:
            horizon_dict["_lottery_like"] = True
            lottery_note = (
                f" ⚠ Lottery-like payoff profile -- POP is only ~{pop:.0f}%, so the modeled MOST LIKELY "
                f"outcome is the max loss; the positive EV comes entirely from a rare large payoff, not "
                f"from this being a favored/high-confidence setup."
            )
            horizon_dict["reward_risk_ratio"] += lottery_note
            if horizon_dict.get("expected_value"):
                horizon_dict["expected_value"] += lottery_note
        else:
            horizon_dict["_implausible_reward_risk"] = True
            horizon_dict["reward_risk_ratio"] += (
                f" ⚠ Above {MAX_PLAUSIBLE_REWARD_RISK_RATIO:g}:1 with no low-POP explanation -- implausible "
                f"for a genuine defined-risk spread; verify premium extraction, strike/leg mapping, and "
                f"buy/sell direction before trusting this."
            )

    # --- Margin (approximate) and Return on Margin ---
    margin, rom = estimate_margin_and_rom(max_loss, max_profit, priced_legs, NIFTY_LOT_SIZE)
    horizon_dict["margin_required"] = (
        f"~₹{margin:,.0f} per lot -- estimated broker margin (SPAN + exposure margin proxy; "
        f"Zerodha, Groww, ICICI Direct, etc. compute this differently, so confirm the exact "
        f"figure in your own broker's margin calculator before sizing)"
    )
    horizon_dict["return_on_margin"] = (
        f"~{rom:.1f}% (assumes the structure is held to expiry and achieves max profit; actual "
        f"realized returns may differ due to early exit, margin changes, or assignment risk)"
        if rom is not None else "n/a"
    )
    # Institutional desks watch both sides of margin efficiency, not just the
    # upside: Profit/Margin (== Return on Margin, best case) and Risk/Margin
    # (worst case) together show how much of the capital actually locked up
    # is exposed to loss vs reward -- a structure can have an attractive ROM
    # while still risking a large share of its own margin (issue #9).
    risk_margin_pct = round(max_loss / margin * 100, 1) if margin else None
    if rom is not None and risk_margin_pct is not None:
        horizon_dict["capital_efficiency"] = (
            f"Profit/Margin ~{rom:.1f}% · Risk/Margin ~{risk_margin_pct:.1f}% "
            f"(share of locked-up margin at stake if max loss is hit)"
        )
    else:
        horizon_dict["capital_efficiency"] = "n/a"

    conf_label, conf_pct, conf_checks = compute_confidence(
        priced_legs, horizon_snap, result["breakevens"], is_eod, vix, spot
    )
    horizon_dict["confidence"] = conf_label
    horizon_dict["confidence_pct"] = conf_pct
    horizon_dict["confidence_reasons"] = conf_checks

    tq_score, tq_breakdown, tq_penalty_notes = compute_trade_quality_score(
        priced_legs, horizon_snap, ev_inr, max_loss, pop, reward_risk_ratio, conf_pct, is_eod
    )
    horizon_dict["_trade_quality_score"] = tq_score
    horizon_dict["trade_quality_breakdown"] = " · ".join(
        f"{name} {sub}/100 ({weight}%)" for name, (sub, weight) in tq_breakdown.items()
    )
    if tq_penalty_notes:
        horizon_dict["trade_quality_breakdown"] += " · ⚠ " + "; ".join(tq_penalty_notes)
    horizon_dict["trade_quality_score"] = f"{tq_score}/100 -- {horizon_dict['trade_quality_breakdown']}"

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

    return f"""Core Objective: You are an expert at analysing Nifty options-related data. A LIVE DATA FEED has already been fetched for you and is provided below -- use those figures as ground truth for this run rather than searching for or guessing them yourself. Using that feed (plus web search only for what it doesn't cover), recommend the best risk-defined strategy -- or combination of strategies -- for the current moment, independently across THREE time horizons:

{live_data_block}

1. Nifty Weekly -- Buy, Sell, or a Combination of both (current/nearest weekly expiry)
2. Nifty Monthly -- Buy, Sell, or a Combination of both (current monthly expiry)
3. Nifty Quarterly -- Buy, Sell, or a Combination of both (nearest quarterly expiry on the Mar/Jun/Sep/Dec cycle)

Horizons are independent from Monthly/Quarterly onward -- they do not need to agree with each other. For example Monthly can be a bullish debit spread while Quarterly is a range-bound Iron Condor, if the data supports it. The Weekly horizon, however, follows the bias-sequencing rule below (constraint #5), which explicitly couples the current week to the next week.

NON-NEGOTIABLE CONSTRAINTS

1. RISK-DEFINED ONLY. Every recommended strategy must have a mathematically capped maximum loss known before entry (e.g. Bull/Bear Call/Put Spreads, Iron Condors, Iron Butterflies, Calendar Spreads, Ratio Spreads with a protective wing, Debit/Credit Spreads). NEVER recommend a naked/undefined-risk position (naked short call, naked short put, uncovered short straddle/strangle) under any circumstance, for any horizon.
   - 2-LEG VERTICAL SPREADS MUST BE SAME OPTION TYPE, DIFFERENT STRIKES, ONE LONG + ONE SHORT. A valid 2-leg spread is always CE+CE or PE+PE -- e.g. "Sell 24600 CE, Buy 24800 CE" (Bear Call Spread) or "Buy 24200 PE, Sell 24000 PE" (Bear Put Spread). A recurring mistake is writing one CE leg and one unrelated PE leg with no strike relationship between them (e.g. "Sell 24600 CE, Buy 24800 PE") -- this is WRONG. It is not a spread at all: the short call has no long call above it capping its loss (naked short call, unbounded upside risk), and the long put is a completely separate, unrelated position. If you cannot find two liquid strikes of the SAME option type to form a real vertical spread, do not force a 2-leg CE+PE pairing -- either use a 4-leg Iron Condor/Butterfly (see constraint #6) or state in bias_reason that no valid 2-leg structure was available at adequate liquidity.
   - REWARD MUST BE PROPORTIONATE TO RISK. Capped risk is necessary but not sufficient -- a spread whose maximum loss is many multiples of its maximum profit is still a bad trade even though it's technically risk-defined. Do NOT pick strikes so far apart that the reward-to-risk ratio falls below {MIN_REWARD_RISK_RATIO:g}, and for credit spreads specifically, do NOT collect a net credit worth less than {MIN_CREDIT_WIDTH_PCT:g}% of the strike width. Both are checked exactly in code after you respond and a violation gets the whole horizon rejected, so prefer strikes CLOSER TOGETHER (narrower width) that collect adequate premium relative to the capital at risk, rather than defaulting to a wide, low-credit spread "for safety."

2. GAP PROTECTION. Every recommendation must explicitly address overnight/weekend gap risk:
   - Prefer structures whose short strikes sit outside the current expected-move band (VIX-implied 1 standard deviation) for that horizon.
   - If global cues (GIFT Nifty pre-market, US markets overnight, Asian markets) suggest elevated gap risk right now, explicitly flag it and bias the recommendation toward tighter risk-defined structures or reduced size -- never toward removing a hedge leg.
   - Do NOT compute or state a rupee gap-risk figure yourself -- for a genuinely risk-defined structure, loss on ANY gap size is capped at max loss, and that figure is calculated from real fetched premiums after you respond (see OUTPUT FORMAT).

3. CAPITAL PROTECTION RULES.
   - Max capital at risk per horizon must not exceed {PER_HORIZON_CAP_PCT:.0f}% of a hypothetical total options-trading capital pool.
   - Worst-case combined maximum loss across all three horizons (the plain sum of each horizon's own max loss, as if all were hit simultaneously -- a conservative ceiling, not a probabilistic portfolio-risk estimate) must not exceed {AGGREGATE_CAP_PCT:.0f}%.
   - Do NOT write a stop-loss/adjustment trigger yourself -- a standard one (short-strike breach, or loss reaching a set multiple of credit/premium) is generated in code from your chosen strikes after you respond.
   - Do NOT compute the rupee max loss/profit or the % of capital yourself -- those are calculated from real fetched premiums after you respond (see OUTPUT FORMAT). Focus only on choosing strikes/legs that keep risk within the caps once priced.

4. This is a stateless, any-time request -- today is {today_str}, and this run is being generated during: {session_label}. Produce a full, self-contained recommendation using only live data you can find right now; do not assume access to any prior recommendation.

5. WEEKLY BIAS SEQUENCING.
   - Determine the Current Week's bias (Bullish or Bearish) strictly from the live market data you find (spot/futures trend, OI buildup, PCR, FII/DII flow, global cues) -- never assume a default direction.
   - CORRECT PCR READING: use these graduated bands, not a bare >1/<1 binary. PCR(OI) is put OI divided by call OI, so PCR 1.61 means roughly 1.6x as much put open interest as call open interest -- NOT "equal calls and puts" (never describe any PCR other than very close to 1.0 as "equal" or "balanced"; that is a factually wrong reading of the ratio and has been a recurring mistake). Bands:
       PCR < 0.7            -> Bearish (call writing dominates)
       PCR 0.7 - 1.2        -> Neutral
       PCR 1.2 - 1.5        -> Bullish (put writing dominates -- put writers expect the market to stay above their strike, building support below spot)
       PCR > 1.5            -> Potentially overbullish / contrarian caution -- elevated put writing this extreme can precede consolidation or a pullback rather than confirming further upside; say so explicitly rather than reading it as simply "more bullish."
     Example of correct wording for PCR 1.61: "PCR(OI) 1.61 indicates relatively strong put writing, suggesting bullish positioning, although elevated PCR can sometimes precede consolidation." Do not state a bias that contradicts the PCR figure you were given without explaining, in bias_reason, exactly why other data overrides it.
   - CORRECT VIX READING: India VIX measures the MAGNITUDE of expected volatility, not direction. A high VIX means bigger expected moves either way (and richer option premiums, favorable for premium-selling structures) -- it does NOT by itself mean bearish. A low VIX (e.g. under ~15) signals calm, range-bound conditions, not bullishness or bearishness. Never write "high/low VIX indicates bullish/bearish trend" -- derive direction only from price/OI/PCR/flow data, and use VIX only for expected-move sizing and premium-selling suitability.
   - DO NOT assume mean reversion for the immediate next weekly expiry. A single week's move is not a quantitative basis for predicting the following week's direction, and the "next_week_bias" field (see schema below) is discarded and overridden in code regardless of what you write here -- do not build the Weekly horizon's legs around a next-week reversal thesis. Size and structure the Weekly horizon for the Current Week only, from the live data you found for it.

6. MONTHLY/QUARTERLY BIAS TOWARD SIDEWAYS/RANGE-BOUND. Unless live data shows a strong, well-supported directional catalyst inside that horizon's window, default the Monthly and Quarterly recommendations to range-bound/theta-positive structures (Iron Condor, Iron Butterfly, Calendar/Diagonal Spread) rather than pure directional debit/credit spreads -- these horizons are for harvesting range and time decay, not chasing the weekly directional call.
   - IRON BUTTERFLY DEFINITION: an Iron Butterfly sells a call AND a put at the SAME at-the-money strike (e.g. Sell 24500 CE + Sell 24500 PE), then buys further OTM wings on each side for protection -- ONE strike in the middle (both shorts), TWO different strikes on the outside (one long call above, one long put below). If the short call and short put strikes differ, that is an Iron Condor, not an Iron Butterfly -- do not mislabel one as the other.
   - LEGS MUST BE PAIRED BY ROLE (short straddle vs wings), NEVER BY STRIKE. A recurring mistake is pairing every CE with a PE at the SAME strike for each of the two strikes used (e.g. "Sell 24000 CE, Buy 24000 PE, Sell 24200 CE, Buy 24200 PE") -- this is WRONG. It puts both calls on the same (short, uncapped-upside) side and both puts on the same (long, capped but directionless) side, which is not risk-defined and is not an Iron Butterfly or Condor at all.
     CORRECT 4-leg pattern for Iron Condor/Butterfly, using two example strikes 24000 (lower wing) and 24500 (short straddle/inner) and 25000 (upper wing):
       Iron Butterfly (single ATM strike 24500, wings at 24000/25000): "Buy 24000 PE, Sell 24500 PE, Sell 24500 CE, Buy 25000 CE"
       Iron Condor (short strikes 24300/24700, wings at 24000/25000): "Buy 24000 PE, Sell 24300 PE, Sell 24700 CE, Buy 25000 CE"
     Notice the shape: the two SHORT legs (one CE, one PE) sit inside, closer to spot; the two LONG legs (one CE, one PE) sit outside as wings, one above and one below. Never write a leg string where both CE legs share one action (Buy/Sell) and both PE legs share the other -- that pattern is always wrong for these structures.

7. NIFTY LIQUIDITY FILTER. Only select strikes with adequate live liquidity for Nifty options -- meaningful open interest, tight bid-ask spread, and strikes close to standard NSE strike intervals. Choose legs ONLY from strikes shown with a live premium in the LIVE DATA FEED above for that horizon; if a theoretically ideal strike isn't listed there, state that explicitly and move to the nearest listed liquid strike instead.
   - STRIKE DISTANCE SANITY CHECK: a strike more than ~{_STRIKE_DISTANCE_FACTOR:g}x that horizon's own expected move (the ATM straddle figure in the live data feed) away from spot is only defensible if there's a real, substantial OI wall at that exact strike -- do not pick a far-OTM strike "for extra safety margin" without that OI backing. This is checked and flagged in code against the real OI data after you respond, so an unjustified far-out strike will show up as a lowered-confidence flag rather than silently passing.
   - IRON CONDOR SHORT-STRIKE-OUTSIDE-EXPECTED-MOVE RULE (hard requirement): for an Iron Condor specifically, BOTH short strikes must sit outside that horizon's own 1-sigma expected move (the ATM straddle figure in the live data feed) from spot -- e.g. if expected move is ±426 pts and spot is 24334, neither short strike may fall between roughly 23908 and 24760. A short strike inside that band is rejected outright in code, not just flagged, because it contradicts the very range-bound thesis an Iron Condor is sold on. (This does not apply to Iron Butterfly, whose shorts are intentionally ATM.)

8. RISK-CONTROLLED SYNTHESIS. Every horizon's final recommendation must simultaneously satisfy constraints #1-#7 above -- risk-defined, gap-protected, capital-capped, correctly bias-sequenced (Weekly), correctly range-biased (Monthly/Quarterly), and liquidity-filtered. If any of these pull in conflicting directions for a given horizon (e.g. the liquid strike sits inside the expected-move band), state the trade-off explicitly in that horizon's bias_reason and resolve it in favor of the tighter risk-defined structure, never in favor of higher theoretical reward.

ADDITIONAL LIVE DATA TO SEARCH FOR (only for what the LIVE DATA FEED above does not already give you -- do not re-search or contradict anything already provided there; if you cannot find a real current figure, say so for that field rather than inventing one):
- Nifty futures price (weekly/monthly, for basis) -- the feed above gives spot, not futures
- Confirmation/context on options-chain levels for any horizon the feed marked unavailable
- FII/DII cash + index-options net positioning (most recent session) -- not covered by the feed
- GIFT Nifty pre-market level (if available) and prior-session US market close (S&P 500, Nasdaq), for gap risk
- Any major near-term event risk (RBI policy, US Fed meeting, major earnings cluster, budget/election dates) that falls within each horizon's window
- Use ONLY primary/official sources: NSE option chain and Bhavcopy, NSE market statistics, India VIX, NSE FII/DII statistics, GIFT Nifty/SGX derivatives data, RBI's policy calendar, company earnings calendars, and official exchange circulars. Never cite social media (Instagram, YouTube, X/Twitter), influencer "prediction" videos, or unsourced broker marketing content as a basis for any figure or bias.

OUTPUT FORMAT -- respond with ONLY raw JSON matching the schema below, and nothing else (no markdown, no code fences, no commentary before or after). Keep every field to plain text/numbers only (no HTML). NOTE: max_loss, max_profit, the two pct_capital fields, breakeven, gap_risk, net Greeks (delta/theta/vega/gamma), probability of profit, margin required, return on margin, capital efficiency, and the confidence score are NOT requested below -- they are all calculated from real fetched premiums/IVs in code after parsing your "legs" field, specifically to avoid the arithmetic errors that come from an LLM inventing option payoff numbers. Your job is only to choose the strikes; be precise and unambiguous in "legs" (format: "Sell 25000 CE, Buy 25200 CE" -- always "Buy"/"Sell", a strike from the live data, and "CE"/"PE"):

{{
  "horizons": [
    {{
      "horizon": "Weekly",
      "expiry_date": "The actual expiry date for this horizon, copied EXACTLY as shown in the LIVE DATA FEED above (e.g. '24 Jul 2026') -- do not calculate or guess your own expiry cycle date; use the real one already provided.",
      "bias": "One of: Bullish / Bearish / Neutral / Range-bound",
      "next_week_bias": "ONLY for the Weekly horizon object: this field is ignored and overridden in code per constraint #5 -- leave it empty or write 'n/a'. Omit entirely for Monthly/Quarterly.",
      "bias_reason": "One or two sentences grounded in the live data you found, internally consistent with the PCR/VIX reading rules in constraint #5",
      "strategy_name": "e.g. 'Bear Call Spread' or 'Iron Condor'",
      "legs": "Full leg structure using ONLY strikes shown in the live data, e.g. 'Sell 25000 CE, Buy 25200 CE (1 lot)'. For Iron Condor/Butterfly, pair legs by ROLE not by strike -- both short legs (one CE, one PE) go together nearer spot, both long legs (one CE, one PE) go together as the outer wings. Never pair a CE and PE at the same strike as one Buy + one Sell for each of two strikes -- that is not a valid risk-defined structure. See constraint #6 for the correct pattern.",
      "strike_rationale": "One sentence, qualitative only (no delta/POP numbers -- those are computed and shown separately): why THIS short strike, e.g. 'anchored beyond the nearest major OI wall on this side'. Do NOT state whether the strike is inside or outside the expected-move band -- that is computed exactly from real fetched premiums and appended automatically after your text; a guessed band claim here risks contradicting it.",
      "confidence": "One of: High / Medium / Low -- your best qualitative guess, but note this is DISCARDED and replaced with a percentage score computed in code from verifiable signals (OI alignment, Max Pain, live-vs-EOD data, VIX regime), same as max_loss/Greeks/POP above.",
      "data_status": "One of: 'live' (found real current data), 'partial' (some fields estimated), 'stale' (data may be several hours old or unavailable) -- your best honest guess, but note this is DISCARDED and overridden in code from the real live-vs-EOD-Bhavcopy signal for that horizon, precisely because a model claiming 'live' in a run where the direct feed was down and EOD Bhavcopy was actually used would contradict the live-feed-status banner elsewhere in the same report."
    }},
    {{ "horizon": "Monthly", ... same fields ... }},
    {{ "horizon": "Quarterly", ... same fields ... }}
  ],
  "portfolio_view": "One paragraph on whether the combined structure over- or under-hedges overall market gap risk across the three horizons, and how the horizons relate to each other qualitatively (e.g. does Weekly's directional stance conflict with Monthly/Quarterly's range-bound stance). Do NOT state or estimate a net long/short gamma verdict for the combined portfolio -- that figure is computed in code from real per-leg Black-Scholes gamma (live IV) after you respond, and any gamma claim you make here will be discarded. Do NOT state or estimate the worst-case combined max-loss percentage or whether it stays within the {AGGREGATE_CAP_PCT:.0f}% cap -- that figure is summed from verified per-horizon numbers in code after you respond, and any claim you make about it here will be discarded."
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
    "not generated": ("#8A8F9C", "#F4F2ED"),
}

_DATA_STATUS_STYLE = {
    "live": ("#2F5233", "#E7EEE4", "Live"),
    "eod": ("#A6812F", "#FDF3D9", "EOD / Last Close"),
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
    style = _BIAS_STYLE.get(key)
    if style is None:
        # Match on prefix too, e.g. "neutral (insufficient evidence)" should
        # still get the plain "neutral" styling rather than falling through
        # to the generic gray default.
        for k, v in _BIAS_STYLE.items():
            if key.startswith(k):
                style = v
                break
    color, bg = style or ("#8A8F9C", "#F4F2ED")
    return _badge(bias or "—", color, bg)


def _confidence_badge(conf):
    key = str(conf or "").strip().lower()
    color, bg = _CONFIDENCE_STYLE.get(key, ("#8A8F9C", "#F4F2ED"))
    return _badge(conf or "—", color, bg)


def _confidence_cell_html(h, sans):
    """Badge + percentage + a ✓/✗ checklist of the verifiable reasons
    behind compute_confidence()'s score (issue #7) -- replaces the old
    bare 'Medium' badge with something a reader can actually audit."""
    badge = _confidence_badge(h.get("confidence"))
    pct = h.get("confidence_pct")
    pct_html = (
        f' <span style="font-family:{sans};font-size:12px;color:#4A5063;">({pct}%)</span>'
        if isinstance(pct, (int, float)) else ""
    )
    reasons = h.get("confidence_reasons") or []
    if not reasons:
        return f"{badge}{pct_html}"
    items = "".join(
        f'<div style="font-family:{sans};font-size:11px;color:{"#2F5233" if ok else "#8B2E2E"};margin-top:3px;">'
        f'{"✓" if ok else "✗"} {html.escape(str(reason))}</div>'
        for ok, reason in reasons
    )
    return f"{badge}{pct_html}{items}"


def _data_status_badge(status):
    key = str(status or "").strip().lower()
    color, bg, label = _DATA_STATUS_STYLE.get(key, ("#8A8F9C", "#F4F2ED", status or "—"))
    return _badge(label, color, bg)


# Sensibull doesn't publish a documented URL spec for pre-filling arbitrary
# custom multi-leg strategies (strikes/actions/expiry) via query params, so
# this deliberately does NOT try to guess one -- a wrong guess would produce
# a link that looks legitimate but silently loads the wrong (or an empty)
# strategy, which is worse than no link at all. What IS a confirmed, stable
# public route is the NIFTY strategy builder itself
# (https://web.sensibull.com/option-strategy-builder?instrument_symbol=NIFTY),
# so every horizon card links there -- the reader lands straight on
# Sensibull's Nifty builder and only needs to pick the expiry (shown right
# next to this link) and add the exact legs already spelled out in the row
# above, rather than starting from Sensibull's home page.
SENSIBULL_STRATEGY_BUILDER_URL = "https://web.sensibull.com/option-strategy-builder?instrument_symbol=NIFTY"


def _execution_cell_html(h, sans, expiry):
    """Structured Broker / Status / Verification block, replacing the old
    bare 'Open Sensibull' link. Status is never the model's own framing --
    it's derived from the same real signals the rest of the card already
    uses: _horizon_rejected (unparseable legs / undefined risk / poor
    reward:risk) and data_status (live NSE chain vs. stale EOD Bhavcopy
    fallback), so a rejected or stale-data horizon can never show 'Ready'
    here while its own Payoff Verification row above says otherwise."""
    if _horizon_rejected(h):
        status_text, status_color = "🛑 Not Ready", "#8B2E2E"
        verification_text = "Rejected -- do not place this trade until the legs are corrected (see Payoff Verification above)."
    elif str(h.get("data_status") or "").strip().lower() == "eod":
        status_text, status_color = "⚠ Verify First", "#A6812F"
        verification_text = "Priced off EOD Bhavcopy (stale close, not live) -- confirm live premiums with your broker before placing this order."
    else:
        status_text, status_color = "✅ Ready", "#2F5233"
        verification_text = "Confirm live premium before placing order."

    broker_line = (
        f'<a href="{SENSIBULL_STRATEGY_BUILDER_URL}" style="color:#8A6D3B;font-weight:600;'
        f'text-decoration:none;">Sensibull</a> '
        f'<span style="color:#8A8F9C;font-size:11px;">(select {expiry} expiry, then add the legs above)</span>'
    )

    def _line(label, value_html):
        return (
            f'<div style="margin-top:4px;">'
            f'<span style="font-family:{sans};font-size:11px;font-weight:700;color:#8A8F9C;'
            f'text-transform:uppercase;letter-spacing:0.04em;">{label}</span><br>'
            f'<span style="font-family:{sans};font-size:12px;color:#14213D;">{value_html}</span></div>'
        )

    return "".join([
        _line("Broker", broker_line),
        _line("Status", f'<span style="font-weight:700;color:{status_color};">{status_text}</span>'),
        _line("Verification", f'<span style="color:#4A5063;">{html.escape(verification_text)}</span>'),
    ])


def _horizon_card_html(h, sans, serif):
    """One risk-defined-strategy card per horizon (Weekly/Monthly/
    Quarterly), styled consistently with the swing-trade note's
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
        raw_row("Execution", _execution_cell_html(h, sans, expiry)),
        raw_row("Directional Bias (Current Week)" if h.get("next_week_bias") else "Directional Bias", _bias_badge(h.get("bias"))),
        *([raw_row("Directional Bias (Next Week)", _bias_badge(h.get("next_week_bias")))] if h.get("next_week_bias") else []),
        row("Bias Rationale", "bias_reason"),
        *([row("OI Trend", "oi_trend")] if h.get("oi_trend") else []),
        row("Strike Selection Rationale", "strike_rationale"),
        row("Expected Move (ATM Straddle)", "expected_move"),
        row("Max Loss", "max_loss", value_color="#8B2E2E", bold=True),
        row("Max Loss (% of horizon capital)", "max_loss_pct_capital", value_color="#8B2E2E"),
        row("Max Profit", "max_profit", value_color="#2F5233", bold=True),
        row("Max Profit (% of horizon capital)", "max_profit_pct_capital", value_color="#2F5233"),
        row("Breakeven", "breakeven"),
        *([row("Probability of Profit", "probability_of_profit")] if h.get("probability_of_profit") else []),
        *([row("Probability of Touch", "probability_of_touch")] if h.get("probability_of_touch") else []),
        *([row("Expected Win Rate", "expected_win_rate")] if h.get("expected_win_rate") else []),
        *([row("Expected Value (EV)", "expected_value",
               value_color="#2F5233" if "positive" in str(h.get("expected_value")) else "#8B2E2E")]
          if h.get("expected_value") else []),
        *([row("Kelly %", "kelly_pct")] if h.get("kelly_pct") else []),
        *([row("Expectancy Ratio (Sharpe-like)", "expectancy_ratio")] if h.get("expectancy_ratio") else []),
        row("Net Greeks (per lot)", "net_greeks"),
        row("Margin Required", "margin_required"),
        row("Return on Margin", "return_on_margin"),
        row("Capital Efficiency", "capital_efficiency"),
        row("Gap Risk", "gap_risk"),
        row("Adjustment / Exit Trigger", "adjustment_trigger"),
        *([row("Reward : Risk", "reward_risk_ratio")] if h.get("reward_risk_ratio") else []),
        row("Trade Quality Score", "trade_quality_score"),
        raw_row("Confidence", _confidence_cell_html(h, sans)),
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


def classify_pcr(pcr):
    """Graduated PCR(OI) reading, replacing the old bare '>1 bullish-ish,
    <1 bearish-ish' binary with the desk's actual bands (issue #6):
      < 0.7          -> Bearish (call writing dominates)
      0.7 - 1.2      -> Neutral
      1.2 - 1.5      -> Bullish (put writing dominates)
      > 1.5          -> Potentially overbullish / contrarian caution
    Returns None if pcr is None -- callers must handle that rather than
    guessing a band for missing data."""
    if pcr is None:
        return None
    if pcr < 0.7:
        return "Bearish (call writing dominates)"
    if pcr <= 1.2:
        return "Neutral"
    if pcr <= 1.5:
        return "Bullish (put writing dominates)"
    return "Potentially overbullish / contrarian caution (elevated put writing can precede consolidation)"


_PCR_EQUAL_CLAIM_RE = re.compile(
    r"\bequal\b[^.!?]*\b(calls?|puts?)\b|\bbalanced\b[^.!?]*\b(calls?|puts?)\b|"
    r"\bbalanced\s+positions?\b|\bequal\s+positions?\b|\bbalanced\s+positioning\b",
    re.IGNORECASE,
)


def _scrub_pcr_mischaracterization(bias_reason, pcr):
    """Discards the model's bias_reason sentence if it mischaracterizes a
    real PCR figure as 'equal' or 'balanced' calls and puts -- a PCR of
    1.61 means roughly 1.6x as much put OI as call OI, not parity, and a
    model has repeatedly written exactly this kind of description despite
    being given the real number. Same whole-sentence-replacement pattern
    as _scrub_false_band_claim: a real graduated PCR reading is computed
    here from the actual fetched figure and substituted in, rather than
    surgically editing the model's wording (which risks leaving a
    grammatically broken fragment behind)."""
    if not bias_reason or pcr is None:
        return bias_reason
    sentences = re.split(r"(?<=[.!?])\s+", bias_reason.strip())
    band = classify_pcr(pcr)
    fixed = []
    for s in sentences:
        if "pcr" in s.lower() and _PCR_EQUAL_CLAIM_RE.search(s):
            fixed.append(f"PCR(OI) {pcr:g} indicates {band.lower()}.")
        else:
            fixed.append(s)
    return " ".join(fixed).strip()


def _pcr_implied_direction(pcr):
    """Coarse directional read from PCR alone, used only to sanity-check
    the model's own 'bias' badge field (issue #5) -- not a substitute for
    compute_horizon_recommendation's full logic, since price trend/OI
    buildup/flow can legitimately make the real bias diverge from PCR
    alone. Both the 1.2-1.5 band and the >1.5 band are put-writing-
    dominant reads and count as 'bullish' here even though classify_pcr()
    labels the >1.5 band as 'overbullish/contrarian caution' -- either
    way it is not 'Neutral' or 'Balanced'."""
    if pcr is None:
        return None
    if pcr < 0.7:
        return "bearish"
    if pcr <= 1.2:
        return "neutral"
    return "bullish"


_NEUTRAL_BIAS_LABELS = ("neutral", "balanced", "range-bound", "range bound", "sideways")


def _scrub_bias_pcr_conflict(bias, pcr):
    """Fixes horizon_dict['bias'] itself (the badge text shown in the
    report), not just bias_reason -- these are two separate fields the
    model fills in, and only bias_reason was being checked against PCR
    before this. A PCR of 1.61 is roughly 1.6x as much put OI as call OI;
    labelling that horizon 'Balanced' or 'Neutral' in the badge is the
    same factual error _scrub_pcr_mischaracterization already guards
    against in the prose, just in the other field. Only overrides when
    PCR is unambiguously outside the 0.7-1.2 neutral band AND the model's
    own badge text is itself a neutral/balanced word -- a genuine
    Bullish-vs-Bearish call from other live data is never touched here,
    since PCR is only one input among several."""
    if pcr is None or not bias:
        return bias
    direction = _pcr_implied_direction(pcr)
    if direction not in ("bullish", "bearish"):
        return bias
    label = str(bias).strip().lower()
    if any(label == n or label.startswith(n) for n in _NEUTRAL_BIAS_LABELS):
        return "Bullish" if direction == "bullish" else "Bearish"
    return bias


_NO_STRATEGY_CLAIM_RE = re.compile(
    r"\bno specific strateg(?:y|ies)\b|\bno strategy recommended\b|"
    r"\bno trade recommended\b|\bwait for confirmation\b|"
    r"\bno risk-defined structure\b|\bno structure is recommended\b|"
    r"\bconflicting\b[^.!?]*\b(?:pcr|vix|signals?|readings?)\b",
    re.IGNORECASE,
)


def _model_declared_no_strategy(h):
    """True if the model's own bias_reason or verification prose says it
    isn't recommending a specific strategy this horizon (e.g. "No specific
    strategy recommended due to conflicting PCR and VIX readings") --
    checked BEFORE any of the payoff-based verdict rules in
    compute_horizon_recommendation, so the top-of-report summary table can
    never show "Consider" for a horizon whose own detail card says no
    trade was recommended. Without this, a horizon where the model punted
    in prose but still left verifiable (even if leftover/default) legs
    behind would sail through the reward/EV/cap/confidence checks and get
    marked Consider -- a direct contradiction between the summary and the
    detail (Weekly Recommendation Contradiction)."""
    for field in ("bias_reason", "verification"):
        text = str(h.get(field) or "")
        if _NO_STRATEGY_CLAIM_RE.search(text):
            return True
    return False


def _scrub_portfolio_view_contradictions(portfolio_view, horizons):
    """Fixes portfolio_view when it contradicts a specific horizon's own,
    actual recommendation (Portfolio Summary Contradiction, highest
    priority). portfolio_view is a separate free-text paragraph the model
    writes independently of each horizon's bias_reason/verification --
    rule 0 in compute_horizon_recommendation already stops a horizon's
    OWN fields from contradicting each other (the earlier "Weekly
    Recommendation Contradiction" fix), but nothing was checking this
    cross-horizon summary paragraph against what each horizon card
    actually says. E.g. the Weekly card genuinely recommends a Bullish
    Vertical Spread (verdict "Consider"), but portfolio_view separately
    claims "No risk-defined structure is recommended for the Weekly" --
    almost certainly stale text carried over from an earlier draft rather
    than a real second opinion, since the model never gets a chance to
    revise portfolio_view after seeing the final per-horizon verdicts.

    Only overrides a sentence that both (a) names a specific horizon and
    (b) makes a no-strategy/no-structure claim about it, while that
    horizon's own deterministic verdict says a real trade WAS
    recommended (i.e. not "No Trade"/"Skip"/"Not Available" -- Consider/
    Neutral/Caution all mean a structure exists and was evaluated, just
    with a quality caveat). Every other sentence is left untouched."""
    if not portfolio_view:
        return portfolio_view
    by_name = {str(h.get("horizon") or "").strip(): h for h in (horizons or []) if h.get("horizon")}
    no_trade_verdicts = {"⚪ No Trade", "❌ Skip", "Not Available"}
    sentences = re.split(r"(?<=[.!?])\s+", portfolio_view.strip())
    fixed = []
    for s in sentences:
        replaced = False
        if _NO_STRATEGY_CLAIM_RE.search(s):
            for name, h in by_name.items():
                if not name or name.lower() not in s.lower():
                    continue
                verdict, _color, _reason = compute_horizon_recommendation(h)
                if verdict not in no_trade_verdicts:
                    strategy = h.get("strategy_name") or "a risk-defined structure"
                    fixed.append(
                        f"{name} actually carries a recommended {strategy} (see the {name} card "
                        f"above) -- correcting contradictory text here."
                    )
                    replaced = True
                break
        if not replaced:
            fixed.append(s)
    return " ".join(fixed).strip()


def _scrub_portfolio_view_structure_type_contradiction(portfolio_view, horizons):
    """Fixes portfolio_view when it names the WRONG specific structure
    type for a horizon -- e.g. "Monthly and Quarterly iron butterflies"
    when Quarterly's card actually shows an Iron Condor (Quarterly
    Strategy Description issue). This is a finer-grained check than
    _scrub_portfolio_view_directional_contradiction: Iron Butterfly and
    Iron Condor are BOTH range-bound structures, so the directional-stance
    check alone wouldn't catch a mix-up between the two -- this compares
    the actual structure name, reusing the same _STRUCTURE_KEYWORDS
    vocabulary _label_conflicts already uses to check the model's own
    strategy_name against its own legs, so every consistency check in
    this file shares one vocabulary rather than three that could
    silently disagree with each other.

    Deliberately conservative: only fires when a sentence mentions
    exactly ONE distinct structure keyword (e.g. "iron butterflies"
    applied to two horizons at once, as in the example above). A
    sentence naming several different structure types for different
    horizons in the same breath is too ambiguous to safely attribute
    which keyword belongs to which horizon, so it's left untouched
    rather than risk a wrong "correction"."""
    if not portfolio_view:
        return portfolio_view
    by_name = {str(h.get("horizon") or "").strip(): h for h in (horizons or []) if h.get("horizon")}
    # Regex per canonical keyword, not a plain substring check -- "iron
    # butterfly" pluralizes irregularly ("iron butterflies"), so a bare
    # `in` check silently misses exactly the plural phrasing this issue
    # was reported with ("Monthly and Quarterly iron butterflies").
    keyword_patterns = {
        "iron condor": r"iron condors?",
        "iron butterfly": r"iron butterfl(?:y|ies)",
        "bull call": r"bull calls?",
        "bear call": r"bear calls?",
        "bull put": r"bull puts?",
        "bear put": r"bear puts?",
        "straddle": r"straddles?",
        "strangle": r"strangles?",
    }
    sentences = re.split(r"(?<=[.!?])\s+", portfolio_view.strip())
    fixed = []
    for s in sentences:
        s_lower = s.lower()
        mentioned = {kw for kw, pat in keyword_patterns.items() if re.search(pat, s_lower)}
        named = [name for name in by_name if name and name.lower() in s_lower]
        if len(mentioned) == 1 and named:
            claimed_kw = next(iter(mentioned))
            contradicted = [
                name for name in named
                if claimed_kw not in str(by_name[name].get("strategy_name") or "").lower()
            ]
            if contradicted:
                descriptions = [
                    f"{name} is {by_name[name].get('strategy_name') or 'an unspecified structure'}"
                    for name in named
                ]
                combined = "; ".join(descriptions) + " -- correcting a structure-type mismatch here."
                fixed.append(combined[:1].upper() + combined[1:])
                continue
        fixed.append(s)
    return " ".join(fixed).strip()


_GAMMA_WORD_RE = re.compile(r"\bgamma\b", re.IGNORECASE)


_RANGE_BOUND_CLAIM_RE = re.compile(r"\brange[- ]bound\b|\bsideways\b", re.IGNORECASE)

# Same non-directional keyword list compute_market_regime already uses for
# its own Weekly range/direction label -- reused here so the two can't
# silently disagree about what counts as "range-bound".
_NON_DIRECTIONAL_STRATEGY_KW = ("iron condor", "iron butterfly", "straddle", "strangle", "butterfly")


def _is_directional_strategy(strategy_name):
    """True if strategy_name reads as a directional structure (vertical
    spread / debit / credit spread with a clear up-or-down bias) rather
    than a range-bound, theta-harvesting one (Iron Condor/Butterfly,
    Straddle/Strangle). False (not directional) if strategy_name is
    empty, since an unlabeled strategy shouldn't be flagged as a
    contradiction either way."""
    s = str(strategy_name or "").strip().lower()
    if not s:
        return False
    return not any(kw in s for kw in _NON_DIRECTIONAL_STRATEGY_KW)


def _scrub_portfolio_view_directional_contradiction(portfolio_view, horizons):
    """Fixes portfolio_view when it calls a horizon "range-bound" or
    "sideways" but that horizon's own actual recommended strategy is a
    directional structure (Portfolio Description issue) -- e.g. "Weekly
    and Monthly maintaining a range-bound stance" when Weekly's card
    actually shows a Bullish Vertical Spread. Same stale-prose problem as
    _scrub_portfolio_view_contradictions (a "no strategy" claim), just a
    mischaracterization of an EXISTING strategy's directionality instead
    of denying one exists.

    A single sentence can legitimately name more than one horizon (as in
    the example above) where only one of them is actually wrong, so
    whenever ANY named horizon disagrees the whole sentence is replaced
    with an explicit per-horizon rebuild -- same whole-sentence-
    replacement principle as _scrub_pcr_mischaracterization, rather than
    surgically deleting just the wrong name and risking a broken
    fragment."""
    if not portfolio_view:
        return portfolio_view
    by_name = {str(h.get("horizon") or "").strip(): h for h in (horizons or []) if h.get("horizon")}
    sentences = re.split(r"(?<=[.!?])\s+", portfolio_view.strip())
    fixed = []
    for s in sentences:
        if _RANGE_BOUND_CLAIM_RE.search(s):
            named = [name for name in by_name if name and name.lower() in s.lower()]
            contradicted = [name for name in named if _is_directional_strategy(by_name[name].get("strategy_name"))]
            if contradicted:
                descriptions = []
                for name in named:
                    strategy = by_name[name].get("strategy_name") or "an unspecified structure"
                    stance = "directional" if name in contradicted else "range-bound"
                    descriptions.append(f"{name} is {stance} ({strategy})")
                combined = "; ".join(descriptions) + " -- correcting contradictory stance text here."
                fixed.append(combined[:1].upper() + combined[1:])
                continue
        fixed.append(s)
    return " ".join(fixed).strip()


def _strip_gamma_claims(text):
    """Removes any sentence where the model asserted a net long/short gamma
    verdict -- the model only sees its own free-text strategy description,
    not the real per-leg IV/Greeks computed in code, so a blanket claim like
    'net short gamma across all horizons' can be wrong (e.g. a debit
    vertical spread is generally long gamma even though condors/butterflies
    elsewhere in the portfolio are short gamma). The real, code-computed
    verdict is prepended separately by compute_portfolio_gamma_summary()."""
    if not text:
        return text
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    kept = [s for s in sentences if not _GAMMA_WORD_RE.search(s)]
    return " ".join(kept).strip()


def compute_portfolio_gamma_summary(horizons):
    """Builds the single authoritative statement of the portfolio's combined
    gamma exposure from the real per-horizon net gamma figures computed in
    apply_verified_payoff (Black-Scholes, from live IV) -- never from the
    model's own portfolio-level claim, which has no way to know that e.g. a
    debit Bull Call Spread is generally long gamma even while Iron Condor/
    Butterfly legs elsewhere are short gamma (issue #2). Returns a plain-
    English sentence, or an honest 'insufficient data' note if fewer than
    two horizons have a usable gamma figure."""
    parts, usable = [], []
    for h in horizons:
        name = str(h.get("horizon") or "").strip()
        gamma = h.get("_net_gamma")
        if gamma is None or not name:
            continue
        usable.append((name, h.get("strategy_name") or "n/a", gamma))

    if not usable:
        return (
            "Combined portfolio gamma: not computable this run (live IV was unavailable "
            "for one or more legs across all horizons)."
        )

    for name, strategy, gamma in usable:
        direction = "long gamma" if gamma > 0 else ("short gamma" if gamma < 0 else "gamma-neutral")
        parts.append(f"{name} ({strategy}): Γ {gamma:+.3f} — {direction}")

    net_total = sum(g for _, _, g in usable)
    if net_total > 0:
        overall = "net LONG gamma overall"
    elif net_total < 0:
        overall = "net SHORT gamma overall"
    else:
        overall = "net gamma-neutral overall"

    coverage_note = (
        "" if len(usable) == len([h for h in horizons if str(h.get("horizon") or "").strip()])
        else " (some horizons omitted -- live IV unavailable for those legs)"
    )

    return (
        f"Per-horizon net gamma (per lot, from live IV): {'; '.join(parts)}. "
        f"Summed across the horizons with a computable figure, the combined structure is "
        f"{overall}{coverage_note} -- not uniformly short gamma across every horizon, since "
        f"long-gamma debit spreads and short-gamma premium-selling structures can coexist and "
        f"partly offset."
    )


_EXPECTED_MOVE_RE = re.compile(r"±[\d,]+(?:\.\d+)?")
_POP_RE = re.compile(r"~[\d.]+%")
_RUPEE_FIGURE_RE = re.compile(r"₹[\d,]+")
_EV_FIGURE_RE = re.compile(r"₹-?[\d,]+")
_RR_RATIO_RE = re.compile(r"^[\d.]+")


def compute_market_regime(live_data, horizons):
    """Builds the one-line 'Market Regime' summary as a scannable header,
    composed ENTIRELY from data already computed or shown elsewhere in the
    report -- never a new inference of its own:

    - Volatility read comes from the same verified VIX figure used
      elsewhere (fetch_live_market_data / compute_confidence), bucketed
      with the same <20 'calm' cutoff already used for confidence scoring
      (so this label can't contradict that one) plus a <13 sub-cutoff for
      a plain 'Low' read.
    - Range-bound vs. Directional comes from the nearest-term (Weekly)
      horizon's own strategy choice: a non-directional structure (Iron
      Condor/Butterfly/Straddle/Strangle) reads as 'Range-bound';
      anything else falls back to whatever directional bias that same
      horizon already states.
    - Bias phrase is a plain-English echo of that same Weekly horizon's
      already-displayed 'bias' field, not a new directional call.

    Returns None if there's neither a usable VIX figure nor a usable
    Weekly horizon, rather than fabricating a regime label."""
    vix = live_data.get("vix") if live_data else None
    vol_label = None
    if isinstance(vix, (int, float)):
        if vix < 13:
            vol_label = f"Low-volatility (VIX {vix:g})"
        elif vix < 20:
            vol_label = f"Moderate-volatility (VIX {vix:g})"
        else:
            vol_label = f"Elevated-volatility (VIX {vix:g})"

    by_name = {str(h.get("horizon") or "").strip(): h for h in (horizons or []) if h.get("horizon")}
    weekly = by_name.get("Weekly") or (horizons[0] if horizons else None)

    range_label = None
    bias_label = None
    if weekly:
        strategy = str(weekly.get("strategy_name") or "").lower()
        bias = str(weekly.get("bias") or "").strip()
        non_directional_kw = ("iron condor", "iron butterfly", "straddle", "strangle", "butterfly")
        if any(kw in strategy for kw in non_directional_kw):
            range_label = "Range-bound"
        elif bias:
            range_label = "Directional"
        if bias:
            bias_lower = bias.lower()
            if "bull" in bias_lower:
                bias_label = "Bullish bias"
            elif "bear" in bias_lower:
                bias_label = "Bearish bias"
            elif "neutral" in bias_lower or "range" in bias_lower:
                bias_label = "Neutral bias"
            else:
                bias_label = f"{bias} bias"

    parts = [p for p in (vol_label, range_label, bias_label) if p]
    return " · ".join(parts) if parts else None


def compute_horizon_recommendation(h):
    """Deterministic verdict for one horizon, built entirely from fields
    apply_verified_payoff / compute_trade_quality_score already computed
    from real premiums -- never from the model's own framing (rule 0's
    "no strategy" check is the one deliberate exception, since it exists
    specifically to catch the model contradicting itself). This is an
    explicit, ordered rule table -- every verdict traces to exactly one
    named rule below rather than an unstated blend of factors, so a reader
    can always see *why* a given horizon landed where it did:

      0. No-strategy admission: the model's own bias_reason/verification
         prose says it isn't recommending a specific strategy this horizon
         (e.g. conflicting PCR/VIX readings) -> "No Trade". Checked FIRST,
         ahead of every payoff-based rule below, because a "no strategy"
         horizon can still carry leftover/default legs that verify cleanly
         -- without this rule those legs would sail through rules 1-6 and
         render as "Consider" directly beside detail text saying no trade
         was recommended (the Weekly Recommendation Contradiction).
      1. Rejected outright (unparseable legs, undefined/naked risk, or
         failed the reward:risk / credit-width quality gate) -> "Skip".
      2. MaxLoss% > PER_HORIZON_CAP_PCT -> "Skip". A trade the account
         genuinely cannot afford at this size, regardless of how good it
         otherwise looks -- this is a sizing constraint, not a quality
         judgment, which is exactly why it's checked before quality is
         even consulted.
      3. EV < 0 at the modeled POP -> "Skip". A defined-risk trade can
         still be a net loser in expectation (see
         compute_expectancy_metrics's docstring).
      4. Reward:Risk ratio is far above what's plausible for a genuine
         defined-risk spread (see MAX_PLAUSIBLE_REWARD_RISK_RATIO), split
         by POP into two distinct "Caution" verdicts so the reason shown
         always matches the detail card underneath:
           - POP is also very low: a genuine long-shot payoff, labeled
             "Lottery-Like Payoff" rather than treated as an error.
           - POP is normal/high/unknown: a premium/strike/direction bug
             is far more likely than a genuinely great trade, labeled
             "Reward:Risk Implausible -- Verify Premiums".
         These stay their own named Caution reasons rather than folding
         into the quality-score rule below: both are data-trust flags a
         reader needs called out explicitly, and a merely mediocre
         quality score doesn't communicate "verify this" or "this is a
         long shot" on its own.
      5. Trade Quality Score >= CONSIDER_QUALITY_THRESHOLD -> "Consider".
         The single number every other verified metric (EV, R:R, POP,
         code-scored confidence) already rolls up into -- so a trade that
         cleared rules 1-4 but scores below this bar isn't waved through
         just because nothing outright failed.
      6. Otherwise -> "Neutral". Cleared every hard gate above (not
         rejected, not over-cap, not negative-EV, not a lottery/implausible
         flag) but didn't clear the quality bar either -- e.g. exactly the
         "Skip on cap, but 63/100 quality with positive EV isn't terrible"
         case: a horizon can fail rule 2 and never even reach rule 6, but
         one that clears the cap with a mediocre score now gets its own
         distinct label instead of being folded into either "Consider" or
         a Skip-adjacent Caution reason it doesn't really deserve.

    Returns (verdict, css_color, reason) where verdict is one of
    "✅ Consider", "◐ Neutral", "⚠ Caution", "❌ Skip", "⚪ No Trade",
    "Not Available"."""
    green, amber, red, gray, blue = "#2F5233", "#A6812F", "#8B2E2E", "#8A8F9C", "#3D6690"

    if _model_declared_no_strategy(h):
        return "⚪ No Trade", gray, "Conflicting Signals"

    if _horizon_rejected(h):
        ml = str(h.get("max_loss") or "")
        if "UNDEFINED RISK" in ml:
            reason = "Undefined Risk"
        elif "POOR REWARD/RISK" in ml:
            reason = "Poor Reward:Risk"
        elif "SHORT STRIKE INSIDE EXPECTED MOVE" in ml:
            reason = "Short Strike Inside Expected Move"
        else:
            reason = "Unverified"
        return "❌ Skip", red, reason

    loss_pct = h.get("max_loss_pct_capital")
    if isinstance(loss_pct, (int, float)) and loss_pct > PER_HORIZON_CAP_PCT:
        return "❌ Skip", red, f"Exceeds {PER_HORIZON_CAP_PCT:.0f}% Per-Horizon Cap"

    ev_text = h.get("expected_value")
    if ev_text and "negative" in str(ev_text).lower():
        return "❌ Skip", red, "Negative EV"

    if h.get("_lottery_like"):
        return "⚠ Caution", amber, "Lottery-Like Payoff"

    if h.get("_implausible_reward_risk"):
        return "⚠ Caution", amber, "Reward:Risk Implausible -- Verify Premiums"

    score = h.get("_trade_quality_score")
    if not isinstance(score, (int, float)):
        return "Not Available", gray, "Quality Score Unavailable"

    if score >= CONSIDER_QUALITY_THRESHOLD:
        return "✅ Consider", green, None

    return "◐ Neutral", blue, f"Trade Quality {score:.0f}/100 -- Below Consider Threshold"


def render_recommendation_summary_table(horizons):
    """Top-of-report Horizon/Recommendation table so the reader gets a
    single fast-scan verdict before the detailed metrics below -- every
    verdict traces back to compute_horizon_recommendation, which only reads
    already-verified fields (plus the model's own no-strategy admission,
    checked first), so this table can never show "Consider" for a horizon
    the detailed card below marks rejected, negative-EV, over-cap, or
    "no specific strategy recommended"."""
    sans = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"

    by_name = {}
    for h in horizons:
        key = str(h.get("horizon") or "").strip()
        if key:
            by_name[key] = h

    def _cell(text, header=False, color="#14213D", bold=False):
        weight = "font-weight:700;" if (header or bold) else ""
        bg = "background:#14213D;color:#ffffff;" if header else f"color:{color};"
        border = "" if header else "border-top:1px solid #EDEAE2;"
        return (
            f'<td style="padding:8px 12px;font-size:12px;{weight}font-family:{sans};'
            f'{bg}{border}text-align:left;">{text}</td>'
        )

    header_row = f'<tr>{_cell("Horizon", header=True)}{_cell("Recommendation", header=True)}</tr>'

    rows = []
    for label in HORIZON_ORDER:
        h = by_name.get(label)
        if not h:
            rows.append(
                f'<tr>{_cell(f"<b>{html.escape(label)}</b>")}'
                f'{_cell("Not Available", color="#8A8F9C")}</tr>'
            )
            continue
        verdict, color, reason = compute_horizon_recommendation(h)
        display = f"{verdict} ({html.escape(reason)})" if reason else verdict
        rows.append(
            f'<tr>{_cell(f"<b>{html.escape(label)}</b>")}'
            f'{_cell(display, color=color, bold=True)}</tr>'
        )

    if not rows:
        return ""

    return f"""
<div style="margin-bottom:16px;">
  <div style="font-family:{sans};font-size:11px;font-weight:700;color:#14213D;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">Overall Recommendation</div>
  <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="border-collapse:collapse;border:1px solid #E7E4DC;border-radius:4px;overflow:hidden;">
    {header_row}
    {''.join(rows)}
  </table>
</div>
"""


def _suggested_action_for_score(score):
    """Maps a Trade Quality Score to a coarse, easy-to-scan sizing action --
    four bands instead of a bare number plus a vague "manual review" aside.
    This is a quick-glance heuristic keyed ONLY to the score, not a
    replacement for compute_suggested_sizing's actual lot-count math (which
    also accounts for per-horizon/aggregate capital caps and the
    MAX_LOTS_PER_HORIZON liquidity ceiling) -- see the Suggested Sizing
    section below for the real, capital-based lot recommendation. Bands:
      80-100 -> Full Size   (highest-conviction tier)
      60-79  -> Half Size   (tradeable, but reduce size)
      40-59  -> Watchlist   (not tradeable as-is; track and revisit)
      <40    -> Skip        (score alone argues against entry)
    """
    if score >= 80:
        return "Full Size", "#2F5233"
    if score >= 60:
        return "Half Size", "#3D6690"
    if score >= 40:
        return "Watchlist", "#A6812F"
    return "Skip", "#8B2E2E"


def render_trade_quality_table(horizons):
    """Simple Horizon / Trade Quality Score table -- the headline number
    from compute_trade_quality_score for each horizon, with a colored bar
    so relative quality is scannable at a glance, plus a bucketed
    Suggested Action column (Full Size / Half Size / Watchlist / Skip, see
    _suggested_action_for_score) so a reader gets a direct read on what to
    do with the number instead of just the score itself. Full component
    breakdown (EV/R:R/POP/Confidence/Liquidity/OI Alignment) is shown
    per-horizon in the detailed card below via the same score's stored
    breakdown text, so a reader who wants "why" doesn't have to leave this
    table to get it -- they just scroll down to the matching horizon's
    card."""
    sans = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"

    by_name = {}
    for h in horizons:
        key = str(h.get("horizon") or "").strip()
        if key:
            by_name[key] = h

    def _cell(text, header=False, color="#14213D"):
        weight = "font-weight:700;" if header else ""
        bg = "background:#14213D;color:#ffffff;" if header else f"color:{color};"
        border = "" if header else "border-top:1px solid #EDEAE2;"
        return (
            f'<td style="padding:8px 12px;font-size:12px;{weight}font-family:{sans};'
            f'{bg}{border}text-align:left;">{text}</td>'
        )

    def _score_color(score):
        if score >= 70:
            return "#2F5233"
        if score >= 45:
            return "#A6812F"
        return "#8B2E2E"

    def _bar_cell(score):
        color = _score_color(score)
        filled = round(score)
        return (
            f'<td style="padding:8px 12px;border-top:1px solid #EDEAE2;">'
            f'<div style="position:relative;height:8px;min-width:44px;max-width:100px;'
            f'background:#EDEAE2;border-radius:4px;overflow:hidden;display:inline-block;'
            f'vertical-align:middle;width:100%;">'
            f'<div style="width:{filled}%;height:100%;background:{color};"></div></div>'
            f'<div style="font-family:{sans};font-size:12px;font-weight:700;color:{color};'
            f'white-space:nowrap;margin-top:3px;">{score}/100</div></td>'
        )

    def _action_cell(score):
        label, color = _suggested_action_for_score(score)
        return (
            f'<td style="padding:8px 12px;border-top:1px solid #EDEAE2;">'
            f'<span style="font-family:{sans};font-size:12px;font-weight:700;color:{color};'
            f'white-space:nowrap;">{label}</span></td>'
        )

    header_row = (
        f'<tr>{_cell("Horizon", header=True)}{_cell("Trade Quality", header=True)}'
        f'{_cell("Suggested Action", header=True)}</tr>'
    )

    rows = []
    for label in HORIZON_ORDER:
        h = by_name.get(label)
        if not h:
            rows.append(
                f'<tr>{_cell(f"<b>{html.escape(label)}</b>")}'
                f'{_cell("Not Available", color="#8A8F9C")}'
                f'{_cell("&mdash;", color="#8A8F9C")}</tr>'
            )
            continue
        score = h.get("_trade_quality_score")
        if score is None:
            rows.append(
                f'<tr>{_cell(f"<b>{html.escape(label)}</b>")}'
                f'{_cell("N/A (Rejected)", color="#8A8F9C")}'
                f'{_cell("Skip", color="#8B2E2E")}</tr>'
            )
            continue
        rows.append(
            f'<tr>{_cell(f"<b>{html.escape(label)}</b>")}{_bar_cell(score)}{_action_cell(score)}</tr>'
        )

    if not rows:
        return ""

    return f"""
<div style="margin-bottom:16px;">
  <div style="font-family:{sans};font-size:11px;font-weight:700;color:#14213D;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">Trade Quality Score</div>
  <div style="font-family:{sans};font-size:10px;color:#8A8F9C;margin-bottom:6px;">Composite of Expected Value (30%), Reward:Risk (20%), Probability of Profit (15%), Confidence (15%), Liquidity (10%), and OI Alignment (10%) -- see each horizon's card below for the exact component breakdown. Suggested Action is a quick-glance score band (80-100 Full Size &middot; 60-79 Half Size &middot; 40-59 Watchlist &middot; &lt;40 Skip); see the Suggested Sizing section below for the actual lot count, which also accounts for capital caps and the per-horizon liquidity ceiling.</div>
  <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="border-collapse:collapse;border:1px solid #E7E4DC;border-radius:4px;overflow:hidden;">
    {header_row}
    {''.join(rows)}
  </table>
</div>
"""


def render_strategy_summary_table(horizons):
    """One-line-per-horizon overview table (Horizon / Strategy / Bias /
    Expected Move / POP / Max Profit / Profit % / Max Loss / Loss % / EV /
    R:R / Confidence) shown before the detailed per-horizon cards, so the
    reader gets the shape of the whole recommendation at a glance. Every
    cell is extracted from the same authoritative, already-computed values
    shown in the detailed cards below (never recomputed or re-derived
    here) -- so the summary can never drift from or contradict the detail
    it's summarizing."""
    sans = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"

    by_name = {}
    for h in horizons:
        key = str(h.get("horizon") or "").strip()
        if key:
            by_name[key] = h

    def _extract(pattern, text, default="n/a"):
        m = pattern.search(text or "")
        return m.group(0) if m else default

    def _pct(value):
        # max_loss_pct_capital / max_profit_pct_capital are either a real
        # rounded float (from apply_verified_payoff) or the literal string
        # "n/a" for a rejected/unverified horizon -- never guess a number
        # for the latter case.
        if isinstance(value, (int, float)):
            return f"{value:g}%"
        return "n/a"

    def _cell(text, header=False, color="#14213D", label=None):
        weight = "font-weight:700;" if header else ""
        bg = "background:#14213D;color:#ffffff;" if header else f"color:{color};"
        border = "" if header else "border-top:1px solid #EDEAE2;"
        data_label = f' data-label="{html.escape(label)}"' if (label and not header) else ""
        return (
            f'<td{data_label} style="padding:7px 10px;font-size:11px;{weight}font-family:{sans};'
            f'{bg}{border}text-align:left;">{text}</td>'
        )

    columns = ("Horizon", "Strategy", "Bias", "Expected Move", "POP", "Max Profit",
               "Profit %", "Max Loss", "Loss %", "EV", "R:R", "Confidence")
    header_cells = "".join(_cell(h, header=True) for h in columns)
    header_row = f'<tr>{header_cells}</tr>'

    body_rows = []
    for label in HORIZON_ORDER:
        h = by_name.get(label)
        if not h:
            continue
        strategy = _esc(h.get("strategy_name"))
        bias = _esc(h.get("bias"))
        exp_move = _extract(_EXPECTED_MOVE_RE, h.get("expected_move"))
        pop = _extract(_POP_RE, h.get("probability_of_profit"))
        max_profit = _extract(_RUPEE_FIGURE_RE, h.get("max_profit"))
        max_loss = _extract(_RUPEE_FIGURE_RE, h.get("max_loss"))
        profit_pct = _pct(h.get("max_profit_pct_capital"))
        loss_pct = _pct(h.get("max_loss_pct_capital"))
        ev_text = h.get("expected_value")
        ev = _extract(_EV_FIGURE_RE, ev_text)
        is_negative_ev = bool(h.get("_negative_ev"))
        ev_color = "#8B2E2E" if is_negative_ev else ("#2F5233" if ev != "n/a" else "#8B2E2E")
        ev_display = f"❌ Avoid ({ev})" if is_negative_ev and ev != "n/a" else ev
        rr = _extract(_RR_RATIO_RE, h.get("reward_risk_ratio"))
        rr_display = f"{rr}" if rr != "n/a" else "n/a"
        conf_pct = h.get("confidence_pct")
        confidence = _esc(h.get("confidence"))
        if isinstance(conf_pct, (int, float)) and confidence != "—":
            confidence = f"{confidence} ({conf_pct}%)"
        cells = "".join([
            _cell(f"<b>{html.escape(label)}</b>", label="Horizon"),
            _cell(html.escape(strategy), label="Strategy"),
            _cell(html.escape(bias), label="Bias"),
            _cell(html.escape(exp_move), label="Expected Move"),
            _cell(html.escape(pop), label="POP"),
            _cell(html.escape(max_profit), color="#2F5233", label="Max Profit"),
            _cell(html.escape(profit_pct), color="#2F5233", label="Profit %"),
            _cell(html.escape(max_loss), color="#8B2E2E", label="Max Loss"),
            _cell(html.escape(loss_pct), color="#8B2E2E", label="Loss %"),
            _cell(html.escape(ev_display), color=ev_color, header=False, label="EV"),
            _cell(html.escape(rr_display), label="R:R"),
            _cell(html.escape(confidence), label="Confidence"),
        ])
        body_rows.append(f'<tr>{cells}</tr>')

    if not body_rows:
        return ""

    return f"""
<div style="margin-bottom:16px;overflow-x:auto;-webkit-overflow-scrolling:touch;">
  <table class="responsive-table" width="100%" cellpadding="0" cellspacing="0" role="presentation" style="border-collapse:collapse;border:1px solid #E7E4DC;border-radius:4px;overflow:hidden;">
    <thead>{header_row}</thead>
    <tbody>{''.join(body_rows)}</tbody>
  </table>
</div>
"""


def compute_suggested_sizing(horizons):
    """Deterministic per-horizon lot-sizing suggestion -- code-derived from
    already-verified max_loss figures and the two capital caps this file
    already enforces (PER_HORIZON_CAP_PCT for a single horizon,
    AGGREGATE_CAP_PCT for all three combined), never from the model's own
    text. This turns "exceeds the aggregate cap" from a bare warning into
    something directly actionable: how many lots (or none at all) to
    actually place per horizon.

    Two passes:
      1. Per-horizon cap: a rejected/negative-EV/low-quality horizon (per
         compute_horizon_recommendation) is sized 0 outright. Otherwise,
         lots = floor(PER_HORIZON_CAP_PCT% of capital / that horizon's own
         verified per-lot max loss) -- if even 1 lot alone breaches the
         per-horizon cap, it's skipped. That figure is then hard-capped at
         MAX_LOTS_PER_HORIZON regardless of how small max_loss is (issue
         #6) -- a tiny per-lot max loss can make the %-of-capital math
         alone suggest an impractically large lot count, ignoring
         liquidity/slippage/execution/margin/gap risk, which don't shrink
         just because max_loss per lot is small.
      2. Aggregate cap: if the combined worst-case loss at that sizing
         still exceeds AGGREGATE_CAP_PCT, lots are trimmed one at a time
         from the LOWEST Trade-Quality-Score horizon first (weakest trade
         gives up size before a stronger one does), down to zero if
         necessary, until the combined figure fits.

    Returns an ordered list of (horizon_name, lots:int|None, note:str|None)
    tuples in HORIZON_ORDER -- lots is None for "skip this horizon
    entirely", with note explaining why."""
    by_name = {str(h.get("horizon") or "").strip(): h for h in horizons}
    per_horizon_cap_inr = PER_HORIZON_CAP_PCT / 100 * TOTAL_CAPITAL_INR
    aggregate_cap_inr = AGGREGATE_CAP_PCT / 100 * TOTAL_CAPITAL_INR

    plan = []
    for name in HORIZON_ORDER:
        h = by_name.get(name)
        if not h:
            plan.append([name, None, "Not available this run"])
            continue
        verdict, _color, reason = compute_horizon_recommendation(h)
        if verdict in ("❌ Skip", "⚪ No Trade", "Not Available"):
            plan.append([name, None, reason])
            continue
        max_loss_per_lot = h.get("_verified_max_loss_inr")
        if not isinstance(max_loss_per_lot, (int, float)) or max_loss_per_lot <= 0:
            plan.append([name, None, "Unverified max loss"])
            continue
        lots = int(per_horizon_cap_inr // max_loss_per_lot)
        if lots < 1:
            plan.append([name, None, f"Even 1 lot exceeds the {PER_HORIZON_CAP_PCT:.0f}% per-horizon cap"])
            continue
        action_label, _action_color = _suggested_action_for_score(h.get("_trade_quality_score", 0) or 0)
        note = None if verdict == "✅ Consider" else f"{reason} -- Trade Quality suggests: {action_label}"
        if lots > MAX_LOTS_PER_HORIZON:
            # The %-of-capital cap alone allowed more lots than is
            # practical to actually execute (liquidity/slippage/margin/
            # gap risk don't scale down just because per-lot max loss is
            # small) -- hard-cap it and say so, rather than silently
            # suggesting a size a retail account likely can't fill cleanly.
            lots = MAX_LOTS_PER_HORIZON
            cap_note = (
                f"Capped at {MAX_LOTS_PER_HORIZON} lots (liquidity/slippage/margin/gap-risk "
                f"ceiling) -- the {PER_HORIZON_CAP_PCT:.0f}% capital cap alone would have allowed "
                f"more; increase OPTIONS_MAX_LOTS_PER_HORIZON if you want to override this"
            )
            note = f"{note}. {cap_note}" if note else cap_note
        plan.append([name, lots, note])

    def total_risk():
        return sum(
            by_name[name].get("_verified_max_loss_inr", 0.0) * lots
            for name, lots, _note in plan if lots
        )

    def quality(name):
        h = by_name.get(name) or {}
        s = h.get("_trade_quality_score")
        return s if s is not None else -1

    guard = 0
    while total_risk() > aggregate_cap_inr and guard < 50:
        guard += 1
        candidates = [row for row in plan if row[1]]
        if not candidates:
            break
        candidates.sort(key=lambda row: quality(row[0]))
        weakest = candidates[0]
        if weakest[1] > 1:
            weakest[1] -= 1
            weakest[2] = "Reduced size to stay within the {:.0f}% aggregate cap".format(AGGREGATE_CAP_PCT)
        else:
            weakest[1] = None
            weakest[2] = f"Skipped to stay within the {AGGREGATE_CAP_PCT:.0f}% aggregate cap"

    return [tuple(row) for row in plan]


def render_suggested_sizing_html(plan, sans):
    """Renders compute_suggested_sizing's plan as the small, directly
    actionable "Suggested Sizing" block the reader can act on immediately
    -- lots per horizon, or an explicit Skip with the real reason."""
    lines = []
    for name, lots, note in plan:
        if lots:
            text = f"{lots} lot{'s' if lots != 1 else ''}"
            color = "#2F5233"
        else:
            text = f"Skip{f' ({html.escape(note)})' if note else ''}"
            color = "#8B2E2E"
        extra = (
            f' <span style="color:#8A6D3B;">— {html.escape(note)}</span>'
            if lots and note else ""
        )
        lines.append(
            f'<div style="display:flex;justify-content:space-between;padding:3px 0;">'
            f'<span style="color:#4A5063;">{html.escape(name)}</span>'
            f'<span style="font-weight:700;color:{color};">{text}</span>{extra}</div>'
        )
    return (
        f'<div style="margin-top:10px;padding:10px 12px;background:#FFFFFF;'
        f'border:1px solid #EDEAE2;border-radius:4px;">'
        f'<div style="font-family:{sans};font-size:11px;font-weight:700;color:#14213D;'
        f'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">Suggested Sizing '
        f'<span style="text-transform:none;font-weight:400;color:#8A8F9C;">(₹{TOTAL_CAPITAL_INR:,.0f} capital pool assumed -- override with OPTIONS_TOTAL_CAPITAL_INR)</span></div>'
        f'<div style="font-family:{sans};font-size:12px;">{"".join(lines)}</div>'
        f'</div>'
    )


def render_horizons_html(horizons, aggregate_pct, portfolio_view, live_data=None):
    sans = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
    serif = "Georgia,'Times New Roman',serif"

    regime_text = compute_market_regime(live_data, horizons)
    regime_html = (
        f'<div style="margin-bottom:12px;padding:8px 14px;background:#F4F2ED;'
        f'border:1px solid #EDEAE2;border-radius:4px;font-family:{sans};font-size:12px;'
        f'font-weight:700;color:#14213D;letter-spacing:0.01em;">'
        f'Market Regime: <span style="font-weight:400;color:#4A5063;">{_esc(regime_text)}</span></div>'
        if regime_text else ""
    )

    # Render in the fixed Weekly -> Monthly -> Quarterly order
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
        f"⚠ EXCEEDS the {AGGREGATE_CAP_PCT:.0f}% worst-case combined cap -- reduce position size before entering."
        if over_cap else
        f"✅ Stays within the {AGGREGATE_CAP_PCT:.0f}% worst-case combined cap."
    )
    # Gamma is a real, code-computed number (from live IV via apply_verified_
    # payoff), so it gets the same treatment as the aggregate-cap verdict
    # above: the model's own gamma claim is stripped out and replaced with
    # the authoritative figure (issue #2).
    gamma_summary = compute_portfolio_gamma_summary(horizons)
    portfolio_text = _strip_gamma_claims(_strip_cap_claims(portfolio_view))
    # Fix any sentence that contradicts a horizon's own actual verdict --
    # e.g. portfolio_view claiming no structure was recommended for a
    # horizon that genuinely has one (Portfolio Summary Contradiction).
    portfolio_text = _scrub_portfolio_view_contradictions(portfolio_text, horizons)
    # Fix wrong specific structure names (e.g. "iron butterflies" for a
    # horizon that's actually an Iron Condor) BEFORE the broader
    # directional-stance check below, so the directional check only ever
    # sees already-correct structure names in its own comparisons.
    portfolio_text = _scrub_portfolio_view_structure_type_contradiction(portfolio_text, horizons)
    # Also fix mischaracterized directionality -- e.g. calling a horizon
    # "range-bound" when its own card shows a directional strategy
    # (Portfolio Description issue).
    portfolio_text = _scrub_portfolio_view_directional_contradiction(portfolio_text, horizons)
    sizing_plan = compute_suggested_sizing(horizons)
    sizing_html = render_suggested_sizing_html(sizing_plan, sans)
    portfolio_html = (
        f'<div style="margin-top:16px;padding:12px 14px;background:#FAF9F6;'
        f'border:1px solid #EDEAE2;border-left:3px solid #B08D57;border-radius:4px;">'
        f'<div style="font-family:{sans};font-size:11px;font-weight:700;color:#14213D;'
        f'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">Portfolio View '
        f'&nbsp;&middot;&nbsp; Worst-Case Combined Max Loss: '
        f'<span style="color:{agg_color};">{agg_display}%</span> '
        f'(cap {AGGREGATE_CAP_PCT:.0f}%)</div>'
        f'<div style="font-family:{sans};font-size:11px;color:#8A8F9C;font-style:italic;margin-bottom:8px;">'
        f'This is a conservative ceiling -- the plain sum of each horizon\'s own max loss, as if Weekly, '
        f'Monthly, and Quarterly ALL hit max loss simultaneously. It is not a probabilistic '
        f'estimate of actual combined portfolio risk, which would normally be lower since the horizons '
        f'won\'t all lose the maximum at the same time.</div>'
        f'<div style="font-family:{sans};font-size:12px;color:#4A5063;line-height:1.65;">{_esc(gamma_summary)}</div>'
        f'<div style="font-family:{sans};font-size:12px;color:#4A5063;line-height:1.65;margin-top:8px;">{_esc(portfolio_text)}</div>'
        f'<div style="font-family:{sans};font-size:12px;font-weight:700;color:{agg_color};margin-top:8px;">{verdict}</div>'
        f'{sizing_html}'
        f'</div>'
    )

    recommendation_table_html = render_recommendation_summary_table(horizons)
    trade_quality_table_html = render_trade_quality_table(horizons)
    summary_table_html = render_strategy_summary_table(horizons)
    return regime_html + recommendation_table_html + trade_quality_table_html + summary_table_html + cards + portfolio_html


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
            "\"Market Data Inputs\" and the live-feed summary above for what was actually used. Options-chain "
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

    sans = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
    serif = "Georgia,'Times New Roman',serif"
    sources_html = render_market_data_inputs_html(live_data, sources, sans)
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
  img {{ max-width:100%; height:auto; }}
  @media screen and (max-width:600px) {{
    body {{ -webkit-text-size-adjust:100%; }}
    .email-container {{ width:100% !important; max-width:100% !important; border-radius:0 !important; }}
    .email-padding {{ padding-left:14px !important; padding-right:14px !important; }}
    h1 {{ font-size:20px !important; }}

    /* Turn the wide multi-column strategy summary table into stacked
       label/value rows on phone screens instead of relying on
       overflow-x scrolling, which many mobile mail apps ignore. */
    table.responsive-table thead {{ display:none !important; }}
    table.responsive-table, table.responsive-table tbody,
    table.responsive-table tr, table.responsive-table td {{
      display:block !important; width:100% !important; box-sizing:border-box;
    }}
    table.responsive-table tr {{
      padding:8px 10px !important; border-top:1px solid #EDEAE2 !important;
    }}
    table.responsive-table tr:first-child {{ border-top:none !important; }}
    table.responsive-table td {{
      padding:2px 0 !important; border-top:none !important;
      text-align:right !important; position:relative;
      padding-left:46% !important; min-height:18px;
    }}
    table.responsive-table td[data-label]:before {{
      content: attr(data-label);
      position:absolute; left:0; top:2px; width:44%;
      text-align:left; font-weight:700; color:#8A8F9C;
      font-size:10px; text-transform:uppercase; letter-spacing:0.03em;
    }}

    /* Other data tables keep their columns but get tighter padding so
       labels and numbers fit comfortably on a phone-width screen. */
    table:not(.responsive-table) td {{ padding:6px 8px !important; }}
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
              <p style="margin:6px 0 0;font-family:{sans};font-size:12px;color:#B7BEC9;">Weekly &middot; Monthly &middot; Quarterly &mdash; Risk-Defined Only</p>
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
# Post-processing: real aggregate-risk arithmetic
# -----------------------------
def finalize_horizons(horizons, live_data):
    """Runs every horizon's legs through apply_verified_payoff() (real
    premium-based math, not LLM arithmetic), and recomputes the aggregate
    capital-at-risk as a plain sum of the per-horizon figures instead of
    trusting a self-reported total (this is what issue #8's arithmetic
    mismatch traces back to). Returns (horizons, aggregate_pct, over_cap)."""
    by_name = {str(h.get("horizon") or "").strip(): h for h in horizons}

    # A single week's move is not a quantitative basis for predicting the
    # following week's direction (issue #1) -- regardless of what the model
    # wrote for next_week_bias, or whether it complied with constraint #5's
    # instruction to leave it blank, force it to an honest, non-predictive
    # placeholder here so it can never contradict itself in the rendered
    # report the way "Current Week Bullish / Next Week Bearish (mean
    # reversion)" did.
    if "Weekly" in by_name:
        by_name["Weekly"]["next_week_bias"] = "Neutral (insufficient evidence)"

    total_at_risk = 0.0
    for name, h in by_name.items():
        if h.get("_verified_max_loss_inr") is not None:
            continue
        snap = (live_data.get("horizons") or {}).get(name, {})
        apply_verified_payoff(h, snap, live_data.get("spot"), live_data.get("vix"))

    for h in by_name.values():
        v = h.get("_verified_max_loss_inr", 0.0)
        if v not in (None, float("inf")):
            total_at_risk += v

    aggregate_pct = round(total_at_risk / TOTAL_CAPITAL_INR * 100, 2) if TOTAL_CAPITAL_INR else 0.0
    over_cap = aggregate_pct > AGGREGATE_CAP_PCT

    ordered = [by_name[h] for h in HORIZON_ORDER if h in by_name]
    return ordered, aggregate_pct, over_cap


def _horizon_rejected(h):
    """True if apply_verified_payoff marked this horizon's legs as
    invalid -- either genuinely unparseable/unpriceable ('Unverified'),
    structurally naked/unbounded ('UNDEFINED RISK'), capped-risk but a bad
    trade on reward-quality grounds ('POOR REWARD/RISK'), or an Iron Condor
    whose short strike sits inside the 1-sigma expected move
    ('SHORT STRIKE INSIDE EXPECTED MOVE', see REJECT_IC_SHORT_INSIDE_EM)."""
    ml = h.get("max_loss")
    return isinstance(ml, str) and (
        "UNDEFINED RISK" in ml or "Unverified" in ml or "POOR REWARD/RISK" in ml
        or "SHORT STRIKE INSIDE EXPECTED MOVE" in ml
    )


def reverify_horizons(horizons, live_data, only_names=None):
    """Re-runs apply_verified_payoff (real premium-based payoff math) for
    the given horizons -- or all of them if only_names is None -- and
    recomputes the aggregate capital-at-risk from scratch. Used after the
    single repair-and-retry pass in run() so the aggregate never drifts
    out of sync with whatever legs actually ended up in the rendered
    report. Unlike finalize_horizons, this does NOT skip horizons that
    already have _verified_max_loss_inr set -- it's meant to be called
    specifically on horizons whose legs just changed."""
    for h in horizons:
        name = h.get("horizon")
        if only_names is not None and name not in only_names:
            continue
        snap = (live_data.get("horizons") or {}).get(name, {})
        apply_verified_payoff(h, snap, live_data.get("spot"), live_data.get("vix"))

    total_at_risk = 0.0
    for h in horizons:
        v = h.get("_verified_max_loss_inr", 0.0)
        if v not in (None, float("inf")):
            total_at_risk += v
    aggregate_pct = round(total_at_risk / TOTAL_CAPITAL_INR * 100, 2) if TOTAL_CAPITAL_INR else 0.0
    over_cap = aggregate_pct > AGGREGATE_CAP_PCT
    return aggregate_pct, over_cap


def build_repair_prompt(rejected_horizons, live_data):
    """Builds a short, targeted follow-up prompt that shows the model
    exactly which of its own legs were rejected and why, plus the same
    live data feed, and asks it to fix ONLY those horizons' legs. This is
    the generator-side fix issue #3 asked for: rather than relying on the
    validator to catch nonsense legs run after run, give the model one
    chance to self-correct against its own concrete mistake before the
    report gives up on that horizon."""
    live_data_block = format_live_data_block(live_data)
    bad_lines = []
    for h in rejected_horizons:
        bad_lines.append(
            f"- {h.get('horizon')}: strategy_name='{h.get('strategy_name')}', "
            f"legs='{h.get('legs')}', bias='{h.get('bias')}' -- REJECTED: {h.get('verification')}"
        )
    bad_block = "\n".join(bad_lines)

    return f"""You previously generated a Nifty options strategy report. The horizons below were REJECTED because their legs did not form a valid risk-defined structure. Fix ONLY the legs for these horizons, using the same live data feed as before -- do not change any other horizon.

{live_data_block}

REJECTED HORIZONS TO FIX:
{bad_block}

RULES YOU MUST FOLLOW EXACTLY:
1. A 2-leg vertical spread must be CE+CE or PE+PE, same expiry, one Buy + one Sell, different strikes. Never pair one CE leg with an unrelated PE leg -- e.g. "Sell 24600 CE, Buy 24800 PE" is INVALID (naked short call plus an unrelated long put, not a spread at all). Valid: "Sell 24600 CE, Buy 24800 CE".
2. A 4-leg Iron Condor/Butterfly must pair legs by ROLE, not by strike: the two SHORT legs (one CE, one PE) sit together nearer spot; the two LONG legs (one CE, one PE) sit together as the outer wings, one above and one below. Never write both CE legs with the same action and both PE legs with the same action -- e.g. "Sell 24000 CE, Buy 24000 PE, Sell 24200 CE, Buy 24200 PE" is INVALID. Valid: "Buy 24000 PE, Sell 24200 PE, Sell 24500 CE, Buy 24700 CE".
3. Use ONLY strikes shown in the live data feed above for that horizon. Keep the SAME directional bias you originally chose unless the live data genuinely cannot support any valid capped-risk structure at that bias -- if so, say so via a Neutral/range-bound structure instead.
4. If the rejection reason mentions poor Reward:Risk or poor credit relative to spread width, the structure was too WIDE for the premium it collects (or paid too much relative to potential reward) -- fix this by choosing strikes CLOSER TOGETHER (a narrower vertical spread width) and/or moving the short strike nearer the current expected-move band so the premium collected is proportionate to the capital at risk. Do not fix this by widening risk or removing a hedge leg -- keep the structure fully risk-defined.
5. If the rejection reason says a short strike is inside the expected move for an Iron Condor, this is the OPPOSITE problem from #4 above: move BOTH short strikes further from spot (outside the ATM straddle's expected-move band shown in the live data feed) while keeping the same wing strikes or moving the wings out proportionally, so the structure keeps its risk-defined shape but no longer contradicts its own range-bound thesis. If no listed liquid strike outside the band exists for this horizon, say so explicitly instead of forcing a strike back inside the band.

Respond with ONLY raw JSON (no markdown, no commentary before or after) in exactly this shape, one object per rejected horizon listed above, same order:
{{
  "horizons": [
    {{"horizon": "<name, exactly as listed above>", "strategy_name": "<corrected strategy name>", "legs": "<corrected legs string>"}}
  ]
}}
"""


def repair_rejected_legs(horizons, live_data):
    """Single self-correction retry (not a loop): if any horizon came back
    rejected from finalize_horizons, send the model one targeted follow-up
    showing its own bad legs and the exact rule it broke, then re-verify
    only the patched horizons against real premiums again. If the model's
    repair prompt fails to parse, or still comes back invalid on
    re-verification, the original rejection stands -- this deliberately
    does not retry more than once, so a persistently confused backend
    still degrades to a clearly-labeled rejected trade rather than looping
    indefinitely."""
    rejected = [h for h in horizons if _horizon_rejected(h)]
    if not rejected:
        return horizons

    repair_prompt = build_repair_prompt(rejected, live_data)
    try:
        repair_text, _repair_sources, _repair_used_search = swing.generate_analysis(repair_prompt)
    except Exception:
        main.log.warning("Repair pass call failed; keeping original rejection(s).")
        return horizons

    if not repair_text:
        main.log.warning("Repair pass produced no output; keeping original rejection(s).")
        return horizons

    fixed_by_name = {}
    try:
        cleaned = swing._strip_code_fences(repair_text)
        data = json.loads(cleaned)
        for item in data.get("horizons", []) if isinstance(data, dict) else []:
            name = str(item.get("horizon") or "").strip()
            if name and item.get("legs"):
                fixed_by_name[name] = item
    except (json.JSONDecodeError, AttributeError, TypeError):
        main.log.warning("Repair pass returned unparseable JSON; keeping original rejection(s).")
        return horizons

    if not fixed_by_name:
        return horizons

    for h in horizons:
        fix = fixed_by_name.get(h.get("horizon"))
        if fix:
            h["legs"] = fix["legs"]
            if fix.get("strategy_name"):
                h["strategy_name"] = fix["strategy_name"]

    reverify_horizons(horizons, live_data, only_names=set(fixed_by_name.keys()))
    still_bad = [h.get("horizon") for h in horizons if h.get("horizon") in fixed_by_name and _horizon_rejected(h)]
    if still_bad:
        main.log.warning(f"Repair pass attempted but still rejected after retry: {', '.join(still_bad)}")
    return horizons


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


def _source_url_title(s):
    """Normalizes whatever shape a single 'source' entry comes in as --
    a dict (url/link + title/name keys), a (title, url) or (url, title)
    2-tuple/list (some live-search backends return these instead of
    dicts), or a plain URL string -- into a consistent (url, title) pair.

    Without this, a bare tuple like ('FII/DII Trading Activity - July
    2026', 'https://...') fell through to `str(s)` on the whole tuple,
    which is exactly why the FII/DII row was showing raw Python
    tuple-repr text instead of a clean link."""
    if isinstance(s, dict):
        url = str(s.get("url") or s.get("link") or "")
        title = str(s.get("title") or s.get("name") or url)
        return url, title
    if isinstance(s, (tuple, list)) and len(s) == 2:
        a, b = s
        # Whichever element looks like a URL is the url; the other is the title.
        if isinstance(a, str) and a.strip().lower().startswith(("http://", "https://")):
            return a, (str(b) if b else a)
        if isinstance(b, str) and b.strip().lower().startswith(("http://", "https://")):
            return b, (str(a) if a else b)
        # Neither looks like a URL -- fall back to treating the first as title.
        return "", str(a)
    text = str(s)
    return (text, text) if text.strip().lower().startswith(("http://", "https://")) else ("", text)


def _filter_sources(sources):
    """Defensive filter over whatever shape swing.generate_analysis()
    returns for 'sources' (list of dicts with a url/link key, (title, url)
    tuples, or plain strings). swing.generate_analysis() is shared with
    swing_trade_advisor.py (a stock-picking report), so its own live-search
    path can just as easily surface breakout-stock roundups, broker
    marketing blogs, or generic "swing trading" content -- none of which
    should back a derivatives report (issue #6). This is now a strict
    ALLOWLIST: a source survives only if its URL matches one of the
    recognized official/primary NSE-adjacent or major-financial-press
    domains in _OFFICIAL_SOURCE_DOMAINS, on top of the existing
    social-media/video blocklist. Anything else -- including a
    legitimate-looking source the allowlist doesn't happen to name -- is
    dropped rather than risk showing an unvetted stock-picking link next
    to option strikes."""
    if not sources:
        return sources
    out = []
    for s in sources:
        url, _title = _source_url_title(s)
        low_url = url.lower()
        if any(bad in low_url for bad in _BLOCKED_SOURCE_HINTS):
            continue
        if not any(dom in low_url for dom in _OFFICIAL_SOURCE_DOMAINS):
            continue
        out.append(s)
    return out


# -----------------------------
# Market Data Inputs (replaces the generic "Sources Consulted" link-dump)
# -----------------------------
_CATEGORY_KEYWORDS = {
    "Nifty Futures": ("nifty futures", "futures price", "futures basis"),
    "FII/DII Activity": (
        "fii", "dii", "foreign institutional", "foreign portfolio investor",
        "domestic institutional", "fii/dii", "fpi",
    ),
    "GIFT Nifty / Pre-Market": ("gift nifty", "sgx nifty", "gift city", "pre-market", "premarket"),
    "Event Calendar": (
        "rbi", "fomc", "federal reserve", "fed meeting", "monetary policy",
        "earnings calendar", "union budget", "budget session", "election",
    ),
}


def _categorize_source(s):
    """Buckets an already domain-allowlisted source (see _filter_sources)
    into one of the Market Data Inputs categories below by keyword match on
    its title/URL. Returns None if it doesn't match a recognized category --
    callers drop those rather than showing them under a generic catch-all
    bucket, so a stray "top breakout stocks this week" link that happened
    to sit on an allowlisted domain still can't surface here."""
    url, title = _source_url_title(s)
    text = f"{url} {title}".lower()
    for cat, keywords in _CATEGORY_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return cat
    return None


def render_market_data_inputs_html(live_data, sources, sans):
    """Structured manifest of the actual data used this run -- Spot,
    Futures, Option Chain/Bhavcopy, India VIX, FII/DII Activity, GIFT
    Nifty, and Event Calendar -- each attributed to where it really came
    from, replacing the old generic "Sources Consulted" link-dump. Spot,
    VIX, and Option Chain/Bhavcopy are deterministic (always known one way
    or another, from the direct NSE/Yahoo fetch) and always shown, even
    when unavailable, since those are core inputs the reader should always
    see the status of. The other four are search-only categories and are
    shown only when a source was actually found and matched to that
    category (issue #8) -- a category with nothing found that run (e.g.
    FII/DII when both the direct NSE fetch and the LLM's own search came
    up empty) is dropped from the table entirely rather than shown as a
    "not found" row, so the manifest only lists what actually informed
    the report.
    """
    live_data = live_data or {}

    def _row(label, value_html):
        return (
            f'<tr><td style="padding:6px 10px;font-size:12px;font-family:{sans};'
            f'color:#4A5063;border-top:1px solid #EDEAE2;width:34%;">{label}</td>'
            f'<td style="padding:6px 10px;font-size:12px;font-family:{sans};'
            f'color:#14213D;border-top:1px solid #EDEAE2;">{value_html}</td></tr>'
        )

    spot_val = _esc(live_data.get("spot"))
    spot_src = _esc(live_data.get("spot_source") or "unavailable this run")
    vix_val = _esc(live_data.get("vix"))
    vix_src = _esc(live_data.get("vix_source") or "unavailable this run")
    oc_src = _esc(
        live_data.get("option_chain_source")
        or "unavailable this run (direct fetch and EOD Bhavcopy fallback both failed)"
    )

    fii_val = live_data.get("fii_net_cr")
    dii_val = live_data.get("dii_net_cr")
    fii_dii_date = live_data.get("fii_dii_date")
    fii_dii_src = live_data.get("fii_dii_source")

    rows = [
        _row("Spot", f'{spot_val} &nbsp;&middot;&nbsp; <span style="color:#8A8F9C;">{spot_src}</span>'),
        _row("Option Chain / Bhavcopy", f'<span style="color:#8A8F9C;">{oc_src}</span>'),
        _row("India VIX", f'{vix_val} &nbsp;&middot;&nbsp; <span style="color:#8A8F9C;">{vix_src}</span>'),
    ]

    iv_rank = live_data.get("iv_rank")
    iv_pct = live_data.get("iv_percentile")
    iv_days = live_data.get("iv_rank_days") or 0
    if iv_rank is not None and iv_pct is not None:
        iv_cell = (
            f'{iv_rank:g} <span style="color:#8A8F9C;">Rank</span> &nbsp;&middot;&nbsp; '
            f'{iv_pct:g} <span style="color:#8A8F9C;">Percentile</span><br>'
            f'<span style="color:#8A8F9C;font-size:11px;">Estimated from India VIX vs its trailing '
            f'{iv_days}-day range (not per-expiry historical IV)</span>'
        )
        rows.append(_row("IV Rank / Percentile", iv_cell))

    by_cat = {}
    for s in (sources or []):
        cat = _categorize_source(s)
        if cat:
            by_cat.setdefault(cat, []).append(s)

    def _links_cell(cat_sources):
        links = []
        for s in cat_sources[:3]:
            url, title = _source_url_title(s)
            if url:
                links.append(f'<a href="{html.escape(url)}" style="color:#8A6D3B;">{html.escape(title)}</a>')
            elif title:
                links.append(html.escape(title))
        if links:
            return "<br>".join(links)
        return None  # nothing found -- caller skips the row entirely

    def _flow_cell(val, label):
        if val is None:
            return None
        sign = "+" if val >= 0 else "&minus;"
        color = "#2F5233" if val >= 0 else "#8B2E2E"
        return f'<span style="color:{color};font-weight:700;">{sign}₹{abs(val):,.0f} Cr</span> <span style="color:#8A8F9C;">({label})</span>'

    if fii_val is not None or dii_val is not None:
        # Real code-fetched net figures (NSE's own FII/DII report) -- shown
        # as structured Net ₹ Cr numbers with a clean source line, instead
        # of the raw search-result link dump this category used to fall
        # back to whenever the LLM's own web search was all that was
        # available.
        parts = [p for p in (_flow_cell(fii_val, "FII, cash mkt"), _flow_cell(dii_val, "DII, cash mkt")) if p]
        date_note = f" &middot; {html.escape(str(fii_dii_date))}" if fii_dii_date else ""
        cell = (
            "<br>".join(parts)
            + f'<div style="margin-top:2px;color:#8A8F9C;font-size:11px;">Source: {html.escape(str(fii_dii_src or "NSE"))}{date_note}</div>'
        )
        rows.append(_row("FII/DII Activity", cell))
    else:
        # Direct NSE fetch failed this run -- fall back to whatever the
        # LLM's own live search turned up, same as the other search-only
        # categories below. If that also came up empty, drop the row
        # entirely (issue #8) rather than show a "not found" placeholder --
        # a cleaner manifest of what data actually informed the report,
        # instead of padding it out with rows that say nothing was there.
        fallback_cell = _links_cell(by_cat.get("FII/DII Activity", []))
        if fallback_cell is not None:
            rows.append(_row("FII/DII Activity", fallback_cell))

    for label, cat_key in (
        ("Nifty Futures", "Nifty Futures"),
        ("GIFT Nifty", "GIFT Nifty / Pre-Market"),
        ("Event Calendar", "Event Calendar"),
    ):
        cell = _links_cell(by_cat.get(cat_key, []))
        if cell is not None:
            rows.append(_row(label, cell))

    return f"""
<div style="margin-top:16px;">
  <div style="font-family:{sans};font-size:11px;font-weight:700;color:#14213D;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">Market Data Inputs</div>
  <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="border-collapse:collapse;border:1px solid #EDEAE2;border-radius:4px;overflow:hidden;">
    {''.join(rows)}
  </table>
</div>
"""


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
        horizons = repair_rejected_legs(horizons, live_data)
        # Recompute the aggregate/over-cap verdict in case the repair pass
        # changed any horizon's verified max loss (a fixed leg goes from
        # "inf" / excluded to a real capital-at-risk number).
        aggregate_pct, over_cap = reverify_horizons(horizons, live_data, only_names=None)
        if over_cap:
            main.log.warning(
                f"Computed worst-case combined max loss ({aggregate_pct}%) exceeds the "
                f"{AGGREGATE_CAP_PCT:.0f}% cap -- flagging in the report rather than silently sending."
            )
        horizons_html = render_horizons_html(horizons, aggregate_pct, portfolio_view, live_data)
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