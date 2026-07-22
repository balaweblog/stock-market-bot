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
from compliance import build_compliance_block_html

# -----------------------------
# Config (env-overridable capital caps, per the prompt's constraint #3)
# -----------------------------
PER_HORIZON_CAP_PCT = float(os.getenv("OPTIONS_PER_HORIZON_CAP_PCT", "5"))
AGGREGATE_CAP_PCT = float(os.getenv("OPTIONS_AGGREGATE_CAP_PCT", "15"))

# Absolute lot ceiling per horizon, independent of the PER_HORIZON_CAP_PCT
# math above -- issue #6.
MAX_LOTS_PER_HORIZON = int(os.getenv("OPTIONS_MAX_LOTS_PER_HORIZON", "5"))

MIN_REWARD_RISK_RATIO = float(os.getenv("OPTIONS_MIN_REWARD_RISK_RATIO", "0.5"))
MIN_CREDIT_WIDTH_PCT = float(os.getenv("OPTIONS_MIN_CREDIT_WIDTH_PCT", "15"))
MAX_PLAUSIBLE_REWARD_RISK_RATIO = float(os.getenv("OPTIONS_MAX_PLAUSIBLE_REWARD_RISK_RATIO", "5"))
LOTTERY_POP_THRESHOLD_PCT = float(os.getenv("OPTIONS_LOTTERY_POP_THRESHOLD_PCT", "15"))
CONSIDER_QUALITY_THRESHOLD = float(os.getenv("OPTIONS_CONSIDER_QUALITY_THRESHOLD", "75"))

REJECT_IC_SHORT_INSIDE_EM = os.getenv("OPTIONS_REJECT_IC_SHORT_INSIDE_EM", "true").lower() == "true"

HORIZON_ORDER = ["Weekly", "Monthly", "Quarterly"]
NIFTY_LOT_SIZE = int(os.getenv("NIFTY_LOT_SIZE", "75"))
TOTAL_CAPITAL_INR = float(os.getenv("OPTIONS_TOTAL_CAPITAL_INR", "1000000"))
RISK_FREE_RATE = float(os.getenv("OPTIONS_RISK_FREE_RATE", "0.065"))

# -----------------------------
# Live data fetch (NSE India option-chain API + Yahoo Finance)
# -----------------------------
_NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/option-chain",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Dest": "empty",
    "Connection": "keep-alive",
    "DNT": "1",
}


def _describe_http_error(e):
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
    if isinstance(e, requests.exceptions.Timeout):
        return False
    resp = getattr(e, "response", None)
    if resp is not None:
        return 500 <= resp.status_code < 600
    return isinstance(e, requests.exceptions.ConnectionError)


def _nse_warm_session(session, timeout, referer_path="/option-chain"):
    session.get("https://www.nseindia.com/", timeout=timeout)
    time.sleep(0.6)
    session.get(f"https://www.nseindia.com{referer_path}", timeout=timeout)
    time.sleep(0.6)


def select_best_strikes(horizon_snap, spot, bias, strategy_type, lot_size=NIFTY_LOT_SIZE):
    """
    Deterministically scans the actual options chain for the best valid structure 
    that clears all strict quality gates, replacing the LLM's blind strike guessing.
    """
    call_ltp = (horizon_snap or {}).get("call_ltp", {})
    put_ltp = (horizon_snap or {}).get("put_ltp", {})
    exp_move = (horizon_snap or {}).get("expected_move")
    
    if not call_ltp or not put_ltp or not spot or not exp_move:
        return {"ok": False, "reason": "Insufficient live premium/spot data for deterministic selection."}

    em_pts = exp_move.get("expected_move_pts", 0)
    band_lo, band_hi = spot - em_pts, spot + em_pts
    
    valid_structures = []

    calls = sorted([s for s in call_ltp.keys() if s > spot])
    puts = sorted([s for s in put_ltp.keys() if s < spot], reverse=True)

    def evaluate_spread(premium, width):
        if width <= 0:
            return None
        max_profit = premium
        max_loss = width - premium
        if max_loss <= 0:
            return None
        rr_ratio = max_profit / max_loss
        credit_width_pct = (premium / width) * 100
        return max_profit, max_loss, rr_ratio, credit_width_pct

    # 1. Evaluate Credit Verticals (Bear Call / Bull Put)
    if strategy_type in ["Bear Call Spread", "Bull Put Spread"]:
        strikes = calls if strategy_type == "Bear Call Spread" else puts
        ltp_map = call_ltp if strategy_type == "Bear Call Spread" else put_ltp
        
        for i, short_strike in enumerate(strikes):
            # Constraint #2: Short strike must be outside expected move
            if (strategy_type == "Bear Call Spread" and short_strike <= band_hi) or \
               (strategy_type == "Bull Put Spread" and short_strike >= band_lo):
                continue
                
            for long_strike in strikes[i+1:]:
                width = abs(long_strike - short_strike)
                if short_strike not in ltp_map or long_strike not in ltp_map:
                    continue
                premium = ltp_map[short_strike] - ltp_map[long_strike]
                
                if premium <= 0:
                    continue
                    
                stats = evaluate_spread(premium, width)
                if not stats:
                    continue
                max_profit, max_loss, rr_ratio, credit_width_pct = stats
                
                # Apply Strict Gates
                if rr_ratio >= MIN_REWARD_RISK_RATIO and credit_width_pct >= MIN_CREDIT_WIDTH_PCT:
                    valid_structures.append({
                        "short_strike": short_strike,
                        "long_strike": long_strike,
                        "rr_ratio": rr_ratio,
                        "credit_width_pct": credit_width_pct,
                        "premium": premium,
                        "width": width
                    })

    # 2. Evaluate Iron Condors (Neutral/Range-Bound)
    elif strategy_type == "Iron Condor":
        for sc in calls:
            if sc <= band_hi or sc not in call_ltp:
                continue
            for lc in calls:
                if lc <= sc or lc not in call_ltp:
                    continue
                call_width = lc - sc
                call_premium = call_ltp[sc] - call_ltp[lc]
                
                for sp in puts:
                    if sp >= band_lo or sp not in put_ltp:
                        continue
                    for lp in puts:
                        if lp >= sp or lp not in put_ltp:
                            continue
                        put_width = sp - lp
                        put_premium = put_ltp[sp] - put_ltp[lp]
                        
                        width = max(call_width, put_width)
                        premium = call_premium + put_premium
                        
                        if premium <= 0:
                            continue
                            
                        stats = evaluate_spread(premium, width)
                        if not stats:
                            continue
                        max_profit, max_loss, rr_ratio, credit_width_pct = stats
                        
                        if rr_ratio >= MIN_REWARD_RISK_RATIO and credit_width_pct >= MIN_CREDIT_WIDTH_PCT:
                            valid_structures.append({
                                "short_call": sc, "long_call": lc,
                                "short_put": sp, "long_put": lp,
                                "rr_ratio": rr_ratio,
                                "credit_width_pct": credit_width_pct,
                                "premium": premium
                            })

    if not valid_structures:
        return {"ok": False, "reason": f"No {strategy_type} cleared the strict {MIN_CREDIT_WIDTH_PCT}% credit/width and {MIN_REWARD_RISK_RATIO} R:R gates in current market conditions."}

    valid_structures.sort(key=lambda x: (x["rr_ratio"], x["credit_width_pct"]), reverse=True)
    best_trade = valid_structures[0]
    
    return {
        "ok": True,
        "best_trade": best_trade,
        "strategy_type": strategy_type
    }


def fetch_nse_option_chain(symbol="NIFTY", timeout=12, max_attempts=3):
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
                time.sleep(1.5 * attempt)
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
    by_dt = {_parse_nse_date(d): d for d in expiry_dates}
    picked = _pick_horizon_expiry_dates(by_dt.keys())
    return {h: by_dt[dt] for h, dt in picked.items()}


def _extract_expiry_snapshot(rows, expiry_str):
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

    call_ltp = {s: c.get("lastPrice") for s, c in calls.items() if c.get("lastPrice") is not None}
    put_ltp = {s: p.get("lastPrice") for s, p in puts.items() if p.get("lastPrice") is not None}
    call_iv = {s: c.get("impliedVolatility") for s, c in calls.items() if c.get("impliedVolatility")}
    put_iv = {s: p.get("impliedVolatility") for s, p in puts.items() if p.get("impliedVolatility")}
    call_oi_chg_pct = {s: c.get("pchangeinOpenInterest") for s, c in calls.items() if c.get("pchangeinOpenInterest") is not None}
    put_oi_chg_pct = {s: p.get("pchangeinOpenInterest") for s, p in puts.items() if p.get("pchangeinOpenInterest") is not None}
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


def describe_max_pain(horizon_snap, spot):
    max_pain = (horizon_snap or {}).get("max_pain")
    if max_pain is None or not spot:
        return None

    diff_pct = ((spot - max_pain) / spot) * 100
    if abs(diff_pct) < 0.15:
        position = "essentially at spot"
    elif diff_pct > 0:
        position = f"{abs(diff_pct):.1f}% below spot"
    else:
        position = f"{abs(diff_pct):.1f}% above spot"

    return (
        f"{max_pain} ({position}) — the strike where option writers' aggregate "
        f"payout is smallest at expiry; a soft magnet for price into expiry, not a target"
    )


def compute_oi_trend(horizon_snap):
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
                    break
    main.log.warning(
        f"NSE Bhavcopy fetch failed for {trade_date.date()} across all hosts "
        f"({', '.join(_BHAVCOPY_FO_HOSTS)}): {_describe_http_error(last_err)}"
    )
    return None


def fetch_latest_nse_bhavcopy_fo(max_days_back=6):
    now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
    skipped_weekdays = []
    for i in range(max_days_back + 1):
        candidate = now_ist - timedelta(days=i)
        if i == 0 and now_ist.time() < dtime(19, 0):
            continue
        rows = fetch_nse_bhavcopy_fo(candidate)
        if rows:
            return rows, candidate.date(), skipped_weekdays
        if candidate.weekday() < 5:
            skipped_weekdays.append(candidate.date())
    return None, None, skipped_weekdays


def _extract_bhavcopy_snapshot(rows, symbol, expiry_dt):
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
        "call_ltp": call_ltp,
        "put_ltp": put_ltp,
        "call_iv": {},
        "put_iv": {},
    }


def _fill_horizons_from_bhavcopy(data, notes, symbol="NIFTY"):
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
        f"markets overnight, event risk). In 'strike_rationale', describe qualitatively "
        f"why you chose your directional or range-bound strategy.",
        f"- Nifty 50 spot: {data.get('spot', 'n/a')}",
        "- India VIX: "
        + str(data.get("vix", "n/a"))
        + (f" ({data['vix_change_pct']:+.2f}% vs prior close)" if data.get("vix_change_pct") is not None else ""),
    ]
    if data.get("iv_rank") is not None and data.get("iv_percentile") is not None:
        lines.append(
            f"- IV Rank: {data['iv_rank']:g} / IV Percentile: {data['iv_percentile']:g}"
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
                f"±{exp_move['expected_move_pts']:g} pts (~{exp_move.get('expected_move_pct', 'n/a')}% of spot)"
            )

    if data.get("notes"):
        lines.append("Fetch notes: " + "; ".join(data["notes"]))

    return "\n".join(lines)


def _market_session_label():
    now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
    is_weekday = now_ist.weekday() < 5
    in_session = is_weekday and dtime(9, 15) <= now_ist.time() <= dtime(15, 30)
    if in_session:
        return "Live NSE trading session", True
    return "Outside NSE trading hours (data reflects last close, not live quotes)", False


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _iv_to_frac(iv):
    if iv is None:
        return None
    iv = float(iv)
    return iv / 100.0 if iv > 3 else iv


def time_to_expiry_years(expiry_dt, now=None):
    now = now or datetime.now(ZoneInfo("Asia/Kolkata"))
    expiry_close = datetime.combine(expiry_dt.date(), dtime(15, 30), tzinfo=ZoneInfo("Asia/Kolkata"))
    return max((expiry_close - now).total_seconds() / (365.0 * 86400), 1e-6)


def bs_greeks(spot, strike, t_years, iv, opt_type, r=RISK_FREE_RATE):
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
    if vix_series is None or current_vix is None or len(vix_series) < 20:
        return None, None, 0
    lo, hi = float(vix_series.min()), float(vix_series.max())
    days = len(vix_series)
    rank = round((current_vix - lo) / (hi - lo) * 100, 1) if hi > lo else 50.0
    below = int((vix_series < current_vix).sum())
    percentile = round(below / days * 100, 1)
    return rank, percentile, days


def compute_expected_move(call_ltp, put_ltp, spot):
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
    sigma = _iv_to_frac(iv)
    if spot is None or sigma is None or t_years is None or t_years <= 0 or sigma <= 0:
        return None

    def cdf_le(K):
        if K <= 0:
            return 0.0
        sqrt_t = math.sqrt(t_years)
        d1 = (math.log(spot / K) + (r + 0.5 * sigma ** 2) * t_years) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t
        return 1.0 - _norm_cdf(d2)

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
    sigma = _iv_to_frac(iv)
    if (
        spot is None or sigma is None or t_years is None or t_years <= 0
        or sigma <= 0 or barrier is None or barrier <= 0
    ):
        return None
    try:
        mu = r - 0.5 * sigma ** 2
        a = math.log(barrier / spot)
        sqrt_t = math.sqrt(t_years)
        if a > 0:
            d1 = (mu * t_years - a) / (sigma * sqrt_t)
            d2 = (-mu * t_years - a) / (sigma * sqrt_t)
        elif a < 0:
            d1 = (a - mu * t_years) / (sigma * sqrt_t)
            d2 = (a + mu * t_years) / (sigma * sqrt_t)
        else:
            return 100.0
        prob = _norm_cdf(d1) + math.exp(2 * mu * a / sigma ** 2) * _norm_cdf(d2)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    return round(max(0.0, min(1.0, prob)) * 100, 1)


def compute_expectancy_metrics(pop_pct, max_profit_inr, max_loss_inr):
    if (
        pop_pct is None or max_profit_inr is None or max_loss_inr is None
        or max_loss_inr <= 0 or not (0 <= pop_pct <= 100)
    ):
        return {"ev_inr": None, "kelly_pct": None, "sharpe_like": None}

    p = pop_pct / 100.0
    q = 1.0 - p
    ev = p * max_profit_inr - q * max_loss_inr

    b = max_profit_inr / max_loss_inr
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
    short_legs = [l for l in priced_legs if l["action"] == "Sell"]
    exposure_buffer = 0.03 * sum(l["strike"] for l in short_legs) * lot_size if short_legs else 0.0
    margin = max(max_loss_inr, exposure_buffer, 1.0)
    rom_pct = round(max_profit_inr / margin * 100, 2) if margin else None
    return round(margin, 0), rom_pct


_LEG_RE = re.compile(r"\b(Buy|Sell)\s+(\d+(?:\.\d+)?)\s*(CE|PE|Call|Put)\b", re.IGNORECASE)

def parse_legs(legs_text):
    """Extracts (action, strike, type) triples from the model's free-text
    'legs' field, normalizing Call -> CE and Put -> PE."""
    out = []
    for action, strike, opt_type in _LEG_RE.findall(legs_text or ""):
        opt_upper = opt_type.upper()
        if opt_upper in ("CALL", "CE"):
            normalized_type = "CE"
        elif opt_upper in ("PUT", "PE"):
            normalized_type = "PE"
        else:
            normalized_type = opt_upper

        out.append({
            "action": action.capitalize(),
            "strike": float(strike),
            "type": normalized_type
        })
    return out


def _leg_premium(horizon_snap, strike, opt_type):
    table = (horizon_snap or {}).get("call_ltp" if opt_type == "CE" else "put_ltp", {})
    return table.get(strike, table.get(int(strike) if float(strike).is_integer() else strike))


def _leg_iv(horizon_snap, strike, opt_type):
    table = (horizon_snap or {}).get("call_iv" if opt_type == "CE" else "put_iv", {})
    return table.get(strike, table.get(int(strike) if float(strike).is_integer() else strike))


def compute_strategy_payoff(legs_text, horizon_snap, lot_size=NIFTY_LOT_SIZE):
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
    m = (model_name or "").lower()
    found_in_model = [kw for kw in _STRUCTURE_KEYWORDS if kw in m]
    if not found_in_model:
        return False
    return not any(kw in classified_name.lower() for kw in found_in_model)


def generate_adjustment_trigger(priced_legs, net_premium):
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
    if not rationale or not any_leg_inside_band:
        return rationale
    if _OUTSIDE_BAND_CLAIM_RE.search(rationale):
        return "Selected primarily on OI/liquidity positioning -- see the computed band status below"
    return rationale


def build_strike_rationale_addendum(priced_legs, horizon_snap, spot):
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

    is_eod = bool(horizon_snap.get("source"))
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
    components = {}

    if ev_inr is not None and max_loss:
        ev_score = _clamp01_100(50 + (ev_inr / max_loss) * 100)
    else:
        ev_score = 0
    components["Expected Value"] = (round(ev_score), 30)

    if reward_risk_ratio is not None and reward_risk_ratio != float("inf"):
        rr_score = _clamp01_100((reward_risk_ratio / 1.5) * 100)
    else:
        rr_score = 0
    components["Reward:Risk"] = (round(rr_score), 20)

    pop_score = _clamp01_100(pop_pct) if pop_pct is not None else 0
    components["Probability of Profit"] = (round(pop_score), 15)

    conf_score = _clamp01_100(conf_pct) if conf_pct is not None else 0
    components["Confidence"] = (round(conf_score), 15)

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

    top_by_type = {
        "CE": top_call_oi[0][0] if top_call_oi else None,
        "PE": top_put_oi[0][0] if top_put_oi else None,
    }
    short_legs = [l for l in (priced_legs or []) if l["action"] == "Sell"]
    if not has_oi_data or not short_legs:
        oi_score = 50
    else:
        aligned = any(top_by_type.get(l["type"]) == l["strike"] for l in short_legs)
        oi_score = 100 if aligned else 35
    components["OI Alignment"] = (round(oi_score), 10)

    total = sum(sub_score * weight / 100 for sub_score, weight in components.values())
    score = round(total)

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

    if vix is not None:
        calm = vix < 20
        checks.append((
            calm,
            f"Calm volatility regime (VIX {vix:g} < 20)" if calm
            else f"Elevated volatility (VIX {vix:g} \u2265 20) -- wider realistic price swings",
        ))
    else:
        checks.append((False, "VIX unavailable this run"))

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
    horizon_dict["bias_reason"] = _scrub_pcr_mischaracterization(
        _strip_cap_claims(horizon_dict.get("bias_reason")),
        (horizon_snap or {}).get("pcr_oi"),
    )
    horizon_dict["bias"] = _scrub_bias_pcr_conflict(
        horizon_dict.get("bias"),
        (horizon_snap or {}).get("pcr_oi"),
    )
    horizon_dict["oi_trend"] = compute_oi_trend(horizon_snap or {})
    horizon_dict["max_pain_note"] = describe_max_pain(horizon_snap, spot)
    legs_text = horizon_dict.get("legs", "")
    result = compute_strategy_payoff(legs_text, horizon_snap)
    is_eod = bool((horizon_snap or {}).get("source"))
    premium_label = "EOD Bhavcopy closing prices (not live)" if is_eod else "live NSE premiums"
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

    max_loss_pts = result["max_loss"] / NIFTY_LOT_SIZE if NIFTY_LOT_SIZE else 0
    width = (max_loss_pts + net_premium) if net_premium > 0 else 0
    rich_credit_flag = ""
    if width > 0 and net_premium > 0 and (net_premium / width) > 0.5:
        pct_of_width = net_premium / width * 100
        rich_credit_flag = (
            f" ⚠ Net credit is {pct_of_width:.0f}% of the {width:g}-point spread width -- unusually rich; "
            f"double-check these premiums against a live broker terminal before trusting this figure."
        )

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
    horizon_dict["_over_ratio_ceiling"] = (
        max_loss > 0 and reward_risk_ratio > MAX_PLAUSIBLE_REWARD_RISK_RATIO
    )
    horizon_dict["reward_risk_ratio"] = (
        f"{reward_risk_ratio:.2f} (₹{max_profit:,.0f} potential vs ₹{max_loss:,.0f} at risk, per lot)"
        if max_loss > 0 else "n/a (no capital at risk)"
    )
    horizon_dict["breakeven"] = be
    horizon_dict["gap_risk"] = (
        f"Capped at max loss (₹{max_loss:,.0f}) even on a gap beyond the strikes -- "
        f"this is a defined-risk structure, so gap risk cannot exceed max loss."
    )
    horizon_dict["adjustment_trigger"] = generate_adjustment_trigger(priced_legs, net_premium)

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

    expiry_dt = None
    try:
        expiry_dt = _parse_nse_date((horizon_snap or {}).get("expiry", ""))
    except (ValueError, TypeError):
        expiry_dt = None
    t_years = time_to_expiry_years(expiry_dt) if expiry_dt else None

    greeks_ok = spot is not None and t_years is not None and not is_eod
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
        if pop is not None else None
    )

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

    horizon_dict["expected_win_rate"] = (
        f"~{pop:.0f}% if held to expiry ({pop_source}) -- typically higher in practice "
        f"if the position is actively managed/closed early rather than held to expiry"
        if pop is not None else None
    )

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

    return f"""Core Objective: You are an expert at analysing Nifty options-related data. A LIVE DATA FEED has already been fetched for you and is provided below -- use those figures as ground truth for this run rather than searching for or guessing them yourself. Using that feed (plus web search only for what it doesn't cover), analyze market conditions and determine the optimal strategy stance independently across THREE time horizons:

{live_data_block}

1. Nifty Weekly -- (current/nearest weekly expiry)
2. Nifty Monthly -- (current monthly expiry)
3. Nifty Quarterly -- (nearest quarterly expiry on the Mar/Jun/Sep/Dec cycle)

OUTPUT FORMAT -- respond with ONLY raw JSON matching the schema below, and nothing else:

{{
  "horizons": [
    {{
      "horizon": "Weekly",
      "expiry_date": "The actual expiry date for this horizon, copied EXACTLY as shown in the LIVE DATA FEED above (e.g. '24 Jul 2026').",
      "bias": "One of: Bullish / Bearish / Neutral / Range-bound",
      "next_week_bias": "n/a",
      "bias_reason": "One or two sentences grounded in the live data you found.",
      "strategy_name": "e.g. 'Bear Call Spread', 'Bull Put Spread', or 'Iron Condor'",
      "legs": "",
      "strike_rationale": "One qualitative sentence describing the logic behind the strategy placement.",
      "confidence": "High",
      "data_status": "live"
    }},
    {{ "horizon": "Monthly", ... same fields ... }},
    {{ "horizon": "Quarterly", ... same fields ... }}
  ],
  "portfolio_view": "One paragraph on overall market gap risk and cross-horizon alignment."
}}
"""


# -----------------------------
# Parsing
# -----------------------------
def _parse_analysis_json(text):
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


SENSIBULL_STRATEGY_BUILDER_URL = "https://web.sensibull.com/option-strategy-builder?instrument_symbol=NIFTY"


def _execution_cell_html(h, sans, expiry):
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
        raw_row("Directional Bias", _bias_badge(h.get("bias"))),
        row("Bias Rationale", "bias_reason"),
        *([row("OI Trend", "oi_trend")] if h.get("oi_trend") else []),
        *([row("Max Pain (OI-Derived)", "max_pain_note")] if h.get("max_pain_note") else []),
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
    if not text:
        return text
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    kept = [
        s for s in sentences
        if not (re.search(r"%", s) and re.search(r"\b(cap|limit)\b", s, re.IGNORECASE))
    ]
    return " ".join(kept).strip()


def classify_pcr(pcr):
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
    if pcr is None:
        return None
    if pcr < 0.7:
        return "bearish"
    if pcr <= 1.2:
        return "neutral"
    return "bullish"


_NEUTRAL_BIAS_LABELS = ("neutral", "balanced", "range-bound", "range bound", "sideways")


def _scrub_bias_pcr_conflict(bias, pcr):
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
    for field in ("bias_reason", "verification"):
        text = str(h.get(field) or "")
        if _NO_STRATEGY_CLAIM_RE.search(text):
            return True
    return False


def _scrub_portfolio_view_contradictions(portfolio_view, horizons):
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
    if not portfolio_view:
        return portfolio_view
    by_name = {str(h.get("horizon") or "").strip(): h for h in (horizons or []) if h.get("horizon")}
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
_NON_DIRECTIONAL_STRATEGY_KW = ("iron condor", "iron butterfly", "straddle", "strangle", "butterfly")


def _is_directional_strategy(strategy_name):
    s = str(strategy_name or "").strip().lower()
    if not s:
        return False
    return not any(kw in s for kw in _NON_DIRECTIONAL_STRATEGY_KW)


def _scrub_portfolio_view_directional_contradiction(portfolio_view, horizons):
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
    if not text:
        return text
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    kept = [s for s in sentences if not _GAMMA_WORD_RE.search(s)]
    return " ".join(kept).strip()


def compute_portfolio_gamma_summary(horizons):
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
    if score >= 80:
        return "Full Size", "#2F5233"
    if score >= 60:
        return "Half Size", "#3D6690"
    if score >= 40:
        return "Watchlist", "#A6812F"
    return "Skip", "#8B2E2E"


def render_trade_quality_table(horizons):
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
  <div style="font-family:{sans};font-size:10px;color:#8A8F9C;margin-bottom:6px;">Composite of Expected Value (30%), Reward:Risk (20%), Probability of Profit (15%), Confidence (15%), Liquidity (10%), and OI Alignment (10%) -- see each horizon's card below for the exact component breakdown.</div>
  <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="border-collapse:collapse;border:1px solid #E7E4DC;border-radius:4px;overflow:hidden;">
    {header_row}
    {''.join(rows)}
  </table>
</div>
"""


def render_strategy_summary_table(horizons):
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
            lots = MAX_LOTS_PER_HORIZON
            cap_note = (
                f"Capped at {MAX_LOTS_PER_HORIZON} lots (liquidity/slippage/margin/gap-risk "
                f"ceiling)"
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
            weakest[2] = "Reduced size to stay within aggregate cap"
        else:
            weakest[1] = None
            weakest[2] = f"Skipped to stay within the {AGGREGATE_CAP_PCT:.0f}% aggregate cap"

    return [tuple(row) for row in plan]


def render_suggested_sizing_html(plan, sans):
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
        f'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">Suggested Sizing </div>'
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

    verdict = (
        f"⚠ EXCEEDS the {AGGREGATE_CAP_PCT:.0f}% worst-case combined cap -- reduce position size before entering."
        if over_cap else
        f"✅ Stays within the {AGGREGATE_CAP_PCT:.0f}% worst-case combined cap."
    )
    gamma_summary = compute_portfolio_gamma_summary(horizons)
    portfolio_text = _strip_gamma_claims(_strip_cap_claims(portfolio_view))
    portfolio_text = _scrub_portfolio_view_contradictions(portfolio_text, horizons)
    portfolio_text = _scrub_portfolio_view_structure_type_contradiction(portfolio_text, horizons)
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
            "levels, IV, and OI figures can still be a few minutes to hours stale. Not investment advice."
        )
    else:
        disclaimer = (
            "Generated by an LLM with no live web search this run -- see the live-feed summary above for "
            "what the direct NSE/Yahoo fetch supplied. Not investment advice."
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
{build_compliance_block_html(report_kind="options", run_note=disclaimer)}
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


def finalize_horizons(horizons, live_data):
    by_name = {str(h.get("horizon") or "").strip(): h for h in horizons}

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
    ml = h.get("max_loss")
    return isinstance(ml, str) and (
        "UNDEFINED RISK" in ml or "Unverified" in ml or "POOR REWARD/RISK" in ml
        or "SHORT STRIKE INSIDE EXPECTED MOVE" in ml
    )


def reverify_horizons(horizons, live_data, only_names=None):
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

Respond with ONLY raw JSON:
{{
  "horizons": [
    {{"horizon": "<name>", "strategy_name": "<corrected strategy name>", "legs": "<corrected legs string>"}}
  ]
}}
"""


def repair_rejected_legs(horizons, live_data):
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


_OFFICIAL_SOURCE_DOMAINS = (
    "nseindia.com", "nsearchives.nseindia.com", "bseindia.com",
    "rbi.org.in", "sebi.gov.in", "sgx.com", "moneycontrol.com",
    "yahoo.com", "finance.yahoo.com", "reuters.com", "bloomberg.com",
    "livemint.com", "economictimes.indiatimes.com", "business-standard.com",
)
_BLOCKED_SOURCE_HINTS = ("instagram.com", "youtube.com", "youtu.be", "twitter.com", "x.com", "facebook.com", "tiktok.com")


def _source_url_title(s):
    if isinstance(s, dict):
        url = str(s.get("url") or s.get("link") or "")
        title = str(s.get("title") or s.get("name") or url)
        return url, title
    if isinstance(s, (tuple, list)) and len(s) == 2:
        a, b = s
        if isinstance(a, str) and a.strip().lower().startswith(("http://", "https://")):
            return a, (str(b) if b else a)
        if isinstance(b, str) and b.strip().lower().startswith(("http://", "https://")):
            return b, (str(a) if a else b)
        return "", str(a)
    text = str(s)
    return (text, text) if text.strip().lower().startswith(("http://", "https://")) else ("", text)


def _filter_sources(sources):
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
    url, title = _source_url_title(s)
    text = f"{url} {title}".lower()
    for cat, keywords in _CATEGORY_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return cat
    return None


def render_market_data_inputs_html(live_data, sources, sans):
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
        return None

    def _flow_cell(val, label):
        if val is None:
            return None
        sign = "+" if val >= 0 else "&minus;"
        color = "#2F5233" if val >= 0 else "#8B2E2E"
        return f'<span style="color:{color};font-weight:700;">{sign}₹{abs(val):,.0f} Cr</span> <span style="color:#8A8F9C;">({label})</span>'

    if fii_val is not None or dii_val is not None:
        parts = [p for p in (_flow_cell(fii_val, "FII, cash mkt"), _flow_cell(dii_val, "DII, cash mkt")) if p]
        date_note = f" &middot; {html.escape(str(fii_dii_date))}" if fii_dii_date else ""
        cell = (
            "<br>".join(parts)
            + f'<div style="margin-top:2px;color:#8A8F9C;font-size:11px;">Source: {html.escape(str(fii_dii_src or "NSE"))}{date_note}</div>'
        )
        rows.append(_row("FII/DII Activity", cell))
    else:
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

    analysis, sources, used_live_search = swing.generate_analysis(prompt)
    if not analysis:
        main.log.error(
            "No LLM backend produced output. Aborting without sending an email."
        )
        sys.exit(1)

    live_feed_ok = live_data.get("status") in ("ok", "eod_fallback", "partial") and (
        live_data.get("spot") or live_data.get("horizons")
    )
    if not used_live_search and not live_feed_ok and os.getenv("REQUIRE_LIVE_DATA", "true").lower() == "true":
        main.log.error(
            "Neither the direct NSE/Yahoo live data fetch nor the LLM's own live "
            "web search succeeded this run. Aborting."
        )
        sys.exit(1)

    horizons, _model_aggregate_pct, portfolio_view = _parse_analysis_json(analysis)
    sources = _filter_sources(sources)
    if horizons:
        # --- DETERMINISTIC STRIKE SELECTION ---
        for h in horizons:
            horizon_name = h.get("horizon")
            strategy_name = str(h.get("strategy_name") or "")
            bias = str(h.get("bias") or "")
            snap = (live_data.get("horizons") or {}).get(horizon_name, {})
            spot = live_data.get("spot")

            strat_lower = strategy_name.lower()
            bias_lower = bias.lower()

            if "bear call" in strat_lower or ("bear" in bias_lower and "call" in strat_lower):
                target_strat = "Bear Call Spread"
            elif "bull put" in strat_lower or ("bull" in bias_lower and "put" in strat_lower):
                target_strat = "Bull Put Spread"
            elif "iron condor" in strat_lower or "condor" in strat_lower:
                target_strat = "Iron Condor"
            elif "bear" in bias_lower:
                target_strat = "Bear Call Spread"
            elif "bull" in bias_lower:
                target_strat = "Bull Put Spread"
            else:
                target_strat = "Iron Condor"

            res = select_best_strikes(snap, spot, bias, target_strat)
            if res.get("ok"):
                best = res["best_trade"]
                st = res["strategy_type"]
                h["strategy_name"] = st
                if st == "Bear Call Spread":
                    h["legs"] = f"Sell {best['short_strike']:g} CE, Buy {best['long_strike']:g} CE"
                elif st == "Bull Put Spread":
                    h["legs"] = f"Sell {best['short_strike']:g} PE, Buy {best['long_strike']:g} PE"
                elif st == "Iron Condor":
                    h["legs"] = f"Buy {best['long_put']:g} PE, Sell {best['short_put']:g} PE, Sell {best['short_call']:g} CE, Buy {best['long_call']:g} CE"
            else:
                main.log.info(f"Deterministic strike selection for {horizon_name} ({target_strat}): {res.get('reason')}")

        horizons, aggregate_pct, over_cap = finalize_horizons(horizons, live_data)
        horizons = repair_rejected_legs(horizons, live_data)
        aggregate_pct, over_cap = reverify_horizons(horizons, live_data, only_names=None)
        if over_cap:
            main.log.warning(
                f"Computed worst-case combined max loss ({aggregate_pct}%) exceeds the "
                f"{AGGREGATE_CAP_PCT:.0f}% cap -- flagging in the report."
            )
        horizons_html = render_horizons_html(horizons, aggregate_pct, portfolio_view, live_data)
    else:
        main.log.error("Could not parse JSON from LLM output; falling back to raw text display.")
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