"""
optionstrategy.py

Standalone companion to main.py, structured the same way as
swing_trade_advisor.py. Runs a single "recommend the best risk-defined
Nifty options strategy across Weekly / Next Week / Next to Next Week
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

from utils import config
from utils.logger import log
from utils.prompt_loader import load_prompt
from controllers import swing_controller as swing
from utils.compliance import build_compliance_block_html

# -----------------------------
# Config (env-overridable capital caps, per the prompt's constraint #3)
# -----------------------------
PER_HORIZON_CAP_PCT = float(os.getenv("OPTIONS_PER_HORIZON_CAP_PCT", "5"))
AGGREGATE_CAP_PCT = float(os.getenv("OPTIONS_AGGREGATE_CAP_PCT", "15"))

# Absolute lot ceiling per horizon, independent of the PER_HORIZON_CAP_PCT
# math above -- issue #6.
MAX_LOTS_PER_HORIZON = int(os.getenv("OPTIONS_MAX_LOTS_PER_HORIZON", "5"))

# Minimum acceptable reward:risk on any strike selection, expressed here as
# max_profit / max_loss (so a "1:2" reward:risk floor -- risking 2 to make
# 1 -- is 0.5; "1:1.5" -- risking 1.5 to make 1 -- is 1/1.5 ~= 0.6667).
# Previously 0.5 (1:2 floor); raised to require a somewhat better payoff per
# unit of risk on every gated strike selection (see rr_ratio checks below
# and poor_reward_risk in apply_verified_payoff).
MIN_REWARD_RISK_RATIO = float(os.getenv("OPTIONS_MIN_REWARD_RISK_RATIO", "0.6667"))
MIN_CREDIT_WIDTH_PCT = float(os.getenv("OPTIONS_MIN_CREDIT_WIDTH_PCT", "15"))
MAX_PLAUSIBLE_REWARD_RISK_RATIO = float(os.getenv("OPTIONS_MAX_PLAUSIBLE_REWARD_RISK_RATIO", "5"))
LOTTERY_POP_THRESHOLD_PCT = float(os.getenv("OPTIONS_LOTTERY_POP_THRESHOLD_PCT", "15"))
CONSIDER_QUALITY_THRESHOLD = float(os.getenv("OPTIONS_CONSIDER_QUALITY_THRESHOLD", "75"))

# Item 19: hard floor on the Liquidity component (see compute_trade_quality_
# score's liq_score -- 30 is both the flat EOD-fallback placeholder AND the
# worst score a live spread can get while still clearing MAX_LEG_SPREAD_PCT).
# Previously liquidity only ever docked 10% off the composite Trade Quality
# Score and never blocked a recommendation outright -- a horizon with
# unknown/thin liquidity across the board could still read "✅ Consider".
# A score at or below this floor now hard-caps the verdict at "⚠ Caution"
# regardless of how good the EV/R:R/POP math looks.
MIN_LIQUIDITY_SCORE_FOR_CONSIDER = float(os.getenv("OPTIONS_MIN_LIQUIDITY_SCORE", "35"))

REJECT_IC_SHORT_INSIDE_EM = os.getenv("OPTIONS_REJECT_IC_SHORT_INSIDE_EM", "true").lower() == "true"

# How far the ATM-straddle-premium expected move and the IV-based 1-sigma
# expected move (see compute_expected_move()) can disagree, in relative
# percent, before that disagreement itself counts against confidence.
EM_DIVERGENCE_THRESHOLD_PCT = float(os.getenv("OPTIONS_EM_DIVERGENCE_THRESHOLD_PCT", "25"))

# Max acceptable bid-ask spread on any single leg, as a percent of the
# leg's mid price. A strike with no two-sided quote at all (one or both
# sides missing/zero) is always treated as illiquid regardless of this
# threshold -- see compute_bid_ask_spread_pct().
MAX_LEG_SPREAD_PCT = float(os.getenv("OPTIONS_MAX_LEG_SPREAD_PCT", "15"))

# Lookback window for the India VIX history used by compute_iv_rank_percentile()
# to place today's VIX into its historical range/percentile. Previously
# hardcoded to a strict trailing 1-year ("YoY") window via yf.Ticker(...).
# history(period="1y") -- that's a reasonable default for a full vol-regime
# cycle, but a strict YoY window can be skewed by a single old outlier event
# (e.g. an election spike or a crash) that's no longer representative of the
# CURRENT regime. Exposed as an env knob so a shorter, more responsive window
# (e.g. "6mo" for a trailing-2-quarter view) can be used instead without a
# code change; "1y" keeps today's TTM-equivalent behavior as the default.
# Any period string accepted by yfinance's history(period=...) works here
# (e.g. "3mo", "6mo", "1y", "2y"). compute_iv_rank_percentile() itself still
# requires at least 20 data points regardless of the window chosen.
IV_RANK_LOOKBACK_PERIOD = os.getenv("OPTIONS_IV_RANK_LOOKBACK_PERIOD", "1y")

# Item 8: a structure that misses the strict MIN_REWARD_RISK_RATIO /
# MIN_CREDIT_WIDTH_PCT gates by only this much (as a percent shaved off
# each threshold) is surfaced as a "Watchlist" near-miss candidate instead
# of being silently dropped when nothing clears the strict gates. 15 means
# a structure clearing 85% of each strict threshold still qualifies as a
# near-miss. Watchlist candidates are never promoted to a live "Consider"
# recommendation and are sized down accordingly (see _sizing_multiplier()).
NEAR_MISS_TOLERANCE_PCT = float(os.getenv("OPTIONS_NEAR_MISS_TOLERANCE_PCT", "15"))

# Item 10: widens the strike "universe" considered for the Watchlist
# near-miss pass on single-sided credit verticals (Bear Call / Bull Put)
# only -- lets a short strike sit as close as this fraction of the full
# expected-move band to spot and still be scanned as a near-miss candidate.
# The strict pass that produces a live "ok" trade is untouched (still
# requires the short strike sit fully outside 1.0x the EM band).
# Deliberately NOT applied to Iron Condor short strikes: that structure
# already has a dedicated hard safety check (REJECT_IC_SHORT_INSIDE_EM)
# against a short strike sitting inside the expected move, so relaxing the
# band here for IC would just fight that check instead of feeding it -- see
# the regime-override logic in run() for how IC handles this case instead.
WATCHLIST_EM_BAND_PCT = float(os.getenv("OPTIONS_WATCHLIST_EM_BAND_PCT", "85"))

# Item 11: "regime override" -- when a neutral Iron Condor can't find ANY
# strike combination with its short legs genuinely outside the expected
# move band (the chain itself isn't range-bound enough this run), and the
# live PCR(OI) shows a directional skew well beyond the plain neutral/
# decisive cutoff (classify_pcr()'s 0.7 / 1.2 bands), retry deterministic
# selection with the PCR-implied directional credit spread instead of
# falling through to the model's raw (also likely ungated) IC guess.
# REGIME_OVERRIDE_PCR_MARGIN is the extra margin required beyond the bare
# 0.7/1.2 cutoff before the override fires, so a borderline/neutral-ish
# PCR reading doesn't flip the whole structure family.
REGIME_OVERRIDE_ENABLED = os.getenv("OPTIONS_REGIME_OVERRIDE_ENABLED", "true").lower() == "true"
REGIME_OVERRIDE_PCR_MARGIN = float(os.getenv("OPTIONS_REGIME_OVERRIDE_PCR_MARGIN", "0.1"))

HORIZON_ORDER = ["Weekly", "Next Week", "Next to Next Week"]
NIFTY_LOT_SIZE = int(os.getenv("NIFTY_LOT_SIZE", "75"))
TOTAL_CAPITAL_INR = float(os.getenv("OPTIONS_TOTAL_CAPITAL_INR", "1000000"))
RISK_FREE_RATE = float(os.getenv("OPTIONS_RISK_FREE_RATE", "0.065"))

# Nifty's constituents pay a real dividend yield (historically ~1.1-1.5%
# annualized). Every Greek/probability calculation below previously priced
# the index as if it paid no dividend at all (q=0), which biases delta,
# theta, POP, and touch-probability for the underlying's expected drift --
# the bias compounds with time-to-expiry, so it matters most for the
# Next to Next Week horizon (the furthest-dated of the three). Default is a reasonable long-run estimate;
# override via env if you track the live trailing yield.
DIVIDEND_YIELD = float(os.getenv("OPTIONS_DIVIDEND_YIELD", "0.012"))

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
        except (AttributeError, ValueError, TypeError) as _e:
            log.debug(f"Could not extract snippet from HTTP error: {_e}")
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


# Item 14: yfinance spot/VIX pulls previously had zero retry -- a single
# transient blip (rate limit, DNS hiccup, Yahoo's own flakiness) meant an
# outright reject for that data point with no second attempt, even though
# NSE's own fetches (fetch_nse_option_chain / fetch_nse_fii_dii above)
# already retry with backoff. This wraps any zero-arg fetch callable with
# the same retry-with-delay pattern used elsewhere in this file.
def _retry_with_delay(fetch_fn, what, max_attempts=3, base_delay=1.5, notes=None):
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fetch_fn()
        except Exception as e:
            last_err = e
            log.warning(f"{what} fetch attempt {attempt}/{max_attempts} failed: {_describe_http_error(e)}")
            if attempt < max_attempts:
                time.sleep(base_delay * attempt)
    msg = f"{what} fetch failed after {max_attempts} attempts: {_describe_http_error(last_err)}"
    log.warning(msg)
    if notes is not None:
        notes.append(msg)
    return None


# Item 14: secondary source for India VIX. yfinance's ^INDIAVIX is the only
# source that gives us history (needed for compute_iv_rank_percentile), but
# if it's unavailable this run, NSE's own live indices endpoint at least
# gives a current VIX level so the prompt isn't missing the figure entirely
# -- IV rank/percentile just stay unavailable (no history from this
# endpoint), which is noted separately by the caller.
def _fetch_nse_vix_snapshot(timeout=12):
    session = requests.Session()
    session.headers.update(_NSE_HEADERS)
    _nse_warm_session(session, timeout, "/market-data/live-market-indices")
    resp = session.get("https://www.nseindia.com/api/allIndices", timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    for row in payload.get("data", []):
        if str(row.get("index", "")).strip().upper() in ("INDIA VIX", "NIFTY VIX"):
            last = row.get("last")
            prev_close = row.get("previousClose")
            if last is None:
                return None
            return {
                "vix": round(float(last), 2),
                "vix_change_pct": (
                    round(((float(last) - float(prev_close)) / float(prev_close)) * 100, 2)
                    if prev_close else None
                ),
            }
    return None


def select_best_strikes(horizon_snap, spot, bias, strategy_type, lot_size=NIFTY_LOT_SIZE):
    """
    Deterministically scans the actual options chain for the best valid structure 
    that clears all strict quality gates, replacing the LLM's blind strike guessing.
    """
    call_ltp = (horizon_snap or {}).get("call_ltp", {})
    put_ltp = (horizon_snap or {}).get("put_ltp", {})
    exp_move = (horizon_snap or {}).get("expected_move")
    
    if not call_ltp or not put_ltp or not spot or not exp_move:
        return {"ok": False, "watchlist": False, "reason": "Insufficient live premium/spot data for deterministic selection."}

    em_pts = exp_move.get("expected_move_pts", 0)
    band_lo, band_hi = spot - em_pts, spot + em_pts
    # Item 10: relaxed band used ONLY for the Watchlist near-miss pass on
    # single-sided credit verticals (see WATCHLIST_EM_BAND_PCT above).
    relaxed_band_pts = em_pts * (WATCHLIST_EM_BAND_PCT / 100)
    relaxed_band_lo, relaxed_band_hi = spot - relaxed_band_pts, spot + relaxed_band_pts

    # Item 8: relaxed R:R / credit-width floors used to classify a
    # structure that misses the strict gates as a Watchlist near-miss
    # rather than being dropped outright.
    near_miss_mult = 1 - (NEAR_MISS_TOLERANCE_PCT / 100)
    rr_near_floor = MIN_REWARD_RISK_RATIO * near_miss_mult
    credit_width_near_floor = MIN_CREDIT_WIDTH_PCT * near_miss_mult

    valid_structures = []
    near_miss_structures = []

    # Real liquidity gate on candidate strikes (see compute_bid_ask_spread_pct /
    # _leg_liquidity_check): when this horizon's snapshot carries actual
    # bid/ask data, drop any strike without a genuine two-sided market or
    # whose spread exceeds MAX_LEG_SPREAD_PCT before ever building a
    # structure out of it -- previously OI *presence* was the only proxy
    # used, and a structure could be "optimal" on paper while sitting on a
    # leg nobody could actually trade at a fair price. Skips (doesn't
    # filter) when there's no quote data for this horizon at all (EOD).
    has_quote_data = bool((horizon_snap or {}).get("call_bid") or (horizon_snap or {}).get("put_bid"))

    def _liquid(strike, opt_type):
        ok, _ = _leg_liquidity_check(horizon_snap, strike, opt_type, has_quote_data)
        return ok

    calls = sorted([s for s in call_ltp.keys() if s > spot and _liquid(s, "CE")])
    puts = sorted([s for s in put_ltp.keys() if s < spot and _liquid(s, "PE")], reverse=True)

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

    def evaluate_debit_spread(premium, width):
        # Mirror of evaluate_spread() above, but for a DEBIT vertical: the
        # premium is paid, not received, so it IS the max loss (capped,
        # defined risk) and (width - premium) is the max profit -- the
        # inverse of the credit-spread relationship. "credit_width_pct" is
        # reused/reported here as the debit's share of the width (same
        # MIN_CREDIT_WIDTH_PCT env knob gates it, just inverted: a debit
        # spread should pay LESS than (100 - MIN_CREDIT_WIDTH_PCT)% of the
        # width, the mirror image of a credit spread collecting AT LEAST
        # MIN_CREDIT_WIDTH_PCT% of it).
        if width <= 0:
            return None
        max_loss = premium
        if max_loss <= 0:
            return None
        max_profit = width - premium
        if max_profit <= 0:
            return None
        rr_ratio = max_profit / max_loss
        debit_width_pct = (premium / width) * 100
        return max_profit, max_loss, rr_ratio, debit_width_pct

    # 1. Evaluate Credit Verticals (Bear Call / Bull Put)
    if strategy_type in ["Bear Call Spread", "Bull Put Spread"]:
        strikes = calls if strategy_type == "Bear Call Spread" else puts
        ltp_map = call_ltp if strategy_type == "Bear Call Spread" else put_ltp
        
        for i, short_strike in enumerate(strikes):
            # Constraint #2: Short strike must be outside expected move for
            # the strict pass; the relaxed band (item 10) only widens how
            # close to spot a strike can sit and still be scanned at all --
            # it does NOT relax the RR/credit-width gates below, so a
            # nearer-to-spot strike still has to clear those on its own.
            if strategy_type == "Bear Call Spread":
                band_ok = short_strike > band_hi
                band_near_ok = short_strike > relaxed_band_hi
            else:
                band_ok = short_strike < band_lo
                band_near_ok = short_strike < relaxed_band_lo
            if not band_near_ok:
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
                candidate = {
                    "short_strike": short_strike,
                    "long_strike": long_strike,
                    "rr_ratio": rr_ratio,
                    "credit_width_pct": credit_width_pct,
                    "premium": premium,
                    "width": width
                }

                # Apply Strict Gates
                if band_ok and rr_ratio >= MIN_REWARD_RISK_RATIO and credit_width_pct >= MIN_CREDIT_WIDTH_PCT:
                    valid_structures.append(candidate)
                elif rr_ratio >= rr_near_floor and credit_width_pct >= credit_width_near_floor:
                    # Item 8/10: missed the strict gate (on band, R:R, or
                    # credit-width) but within tolerance -- Watchlist tier.
                    near_miss_structures.append(candidate)

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

                        total_premium = call_premium + put_premium
                        if total_premium <= 0:
                            continue

                        # BUG FIX: for an Iron Condor the theoretical max loss is
                        # per-wing, not the wider width against the total premium.
                        # With unequal wings the call side and put side can each
                        # have a different max loss (wing_width - wing_premium);
                        # the trade max loss is the larger of the two. Previously
                        # max(call_width, put_width) was used as a single width
                        # against the combined premium -- correct only when wings
                        # are equal, wrong otherwise (overstates max-loss for a
                        # wider-call / richer-put condor, understates for the
                        # reverse). Compute per-wing then take the max.
                        call_max_loss = call_width - call_premium
                        put_max_loss = put_width - put_premium
                        if call_max_loss <= 0 or put_max_loss <= 0:
                            # Either wing is fully funded by its own credit -- no
                            # genuine risk on that side; skip (data anomaly).
                            continue
                        max_loss = max(call_max_loss, put_max_loss)
                        max_profit = total_premium
                        rr_ratio = max_profit / max_loss
                        # credit_width_pct: use the wider wing as the reference
                        # width, consistent with how brokers display it.
                        ref_width = max(call_width, put_width)
                        credit_width_pct = (total_premium / ref_width) * 100
                        candidate = {
                            "short_call": sc, "long_call": lc,
                            "short_put": sp, "long_put": lp,
                            "rr_ratio": rr_ratio,
                            "credit_width_pct": credit_width_pct,
                            "premium": total_premium,
                            "_call_max_loss": call_max_loss,
                            "_put_max_loss": put_max_loss,
                        }

                        if rr_ratio >= MIN_REWARD_RISK_RATIO and credit_width_pct >= MIN_CREDIT_WIDTH_PCT:
                            valid_structures.append(candidate)
                        elif rr_ratio >= rr_near_floor and credit_width_pct >= credit_width_near_floor:
                            # Item 8: R:R/credit-width near-miss only -- the
                            # band constraint above (sc <= band_hi / sp >=
                            # band_lo) is NOT relaxed for Iron Condor (see
                            # WATCHLIST_EM_BAND_PCT docstring for why).
                            near_miss_structures.append(candidate)

    # 3. Evaluate Debit Verticals (Bull Call / Bear Put)
    # BUG FIX: previously select_best_strikes() only implemented the three
    # credit/neutral structures above, so a Bull Call or Bear Put Spread
    # skipped deterministic optimization entirely and fell back to
    # whatever strikes the LLM guessed (only run through apply_verified_
    # payoff for payoff math afterwards, never through the strike-quality
    # optimizer the other three structures get). This scans the real
    # chain the same way, just with the debit/credit relationship
    # inverted -- see evaluate_debit_spread() above.
    elif strategy_type in ["Bull Call Spread", "Bear Put Spread"]:
        is_bull_call = strategy_type == "Bull Call Spread"
        opt_type = "CE" if is_bull_call else "PE"
        ltp_map = call_ltp if is_bull_call else put_ltp

        # Debit spreads legitimately buy a strike on either side of spot
        # (ATM/ITM for real delta exposure), unlike the strictly-OTM
        # `calls`/`puts` lists built above for the credit verticals -- so
        # scan a fresh, liquidity-filtered, full-chain strike list here
        # instead of reusing those. Ordered so `long_strike` (the leg
        # actually bought) is always the nearer/more-expensive strike and
        # `short_strike` (the leg sold to fund it) is always farther out,
        # matching how the pair is priced below.
        all_strikes = sorted(
            (s for s in ltp_map.keys() if _liquid(s, opt_type)),
            reverse=not is_bull_call,
        )

        if not all_strikes:
            return {
                "ok": False,
                "watchlist": False,
                "reason": (
                    f"No liquid {opt_type} strikes available for {strategy_type} scan "
                    f"(chain empty or every strike failed the liquidity gate) -- "
                    f"not a gate-quality failure, there was nothing to evaluate."
                ),
            }

        for i, long_strike in enumerate(all_strikes):
            # Mirror of constraint #2 on the credit verticals: don't buy a
            # long leg that's already past the expected-move band on the
            # wrong side -- that's paying up for a strike the underlying
            # isn't expected to reach by expiry, not a genuine entry.
            if is_bull_call and long_strike > band_hi:
                continue
            if not is_bull_call and long_strike < band_lo:
                continue
            # Items 16/17: symmetric guard on the OTHER side. Nothing above
            # stopped the long leg from being bought arbitrarily deep ITM
            # (e.g. 4x+ the expected move on the wrong side of spot) --
            # a candidate like that prices almost entirely as intrinsic
            # value, which the buggy sort below used to actively favor
            # (see the sort-key fix a few lines down). A genuine directional
            # debit-spread entry has no business buying a leg this far
            # outside the expected-move band on either side.
            if is_bull_call and long_strike < band_lo:
                continue
            if not is_bull_call and long_strike > band_hi:
                continue

            for short_strike in all_strikes[i + 1:]:
                width = abs(short_strike - long_strike)
                if long_strike not in ltp_map or short_strike not in ltp_map:
                    continue
                premium = ltp_map[long_strike] - ltp_map[short_strike]

                if premium <= 0:
                    continue

                stats = evaluate_debit_spread(premium, width)
                if not stats:
                    continue
                max_profit, max_loss, rr_ratio, debit_width_pct = stats
                candidate = {
                    "long_strike": long_strike,
                    "short_strike": short_strike,
                    "rr_ratio": rr_ratio,
                    "credit_width_pct": debit_width_pct,
                    "premium": premium,
                    "width": width,
                }

                # Apply Strict Gates (see evaluate_debit_spread() docstring
                # for why the credit-width gate is inverted here).
                if rr_ratio >= MIN_REWARD_RISK_RATIO and debit_width_pct <= (100 - MIN_CREDIT_WIDTH_PCT):
                    valid_structures.append(candidate)
                elif rr_ratio >= rr_near_floor and debit_width_pct <= (100 - credit_width_near_floor):
                    # Item 8: near-miss on R:R and/or how much of the width
                    # was paid away -- band constraint above is unchanged
                    # (not part of item 10's relaxation for debit spreads).
                    near_miss_structures.append(candidate)

    # Items 16/17: "credit_width_pct" means opposite things for a credit
    # vertical/Iron Condor (premium COLLECTED as % of width -- higher is
    # better) vs. a debit vertical (premium PAID as % of width -- lower is
    # better). Sorting both the same way (descending) silently picked the
    # WORST debit spread that still cleared the gates -- the one that paid
    # away the most of the width, which is what a deep-ITM long leg does
    # since its price is almost pure intrinsic value. Sort debit spreads by
    # rr_ratio first (directly measures reward per unit of risk, and is
    # unaffected by this sign confusion), tiebreaking on the LEAST width
    # paid away (ascending credit_width_pct/debit_width_pct).
    is_debit_spread = strategy_type in ["Bull Call Spread", "Bear Put Spread"]

    def _sort_key(candidates):
        if is_debit_spread:
            candidates.sort(key=lambda x: (x["rr_ratio"], -x["credit_width_pct"]), reverse=True)
        else:
            candidates.sort(key=lambda x: (x["credit_width_pct"], x["rr_ratio"]), reverse=True)

    if not valid_structures:
        # Item 8: nothing cleared the strict gates -- before giving up
        # entirely, check whether a near-miss candidate qualifies as a
        # Watchlist pick instead of reporting nothing.
        if near_miss_structures:
            _sort_key(near_miss_structures)
            best_near_miss = near_miss_structures[0]
            return {
                "ok": False,
                "watchlist": True,
                "near_miss_trade": best_near_miss,
                "strategy_type": strategy_type,
                "reason": (
                    f"No {strategy_type} cleared the strict {MIN_CREDIT_WIDTH_PCT}% credit/width and "
                    f"{MIN_REWARD_RISK_RATIO} R:R gates, but the closest candidate landed within "
                    f"{NEAR_MISS_TOLERANCE_PCT:.0f}% of those thresholds -- surfaced as a Watchlist "
                    f"candidate instead of being dropped."
                ),
            }
        liquidity_hint = (
            " (or every nearby strike failed the real bid-ask liquidity gate)"
            if has_quote_data else ""
        )
        return {
            "ok": False,
            "watchlist": False,
            "reason": (
                f"No {strategy_type} cleared the strict {MIN_CREDIT_WIDTH_PCT}% credit/width and "
                f"{MIN_REWARD_RISK_RATIO} R:R gates in current market conditions{liquidity_hint}, "
                f"and no near-miss candidate landed within {NEAR_MISS_TOLERANCE_PCT:.0f}% of those "
                f"thresholds either."
            ),
        }

    # Sort by credit_width_pct (premium collected as % of width) first, then
    # rr_ratio as tiebreaker, for credit verticals/Iron Condor. The R:R
    # floor is already enforced by the MIN_REWARD_RISK_RATIO gate above, so
    # every candidate here already clears the minimum acceptable risk/reward
    # -- the remaining objective is to maximize premium captured, not to
    # keep optimizing R:R once it's already "good enough". (Previously
    # sorted (rr_ratio, credit_width_pct), which could pick a lower-premium
    # structure over a materially richer one just for a marginally better
    # R:R.) Debit verticals (Bull Call / Bear Put) use the inverted
    # objective via _sort_key() above -- see items 16/17.
    _sort_key(valid_structures)
    best_trade = valid_structures[0]
    
    return {
        "ok": True,
        "watchlist": False,
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
            log.warning(
                f"NSE option-chain fetch attempt {attempt}/{max_attempts} failed for "
                f"{symbol}: {_describe_http_error(e)}"
            )
            if attempt < max_attempts and _is_retryable_nse_error(e):
                time.sleep(1.5 * attempt)
            elif attempt < max_attempts:
                log.warning(
                    f"NSE option-chain fetch for {symbol}: error looks like a block "
                    f"rather than a transient blip -- skipping remaining retries."
                )
                break
    log.warning(
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
                log.warning(
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
            log.warning(
                f"NSE FII/DII fetch attempt {attempt}/{max_attempts} failed: {_describe_http_error(e)}"
            )
            if attempt < max_attempts and _is_retryable_nse_error(e):
                time.sleep(1.5 * attempt)
            elif attempt < max_attempts:
                log.warning("NSE FII/DII fetch: error looks like a block -- skipping remaining retries.")
                break
    log.warning(f"NSE FII/DII fetch failed: {_describe_http_error(last_err)}")
    return None


def _pick_horizon_expiry_dates(dt_list):
    dts = sorted(set(dt_list))
    if not dts:
        return {}
    weekly_dt = dts[0]
    next_week_dt = dts[1] if len(dts) > 1 else dts[0]
    # Third-nearest expiry -- i.e. the expiry that follows "Next Week".
    # Falls back to the last available date (or Next Week's / Weekly's own
    # date) when the feed doesn't have three distinct expiries yet.
    next_to_next_week_dt = dts[2] if len(dts) > 2 else (dts[-1] if len(dts) > 1 else weekly_dt)
    return {"Weekly": weekly_dt, "Next Week": next_week_dt, "Next to Next Week": next_to_next_week_dt}


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

    # Real liquidity data: NSE's option-chain payload carries per-strike
    # top-of-book bid/ask (field names are inconsistently cased in NSE's
    # own API -- "bidprice" lowercase, "askPrice" uppercase -- this is not
    # a typo). A strike with no live two-sided quote comes back as 0 or
    # missing on one or both sides; only keep strikes where both sides are
    # a genuine positive price, since that's the only case a spread is
    # actually meaningful. See compute_bid_ask_spread_pct() / the
    # liquidity gate in compute_strategy_payoff() and select_best_strikes().
    def _quote_map(book, side_key):
        return {s: v for s, o in book.items() if (v := o.get(side_key)) and v > 0}

    call_bid = _quote_map(calls, "bidprice")
    call_ask = _quote_map(calls, "askPrice")
    put_bid = _quote_map(puts, "bidprice")
    put_ask = _quote_map(puts, "askPrice")

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
        "call_bid": call_bid,
        "call_ask": call_ask,
        "put_bid": put_bid,
        "put_ask": put_ask,
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


def compute_expected_move(call_ltp, put_ltp, spot, vix=None, expiry=None, call_iv=None, put_iv=None):
    """
    Two independent expected-move estimates for one expiry, reconciled into
    a single gating figure instead of left to silently disagree:

      - Straddle-premium EM: ATM call + ATM put last-traded premium (the
        market's own breakeven-to-breakeven range for this expiry).
      - IV-based 1-sigma EM: spot * ATM_IV * sqrt(T) -- the standard
        lognormal 1-sigma move implied by ATM implied volatility, the same
        method compute_pop()/_pop_diagnostics() use for POP.

    These can genuinely diverge -- thin ATM liquidity distorts the straddle
    premium, a stale/skewed IV quote distorts the IV estimate -- so this
    always returns BOTH figures (straddle_move_pts, iv_move_pts) plus a
    single `expected_move_pts` that is the WIDER of the two, tagged via
    `expected_move_basis`. Anything gating "is this short strike safely
    outside the expected move" (select_best_strikes, the Iron Condor
    short-inside-EM check, compute_confidence's strike-distance check) reads
    `expected_move_pts` and so automatically becomes the more conservative
    of the two checks, rather than trusting whichever formula happens to be
    plugged into that call site. `move_divergence_pct` flags when the two
    disagree by more than OPTIONS_EM_DIVERGENCE_THRESHOLD_PCT (default 25%)
    so compute_confidence can penalize an unreliable print instead of
    quietly picking one.

    call_ltp / put_ltp: {strike: last_traded_price} maps for one expiry.
    call_iv / put_iv: {strike: implied_volatility_pct} maps for the same
    expiry -- pass the live NSE feed's tables; omit (or pass {}) for EOD
    Bhavcopy data, which has no IV column, and the IV-based estimate is
    simply skipped, leaving expected_move_pts as the straddle figure alone
    (unchanged behavior for EOD runs).

    Returns None (never raises) if there's no strike with both a call AND
    a put premium to form a straddle from -- callers already handle a None
    expected_move by falling back to "n/a" / skipping deterministic strike
    selection for that horizon.
    """
    if not call_ltp or not put_ltp or not spot:
        return None

    common_strikes = set(call_ltp.keys()) & set(put_ltp.keys())
    if not common_strikes:
        return None

    atm_strike = min(common_strikes, key=lambda k: abs(k - spot))
    call_premium = call_ltp.get(atm_strike)
    put_premium = put_ltp.get(atm_strike)
    if not call_premium or not put_premium:
        return None

    straddle_move_pts = call_premium + put_premium

    iv_move_pts = None
    atm_iv = None
    if call_iv and put_iv and expiry:
        # Use call-side ATM IV with put as fallback -- the canonical quant
        # convention for index options (put-call parity keeps them close, but
        # averaging can halve the estimate if one side is stale/zero).
        atm_iv_val = call_iv.get(atm_strike) or put_iv.get(atm_strike)
        if atm_iv_val:
            atm_iv = atm_iv_val
            try:
                t_years = time_to_expiry_years(_parse_nse_date(expiry))
                sigma = _iv_to_frac(atm_iv)
                if sigma and sigma > 0 and t_years > 0:
                    iv_move_pts = spot * sigma * math.sqrt(t_years)
            except (ValueError, TypeError):
                iv_move_pts = None

    move_divergence_pct = None
    if iv_move_pts and straddle_move_pts:
        # BUG FIX: guard zero denominator -- min() can be 0 if one estimate is
        # 0.0, which would crash with ZeroDivisionError. Skip divergence calc
        # in that edge case; callers treat None divergence as "no penalty".
        min_move = min(straddle_move_pts, iv_move_pts)
        if min_move > 0:
            move_divergence_pct = round(
                abs(straddle_move_pts - iv_move_pts) / min_move * 100, 1
            )

    if iv_move_pts and iv_move_pts > straddle_move_pts:
        expected_move_pts, expected_move_basis = iv_move_pts, "iv"
    else:
        expected_move_pts, expected_move_basis = straddle_move_pts, "straddle"

    expected_move_pct = round((expected_move_pts / spot) * 100, 2) if spot else None

    return {
        "atm_strike": atm_strike,
        "atm_call_premium": call_premium,
        "atm_put_premium": put_premium,
        "atm_iv": round(atm_iv, 2) if atm_iv else None,
        "straddle_move_pts": round(straddle_move_pts, 2),
        "iv_move_pts": round(iv_move_pts, 2) if iv_move_pts else None,
        "move_divergence_pct": move_divergence_pct,
        "expected_move_pts": round(expected_move_pts, 2),
        "expected_move_pct": expected_move_pct,
        "expected_move_basis": expected_move_basis,
    }


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
                log.warning(
                    f"NSE Bhavcopy fetch failed for {trade_date.date()} via {host} "
                    f"(attempt {attempt}/{max_attempts_per_host}): {_describe_http_error(e)}"
                )
                if attempt < max_attempts_per_host and _is_retryable_nse_error(e):
                    time.sleep(1.0 * attempt)
                else:
                    break
    log.warning(
        f"NSE Bhavcopy fetch failed for {trade_date.date()} across all hosts "
        f"({', '.join(_BHAVCOPY_FO_HOSTS)}): {_describe_http_error(last_err)}"
    )
    return None


def fetch_latest_nse_bhavcopy_fo(max_days_back=6):
    now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
    skipped_weekdays = []
    for i in range(max_days_back + 1):
        candidate = now_ist - timedelta(days=i)
        # BUG FIX: NSE publishes Bhavcopy files by ~18:30 IST on most trading
        # days. The old 19:00 cutoff caused the script to skip today's already-
        # published file during the 18:30-19:00 window and fall back to
        # yesterday's data unnecessarily. 18:30 is a safer cutoff.
        if i == 0 and now_ist.time() < dtime(18, 30):
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
            # BUG FIX: for options the official settlement price (SttlmPric)
            # is the theoretically correct figure for daily MTM and end-of-day
            # premium. ClsPric is the last traded price and can be stale if the
            # option had no trades near close. Try SttlmPric first; fall back
            # to ClsPric if missing (e.g. on far-OTM strikes with no settlement).
            settle_px = float(r.get("SttlmPric") or r.get("ClsPric") or 0) or None
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
        "call_bid": {},
        "call_ask": {},
        "put_bid": {},
        "put_ask": {},
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
            snap["expected_move"] = compute_expected_move(
                snap["call_ltp"], snap["put_ltp"], data["spot"], data.get("vix"), snap["expiry"],
                call_iv=snap["call_iv"], put_iv=snap["put_iv"],
            )
            data["horizons"][horizon] = snap

        notes.append(
            f"Live NSE option-chain feed was unavailable this run -- per-horizon OI/PCR/max-pain "
            f"were filled from NSE's official EOD Bhavcopy dated {bhav_date.strftime('%d %b %Y')} "
            f"instead (last trading day's CLOSE, not live/intraday){staleness_note}."
        )
        return True
    except (ValueError, TypeError, KeyError, IndexError, AttributeError) as e:
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
        # Item 14: retry-with-delay instead of a single try -- transient
        # Yahoo Finance blips no longer cost us the whole data point.
        vix_hist = _retry_with_delay(
            lambda: yf.Ticker("^INDIAVIX").history(period=IV_RANK_LOOKBACK_PERIOD),
            "Yahoo Finance VIX",
            max_attempts=3,
            notes=notes,
        )
        if vix_hist is not None and not vix_hist.empty:
            data["vix"] = round(float(vix_hist["Close"].iloc[-1]), 2)
            data["vix_source"] = "Yahoo Finance (^INDIAVIX)"
            if len(vix_hist) >= 2:
                prev = float(vix_hist["Close"].iloc[-2])
                data["vix_change_pct"] = round(((data["vix"] - prev) / prev * 100) if prev and prev > 0 else 0.0, 2)
            rank, pct, days_used = compute_iv_rank_percentile(vix_hist["Close"], data["vix"])
            data["iv_rank"] = rank
            data["iv_percentile"] = pct
            data["iv_rank_days"] = days_used
        else:
            # Item 14: secondary source. yfinance gave us nothing after
            # retries -- try NSE's own live indices endpoint for at least a
            # current VIX level (no history there, so IV rank/percentile
            # stay unavailable and are noted as such rather than guessed).
            nse_vix = _retry_with_delay(
                _fetch_nse_vix_snapshot, "NSE India VIX (secondary source)", max_attempts=2, notes=notes,
            )
            if nse_vix:
                data["vix"] = nse_vix["vix"]
                data["vix_change_pct"] = nse_vix.get("vix_change_pct")
                data["vix_source"] = "NSE live indices API (secondary source -- current level only, no IV rank/percentile)"
                notes.append(
                    "India VIX came from the NSE secondary source after Yahoo Finance failed -- "
                    "no history available from this source, so IV rank/percentile are unavailable this run."
                )

        spot_hist = _retry_with_delay(
            lambda: yf.Ticker("^NSEI").history(period="1d"),
            "Yahoo Finance Nifty spot",
            max_attempts=3,
            notes=notes,
        )
        if spot_hist is not None and not spot_hist.empty:
            data["spot"] = round(float(spot_hist["Close"].iloc[-1]), 2)
            data["spot_source"] = "Yahoo Finance (^NSEI)"
        # Note: if this also fails, data["spot"] is still backfilled below
        # from the NSE option-chain's own underlyingValue (existing
        # secondary source), or the EOD Bhavcopy futures-close proxy in
        # _fill_horizons_from_bhavcopy() as a tertiary fallback.
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
            snap["expected_move"] = compute_expected_move(
                snap["call_ltp"], snap["put_ltp"], data["spot"], data.get("vix"), snap["expiry"],
                call_iv=snap["call_iv"], put_iv=snap["put_iv"],
            )
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
    """Convert IV to decimal fraction. Values > 1.0 are treated as percentages."""
    if iv is None:
        return None
    iv = float(iv)
    return iv / 100.0 if iv > 1.0 else iv


def time_to_expiry_years(expiry_dt, now=None):
    now = now or datetime.now(ZoneInfo("Asia/Kolkata"))
    expiry_close = datetime.combine(expiry_dt.date(), dtime(15, 30), tzinfo=ZoneInfo("Asia/Kolkata"))
    return max((expiry_close - now).total_seconds() / (365.0 * 86400), 1e-6)


def bs_greeks(spot, strike, t_years, iv, opt_type, r=RISK_FREE_RATE, q=DIVIDEND_YIELD):
    """
    Black-Scholes-Merton greeks with a continuous dividend yield q on the
    underlying. Nifty is a dividend-paying index, so q=0 (the previous
    behavior) understates the cost-of-carry drag on calls and overstates it
    on puts -- every d1/d2, delta, gamma, theta, and vega below carries the
    standard e^{-qT} discount factor to correct for that.
    """
    try:
        sigma = _iv_to_frac(iv)
        if spot is None or strike is None or sigma is None or t_years is None:
            return None
        if t_years <= 0 or sigma <= 0 or spot <= 0 or strike <= 0:
            return None
        sqrt_t = math.sqrt(t_years)
        d1 = (math.log(spot / strike) + (r - q + 0.5 * sigma ** 2) * t_years) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t
        pdf_d1 = _norm_pdf(d1)
        disc_q = math.exp(-q * t_years)
        disc_r = math.exp(-r * t_years)
        if opt_type == "CE":
            delta = disc_q * _norm_cdf(d1)
            theta = (
                -(spot * disc_q * pdf_d1 * sigma) / (2 * sqrt_t)
                - r * strike * disc_r * _norm_cdf(d2)
                + q * spot * disc_q * _norm_cdf(d1)
            ) / 365.0
        else:
            delta = disc_q * (_norm_cdf(d1) - 1.0)
            theta = (
                -(spot * disc_q * pdf_d1 * sigma) / (2 * sqrt_t)
                + r * strike * disc_r * _norm_cdf(-d2)
                - q * spot * disc_q * _norm_cdf(-d1)
            ) / 365.0
        gamma = disc_q * pdf_d1 / (spot * sigma * sqrt_t)
        vega = spot * disc_q * pdf_d1 * sqrt_t / 100.0
        return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}
    except (ValueError, ZeroDivisionError, OverflowError, TypeError):
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


def compute_pop(spot, t_years, iv, payoff_fn, breakevens, r=RISK_FREE_RATE, q=DIVIDEND_YIELD):
    sigma = _iv_to_frac(iv)
    if spot is None or sigma is None or t_years is None or t_years <= 0 or sigma <= 0:
        return None

    def cdf_le(K):
        if K <= 0:
            return 0.0
        sqrt_t = math.sqrt(t_years)
        d1 = (math.log(spot / K) + (r - q + 0.5 * sigma ** 2) * t_years) / (sigma * sqrt_t)
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
    except (ValueError, ZeroDivisionError, OverflowError, TypeError):
        return None
    return round(max(0.0, min(1.0, prob_profit)) * 100, 1)


def compute_touch_probability(spot, t_years, iv, barrier, r=RISK_FREE_RATE, q=DIVIDEND_YIELD):
    """
    BUG FIX: the previous implementation used an asymmetric branching formula
    (different d1/d2 formulas for a > 0 vs a < 0) which is non-standard and
    gives wrong probabilities. The correct formula is the standard
    reflection-principle (first-passage) probability for a log-normal process:

        P(touch) = N(d+) + exp(2*mu*a / sigma^2) * N(d-)

    where a = ln(barrier/spot), d+ = (a - mu*T)/(sigma*sqrt(T)),
    d- = (-a - mu*T)/(sigma*sqrt(T)), and mu = r - q - 0.5*sigma^2.

    This formula is symmetric -- the sign of `a` already encodes direction
    (positive = upward barrier, negative = downward barrier). The same
    formula handles both cases without branching.
    """
    sigma = _iv_to_frac(iv)
    if (
        spot is None or sigma is None or t_years is None or t_years <= 0
        or sigma <= 0 or barrier is None or barrier <= 0
    ):
        return None
    try:
        if barrier == spot:
            return 100.0
        mu = r - q - 0.5 * sigma ** 2
        a = math.log(barrier / spot)
        sqrt_t = math.sqrt(t_years)
        # Standard reflection-principle formula (identical for up/down barriers):
        d_plus  = (a - mu * t_years) / (sigma * sqrt_t)   # towards barrier
        d_minus = (-a - mu * t_years) / (sigma * sqrt_t)  # reflected path
        exponent = 2.0 * mu * a / (sigma ** 2)
        # Clamp exponent to avoid overflow in exp() for extreme parameters
        prob = _norm_cdf(d_plus) + math.exp(min(exponent, 700.0)) * _norm_cdf(d_minus)
    except (ValueError, ZeroDivisionError, OverflowError, TypeError):
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


_DEFINED_RISK_MARGIN_MULTIPLIER = float(os.getenv("OPTIONS_MARGIN_MULTIPLIER", "1.2"))


def estimate_margin_and_rom(max_loss_inr, max_profit_inr, priced_legs, lot_size):
    """
    Approximate NSE SPAN + Exposure margin for a VERIFIED defined-risk
    options spread (bull put / bear call / iron condor -- this function is
    only ever reached after apply_verified_payoff has already rejected any
    unbounded-risk structure).

    BUG FIXED: this previously computed an `exposure_buffer` as 3% of the
    short leg's full notional (strike x lot_size) and took the max of that
    against max_loss_inr. For a typical Nifty spread that buffer dwarfs the
    real max loss -- e.g. spot 25000, lot_size 75, one short leg: 0.03 x
    25000 x 75 = Rs 56,250, versus a real max loss on a 100-150pt wide
    spread of roughly Rs 5,000-8,000. That 3% figure is the right order of
    magnitude for margining a NAKED/uncovered short option, not a
    risk-defined spread. Since Feb 2021 SEBI peak-margin norms, exchanges
    grant same-underlying/same-expiry spread margin benefit on these
    structures, so the real SPAN+Exposure margin blocked is close to (and
    only modestly above) the worst-case loss -- commonly ~1.15-1.3x max
    loss in practice. Using the 3% notional buffer silently overstated
    margin by 5-10x and understated Return on Margin by the same factor,
    making genuinely good defined-risk trades look capital-inefficient.
    """
    if max_loss_inr is None or max_loss_inr <= 0:
        margin = max(max_loss_inr or 0.0, 1.0)
    else:
        margin = max_loss_inr * _DEFINED_RISK_MARGIN_MULTIPLIER
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


def _leg_bid_ask(horizon_snap, strike, opt_type):
    key = float(strike)
    lookup_key = int(key) if key.is_integer() else key
    bid_table = (horizon_snap or {}).get("call_bid" if opt_type == "CE" else "put_bid", {})
    ask_table = (horizon_snap or {}).get("call_ask" if opt_type == "CE" else "put_ask", {})
    bid = bid_table.get(strike, bid_table.get(lookup_key))
    ask = ask_table.get(strike, ask_table.get(lookup_key))
    return bid, ask


def compute_bid_ask_spread_pct(bid, ask):
    """
    Relative bid-ask spread as a percent of mid price. Returns None (not a
    rejection by itself -- caller decides) when either side is missing or
    non-positive, or when the quote is crossed/locked (ask <= bid), since
    none of those represent a computable two-sided spread.
    """
    if not bid or not ask or bid <= 0 or ask <= 0 or ask <= bid:
        return None
    mid = (bid + ask) / 2.0
    return round((ask - bid) / mid * 100, 2)


def _leg_liquidity_check(horizon_snap, strike, opt_type, has_quote_data):
    """
    Real liquidity gate for one leg, using actual bid/ask instead of OI
    presence as a proxy. Returns (ok, reason_or_spread_pct).

    If this horizon's snapshot has no bid/ask data at all (EOD Bhavcopy,
    or a live fetch where the book fields didn't come through), the gate
    can't evaluate anything and does NOT reject -- has_quote_data being
    False means "skip", not "pass"; degrades to premium-only verification,
    same behavior as before this gate existed. If quote data IS present
    for this horizon but this specific leg has no two-sided market, that's
    a real illiquidity signal and the leg is rejected.
    """
    if not has_quote_data:
        return True, None
    bid, ask = _leg_bid_ask(horizon_snap, strike, opt_type)
    spread_pct = compute_bid_ask_spread_pct(bid, ask)
    if spread_pct is None:
        return False, "no live two-sided quote (missing/zero bid or ask)"
    if spread_pct > MAX_LEG_SPREAD_PCT:
        return False, f"bid-ask spread {spread_pct:g}% of mid exceeds the {MAX_LEG_SPREAD_PCT:g}% liquidity gate"
    return True, spread_pct


def compute_strategy_payoff(legs_text, horizon_snap, lot_size=NIFTY_LOT_SIZE):
    legs = parse_legs(legs_text)
    if not legs:
        return {"ok": False, "reason": "Could not parse strikes/legs from the model's output."}

    # Liquidity gate is only "live" (able to reject) when THIS horizon's
    # snapshot actually carries bid/ask data -- EOD Bhavcopy and any live
    # fetch that came back without book fields have none of either side,
    # so the gate skips rather than rejecting on absent data it never had
    # a chance to evaluate.
    has_quote_data = bool(
        (horizon_snap or {}).get("call_bid") or (horizon_snap or {}).get("put_bid")
    )

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
        liquid_ok, liquidity_info = _leg_liquidity_check(
            horizon_snap, leg["strike"], leg["type"], has_quote_data
        )
        if not liquid_ok:
            return {
                "ok": False,
                "reason": (
                    f"{leg['action']} {leg['strike']:g} {leg['type']} rejected on liquidity: "
                    f"{liquidity_info} -- this leg is not tradeable at a fair price right now."
                ),
            }
        leg_spread_pct = liquidity_info if isinstance(liquidity_info, (int, float)) else None
        priced.append({**leg, "premium": float(premium), "spread_pct": leg_spread_pct})

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
    # Bug fix: the right tail (S -> +inf, relevant to calls) has genuine unbounded
    # risk when payoff keeps FALLING as S rises, i.e. slope_right < 0 -- that part
    # was correct. But the left tail (S -> 0, relevant to puts) has genuine
    # unbounded/naked risk when payoff keeps FALLING as S drops, which in terms of
    # this slope (measured in the increasing-S direction) means slope_left must be
    # POSITIVE, not negative. The previous `slope_left < -0.01` test had the sign
    # backwards for puts: it flagged net-long/defined-risk put structures (where
    # payoff RISES on further downside, slope_left negative) as "undefined risk",
    # while a genuinely naked short put (slope_left positive) would slip through
    # unflagged. Verified against naked-short-put, naked-short-call, and standard
    # 2-leg spread cases before landing this fix.
    unbounded_risk = slope_left > 0.01 or slope_right < -0.01

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


_STRIKE_MENTION_RE = re.compile(
    r"\b(?:(short|long)\s+)?(\d{4,6}(?:\.\d+)?)\s*(call|put|ce|pe)\b",
    re.IGNORECASE,
)


def _scrub_stale_strike_claims(rationale, priced_legs):
    """
    BUG FIX: the model's freeform strike_rationale sentence is written
    independently of the 'legs' field, and can name a strike that doesn't
    match what's actually in priced_legs -- e.g. rationale says "the short
    23500 put" while the real (verified, tradeable) short leg is 22700 PE.
    This happens whenever the model's own prose and its own legs disagree,
    and separately whenever deterministic strike selection (select_best_
    strikes) or the repair pass swaps in different strikes than the model
    originally wrote about. A rationale naming the wrong strike is worse
    than no rationale -- a trader skimming the report could mistake the
    quoted number for the strike to actually sell/buy.

    BUG FIX: this used to only check whether the mentioned strike number
    existed *somewhere* among the real legs, and only scrubbed when *every*
    mentioned strike was absent. Two failure modes slipped through as a
    result: (1) a sentence calling a real strike "the short X" when X is
    actually the long leg (the number passes the existence check even
    though the claimed role is wrong), and (2) a sentence mixing one stale
    strike with other, coincidentally-correct strikes (e.g. injected by a
    substitution note appended before this scrub runs) -- since not *all*
    mentions were wrong, nothing got scrubbed even though one of them was
    actively misleading. Now: each mention is checked against the real
    (strike, option-type) pair for a match, and if the mention names a
    short/long role, that role must match the real leg's actual action.
    A single bad mention is enough to scrub the whole sentence.
    """
    if not rationale or not priced_legs:
        return rationale
    mentioned = _STRIKE_MENTION_RE.findall(rationale)
    if not mentioned:
        return rationale

    real_action_by_strike_type = {}
    for leg in priced_legs:
        real_action_by_strike_type[(leg["strike"], leg["type"].upper())] = leg["action"]

    def _mention_is_wrong(role, num, opt_word):
        strike = float(num)
        opt_type = "CE" if opt_word.lower() in ("call", "ce") else "PE"
        action = real_action_by_strike_type.get((strike, opt_type))
        if action is None:
            return True  # strike/type combo isn't in the real legs at all
        if role and role.lower() == "short" and action != "Sell":
            return True  # claimed short, but that strike is actually the long leg
        if role and role.lower() == "long" and action != "Buy":
            return True  # claimed long, but that strike is actually the short leg
        return False

    if any(_mention_is_wrong(role, num, opt_word) for role, num, opt_word in mentioned):
        return (
            "Selected primarily on OI/liquidity/expected-move positioning "
            "-- see the verified short-strike detail below (the model's "
            "original strike-number claim here didn't match the actual "
            "priced legs and was dropped to avoid a misleading strike)"
        )
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


# Item 22: stale/EOD-derived figures were being displayed with the same
# precision as live-tick data -- a breakeven computed off yesterday's
# closing prices (no live IV, no live bid/ask) doesn't actually resolve to
# the nearest paisa, and a Return-on-Margin/Capital-Efficiency percentage
# built on top of that same stale premium is just as imprecise, however
# large the number gets. These round display precision down to match what
# the underlying data can actually support, rather than implying a false
# level of certainty. Does not touch the underlying float used for any
# gating/math -- display only.
def _fmt_breakeven_pts(value, is_eod):
    return f"{value:,.0f}" if is_eod else f"{value:,.2f}"


def _fmt_pct_precision(value, is_eod, implausible_threshold=1000):
    """Formats a percentage figure (ROI/ROM/Capital Efficiency) at a
    precision matching data quality. EOD-derived figures round to the
    nearest whole percent instead of one decimal place. A figure past
    implausible_threshold is itself a sign the underlying premiums are
    mismatched/stale rather than a genuine return, so it's flagged instead
    of displayed as a bare huge number."""
    if value is None:
        return "n/a"
    if abs(value) >= implausible_threshold:
        return f">{implausible_threshold:,.0f}% (implausible at this precision -- verify premiums)"
    return f"{value:.0f}%" if is_eod else f"{value:.1f}%"


def compute_trade_quality_score(priced_legs, horizon_snap, ev_inr, max_loss, pop_pct, reward_risk_ratio, conf_pct, is_eod, sources=None):
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
    leg_spreads = [l["spread_pct"] for l in (priced_legs or []) if l.get("spread_pct") is not None]
    if leg_spreads:
        worst_spread_pct = max(leg_spreads)
        # 0% spread -> 100; at the MAX_LEG_SPREAD_PCT gate itself -> 30
        # (a leg wider than that never reaches here -- compute_strategy_payoff
        # already rejected the whole trade before scoring runs).
        liq_score = _clamp01_100(100 - (worst_spread_pct / MAX_LEG_SPREAD_PCT) * 70)
    elif is_eod:
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

    # BUG FIX: _categorize_source() tags today's news sources into an
    # "Event Calendar" bucket (RBI/FOMC/budget/election keywords) purely
    # for the display table in render_market_data_inputs_html() -- an
    # RBI-policy or FOMC headline in today's sources previously had zero
    # effect on either the confidence check-list or this score. A live
    # event-calendar hit is a real risk factor for a risk-defined options
    # structure (a policy surprise can gap the underlying through both
    # short strikes), so it now knocks points off here too, independent of
    # whatever compute_confidence already deducted for it.
    event_sources = [s for s in (sources or []) if _categorize_source(s) == "Event Calendar"]
    if event_sources and score > 0:
        score = max(0, score - 10)
        penalty_notes.append(
            f"{len(event_sources)} event-calendar headline(s) in today's sources "
            f"(RBI/FOMC/budget/election) -- score reduced 10 pts; verify this "
            f"horizon's expiry doesn't span the event before sizing"
        )

    return score, components, penalty_notes


def compute_confidence(priced_legs, horizon_snap, breakevens, is_eod, vix, spot=None, sources=None, gamma_available=None):
    checks = []

    # Item 21: gamma (and the rest of the Greeks) being unavailable used to
    # be purely a footnote on net_greeks/the portfolio gamma summary --
    # it never touched confidence or the Trade Quality Score, so a horizon
    # with literally no verified gap-risk read could still score however
    # the other checks happened to land. Treat it as its own confidence
    # check, same as OI/VIX/liquidity availability below. gamma_available
    # is None when the caller hasn't determined it yet (skips the check
    # rather than guessing) -- see apply_verified_payoff's greeks_ok.
    if gamma_available is not None:
        checks.append((
            gamma_available,
            "Live IV available -- Greeks (incl. gamma) verified for this structure" if gamma_available
            else "Gamma/Greeks unavailable this run -- gap risk on this structure is unverified",
        ))

    # BUG FIX: an "Event Calendar" source hit (RBI/FOMC/budget/election --
    # see _categorize_source()/_CATEGORY_KEYWORDS) used to be display-only,
    # surfaced in the Market Data Inputs table but never fed into any of
    # the checks below. Treat it like the other risk checks in this
    # function: a hit counts against confidence, since a scheduled macro
    # event inside the expiry window is exactly the kind of gap risk a
    # risk-defined spread's static OI/VIX/max-pain checks can't see.
    event_sources = [s for s in (sources or []) if _categorize_source(s) == "Event Calendar"]
    if event_sources:
        checks.append((
            False,
            f"{len(event_sources)} event-calendar headline(s) in today's sources "
            f"(RBI/FOMC/budget/election) -- confirm none fall inside this horizon's "
            f"expiry window before sizing",
        ))

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

    leg_spreads = [l["spread_pct"] for l in (priced_legs or []) if l.get("spread_pct") is not None]
    if leg_spreads:
        worst_spread_pct = max(leg_spreads)
        tight = worst_spread_pct <= MAX_LEG_SPREAD_PCT / 2
        checks.append((
            tight,
            f"Widest leg spread {worst_spread_pct:g}% of mid -- comfortably inside the "
            f"{MAX_LEG_SPREAD_PCT:g}% liquidity gate" if tight
            else f"Widest leg spread {worst_spread_pct:g}% of mid -- close to the "
                 f"{MAX_LEG_SPREAD_PCT:g}% liquidity gate; expect real slippage vs. mid on entry/exit",
        ))

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
    move_divergence_pct = exp_move.get("move_divergence_pct")
    if move_divergence_pct is not None:
        agree = move_divergence_pct < EM_DIVERGENCE_THRESHOLD_PCT
        checks.append((
            agree,
            f"Straddle and IV-based expected-move estimates agree within "
            f"{move_divergence_pct:g}%" if agree
            else f"Straddle (±{exp_move.get('straddle_move_pts'):g}) and IV-based 1σ "
                 f"(±{exp_move.get('iv_move_pts'):g}) expected-move estimates diverge by "
                 f"{move_divergence_pct:g}% -- one is likely distorted by thin ATM liquidity "
                 f"or a stale/skewed IV quote; using the wider figure for gating",
        ))

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


def apply_verified_payoff(horizon_dict, horizon_snap, spot=None, vix=None, sources=None):
    horizon_dict["bias_reason"] = _scrub_pcr_direction_claim(
        _scrub_pcr_mischaracterization(
            _strip_cap_claims(horizon_dict.get("bias_reason")),
            (horizon_snap or {}).get("pcr_oi"),
        ),
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
    existing = _scrub_stale_strike_claims(existing, priced_legs)
    # BUG FIX: run() used to bake any deterministic-strike-selection
    # substitution_note directly into horizon_dict["strike_rationale"]
    # (concatenated onto the model's original, still-unscrubbed prose)
    # *before* this function ever ran. Since "_raw_strike_rationale" is
    # only cached once, that first cache captured the corrupted text --
    # stale strike numbers sitting right next to the substitution note's
    # correct ones -- and _scrub_stale_strike_claims saw some correct
    # numbers in the sentence and left the stale one in place. The
    # substitution note is now stashed separately (h["_substitution_note"])
    # and only appended here, after scrubbing has already cleaned the
    # model's original text -- the same pattern rationale_addendum uses.
    substitution_note = horizon_dict.get("_substitution_note")
    suffixes = [s for s in (rationale_addendum, substitution_note) if s]
    if suffixes:
        suffix_text = " ".join(f"({s})" for s in suffixes)
        horizon_dict["strike_rationale"] = f"{existing} {suffix_text}" if existing else " ".join(suffixes)
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
                em_basis_for_filter = "IV-based 1σ" if exp_move_for_filter.get("expected_move_basis") == "iv" else "ATM straddle"
                reason_text = (
                    f"Short strike {nearest_short_for_filter['strike']:g} is only "
                    f"{near_distance_for_filter:.0f} pts from spot ({near_multiple_for_filter:.2f}x the "
                    f"±{em_pts:g}-pt expected move, {em_basis_for_filter}-derived) -- an Iron Condor's short "
                    f"strikes must sit outside the expected-move band by construction; this one contradicts "
                    f"its own range-bound thesis."
                )
                horizon_dict["max_loss"] = (
                    f"SHORT STRIKE INSIDE EXPECTED MOVE -- reject this trade "
                    f"({nearest_short_for_filter['strike']:g} is {near_distance_for_filter:.0f} pts from spot, "
                    f"{near_multiple_for_filter:.2f}x expected move)"
                )
                horizon_dict["max_profit"] = "n/a"
                horizon_dict["max_loss_pct_capital"] = "n/a"
                horizon_dict["max_profit_pct_capital"] = "n/a"
                horizon_dict["breakeven"] = ", ".join(_fmt_breakeven_pts(b, is_eod) for b in result["breakevens"]) if result["breakevens"] else "n/a"
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
    be = ", ".join(_fmt_breakeven_pts(b, is_eod) for b in result["breakevens"]) if result["breakevens"] else "n/a"

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
    # Item 20: max_loss <= 0 on a real defined-risk structure means the
    # premium collected/paid is claiming to fully fund (or overfund) the
    # widest wing -- an arbitrage-looking price that's a data artifact
    # (mismatched stale EOD last-traded prices across strikes/timestamps),
    # not a genuine riskless trade. This used to fall through as
    # reward_risk_ratio=inf and read downstream as "no capital at risk,
    # ~100% POP" -- Confidence (computed independently, see
    # compute_confidence) never saw it, so a horizon could show 76%
    # confidence in the same report where Suggested Sizing separately (and
    # correctly) rejected it as "Unverified max loss". Reject it here too,
    # the same way poor_reward_risk does, so both parts of the report agree.
    invalid_max_loss = max_loss <= 0
    poor_reward_risk = max_loss > 0 and reward_risk_ratio < MIN_REWARD_RISK_RATIO
    poor_credit_width = credit_width_pct is not None and credit_width_pct < MIN_CREDIT_WIDTH_PCT

    if invalid_max_loss:
        reason_text = (
            f"Computed max loss is ₹{max_loss:,.0f} (<= 0) on a defined-risk structure -- the "
            f"priced legs imply the premium fully funds (or overfunds) the widest wing, which "
            f"isn't a genuine riskless trade. This is almost always mismatched/stale leg prices "
            f"({premium_label}), not a real arbitrage -- reverify every leg's premium against a "
            f"live broker terminal before trusting this structure at all."
        )
        horizon_dict["max_loss"] = "UNVERIFIABLE MAX LOSS -- reject this trade"
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
        horizon_dict["reward_risk_ratio"] = "n/a -- max loss unverifiable (rejected)"
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

    if poor_reward_risk:
        reason_text = (
            f"Reward:Risk is only {reward_risk_ratio:.2f} (₹{max_profit:,.0f} max profit vs "
            f"₹{max_loss:,.0f} max loss per lot) -- below the {MIN_REWARD_RISK_RATIO:g} minimum."
        )
        horizon_dict["max_loss"] = f"POOR REWARD/RISK -- reject this trade"
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
        horizon_dict["reward_risk_ratio"] = f"{reward_risk_ratio:.2f} -- below minimum (rejected)"
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

    if poor_credit_width:
        rich_credit_flag += (
            f" ⚠ Low Premium Environment: Net credit is only {credit_width_pct:.1f}% of the {width:g}-point "
            f"spread width -- below your {MIN_CREDIT_WIDTH_PCT:g}% target."
        )

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
            band = f" -- probability band used for gating: {lo:,.0f}–{hi:,.0f}"
        basis = exp_move.get("expected_move_basis", "straddle")
        basis_label = "IV-based 1σ" if basis == "iv" else "ATM straddle"
        both_note = ""
        iv_pts = exp_move.get("iv_move_pts")
        straddle_pts = exp_move.get("straddle_move_pts")
        if iv_pts and straddle_pts:
            div_pct = exp_move.get("move_divergence_pct")
            both_note = f" [straddle: ±{straddle_pts:g}, IV 1σ: ±{iv_pts:g}"
            if div_pct is not None and div_pct >= EM_DIVERGENCE_THRESHOLD_PCT:
                both_note += f", diverge {div_pct:g}% -- using the wider of the two"
            both_note += "]"
        horizon_dict["expected_move"] = (
            f"±{exp_move['expected_move_pts']:g} pts (~{exp_move.get('expected_move_pct', 'n/a')}% of spot) "
            f"by expiry, from {basis_label}{band}{both_note}"
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
        f"~{_fmt_pct_precision(rom, is_eod)} (assumes the structure is held to expiry and achieves max profit; actual "
        f"realized returns may differ due to early exit, margin changes, or assignment risk)"
        if rom is not None else "n/a"
    )
    risk_margin_pct = round(max_loss / margin * 100, 1) if margin else None
    if rom is not None and risk_margin_pct is not None:
        horizon_dict["capital_efficiency"] = (
            f"Profit/Margin ~{_fmt_pct_precision(rom, is_eod)} · Risk/Margin ~{_fmt_pct_precision(risk_margin_pct, is_eod)} "
            f"(share of locked-up margin at stake if max loss is hit)"
        )
    else:
        horizon_dict["capital_efficiency"] = "n/a"

    conf_label, conf_pct, conf_checks = compute_confidence(
        priced_legs, horizon_snap, result["breakevens"], is_eod, vix, spot, sources,
        gamma_available=greeks_ok,
    )
    horizon_dict["confidence"] = conf_label
    horizon_dict["confidence_pct"] = conf_pct
    horizon_dict["confidence_reasons"] = conf_checks

    tq_score, tq_breakdown, tq_penalty_notes = compute_trade_quality_score(
        priced_legs, horizon_snap, ev_inr, max_loss, pop, reward_risk_ratio, conf_pct, is_eod, sources
    )
    horizon_dict["_trade_quality_score"] = tq_score
    # Item 19: liquidity previously only ever fed the Trade Quality Score as
    # a 10%-weighted component -- a horizon could score "✅ Consider" on the
    # strength of EV/R:R/POP alone while liquidity sat at its worst
    # acceptable value (30/100, the floor for a spread that just barely
    # clears MAX_LEG_SPREAD_PCT -- or the flat EOD placeholder when there's
    # no live quote data at all) with nothing ever blocking the
    # recommendation on that basis. Stashed here so
    # compute_horizon_recommendation() can gate on it directly.
    horizon_dict["_liquidity_score"] = tq_breakdown.get("Liquidity", (None, None))[0]
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

    return load_prompt("option/main", live_data_block=live_data_block)


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


def _normalize_weekly_recommendations(h):
    raw_recommendations = h.get("weekly_recommendations") or []
    if isinstance(raw_recommendations, list) and raw_recommendations:
        normalized = []
        for entry in raw_recommendations:
            if not isinstance(entry, dict):
                continue
            normalized.append({
                "label": str(entry.get("label") or "Option").strip() or "Option",
                "strategy_name": str(entry.get("strategy_name") or h.get("strategy_name") or "").strip(),
                "legs": str(entry.get("legs") or h.get("legs") or "").strip(),
                "reason": str(entry.get("reason") or entry.get("strike_rationale") or h.get("strike_rationale") or "").strip(),
            })
        if normalized:
            return normalized

    primary = {
        "label": "Primary",
        "strategy_name": str(h.get("strategy_name") or "").strip(),
        "legs": str(h.get("legs") or "").strip(),
        "reason": str(h.get("strike_rationale") or h.get("bias_reason") or "").strip(),
    }
    alternative = {
        "label": "Alternative",
        "strategy_name": primary["strategy_name"],
        "legs": primary["legs"],
        "reason": primary["reason"],
    }
    return [primary, alternative]


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

    recommendations = _normalize_weekly_recommendations(h)
    recommendation_rows = "".join(
        f'<tr><td style="padding:8px 10px;font-size:12px;font-family:{sans};color:#4A5063;border-top:1px solid #EDEAE2;width:38%;">{html.escape(rec["label"])}</td><td style="padding:8px 10px;font-size:12px;font-family:{sans};color:#14213D;border-top:1px solid #EDEAE2;">{html.escape(rec["strategy_name"])}<br><span style="color:#4A5063;">{html.escape(rec["legs"])}<br>{html.escape(rec["reason"])}</span></td></tr>'
        for rec in recommendations
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
        raw_row("Weekly Recommendations", recommendation_rows),
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


_PCR_CALL_DOMINANCE_BULLISH_RE = re.compile(
    r"call[^.!?]{0,20}dominan\w*[^.!?]*?(?:upward|upside|bullish|support)",
    re.IGNORECASE,
)
_PCR_PUT_DOMINANCE_BEARISH_RE = re.compile(
    r"put[^.!?]{0,20}dominan\w*[^.!?]*?(?:downward|downside|bearish|resistance)",
    re.IGNORECASE,
)


def _scrub_pcr_direction_claim(bias_reason, pcr):
    """
    BUG FIX: the model would sometimes write a bias_reason sentence like
    "PCR is 0.83 (<1) indicating call-side dominance ... suggesting upward
    pressure" -- this has the OI-analysis causality backwards. This file's
    own classify_pcr()/_pcr_implied_direction() (used elsewhere to flag
    bias/PCR conflicts) encode the standard convention: heavy CALL open
    interest is a resistance/bearish signal (call writers are betting price
    stays below that strike), heavy PUT open interest is a support/bullish
    signal (put writers are betting price stays above that strike) -- the
    OPPOSITE of the naive "more calls being traded = more bullish
    speculation" reading. _scrub_bias_pcr_conflict only catches the case
    where the model's bias LABEL contradicts a decisive PCR reading; it
    never catches a directionally-backwards CAUSAL CLAIM inside the
    rationale text itself, which is what actually shipped in this report
    for a PCR of 0.83 (itself within the neutral 0.7-1.2 band, so not even
    a decisive reading either way). This scrubber replaces such backwards
    claims with an accurate, convention-consistent description.
    """
    if not bias_reason or pcr is None:
        return bias_reason
    band = classify_pcr(pcr)
    sentences = re.split(r"(?<=[.!?])\s+", bias_reason.strip())
    fixed = []
    for s in sentences:
        if _PCR_CALL_DOMINANCE_BULLISH_RE.search(s) or _PCR_PUT_DOMINANCE_BEARISH_RE.search(s):
            fixed.append(
                f"PCR (OI) is {pcr:g} -- {band}. (By OI-analysis convention, heavy call OI "
                f"signals overhead resistance and heavy put OI signals underlying support -- "
                f"the opposite of reading heavy call OI as bullish.)"
            )
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
        # Item 21: portfolio-wide gamma unavailability used to be buried as
        # a single gray-text sentence inside the Portfolio View paragraph --
        # easy to skim past even though it means gap risk across the ENTIRE
        # portfolio is unverified this run. The caller now also gets a bool
        # so it can render this as its own bold warning line, the same
        # visual tier as the aggregate-cap verdict, instead of a footnote.
        return (
            "Combined portfolio gamma: not computable this run (live IV was unavailable "
            "for one or more legs across all horizons).",
            True,
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
        f"partly offset.",
        False,
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
    green, amber, red, gray, blue, teal = "#2F5233", "#A6812F", "#8B2E2E", "#8A8F9C", "#3D6690", "#2E6E73"

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
        elif "UNVERIFIABLE MAX LOSS" in ml:
            reason = "Unverifiable Max Loss -- Pricing Anomaly"
        else:
            reason = "Unverified"
        return "❌ Skip", red, reason

    # Item 8: select_best_strikes() now returns a Watchlist near-miss
    # candidate (see NEAR_MISS_TOLERANCE_PCT / WATCHLIST_EM_BAND_PCT) when
    # nothing cleared the strict live gates. It's a real, priced, defined-
    # risk structure (so it still passes the _horizon_rejected check above),
    # just one that missed the strict thresholds -- give it its own tier
    # rather than letting it silently read as a full "Consider".
    if h.get("_watchlist_tier"):
        return "🔍 Watchlist", teal, "Near-Miss on Strict Gates"

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

    # Item 19: liquidity hard floor -- see MIN_LIQUIDITY_SCORE_FOR_CONSIDER.
    # Placed after the other hard-gate checks (so a genuinely broken trade
    # still reports its own real rejection reason first) but before the
    # quality-score branch, so thin/unknown liquidity can never read as a
    # full "✅ Consider" no matter how the rest of the math looks.
    liq_score = h.get("_liquidity_score")
    if isinstance(liq_score, (int, float)) and liq_score <= MIN_LIQUIDITY_SCORE_FOR_CONSIDER:
        return "⚠ Caution", amber, f"Liquidity Too Thin ({liq_score:.0f}/100) -- Verify Fill Quality"

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


def _sizing_multiplier(h, action_label):
    """
    Item 13: previously this label was purely cosmetic -- compute_suggested_
    sizing() looked up action_label just to append it to a footnote, while
    `lots` itself was still sized off the per-horizon capital cap alone. A
    "Half Size" or "Watchlist" trade got exactly the same lot count as a
    "Full Size" one; only the text next to it differed. This derates the
    actual lot count so confidence tier changes the real position, not just
    the label.
    """
    if h.get("_watchlist_tier"):
        # Item 8's near-miss candidates never cleared the strict live gates
        # at all -- size well below "Half Size" rather than at half.
        return 0.25
    if action_label == "Half Size":
        return 0.5
    if action_label == "Watchlist":
        return 0.25
    if action_label == "Skip":
        return 0.0
    return 1.0


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

        # Item 13: actually apply the confidence-tier derating instead of
        # only noting it -- see _sizing_multiplier(). Applied after the
        # MAX_LOTS_PER_HORIZON ceiling so a Watchlist/Half-Size trade is a
        # fraction of the same ceiling everything else is capped at, not a
        # fraction of an already-uncapped number.
        mult = _sizing_multiplier(h, action_label)
        if mult < 1.0:
            full_size_lots = lots
            lots = int(lots * mult)
            size_note = f"{action_label} tier -- sized at {mult:g}x ({full_size_lots} -> {lots} lots)"
            note = f"{note}. {size_note}" if note else size_note
            if lots < 1:
                plan.append([name, None, f"{note} -- rounds down to 0 lots, skip"])
                continue

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
    except (TypeError, ValueError) as e:
        log.warning(f"Failed to parse aggregate_pct '{aggregate_pct}': {e}")

    verdict = (
        f"⚠ EXCEEDS the {AGGREGATE_CAP_PCT:.0f}% worst-case combined cap -- reduce position size before entering."
        if over_cap else
        f"✅ Stays within the {AGGREGATE_CAP_PCT:.0f}% worst-case combined cap."
    )
    gamma_summary, gamma_unavailable_all = compute_portfolio_gamma_summary(horizons)
    portfolio_text = _strip_gamma_claims(_strip_cap_claims(portfolio_view))
    portfolio_text = _scrub_portfolio_view_contradictions(portfolio_text, horizons)
    portfolio_text = _scrub_portfolio_view_structure_type_contradiction(portfolio_text, horizons)
    portfolio_text = _scrub_portfolio_view_directional_contradiction(portfolio_text, horizons)
    sizing_plan = compute_suggested_sizing(horizons)
    sizing_html = render_suggested_sizing_html(sizing_plan, sans)
    # Item 21: bold, same-tier warning line as the aggregate-cap verdict --
    # not just the gray-text sentence inside gamma_summary below -- so
    # portfolio-wide unverified gap risk can't be skimmed past.
    gamma_gate_html = (
        f'<div style="font-family:{sans};font-size:12px;font-weight:700;color:#8B2E2E;margin-top:8px;">'
        f'⚠ Portfolio gamma unavailable this run -- gap risk across every horizon is unverified; '
        f'treat aggregate sizing above as an upper bound, not a confirmed figure.</div>'
        if gamma_unavailable_all else ""
    )
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
        f'{gamma_gate_html}'
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
              <p style="margin:6px 0 0;font-family:{sans};font-size:12px;color:#B7BEC9;">Weekly &middot; Next Week &middot; Next to Next Week &mdash; Risk-Defined Only</p>
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
    subject = f"Nifty Option Strategy Note — {config.get_date_with_suffix(now_ist)} · {time_str}"

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
            server.quit()
        log.info("Nifty options strategy email sent successfully.")
        return True
    except smtplib.SMTPAuthenticationError:
        log.error(
            "SMTP Authentication Error: check EMAIL_FROM/EMAIL_PASSWORD "
            "(use a Gmail App Password, not the account password)."
        )
        return False
    except Exception as e:
        log.error(f"Failed to send Nifty options strategy email: {e}")
        traceback.print_exc()
    return False


def finalize_horizons(horizons, live_data, sources=None):
    by_name = {str(h.get("horizon") or "").strip(): h for h in horizons}

    if "Weekly" in by_name:
        by_name["Weekly"]["next_week_bias"] = "Neutral (insufficient evidence)"

    total_at_risk = 0.0
    for name, h in by_name.items():
        if h.get("_verified_max_loss_inr") is not None:
            continue
        snap = (live_data.get("horizons") or {}).get(name, {})
        apply_verified_payoff(h, snap, live_data.get("spot"), live_data.get("vix"), sources)

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
        or "SHORT STRIKE INSIDE EXPECTED MOVE" in ml or "UNVERIFIABLE MAX LOSS" in ml
    )


def _aggregate_risk_from_verified(horizons):
    """
    Sum the already-computed per-lot max loss across horizons without
    re-running apply_verified_payoff. BUG FIXED: run() previously called
    reverify_horizons(..., only_names=None) as a final step *after*
    repair_rejected_legs had already reverified everything it touched --
    that unconditionally re-applied apply_verified_payoff to every horizon
    a second (sometimes third) time purely to get an aggregate percentage,
    which is wasted work with no effect on the numbers (the computation is
    idempotent) but made the aggregate-cap check dependent on re-deriving
    state that was already sitting on each horizon dict.
    """
    total_at_risk = sum(
        v for v in (h.get("_verified_max_loss_inr", 0.0) for h in horizons)
        if v not in (None, float("inf"))
    )
    aggregate_pct = round(total_at_risk / TOTAL_CAPITAL_INR * 100, 2) if TOTAL_CAPITAL_INR else 0.0
    over_cap = aggregate_pct > AGGREGATE_CAP_PCT
    return aggregate_pct, over_cap


def reverify_horizons(horizons, live_data, only_names=None, sources=None):
    for h in horizons:
        name = h.get("horizon")
        if only_names is not None and name not in only_names:
            continue
        snap = (live_data.get("horizons") or {}).get(name, {})
        apply_verified_payoff(h, snap, live_data.get("spot"), live_data.get("vix"), sources)

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

    return load_prompt("option/repair", live_data_block=live_data_block, bad_block=bad_block)


def repair_rejected_legs(horizons, live_data, sources=None):
    """
    Reasons over horizons + live_data that were ALREADY fetched earlier in
    the run -- no new facts to find -- so this uses the non-live
    generate_synthesis() tier (plain Groq -> plain Gemini) rather than the
    full live-search cascade. Keeps groq/compound, Tavily, Gemini-
    grounding, and Mistral quota free for the calls in this run that
    actually depend on live search.
    """
    rejected = [h for h in horizons if _horizon_rejected(h)]
    if not rejected:
        return horizons

    repair_prompt = build_repair_prompt(rejected, live_data)
    try:
        repair_text = swing.generate_synthesis(
            repair_prompt,
            validate_fn=lambda t: _parse_analysis_json(t)[0] is not None,
            log_label="option repair pass",
        )
    except Exception as e:
        log.warning(f"Repair pass call failed; keeping original rejection(s). Exception: {e}", exc_info=True)
        return horizons

    if not repair_text:
        log.warning("Repair pass produced no output; keeping original rejection(s).")
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
        log.warning("Repair pass returned unparseable JSON; keeping original rejection(s).")
        return horizons

    if not fixed_by_name:
        return horizons

    for h in horizons:
        fix = fixed_by_name.get(h.get("horizon"))
        if fix:
            h["legs"] = fix["legs"]
            if fix.get("strategy_name"):
                h["strategy_name"] = fix["strategy_name"]
            # BUG FIX: apply_verified_payoff only ever populates
            # "_raw_strike_rationale" once (`if "_raw_strike_rationale"
            # not in horizon_dict`), caching whatever strike_rationale
            # prose existed from the FIRST pass -- i.e. the version that
            # described the now-rejected legs we just overwrote above.
            # Left in place, the repaired horizon would reverify with new
            # strikes but keep displaying a rationale sentence written
            # about the old (rejected, structurally different) strikes.
            # Clear both so the repaired trade gets an accurate rationale
            # derived from its actual new legs instead of a stale
            # description of a trade that no longer exists.
            h.pop("strike_rationale", None)
            h.pop("_raw_strike_rationale", None)
            h.pop("_substitution_note", None)

    reverify_horizons(horizons, live_data, only_names=set(fixed_by_name.keys()), sources=sources)
    still_bad = [h.get("horizon") for h in horizons if h.get("horizon") in fixed_by_name and _horizon_rejected(h)]
    if still_bad:
        log.warning(f"Repair pass attempted but still rejected after retry: {', '.join(still_bad)}")
    return horizons


def build_reformat_prompt(raw_analysis, live_data):
    """
    BUG FIX: _parse_analysis_json() previously had no fallback when the model
    ignored the "respond with ONLY raw JSON" instruction entirely and wrote a
    prose/markdown report instead (this happens most often on the weaker
    fallback backends -- Gemini Flash free tier or the local Qwen2.5-1.5B --
    which are more prone to dropping strict output-format instructions than
    the primary backend). In that case _parse_analysis_json() has no `{...}`
    to extract at all, returns (None, None, None), and run() gave up straight
    to the raw-text-dump branch with no retry.

    This builds a second-pass prompt that hands the model its own prior
    (unparseable) answer back and asks it to losslessly convert it into the
    required JSON shape -- no new analysis, no re-reading the live data, just
    a reformat. Reusing the model's own content (rather than re-running the
    full analysis prompt) keeps this cheap and avoids getting a second,
    possibly-inconsistent, set of numbers.
    """
    live_data_block = format_live_data_block(live_data)
    return load_prompt(
        "option/reformat",
        raw_analysis=raw_analysis,
        live_data_block=live_data_block,
    )


def reformat_unparseable_analysis(analysis, live_data):
    """
    Second-pass rescue for a totally unparseable first response (see
    build_reformat_prompt() docstring). Returns (horizons, aggregate_pct,
    portfolio_view, reformatted_text) -- reformatted_text is the text that
    should now be treated as "the analysis" going forward (for the raw-text
    fallback branch, if this rescue attempt also fails), or the original
    `analysis` unchanged if the reformat call itself couldn't be attempted.

    This is a lossless reformat of the model's own prior answer -- no new
    analysis, no new facts, no re-reading live data -- so it uses the
    non-live generate_synthesis() tier (plain Groq -> plain Gemini)
    instead of the full live-search cascade, keeping groq/compound,
    Tavily, Gemini-grounding, and Mistral quota free for calls in this
    run that actually depend on live search.
    """
    reformat_prompt = build_reformat_prompt(analysis, live_data)
    try:
        reformatted_text = swing.generate_synthesis(
            reformat_prompt,
            validate_fn=lambda t: _parse_analysis_json(t)[0] is not None,
            log_label="option reformat pass",
        )
    except Exception as e:
        log.warning(f"Reformat pass call failed; keeping original unparseable output. Exception: {e}", exc_info=True)
        return None, None, None, analysis

    if not reformatted_text:
        log.warning("Reformat pass produced no output; keeping original unparseable output.")
        return None, None, None, analysis

    horizons, aggregate_pct, portfolio_view = _parse_analysis_json(reformatted_text)
    if not horizons:
        log.warning("Reformat pass still produced unparseable JSON; falling back to raw text display.")
        return None, None, None, reformatted_text

    log.info("Reformat pass recovered valid JSON from an initially unparseable response.")
    return horizons, aggregate_pct, portfolio_view, reformatted_text


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
    log.info(
        f"Live data feed status: {live_data['status']}"
        + (f" -- {'; '.join(live_data['notes'])}" if live_data["notes"] else "")
    )
    prompt = build_prompt(live_data)

    analysis, sources, used_live_search = swing.generate_analysis(
        prompt, validate_fn=lambda t: _parse_analysis_json(t)[0] is not None
    )
    if not analysis:
        log.error(
            "No LLM backend produced output. Aborting without sending an email."
        )
        sys.exit(1)

    live_feed_ok = live_data.get("status") in ("ok", "eod_fallback", "partial") and (
        live_data.get("spot") or live_data.get("horizons")
    )
    if not used_live_search and not live_feed_ok and os.getenv("REQUIRE_LIVE_DATA", "true").lower() == "true":
        log.error(
            "Neither the direct NSE/Yahoo live data fetch nor the LLM's own live "
            "web search succeeded this run. Aborting."
        )
        sys.exit(1)

    horizons, _model_aggregate_pct, portfolio_view = _parse_analysis_json(analysis)
    if not horizons:
        # The model ignored the "respond with ONLY raw JSON" instruction and
        # returned prose/markdown instead -- try one cheap reformat pass
        # before giving up and emailing an unstructured raw-text dump.
        log.warning("Initial response was not valid JSON; attempting a reformat pass.")
        horizons, _model_aggregate_pct, portfolio_view, analysis = reformat_unparseable_analysis(analysis, live_data)
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

            # BUG FIX: select_best_strikes() used to only implement the three
            # CREDIT/neutral structures (Bear Call Spread, Bull Put Spread,
            # Iron Condor) -- it had no debit-spread optimizer, so a model-
            # proposed Bull Call or Bear Put Spread was previously either
            # skipped outright, or (worse, before that fix) silently
            # remapped to the wrong-direction credit structure by the
            # bias-only fallback branch below. select_best_strikes() now
            # implements Bull Call / Bear Put too (see its "3. Evaluate
            # Debit Verticals" branch), so route them there like the other
            # three instead of bypassing deterministic optimization.
            if "bull call" in strat_lower or ("bull" in bias_lower and "call" in strat_lower):
                target_strat = "Bull Call Spread"
            elif "bear put" in strat_lower or ("bear" in bias_lower and "put" in strat_lower):
                target_strat = "Bear Put Spread"
            elif "bear call" in strat_lower or ("bear" in bias_lower and "call" in strat_lower):
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

            # Item 11 -- regime override: an Iron Condor that can't find ANY
            # structure with its short legs genuinely outside the expected
            # move band means the chain itself isn't range-bound enough this
            # run (the same condition REJECT_IC_SHORT_INSIDE_EM guards
            # against downstream) -- deliberately not given a Watchlist
            # near-miss on band for this reason (see WATCHLIST_EM_BAND_PCT).
            # If live PCR(OI) shows a clear directional skew instead of a
            # neutral reading, retry deterministic selection with the
            # PCR-implied directional credit spread rather than falling
            # through to the model's raw (also likely ungated) IC guess.
            is_regime_override = False
            if (
                not res.get("ok") and not res.get("watchlist")
                and target_strat == "Iron Condor"
                and REGIME_OVERRIDE_ENABLED
            ):
                pcr = snap.get("pcr_oi")
                strong_bearish = isinstance(pcr, (int, float)) and pcr <= (0.7 - REGIME_OVERRIDE_PCR_MARGIN)
                strong_bullish = isinstance(pcr, (int, float)) and pcr >= (1.2 + REGIME_OVERRIDE_PCR_MARGIN)
                if strong_bearish or strong_bullish:
                    override_strat = "Bear Call Spread" if strong_bearish else "Bull Put Spread"
                    override_res = select_best_strikes(snap, spot, bias, override_strat)
                    if override_res.get("ok") or override_res.get("watchlist"):
                        log.info(
                            f"Regime override for {horizon_name}: Iron Condor found no valid "
                            f"structure outside the expected-move band and PCR(OI)={pcr:g} shows a "
                            f"strong directional skew -- retrying with {override_strat} instead."
                        )
                        res = override_res
                        target_strat = override_strat
                        is_regime_override = True

            if res.get("ok") or res.get("watchlist"):
                is_watchlist = bool(res.get("watchlist"))
                best = res["best_trade"] if res.get("ok") else res["near_miss_trade"]
                st = res["strategy_type"]
                # Item 8: near-miss candidates never get a live "Consider"
                # recommendation and are sized down -- see
                # compute_horizon_recommendation() / _sizing_multiplier().
                if is_watchlist:
                    h["_watchlist_tier"] = True
                    h["_watchlist_reason"] = res.get("reason")
                # Make any substitution visible in the report instead of silent:
                # the model's original strategy_name/legs are being replaced by
                # a deterministically-optimized structure, which can be a
                # different structure family than what bias_reason/
                # strike_rationale were written to justify.
                if is_regime_override:
                    substitution_note = (
                        f"Regime override: Iron Condor found no strike combination outside the "
                        f"expected-move band this run; live PCR(OI) showed a strong directional "
                        f"skew, so a {st} was selected deterministically instead."
                        + (" (Watchlist -- see note below.)" if is_watchlist else "")
                    )
                elif is_watchlist:
                    if st == "Bear Call Spread":
                        chosen_desc = f"short {best['short_strike']:g} CE / long {best['long_strike']:g} CE"
                    elif st == "Bull Put Spread":
                        chosen_desc = f"short {best['short_strike']:g} PE / long {best['long_strike']:g} PE"
                    elif st == "Bull Call Spread":
                        chosen_desc = f"long {best['long_strike']:g} CE / short {best['short_strike']:g} CE"
                    elif st == "Bear Put Spread":
                        chosen_desc = f"long {best['long_strike']:g} PE / short {best['short_strike']:g} PE"
                    elif st == "Iron Condor":
                        chosen_desc = (
                            f"short {best['short_put']:g} PE/{best['short_call']:g} CE, "
                            f"long {best['long_put']:g} PE/{best['long_call']:g} CE"
                        )
                    else:
                        chosen_desc = None
                    substitution_note = (
                        f"WATCHLIST: no {st} cleared the strict live R:R/credit-width gates this "
                        f"run; the closest near-miss candidate ({chosen_desc or 'see legs above'}) is "
                        f"shown for reference only, not a live gate-cleared entry -- sized down "
                        f"accordingly in the suggested sizing table below."
                    )
                elif strat_lower and st.lower() not in strat_lower:
                    substitution_note = (
                        f"Deterministic strike selection replaced the model's proposed "
                        f"'{strategy_name}' with an optimized {st} that clears the live "
                        f"R:R/credit-width gates -- verify this still matches your intended thesis."
                    )
                else:
                    # Same strategy family, but the optimizer may still have picked
                    # different strikes than the model's original (possibly
                    # hallucinated) guess. h["legs"] is about to be overwritten below,
                    # so any strike numbers mentioned in the model's existing
                    # strike_rationale text can go stale/wrong relative to the real
                    # trade. Always surface the actual chosen strikes here rather
                    # than only when the structure family itself changed.
                    if st == "Bear Call Spread":
                        chosen_desc = f"short {best['short_strike']:g} CE / long {best['long_strike']:g} CE"
                    elif st == "Bull Put Spread":
                        chosen_desc = f"short {best['short_strike']:g} PE / long {best['long_strike']:g} PE"
                    elif st == "Bull Call Spread":
                        chosen_desc = f"long {best['long_strike']:g} CE / short {best['short_strike']:g} CE"
                    elif st == "Bear Put Spread":
                        chosen_desc = f"long {best['long_strike']:g} PE / short {best['short_strike']:g} PE"
                    elif st == "Iron Condor":
                        chosen_desc = (
                            f"short {best['short_put']:g} PE/{best['short_call']:g} CE, "
                            f"long {best['long_put']:g} PE/{best['long_call']:g} CE"
                        )
                    else:
                        chosen_desc = None
                    substitution_note = (
                        f"Deterministic strike selection chose {chosen_desc} (verified against "
                        f"live/EOD premiums) -- this may differ from strike(s) mentioned above if "
                        f"the model's original text guessed different levels."
                        if chosen_desc else None
                    )
                if substitution_note:
                    # BUG FIX: this used to concatenate directly onto
                    # h["strike_rationale"] right here -- before
                    # apply_verified_payoff/_scrub_stale_strike_claims ever
                    # runs on it. That meant the *first* apply_verified_payoff
                    # call cached this already-corrupted text as
                    # "_raw_strike_rationale" (stale model strikes sitting
                    # next to this note's correct ones), and the scrub saw
                    # some correct numbers in the sentence and let the stale
                    # one survive. Stash it separately instead; apply_verified_
                    # payoff appends it after scrubbing has cleaned the
                    # model's original prose against the real, final legs.
                    h["_substitution_note"] = substitution_note
                h["strategy_name"] = st
                if st == "Bear Call Spread":
                    h["legs"] = f"Sell {best['short_strike']:g} CE, Buy {best['long_strike']:g} CE"
                elif st == "Bull Put Spread":
                    h["legs"] = f"Sell {best['short_strike']:g} PE, Buy {best['long_strike']:g} PE"
                elif st == "Bull Call Spread":
                    h["legs"] = f"Buy {best['long_strike']:g} CE, Sell {best['short_strike']:g} CE"
                elif st == "Bear Put Spread":
                    h["legs"] = f"Buy {best['long_strike']:g} PE, Sell {best['short_strike']:g} PE"
                elif st == "Iron Condor":
                    h["legs"] = f"Buy {best['long_put']:g} PE, Sell {best['short_put']:g} PE, Sell {best['short_call']:g} CE, Buy {best['long_call']:g} CE"
            else:
                log.info(f"Deterministic strike selection for {horizon_name} ({target_strat}): {res.get('reason')}")

        horizons, aggregate_pct, over_cap = finalize_horizons(horizons, live_data, sources)
        horizons = repair_rejected_legs(horizons, live_data, sources)
        aggregate_pct, over_cap = _aggregate_risk_from_verified(horizons)
        if over_cap:
            log.warning(
                f"Computed worst-case combined max loss ({aggregate_pct}%) exceeds the "
                f"{AGGREGATE_CAP_PCT:.0f}% cap -- flagging in the report."
            )
        horizons_html = render_horizons_html(horizons, aggregate_pct, portfolio_view, live_data)
    else:
        log.error("Could not parse JSON from LLM output; falling back to raw text display.")
        sans = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
        horizons_html = (
            f'<div style="font-family:{sans};font-size:12px;color:#8B2E2E;margin-bottom:8px;">'
            f"Note: the model's response could not be parsed as structured data; showing raw output below.</div>"
            f'<pre style="white-space:pre-wrap;font-family:{sans};font-size:12px;color:#14213D;">{html.escape(swing._strip_code_fences(analysis))}</pre>'
        )

    email_html = build_email_html(horizons_html, today_str, sources, used_live_search, session_label, live_data)

    if os.getenv("DRY_RUN", "false").lower() == "true":
        with open("option_strategy_report.html", "w", encoding="utf-8") as f:
            f.write(email_html)
        log.info("DRY_RUN enabled -- wrote option_strategy_report.html instead of emailing.")
        return

    if not send_option_strategy_email(email_html):
        log.critical("Email delivery failed")


if __name__ == "__main__":
    run()