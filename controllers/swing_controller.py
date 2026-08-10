"""
swing_trade_advisor.py

Standalone companion to main.py. Runs a single, open-ended "find me a
high-conviction 3-5 month swing trade" prompt against whichever free LLM
backend is available, then emails the result to the same recipients
configured for the stock report (EMAIL_TO / EMAIL_CC in config.py / the
workflow yaml's env vars).

LLM init, model tiers, retry/backoff, and the live-search fallback chain
all live in llm_backend.py now (shared with main.py's AI Stocks Story and,
via this module's generate_analysis(), optionstrategy.py) -- see that
file's docstring for the full chain order and rationale. This module only
owns what's specific to the swing-trade prompt: which Tavily queries to
run for the grounded-context tier (_gather_tavily_context), the stock
qualification/verification logic below, and result formatting/email.

CAVEAT: this is still not a verified real-time trade signal. Web search
results can be a few hours stale, incomplete, or misread by the model.
Treat every price level, %, and "recent" news item as a starting hypothesis
to verify against a live quote/news source yourself -- not investment advice.
"""

import os
import re
import sys
import csv
import json
import html
import time
import requests
import threading
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

import smtplib
from email.mime.text import MIMEText
import pandas as pd
from utils import email_service

from utils import config
from utils.logger import log
from utils.prompt_loader import load_prompt
from services.stock_fetcher import fetch_stock_data
from models.market_context import classify_market
from utils.yf_throttle import get_shared_session, call_with_retries
from llm import llm_backend  # shared LLM init + fallback chain (see llm_backend.py)
from utils.compliance import build_compliance_block_html

from models import swing_trade_risk as risk
from models import swing_trade_regime as regime
from models import swing_trade_scoring as scoring
from models import swing_trade_outcomes as outcomes
from models import swing_trade_universe as universe  # static ticker seed list for the deterministic screen (see below)

# -----------------------------
# Qualifying-stock gate
# -----------------------------
# The model's own JSON output is not trusted at face value: _verify_stock_claims
# independently checks every mandatory filter from the prompt (uptrend, RSI/MACD,
# growth thresholds, risk:reward minimum, debt/ROE) against real data. A stock
# with a "hard" contradiction -- i.e. one where the independent check actively
# disagrees with the model's claim, not just "couldn't be verified" -- on
# debt-to-equity, ROE, or a data-integrity check (missing price, hallucinated
# ticker) always fails its own strategy's entry criteria. A hard contradiction
# on one of the five CORE setup filters (uptrend/RSI/MACD/growth/risk-reward)
# is more forgiving: MIN_CORE_FILTERS_REQUIRED lets up to two fail by default,
# and a high enough composite score can override even that (see
# _split_qualifying). REQUIRE_QUALIFYING_STOCK (default true) enforces the
# overall gate; set to "false" to restore the old behavior of reporting every
# candidate regardless of contradictions.
REQUIRE_QUALIFYING_STOCK = os.getenv("REQUIRE_QUALIFYING_STOCK", "true").lower() == "true"
# How many times to re-prompt the model (with the specific rejection reasons fed
# back in) before giving up and reporting "no qualifying trade found" instead of
# emailing a pick that fails its own criteria. Each attempt now runs a full
# two-stage pipeline (fundamentals screen, then technicals -- see below), so
# this is 2 LLM calls per attempt, not 1. 3 is a reasonable default for a
# WEEKLY run (see SCHEDULE note below) -- it wouldn't be for a daily one.
# _env_int lives in llm_backend.py now (shared with main.py's config too).
_env_int = llm_backend._env_int

MAX_GENERATION_ATTEMPTS = _env_int("MAX_GENERATION_ATTEMPTS", 3)


def _env_float(name, default):
    """Same idea as _env_int, but for a float threshold (risk:reward, RSI,
    debt-to-equity %, ROE %, growth %) -- falls back to `default` (with a
    warning) on anything unset/empty/unparseable.

    Moved up here (was previously defined further down, after
    REGIME_SOFTEN_MAX_PCT's module-level call to it) -- that ordering was a
    latent NameError waiting to happen: Python executes top-level module
    code in order, so a function used at import time must be DEFINED above
    its first call, not just present somewhere later in the file. It never
    surfaced locally because whichever module imports swing_trade_advisor
    first usually does so after some other import already pulled in a
    same-named helper into globals() by coincidence in dev, but a clean
    process (e.g. optionstrategy.py as the actual first importer in CI)
    hits it every time."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw.strip())
    except ValueError:
        log.warning(f"env var {name}='{raw}' is not a valid number -- using default {default}.")
        return default


def _fmt_num(x):
    """Formats a threshold for display in prompts/messages without a
    trailing '.0' when it's a whole number (e.g. 20.0 -> '20', 17.5 -> '17.5').
    Moved up alongside _env_float for the same reason -- see that
    docstring."""
    return f"{x:g}"


REQUIRE_PROFESSIONAL_QUALITY_GATE = os.getenv("REQUIRE_PROFESSIONAL_QUALITY_GATE", "true").lower() == "true"
MIN_COMPOSITE_SCORE = _env_float("MIN_COMPOSITE_SCORE", 60.0)
MAX_POSITION_SIZE_PCT = _env_float("MAX_POSITION_SIZE_PCT", 5.0)


# -----------------------------
# Deterministic fundamentals screen (replaces LLM-driven Stage-1 discovery)
# -----------------------------
# Previously, Stage 1 asked an LLM to "search" for 8-12 candidate names per
# sector meeting the growth bar, and only THEN independently re-checked
# whatever it found (_prefilter_by_fundamentals). That made candidate
# DISCOVERY -- unlike every other stage of this pipeline -- entirely
# dependent on the model's web search actually surfacing the right small/
# mid-cap names, which is slow, token-expensive, and produces a lot of
# "rejected: nodata" candidates when the model guesses/misremembers a
# ticker. _fetch_fundamentals() already fetches real data deterministically
# (used today only to VERIFY the model's claims) -- when true, this flag
# uses it to screen swing_trade_universe.py's static ticker list directly,
# so Stage 1 becomes a real, complete, zero-LLM-token scan instead of a
# sample-and-hope search. Stage 2 (sentiment/catalyst/trade-plan, which
# genuinely needs live web search) is unchanged either way.
# Set to "false" to restore the old LLM-search Stage 1 (e.g. if you don't
# trust/haven't audited swing_trade_universe.py's ticker list yet, or want
# to compare the two approaches side by side).
USE_DETERMINISTIC_SCREEN = os.getenv("USE_DETERMINISTIC_SCREEN", "true").lower() == "true"

# -----------------------------
# Universe widening (optional supplement to swing_trade_universe.py)
# -----------------------------
# swing_trade_universe.py's static seed list is what actually determines
# how many/which candidates get checked -- widening it for real (adding
# mid-cap names or a broader index) means editing that file directly.
# EXTRA_UNIVERSE_FILE is a lower-friction supplement that doesn't require
# touching swing_trade_universe.py at all: point it at a CSV
# (columns: name,ticker,sector,bucket) and those rows get merged into the
# deterministic screen's candidate pool alongside the seed list, for
# whichever sectors are in play on a given attempt. Unset by default (no
# behavior change). See _load_extra_universe_tickers.
EXTRA_UNIVERSE_FILE = os.getenv("EXTRA_UNIVERSE_FILE", "").strip()

# -----------------------------
# Regime-aware fundamentals softening -- bounded, transparent, this-run-only
# -----------------------------
# A fixed MIN_GROWTH_YOY_PCT bar is calibrated for an average market. In a
# genuinely weak/choppy tape (see swing_trade_regime.py's classification)
# the honest answer some weeks really is "nothing qualifies" -- but a bar
# that NEVER flexes also means the strategy simply stops producing any
# signal at all during the exact stretches (broad-based small/mid-cap
# weakness) this run is likeliest to hit. When enabled, this softens
# MIN_GROWTH_YOY_PCT by a small, capped percentage for THIS RUN ONLY when
# regime.check_market_regime()'s classification looks weak/mixed -- never
# more than REGIME_SOFTEN_MAX_PCT below the configured value, and always
# disclosed in the email (see _regime_softening_html) so it's never a
# silent bar-lowering. This is independent of, and separate from,
# AUTO_ADJUST_THRESHOLDS (which is history-driven and persists across
# runs) -- this one is regime-driven and resets every run.
REGIME_SOFTEN_GROWTH_BAR = os.getenv("REGIME_SOFTEN_GROWTH_BAR", "true").lower() == "true"
REGIME_SOFTEN_MAX_PCT = _env_float("REGIME_SOFTEN_MAX_PCT", 15.0)

# -----------------------------
# Regime-gate override -- relative-strength exception (explicit, opt-in)
# -----------------------------
# When regime.REQUIRE_MARKET_REGIME_FILTER is on (the default) and the
# broad-market check fails, run() has always skipped the entire scan --
# see the "Market-regime gate" block in run(). That's a deliberate,
# explicit block, not the thing this section is about.
#
# What THIS controls is a separate, opt-in exception: when enabled, a
# failed regime gate no longer skips the run outright -- instead, every
# candidate this run must additionally clear a stricter, substitute
# per-stock bar (_verify_relative_strength_override) in place of the
# market-wide check that's being skipped for it. That override check is
# an absolute blocker (like debt/equity or ROE), not one of the five
# flexible core filters -- it can't be waived by MIN_CORE_FILTERS_REQUIRED
# slack or a high composite score, since it exists specifically to
# substitute for the disabled market gate rather than to be one more
# negotiable setup filter. Off by default: a failed regime gate still
# skips the run unless you explicitly opt into this.
REGIME_OVERRIDE_ON_RS = os.getenv("REGIME_OVERRIDE_ON_RS", "false").lower() == "true"
REGIME_OVERRIDE_MIN_RSI = _env_float("REGIME_OVERRIDE_MIN_RSI", 55.0)
REGIME_OVERRIDE_SMA50D_MARGIN_PCT = _env_float("REGIME_OVERRIDE_SMA50D_MARGIN_PCT", 3.0)

# -----------------------------
# Cross-run near-miss watchlist
# -----------------------------
# A stock that fully clears fundamentals but fails ONLY on a technical
# filter (e.g. RSI still falling, no bullish MACD signal yet) is exactly the
# kind of candidate worth re-checking next week rather than re-discovering
# from scratch -- growth doesn't change week to week, but a technical setup
# can complete within days. WATCHLIST_LOG persists these across runs;
# _load_and_recheck_watchlist() re-verifies each entry's CURRENT technicals
# (and re-confirms fundamentals still hold, since a quarter can roll over)
# at the start of every run, ahead of the normal sector-rotation scan.
WATCHLIST_LOG = os.getenv("WATCHLIST_LOG", "swing_trade_watchlist.csv")
WATCHLIST_MAX_AGE_DAYS = _env_int("WATCHLIST_MAX_AGE_DAYS", 42)  # ~6 weeks, then drop stale entries


# -----------------------------
# Configurable mandatory-filter thresholds
# -----------------------------
# These were previously hardcoded in TWO places that had to be edited
# together to actually change the bar: the prompt text sent to the model,
# and the independent verifier that re-checks the model's claims against
# real data (_verify_fundamentals / _verify_risk_reward / _verify_technicals).
# They're env vars now so the bar can be tuned per-run (e.g. in the
# workflow yaml) without touching code -- and every prompt/message below
# now always describes the actual enforced value instead of a hardcoded
# number that could silently drift out of sync with the verifier (as the
# risk:reward text and check previously had -- the prompt said "1:2.5" but
# the code enforced 2.0). MIN_RISK_REWARD's default below was lowered from
# that previously-enforced 2.0 to 1.5 -- a deliberate loosening, not drift --
# and MIN_GROWTH_YOY_PCT's default was similarly lowered from 20.0 to 15.0
# (see _verify_fundamentals for the accompanying "either revenue or profit
# growth, not both" change).
MIN_GROWTH_YOY_PCT = _env_float("MIN_GROWTH_YOY_PCT", 15.0)

# -----------------------------
# Growth lookback window
# -----------------------------
# A strict single-quarter YoY comparison penalizes one lumpy/weak quarter
# even when the underlying trend is fine -- e.g. a one-off delayed order or
# a tough prior-year comp can fail an otherwise-strong company on a single
# data point. "trailing_2q" sums the latest 2 quarters and compares against
# the 2 quarters bracketing the point 1 year ago, so one bad quarter is
# averaged against its neighbor rather than deciding the outcome alone.
# "ttm" compares trailing-twelve-months against the prior TTM (8 quarters
# of history needed) -- the smoothest option, but yfinance's
# quarterly_financials frequently only exposes ~4-5 quarters, so this mode
# commonly falls back to "yoy" per-ticker (logged when it happens) rather
# than silently skipping growth verification for that name. See
# _growth_lookback_periods.
GROWTH_LOOKBACK_MODE = os.getenv("GROWTH_LOOKBACK_MODE", "yoy").strip().lower()
if GROWTH_LOOKBACK_MODE not in ("yoy", "trailing_2q", "ttm"):
    log.warning(f"GROWTH_LOOKBACK_MODE='{GROWTH_LOOKBACK_MODE}' not recognized -- falling back to 'yoy'.")
    GROWTH_LOOKBACK_MODE = "yoy"

MIN_RISK_REWARD = _env_float("MIN_RISK_REWARD", 1.5)
MAX_RSI_OVERBOUGHT = _env_float("MAX_RSI_OVERBOUGHT", 70.0)
# RSI filter loosened from "trending up AND below MAX_RSI_OVERBOUGHT" to a
# plain band check: anywhere in [MIN_RSI_OVERSOLD, MAX_RSI_OVERBOUGHT)
# qualifies regardless of whether RSI ticked up or down since the prior
# session -- day-to-day direction is noisy and was rejecting otherwise-fine
# setups on a single down-week.
MIN_RSI_OVERSOLD = _env_float("MIN_RSI_OVERSOLD", 40.0)
MAX_DEBT_TO_EQUITY_PCT = _env_float("MAX_DEBT_TO_EQUITY_PCT", 100.0)
MIN_ROE_PCT = _env_float("MIN_ROE_PCT", 10.0)
# When false, the uptrend requirement is dropped from both the prompt and
# the verifier entirely (RSI/MACD/growth/risk-reward filters still apply)
# -- use this if you want to consider pullback/basing setups, not just
# confirmed uptrends. When true, the requirement itself is now just "price
# above its 50-day MA" -- loosened from the previous stricter "price above
# BOTH the 20-week and 50-week SMA" structure (see _verify_technicals).
REQUIRE_UPTREND_FILTER = os.getenv("REQUIRE_UPTREND_FILTER", "true").lower() == "true"

# -----------------------------
# "N of 5 core filters must pass" -- and a high-composite-score override
# -----------------------------
# Previously a candidate qualified only if it had ZERO hard contradictions
# across every mandatory filter. The five CORE swing-setup filters this
# strategy's own prompt describes as "the setup" are: uptrend (50-day MA),
# RSI band, MACD signal, growth threshold, and risk:reward minimum.
# MIN_CORE_FILTERS_REQUIRED lets a candidate qualify even if up to
# (CORE_FILTER_COUNT - MIN_CORE_FILTERS_REQUIRED) of those five hard-fail --
# lowered from 4 (any 4 of 5) to 3 (any 3 of 5), i.e. up to two of the five
# core filters can now hard-fail and a candidate still qualifies strict.
# This flexibility does NOT extend to debt-to-equity, ROE, or data-
# integrity checks (missing price, hallucinated ticker, etc.) -- those
# remain absolute blockers no matter how many core filters pass, since
# they're basic quality/fraud checks rather than part of the setup itself.
#
# Separately, COMPOSITE_SCORE_CORE_FILTER_OVERRIDE gives a second path to
# qualify: a candidate whose 0-100 composite score clears this (lowered
# from 75 to 70) bar can be reported even if it doesn't clear
# MIN_CORE_FILTERS_REQUIRED -- the idea being a sufficiently strong overall
# setup can outweigh a weak filter or two. This override still respects
# the non-core absolute blockers above, and still goes through the
# existing MIN_COMPOSITE_SCORE floor in _passes_professional_quality_gate.
# See _split_qualifying.
CORE_FILTER_COUNT = 5
MIN_CORE_FILTERS_REQUIRED = _env_int("MIN_CORE_FILTERS_REQUIRED", 3)
COMPOSITE_SCORE_CORE_FILTER_OVERRIDE = _env_float("COMPOSITE_SCORE_CORE_FILTER_OVERRIDE", 70.0)

# -----------------------------
# Watchlist tier (this run's email, not to be confused with the cross-run
# near-miss CSV below)
# -----------------------------
# Previously, a candidate that didn't clear the strict bar above just fell
# into "rejected" -- on a genuinely weak day that can mean the whole run
# reports "no qualifying trade" even though one or two names came close. A
# candidate that clears every ABSOLUTE blocker (debt/equity, ROE, data
# integrity -- never negotiable) but is one core filter short of the
# strict bar is exactly the "fails one filter but close" case worth
# surfacing on its own, clearly-labeled tier rather than either promoting
# it into strict (which would quietly lower the bar) or burying it in the
# rejected list (which drops it from the email's headline output
# entirely). WATCHLIST_EXTRA_CORE_FILTER_SLACK widens the core-filter bar
# by this many additional failures beyond what strict allows -- default 1
# means "one filter more than strict tolerates" lands in the watchlist
# tier instead of being flatly rejected. Set WATCHLIST_TIER_ENABLED=false
# to disable and restore the old "strict or rejected" behavior. See
# _split_qualifying.
WATCHLIST_TIER_ENABLED = os.getenv("WATCHLIST_TIER_ENABLED", "true").lower() == "true"
WATCHLIST_EXTRA_CORE_FILTER_SLACK = _env_int("WATCHLIST_EXTRA_CORE_FILTER_SLACK", 1)

# -----------------------------
# Confidence-adjusted position sizing for the watchlist tier
# -----------------------------
# Alternative to relaxing entry filters for close-but-imperfect setups:
# _split_qualifying's filters above are untouched -- a watchlist candidate
# still has to clear every absolute blocker and land within
# WATCHLIST_EXTRA_CORE_FILTER_SLACK of the strict bar, exactly as before.
# What changes is that a watchlist pick is no longer sized the same as a
# strict one -- _apply_confidence_sizing scales BOTH the model's flat
# allocation_pct AND the independently-computed ATR-based share count/
# position value (see _attach_risk_plan) by this multiplier, once, right
# after _split_qualifying. This is exposure-management, not a lowered bar:
# a watchlist stock can still never end up sized larger than a strict one,
# since the multiplier applies on top of whatever already survived
# MAX_POSITION_SIZE_PCT's cap. Set to 1.0 to disable (watchlist sized the
# same as strict). See _apply_confidence_sizing.
WATCHLIST_POSITION_SIZE_MULTIPLIER = _env_float("WATCHLIST_POSITION_SIZE_MULTIPLIER", 0.5)

# -----------------------------
# Target-distance-based time horizon (replaces one fixed 3-5 month window
# for every candidate)
# -----------------------------
# Every candidate used to be pushed into the same "3-5 months" holding
# period in the prompt, regardless of how far its own target actually was.
# _compute_flexible_horizon derives a per-stock window instead, from real
# data already on hand: the ATR-based weekly volatility computed in
# _attach_risk_plan (this stock's own typical weekly % move) says roughly
# how many weeks it should take THIS stock, at ITS OWN pace, to travel the
# distance to target1_pct. A modest, close target on a fast mover can
# genuinely resolve in under 3 months; a distant, aggressive target on a
# slow mover legitimately needs more than 5 -- that's a timing fact, not a
# reason to loosen the entry bar (the same principle
# WATCHLIST_POSITION_SIZE_MULTIPLIER applies to size instead of horizon).
# Set HORIZON_STRETCH_ENABLED=false to fall back to showing only the
# model's own exit_date guess with no computed window.
HORIZON_STRETCH_ENABLED = os.getenv("HORIZON_STRETCH_ENABLED", "true").lower() == "true"
# Multiplier on top of the raw straight-line ATR-pace estimate to get the
# window's upper bound -- price doesn't move in a straight line every
# week, so the realistic worst case allows more calendar time than the
# straight-line number alone.
HORIZON_BUFFER_MULTIPLIER = _env_float("HORIZON_BUFFER_MULTIPLIER", 1.6)
# Absolute floor/ceiling so a near-zero ATR or an extreme target can't
# produce a nonsensical multi-day or multi-year "swing trade" horizon --
# a sanity clamp, not a reintroduction of the old fixed window.
HORIZON_FLOOR_MONTHS = _env_float("HORIZON_FLOOR_MONTHS", 1.5)
HORIZON_CEILING_MONTHS = _env_float("HORIZON_CEILING_MONTHS", 9.0)

# -----------------------------
# Multi-session confirmation for weekly RSI/MACD (reduces noise-driven
# rejections without changing the actual bar)
# -----------------------------
# _fetch_weekly_technicals_uncached used to read weekly RSI and the MACD-
# vs-signal comparison off a single latest session -- a stock could get
# hard-rejected purely because ITS ONE MOST RECENT weekly close happened to
# be a noisy outlier, even though it was solidly inside the RSI band or
# above signal on every nearby session. This averages both readings over
# the last MULTI_SESSION_CONFIRMATION_WINDOW weekly sessions instead of
# taking a single snapshot. MIN_RSI_OVERSOLD/MAX_RSI_OVERBOUGHT (the actual
# bar) are untouched -- this only smooths what gets compared against them.
MULTI_SESSION_CONFIRMATION_WINDOW = _env_int("MULTI_SESSION_CONFIRMATION_WINDOW", 3)

# -----------------------------
# Rejection-history logging (for deciding whether a threshold is genuinely
# too tight, versus one run's coincidence)
# -----------------------------
# Every "hard" rejection note already contains the real recomputed number in
# its text (e.g. "Revenue growth YoY is 12.5%"). Rather than re-deriving that
# number from scratch, these patterns just pull it back out of the note text
# so it can be logged alongside the threshold that was active for that run.
# This is a FUNCTION rather than a fixed list because _apply_auto_adjustments
# (below) can change MIN_GROWTH_YOY_PCT etc. mid-run -- callers must always
# get the CURRENT threshold value, not whatever was in effect at import time.
# Each tuple: (metric_name, regex-with-one-capture-group, current_threshold, "min"|"max")
# "min" means the candidate needed to be >= threshold (growth, ROE, risk:reward);
# "max" means it needed to be <= threshold (debt-to-equity, RSI overbought ceiling).
def _metric_patterns():
    return [
        ("revenue_growth_yoy_pct", re.compile(r"Revenue growth YoY is (-?[\d.]+)%"), MIN_GROWTH_YOY_PCT, "min"),
        ("profit_growth_yoy_pct", re.compile(r"Net profit growth YoY is (-?[\d.]+)%"), MIN_GROWTH_YOY_PCT, "min"),
        ("debt_to_equity_pct", re.compile(r"Debt-to-equity is (-?[\d.]+)%"), MAX_DEBT_TO_EQUITY_PCT, "max"),
        ("roe_pct", re.compile(r"ROE is (-?[\d.]+)%"), MIN_ROE_PCT, "min"),
        ("risk_reward_ratio", re.compile(r"Risk:reward of 1 : (-?[\d.]+) is below"), MIN_RISK_REWARD, "min"),
        ("weekly_rsi_overbought", re.compile(r"Weekly RSI is (-?[\d.]+) \(>="), MAX_RSI_OVERBOUGHT, "max"),
        ("weekly_rsi_oversold", re.compile(r"Weekly RSI is (-?[\d.]+) \(<"), MIN_RSI_OVERSOLD, "min"),
    ]


REJECTION_HISTORY_LOG = os.getenv("REJECTION_HISTORY_LOG", "swing_trade_rejection_history.csv")


def _log_rejection_history(rejected, today_str, log_path=REJECTION_HISTORY_LOG):
    """
    Appends one row per (ticker, matched threshold-miss) to a local CSV, so
    near-miss numbers accumulate across runs instead of only being visible in
    a single run's email. Only logs threshold-comparison misses (the ones in
    _metric_patterns()) -- not every "hard" note, since some hard notes (e.g.
    a model/recomputed risk:reward MISMATCH, or "price below its SMA") aren't
    a "how close to the threshold was this" comparison and would just add
    noise to a threshold-tuning analysis.

    Best-effort and silent-on-failure by design: a broken log write should
    never abort or change the outcome of an actual run.
    """
    try:
        rows = []
        for s in rejected:
            ticker = (s.get("ticker") or "").strip()
            name = s.get("name") or ticker or "?"
            for note_text, sev in (s.get("_verification_notes") or []):
                if sev != "hard":
                    continue
                for metric, pattern, threshold, direction in _metric_patterns():
                    m = pattern.search(note_text)
                    if not m:
                        continue
                    try:
                        actual = float(m.group(1))
                    except ValueError:
                        continue
                    margin = (threshold - actual) if direction == "min" else (actual - threshold)
                    rows.append({
                        "date": today_str,
                        "ticker": ticker,
                        "name": name,
                        "metric": metric,
                        "threshold": threshold,
                        "actual_value": actual,
                        "margin_missed_by": round(margin, 2),
                    })
        if not rows:
            return
        path = Path(log_path)
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["date", "ticker", "name", "metric", "threshold", "actual_value", "margin_missed_by"]
            )
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
    except OSError as e:
        log.warning(f"Could not write rejection history log: {e}")


# -----------------------------
# Automatic threshold self-tuning -- OPT-IN, bounded, and fully logged
# -----------------------------
# Off by default. When enabled, each run reads the accumulated rejection-
# history log and, ONLY where several DIFFERENT runs and DIFFERENT tickers
# show a recurring near-miss on the same threshold, nudges that threshold a
# small bounded step -- never past MAX_THRESHOLD_DRIFT_PCT away from the
# value you actually configured, and never more than ADJUST_STEP_PCT of that
# original value in a single run. It never tightens a threshold and never
# acts on a single run's data (that's "coincidence, not pattern" -- same rule
# analyze_rejection_history.py applies). Every change it makes is written to
# THRESHOLD_ADJUSTMENT_LOG and disclosed in the email itself -- this is meant
# to replace you manually re-reading the history and editing the workflow
# yaml, not to quietly drift the strategy's own bar with no visible trail.
AUTO_ADJUST_THRESHOLDS = os.getenv("AUTO_ADJUST_THRESHOLDS", "false").lower() == "true"
MIN_RUNS_BEFORE_ADJUST = _env_int("MIN_RUNS_BEFORE_ADJUST", 4)
MAX_THRESHOLD_DRIFT_PCT = _env_float("MAX_THRESHOLD_DRIFT_PCT", 25.0)
ADJUST_STEP_PCT = _env_float("ADJUST_STEP_PCT", 5.0)
THRESHOLD_ADJUSTMENT_LOG = os.getenv("THRESHOLD_ADJUSTMENT_LOG", "swing_trade_threshold_adjustments.csv")

# SAFETY GATE (review item 8): an auto-loosening mechanism can quietly
# degrade signal quality specifically in a regime where quality setups are
# genuinely rare -- the worst possible time to lower the bar. Rejection
# history alone (which is all AUTO_ADJUST_THRESHOLDS looks at) cannot tell
# the difference between "this threshold is miscalibrated" and "the market
# just isn't offering this setup right now"; only a walk-forward backtest
# (see swing_trade_backtest.py) against several years of real price data
# can distinguish the two. So AUTO_ADJUST_THRESHOLDS is forced back off at
# runtime -- even if the env var is set -- unless the operator has also
# set CONFIRM_AUTO_ADJUST_BACKTESTED=true, an explicit acknowledgment that
# they've actually run the backtest harness and reviewed its numbers
# before trusting this feature. This is a deliberate speed bump, not a
# hard technical dependency (there's no way to programmatically verify a
# human actually looked at backtest output) -- but it makes "I forgot this
# was dangerous" require a second, separate, explicit opt-in rather than
# one boolean flag someone set months ago and forgot about.
_CONFIRM_AUTO_ADJUST_BACKTESTED = os.getenv("CONFIRM_AUTO_ADJUST_BACKTESTED", "false").lower() == "true"
if AUTO_ADJUST_THRESHOLDS and not _CONFIRM_AUTO_ADJUST_BACKTESTED:
    log.warning(
        "AUTO_ADJUST_THRESHOLDS=true but CONFIRM_AUTO_ADJUST_BACKTESTED is "
        "not set -- forcing auto-adjustment OFF for this run. Rejection-history "
        "near-misses alone cannot distinguish 'this threshold is miscalibrated' from "
        "'quality setups are genuinely rare right now' -- only a walk-forward "
        "backtest (see swing_trade_backtest.py) can. Run the backtest, review its "
        "win-rate/drawdown numbers, and set CONFIRM_AUTO_ADJUST_BACKTESTED=true "
        "once you actually trust this feature."
    )
    AUTO_ADJUST_THRESHOLDS = False

# Captured once, before any auto-adjustment ever runs, so drift is always
# measured from the value YOU set in the workflow yaml -- not from an
# already-adjusted value from a previous run (which would let drift compound
# past MAX_THRESHOLD_DRIFT_PCT over many runs instead of being capped by it).
_ORIGINAL_THRESHOLDS = {
    "MIN_GROWTH_YOY_PCT": MIN_GROWTH_YOY_PCT,
    "MIN_RISK_REWARD": MIN_RISK_REWARD,
    "MAX_RSI_OVERBOUGHT": MAX_RSI_OVERBOUGHT,
    "MIN_RSI_OVERSOLD": MIN_RSI_OVERSOLD,
    "MAX_DEBT_TO_EQUITY_PCT": MAX_DEBT_TO_EQUITY_PCT,
    "MIN_ROE_PCT": MIN_ROE_PCT,
}
_GLOBAL_DIRECTION = {
    "MIN_GROWTH_YOY_PCT": "min",
    "MIN_RISK_REWARD": "min",
    "MAX_RSI_OVERBOUGHT": "max",
    "MIN_RSI_OVERSOLD": "min",
    "MAX_DEBT_TO_EQUITY_PCT": "max",
    "MIN_ROE_PCT": "min",
}
_METRIC_TO_GLOBAL = {
    "revenue_growth_yoy_pct": "MIN_GROWTH_YOY_PCT",
    "profit_growth_yoy_pct": "MIN_GROWTH_YOY_PCT",
    "debt_to_equity_pct": "MAX_DEBT_TO_EQUITY_PCT",
    "roe_pct": "MIN_ROE_PCT",
    "risk_reward_ratio": "MIN_RISK_REWARD",
    "weekly_rsi_overbought": "MAX_RSI_OVERBOUGHT",
    "weekly_rsi_oversold": "MIN_RSI_OVERSOLD",
}


def _load_history_rows(log_path=REJECTION_HISTORY_LOG):
    path = Path(log_path)
    if not path.exists():
        return []
    rows = []
    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    r["threshold"] = float(r["threshold"])
                    r["actual_value"] = float(r["actual_value"])
                    r["margin_missed_by"] = float(r["margin_missed_by"])
                except (KeyError, ValueError, TypeError) as e:
                    log.debug(f"Skipping malformed row: {e}")
                    continue
                rows.append(r)
    except OSError as e:
        log.warning(f"Could not read rejection history log '{log_path}': {e}")
    return rows


def _log_threshold_adjustments(applied, log_path=THRESHOLD_ADJUSTMENT_LOG):
    """Audit trail of every auto-adjustment ever applied, independent of the
    (transient) email/console log for that one run -- so the full history of
    what the strategy's own bar used to be is always reconstructable."""
    if not applied:
        return
    today_str = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d")
    try:
        path = Path(log_path)
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["date", "constant", "old_value", "new_value", "reason"])
            if write_header:
                writer.writeheader()
            for gname, old, new, reason in applied:
                writer.writerow({"date": today_str, "constant": gname, "old_value": old, "new_value": new, "reason": reason})
    except OSError as e:
        log.warning(f"Could not write threshold adjustment log: {e}")


def _apply_auto_adjustments(log_path=REJECTION_HISTORY_LOG):
    """
    Mutates the module-level threshold globals directly (so every downstream
    prompt, verifier, and email-text reference already picks up the new
    value with no other code changes needed) and returns a list of
    (constant_name, old_value, new_value, reason) tuples for anything it
    changed -- empty if AUTO_ADJUST_THRESHOLDS is off, there's no history
    yet, or nothing showed a strong enough pattern this run.
    """
    if not AUTO_ADJUST_THRESHOLDS:
        return []

    rows = _load_history_rows(log_path)
    if not rows:
        return []

    by_global = defaultdict(list)
    for r in rows:
        gname = _METRIC_TO_GLOBAL.get(r.get("metric"))
        if gname:
            by_global[gname].append(r)

    applied = []
    for gname, entries in by_global.items():
        direction = _GLOBAL_DIRECTION[gname]
        original = _ORIGINAL_THRESHOLDS[gname]
        current = globals()[gname]

        run_dates = set(e["date"] for e in entries)
        if len(run_dates) < MIN_RUNS_BEFORE_ADJUST:
            continue  # not enough independent runs yet -- could easily be one bad stretch

        near_misses = [e for e in entries if 0 < e["margin_missed_by"] <= abs(original) * 0.25]
        near_miss_dates = set(e["date"] for e in near_misses)
        # Require the near-misses to be spread across at least half the runs
        # that hit this threshold at all -- not just clustered in one run --
        # so one unusually-close week can't look like a durable pattern.
        if not near_misses or len(near_miss_dates) < max(2, len(run_dates) // 2):
            continue

        near_miss_values = sorted(e["actual_value"] for e in near_misses)
        median_near_miss = near_miss_values[len(near_miss_values) // 2]

        cap = abs(original) * (MAX_THRESHOLD_DRIFT_PCT / 100.0)
        step = abs(original) * (ADJUST_STEP_PCT / 100.0)

        if direction == "min":
            target = max(median_near_miss, original - cap)   # never loosen past the drift cap
            target = min(target, original)                    # never "loosen" upward past the original by mistake
            new_value = current if target >= current else max(current - step, target)
        else:
            target = min(median_near_miss, original + cap)
            target = max(target, original)
            new_value = current if target <= current else min(current + step, target)

        new_value = round(new_value, 2)
        if new_value != current:
            reason = (
                f"{len(near_misses)} near-miss(es) across {len(near_miss_dates)} of "
                f"{len(run_dates)} run(s) that hit this threshold (median near-miss "
                f"value: {median_near_miss})"
            )
            applied.append((gname, current, new_value, reason))
            globals()[gname] = new_value

    if applied:
        _log_threshold_adjustments(applied)
    return applied


def _adjustments_html(applied):
    """Small notice box for the email disclosing exactly what was auto-adjusted
    this run and why -- auto-tuning with no visible trail is the same failure
    mode as a human quietly loosening the bar until something passes."""
    if not applied:
        return ""
    sans = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
    rows = "".join(
        f'<div style="margin-top:6px;">'
        f'<strong>{html.escape(gname)}</strong>: {old} &rarr; {new} '
        f'<span style="color:#8A8F9C;">({html.escape(reason)})</span></div>'
        for gname, old, new, reason in applied
    )
    return (
        f'<div style="font-family:{sans};font-size:12px;color:#5C4A1E;line-height:1.6;'
        f'padding:12px 16px;background:#FBF3DC;border-radius:4px;border:1px solid #E9DCB0;'
        f'margin-bottom:14px;">'
        f'<strong>Thresholds auto-adjusted this run</strong> (recurring near-misses in the '
        f'rejection history -- see {html.escape(THRESHOLD_ADJUSTMENT_LOG)} for the full trail):'
        f"{rows}"
        "</div>"
    )


def _regime_soften_growth_bar(regime_detail):
    """
    Bounded, transparent, THIS-RUN-ONLY softening of MIN_GROWTH_YOY_PCT when
    the broad market looks weak/mixed -- see REGIME_SOFTEN_GROWTH_BAR's
    docstring above for the rationale. Mutates the module-level
    MIN_GROWTH_YOY_PCT global (same pattern _apply_auto_adjustments already
    uses) so every downstream prompt/verifier picks up the softened value
    automatically. Returns (old_value, new_value, reason) or None if nothing
    changed (flag off, regime looks fine, or regime_detail doesn't expose a
    classification this function recognizes).

    Deliberately conservative: this NEVER tightens, never exceeds
    REGIME_SOFTEN_MAX_PCT below the value configured at process start, and
    only fires on a small set of explicitly weak/mixed classifications --
    an unrecognized or missing classification is treated as "don't touch
    it" rather than guessed at.
    """
    global MIN_GROWTH_YOY_PCT
    if not REGIME_SOFTEN_GROWTH_BAR:
        return None

    classification = str((regime_detail or {}).get("classification") or "").strip().lower()
    # swing_trade_regime.check_market_regime() only ever classifies as
    # "bullish", "caution" (mixed breadth -- above one of 20w/50w SMA but
    # not both), "bearish", or "unknown" (index data unavailable). Softening
    # only makes sense for the two weak-but-not-gated-out states; "unknown"
    # is deliberately left alone (no trend read at all, nothing to react
    # to) and "bullish" obviously doesn't need softening. Note that if
    # REQUIRE_MARKET_REGIME_FILTER is on (the default) and
    # MARKET_REGIME_ALLOW_CAUTION is off (also the default), a "caution"
    # classification never reaches this function at all -- the regime gate
    # above already returns early on it. This only fires once a run has
    # actually been allowed to proceed under a weak regime.
    weak_classifications = {"caution", "bearish"}
    if not any(w in classification for w in weak_classifications):
        return None

    original = MIN_GROWTH_YOY_PCT
    floor = original * (1 - REGIME_SOFTEN_MAX_PCT / 100.0)
    new_value = round(max(floor, original * 0.925), 2)  # one fixed, small step -- not a search for "whatever passes"
    if new_value >= original:
        return None

    MIN_GROWTH_YOY_PCT = new_value
    reason = (
        f"market regime classified '{classification}' this run -- growth bar "
        f"softened by up to {_fmt_num(REGIME_SOFTEN_MAX_PCT)}% (capped) rather "
        "than leaving the strategy structurally unable to produce a signal "
        "during broad small/mid-cap weakness"
    )
    return (original, new_value, reason)


def _regime_softening_html(softening):
    """Small disclosure box, same visual language as _adjustments_html, so a
    regime-driven bar change is exactly as visible as a history-driven one --
    never a silent loosening."""
    if not softening:
        return ""
    old, new, reason = softening
    sans = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
    return (
        f'<div style="font-family:{sans};font-size:12px;color:#5C4A1E;line-height:1.6;'
        f'padding:12px 16px;background:#FBF3DC;border-radius:4px;border:1px solid #E9DCB0;'
        f'margin-bottom:14px;">'
        f'<strong>MIN_GROWTH_YOY_PCT softened this run</strong>: {old} &rarr; {new} '
        f'<span style="color:#8A8F9C;">({html.escape(reason)})</span></div>'
    )


def _regime_override_html(active, regime_detail):
    """Disclosure box for a run where the regime gate failed but
    REGIME_OVERRIDE_ON_RS let the scan proceed anyway -- same visual
    language as _regime_softening_html, so this is never a silent
    exception."""
    if not active:
        return ""
    sans = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
    classification = html.escape(str((regime_detail or {}).get("classification") or "unknown"))
    return (
        f'<div style="font-family:{sans};font-size:12px;color:#5C4A1E;line-height:1.6;'
        f'padding:12px 16px;background:#FBF3DC;border-radius:4px;border:1px solid #E9DCB0;'
        f'margin-bottom:14px;">'
        f'<strong>Regime gate overridden this run</strong>: broad market regime classified '
        f'&#39;{classification}&#39; (would normally skip the run entirely), but '
        f'REGIME_OVERRIDE_ON_RS is enabled, so every candidate below additionally had to clear '
        f'a stricter relative-strength bar (momentum &ge; {_fmt_num(REGIME_OVERRIDE_MIN_RSI)}, '
        f'price &ge;{_fmt_num(REGIME_OVERRIDE_SMA50D_MARGIN_PCT)}% above its 50-day moving '
        f'average) in place of the market-wide check.</div>'
    )


# -----------------------------
# Cross-run near-miss watchlist
# -----------------------------
def _is_technical_only_near_miss(rejected_stock):
    """
    True if a rejected candidate's ONLY 'hard' contradictions are technical
    (uptrend/RSI/MACD) -- i.e. fundamentals fully passed. These are exactly
    the candidates worth re-checking next run rather than re-discovering:
    growth doesn't change week to week, but a technical setup (a crossover,
    a pullback completing) can flip within days.
    """
    notes = rejected_stock.get("_verification_notes") or []
    hard = [n for n, sev in notes if sev == "hard"]
    if not hard:
        return False
    fundamentals_markers = ("Debt-to-equity", "ROE is", "Revenue growth YoY", "Net profit growth YoY")
    return not any(any(m in n for m in fundamentals_markers) for n in hard)


def _log_watchlist(rejected, today_str_iso, log_path=WATCHLIST_LOG):
    """
    Best-effort, silent-on-failure (same contract as _log_rejection_history):
    appends technical-only near-misses to a small CSV so they can be
    re-checked at the start of the NEXT run instead of only living in this
    run's rejection list.
    """
    try:
        rows = []
        for s in rejected:
            ticker = (s.get("ticker") or "").strip()
            if not ticker or not _is_technical_only_near_miss(s):
                continue
            rows.append({
                "date_added": today_str_iso,
                "ticker": ticker,
                "name": s.get("name") or ticker,
                "sector": s.get("sector") or "",
            })
        if not rows:
            return
        path = Path(log_path)
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["date_added", "ticker", "name", "sector"])
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
    except OSError as e:
        log.warning(f"Could not write watchlist log: {e}")


def _load_and_recheck_watchlist(log_path=WATCHLIST_LOG, max_age_days=WATCHLIST_MAX_AGE_DAYS):
    """
    Reads the persisted watchlist, drops stale entries (older than
    max_age_days) and duplicate tickers, and re-verifies each survivor's
    CURRENT fundamentals + technicals against real data right now.

    Returns (fundamentally_qualified, rewritten_rows) where
    fundamentally_qualified is a Stage-1-shaped candidate list ready to feed
    straight into build_technical_prompt (exactly like a fresh sector-scan
    result), and rewritten_rows is what should be written back to the CSV
    (stale/no-longer-fundamentally-qualified/now-fully-qualified entries
    removed; still-technicals-only-near-miss entries kept for next time).

    Never raises: a missing/corrupt log is treated as "empty watchlist",
    consistent with the rest of this file's best-effort logging.
    """
    path = Path(log_path)
    if not path.exists():
        return [], []

    try:
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except OSError as e:
        log.warning(f"Could not read watchlist log '{log_path}': {e}")
        return [], []

    today = datetime.now(ZoneInfo("Asia/Kolkata")).date()
    qualified = []
    keep_rows = []
    seen = set()
    for row in rows:
        ticker = (row.get("ticker") or "").strip()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        try:
            added = datetime.strptime(row["date_added"], "%Y-%m-%d").date()
            if (today - added).days > max_age_days:
                continue  # stale -- drop silently, it's had its chance
        except (KeyError, ValueError) as e:
            log.warning(f"Skipping malformed date for {ticker}: {e}")
            pass  # malformed date -- keep checking it rather than losing it

        stub = {"name": row.get("name") or ticker, "ticker": ticker, "sector": row.get("sector") or ""}
        fund_notes = _verify_fundamentals(stub)
        if any(sev in ("hard", "nodata") for _, sev in fund_notes):
            continue  # no longer fundamentally qualified (or ticker gone bad) -- drop

        tech_notes = _verify_technicals(stub)
        if any(sev == "hard" for _, sev in tech_notes):
            keep_rows.append(row)  # still a technical near-miss -- keep watching
            continue

        # Now clears BOTH fundamentals and technicals -- feed straight into
        # Stage 2 as a real candidate, and drop it from the watchlist (it's
        # graduated, not still "watching").
        data = _fetch_fundamentals(ticker) or {}
        qualified.append({
            "name": stub["name"],
            "ticker": ticker,
            "sector": stub["sector"],
            "market_cap_bucket": "?",
            "revenue_growth_yoy_pct": data.get("revenue_growth_yoy"),
            "profit_growth_yoy_pct": data.get("profit_growth_yoy"),
            "why": "Graduated from the near-miss watchlist -- technical setup has now completed.",
        })

    return qualified, keep_rows


def _rewrite_watchlist(rows, log_path=WATCHLIST_LOG):
    """Overwrites the watchlist CSV with exactly `rows` (already filtered by
    _load_and_recheck_watchlist) -- best-effort, silent-on-failure."""
    try:
        path = Path(log_path)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["date_added", "ticker", "name", "sector"])
            writer.writeheader()
            writer.writerows(rows)
    except OSError as e:
        log.warning(f"Could not rewrite watchlist log: {e}")


# -----------------------------
# Sector rotation across attempts
# -----------------------------
# Each retry attempt covers a fresh slice of sectors it hasn't already
# covered this run, instead of re-running a broad search over the same
# ground and hoping for different names. SECTORS_PER_ATTEMPT controls how
# many sectors go into one Stage-1 fundamentals-screen call.
#
# NOTE for USE_DETERMINISTIC_SCREEN=true (the default): the original
# reason for rotating ("hope the LLM finds different names this time") no
# longer applies -- the deterministic screen isn't stochastic, so re-
# checking the same sector slice would always produce the same result.
# Rotation still earns its keep here for a different reason: it bounds how
# many tickers' fundamentals get fetched (network calls) before giving up,
# by only pulling in a new slice of swing_trade_universe.py's ~208-ticker
# seed list per attempt rather than checking the entire universe on every
# single attempt regardless of whether an earlier attempt already found a
# qualifying stock. Raising SECTORS_PER_ATTEMPT trades "fewer attempts
# needed to see the whole universe" for "more yfinance calls up front even
# when attempt 1 would have succeeded on its own."
SECTORS = [
    "IT & Technology", "Pharma & Healthcare", "Banking & NBFC",
    "Capital Goods & Infrastructure", "Auto & Auto Ancillaries",
    "Chemicals & Fertilizers", "Defence", "Consumer & FMCG",
    "Metals & Mining", "Realty & Construction", "Energy & Power",
    "Textiles & Apparel", "Cement", "Telecom",
]
SECTORS_PER_ATTEMPT = _env_int("SECTORS_PER_ATTEMPT", 14)


def _sectors_for_attempt(attempt_idx):
    """Returns a rotating slice of SECTORS for this attempt (0-indexed) so
    successive attempts within one run cover new ground instead of
    re-searching the same sectors."""
    n = len(SECTORS)
    # Guard against a misconfigured (<=0) SECTORS_PER_ATTEMPT producing an
    # empty slice, which would send the model a prompt with no sectors to
    # search at all -- fall back to at least 1.
    k = max(1, min(SECTORS_PER_ATTEMPT, n))
    start = (attempt_idx * k) % n
    return [SECTORS[(start + i) % n] for i in range(k)]


# -----------------------------
# Schedule
# -----------------------------
# This script is intended to run WEEKLY, on Monday mornings IST (not daily).
# Suggested cron for the workflow yaml (03:00 UTC Monday = 08:30 IST Monday,
# comfortably before the 09:15 IST market open):
#   cron: '0 3 * * 1'
# _run_context() below detects Monday and adjusts prompt language (news/
# catalyst lookback window) accordingly -- see its docstring.
def _run_context():
    """
    Returns (today_str, is_monday, lookback_note).
    Since this runs weekly rather than daily, a plain "as of today" search
    misses most of the week's actual news flow (results, orders, upgrades,
    FII/DII activity) -- those all happened on days this script wasn't run.
    On a Monday run, lookback_note tells the model to treat the past week
    (last 5-7 trading sessions), not just the last 24 hours, as the relevant
    catalyst window, and to note that "current" price is effectively last
    Friday's close since markets are shut over the weekend.
    """
    now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
    today_str = now_ist.strftime("%d %B %Y")
    is_monday = now_ist.weekday() == 0
    if is_monday:
        lookback_note = (
            "and note that this is a WEEKLY scan run on Monday morning -- markets "
            "were closed over the weekend, so the most recent close is effectively "
            "last Friday's. Because a full week has passed since the previous scan, "
            "treat the ENTIRE PAST WEEK (the last 5-7 trading sessions), not just "
            "the last 24 hours, as the relevant window for news, results, broker "
            "upgrades/downgrades, and FII/DII activity -- do not limit searches to "
            "\"today\" only."
        )
    else:
        lookback_note = (
            "using the most recent available trading data and news as of right now."
        )
    return today_str, is_monday, lookback_note


# -----------------------------
# Prompts -- two-stage pipeline
# -----------------------------
# Stage 1 screens purely on fundamentals (a narrower, cheaper-to-verify
# universe), and every candidate it returns is independently re-checked
# against real data (_prefilter_by_fundamentals) BEFORE Stage 2 spends any
# model effort on technicals/entry-exit/sentiment for it. This avoids asking
# one model call to juggle fundamentals + technicals + sentiment + arithmetic
# all at once, which is a lot to get right simultaneously from search
# snippets alone.
STRATEGY_TYPES_BLOCK = """Stock Selection Strategy (choose one per stock in Stage 2):
- Momentum Breakout: a large-cap or quality mid-cap breaking out of a multi-month consolidation (Cup and Handle, Multi-Year Base, Symmetrical Triangle on a weekly chart) on significantly above-average volume, signaling a new sustained intermediate-term uptrend.
- Event-Driven: positioned to gain from a major near-term catalyst (regulatory approval, large contract win, demerger/spinoff, M&A arbitrage) with a clearly quantifiable price impact within a realistic multi-month window (the report derives each pick's actual horizon separately from its target distance and volatility -- see below -- rather than assuming a fixed period).
- Technical Swing Trade: a stock in a confirmed strong secular uptrend (above 50-WMA and 200-WMA) that has pulled back to a key support level (20-WMA, 38.2%/50% Fibonacci retracement, or horizontal support) with a clear reversal candlestick pattern.
- Fundamental Short-Term Bet: compelling valuation plus an exceptionally strong recent quarter (YoY and QoQ growth, clear beat on analyst consensus) and strongly positive guidance -- a re-rating play."""


def _mega_large_cap_caution():
    """
    Was previously a module-level string constant, formatted once at import
    time with whatever MIN_GROWTH_YOY_PCT happened to be then. That's a
    latent bug: _apply_auto_adjustments() mutates the MIN_GROWTH_YOY_PCT
    global mid-run (via globals()[gname] = new_value), but a string that was
    already formatted at import time can't retroactively pick that up -- the
    model would keep being told the ORIGINAL threshold forever, silently out
    of sync with the number the verifier is actually enforcing that run.
    Made into a function so every call re-reads the current global, same as
    every other threshold-bearing prompt string in this file already does.
    """
    return (
        "IMPORTANT market-cap steering: well-known large-caps and megacaps (e.g. "
        "TCS, Infosys, HCL Tech, Wipro, Reliance, HDFC Bank, ICICI Bank, Sun "
        f"Pharma, Bandhan Bank, and similar Nifty 50/Nifty 100 constituents) almost "
        f"never post EVEN ONE of >={_fmt_num(MIN_GROWTH_YOY_PCT)}% YoY revenue growth OR "
        f">={_fmt_num(MIN_GROWTH_YOY_PCT)}% YoY profit growth in "
        "a given quarter -- their revenue base is too large for that pace of growth "
        "except in rare one-off years, on either line. Repeatedly proposing "
        "these names and having them fail this filter wastes this search. "
        "Deprioritize them unless you have concrete, verifiable evidence of an "
        "unusual one-off beat this specific quarter. Spend most of your search "
        "effort instead on SMALL-CAP and MID-CAP stocks (BSE SmallCap 250 / BSE "
        "MidCap 150 universe, sub-Rs. 50,000 crore market cap) -- growth of this "
        "magnitude is far more common off a smaller revenue base, e.g. a company "
        "scaling from Rs. 200cr to Rs. 260cr quarterly revenue.\n\n"
        "Also avoid large, capital-intensive diversified conglomerates (e.g. "
        "Grasim, Godrej Industries, and similar cement/chemicals/infra-heavy "
        "holding companies) -- they structurally carry high debt-to-equity from "
        "their asset base, which fails the low-debt filter almost by default. "
        "Asset-light business models (IT services, specialty chemicals, "
        "formulation-focused pharma, consumer brands, defence electronics) are "
        "far more likely to combine high growth with low debt."
    )

SOURCE_QUALITY_NOTE = (
    "Source quality: do NOT rely on or cite social media posts (Instagram, "
    "Facebook, X/Twitter, Telegram) -- they are unverifiable and frequently "
    "wrong. If a search result is a prebuilt stock-screener page (screener.in, "
    "Trendlyne, Chartink, etc.) that looks like it already matches these "
    "filters, actually open/fetch that page and read the real list of stocks "
    "in its results table -- don't just note the link exists without reading "
    "what's on it."
)


def build_growth_screen_prompt(sectors, exclude_tickers, today_str, lookback_note):
    """
    STAGE 1 -- LLM-SEARCH FALLBACK PATH ONLY. Only called when
    USE_DETERMINISTIC_SCREEN=false; with the default (true), Stage 1 is
    _deterministic_fundamentals_screen instead, which checks every ticker
    in swing_trade_universe.py's seed list (208 tickers across 14 sectors
    as of the last audit -- see that file's own docstring) with no
    sampling cap and no LLM call. This function and its "search for 8-12
    companies" / "list up to 20 candidates" guidance below describe the
    OLD behavior and are dead code in the default configuration -- they
    still matter only if you explicitly set USE_DETERMINISTIC_SCREEN=false
    (e.g. to compare the two approaches, or if the seed universe hasn't
    been kept current). Don't read the "8-12 / up to 20" language below as
    a description of how many candidates a normal run actually checks --
    a normal run checks the whole seed list for the sector slice.

    Scoped to a rotating slice of sectors (see SECTORS / _sectors_for_attempt)
    so each attempt this run searches genuinely new ground instead of
    re-covering the same sectors. Deliberately does NOT ask for
    technicals/entry-exit/risk-reward yet -- that's Stage 2, run only
    against whichever candidates survive independent fundamentals
    verification.
    """
    sector_list = ", ".join(sectors)
    exclude_block = (
        "Do NOT propose any of these tickers again -- already checked and "
        "rejected earlier this run: " + ", ".join(sorted(t for t in exclude_tickers if t)) + "."
        if exclude_tickers else ""
    )
    return load_prompt(
        "swing/growth_screen",
        today_str=today_str,
        lookback_note=lookback_note,
        mega_large_cap_caution=_mega_large_cap_caution(),
        SOURCE_QUALITY_NOTE=SOURCE_QUALITY_NOTE,
        sector_list=sector_list,
        min_growth_yoy_pct=_fmt_num(MIN_GROWTH_YOY_PCT),
        exclude_block=exclude_block,
    )


def build_technical_prompt(candidates, exclude_tickers, today_str, lookback_note):
    """
    STAGE 2: technicals, sentiment, and trade-plan construction, run ONLY
    against the Stage-1 candidates that already passed independent
    fundamentals verification (_prefilter_by_fundamentals). The model isn't
    asked to re-justify growth here -- just to check the technical/sentiment/
    risk filters and build a trade plan for whichever names genuinely pass.
    """
    listing = "\n".join(
        f"- {c.get('name')} ({c.get('ticker')}) -- sector: {c.get('sector', '?')}, "
        f"cap: {c.get('market_cap_bucket', '?')}, independently-confirmed growth: "
        f"revenue {c.get('revenue_growth_yoy_pct', '?')}% / "
        f"profit {c.get('profit_growth_yoy_pct', '?')}% YoY"
        for c in candidates
    )
    exclude_block = (
        "Do NOT propose any of these tickers -- already checked and rejected "
        "earlier this run: " + ", ".join(sorted(t for t in exclude_tickers if t)) + "."
        if exclude_tickers else ""
    )
    uptrend_clause = "price above its 50-day MA; " if REQUIRE_UPTREND_FILTER else ""
    return load_prompt(
        "swing/technical",
        today_str=today_str,
        lookback_note=lookback_note,
        min_growth_yoy_pct=_fmt_num(MIN_GROWTH_YOY_PCT),
        listing=listing,
        exclude_block=exclude_block,
        STRATEGY_TYPES_BLOCK=STRATEGY_TYPES_BLOCK,
        uptrend_clause=uptrend_clause,
        min_rsi_oversold=_fmt_num(MIN_RSI_OVERSOLD),
        max_rsi_overbought=_fmt_num(MAX_RSI_OVERBOUGHT),
        min_risk_reward=_fmt_num(MIN_RISK_REWARD),
    )


# -----------------------------
# LLM call (larger token budget than main.py's per-stock reasoning calls,
# since this is one long-form response rather than a short per-stock blurb)
# -----------------------------
def _tavily_search(query, max_results=4, include_domains=None):
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
        payload = {
            "api_key": api_key,
            "query": query,
            # "advanced" costs more of the monthly search quota than "basic"
            # (2 credits vs 1) but is materially more relevance-ranked --
            # worth it here since this only runs a handful of queries per
            # invocation and "basic" was observed returning off-topic hits
            # (e.g. an unrelated motorcycle brand for a query containing
            # "Indian") for these short, finance-generic queries.
            "search_depth": "advanced",
            "max_results": max_results,
            "include_answer": False,
        }
        if include_domains:
            payload["include_domains"] = include_domains
        resp = requests.post(
            "https://api.tavily.com/search",
            json=payload,
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
        log.warning(f"Tavily search failed for query '{query}': {e}")
        return []


# Reputable Indian financial/news sources to scope Tavily searches to.
# Without this, short generic queries (e.g. containing just "Indian" or
# "stock market") can drift to off-topic results outside this domain
# entirely -- restricting to known finance sources fixes that at the
# search layer instead of trying to filter noise out afterwards.
_TAVILY_FINANCE_DOMAINS = [
    "moneycontrol.com", "nseindia.com", "bseindia.com", "screener.in",
    "economictimes.indiatimes.com", "livemint.com", "business-standard.com",
    "trendlyne.com", "tradingview.com",
]


def _gather_tavily_context(today_str, extra_queries=None):
    """
    Runs a small, fixed set of targeted queries covering the areas the
    prompt actually needs (momentum/breakout setups, FII/DII activity,
    broker calls) and assembles the results into a compact context block
    plus a deduplicated (title, url) source list. Returns (context_text,
    sources) -- context_text is "" if Tavily isn't configured or every
    query failed.

    extra_queries: optional list of additional, caller-supplied queries --
    e.g. the specific stock or fund names in the batch currently being
    analyzed. Without these, this tier only ever searches generic
    market-wide terms, so a name that isn't already part of the day's
    broad momentum/FII-DII/broker-call chatter gets no grounding at all
    even when this tier is the one that ends up serving the call.
    """
    queries = [
        f"NSE BSE India stock momentum breakout {today_str}",
        f"India stock market FII DII buying activity {today_str}",
        f"Indian stock brokerage buy rating target price upgrade {today_str}",
    ]
    if extra_queries:
        queries = queries + list(extra_queries)
    sources = []
    blocks = []
    for q in queries:
        for r in _tavily_search(q, include_domains=_TAVILY_FINANCE_DOMAINS):
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


def generate_analysis(prompt, max_tokens=1200, extra_context_queries=None, validate_fn=None):
    """
    Thin wrapper around llm_backend.generate_analysis() -- this module used
    to hand-roll its own copy of the entire fallback chain (groq/compound ->
    compound-mini -> Tavily+synthesis -> Gemini -> Mistral); that logic now
    lives once in llm_backend.py, shared with main.py's AI Stocks Story and
    (via this function) optionstrategy.py.

    Only the swing-trade-specific piece stays here: which Tavily queries to
    run for the "grounded" tier (see _gather_tavily_context above).

    This is the LIVE-SEARCH tier -- use it only for calls that need to find
    or verify current facts (prices, fundamentals, news). For a call that's
    reasoning over data already gathered earlier in the same run (a
    reformat/repair pass, a final synthesis stage), use
    llm_backend.generate_synthesis() instead -- it skips the search
    cascade entirely rather than spending live-search quota on a call that
    can't use it.

    extra_context_queries: optional list of extra search terms to fold into
    the Tavily grounding tier -- pass the specific stock/fund names a batch
    call is about here so that tier, if it ends up serving the call, is
    actually searching for them instead of only generic market terms.

    validate_fn: optional text -> bool, forwarded to llm_backend.generate_analysis.
    Without this, the chain's default validator only checks "non-empty" --
    so a tier that ignores the "respond with ONLY raw JSON" instruction and
    returns commentary/preamble text still counts as a "success", and the
    chain never falls through to a later tier (Gemini/Mistral) that
    might have actually returned parseable JSON. Callers building Stage 1/
    Stage 2 prompts should pass a validator that confirms the response
    parses as their expected schema (see run()'s call sites) so a
    malformed-but-nonempty reply is treated as this tier failing, not as a
    real "zero candidates" result.

    Returns (text, sources, used_live_search). `text` is "" (falsy) on total
    failure, same as before when it returned None -- existing `if not text:`
    checks in callers (this file and optionstrategy.py) work unchanged.
    """
    today_str = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%d %B %Y")

    def _gather_context():
        return _gather_tavily_context(today_str, extra_queries=extra_context_queries)

    return llm_backend.generate_analysis(
        prompt,
        max_tokens=max_tokens,
        gather_context_fn=_gather_context,
        validate_fn=validate_fn,
        log_label="swing-trade generation",
    )




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


def _parse_candidates_json(text):
    """Same parsing approach as _parse_analysis_json, but for Stage 1's
    {"candidates": [...]} schema instead of Stage 2's {"stocks": [...]}."""
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
    candidates = data.get("candidates") if isinstance(data, dict) else None
    if not isinstance(candidates, list):
        return None
    return candidates


def _fetch_current_price(ticker):
    ticker = (ticker or "").strip()
    if not ticker:
        return None, None
    try:
        df = fetch_stock_data(ticker)
        latest_close = float(df.iloc[-1].get("close")) if df is not None and not df.empty else None
        if latest_close is None:
            return None, None
        market = classify_market(ticker)
        currency_symbol = "₹" if market == "India" else "$"
        return latest_close, currency_symbol
    except Exception as e:
        log.warning(f"Could not fetch live price for '{ticker}': {e}")
        return None, None


def _attach_live_prices(stocks):
    for stock in stocks:
        price, currency_symbol = _fetch_current_price(stock.get("ticker"))
        if price is not None:
            stock["current_price_display"] = f"{currency_symbol}{price:,.2f}"
        else:
            stock["current_price_display"] = None
    return stocks


def _attach_risk_plan(stocks):
    """
    Attaches an independently-computed, ATR-based stop/target/position-size
    plan to each stock (review items 2+3) -- computed purely from real price
    history via swing_trade_risk, never from the model's flat stop_loss_pct/
    target1_pct. Deliberately additive: the model's flat-% fields are left
    untouched so the report can show both side by side (see
    _risk_plan_display) rather than silently overwriting a value that might
    itself be informative (e.g. if the two disagree sharply, that's worth
    seeing, not hiding).
    """
    for stock in stocks:
        try:
            plan = risk.compute_volatility_adjusted_plan(stock.get("ticker"), min_risk_reward=MIN_RISK_REWARD)
        except Exception as e:
            log.warning(f"Risk plan computation failed for {stock.get('ticker')}: {e}")
            plan = {"error": str(e)}
        stock["_atr_risk_plan"] = plan
    return stocks


def _risk_plan_display(stock):
    plan = stock.get("_atr_risk_plan")
    if not plan:
        return '<span style="color:#8A8F9C;">Could not compute (insufficient price history for ATR).</span>'
    lines = [
        f'Stop {plan["stop_loss_pct"]}% / Target {plan["target1_pct"]}% '
        f'<span style="color:#8A8F9C;">(from {risk.ATR_STOP_MULTIPLE:g}&times; weekly ATR of '
        f'{plan["atr_weekly"]} on a {plan["latest_close"]} close)</span>'
    ]
    if plan.get("shares_for_1pct_risk") is not None:
        lines.append(
            f'<div style="margin-top:2px;font-size:11px;color:#8A8F9C;">'
            f'Position size for {risk.RISK_PCT_PER_TRADE:g}% portfolio risk: '
            f'{plan["shares_for_1pct_risk"]} shares '
            f'(&asymp;{plan["position_value_for_1pct_risk"]:,.0f})</div>'
        )
    else:
        fallback_msg = html.escape(plan.get("position_size_note") or "")
        lines.append(
            '<div style="margin-top:2px;font-size:11px;color:#8A8F9C;">'
            + (fallback_msg or "Set PORTFOLIO_VALUE to get a share-count position size, not just a %.")
            + "</div>"
        )
    return "".join(lines)


def _compute_flexible_horizon(stocks):
    """
    Attaches a per-stock "_horizon_window" dict (or None if it can't be
    computed) using target1_pct/upside_target_pct against the ATR-based
    weekly move already sitting in _atr_risk_plan (see _attach_risk_plan) --
    see the HORIZON_* constants' docstring above for the full rationale.
    Deliberately additive, same pattern as _attach_risk_plan: doesn't touch
    the model's own "exit_date" field, so the report can show both the
    model's rough guess and the independently-derived window side by side.

    Must run AFTER _attach_risk_plan (needs _atr_risk_plan) and BEFORE
    _verify_stock_claims (whose sanity-bounds check references the window
    when flagging an aggressive upside target -- see _verify_sanity_bounds).
    """
    if not HORIZON_STRETCH_ENABLED:
        for s in stocks:
            s["_horizon_window"] = None
        return stocks

    for s in stocks:
        plan = s.get("_atr_risk_plan") or {}
        atr_weekly = plan.get("atr_weekly")
        latest_close = plan.get("latest_close")
        target_pct = _parse_first_number(s.get("target1_pct")) or _parse_first_number(s.get("upside_target_pct"))

        if not atr_weekly or not latest_close or not target_pct or target_pct <= 0:
            s["_horizon_window"] = None
            continue

        weekly_pct_move = (atr_weekly / latest_close) * 100
        if weekly_pct_move <= 0:
            s["_horizon_window"] = None
            continue

        # Straight-line estimate of how many weeks it takes THIS stock, at
        # its own typical weekly ATR-based move, to cover the distance to
        # target -- then buffered upward for the upper bound since price
        # doesn't move in a straight line every week, and clamped to a sane
        # floor/ceiling so extreme inputs can't produce a nonsensical window.
        weeks_needed = target_pct / weekly_pct_move
        min_months = weeks_needed / 4.345  # average weeks per month
        max_months = min_months * HORIZON_BUFFER_MULTIPLIER

        min_months = max(HORIZON_FLOOR_MONTHS, min(min_months, HORIZON_CEILING_MONTHS))
        max_months = max(min_months + 0.5, min(max_months, HORIZON_CEILING_MONTHS))

        s["_horizon_window"] = {
            "min_months": round(min_months, 1),
            "max_months": round(max_months, 1),
            "weekly_pct_move": round(weekly_pct_move, 2),
            "weeks_needed_at_atr_pace": round(weeks_needed, 1),
        }
    return stocks


def _horizon_window_display(stock):
    window = stock.get("_horizon_window")
    if not window:
        return (
            '<span style="color:#8A8F9C;">Could not compute (needs both a target % and ATR '
            "data) -- see the model's own Exit Date estimate above instead.</span>"
        )
    return (
        f'{window["min_months"]:g}&ndash;{window["max_months"]:g} months '
        f'<span style="color:#8A8F9C;">(target implies &asymp;{window["weeks_needed_at_atr_pace"]:g} weeks '
        f'at this stock&rsquo;s own {window["weekly_pct_move"]:g}%/week ATR pace)</span>'
    )


def _apply_confidence_sizing(watchlist):
    """
    Scales position size down for every watchlist-tier stock by
    WATCHLIST_POSITION_SIZE_MULTIPLIER (default: half) instead of loosening
    any entry filter to admit more close-but-imperfect candidates at full
    size -- see that constant's docstring above. Must run AFTER
    _split_qualifying (only applies to the watchlist tier) and after
    MAX_POSITION_SIZE_PCT's cap has already been applied in
    _passes_professional_quality_gate, so a watchlist stock's size is always
    scaled down from whatever already survived that cap -- never larger
    than a strict-tier stock's.

    Scales two independent things that both represent position size and
    would otherwise drift out of sync in the report:
    1. allocation_pct -- the model's own flat %.
    2. _atr_risk_plan's shares_for_1pct_risk / position_value_for_1pct_risk
       -- the independently-computed ATR-based share count (see
       _attach_risk_plan). Both need to move together, or the report would
       show a halved % next to a full-size share count.
    """
    if WATCHLIST_POSITION_SIZE_MULTIPLIER >= 1.0:
        return watchlist

    for s in watchlist:
        alloc = _parse_first_number(s.get("allocation_pct"))
        if alloc is not None:
            s["allocation_pct"] = f"{alloc * WATCHLIST_POSITION_SIZE_MULTIPLIER:g}%"

        plan = s.get("_atr_risk_plan")
        if plan and plan.get("shares_for_1pct_risk") is not None:
            try:
                plan["shares_for_1pct_risk"] = int(plan["shares_for_1pct_risk"] * WATCHLIST_POSITION_SIZE_MULTIPLIER)
                if plan.get("position_value_for_1pct_risk") is not None:
                    plan["position_value_for_1pct_risk"] = (
                        plan["position_value_for_1pct_risk"] * WATCHLIST_POSITION_SIZE_MULTIPLIER
                    )
            except (TypeError, ValueError):
                pass

        s["_confidence_sizing_note"] = (
            f"Watchlist tier -- position size cut to {WATCHLIST_POSITION_SIZE_MULTIPLIER:g}x strict sizing. "
            "Entry filters were NOT relaxed to admit this candidate; only size was, to limit exposure "
            "to a still-forming setup while still getting some exposure to it."
        )
    return watchlist


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
    if true_ratio < MIN_RISK_REWARD:
        notes.append((f"Risk:reward of 1 : {true_ratio} is below the 1:{_fmt_num(MIN_RISK_REWARD)} minimum the prompt requires.", "hard"))
    return notes


_technicals_cache = {}
_technicals_cache_lock = threading.Lock()


def _fetch_weekly_technicals(ticker):
    """
    Fetches OHLC price history for `ticker` and derives the weekly
    technicals (SMA20w/50w, RSI, MACD) used to independently verify the
    strategy's uptrend / RSI / MACD filters.

    NOTE: this used to call fetch_stock_data(ticker) once, bare, with no
    retry and no cache -- called repeatedly for the same ticker across a
    run (watchlist recheck, Stage-2 verification, near-miss reporting),
    and from concurrent candidate checks alongside stock_controller's and
    _fetch_fundamentals's own yfinance/NSE traffic. A single transient
    hiccup (rate-limit, timeout, momentary session/crumb failure) was
    enough to permanently mark an otherwise-good candidate "price history
    fetch failed" for the rest of the run -- exactly the same failure mode
    _fetch_fundamentals_uncached had before it was moved onto
    call_with_retries (see that function's docstring). This now retries
    transient failures the same way, and caches the result per ticker for
    the life of the process so a retried, successful fetch doesn't get
    thrown away and re-fetched (and re-risked) on the next call site.
    """
    ticker = (ticker or "").strip()
    if not ticker:
        return None
    if ticker in _technicals_cache:
        return _technicals_cache[ticker]
    with _technicals_cache_lock:
        if ticker in _technicals_cache:
            return _technicals_cache[ticker]
        result = _fetch_weekly_technicals_uncached(ticker)
        _technicals_cache[ticker] = result
        return result


def _fetch_price_history_secondary(ticker):
    """
    Backfill data source for OHLC price history, tried only when the
    primary fetch_stock_data() source (services.stock_fetcher) fails
    outright -- i.e. call_with_retries already exhausted its 5 attempts
    against transient errors and still came back empty -- rather than for
    every call. Goes directly through yfinance's own daily history() call,
    on the same shared session + process-wide throttle _fetch_fundamentals
    already uses (utils/yf_throttle.py), so it's independent of whatever
    fetch_stock_data wraps internally: an outage or bug specific to that
    ONE source no longer permanently disqualifies a stock on "technicals
    could not be verified" when a second source would have real data (seen
    in practice for names like Persistent Systems / Coforge, which have
    perfectly ordinary price history everywhere else). Returns a DataFrame
    with a lowercase "close" column indexed by date, or None if this
    source also fails.
    """
    try:
        import yfinance as yf
    except ImportError:
        return None

    def _raw():
        yt = yf.Ticker(ticker, session=get_shared_session())
        hist = yt.history(period="2y", interval="1d", auto_adjust=False)
        if hist is None or hist.empty:
            return None
        hist = hist.rename(columns={c: c.lower() for c in hist.columns})
        return hist[["close"]] if "close" in hist.columns else None

    try:
        return call_with_retries(_raw, max_attempts=3)
    except Exception as e:
        log.warning(f"Secondary (yfinance) price-history source also failed for '{ticker}': {e}")
        return None


def _fetch_weekly_technicals_uncached(ticker):
    try:
        used_secondary_source = False
        df = call_with_retries(lambda: fetch_stock_data(ticker), max_attempts=5)
        if df is None or len(df) < 30 or "close" not in getattr(df, "columns", []):
            log.info(
                f"Primary price-history source failed/insufficient for '{ticker}' -- "
                "trying secondary (yfinance) source before giving up."
            )
            df = _fetch_price_history_secondary(ticker)
            used_secondary_source = df is not None
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

        # Daily 50-day SMA -- used for the uptrend filter, loosened from a
        # stricter "price above BOTH the 20-week and 50-week SMA" structure
        # down to simply "price above its 50-day MA" (see _verify_technicals).
        # Computed on the raw daily series so it's checkable with far less
        # history (50 daily sessions is ~10 weeks) than the weekly RSI/MACD
        # below need, and independent of the weekly-history gate further down.
        latest_close_daily = round(float(close.iloc[-1]), 2)
        sma50d = round(float(close.rolling(50).mean().iloc[-1]), 2) if len(close) >= 50 else None

        weekly_close = close.resample("W").last().dropna()
        # Drop current incomplete week to avoid distorted signals
        if len(weekly_close) > 1 and weekly_close.index[-1] > pd.Timestamp.now(tz=weekly_close.index.tz) - pd.Timedelta(days=2):
            weekly_close = weekly_close.iloc[:-1]
        if len(weekly_close) < 55:
            return {
                "insufficient_history": True,
                "weeks_available": len(weekly_close),
                "latest_close": latest_close_daily,
                "sma50d": sma50d,
                "used_secondary_source": used_secondary_source,
            }

        delta = weekly_close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / loss.replace(0, float("nan"))
        rsi = 100 - (100 / (1 + rs))

        ema12 = weekly_close.ewm(span=12, adjust=False).mean()
        ema26 = weekly_close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - macd_signal

        # MACD bullish signal, broadened beyond "just crossed over": a fresh
        # bullish crossover (within last 3 bars) is still one qualifying
        # path, but so is simply being above signal right now, or the
        # histogram having risen for 2+ consecutive weekly sessions (i.e.
        # momentum building even before/without a crossover). Requiring a
        # FRESH crossover specifically was too narrow -- a stock that
        # crossed over 4 weeks ago and has stayed above signal since is
        # still bullish, it just doesn't get penalized for the crossover
        # itself being "stale".
        recent_bullish_crossover = False
        for i in range(-1, -4, -1):
            try:
                if macd.iloc[i] > macd_signal.iloc[i] and macd.iloc[i-1] <= macd_signal.iloc[i-1]:
                    recent_bullish_crossover = True
                    break
            except IndexError:
                break

        # "Above signal" is now read off the average MACD-vs-signal spread
        # over the last MULTI_SESSION_CONFIRMATION_WINDOW weekly sessions
        # (default 3), not a single session's snapshot -- see that
        # constant's docstring. A stock that's genuinely above signal stays
        # above signal on average; a stock that's only above signal because
        # of one noisy session's close correctly stops qualifying here,
        # without touching the 0-line threshold itself.
        window = min(MULTI_SESSION_CONFIRMATION_WINDOW, len(macd))
        spread_recent = (macd - macd_signal).iloc[-window:]
        macd_above_signal = bool(spread_recent.mean() > 0) if len(spread_recent) > 0 else bool(macd.iloc[-1] > macd_signal.iloc[-1])

        histogram_rising_2session = False
        if len(macd_hist) >= 3:
            try:
                histogram_rising_2session = bool(
                    macd_hist.iloc[-1] > macd_hist.iloc[-2] > macd_hist.iloc[-3]
                )
            except IndexError:
                histogram_rising_2session = False

        macd_bullish_signal = recent_bullish_crossover or macd_above_signal or histogram_rising_2session

        # RSI is likewise read as an average over the same
        # MULTI_SESSION_CONFIRMATION_WINDOW sessions rather than the single
        # latest session -- this is purely a noise-reduction change to WHAT
        # gets compared against MIN_RSI_OVERSOLD/MAX_RSI_OVERBOUGHT in
        # _verify_technicals; those threshold values themselves are
        # untouched, so the bar isn't loosened, just the reading of where a
        # stock sits relative to it is steadier. rsi14w_latest keeps the
        # single-session snapshot available for display/debugging.
        rsi_valid = rsi.dropna()
        rsi_window = rsi_valid.iloc[-window:] if len(rsi_valid) > 0 else rsi_valid
        rsi_avg = rsi_window.mean() if len(rsi_window) > 0 else None
        rsi_now = rsi.iloc[-1]
        rsi_prev = rsi.iloc[-3] if len(rsi) > 3 else None

        return {
            "insufficient_history": False,
            "latest_close": latest_close_daily,
            "sma50d": sma50d,
            "rsi14w": round(float(rsi_avg), 1) if rsi_avg is not None and pd.notna(rsi_avg) else None,
            "rsi14w_latest": round(float(rsi_now), 1) if pd.notna(rsi_now) else None,
            "rsi14w_prev": round(float(rsi_prev), 1) if rsi_prev is not None and pd.notna(rsi_prev) else None,
            "macd": round(float(macd.iloc[-1]), 3),
            "macd_signal": round(float(macd_signal.iloc[-1]), 3),
            "recent_bullish_crossover": recent_bullish_crossover,
            "macd_above_signal": macd_above_signal,
            "histogram_rising_2session": histogram_rising_2session,
            "macd_bullish_signal": macd_bullish_signal,
            "used_secondary_source": used_secondary_source,
        }
    except Exception as e:
        log.warning(f"Could not compute weekly technicals for '{ticker}': {e}")
        return None


def _verify_technicals(stock):
    """Returns a list of (note_text, severity) tuples -- see _verify_risk_reward."""
    ticker = (stock.get("ticker") or "").strip()
    if not ticker:
        return [("No ticker provided -- technicals could not be verified.", "soft")]

    tech = _fetch_weekly_technicals(ticker)
    if tech is None:
        return [("Technicals could not be independently verified (price history fetch failed).", "nodata")]

    notes = []

    if tech.get("used_secondary_source"):
        notes.append((
            "Price history came from the secondary (yfinance) backfill source -- the primary "
            "source failed for this ticker after retries -- see _fetch_price_history_secondary.",
            "soft",
        ))

    # Uptrend filter: a single "price above its 50-day MA" check, checkable
    # off the daily series regardless of whether there's enough weekly
    # history for RSI/MACD below.
    if REQUIRE_UPTREND_FILTER:
        price = tech.get("latest_close")
        sma50d = tech.get("sma50d")
        if price is None or sma50d is None:
            notes.append(("Not enough daily price history (<50 sessions) to verify the 50-day MA uptrend filter -- unverified.", "soft"))
        elif price < sma50d:
            notes.append((f"Price ({price}) is BELOW its 50-day MA ({sma50d}) -- uptrend filter failed.", "hard"))

    if tech.get("insufficient_history"):
        notes.append((
            f"Only {tech.get('weeks_available')} weeks of price history available -- "
            "not enough to verify weekly RSI or MACD; those claims are unverified.", "soft"
        ))
        return notes

    if tech["rsi14w"] is not None:
        if tech["rsi14w"] >= MAX_RSI_OVERBOUGHT:
            notes.append((f"Weekly RSI (avg of last {MULTI_SESSION_CONFIRMATION_WINDOW} sessions) is {tech['rsi14w']} (>={_fmt_num(MAX_RSI_OVERBOUGHT)}, overbought) -- outside the {_fmt_num(MIN_RSI_OVERSOLD)}-{_fmt_num(MAX_RSI_OVERBOUGHT)} band the strategy requires.", "hard"))
        elif tech["rsi14w"] < MIN_RSI_OVERSOLD:
            notes.append((f"Weekly RSI (avg of last {MULTI_SESSION_CONFIRMATION_WINDOW} sessions) is {tech['rsi14w']} (<{_fmt_num(MIN_RSI_OVERSOLD)}) -- outside the {_fmt_num(MIN_RSI_OVERSOLD)}-{_fmt_num(MAX_RSI_OVERBOUGHT)} band the strategy requires.", "hard"))
        # Day-to-day direction (rising/falling) no longer matters -- any RSI
        # inside the band qualifies regardless of which way it moved since
        # the prior weekly bar.

    if not tech.get("macd_bullish_signal"):
        notes.append((
            "MACD line is below its signal line and the histogram hasn't been "
            "rising for 2+ consecutive weekly sessions -- bullish MACD signal "
            "(above signal, rising histogram, or a fresh crossover) not yet met.",
            "soft",
        ))

    return notes


def _verify_relative_strength_override(stock):
    """
    Substitute, per-stock gate used ONLY for a run where the broad
    market-regime gate failed AND REGIME_OVERRIDE_ON_RS is enabled (see
    run() and REGIME_OVERRIDE_ON_RS's docstring above). This is what "the
    stock's own trend is strong enough to override a failed regime gate"
    actually checks: a materially stronger uptrend (price well above its
    50-day MA, by REGIME_OVERRIDE_SMA50D_MARGIN_PCT) AND a stronger weekly
    RSI floor (REGIME_OVERRIDE_MIN_RSI) than the normal filters require.

    Always returns "hard" notes (or none) -- deliberately an absolute
    blocker, not a core filter eligible for MIN_CORE_FILTERS_REQUIRED
    slack or the composite-score override, since it exists specifically to
    replace the market-wide check being skipped for this stock.
    """
    ticker = (stock.get("ticker") or "").strip()
    if not ticker:
        return [("No ticker provided -- relative-strength override could not be verified.", "hard")]

    tech = _fetch_weekly_technicals(ticker)
    if tech is None or tech.get("insufficient_history"):
        return [(
            "Market-regime gate failed this run, and the relative-strength override "
            f"requires verified weekly technicals to grant an exception for '{ticker}' -- "
            "none available, so no override is granted.", "hard"
        )]

    # NOTE: phrasing below deliberately avoids the literal substrings
    # "50-day MA" and "RSI" that _classify_core_filter() scans for -- this
    # override is meant to be an absolute blocker, not something that gets
    # (mis)classified as one of the five negotiable core filters just
    # because it also looks at price-vs-moving-average and momentum.
    notes = []
    price = tech.get("latest_close")
    sma50d = tech.get("sma50d")
    required_price = sma50d * (1 + REGIME_OVERRIDE_SMA50D_MARGIN_PCT / 100.0) if sma50d else None
    if price is None or required_price is None or price < required_price:
        notes.append((
            f"Relative-strength override requires price at least "
            f"{_fmt_num(REGIME_OVERRIDE_SMA50D_MARGIN_PCT)}% above its 50-day moving average (a "
            "materially stronger uptrend than the normal filter) since the broad market-regime "
            "gate failed this run -- not met.", "hard"
        ))

    momentum = tech.get("rsi14w")
    if momentum is None or momentum < REGIME_OVERRIDE_MIN_RSI:
        notes.append((
            f"Relative-strength override requires a weekly momentum reading >= "
            f"{_fmt_num(REGIME_OVERRIDE_MIN_RSI)} (stronger than the normal band) since the "
            f"broad market-regime gate failed this run -- not met (reading: {momentum}).", "hard"
        ))

    return notes


def _fetch_fundamentals(ticker):
    """
    Fetches point-in-time fundamentals (debt/equity, ROE) plus a quarterly
    revenue/net-income series directly via yfinance -- independent of
    whatever the LLM claimed about the company's financials. Returns a
    dict (fields may be None if unavailable) or None if the fetch fails
    entirely / yfinance isn't installed.

    NOTE: this used to build its own bare `yf.Ticker(ticker)` with no
    shared session and no throttle/retry, called from up to 10 concurrent
    threads (see the ThreadPoolExecutor in _deterministic_fundamentals_screen)
    across the entire ticker universe (100+ symbols) -- completely
    uncoordinated with controllers.stock_controller's and
    services.stock_fetcher's own yfinance traffic. That kind of unthrottled
    concurrent burst is exactly what produces the intermittent "quote not
    found" 404s seen in practice for perfectly valid tickers (LTIM.NS,
    TATAMOTORS.NS, etc.) -- those errors are a symptom of Yahoo's
    session/crumb handling breaking down under load, not those symbols
    actually being delisted. This now routes through the same shared
    session + process-wide throttle + rate-limit-aware retry used
    everywhere else (utils/yf_throttle.py), and results are cached per
    ticker for the life of the process since _verify_fundamentals() and
    check_candidate() were each independently calling this for the same
    ticker (2x the necessary yfinance calls for every qualifying
    candidate).
    """
    ticker = (ticker or "").strip()
    if not ticker:
        return None
    if ticker in _fundamentals_cache:
        return _fundamentals_cache[ticker]
    with _fundamentals_cache_lock:
        if ticker in _fundamentals_cache:
            return _fundamentals_cache[ticker]
        result = _fetch_fundamentals_uncached(ticker)
        _fundamentals_cache[ticker] = result
        return result


_fundamentals_cache = {}
_fundamentals_cache_lock = threading.Lock()


def _fetch_fundamentals_uncached(ticker):
    try:
        import yfinance as yf
    except ImportError:
        log.warning("yfinance not installed -- fundamentals verification skipped.")
        return None

    def _raw():
        yt = yf.Ticker(ticker, session=get_shared_session())
        info = yt.info or {}
        result = {
            "debt_to_equity": info.get("debtToEquity"),
            "roe": info.get("returnOnEquity"),
            "revenue_growth_yoy": None,
            "profit_growth_yoy": None,
            "growth_basis": None,
        }
        try:
            qf = yt.quarterly_financials  # rows = line items, cols = quarter-end dates, most-recent first
            if qf is not None and not qf.empty and qf.shape[1] >= 2:
                columns = list(qf.columns)
                periods = _growth_lookback_periods(columns, GROWTH_LOOKBACK_MODE)
                if periods is None and GROWTH_LOOKBACK_MODE != "yoy":
                    log.info(
                        f"'{ticker}': not enough quarterly history for "
                        f"GROWTH_LOOKBACK_MODE='{GROWTH_LOOKBACK_MODE}' -- falling back "
                        "to a plain YoY comparison for this ticker only."
                    )
                    periods = _growth_lookback_periods(columns, "yoy")
                if periods is None:
                    log.warning(
                        f"'{ticker}': no quarterly_financials column falls within "
                        "45 days of 1 year before the latest quarter -- skipping "
                        "growth calc (irregular/missing quarterly history) "
                        "rather than comparing against the wrong period."
                    )
                else:
                    current_idxs, prior_idxs, basis = periods
                    revenue_row = next((r for r in qf.index if "total revenue" in r.lower()), None)
                    income_row = next((r for r in qf.index if r.lower() == "net income"), None)

                    def _period_sum(row):
                        current_vals = qf.loc[row].iloc[current_idxs]
                        prior_vals = qf.loc[row].iloc[prior_idxs]
                        if current_vals.isna().any() or prior_vals.isna().any():
                            return None, None
                        return float(current_vals.sum()), float(prior_vals.sum())

                    if revenue_row is not None:
                        latest, year_ago = _period_sum(revenue_row)
                        if latest is not None and year_ago not in (None, 0):
                            result["revenue_growth_yoy"] = round(((latest - year_ago) / abs(year_ago)) * 100, 1)
                    if income_row is not None:
                        latest, year_ago = _period_sum(income_row)
                        if latest is not None and year_ago not in (None, 0):
                            if year_ago <= 0 or latest <= 0:
                                result["profit_growth_yoy"] = None  # Turnaround case, not standard growth
                            else:
                                result["profit_growth_yoy"] = round(((latest - year_ago) / abs(year_ago)) * 100, 1)
                    result["growth_basis"] = basis
            elif qf is None or qf.empty:
                pass
        except Exception as e:
            log.warning(f"Could not compute quarterly growth for '{ticker}': {e}")
        return result

    try:
        return call_with_retries(_raw, max_attempts=5)
    except Exception as e:
        log.warning(f"Could not fetch fundamentals for '{ticker}': {e}")
        return None


def _find_year_ago_index(columns, tolerance_days=45):
    """
    Given quarterly_financials' columns (most-recent-first, each a
    quarter-end Timestamp), returns the positional index of whichever
    column falls closest to 365 days before the most recent column --
    instead of assuming a fixed offset of 4 quarters back. A fixed offset
    silently compares against the wrong period whenever a company's
    reported quarterly history has a gap or an irregular (non-quarterly)
    reporting cadence. Returns None if no column falls within
    tolerance_days of the 1-year-ago target, so callers can skip the
    growth calc rather than compute it against a mismatched period.
    """
    if len(columns) < 2:
        return None
    latest_date = columns[0]
    target = latest_date - pd.Timedelta(days=365)
    best_idx, best_diff = None, None
    for i, col in enumerate(columns[1:], start=1):
        diff = abs((col - target).days)
        if diff <= tolerance_days and (best_diff is None or diff < best_diff):
            best_idx, best_diff = i, diff
    return best_idx


def _growth_lookback_periods(columns, mode):
    """
    Returns (current_idxs, prior_idxs, basis_label) -- positional indices
    into `columns` (quarterly_financials' most-recent-first column order)
    to sum for the "current" and "prior" side of a growth comparison, plus
    a short human-readable label for the basis used. Returns None if there
    isn't enough quarterly history for the requested mode, so callers can
    fall back (or skip) rather than compute against a mismatched/incomplete
    period. See GROWTH_LOOKBACK_MODE's docstring above for the rationale.

    "yoy": single latest quarter vs. whichever column falls closest to 1
    year before it (_find_year_ago_index) -- the original behavior.

    "trailing_2q": sum of the latest 2 quarters vs. the 2 quarters
    bracketing the point 1 year ago -- smooths a single weak quarter by
    averaging it against its neighbor on both sides of the comparison.

    "ttm": trailing-twelve-months (latest 4 quarters) vs. the prior TTM
    (the 4 quarters before that) -- needs 8+ quarters of history, which
    yfinance's quarterly_financials often doesn't expose; a None return
    here is expected and callers should fall back to "yoy" per-ticker.
    """
    if mode == "trailing_2q":
        if len(columns) < 2:
            return None
        year_ago_idx = _find_year_ago_index(columns)
        if year_ago_idx is None or year_ago_idx < 1:
            return None
        return [0, 1], [year_ago_idx - 1, year_ago_idx], "trailing 2-quarter"
    if mode == "ttm":
        if len(columns) < 8:
            return None
        return [0, 1, 2, 3], [4, 5, 6, 7], "trailing-twelve-month"
    # "yoy" (default / fallback)
    if len(columns) < 2:
        return None
    year_ago_idx = _find_year_ago_index(columns)
    if year_ago_idx is None:
        return None
    return [0], [year_ago_idx], "YoY"


# Growth this far above the mandatory threshold is disproportionately
# likely to be a one-off (impairment reversal, asset sale, exceptional
# item) rather than organic operating improvement -- e.g. a -308% "profit
# growth" figure for a name in an earlier run's rejected list smelled like
# exactly this, not genuine deterioration. yfinance's quarterly numbers are
# taken at face value elsewhere in this file; this check doesn't reject an
# anomalous figure (a real business CAN grow profit 80%+ YoY off a small
# base), it flags it as needing a human look at the actual exchange filing
# before being trusted, and asks the Stage 2 model to specifically address
# it rather than silently repeating the headline number.
ANOMALOUS_GROWTH_MULTIPLE = _env_float("ANOMALOUS_GROWTH_MULTIPLE", 3.0)  # 3x the mandatory threshold


def _check_anomalous_growth(ticker, data):
    """
    Returns a list of (note_text, "soft") tuples flagging growth figures
    that are large enough to warrant manual cross-verification against the
    actual quarterly result / exchange filing, rather than trusting
    yfinance's derived YoY number at face value. Two independent checks:

    1. Magnitude: revenue or profit growth >= ANOMALOUS_GROWTH_MULTIPLE x
       the mandatory threshold.
    2. Divergence: profit growth far outpacing revenue growth (e.g. profit
       +150% on flat/modest revenue growth) is the classic signature of a
       non-operating gain (asset sale, tax credit, one-off write-back)
       inflating net income without the underlying business actually
       growing that fast -- worth a second look even if neither figure
       alone crosses the magnitude threshold.

    Always "soft" (informational), never "hard" -- this module has no way
    to confirm from yfinance data alone whether a large number is a real
    beat or a one-off; that confirmation is exactly what it's asking a
    human (or the Stage 2 model's own web search) to go do.
    """
    notes = []
    rev_g = data.get("revenue_growth_yoy")
    profit_g = data.get("profit_growth_yoy")
    cap = MIN_GROWTH_YOY_PCT * ANOMALOUS_GROWTH_MULTIPLE

    if profit_g is not None and abs(profit_g) >= cap:
        notes.append((
            f"Net profit growth of {profit_g}% is {ANOMALOUS_GROWTH_MULTIPLE:g}x+ the "
            f"{_fmt_num(MIN_GROWTH_YOY_PCT)}% mandatory threshold -- unusually large swings "
            "like this are disproportionately likely to reflect a one-off item (impairment "
            "reversal, asset sale, exceptional charge) rather than organic growth. "
            "Cross-check against the actual quarterly result/exchange filing before trusting "
            f"this figure for '{ticker}'.", "soft"
        ))
    if rev_g is not None and abs(rev_g) >= cap:
        notes.append((
            f"Revenue growth of {rev_g}% is {ANOMALOUS_GROWTH_MULTIPLE:g}x+ the "
            f"{_fmt_num(MIN_GROWTH_YOY_PCT)}% mandatory threshold for '{ticker}' -- verify "
            "against the exchange filing (this can be a real scale-up, but can also be a "
            "one-off bulk order, M&A-driven consolidation, or restated prior-year base).", "soft"
        ))
    if (
        profit_g is not None and rev_g is not None and rev_g > 0
        and profit_g > rev_g * 3 and profit_g >= MIN_GROWTH_YOY_PCT
    ):
        notes.append((
            f"Profit growth ({profit_g}%) is far outpacing revenue growth ({rev_g}%) for "
            f"'{ticker}' -- this divergence is the classic signature of a non-operating gain "
            "(asset sale, tax credit, write-back) inflating net income rather than the "
            "underlying business growing this fast. Worth confirming against the P&L's "
            "'exceptional items' / 'other income' line before trusting it.", "soft"
        ))
    return notes


def _verify_fundamentals(stock):
    """
    Independently checks the prompt's mandatory fundamentals filters
    (low debt-to-equity, high/improving ROE, >=MIN_GROWTH_YOY_PCT% YoY
    revenue OR profit growth -- only one of the two needs to clear the
    bar, not both) against real data instead of trusting the model's
    fundamental narrative. Returns a list of (note_text, severity) tuples.
    """
    ticker = (stock.get("ticker") or "").strip()
    if not ticker:
        return [("No ticker provided -- fundamentals could not be verified.", "soft")]

    data = _fetch_fundamentals(ticker)
    if data is None:
        return [("Fundamentals could not be independently verified (data fetch failed).", "nodata")]

    # yfinance can resolve a ticker symbol (so _fetch_fundamentals doesn't
    # raise / return None) while returning literally no financial data for
    # it -- this is the common signature of a delisted, renamed, or
    # hallucinated ticker, not a working symbol with some fields missing.
    # Treat that case the same as a total fetch failure ("nodata") rather
    # than four separate "soft" notes -- otherwise _verify_stock_claims'
    # "price + technicals + fundamentals all unfetchable => reject" rule
    # never fires, because it specifically looks for a "nodata" fundamentals
    # note and four individually-soft notes don't produce one.
    if all(data.get(k) is None for k in (
        "debt_to_equity", "roe", "revenue_growth_yoy", "profit_growth_yoy"
    )):
        return [(
            "No fundamentals data at all came back for this ticker (debt/equity, "
            "ROE, and both growth figures are all empty) -- this is the usual "
            "signature of a wrong, delisted, or hallucinated symbol rather than "
            "a data provider missing one or two fields.", "nodata"
        )]

    notes = []

    dte = data.get("debt_to_equity")
    if dte is not None:
        if dte > MAX_DEBT_TO_EQUITY_PCT:
            notes.append((f"Debt-to-equity is {dte:.0f}% -- elevated (above the {_fmt_num(MAX_DEBT_TO_EQUITY_PCT)}% threshold), contradicts the 'low debt-to-equity' requirement.", "hard"))
    else:
        notes.append(("Debt-to-equity not available from data provider -- unverified.", "soft"))

    roe = data.get("roe")
    if roe is not None:
        # yfinance returns ROE as a decimal ratio (e.g., 0.15 = 15%)
        # Values with abs > 10 are likely already in percentage form
        roe_pct = roe * 100 if abs(roe) < 10 else roe
        if roe_pct < MIN_ROE_PCT:
            notes.append((f"ROE is {roe_pct:.1f}% -- weak (below the {_fmt_num(MIN_ROE_PCT)}% threshold), contradicts the 'high/improving ROCE/ROE' requirement.", "hard"))
    else:
        notes.append(("ROE not available from data provider -- unverified.", "soft"))

    rev_g = data.get("revenue_growth_yoy")
    profit_g = data.get("profit_growth_yoy")
    basis = data.get("growth_basis")
    # Note text below always keeps the literal "growth YoY is X%" phrasing
    # regardless of GROWTH_LOOKBACK_MODE -- _metric_patterns() regex-parses
    # that exact substring for rejection-history logging, so the basis is
    # disclosed as a suffix after the number instead of changing the label.
    basis_suffix = f" (computed on a {basis} basis)" if basis and basis != "YoY" else ""
    growths = [g for g in (rev_g, profit_g) if g is not None]
    if growths:
        best_growth = max(growths)
        if best_growth < MIN_GROWTH_YOY_PCT:
            notes.append((f"Neither revenue growth ({rev_g}%) nor net profit growth ({profit_g}%) meets the {_fmt_num(MIN_GROWTH_YOY_PCT)}% threshold{basis_suffix}.", "hard"))
        elif rev_g is not None and rev_g < MIN_GROWTH_YOY_PCT:
            notes.append((f"Revenue growth YoY is {rev_g}% (below {_fmt_num(MIN_GROWTH_YOY_PCT)}%){basis_suffix}, but net profit growth YoY is {profit_g}% (qualifies on profit growth).", "soft"))
        elif profit_g is not None and profit_g < MIN_GROWTH_YOY_PCT:
            notes.append((f"Net profit growth YoY is {profit_g}% (below {_fmt_num(MIN_GROWTH_YOY_PCT)}%){basis_suffix}, but revenue growth YoY is {rev_g}% (qualifies on revenue growth).", "soft"))
    else:
        notes.append(("Growth could not be computed (insufficient quarterly history from data provider).", "soft"))

    notes.extend(_check_anomalous_growth(ticker, data))

    return notes


def _prefilter_by_fundamentals(candidates):
    """
    Independently re-checks each Stage-1 candidate's fundamentals against
    real data (via the same _verify_fundamentals used on final picks) --
    the model's self-reported growth numbers are not trusted at face value
    here either. Only candidates with zero 'hard' contradictions move on to
    the Stage-2 technicals call; this is what makes the two-stage split
    actually save effort (no technicals prompt is ever built for a stock
    whose growth claim doesn't hold up).

    Returns (qualified, rejected) -- qualified is the original candidate
    dicts (unchanged, for building the Stage 2 prompt); rejected candidates
    carry a "_verification_notes" key so they can be reported in the "no
    qualifying trade" summary same as final-stage rejections.

    A candidate is also rejected here if fundamentals data couldn't be
    fetched at all (severity "nodata", e.g. yfinance has no such ticker) --
    this stage exists specifically to confirm >=MIN_GROWTH_YOY_PCT% YoY growth, and "no data
    to confirm it with" is functionally the same failure as "confirmed and
    it's below threshold". It's also usually a sign the ticker itself is
    wrong/hallucinated rather than a transient data-provider hiccup, so
    catching it here avoids spending a Stage 2 LLM call on it.
    """
    qualified, rejected = [], []
    for c in candidates:
        ticker = (c.get("ticker") or "").strip()
        if not ticker:
            record = dict(c)
            record["_verification_notes"] = [("No ticker provided by the model for this candidate -- cannot verify or trade it.", "hard")]
            rejected.append(record)
            continue
        notes = _verify_fundamentals({"ticker": ticker})
        blocking = [n for n, sev in notes if sev in ("hard", "nodata")]
        if blocking:
            record = dict(c)
            record["_verification_notes"] = notes
            rejected.append(record)
        else:
            qualified.append(c)
    return qualified, rejected


def _load_extra_universe_tickers():
    """
    Optional supplement to swing_trade_universe.py's static seed list -- see
    EXTRA_UNIVERSE_FILE's docstring above. Reads a CSV of
    name,ticker,sector,bucket rows and returns them in the same 4-tuple
    shape universe.tickers_for_sectors() yields, so the two sources can be
    merged with identical dedupe/exclude handling. Best-effort: a missing,
    unset, or malformed file is logged and treated as "no extra tickers"
    rather than failing the run.
    """
    if not EXTRA_UNIVERSE_FILE:
        return []
    path = Path(EXTRA_UNIVERSE_FILE)
    if not path.exists():
        log.warning(f"EXTRA_UNIVERSE_FILE='{EXTRA_UNIVERSE_FILE}' does not exist -- ignoring.")
        return []
    try:
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except OSError as e:
        log.warning(f"Could not read EXTRA_UNIVERSE_FILE '{EXTRA_UNIVERSE_FILE}': {e}")
        return []

    out = []
    for row in rows:
        ticker = (row.get("ticker") or "").strip()
        sector = (row.get("sector") or "").strip()
        if not ticker or not sector:
            continue
        name = (row.get("name") or "").strip() or ticker
        bucket = (row.get("bucket") or row.get("market_cap_bucket") or "?").strip()
        out.append((name, ticker, sector, bucket))
    return out


def _deterministic_fundamentals_screen(sectors, exclude_tickers):
    """
    Stage-1 replacement for USE_DETERMINISTIC_SCREEN=true: iterates
    swing_trade_universe.py's static ticker list for the given sectors and
    checks EACH one against real data via the exact same _verify_fundamentals
    the old LLM-discovered candidates were checked against -- no LLM call,
    no "did the model actually find good candidates" uncertainty, and no
    sample-of-8-12-per-sector limit (the whole seed list gets checked).

    Returns (qualified_candidates, rejected) in the same shapes
    _prefilter_by_fundamentals produces, so run() can feed the result
    straight into Stage 2 (build_technical_prompt) unchanged.

    qualified_candidates carry a "why" field describing the real numbers
    found, instead of an LLM's freeform one-sentence claim -- there's no
    model output to summarize here, this candidate qualified purely on
    fetched data.
    """
    qualified, rejected = [], []
    seen_this_call = set()
    
    # Gather candidates: the static seed list first, then any widened
    # supplement (see EXTRA_UNIVERSE_FILE) filtered to this attempt's
    # sectors -- same dedupe/exclude rules for both sources so a ticker
    # present in both is only checked once.
    candidates = []
    for name, ticker, sector, bucket in universe.tickers_for_sectors(sectors):
        ticker_u = ticker.strip().upper()
        if ticker_u in exclude_tickers or ticker_u in seen_this_call:
            continue
        seen_this_call.add(ticker_u)
        candidates.append((name, ticker, sector, bucket))

    for name, ticker, sector, bucket in _load_extra_universe_tickers():
        if sector not in sectors:
            continue
        ticker_u = ticker.strip().upper()
        if ticker_u in exclude_tickers or ticker_u in seen_this_call:
            continue
        seen_this_call.add(ticker_u)
        candidates.append((name, ticker, sector, bucket))

    def check_candidate(cand):
        name, ticker, sector, bucket = cand
        stub = {"name": name, "ticker": ticker, "sector": sector, "market_cap_bucket": bucket}
        notes = _verify_fundamentals(stub)
        blocking = [n for n, sev in notes if sev in ("hard", "nodata")]

        if blocking:
            record = dict(stub)
            record["_verification_notes"] = notes
            return (False, record)

        data = _fetch_fundamentals(ticker) or {}
        qual = {
            "name": name,
            "ticker": ticker,
            "sector": sector,
            "market_cap_bucket": bucket,
            "revenue_growth_yoy_pct": data.get("revenue_growth_yoy"),
            "profit_growth_yoy_pct": data.get("profit_growth_yoy"),
            "why": (
                f"Deterministic screen: revenue +{data.get('revenue_growth_yoy')}% / "
                f"profit +{data.get('profit_growth_yoy')}% YoY (fetched directly, not "
                "model-reported)."
            ),
        }
        return (True, qual)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(check_candidate, cand): cand for cand in candidates}
        for future in as_completed(futures):
            try:
                is_qual, record = future.result()
                if is_qual:
                    qualified.append(record)
                else:
                    rejected.append(record)
            except Exception as e:
                log.error(f"Error checking candidate in deterministic screen: {e}")

    return qualified, rejected


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
        window = stock.get("_horizon_window")
        horizon_clause = (
            f"implied horizon ~{window['min_months']:g}-{window['max_months']:g} months at this stock's own ATR pace"
            if window else "no computed horizon available to sanity-check the pace"
        )
        notes.append((f"Upside target of {upside}% ({horizon_clause}) is unusually aggressive -- treat as a stretch case, not a base case.", "soft"))

    risk_level = (stock.get("risk_level") or "").strip().lower()
    if risk_level not in ("medium", "high"):
        notes.append((f"Risk level '{stock.get('risk_level')}' is not one of the expected 'Medium'/'High' values.", "soft"))

    return notes


def _verify_stock_claims(stocks, regime_override_active=False):
    for stock in stocks:
        rr_notes = _verify_risk_reward(stock)
        tech_notes = _verify_technicals(stock)
        fund_notes = _verify_fundamentals(stock)
        sanity_notes = _verify_sanity_bounds(stock)
        rs_override_notes = _verify_relative_strength_override(stock) if regime_override_active else []

        price_missing = not stock.get("current_price_display")
        tech_no_data = any(sev == "nodata" for _, sev in tech_notes)
        fund_no_data = any(sev == "nodata" for _, sev in fund_notes)

        notes = rr_notes + tech_notes + fund_notes + sanity_notes + rs_override_notes
        if price_missing:
            notes.append(("Live quote lookup failed for this ticker -- confirm it's a real, currently-listed symbol before trading it.", "nodata"))

        # If EVERY independent real-data source failed for this ticker --
        # price, technicals, AND fundamentals -- that's a much stronger
        # signal than three isolated "couldn't verify" notes: it usually
        # means the ticker itself is wrong, delisted, or hallucinated by
        # the model, not that three unrelated data sources all happened to
        # have an outage at once. "Zero verified data" is not the same
        # thing as "zero contradictions", so escalate this specific
        # combination to a hard contradiction -- _split_qualifying then
        # excludes it instead of letting a totally unverifiable pick
        # through with a star rating on a technicality. (Ordinarily this
        # is already caught one stage earlier by _prefilter_by_fundamentals;
        # this is a second layer for the case where fundamentals data
        # happened to be fetchable but price/technicals weren't.)
        if price_missing and tech_no_data and fund_no_data:
            notes.append((
                f"No real data source (live price, price history, or fundamentals) could be "
                f"found for ticker '{stock.get('ticker') or '?'}' -- this usually means the "
                "ticker is wrong, delisted, or hallucinated rather than a temporary data "
                "outage. Treating as disqualifying rather than merely unverifiable.", "hard"
            ))

        stock["_verification_notes"] = notes
        _adjust_confidence(stock, notes)

        # Composite score (review item 6): the hard-gate pass/fail above is
        # unchanged and still decides qualify/reject -- this adds a ranked,
        # 0-100 diagnostic on top so "how close did this candidate actually
        # come" is visible instead of thrown away, and so multiple
        # qualifying candidates in one run can be ranked rather than taken
        # in whatever order the model listed them.
        composite, breakdown = scoring.compute_composite_score(rr_notes, tech_notes, fund_notes, sanity_notes)
        stock["_composite_score"] = composite
        stock["_composite_breakdown"] = breakdown
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
    soft_count = sum(1 for _, sev in notes if sev != "hard")  # "soft" and "nodata" both count as unverifiable here
    penalty = hard_count * 0.8 + soft_count * 0.2
    stock["confidence_score_adjusted"] = max(0.0, round(original - penalty, 1))
    stock["_confidence_penalty_detail"] = f"{hard_count} contradiction(s), {soft_count} unverifiable item(s)"


def _hard_contradictions(stock):
    """Text of every 'hard' verification note -- i.e. an independent check that
    actively disagrees with the model's claim, as opposed to a 'soft' note where
    something simply couldn't be verified either way."""
    return [n for n, sev in (stock.get("_verification_notes") or []) if sev == "hard"]


# The five CORE swing-setup filters -- see MIN_CORE_FILTERS_REQUIRED's
# docstring above. Matched by keyword against the hard-note text each
# verifier already produces, in the same spirit as _metric_patterns().
# A hard note matching none of these (debt-to-equity, ROE, missing price,
# hallucinated ticker, position sizing, etc.) is NOT a core filter and
# always blocks regardless of MIN_CORE_FILTERS_REQUIRED or a high
# composite score -- see _split_qualifying.
_CORE_FILTER_KEYWORDS = (
    ("uptrend", ("50-day MA",)),
    ("rsi", ("RSI",)),
    ("macd", ("MACD",)),
    ("growth", ("revenue growth", "profit growth")),
    ("risk_reward", ("Risk:reward", "risk:reward")),
)


def _classify_core_filter(note_text):
    """Returns the core-filter name a hard note belongs to, or None if it's
    outside the five-filter flexibility (see _CORE_FILTER_KEYWORDS)."""
    for name, keywords in _CORE_FILTER_KEYWORDS:
        if any(kw in note_text for kw in keywords):
            return name
    return None


def _passes_professional_quality_gate(stock, enforce_composite_floor=True):
    """
    enforce_composite_floor=False is used for the watchlist tier (see
    _split_qualifying): a watchlist candidate is EXPECTED to score lower
    than MIN_COMPOSITE_SCORE (that's part of why it's watchlist and not
    strict), so that floor doesn't apply there -- but the live-price data-
    integrity check and the position-size cap always still apply to any
    stock reaching this function, strict or watchlist.
    """
    if not stock.get("current_price_display"):
        return False, "Missing a verified live price for the trade setup"

    # Auto-cap position size instead of hard-rejecting
    alloc = _parse_first_number(stock.get("allocation_pct"))
    if alloc is not None and alloc > MAX_POSITION_SIZE_PCT:
        stock["allocation_pct"] = f"{MAX_POSITION_SIZE_PCT:g}%"
        stock["_position_cap_applied"] = True

    if enforce_composite_floor:
        composite = stock.get("_composite_score")
        if composite is not None and composite < MIN_COMPOSITE_SCORE:
            return False, f"Composite score {composite:.1f}/100 is below the threshold of {MIN_COMPOSITE_SCORE:.1f}/100"

    return True, None


def _split_qualifying(stocks):
    """
    Splits verified stocks into (qualifying, watchlist, rejected).

    Previously a stock qualified only if it had ZERO 'hard' contradictions
    anywhere. Now:

    1. A hard contradiction OUTSIDE the five core setup filters (debt-to-
       equity, ROE, missing price, hallucinated ticker, a failed
       relative-strength override, etc. -- see _classify_core_filter)
       always disqualifies into `rejected`, same as before. These are
       basic quality/data-integrity/absolute checks, not part of the setup
       itself, so they're never up for negotiation by either tier below.
    2. Among the five core filters (uptrend, RSI, MACD, growth, risk:reward),
       a stock qualifies STRICT if at least MIN_CORE_FILTERS_REQUIRED of
       the 5 hard-pass -- i.e. up to (CORE_FILTER_COUNT -
       MIN_CORE_FILTERS_REQUIRED) of them can hard-fail and it still
       qualifies strict (default: any 3 of 5).
    3. A stock that doesn't clear that core-filter bar can still qualify
       STRICT via COMPOSITE_SCORE_CORE_FILTER_OVERRIDE: a sufficiently high
       overall composite score (default >=70/100) is treated as strong
       enough evidence on its own, even with more than one core filter
       failing -- still subject to rule 1 above.
    4. A stock that clears rule 1 (no absolute blocker) but misses both
       rule 2 and rule 3 lands in WATCHLIST instead of rejected, provided
       it's within WATCHLIST_EXTRA_CORE_FILTER_SLACK core filters of the
       strict bar (default: one filter more than strict tolerates) and
       WATCHLIST_TIER_ENABLED is on. This is "fails one filter but close"
       -- shown as its own tier in the email rather than silently promoted
       into strict or silently dropped. The composite-score floor
       (MIN_COMPOSITE_SCORE) does NOT apply to this tier -- scoring below
       that floor is expected for a watchlist candidate -- but the live-
       price/data-integrity check and position-size cap still do.

    'Soft' notes (couldn't be verified either way) never block, in any
    tier.
    """
    qualifying, watchlist, rejected = [], [], []
    for s in stocks:
        hard_notes = _hard_contradictions(s)
        non_core_failed = [n for n in hard_notes if _classify_core_filter(n) is None]
        core_failed = {cat for cat in (_classify_core_filter(n) for n in hard_notes) if cat}

        if non_core_failed:
            rejected.append(s)
            continue

        core_passed = CORE_FILTER_COUNT - len(core_failed)
        composite = s.get("_composite_score")
        passes_core_filter_bar = core_passed >= MIN_CORE_FILTERS_REQUIRED
        passes_composite_override = composite is not None and composite >= COMPOSITE_SCORE_CORE_FILTER_OVERRIDE

        if passes_core_filter_bar or passes_composite_override:
            if REQUIRE_PROFESSIONAL_QUALITY_GATE:
                passes, reason = _passes_professional_quality_gate(s, enforce_composite_floor=True)
                if not passes:
                    s["_verification_notes"] = list(s.get("_verification_notes") or []) + [(reason, "hard")]
                    rejected.append(s)
                    continue
            qualifying.append(s)
            continue

        watchlist_min_core_passed = MIN_CORE_FILTERS_REQUIRED - WATCHLIST_EXTRA_CORE_FILTER_SLACK
        if WATCHLIST_TIER_ENABLED and core_passed >= watchlist_min_core_passed:
            passes, reason = (True, None)
            if REQUIRE_PROFESSIONAL_QUALITY_GATE:
                passes, reason = _passes_professional_quality_gate(s, enforce_composite_floor=False)
            if passes:
                s["_watchlist_reason"] = (
                    f"{core_passed}/{CORE_FILTER_COUNT} core setup filters passed this run "
                    f"(strict needs {MIN_CORE_FILTERS_REQUIRED}/{CORE_FILTER_COUNT}) -- close, "
                    "not a full qualifier."
                )
                watchlist.append(s)
                continue
            s["_verification_notes"] = list(s.get("_verification_notes") or []) + [(reason, "hard")]
            rejected.append(s)
            continue

        composite_clause = (
            f"composite score ({composite:.1f}/100) is below the "
            f"{_fmt_num(COMPOSITE_SCORE_CORE_FILTER_OVERRIDE)}/100 override threshold"
            if composite is not None else
            "no composite score is available to fall back on"
        )
        note = (
            f"Only {core_passed}/{CORE_FILTER_COUNT} core setup filters passed "
            f"(needs {MIN_CORE_FILTERS_REQUIRED}/{CORE_FILTER_COUNT} for strict, "
            f"{watchlist_min_core_passed}/{CORE_FILTER_COUNT} for watchlist), and {composite_clause}."
        )
        s["_verification_notes"] = list(s.get("_verification_notes") or []) + [(note, "hard")]
        rejected.append(s)
    return qualifying, watchlist, rejected


def _verification_display(stock):
    notes = stock.get("_verification_notes") or []
    if not notes:
        return '<span style="color:#2F5233;font-weight:700;">Verified -- no contradictions found</span>'
    hard = [n for n, sev in notes if sev == "hard"]
    soft = [n for n, sev in notes if sev != "hard"]
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


def _risk_reward_display(stock):
    verified = stock.get("risk_reward_ratio_verified")
    stated = stock.get("risk_reward_ratio")
    stated_str = str(stated).strip() if stated not in (None, "") else None
    if verified is None:
        return html.escape(stated_str) if stated_str else None
    if stated_str and stated_str != verified:
        return (
            f'<strong>{html.escape(verified)}</strong> '
            f'<span style="font-size:11px;color:#8A8F9C;">(model stated {html.escape(stated_str)})</span>'
        )
    return f'<strong>{html.escape(verified)}</strong>'


def _render_one_stock_card(stock, idx, sans, tier_badge=None):
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
        raw_row("Composite Score (0-100)", scoring.composite_score_html),
        raw_row("Risk Level", lambda s: _risk_level_badge(s.get("risk_level"))),
        row("Key Catalysts", "key_catalysts"),
        raw_row("Risk : Reward", _risk_reward_display),
        row("Allocation (% of capital)", "allocation_pct"),
        row("Entry Date (Targeted)", "entry_date"),
        row("Exit Date (Model's Estimate)", "exit_date"),
        raw_row("Time Horizon (Target-Based)", _horizon_window_display),
        row("Strategy Type", "strategy_type"),
        row("Upside Target %", "upside_target_pct", value_color="#2F5233", bold=True),
        row("Stop-Loss %", "stop_loss_pct", value_color="#8B2E2E", bold=True),
        row("Target 1 (T1) %", "target1_pct"),
        row("Target 2 (T2) %", "target2_pct"),
        raw_row("ATR-Based Risk Plan", _risk_plan_display),
        row("Recent Top Buyers (FII/DII)", "top_buyers"),
        row("Broker Recommendations", "broker_recommendations"),
        raw_row("Data Verification", _verification_display),
    ])

    name = esc(stock.get("name"))
    ticker = esc(stock.get("ticker"))
    rationale = esc(stock.get("rationale"))
    badge_html = (
        f' <span style="background:#B08D57;color:#14213D;border-radius:3px;padding:2px 7px;'
        f'font-size:10px;font-weight:700;letter-spacing:0.03em;vertical-align:middle;">{html.escape(tier_badge)}</span>'
        if tier_badge else ""
    )
    watchlist_reason_html = (
        f'<div style="margin-top:6px;font-family:{sans};font-size:11px;color:#8A6D1E;">'
        f'<strong>Why watchlist, not strict:</strong> {esc(stock.get("_watchlist_reason"))}</div>'
        if stock.get("_watchlist_reason") else ""
    )
    sizing_note_html = (
        f'<div style="margin-top:6px;font-family:{sans};font-size:11px;color:#8A6D1E;">'
        f'<strong>Position sizing:</strong> {esc(stock.get("_confidence_sizing_note"))}</div>'
        if stock.get("_confidence_sizing_note") else ""
    )

    return f"""<div style="margin-top:{0 if idx == 0 else 22}px;">
<table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="border:1px solid #E7E4DC;border-radius:4px;overflow:hidden;border-collapse:collapse;">
<tr style="background:#14213D;"><td colspan="2" style="padding:9px 10px;font-family:{sans};font-size:11px;font-weight:700;color:#ffffff;text-transform:uppercase;letter-spacing:0.05em;">{idx + 1}. {name} <span style="color:#B08D57;">({ticker})</span>{badge_html}</td></tr>
{rows}
</table>
{watchlist_reason_html}
{sizing_note_html}
<div style="margin-top:10px;font-family:{sans};font-size:12px;color:#4A5063;line-height:1.65;"><strong style="color:#14213D;">Investment Rationale:</strong> {rationale}</div>
{_trade_execution_plan_html(stock, sans)}
</div>"""


def render_stock_table_html(stocks, tier_badge=None):
    sans = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
    if not stocks:
        return _no_qualifying_stock_html([])
    return "".join(
        _render_one_stock_card(stock, idx, sans, tier_badge=tier_badge) for idx, stock in enumerate(stocks)
    )


def _watchlist_tier_html(watchlist):
    """
    Renders the watchlist tier as its own clearly-labeled section --
    distinct from the strict-tier table above it (if any) and from the
    cross-run near-miss CSV (_log_watchlist/_load_and_recheck_watchlist),
    which is a separate mechanism for re-checking technical-only near-misses
    on a FUTURE run. This is THIS run's "close, but not a full qualifier"
    output.
    """
    if not watchlist:
        return ""
    sans = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
    header = (
        f'<div style="margin-top:26px;padding:10px 16px;background:#FBF3DC;border:1px solid #E9DCB0;'
        f'border-radius:4px;font-family:{sans};">'
        f'<div style="font-size:11px;font-weight:700;color:#5C4A1E;text-transform:uppercase;'
        f'letter-spacing:0.05em;">Watchlist Picks</div>'
        f'<div style="margin-top:4px;font-size:12px;color:#5C4A1E;line-height:1.55;">'
        "Close, but not full strict-tier qualifiers -- each one below failed at least one more "
        "core setup filter than the strict bar allows (or scored below the composite floor), "
        "with no absolute blocker (debt/equity, ROE, data integrity) against it. Treat these as "
        "worth a second look, not a vetted recommendation."
        "</div></div>"
    )
    return header + render_stock_table_html(watchlist, tier_badge="WATCHLIST")


def _dedupe_and_rank_watchlist(watchlist, max_shown=5):
    """De-dupes accumulated watchlist candidates (a ticker can surface more
    than once across this run's attempts) by ticker, keeping the
    best-scoring instance, then ranks by composite score (best first) and
    caps the count shown in the email."""
    best_by_ticker = {}
    for s in watchlist:
        ticker = (s.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        existing = best_by_ticker.get(ticker)
        if existing is None or (s.get("_composite_score") or 0) > (existing.get("_composite_score") or 0):
            best_by_ticker[ticker] = s
    ranked = scoring.rank_by_composite(list(best_by_ticker.values())) if scoring.USE_COMPOSITE_SCORE else list(best_by_ticker.values())
    return ranked[:max_shown]


def _choose_analysis_html(qualifying, candidates, rejected, watchlist=None, require_qualifying_stock=True):
    watchlist = watchlist or []
    if qualifying:
        return render_stock_table_html(qualifying) + _watchlist_tier_html(watchlist)
    if watchlist:
        # No strict qualifier this run, but one or more close candidates
        # exist -- surface them as the headline output instead of falling
        # all the way back to "no qualifying trade found" (see
        # WATCHLIST_TIER_ENABLED's docstring).
        sans = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
        note = (
            f'<div style="font-family:{sans};font-size:13px;color:#14213D;line-height:1.65;'
            f'padding:14px 16px;background:#F4F2ED;border-radius:4px;border:1px solid #E7E4DC;">'
            "<strong>No strict qualifying trade found this run.</strong> "
            f"{len(watchlist)} candidate(s) came close enough to land on the watchlist tier "
            "below instead -- see each one's \"why watchlist, not strict\" note for exactly "
            "what fell short."
            "</div>"
        )
        return note + _watchlist_tier_html(watchlist)
    if require_qualifying_stock:
        return _no_qualifying_stock_html(rejected or candidates)
    return render_stock_table_html(candidates)


def _no_qualifying_stock_html(rejected):
    """
    Rendered instead of a recommendation table when every candidate this run
    failed independent verification against its own strategy's mandatory
    filters (even after retries). Being honest that nothing qualified today is
    the correct output here -- forcing a pick that fails its own criteria is
    the bug this replaces.
    """
    sans = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
    uptrend_phrase = "50-day MA uptrend, " if REQUIRE_UPTREND_FILTER else ""
    out = (
        f'<div style="font-family:{sans};font-size:13px;color:#14213D;line-height:1.65;'
        f'padding:14px 16px;background:#F4F2ED;border-radius:4px;border:1px solid #E7E4DC;">'
        "<strong>No qualifying trade found for this run.</strong> Every candidate considered "
        f"failed too many of the five core setup filters ({uptrend_phrase}RSI {_fmt_num(MIN_RSI_OVERSOLD)}-{_fmt_num(MAX_RSI_OVERBOUGHT)}, "
        f"bullish MACD signal, &ge;{_fmt_num(MIN_GROWTH_YOY_PCT)}% YoY revenue/profit growth, &ge;1:{_fmt_num(MIN_RISK_REWARD)} risk:reward -- "
        f"fewer than {MIN_CORE_FILTERS_REQUIRED} of {CORE_FILTER_COUNT} passing, with no composite score reaching the "
        f"{_fmt_num(COMPOSITE_SCORE_CORE_FILTER_OVERRIDE)}/100 override) or a non-negotiable check (debt/equity, ROE, or a "
        "verifiable ticker) once checked against independently-verified data, even after retrying with feedback. No pick is "
        "being reported rather than recommending one that fails its own entry criteria. "
        + (
            "No candidate was even close enough to land on the watchlist tier either "
            f"(needs within {WATCHLIST_EXTRA_CORE_FILTER_SLACK} core filter(s) of the strict bar, no absolute blocker)."
            if WATCHLIST_TIER_ENABLED else
            "(WATCHLIST_TIER_ENABLED is off, so no near-miss tier was considered.)"
        )
        + "</div>"
    )
    if rejected:
        # Ranked by composite score (best near-misses first) rather than
        # just "whichever 6 were seen last" -- this is the diagnostic value
        # a pure pass/fail throws away: seeing how CLOSE the strongest
        # rejected candidates actually came (review item 6).
        ranked = scoring.rank_by_composite([s for s in rejected if "_composite_score" in s])
        unscored = [s for s in rejected if "_composite_score" not in s]  # e.g. Stage-1 fundamentals-only rejections
        display_list = (ranked + unscored)[-6:] if not ranked else ranked[:6]

        def _summary_for_rejected(s):
            notes = [n for n, sev in (s.get("_verification_notes") or []) if sev in ("hard", "nodata")]
            if not notes:
                return "no hard contradiction surfaced"
            concise = []
            for note in notes[:3]:
                text = note.split(" -- ", 1)[0]
                text = text.replace("contradicts the 'low debt-to-equity' requirement.", "failed low debt/equity")
                text = text.replace("contradicts the 'high/improving ROCE/ROE' requirement.", "failed ROE strength")
                text = text.replace("the prompt requires.", "missed the required threshold")
                concise.append(text)
            return "; ".join(concise)

        items = "".join(
            f'<div style="margin-top:8px;font-family:{sans};font-size:12px;color:#4A5063;">'
            f'<strong style="color:#14213D;">{html.escape(str(s.get("name") or s.get("ticker") or "Unnamed"))}</strong>'
            + (f' &mdash; composite {s["_composite_score"]:.1f}/100' if "_composite_score" in s else "")
            + f' &mdash; {html.escape(_summary_for_rejected(s))}'
            "</div>"
            for s in display_list
        )
        out += (
            f'<div style="margin-top:14px;padding-top:10px;border-top:1px solid #EDEAE2;'
            f'font-family:{sans};font-size:11px;font-weight:700;color:#8A8F9C;text-transform:uppercase;'
            f'letter-spacing:0.05em;">Candidates considered and rejected this run '
            f'(ranked by how close they came)</div>{items}'
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
        f'<a href="{html.escape(url, quote=True)}" style="color:#14213D;text-decoration:none;'
        f'border-bottom:1px solid #B08D57;">{html.escape(title)}</a></div>'
        for title, url in sources[:12]
        if url and url.strip().lower().startswith(("http://", "https://"))
    )
    if not items:
        return ""
    return f"""
        <div style="margin-top:14px;padding-top:12px;border-top:1px solid #EDEAE2;">
          <div style="font-family:{sans};font-size:11px;font-weight:700;color:#14213D;text-transform:uppercase;letter-spacing:0.06em;">Sources Consulted &nbsp;&middot;&nbsp; Live Web Search</div>
          {items}
        </div>
    """


def build_email_html(analysis_html, today_str, sources, used_live_search, adjustments_html=""):
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
<meta http-equiv="X-UA-Compatible" content="IE=edge">
<meta name="x-apple-disable-message-reformatting">
<meta name="format-detection" content="telephone=no, date=no, address=no, email=no">
<meta name="color-scheme" content="light">
<meta name="supported-color-schemes" content="light">
<title>Swing Trade Research Note</title>
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
<body style="margin:0;padding:0;background:#F2F0EC;font-family:{serif};color:#1B2233;">
  <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="background:#F2F0EC;width:100%;">
    <tr>
      <td align="center" style="padding:20px 16px;" class="email-padding">
        <table width="100%" cellpadding="0" cellspacing="0" role="presentation" class="email-container" style="max-width:680px;min-width:280px;background:#ffffff;border:1px solid #DAD5CB;border-radius:4px;overflow:hidden;">
          <tr>
            <td style="background:#14213D;padding:26px 28px 22px;" class="email-padding">
              <div style="font-family:{sans};font-size:10px;font-weight:700;letter-spacing:0.16em;text-transform:uppercase;color:#B08D57;">Market Intelligence &nbsp;&bull;&nbsp;</div>
              <h1 style="margin:8px 0 0;font-family:{serif};font-weight:400;font-size:23px;line-height:1.3;color:#ffffff;letter-spacing:0.01em;">Swing Trade Research Analysis</h1>
              <p style="margin:6px 0 0;font-family:{sans};font-size:12px;color:#B7BEC9;">3&ndash;5 Month Positioning</p>
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
              {adjustments_html}
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
    now_ist = datetime.now(ZoneInfo("UTC")).astimezone(ZoneInfo("Asia/Kolkata"))
    time_str = now_ist.strftime("%I:%M %p IST")
    note_label = "Weekly Swing Trade Research Note" if now_ist.weekday() == 0 else "Swing Trade Research Note"
    subject = f"{note_label} — {config.get_date_with_suffix(now_ist)} · {time_str}"

    return email_service.send_email(
        subject=subject,
        html_body=html_body
    )


def _require_live_or_abort(used_live, stage_label):
    if not used_live and os.getenv("REQUIRE_LIVE_DATA", "true").lower() == "true":
        msg = (
            f"Live web search was not used for {stage_label} this run (Groq's "
            "live-search model was unavailable or the backend fell back to "
            "Gemini/local), so the output would only reflect stale training-data "
            "prices/news. Aborting without sending an email. Set "
            "REQUIRE_LIVE_DATA=false to override and allow a clearly-labeled "
            "stale-data email instead."
        )
        log.error(msg)
        raise RuntimeError(msg)


def run():
    applied_adjustments = _apply_auto_adjustments()
    if applied_adjustments:
        log.info("Auto-adjusted thresholds this run based on rejection history:")
        for gname, old, new, reason in applied_adjustments:
            log.info(f"  {gname}: {old} -> {new}  ({reason})")

    today_str, is_monday, lookback_note = _run_context()
    if is_monday:
        log.info("Monday run detected -- widening news/catalyst lookback to the past week.")

    analysis_html = None
    sources = []
    used_live_search = False
    all_rejected = []
    all_watchlist = []  # accumulated across attempts -- see WATCHLIST_TIER_ENABLED
    qualifying = []  # default if every attempt "continue"s before ever assigning it
    stocks = []  # default if every attempt "continue"s before ever assigning it
    regime_override_active = False
    regime_ok, regime_detail = regime.check_market_regime()
    log.info(f"Market-regime check: {regime_detail}")

    # Every ticker rejected so far this run (at either the fundamentals or
    # technicals stage) -- excluded from later attempts' prompts AND
    # enforced here directly, since prompt instructions alone aren't
    # reliably honored by the model.
    seen_tickers = set()

    # Market-regime gate (review item 4): buying individual bullish setups
    # during a broad market downtrend has a materially worse hit rate than
    # the same setup in a bullish tape -- gate the WHOLE run behind the
    # index trend rather than letting a strong-looking individual stock
    # override a bearish broader market. This also saves every Stage 1/
    # Stage 2 LLM call this run would otherwise have spent, since there's
    # no point screening candidates the strategy itself says not to trust
    # right now. Set REQUIRE_MARKET_REGIME_FILTER=false (in
    # swing_trade_regime.py's env) to disable and fall back to the old
    # per-stock-only behavior.
    if regime.REQUIRE_MARKET_REGIME_FILTER and not regime_ok:
        if REGIME_OVERRIDE_ON_RS:
            # Explicit, opt-in exception (see REGIME_OVERRIDE_ON_RS's
            # docstring above): don't skip the run -- proceed, but every
            # candidate must additionally clear _verify_relative_strength_
            # override in place of the market-wide check being skipped.
            # This is what makes the existing fail-open behavior a
            # deliberate rule instead of an accident.
            regime_override_active = True
            log.warning(
                f"Market regime gate failed ({regime_detail.get('classification')}) -- "
                "REGIME_OVERRIDE_ON_RS is enabled, so proceeding with the scan instead of "
                "skipping the run. Every candidate this run must additionally clear a "
                "stricter relative-strength bar in place of the market-wide check."
            )
        else:
            log.warning(
                f"Market regime gate failed ({regime_detail.get('classification')}) -- "
                "skipping this run's scan entirely rather than screening individual "
                "stocks against an unfavorable broad-market backdrop."
            )
            analysis_html = (
                _no_qualifying_stock_html([])
                + regime.regime_note_html(regime_detail)
            )
            email_html = build_email_html(analysis_html, today_str, [], False, _adjustments_html(applied_adjustments))
            if os.getenv("DRY_RUN", "false").lower() == "true":
                with open("swing_trade_report.html", "w", encoding="utf-8") as f:
                    f.write(email_html)
                log.info("DRY_RUN enabled -- wrote swing_trade_report.html instead of emailing.")
                return
            send_swing_trade_email(email_html)
            return

    regime_softening = _regime_soften_growth_bar(regime_detail)
    if regime_softening:
        log.info(
            f"Regime-driven softening this run: MIN_GROWTH_YOY_PCT "
            f"{regime_softening[0]} -> {regime_softening[1]} ({regime_softening[2]})"
        )

    watchlist_graduates, watchlist_keep_rows = _load_and_recheck_watchlist()
    if watchlist_graduates:
        seen_tickers.update(
            (c.get("ticker") or "").strip().upper() for c in watchlist_graduates if c.get("ticker")
        )

    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        sectors = _sectors_for_attempt(attempt - 1)
        log.info(
            f"Attempt {attempt}/{MAX_GENERATION_ATTEMPTS} -- Stage 1 (fundamentals "
            f"screen) in sectors: {', '.join(sectors)}"
        )

        if USE_DETERMINISTIC_SCREEN:
            # Real, complete, zero-LLM-token screen of the static universe
            # for these sectors (see _deterministic_fundamentals_screen) --
            # no "did the model's search find anything" uncertainty, and no
            # 8-12-per-sector sampling limit.
            log.info(
                f"Attempt {attempt}: deterministic fundamentals screen "
                f"(no LLM call) across {', '.join(sectors)}."
            )
            fundamentally_qualified, rejected_fund = _deterministic_fundamentals_screen(sectors, seen_tickers)
            all_rejected.extend(rejected_fund)
            seen_tickers.update(
                (c.get("ticker") or "").strip().upper() for c in rejected_fund if c.get("ticker")
            )
            seen_tickers.update(
                (c.get("ticker") or "").strip().upper() for c in fundamentally_qualified if c.get("ticker")
            )
        else:
            growth_prompt = build_growth_screen_prompt(sectors, seen_tickers, today_str, lookback_note)
            # Larger budget than the default: this call has to search several
            # sectors, check 8-12 companies, and enumerate up to 20 candidates --
            # 1200 tokens was silently truncating that down to 1-2 stocks checked.
            # validate_fn requires the reply to actually parse as the expected
            # {"candidates": [...]} shape -- without this, a tier that ignores
            # the "ONLY raw JSON" instruction and returns commentary text still
            # counts as a chain "success", and the whole attempt gets treated as
            # zero candidates instead of falling through to a tier that might
            # have returned usable JSON.
            growth_analysis, growth_sources, growth_live = generate_analysis(
                growth_prompt, max_tokens=3000,
                validate_fn=lambda t: _parse_candidates_json(t) is not None,
            )

            if not growth_analysis:
                msg = (
                    "No LLM backend produced Stage 1 output (no GROQ_API_KEY/"
                    "GOOGLE_API_KEY set and local model unavailable/failed). "
                    "Aborting without sending an email."
                )
                log.error(msg)
                raise RuntimeError(msg)
            _require_live_or_abort(growth_live, "Stage 1 (fundamentals screen)")

            for s in growth_sources:
                if s not in sources:
                    sources.append(s)
            used_live_search = used_live_search or growth_live

            candidates = _parse_candidates_json(growth_analysis)
            if candidates is None:
                log.warning(
                    f"Attempt {attempt}: Stage 1 output could not be parsed as "
                    "candidate JSON -- treating as zero candidates for this attempt."
                )
                candidates = []

            candidates = [
                c for c in candidates
                if (c.get("ticker") or "").strip().upper() not in seen_tickers
            ]
            if not candidates:
                log.info(f"Attempt {attempt}: no new candidates found in {', '.join(sectors)}.")
                fundamentally_qualified, rejected_fund = [], []
            else:
                fundamentally_qualified, rejected_fund = _prefilter_by_fundamentals(candidates)
                all_rejected.extend(rejected_fund)
                seen_tickers.update(
                    (c.get("ticker") or "").strip().upper() for c in rejected_fund if c.get("ticker")
                )

        # Watchlist graduates (near-misses from a PREVIOUS run that now also
        # clear technicals) get first shot at this run's single Stage 2 call
        # rather than waiting for their own attempt -- fold them in once,
        # on the first attempt, rather than duplicating the whole Stage 2
        # block for them separately.
        if attempt == 1 and watchlist_graduates:
            log.info(
                f"{len(watchlist_graduates)} watchlist candidate(s) now clear both "
                "fundamentals and technicals -- adding to this attempt's Stage 2 batch: "
                + ", ".join(c.get("name") or c.get("ticker") or "?" for c in watchlist_graduates)
            )
            fundamentally_qualified = list(fundamentally_qualified) + watchlist_graduates
            seen_tickers.update(
                (c.get("ticker") or "").strip().upper() for c in watchlist_graduates if c.get("ticker")
            )

        if not fundamentally_qualified:
            log.info(
                f"Attempt {attempt}: no candidate from {', '.join(sectors)} "
                "passed independent fundamentals verification -- none reached Stage 2."
            )
            continue

        log.info(
            f"Attempt {attempt} -- Stage 2 (technicals) for "
            f"{len(fundamentally_qualified)} fundamentally-qualified candidate(s): "
            + ", ".join(c.get("name") or c.get("ticker") or "?" for c in fundamentally_qualified)
        )

        tech_prompt = build_technical_prompt(fundamentally_qualified, seen_tickers, today_str, lookback_note)
        tech_analysis, tech_sources, tech_live = generate_analysis(
            tech_prompt, max_tokens=2200,
            validate_fn=lambda t: _parse_analysis_json(t) is not None,
        )

        if not tech_analysis:
            msg = "No LLM backend produced Stage 2 output. Aborting without sending an email."
            log.error(msg)
            raise RuntimeError(msg)
        _require_live_or_abort(tech_live, "Stage 2 (technicals)")

        for s in tech_sources:
            if s not in sources:
                sources.append(s)
        used_live_search = used_live_search or tech_live

        stocks = _parse_analysis_json(tech_analysis)
        if stocks is None:
            log.warning(
                f"Attempt {attempt}: Stage 2 output could not be parsed as stock "
                "JSON -- treating as zero candidates for this attempt."
            )
            stocks = []

        # Guard against the model drifting outside the fundamentals-vetted
        # shortlist it was explicitly given.
        allowed = {(c.get("ticker") or "").strip().upper() for c in fundamentally_qualified}
        stocks = [s for s in stocks if (s.get("ticker") or "").strip().upper() in allowed]

        if not stocks:
            log.info(f"Attempt {attempt}: no candidate passed Stage 2 technicals.")
            continue

        stocks = _attach_live_prices(stocks)
        stocks = _attach_risk_plan(stocks)
        stocks = _compute_flexible_horizon(stocks)
        stocks = _verify_stock_claims(stocks, regime_override_active=regime_override_active)
        qualifying, watchlist, rejected = _split_qualifying(stocks)
        watchlist = _apply_confidence_sizing(watchlist)

        # When enabled, rank qualifying (and watchlist) candidates by
        # composite score before the concentration cap below picks which
        # one(s) to keep per sector -- otherwise "which pick survives the
        # cap" is just "whichever the model happened to list first"
        # (review item 6).
        if scoring.USE_COMPOSITE_SCORE:
            qualifying = scoring.rank_by_composite(qualifying)
            watchlist = scoring.rank_by_composite(watchlist)

        # Sector-concentration cap (review item 9): if this attempt's Stage 2
        # call returned multiple qualifying names, don't let several
        # same-sector picks (often the same underlying factor bet) all
        # through in one run. Applied to each tier independently so a
        # crowded sector doesn't crowd out the OTHER tier's cap budget.
        qualifying, dropped_for_concentration = risk.apply_sector_concentration_cap(qualifying)
        for d in dropped_for_concentration:
            d["_verification_notes"] = (d.get("_verification_notes") or []) + [(d["_concentration_note"], "hard")]
        all_rejected.extend(dropped_for_concentration)

        watchlist, watchlist_dropped_for_concentration = risk.apply_sector_concentration_cap(watchlist)
        for d in watchlist_dropped_for_concentration:
            d["_verification_notes"] = (d.get("_verification_notes") or []) + [(d["_concentration_note"], "hard")]
        all_rejected.extend(watchlist_dropped_for_concentration)

        all_rejected.extend(rejected)
        all_watchlist.extend(watchlist)
        seen_tickers.update(
            (s.get("ticker") or "").strip().upper() for s in rejected if s.get("ticker")
        )

        if qualifying or not REQUIRE_QUALIFYING_STOCK:
            break

        log.info(
            f"Attempt {attempt}/{MAX_GENERATION_ATTEMPTS}: {len(rejected)} "
            "candidate(s) failed independent verification at Stage 2, "
            f"{len(watchlist)} landed on the watchlist tier."
        )

    # De-dupe/rank the watchlist tier accumulated across every attempt this
    # run made (a ticker can resurface across attempts) before it's shown.
    all_watchlist = _dedupe_and_rank_watchlist(all_watchlist)

    if analysis_html is None and (qualifying or all_watchlist or not REQUIRE_QUALIFYING_STOCK):
        analysis_html = _choose_analysis_html(
            qualifying=qualifying,
            candidates=stocks,
            rejected=all_rejected,
            watchlist=all_watchlist,
            require_qualifying_stock=REQUIRE_QUALIFYING_STOCK,
        )

    # Log regardless of whether this run ultimately found a qualifying stock --
    # a threshold can be worth revisiting even in a run that DID produce a
    # pick, if other candidates that run missed it by a hair.
    _log_rejection_history(all_rejected, today_str)

    # Persist the near-miss watchlist: keep whatever survived recheck at the
    # top of this run (still-watching entries, minus graduates/stale/no-
    # longer-fundamentally-qualified ones _load_and_recheck_watchlist already
    # dropped), plus any NEW technical-only near-misses from this run's own
    # rejections. Rewritten wholesale rather than appended, since the recheck
    # step already decided which old rows survive.
    today_iso = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d")
    new_near_miss_rows = [
        {"date_added": today_iso, "ticker": (s.get("ticker") or "").strip(),
         "name": s.get("name") or (s.get("ticker") or "").strip(), "sector": s.get("sector") or ""}
        for s in all_rejected
        if (s.get("ticker") or "").strip() and _is_technical_only_near_miss(s)
    ]
    merged_by_ticker = {row["ticker"]: row for row in watchlist_keep_rows}
    for row in new_near_miss_rows:
        merged_by_ticker.setdefault(row["ticker"], row)  # keep the older date_added if already watching
    _rewrite_watchlist(list(merged_by_ticker.values()))

    if analysis_html is None:
        log.warning(
            f"All {MAX_GENERATION_ATTEMPTS} attempt(s) failed to produce a stock "
            "that passes its own strategy's mandatory filters against real data. "
            "Reporting 'no qualifying trade' instead of a contradicted pick."
        )
        analysis_html = _no_qualifying_stock_html(all_rejected)

    analysis_html = (analysis_html or "") + regime.regime_note_html(regime_detail)
    disclosures_html = (
        _adjustments_html(applied_adjustments)
        + _regime_softening_html(regime_softening)
        + _regime_override_html(regime_override_active, regime_detail)
    )
    email_html = build_email_html(analysis_html, today_str, sources, used_live_search, disclosures_html)

    # Outcome-tracking feedback loop (review item 7): log every stock this
    # run actually emailed so swing_trade_outcomes.py can later check real
    # price history and report whether it hit target, hit stop, or neither.
    # Runs regardless of DRY_RUN so local testing doesn't silently pollute
    # (or silently skip populating) the outcomes log inconsistently -- if
    # you don't want DRY_RUN runs logged, don't run with picks that qualify.
    if qualifying:
        for s in qualifying:
            outcomes.log_recommendation(s, today_str_iso=datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d"))

    if os.getenv("DRY_RUN", "false").lower() == "true":
        with open("swing_trade_report.html", "w", encoding="utf-8") as f:
            f.write(email_html)
        log.info("DRY_RUN enabled -- wrote swing_trade_report.html instead of emailing.")
        return

    send_swing_trade_email(email_html)


# -----------------------------------------------------------------------
# Backward-compat shim (PEP 562)
# -----------------------------------------------------------------------
# nifty_stock_controller.py and mutual_fund_controller.py import
# generate_synthesis (the non-live reasoning-only tier -- see its
# docstring in llm_backend.py) via
#   from controllers.swing_controller import (..., generate_synthesis, ...)
# This module doesn't define that name itself -- it only wraps
# llm_backend.generate_analysis() -- so forward any attribute this module
# doesn't define itself to llm_backend, where it actually lives. This
# also covers the now-removed _generate_local name and anything else
# callers may still expect at swing_trade_advisor.<name> from the
# pre-consolidation layout, without re-adding stale duplicate code here.
def __getattr__(name):
    if hasattr(llm_backend, name):
        return getattr(llm_backend, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    run()