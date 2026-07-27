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
import traceback
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

import smtplib
from email.mime.text import MIMEText

import pandas as pd

import stockpredictor  # reuses email config/credentials and helpers
import llm_backend  # shared LLM init + fallback chain (see llm_backend.py)
from compliance import build_compliance_block_html

# Independent-of-the-LLM risk/regime/scoring/tracking modules -- see each
# file's own docstring for the rationale. Kept as separate modules rather
# than folded into this already-1800-line file so each concern (position
# sizing, market regime, composite scoring, outcome tracking) is
# independently testable and can be reused by optionstrategy.py /
# stock_market_advisor.py later without dragging in this file's Stage-1/
# Stage-2 prompt-building code.
import swing_trade_risk as risk
import swing_trade_regime as regime
import swing_trade_scoring as scoring
import swing_trade_outcomes as outcomes
import swing_trade_universe as universe  # static ticker seed list for the deterministic screen (see below)

# -----------------------------
# Qualifying-stock gate
# -----------------------------
# The model's own JSON output is not trusted at face value: _verify_stock_claims
# independently checks every mandatory filter from the prompt (uptrend, RSI/MACD,
# growth thresholds, risk:reward minimum, debt/ROE) against real data. A stock
# with any "hard" contradiction -- i.e. one where the independent check actively
# disagrees with the model's claim, not just "couldn't be verified" -- fails its
# own strategy's entry criteria and must not be recommended. REQUIRE_QUALIFYING_STOCK
# (default true) enforces that; set to "false" to restore the old behavior of
# reporting every candidate regardless of contradictions.
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
        print(f"WARNING: env var {name}='{raw}' is not a valid number -- using default {default}.")
        return default


def _fmt_num(x):
    """Formats a threshold for display in prompts/messages without a
    trailing '.0' when it's a whole number (e.g. 20.0 -> '20', 17.5 -> '17.5').
    Moved up alongside _env_float for the same reason -- see that
    docstring."""
    return f"{x:g}"


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
# Regime-aware fundamentals softening -- bounded, transparent, this-run-only
# -----------------------------
# A fixed 20%/20% YoY growth bar is calibrated for an average market. In a
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
REGIME_SOFTEN_GROWTH_BAR = os.getenv("REGIME_SOFTEN_GROWTH_BAR", "false").lower() == "true"
REGIME_SOFTEN_MAX_PCT = _env_float("REGIME_SOFTEN_MAX_PCT", 15.0)

# -----------------------------
# Cross-run near-miss watchlist
# -----------------------------
# A stock that fully clears fundamentals but fails ONLY on a technical
# filter (e.g. RSI still falling, no MACD crossover yet) is exactly the
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
# the code enforced 2.0; MIN_RISK_REWARD's default below preserves that
# previously-enforced 2.0 rather than silently tightening it).
MIN_GROWTH_YOY_PCT = _env_float("MIN_GROWTH_YOY_PCT", 20.0)
MIN_RISK_REWARD = _env_float("MIN_RISK_REWARD", 2.0)
MAX_RSI_OVERBOUGHT = _env_float("MAX_RSI_OVERBOUGHT", 70.0)
MAX_DEBT_TO_EQUITY_PCT = _env_float("MAX_DEBT_TO_EQUITY_PCT", 100.0)
MIN_ROE_PCT = _env_float("MIN_ROE_PCT", 10.0)
# When false, the price-above-20/50-week-SMA uptrend requirement is dropped
# from both the prompt and the verifier entirely (RSI/MACD/growth/risk-reward
# filters still apply) -- use this if you want to consider pullback/basing
# setups, not just confirmed uptrends.
REQUIRE_UPTREND_FILTER = os.getenv("REQUIRE_UPTREND_FILTER", "true").lower() == "true"

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
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["date", "ticker", "name", "metric", "threshold", "actual_value", "margin_missed_by"]
            )
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
    except Exception as e:
        stockpredictor.log.warning(f"Could not write rejection history log: {e}")


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
    print(
        "WARNING: AUTO_ADJUST_THRESHOLDS=true but CONFIRM_AUTO_ADJUST_BACKTESTED is "
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
    "MAX_DEBT_TO_EQUITY_PCT": MAX_DEBT_TO_EQUITY_PCT,
    "MIN_ROE_PCT": MIN_ROE_PCT,
}
_GLOBAL_DIRECTION = {
    "MIN_GROWTH_YOY_PCT": "min",
    "MIN_RISK_REWARD": "min",
    "MAX_RSI_OVERBOUGHT": "max",
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
}


def _load_history_rows(log_path=REJECTION_HISTORY_LOG):
    path = Path(log_path)
    if not path.exists():
        return []
    rows = []
    try:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    r["threshold"] = float(r["threshold"])
                    r["actual_value"] = float(r["actual_value"])
                    r["margin_missed_by"] = float(r["margin_missed_by"])
                except (KeyError, ValueError, TypeError):
                    continue
                rows.append(r)
    except Exception as e:
        stockpredictor.log.warning(f"Could not read rejection history log '{log_path}': {e}")
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
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["date", "constant", "old_value", "new_value", "reason"])
            if write_header:
                writer.writeheader()
            for gname, old, new, reason in applied:
                writer.writerow({"date": today_str, "constant": gname, "old_value": old, "new_value": new, "reason": reason})
    except Exception as e:
        stockpredictor.log.warning(f"Could not write threshold adjustment log: {e}")


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
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["date_added", "ticker", "name", "sector"])
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
    except Exception as e:
        stockpredictor.log.warning(f"Could not write watchlist log: {e}")


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
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception as e:
        stockpredictor.log.warning(f"Could not read watchlist log '{log_path}': {e}")
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
        except (KeyError, ValueError):
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
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["date_added", "ticker", "name", "sector"])
            writer.writeheader()
            writer.writerows(rows)
    except Exception as e:
        stockpredictor.log.warning(f"Could not rewrite watchlist log: {e}")


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
SECTORS_PER_ATTEMPT = _env_int("SECTORS_PER_ATTEMPT", 6)


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
- Event-Driven: positioned to gain from a major near-term catalyst (regulatory approval, large contract win, demerger/spinoff, M&A arbitrage) with a clearly quantifiable price impact inside the 3-5 month window.
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
        f"never post BOTH >={_fmt_num(MIN_GROWTH_YOY_PCT)}% YoY revenue growth AND "
        f">={_fmt_num(MIN_GROWTH_YOY_PCT)}% YoY profit growth in "
        "the same quarter simultaneously -- their revenue base is too large for "
        "that pace of growth except in rare one-off years. Repeatedly proposing "
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
    return f"""STAGE 1 OF 2 -- FUNDAMENTALS SCREEN ONLY. Using the most current data as of {today_str}, {lookback_note}

Your ONLY job in this stage is to find genuine candidate stocks with an exceptionally strong recent quarter. Do NOT evaluate technicals (SMA/RSI/MACD), entry/exit levels, or risk:reward yet -- that happens in a separate Stage 2 call, only for whichever of your candidates survive independent verification against real financial data. Do not fabricate a growth figure -- if you cannot verify a real current number, omit the stock rather than guessing.

{_mega_large_cap_caution()}

{SOURCE_QUALITY_NOTE}

Search ONLY within these sectors this pass: {sector_list}. (Other sectors are covered in separate passes this run -- stay focused here so you search a handful of names deeply rather than many names thinly.)

Search Strategy:
- Do NOT rely on generic "stocks to buy today" / "top picks" / "5 shares to buy" listicle articles -- these recycle the same handful of already-popular, already-large names.
- Run systematic, screener-style searches per sector, biased toward small/mid-cap universes, e.g.: "<sector> smallcap midcap NSE BSE India net profit growth above {_fmt_num(MIN_GROWTH_YOY_PCT)}% YoY Q1 FY27", "<sector> India smallcap quarterly results revenue growth {_fmt_num(MIN_GROWTH_YOY_PCT)} percent {today_str}", "BSE SmallCap 250 <sector> results beat estimates", "BSE MidCap 150 <sector> Q1 FY27 results", company investor-relations / exchange-filing results pages, and sector-specific earnings roundups that explicitly cover smaller names, not just index heavyweights.
- Aim to individually check at least 8-12 distinct real companies across the sectors above (weighted toward small/mid-cap) before concluding few or none qualify.
{exclude_block}

Mandatory fundamentals filters (only stocks meeting ALL of these belong in your output):
- Latest quarter: net profit AND revenue growth both above {_fmt_num(MIN_GROWTH_YOY_PCT)}% YoY, with margin expansion.
- Low debt-to-equity (or strong asset quality for financials).
- High/improving ROCE/ROE, and check for promoter/institutional buying last quarter if known.

OUTPUT FORMAT -- respond with ONLY raw JSON matching the schema below, nothing else (no markdown, no code fences, no commentary before or after):

{{
  "candidates": [
    {{
      "name": "Stock name",
      "ticker": "Exact, currently-listed Yahoo Finance ticker (e.g. 'RELIANCE.NS') -- must be a real symbol you are confident is correct",
      "sector": "One of the sectors listed above",
      "market_cap_bucket": "One of: 'Small-cap', 'Mid-cap', 'Large-cap' -- your best estimate",
      "revenue_growth_yoy_pct": "e.g. 24.5",
      "profit_growth_yoy_pct": "e.g. 31.2",
      "why": "One sentence on the growth driver"
    }}
  ]
}}
List every genuine candidate you find meeting the bar in these sectors -- up to 20, and favor small/mid-cap names per the market-cap steering above. It is normal for very few (even zero) to qualify in a given sector slice -- return "candidates": [] rather than padding the list.
"""


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
    return f"""STAGE 2 OF 2 -- TECHNICALS, SENTIMENT, AND TRADE PLAN. Using the most current market data as of {today_str}, {lookback_note}

The stocks below have ALREADY passed independent fundamentals verification (>={_fmt_num(MIN_GROWTH_YOY_PCT)}% YoY revenue and profit growth, confirmed against real financial data, low debt, strong ROE) -- do not re-justify growth in your rationale beyond a brief mention. Your job now is to check EACH of these against the technical and sentiment filters below and build a trade plan ONLY for the ones that genuinely pass. It is fine, and expected, for some or all of these to fail on technicals (e.g. overbought, no MACD crossover) -- do not force a pick that doesn't qualify.

Fundamentally-qualified shortlist to evaluate (do not propose any stock outside this list):
{listing}
{exclude_block}

{STRATEGY_TYPES_BLOCK}

Mandatory technical / sentiment / risk filters:
- Technicals (3-5M view): {"price above 20-week AND 50-week SMA; " if REQUIRE_UPTREND_FILTER else ""}weekly RSI trending up but below {_fmt_num(MAX_RSI_OVERBOUGHT)}; bullish MACD crossover.
- Sentiment: recent positive catalysts (analyst upgrades, sector tailwinds, large orders) and supportive FII/DII activity.
- Risk/reward: minimum 1:{_fmt_num(MIN_RISK_REWARD)} based on your own proposed stop-loss and target -- before answering, verify the arithmetic yourself: risk_reward_ratio must equal (target1_pct / stop_loss_pct) to one decimal place; if it doesn't, adjust the target or stop-loss rather than reporting a mismatched ratio.
- Do not fabricate a price, RSI value, or news item -- if you cannot verify a real current number, say so in "rationale" instead of inventing one.

OUTPUT FORMAT -- respond with ONLY raw JSON matching the schema below, and nothing else (no markdown, no code fences, no commentary before or after). Plain text/numbers only (no HTML):

{{
  "stocks": [
    {{
      "name": "Stock name",
      "ticker": "Exact ticker from the shortlist above",
      "allocation_pct": "e.g. 5-10%",
      "entry_date": "Targeted entry date",
      "exit_date": "Expected exit date, 3-5 months from entry",
      "strategy_type": "Strategy name used",
      "confidence_score": "Conviction out of 10 (e.g. 8.8) -- weigh fundamental + technical + sentiment strength together",
      "risk_level": "One word: 'Medium' or 'High'",
      "key_catalysts": "2-4 near-term catalysts, comma-separated, e.g. 'Earnings, Order Win, Sector Upgrade'",
      "risk_reward_ratio": "e.g. '1 : 2.5' -- must arithmetically match stop_loss_pct and target1_pct below",
      "upside_target_pct": "Favourable % for 3-5 months",
      "stop_loss_pct": "Risk % (Stop-Loss)",
      "target1_pct": "Expected Profit % (T1)",
      "target2_pct": "Expected Profit % (T2), optional",
      "top_buyers": "Recent FII/DII activity, if known",
      "broker_recommendations": "e.g. 'Buy' with target X from a named brokerage, if known",
      "rationale": "Two to three sentences covering technical + sentiment rationale (fundamentals already confirmed) and the key risk to watch"
    }}
  ]
}}
Only include a stock if it truly satisfies every mandatory technical/sentiment/risk filter with real, verifiable current data. Return "stocks": [] if none from the shortlist genuinely pass right now -- do not force a pick.
"""


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
        stockpredictor.log.warning(f"Tavily search failed for query '{query}': {e}")
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
    compound-mini -> Tavily+synthesis -> Gemini -> Mistral -> local); that
    logic now lives once in llm_backend.py, shared with main.py's AI Stocks
    Story and (via this function) optionstrategy.py.

    Only the swing-trade-specific piece stays here: which Tavily queries to
    run for the "grounded" tier (see _gather_tavily_context above).

    extra_context_queries: optional list of extra search terms to fold into
    the Tavily grounding tier -- pass the specific stock/fund names a batch
    call is about here so that tier, if it ends up serving the call, is
    actually searching for them instead of only generic market terms.

    validate_fn: optional text -> bool, forwarded to llm_backend.generate_analysis.
    Without this, the chain's default validator only checks "non-empty" --
    so a tier that ignores the "respond with ONLY raw JSON" instruction and
    returns commentary/preamble text still counts as a "success", and the
    chain never falls through to a later tier (Gemini/Mistral/local) that
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
        df = stockpredictor.fetch_data(ticker)
        latest_close = stockpredictor._safe_float(df.iloc[-1].get("close"))
        if latest_close is None:
            return None, None
        market = stockpredictor.classify_market(ticker)
        currency_symbol = "₹" if market == "India" else "$"
        return latest_close, currency_symbol
    except Exception as e:
        stockpredictor.log.warning(f"Could not fetch live price for '{ticker}': {e}")
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
        plan = risk.compute_volatility_adjusted_plan(stock.get("ticker"), min_risk_reward=MIN_RISK_REWARD)
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


def _fetch_weekly_technicals(ticker):
    try:
        df = stockpredictor.fetch_data(ticker)
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

        weekly_close = close.resample("W").last().dropna()
        if len(weekly_close) < 55:
            result = {"insufficient_history": True, "weeks_available": len(weekly_close)}
            if len(weekly_close) >= 20:
                result["sma20w"] = round(float(weekly_close.rolling(20).mean().iloc[-1]), 2)
                result["latest_close"] = round(float(weekly_close.iloc[-1]), 2)
            return result

        sma20w = weekly_close.rolling(20).mean()
        sma50w = weekly_close.rolling(50).mean()

        delta = weekly_close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, float("nan"))
        rsi = 100 - (100 / (1 + rs))

        ema12 = weekly_close.ewm(span=12, adjust=False).mean()
        ema26 = weekly_close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()

        rsi_now = rsi.iloc[-1]
        rsi_prev = rsi.iloc[-3] if len(rsi) > 3 else None

        return {
            "insufficient_history": False,
            "latest_close": round(float(weekly_close.iloc[-1]), 2),
            "sma20w": round(float(sma20w.iloc[-1]), 2),
            "sma50w": round(float(sma50w.iloc[-1]), 2),
            "rsi14w": round(float(rsi_now), 1) if pd.notna(rsi_now) else None,
            "rsi14w_prev": round(float(rsi_prev), 1) if rsi_prev is not None and pd.notna(rsi_prev) else None,
            "macd": round(float(macd_line.iloc[-1]), 3),
            "macd_signal": round(float(signal_line.iloc[-1]), 3),
        }
    except Exception as e:
        stockpredictor.log.warning(f"Could not compute weekly technicals for '{ticker}': {e}")
        return None


def _verify_technicals(stock):
    """Returns a list of (note_text, severity) tuples -- see _verify_risk_reward."""
    ticker = (stock.get("ticker") or "").strip()
    if not ticker:
        return [("No ticker provided -- technicals could not be verified.", "soft")]

    tech = _fetch_weekly_technicals(ticker)
    if tech is None:
        return [("Technicals could not be independently verified (price history fetch failed).", "nodata")]
    if tech.get("insufficient_history"):
        notes = [(
            f"Only {tech.get('weeks_available')} weeks of price history available -- "
            "not enough to verify the 50-week SMA, RSI, or MACD; those claims are "
            "unverified.", "soft"
        )]
        # Even with <55 weeks we may still have enough for the 20-week SMA
        # (_fetch_weekly_technicals populates it once >=20 weeks are available)
        # -- that much is checkable, so don't leave it as a blanket "unverified".
        # Only enforced when REQUIRE_UPTREND_FILTER is true (default).
        if REQUIRE_UPTREND_FILTER and tech.get("sma20w") is not None and tech.get("latest_close") is not None:
            if tech["latest_close"] < tech["sma20w"]:
                notes.append((
                    f"Price ({tech['latest_close']}) is BELOW the 20-week SMA "
                    f"({tech['sma20w']}) -- contradicts the required uptrend filter "
                    "(this much is checkable even with limited history).", "hard"
                ))
        return notes

    notes = []
    price = tech["latest_close"]
    if REQUIRE_UPTREND_FILTER:
        if price < tech["sma20w"]:
            notes.append((f"Price ({price}) is BELOW the 20-week SMA ({tech['sma20w']}) -- contradicts the required uptrend filter.", "hard"))
        if price < tech["sma50w"]:
            notes.append((f"Price ({price}) is BELOW the 50-week SMA ({tech['sma50w']}) -- contradicts the required uptrend filter.", "hard"))

    if tech["rsi14w"] is not None:
        if tech["rsi14w"] >= MAX_RSI_OVERBOUGHT:
            notes.append((f"Weekly RSI is {tech['rsi14w']} (>={_fmt_num(MAX_RSI_OVERBOUGHT)}, overbought) -- contradicts the 'RSI below {_fmt_num(MAX_RSI_OVERBOUGHT)}' requirement.", "hard"))
        if tech.get("rsi14w_prev") is not None and tech["rsi14w"] < tech["rsi14w_prev"]:
            notes.append((f"Weekly RSI is falling ({tech['rsi14w_prev']} to {tech['rsi14w']}), not rising as the strategy requires.", "soft"))

    if tech["macd"] < tech["macd_signal"]:
        notes.append(("MACD line is currently below its signal line -- no bullish crossover in effect right now.", "hard"))

    return notes


def _fetch_fundamentals(ticker):
    """
    Fetches point-in-time fundamentals (debt/equity, ROE) plus a quarterly
    revenue/net-income series directly via yfinance -- independent of
    whatever the LLM claimed about the company's financials. Returns a
    dict (fields may be None if unavailable) or None if the fetch fails
    entirely / yfinance isn't installed.
    """
    ticker = (ticker or "").strip()
    if not ticker:
        return None
    try:
        import yfinance as yf
    except ImportError:
        stockpredictor.log.warning("yfinance not installed -- fundamentals verification skipped.")
        return None
    try:
        yt = yf.Ticker(ticker)
        info = yt.info or {}
        result = {
            "debt_to_equity": info.get("debtToEquity"),
            "roe": info.get("returnOnEquity"),
            "revenue_growth_yoy": None,
            "profit_growth_yoy": None,
        }
        try:
            qf = yt.quarterly_financials  # rows = line items, cols = quarter-end dates, most-recent first
            if qf is not None and not qf.empty and qf.shape[1] >= 2:
                year_ago_idx = _find_year_ago_index(list(qf.columns))
                if year_ago_idx is None:
                    stockpredictor.log.warning(
                        f"'{ticker}': no quarterly_financials column falls within "
                        "45 days of 1 year before the latest quarter -- skipping "
                        "YoY growth calc (irregular/missing quarterly history) "
                        "rather than comparing against the wrong period."
                    )
                else:
                    revenue_row = next((r for r in qf.index if "total revenue" in r.lower()), None)
                    income_row = next((r for r in qf.index if r.lower() == "net income"), None)
                    if revenue_row is not None:
                        latest, year_ago = qf.loc[revenue_row].iloc[0], qf.loc[revenue_row].iloc[year_ago_idx]
                        if year_ago and year_ago != 0 and pd.notna(latest) and pd.notna(year_ago):
                            result["revenue_growth_yoy"] = round(((latest - year_ago) / abs(year_ago)) * 100, 1)
                    if income_row is not None:
                        latest, year_ago = qf.loc[income_row].iloc[0], qf.loc[income_row].iloc[year_ago_idx]
                        if year_ago and year_ago != 0 and pd.notna(latest) and pd.notna(year_ago):
                            result["profit_growth_yoy"] = round(((latest - year_ago) / abs(year_ago)) * 100, 1)
        except Exception as e:
            stockpredictor.log.warning(f"Could not compute quarterly growth for '{ticker}': {e}")
        return result
    except Exception as e:
        stockpredictor.log.warning(f"Could not fetch fundamentals for '{ticker}': {e}")
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
    (low debt-to-equity, high/improving ROE, >=20% YoY revenue and
    profit growth) against real data instead of trusting the model's
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
        roe_pct = roe * 100 if abs(roe) <= 1 else roe
        if roe_pct < MIN_ROE_PCT:
            notes.append((f"ROE is {roe_pct:.1f}% -- weak (below the {_fmt_num(MIN_ROE_PCT)}% threshold), contradicts the 'high/improving ROCE/ROE' requirement.", "hard"))
    else:
        notes.append(("ROE not available from data provider -- unverified.", "soft"))

    rev_g = data.get("revenue_growth_yoy")
    if rev_g is not None:
        if rev_g < MIN_GROWTH_YOY_PCT:
            notes.append((f"Revenue growth YoY is {rev_g}% -- below the {_fmt_num(MIN_GROWTH_YOY_PCT)}% threshold the prompt requires.", "hard"))
    else:
        notes.append(("Revenue YoY growth could not be computed (insufficient quarterly history from data provider).", "soft"))

    profit_g = data.get("profit_growth_yoy")
    if profit_g is not None:
        if profit_g < MIN_GROWTH_YOY_PCT:
            notes.append((f"Net profit growth YoY is {profit_g}% -- below the {_fmt_num(MIN_GROWTH_YOY_PCT)}% threshold the prompt requires.", "hard"))
    else:
        notes.append(("Net profit YoY growth could not be computed (insufficient quarterly history from data provider).", "soft"))

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
    this stage exists specifically to confirm >=20% YoY growth, and "no data
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
    for name, ticker, sector, bucket in universe.tickers_for_sectors(sectors):
        ticker_u = ticker.strip().upper()
        if ticker_u in exclude_tickers or ticker_u in seen_this_call:
            continue
        seen_this_call.add(ticker_u)

        stub = {"name": name, "ticker": ticker, "sector": sector, "market_cap_bucket": bucket}
        notes = _verify_fundamentals(stub)
        blocking = [n for n, sev in notes if sev in ("hard", "nodata")]

        if blocking:
            record = dict(stub)
            record["_verification_notes"] = notes
            rejected.append(record)
            continue

        data = _fetch_fundamentals(ticker) or {}
        qualified.append({
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
        })
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
        notes.append((f"Upside target of {upside}% in a 3-5 month window is unusually aggressive -- treat as a stretch case, not a base case.", "soft"))

    risk_level = (stock.get("risk_level") or "").strip().lower()
    if risk_level not in ("medium", "high"):
        notes.append((f"Risk level '{stock.get('risk_level')}' is not one of the expected 'Medium'/'High' values.", "soft"))

    return notes


def _verify_stock_claims(stocks):
    for stock in stocks:
        rr_notes = _verify_risk_reward(stock)
        tech_notes = _verify_technicals(stock)
        fund_notes = _verify_fundamentals(stock)
        sanity_notes = _verify_sanity_bounds(stock)

        price_missing = not stock.get("current_price_display")
        tech_no_data = any(sev == "nodata" for _, sev in tech_notes)
        fund_no_data = any(sev == "nodata" for _, sev in fund_notes)

        notes = rr_notes + tech_notes + fund_notes + sanity_notes
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


def _split_qualifying(stocks):
    """
    Splits verified stocks into (qualifying, rejected). A stock qualifies only if
    it has zero 'hard' contradictions -- i.e. nothing in its own strategy's
    mandatory filters (uptrend, RSI/MACD, growth thresholds, risk:reward minimum,
    debt/ROE) was independently found to be false. 'Soft' notes (couldn't be
    verified) are still disclosed in the report but don't block a recommendation.
    """
    qualifying, rejected = [], []
    for s in stocks:
        (rejected if _hard_contradictions(s) else qualifying).append(s)
    return qualifying, rejected


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


def _render_one_stock_card(stock, idx, sans):
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
        row("Exit Date (Expected)", "exit_date"),
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

    return f"""<div style="margin-top:{0 if idx == 0 else 22}px;">
<table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="border:1px solid #E7E4DC;border-radius:4px;overflow:hidden;border-collapse:collapse;">
<tr style="background:#14213D;"><td colspan="2" style="padding:9px 10px;font-family:{sans};font-size:11px;font-weight:700;color:#ffffff;text-transform:uppercase;letter-spacing:0.05em;">{idx + 1}. {name} <span style="color:#B08D57;">({ticker})</span></td></tr>
{rows}
</table>
<div style="margin-top:10px;font-family:{sans};font-size:12px;color:#4A5063;line-height:1.65;"><strong style="color:#14213D;">Investment Rationale:</strong> {rationale}</div>
{_trade_execution_plan_html(stock, sans)}
</div>"""


def render_stock_table_html(stocks):
    sans = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
    if not stocks:
        return _no_qualifying_stock_html([])
    return "".join(
        _render_one_stock_card(stock, idx, sans) for idx, stock in enumerate(stocks)
    )



def _no_qualifying_stock_html(rejected):
    """
    Rendered instead of a recommendation table when every candidate this run
    failed independent verification against its own strategy's mandatory
    filters (even after retries). Being honest that nothing qualified today is
    the correct output here -- forcing a pick that fails its own criteria is
    the bug this replaces.
    """
    sans = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
    uptrend_phrase = "uptrend, " if REQUIRE_UPTREND_FILTER else ""
    out = (
        f'<div style="font-family:{sans};font-size:13px;color:#14213D;line-height:1.65;'
        f'padding:14px 16px;background:#F4F2ED;border-radius:4px;border:1px solid #E7E4DC;">'
        "<strong>No qualifying trade found for this run.</strong> Every candidate considered "
        f"failed at least one of the strategy's mandatory filters ({uptrend_phrase}rising RSI below {_fmt_num(MAX_RSI_OVERBOUGHT)}, "
        f"bullish MACD crossover, &ge;{_fmt_num(MIN_GROWTH_YOY_PCT)}% YoY revenue/profit growth, or &ge;1:{_fmt_num(MIN_RISK_REWARD)} risk:reward) once "
        "checked against independently-verified data, even after retrying with feedback. No pick is "
        "being reported rather than recommending one that fails its own entry criteria."
        "</div>"
    )
    if rejected:
        # Ranked by composite score (best near-misses first) rather than
        # just "whichever 6 were seen last" -- this is the diagnostic value
        # a pure pass/fail throws away: seeing how CLOSE the strongest
        # rejected candidates actually came (review item 6).
        ranked = scoring.rank_by_composite([s for s in rejected if "_composite_score" in s])
        unscored = [s for s in rejected if "_composite_score" not in s]  # e.g. Stage-1 fundamentals-only rejections
        display_list = (ranked + unscored)[-6:] if not ranked else ranked[:6]
        items = "".join(
            f'<div style="margin-top:8px;font-family:{sans};font-size:12px;color:#4A5063;">'
            f'<strong style="color:#14213D;">{html.escape(str(s.get("name") or s.get("ticker") or "Unnamed"))}</strong>'
            + (f' &mdash; composite {s["_composite_score"]:.1f}/100' if "_composite_score" in s else "")
            + f' &mdash; rejected: {html.escape("; ".join(n for n, sev in (s.get("_verification_notes") or []) if sev in ("hard", "nodata")) or "unspecified")}'
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
<title>Swing Trade Research Note</title>
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
              <div style="font-family:{sans};font-size:10px;font-weight:700;letter-spacing:0.16em;text-transform:uppercase;color:#B08D57;">Market Intelligence &nbsp;&bull;&nbsp; Idea Generation</div>
              <h1 style="margin:8px 0 0;font-family:{serif};font-weight:400;font-size:23px;line-height:1.3;color:#ffffff;letter-spacing:0.01em;">Swing Trade Research Note</h1>
              <p style="margin:6px 0 0;font-family:{sans};font-size:12px;color:#B7BEC9;">3&ndash;5 Month Positioning Horizon</p>
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
    if not all([stockpredictor.EMAIL_FROM, stockpredictor.EMAIL_PASSWORD, stockpredictor.EMAIL_TO]):
        stockpredictor.log.error(
            "Email credentials not found. Please set EMAIL_FROM, EMAIL_PASSWORD, "
            "and EMAIL_TO (the same env vars main.py uses)."
        )
        return False

    to_recipients = stockpredictor.parse_email_list(stockpredictor.EMAIL_TO)
    cc_recipients = stockpredictor.parse_email_list(getattr(stockpredictor, "EMAIL_CC", "") or "")

    if not to_recipients:
        stockpredictor.log.error("No valid TO recipients found in EMAIL_TO.")
        return False

    now_ist = datetime.now(ZoneInfo("UTC")).astimezone(ZoneInfo("Asia/Kolkata"))
    time_str = now_ist.strftime("%I:%M %p IST")
    note_label = "Weekly Swing Trade Research Note" if now_ist.weekday() == 0 else "Swing Trade Research Note"
    subject = f"{note_label} — {stockpredictor.get_date_with_suffix(now_ist)} · {time_str}"

    msg = MIMEText(html_body, "html")
    msg["Subject"] = subject
    msg["From"] = stockpredictor.EMAIL_FROM
    msg["To"] = ", ".join(to_recipients)
    if cc_recipients:
        msg["Cc"] = ", ".join(cc_recipients)

    all_recipients = to_recipients + cc_recipients

    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
            server.starttls()
            server.login(stockpredictor.EMAIL_FROM, stockpredictor.EMAIL_PASSWORD)
            server.sendmail(stockpredictor.EMAIL_FROM, all_recipients, msg.as_string())
        stockpredictor.log.info("Swing trade email sent successfully.")
        return True
    except smtplib.SMTPAuthenticationError:
        stockpredictor.log.error(
            "SMTP Authentication Error: check EMAIL_FROM/EMAIL_PASSWORD "
            "(use a Gmail App Password, not the account password)."
        )
    except Exception as e:
        stockpredictor.log.error(f"Failed to send swing trade email: {e}")
        traceback.print_exc()
    return False


def _require_live_or_abort(used_live, stage_label):
    if not used_live and os.getenv("REQUIRE_LIVE_DATA", "true").lower() == "true":
        stockpredictor.log.error(
            f"Live web search was not used for {stage_label} this run (Groq's "
            "live-search model was unavailable or the backend fell back to "
            "Gemini/local), so the output would only reflect stale training-data "
            "prices/news. Aborting without sending an email. Set "
            "REQUIRE_LIVE_DATA=false to override and allow a clearly-labeled "
            "stale-data email instead."
        )
        sys.exit(1)


def run():
    applied_adjustments = _apply_auto_adjustments()
    if applied_adjustments:
        stockpredictor.log.info("Auto-adjusted thresholds this run based on rejection history:")
        for gname, old, new, reason in applied_adjustments:
            stockpredictor.log.info(f"  {gname}: {old} -> {new}  ({reason})")

    today_str, is_monday, lookback_note = _run_context()
    if is_monday:
        stockpredictor.log.info("Monday run detected -- widening news/catalyst lookback to the past week.")

    analysis_html = None
    sources = []
    used_live_search = False
    all_rejected = []
    qualifying = []  # default if every attempt "continue"s before ever assigning it
    regime_ok, regime_detail = regime.check_market_regime()
    stockpredictor.log.info(f"Market-regime check: {regime_detail}")

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
        stockpredictor.log.warning(
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
            with open("swing_trade_report.html", "w") as f:
                f.write(email_html)
            stockpredictor.log.info("DRY_RUN enabled -- wrote swing_trade_report.html instead of emailing.")
            return
        send_swing_trade_email(email_html)
        return

    regime_softening = _regime_soften_growth_bar(regime_detail)
    if regime_softening:
        stockpredictor.log.info(
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
        stockpredictor.log.info(
            f"Attempt {attempt}/{MAX_GENERATION_ATTEMPTS} -- Stage 1 (fundamentals "
            f"screen) in sectors: {', '.join(sectors)}"
        )

        if USE_DETERMINISTIC_SCREEN:
            # Real, complete, zero-LLM-token screen of the static universe
            # for these sectors (see _deterministic_fundamentals_screen) --
            # no "did the model's search find anything" uncertainty, and no
            # 8-12-per-sector sampling limit.
            stockpredictor.log.info(
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
                stockpredictor.log.error(
                    "No LLM backend produced Stage 1 output (no GROQ_API_KEY/"
                    "GOOGLE_API_KEY set and local model unavailable/failed). "
                    "Aborting without sending an email."
                )
                sys.exit(1)
            _require_live_or_abort(growth_live, "Stage 1 (fundamentals screen)")

            for s in growth_sources:
                if s not in sources:
                    sources.append(s)
            used_live_search = used_live_search or growth_live

            candidates = _parse_candidates_json(growth_analysis)
            if candidates is None:
                stockpredictor.log.warning(
                    f"Attempt {attempt}: Stage 1 output could not be parsed as "
                    "candidate JSON -- treating as zero candidates for this attempt."
                )
                candidates = []

            candidates = [
                c for c in candidates
                if (c.get("ticker") or "").strip().upper() not in seen_tickers
            ]
            if not candidates:
                stockpredictor.log.info(f"Attempt {attempt}: no new candidates found in {', '.join(sectors)}.")
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
            stockpredictor.log.info(
                f"{len(watchlist_graduates)} watchlist candidate(s) now clear both "
                "fundamentals and technicals -- adding to this attempt's Stage 2 batch: "
                + ", ".join(c.get("name") or c.get("ticker") or "?" for c in watchlist_graduates)
            )
            fundamentally_qualified = list(fundamentally_qualified) + watchlist_graduates
            seen_tickers.update(
                (c.get("ticker") or "").strip().upper() for c in watchlist_graduates if c.get("ticker")
            )

        if not fundamentally_qualified:
            stockpredictor.log.info(
                f"Attempt {attempt}: no candidate from {', '.join(sectors)} "
                "passed independent fundamentals verification -- none reached Stage 2."
            )
            continue

        stockpredictor.log.info(
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
            stockpredictor.log.error(
                "No LLM backend produced Stage 2 output. Aborting without sending an email."
            )
            sys.exit(1)
        _require_live_or_abort(tech_live, "Stage 2 (technicals)")

        for s in tech_sources:
            if s not in sources:
                sources.append(s)
        used_live_search = used_live_search or tech_live

        stocks = _parse_analysis_json(tech_analysis)
        if stocks is None:
            stockpredictor.log.warning(
                f"Attempt {attempt}: Stage 2 output could not be parsed as stock "
                "JSON -- treating as zero candidates for this attempt."
            )
            stocks = []

        # Guard against the model drifting outside the fundamentals-vetted
        # shortlist it was explicitly given.
        allowed = {(c.get("ticker") or "").strip().upper() for c in fundamentally_qualified}
        stocks = [s for s in stocks if (s.get("ticker") or "").strip().upper() in allowed]

        if not stocks:
            stockpredictor.log.info(f"Attempt {attempt}: no candidate passed Stage 2 technicals.")
            continue

        stocks = _attach_live_prices(stocks)
        stocks = _attach_risk_plan(stocks)
        stocks = _verify_stock_claims(stocks)
        qualifying, rejected = _split_qualifying(stocks)

        # When enabled, rank qualifying candidates by composite score before
        # the concentration cap below picks which one(s) to keep per sector --
        # otherwise "which pick survives the cap" is just "whichever the
        # model happened to list first" (review item 6).
        if scoring.USE_COMPOSITE_SCORE:
            qualifying = scoring.rank_by_composite(qualifying)

        # Sector-concentration cap (review item 9): if this attempt's Stage 2
        # call returned multiple qualifying names, don't let several
        # same-sector picks (often the same underlying factor bet) all
        # through in one run.
        qualifying, dropped_for_concentration = risk.apply_sector_concentration_cap(qualifying)
        for d in dropped_for_concentration:
            d["_verification_notes"] = (d.get("_verification_notes") or []) + [(d["_concentration_note"], "hard")]
        all_rejected.extend(dropped_for_concentration)
        all_rejected.extend(rejected)
        seen_tickers.update(
            (s.get("ticker") or "").strip().upper() for s in rejected if s.get("ticker")
        )

        if qualifying or not REQUIRE_QUALIFYING_STOCK:
            analysis_html = render_stock_table_html(qualifying or stocks)
            break

        stockpredictor.log.info(
            f"Attempt {attempt}/{MAX_GENERATION_ATTEMPTS}: {len(rejected)} "
            "candidate(s) failed independent verification at Stage 2."
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
        stockpredictor.log.warning(
            f"All {MAX_GENERATION_ATTEMPTS} attempt(s) failed to produce a stock "
            "that passes its own strategy's mandatory filters against real data. "
            "Reporting 'no qualifying trade' instead of a contradicted pick."
        )
        analysis_html = _no_qualifying_stock_html(all_rejected)

    analysis_html = (analysis_html or "") + regime.regime_note_html(regime_detail)
    disclosures_html = _adjustments_html(applied_adjustments) + _regime_softening_html(regime_softening)
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
        with open("swing_trade_report.html", "w") as f:
            f.write(email_html)
        stockpredictor.log.info("DRY_RUN enabled -- wrote swing_trade_report.html instead of emailing.")
        return

    send_swing_trade_email(email_html)


# -----------------------------------------------------------------------
# Backward-compat shim (PEP 562)
# -----------------------------------------------------------------------
# stock_market_advisor.py and mutual_fund_advisor.py still do
#   from swing_trade_advisor import (..., _generate_local, ...)
# from before the LLM fallback chain (model init, groq/gemini clients,
# _generate_local, etc.) was consolidated into llm_backend.py. This
# module no longer defines those names itself -- it only calls
# llm_backend.generate_analysis() -- so a plain `from swing_trade_advisor
# import _generate_local` now raises ImportError.
#
# Rather than either breaking those two callers or re-adding a stale
# duplicate implementation here, forward any attribute this module
# doesn't define itself to llm_backend, where it actually lives. This
# covers _generate_local specifically (the one causing the ImportError)
# as well as anything else callers may still expect at
# swing_trade_advisor.<name> from the pre-consolidation layout.
def __getattr__(name):
    if hasattr(llm_backend, name):
        return getattr(llm_backend, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    run()