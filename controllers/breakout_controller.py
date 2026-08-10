"""
controllers/breakout_controller.py

Daily chart-pattern breakout screener over the NIFTY 500 universe.
Separate from wealth_controller.py's monthly SIP checkup and from
stock_controller.py's daily briefing on the fixed watchlist -- this one
scans a much wider universe (NIFTY 500) purely on price/volume
mechanics, no AI narrative involved, and only surfaces a signal after
backtesting that exact pattern against the stock's own history.

Pipeline, per run:
  1. Get today's NIFTY 500 symbol list (utils.nse_data).
  2. Get today's NSE bhavcopy for a same-day price/volume cross-check
     (fail-soft -- proceeds without it if NSE is unreachable, and says
     so in the email).
  3. Pull ~2 years of daily OHLCV per symbol from yfinance, in batches.
  4. Per symbol: data-quality gate (utils.nse_data.data_quality_check) ->
     skip if it fails outright, else carry any caution notes forward.
  5. Run every pattern detector against today's bar
     (utils.breakout_patterns.scan_all_patterns).
  6. For every pattern that fired, replay that SAME detector across the
     symbol's own trailing history (utils.breakout_backtest) to get a
     live hit-rate/avg-return read -- this is what "confirmed" means
     below, never a canned/pre-baked statistic.
  7. Split into "Confirmed Breakouts" (backtest cleared the bar) and
     "Unconfirmed / Watch List" (pattern fired today but backtest sample
     was too thin or the historical hit-rate was weak) -- nothing is
     silently dropped, everything shown is labeled for what it is.
  8. Layer an Entry Classification on every signal (utils.entry_classification):
     not every breakout is an immediate buy. Each row is tagged Fresh
     Breakout, Retest, or (for symbols that haven't broken out yet but are
     close) Near Breakout -- with a concrete entry zone or trigger price.
     Retest is flagged as the preferred entry where it applies. This is a
     generic price-structure overlay (trailing-high resistance), independent
     of whichever specific pattern detector fired.
  9. For any row with an exact entry price (Fresh Breakout / Retest),
     compute three technical price targets (utils.breakout_targets):
     Target 1 (nearest prior swing-high resistance), Target 2 (the
     pattern's own measured move, e.g. cup depth or flagpole height
     projected from the breakout), and Target 3 (Fibonacci extension of
     that same base, or a further proven swing high, whichever is
     farther). This turns the backtest's average-return number into
     concrete price levels.
  10. Compute Risk/Reward for every row that has an entry price, a stop
      loss, and at least one target: reward is measured against the
      NEAREST available target, risk against the stop from step 8. This
      is a CORE FILTER, not decoration -- rows with R:R below
      RISK_REWARD_MIN_THRESHOLD (1.5) are pulled out of Confirmed/Watch
      into their own "Filtered -- Poor Risk/Reward" section rather than
      shown as if they were equally good, and R:R at/above
      RISK_REWARD_PREFERRED_THRESHOLD (2.0) is called out as worth
      prioritizing. Rows with no R:R data (missing entry/stop/target)
      are left alone rather than penalized for a gap elsewhere in the
      pipeline.
  11. Historical Failure Risk read (utils.breakout_failure): how often
      this exact pattern has fallen back below its own breakout-day low
      within utils.breakout_failure.FAILURE_WINDOW_DAYS days on this
      exact stock before -- informational context on every row, not a
      gate. (Earlier versions of this pipeline also ran a same-day
      bull-trap check -- below-average volume, a long upper wick, a weak
      close, giving back most of the day's gain -- and pulled flagged
      rows into a dedicated "🔴 Failed Breakout" section. That section
      has been removed; rows are no longer segregated on same-day
      characteristics and are bucketed by the normal
      backtest/Risk:Reward rules below like everything else.)
  12a. Fundamental quality gate, Confirmed Breakouts ONLY: for any row
      still on track for Confirmed after steps 10-11, fetch yfinance's
      .info for that symbol (lazily -- only for rows about to qualify,
      not the whole universe) and require ROE, Debt/Equity, and latest
      earnings growth to each clear a classic quality-stock bar where
      that metric is available. A row that clears the technical bar but
      fails an available fundamentals check is pulled into Watch List
      with a note, never dropped; a symbol with none of the three metrics
      available is left ungated (data gap, not a fail). This makes
      "Confirmed" mean high quality on the chart AND the balance sheet,
      not chart mechanics alone.
  12b. Breakout Confirmation Score (utils.breakout_confirmation): a
      9-point checklist tally (price vs resistance, volume, close
      position, RSI band, moving averages, relative strength vs NIFTY,
      sector confirmation, the NIFTY regime read from step 6, and
      extension past resistance) attached to every fired signal as
      "(passed)/(available)" context -- e.g. "8/8" while sector data
      remains unavailable. Purely informational, same as the Quality
      Score -- doesn't change which bucket a row lands in.
  13. Render + send the email, in the same visual style as the other
      reports in this project.

This is deliberately NOT wired to llm_backend -- every number in the
email traces back to a specific, replayable calculation, not a model's
narrative gloss on the numbers.
"""

import argparse
import datetime as dt
import html
import time

import pandas as pd
import yfinance as yf
from zoneinfo import ZoneInfo

from utils.config import get_date_with_suffix
from utils.logger import log
from utils import email_service
from utils.nse_data import get_nifty500_symbols, get_bhavcopy, data_quality_check
from utils.breakout_patterns import scan_all_patterns, PATTERN_DETECTOR_BY_NAME
from utils.breakout_backtest import (
    backtest_signal, compute_quality_score,
    PRIMARY_HORIZON, MIN_SAMPLES_FOR_CONFIDENCE, CONFIRM_HIT_RATE_THRESHOLD,
)
from utils.entry_classification import classify_entry, EntryState, ClassifierConfig
from utils.breakout_targets import (
    compute_targets, TargetsConfig, compute_risk_reward,
    RISK_REWARD_MIN_THRESHOLD, RISK_REWARD_PREFERRED_THRESHOLD,
)
from utils.market_regime import (
    compute_market_regime, MarketRegime, RegimeResult,
    CHECK_LABELS as REGIME_CHECK_LABELS, BEAR_MARKET_MIN_QUALITY_SCORE,
)
from utils.breakout_failure import (
    evaluate_failure_risk, FailureRisk,
    FAILURE_WINDOW_DAYS, MIN_SAMPLES_FOR_FAILURE_RATE, HIGH_FAILURE_RATE_CEILING,
)
from utils.breakout_confirmation import compute_confirmation_score, SECTOR_DATA_NOTE
from utils.constants import STOCKS

SERIF = "Georgia, 'Times New Roman', serif"
SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"

HISTORY_PERIOD = "2y"
BATCH_SIZE = 40          # symbols per yfinance batch download
BATCH_PAUSE_SECONDS = 1  # be polite to the endpoint between batches
MAX_CONFIRMED_ROWS = 25  # keep the email skimmable even on a big breakout day
MAX_WATCH_ROWS = 15
MAX_NEAR_BREAKOUT_ROWS = 15
MAX_FILTERED_RR_ROWS = 15

ENTRY_CLASSIFIER_CONFIG = ClassifierConfig()
TARGETS_CONFIG = TargetsConfig()

# -----------------------------------------------------------------------
# Fundamental quality gate -- Confirmed Breakouts ONLY.
#
# The rest of this pipeline is purely technical (price/volume/pattern
# mechanics, backtested against the stock's own history). This gate adds
# one more requirement before a row is allowed to sit in the "Confirmed"
# bucket: the underlying business has to look financially sound too, not
# just the chart. It is fetched from yfinance's .info (P/E, ROE,
# debt/equity, earnings growth, margins) -- the same "classic quality
# stock" screen: low debt, strong return on equity, growing earnings.
#
# Fail-soft, same convention as every other filter in this file:
#   - Fetched LAZILY, only for rows that already cleared the technical
#     backtest bar (i.e. rows that would otherwise land in Confirmed) --
#     not for the whole NIFTY 500 universe, to keep the run fast.
#   - If yfinance has NONE of the three metrics for a symbol, the gate is
#     treated as unavailable and does NOT block the row -- a data gap
#     elsewhere in the pipeline never silently disqualifies a signal.
#   - If AT LEAST ONE metric is available and it fails the bar, the row
#     is pulled out of Confirmed into Watch List (never dropped), same as
#     the Bearish-regime downgrade -- see row["fundamental_downgraded"].
FUNDAMENTAL_MIN_ROE = 0.15                 # Return on Equity >= 15%
FUNDAMENTAL_MAX_DEBT_TO_EQUITY = 100.0     # Debt/Equity <= 100 (yfinance reports this as a percentage, e.g. 45.2 == 45.2%)
FUNDAMENTAL_MIN_EARNINGS_GROWTH = 0.0      # Latest earnings growth must be non-negative, not shrinking


def _fetch_fundamentals_info(symbol, cache):
    """Raw yfinance .info dict for one symbol, cached per run. Fail-soft --
    returns {} (not None) on any fetch error so callers don't need a
    separate None-check path; an empty dict is treated the same as "no
    metrics available" by evaluate_fundamental_quality."""
    if symbol in cache:
        return cache[symbol]
    info = {}
    try:
        info = yf.Ticker(f"{symbol}.NS").info or {}
    except Exception as e:
        log.warning(f"Breakout Screener: fundamentals fetch failed for {symbol}: {e}")
    cache[symbol] = info
    return info


def evaluate_fundamental_quality(symbol, cache):
    """Returns a dict:
      available: True if yfinance returned at least one of the three
        metrics for this symbol.
      passed: True only if available AND every metric that WAS returned
        clears its threshold (missing individual metrics don't count
        against the row, same fail-soft principle as Risk:Reward).
      checks: list of (label, passed, display_value) for the metrics that
        were available, for the email to show its work.
    """
    info = _fetch_fundamentals_info(symbol, cache)
    roe = info.get("returnOnEquity")
    debt_to_equity = info.get("debtToEquity")
    earnings_growth = info.get("earningsGrowth")
    if earnings_growth is None:
        earnings_growth = info.get("earningsQuarterlyGrowth")

    checks = []
    if roe is not None:
        checks.append(("ROE", roe >= FUNDAMENTAL_MIN_ROE, f"{roe*100:.1f}%"))
    if debt_to_equity is not None:
        checks.append(("Debt/Equity", debt_to_equity <= FUNDAMENTAL_MAX_DEBT_TO_EQUITY, f"{debt_to_equity:.0f}%"))
    if earnings_growth is not None:
        checks.append(("Earnings growth", earnings_growth >= FUNDAMENTAL_MIN_EARNINGS_GROWTH, f"{earnings_growth*100:+.1f}%"))

    available = bool(checks)
    passed = available and all(ok for _, ok, _ in checks)
    return {"available": available, "passed": passed, "checks": checks}

# Bare NSE symbols (constants.STOCKS keys, e.g. "BEL", "SBIN", "ITC") from
# Bala's own direct-equity list -- NOT constants.WATCHLIST, which holds
# full company names in a different format. Uppercased for a
# case-insensitive match against the screener's own symbol strings.
MY_STOCK_SYMBOLS = {s.upper() for s in STOCKS.keys()}
MY_STOCK_ROW_BG = "#E9F7ED"  # light green -- "this is one of your own stocks"

ENTRY_STATE_BADGE_COLORS = {
    # EntryState -> (background, text)
    EntryState.FRESH_BREAKOUT: ("#DCEFE0", "#0f5132"),
    EntryState.RETEST: ("#FCF1D8", "#7a5b00"),
    EntryState.NEAR_BREAKOUT: ("#DCE8F5", "#1c4a8a"),
    EntryState.NONE: ("#EDEAE2", "#8A8F9C"),
}

PATTERN_STYLES = {
    # pattern -> emoji, used purely for quick visual scanning of the table
    "Resistance Breakout": "📈",
    "Volume Surge Breakout": "🔊",
    "52-Week High Breakout": "🚀",
    "Golden Cross (50/200 DMA)": "✳️",
    "Bollinger Squeeze Breakout": "🎯",
    "Bull Flag Breakout": "🚩",
    "Triangle Breakout": "🔺",
    "Cup and Handle Breakout": "☕",
}


# -----------------------------------------------------------------------
# Data pull
# -----------------------------------------------------------------------
def _fetch_history_batch(tickers):
    """One yfinance batch download. Returns {symbol: df} for symbols that
    came back with usable data -- silently omits ones that didn't
    (logged), since a handful of delisted/renamed symbols in any index
    list is normal and shouldn't stop the run."""
    out = {}
    try:
        data = yf.download(
            tickers, period=HISTORY_PERIOD, interval="1d",
            group_by="ticker", auto_adjust=False, threads=True, progress=False,
        )
    except Exception as e:
        log.warning(f"Breakout Screener: batch download failed for {len(tickers)} tickers: {e}")
        return out

    for t in tickers:
        try:
            if len(tickers) == 1:
                df = data
            else:
                df = data[t]
            df = df.dropna(subset=["Close"])
            if not df.empty:
                out[t] = df
        except Exception:
            continue
    return out


def fetch_universe_history(symbols):
    """symbols: NSE symbols without ".NS". Returns {symbol: OHLCV df}."""
    tickers = [f"{s}.NS" for s in symbols]
    symbol_by_ticker = {f"{s}.NS": s for s in symbols}
    result = {}

    for start in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[start:start + BATCH_SIZE]
        batch_data = _fetch_history_batch(batch)
        for ticker, df in batch_data.items():
            result[symbol_by_ticker[ticker]] = df
        log.info(f"Breakout Screener: fetched history for {len(result)}/{len(symbols)} symbols so far...")
        if start + BATCH_SIZE < len(tickers):
            time.sleep(BATCH_PAUSE_SECONDS)

    missing = len(symbols) - len(result)
    if missing:
        log.warning(f"Breakout Screener: {missing} symbols returned no usable yfinance history (delisted/renamed/thin -- skipped).")
    return result


# -----------------------------------------------------------------------
# Setup Strength -- composite read across everything ELSE this pipeline
# already computed for a row (Quality Score, Risk:Reward, Failure Risk,
# Confirmation Score, and the Fundamentals gate where it ran). This is
# NOT a new, independent signal -- it doesn't backtest anything itself --
# it's a single weighted-average summary of the other columns, so a row
# that's strong on every axis stands out at a glance instead of making
# you scan five separate cells.
#
# Explicitly NOT a probability or a guarantee, and it does NOT override
# the Confirmed/Watch/Filtered/Failed bucket a row already landed in --
# see build_report_html's disclaimer. Weighted like this when every
# component is available:
#   Quality Score        35%  (backtest hit-rate/sample/consistency, already blended)
#   Risk:Reward           25%  (capped at 3:1 -- beyond that treated as maxed out)
#   Failure Risk          20%  (inverse of this pattern's historical failure rate)
#   Confirmation Score    15%  (today's 9-point technical checklist)
#   Fundamentals           5%  (ROE/Debt-Equity/earnings growth, Confirmed-only)
# Fail-soft, same convention as every other score in this file: a
# component that wasn't computed for this row (e.g. no R:R because there
# was no exact entry price) is simply dropped and the remaining weights
# are renormalized -- a data gap elsewhere never silently drags the score
# down or up.
SETUP_STRENGTH_WEIGHTS = {
    "quality": 35,
    "risk_reward": 25,
    "failure_risk": 20,
    "confirmation": 15,
    "fundamentals": 5,
}
SETUP_STRENGTH_RR_CAP = 3.0  # R:R at/above this is treated as fully maxed out on that component

SETUP_STRENGTH_BANDS = (
    (80, "Very Strong"),
    (65, "Strong"),
    (50, "Moderate"),
    (0, "Weak"),
)


def compute_setup_strength(row):
    """Returns {"score": int 0-100, "label": str, "components_used": int,
    "components_total": int} or None if NONE of the underlying components
    were available for this row (e.g. an entirely unscored, un-classified
    signal)."""
    components = []  # (weight, fraction 0-1)

    quality = row.get("quality")
    if quality:
        components.append((SETUP_STRENGTH_WEIGHTS["quality"], quality["score"] / 100.0))

    risk_reward = row.get("risk_reward")
    if risk_reward:
        rr_frac = min(risk_reward["ratio"] / SETUP_STRENGTH_RR_CAP, 1.0)
        components.append((SETUP_STRENGTH_WEIGHTS["risk_reward"], rr_frac))

    failure_risk = row.get("failure_risk")
    if failure_risk and failure_risk.backtest and failure_risk.backtest.failure_rate is not None:
        components.append((SETUP_STRENGTH_WEIGHTS["failure_risk"], 1.0 - failure_risk.backtest.failure_rate))

    confirmation = row.get("confirmation")
    if confirmation and confirmation.available_count:
        components.append((SETUP_STRENGTH_WEIGHTS["confirmation"], confirmation.passed_count / confirmation.available_count))

    fundamentals = row.get("fundamentals")
    if fundamentals and fundamentals["available"] and fundamentals["checks"]:
        passed_frac = sum(1 for _, ok, _ in fundamentals["checks"] if ok) / len(fundamentals["checks"])
        components.append((SETUP_STRENGTH_WEIGHTS["fundamentals"], passed_frac))

    if not components:
        return None

    total_weight = sum(w for w, _ in components)
    score = round(sum(w * frac for w, frac in components) / total_weight * 100)
    label = next(lbl for threshold, lbl in SETUP_STRENGTH_BANDS if score >= threshold)
    return {
        "score": score,
        "label": label,
        "components_used": len(components),
        "components_total": len(SETUP_STRENGTH_WEIGHTS),
    }


# -----------------------------------------------------------------------
# Best Execution Case -- a rule-based read across signals ALREADY computed
# for a row (which bucket it landed in, Entry State, Setup Strength,
# Risk:Reward, Failure Risk). Not a new backtest, not a probability, and
# not a replacement for Setup Strength -- it's a stricter, all-of-the-above
# filter on top of it, meant to flag only the handful of rows an
# at-a-glance read would actually be comfortable acting on TODAY, out of
# everything that fired. Same fail-soft convention as the rest of this
# file: a row missing a needed component simply doesn't qualify -- it's
# never flagged as bad, just not called out as a best case.
#
# A row qualifies ONLY when ALL of the following hold:
#   1. It landed in the Confirmed Breakouts bucket (checked by the caller
#      -- Watch/Filtered/Near-Breakout rows never qualify, no matter how
#      high their component scores are).
#   2. Entry State is Fresh Breakout or Retest -- there's a concrete entry
#      price today, not just a pattern fire with no actionable zone.
#   3. Setup Strength >= BEST_EXECUTION_MIN_SETUP_STRENGTH (the composite
#      already blends Quality / Risk:Reward / Failure Risk / Confirmation
#      / Fundamentals -- see compute_setup_strength above).
#   4. Risk:Reward is at/above RISK_REWARD_PREFERRED_THRESHOLD, not just
#      the bare minimum that keeps a row out of the Filtered section.
#   5. Failure Risk isn't flagged elevated on a reliable sample (reuses
#      utils.breakout_failure's own "elevated" read rather than
#      re-deriving a second threshold here).
BEST_EXECUTION_MIN_SETUP_STRENGTH = 80  # matches the "Very Strong" Setup Strength band
BEST_EXECUTION_ROW_BG = "#87CEFA"       # light sky blue -- "best case for execution today"
BEST_EXECUTION_LABEL = "\U0001F3AF Best Execution Case"


def is_best_execution_case(row):
    """row is expected to already carry entry/setup_strength/risk_reward/
    failure_risk (i.e. called after those are set on the row, and only
    for rows about to be placed in the Confirmed bucket)."""
    entry = row.get("entry")
    if not entry or entry.state not in (EntryState.FRESH_BREAKOUT, EntryState.RETEST):
        return False

    strength = row.get("setup_strength")
    if not strength or strength["score"] < BEST_EXECUTION_MIN_SETUP_STRENGTH:
        return False

    risk_reward = row.get("risk_reward")
    if not risk_reward or risk_reward["ratio"] < RISK_REWARD_PREFERRED_THRESHOLD:
        return False

    failure_risk = row.get("failure_risk")
    if failure_risk and failure_risk.backtest and failure_risk.backtest.elevated:
        return False

    return True


# -----------------------------------------------------------------------
# Scan
# -----------------------------------------------------------------------
def scan_universe(histories, bhav_df, regime: RegimeResult):
    """
    histories: {symbol: OHLCV df}
    regime: RegimeResult from utils.market_regime.compute_market_regime(),
        computed once per run from the SAME histories dict. Two effects:
          - regime.entries_supportive feeds every fresh-breakout entry
            classification's "Market trend supportive" check (True in
            Bullish/Neutral, False in Bearish).
          - regime.only_strongest (Bearish only) raises the Confirmed bar
            below: a signal must ALSO carry a Strong/Excellent Quality
            score and Risk:Reward at the PREFERRED threshold, not just
            clear the backtest -- see REGIME_INTERPRETATION in
            utils.market_regime. Rows that clear the backtest but not
            this stricter bar are downgraded to Watch, never dropped.
    Returns (confirmed, watch_list, filtered_low_rr, near_breakout_watch, skipped_count):
      confirmed: list of signal dicts that cleared the backtest bar AND
        (where R:R data was available) cleared RISK_REWARD_MIN_THRESHOLD
        AND, in a Bearish regime, the stricter regime bar above
      watch_list: signals that fired today but didn't clear the backtest
        bar (thin sample or weak historical hit-rate), OR that cleared it
        but were downgraded by the Bearish-regime bar (row["regime_downgraded"]
        marks these so the email can label them honestly)
      filtered_low_rr: signals that would otherwise have qualified for
        Confirmed/Watch but were pulled out because R:R < RISK_REWARD_MIN_THRESHOLD
        -- shown separately, never silently dropped, per this project's
        "everything shown is labeled for what it is" convention
      near_breakout_watch: symbols that haven't broken out yet but are
        close, per utils.entry_classification -- surfaced even though no
        pattern fired today, so the screener is useful before the breakout too
      skipped_count: symbols dropped entirely by the data-quality gate

      Every row (in every bucket above) also carries row["failure_risk"].backtest
      -- the historical probability THIS exact pattern has failed within
      {FAILURE_WINDOW_DAYS} days on THIS stock before, for context. There is
      no longer a same-day bull-trap gate that segregates rows into their own
      section -- rows are bucketed purely on the backtest/R:R rules above.
    """
    confirmed, watch_list, filtered_low_rr, near_breakout_watch = [], [], [], []
    skipped_count = 0
    market_trend_supportive = regime.entries_supportive
    fundamentals_cache = {}  # per-run cache, symbol -> yfinance .info dict

    for symbol, df in histories.items():
        ok, dq_notes = data_quality_check(symbol, df, bhav_df)
        if not ok:
            skipped_count += 1
            continue

        i = len(df) - 1
        entry_result = classify_entry(symbol, df, market_trend_supportive, ENTRY_CLASSIFIER_CONFIG)

        today_signals = scan_all_patterns(df, i)
        if not today_signals:
            # No pattern fired today -- still worth surfacing if price
            # structure alone says we're approaching a breakout.
            if entry_result and entry_result.state == EntryState.NEAR_BREAKOUT:
                near_breakout_watch.append({
                    "symbol": symbol,
                    "entry": entry_result,
                    "dq_notes": dq_notes,
                    "current_price": float(df["Close"].iloc[-1]),
                })
            continue

        for sig in today_signals:
            detector_fn = PATTERN_DETECTOR_BY_NAME.get(sig["pattern"])
            bt = backtest_signal(symbol, df, detector_fn, i) if detector_fn else None
            quality = compute_quality_score(bt, dq_notes)
            failure_risk = evaluate_failure_risk(symbol, df, i, detector_fn)
            confirmation = compute_confirmation_score(symbol, df, i, regime)

            # Targets need an actual actionable entry price to project
            # from -- only Fresh Breakout / Retest rows have one. Near
            # Breakout hasn't triggered yet and No Setup isn't a buy, so
            # there's nothing to project a target from in either case.
            targets = None
            risk_reward = None
            if entry_result and entry_result.exact_entry_price:
                targets = compute_targets(
                    symbol, df, i, sig["pattern"], sig,
                    entry_result.exact_entry_price, entry_result.atr,
                    TARGETS_CONFIG,
                )
                risk_reward = compute_risk_reward(
                    entry_result.exact_entry_price, entry_result.stop_loss, targets,
                )

            # Cleared the backtest bar on its own merits (the ONLY bar
            # outside a Bearish regime).
            cleared_backtest = bool(bt and bt.get("confirmed"))

            # In a Bearish regime, clearing the backtest is necessary but
            # no longer sufficient -- "only the strongest breakouts are
            # allowed through" also means a Strong/Excellent Quality score
            # and Risk:Reward at the PREFERRED bar, not just the normal
            # minimum (which the filtered_low_rr gate below already enforces
            # regardless of regime).
            if regime.only_strongest:
                meets_confirmed_bar = (
                    cleared_backtest
                    and quality is not None and quality["score"] >= BEAR_MARKET_MIN_QUALITY_SCORE
                    and risk_reward is not None and risk_reward.get("preferred")
                )
            else:
                meets_confirmed_bar = cleared_backtest

            # Fundamental quality gate -- Confirmed Breakouts only (see
            # FUNDAMENTAL_MIN_ROE et al. above). Only bother fetching
            # fundamentals for rows that are, on technicals alone, about
            # to qualify for Confirmed -- keeps this an occasional
            # per-signal yfinance call rather than a 500-symbol sweep.
            fundamentals = None
            fundamental_downgraded = False
            if meets_confirmed_bar:
                fundamentals = evaluate_fundamental_quality(symbol, fundamentals_cache)
                if fundamentals["available"] and not fundamentals["passed"]:
                    meets_confirmed_bar = False
                    fundamental_downgraded = True

            row = {
                "symbol": symbol,
                "pattern": sig["pattern"],
                "signal_price": sig["signal_price"],
                "detail": sig["detail"],
                "dq_notes": dq_notes,
                "backtest": bt,
                "quality": quality,
                "entry": entry_result,
                "targets": targets,
                "risk_reward": risk_reward,
                "failure_risk": failure_risk,
                "confirmation": confirmation,
                "fundamentals": fundamentals,
                # Cleared the backtest but was pulled out of Confirmed
                # specifically by the Bearish-regime bar (not by R:R,
                # which has its own dedicated section) -- lets the email
                # say WHY this row is sitting in Watch instead of leaving
                # it unexplained.
                "regime_downgraded": bool(regime.only_strongest and cleared_backtest and not meets_confirmed_bar and not fundamental_downgraded),
                # Cleared backtest AND the regime bar, but pulled out of
                # Confirmed specifically by weak/high-debt/shrinking
                # fundamentals -- see evaluate_fundamental_quality above.
                "fundamental_downgraded": fundamental_downgraded,
            }
            # Computed last -- needs quality/risk_reward/failure_risk/
            # confirmation/fundamentals to already be in the row dict
            # above, since it's a weighted blend of those, not an
            # independent calculation. See compute_setup_strength.
            row["setup_strength"] = compute_setup_strength(row)

            # Risk/Reward is a CORE FILTER, not decoration: a row with
            # computed R:R below threshold gets pulled into its own
            # section regardless of how good the backtest/quality looked,
            # since a great hit-rate on a bad R:R setup still loses money
            # over time. Rows with no R:R data (missing entry/stop/target)
            # are left in their normal bucket -- don't punish a row for a
            # gap elsewhere in the pipeline.
            if risk_reward and not risk_reward["meets_threshold"]:
                row["best_execution"] = False
                filtered_low_rr.append(row)
            elif meets_confirmed_bar:
                # Best Execution Case is only ever evaluated for rows
                # about to land in Confirmed -- see is_best_execution_case.
                row["best_execution"] = is_best_execution_case(row)
                confirmed.append(row)
            else:
                row["best_execution"] = False
                watch_list.append(row)

    def _rank_key(row):
        # Primary sort: Best Execution Case rows lead every list they're
        # eligible for (see is_best_execution_case -- Confirmed only).
        # Then Setup Strength score (None -- no components available at
        # all -- ranks last, not first, since an unscored row is the
        # least-trustworthy one to lead with). Ties broken by the same
        # keys as before: Breakout Quality Score, then Risk/Reward ratio,
        # then backtest sample size.
        best_execution = 1 if row.get("best_execution") else 0
        strength = row.get("setup_strength")
        strength_score = strength["score"] if strength else -1
        quality_score = row["quality"]["score"] if row["quality"] else -1
        rr = row.get("risk_reward")
        rr_ratio = rr["ratio"] if rr else 0
        bt = row["backtest"] or {}
        return (best_execution, strength_score, quality_score, rr_ratio, bt.get("sample_size", 0))

    confirmed.sort(key=_rank_key, reverse=True)
    watch_list.sort(key=_rank_key, reverse=True)
    filtered_low_rr.sort(key=lambda r: (r["risk_reward"] or {}).get("ratio", 0), reverse=True)
    near_breakout_watch.sort(key=lambda r: r["entry"].distance_to_breakout_pct or 999)
    return confirmed, watch_list, filtered_low_rr, near_breakout_watch, skipped_count


# -----------------------------------------------------------------------
# Email rendering -- same visual language as wealth_controller.py's report
# -----------------------------------------------------------------------
def _backtest_cell(bt):
    if not bt or not bt.get("horizons"):
        return '<span style="color:#8A8F9C;">No historical sample yet</span>'
    primary = bt["horizons"].get(PRIMARY_HORIZON)
    if not primary:
        return '<span style="color:#8A8F9C;">No historical sample yet</span>'
    return (
        f'{primary["hit_rate"]*100:.0f}% hit-rate over {bt["sample_size"]} past occurrences '
        f'(avg {primary["avg_return"]*100:+.1f}% at {PRIMARY_HORIZON}d)'
    )


QUALITY_BADGE_COLORS = {
    # label -> (background, text)
    "Excellent": ("#DCEFE0", "#0f5132"),
    "Strong": ("#E4F0E9", "#3d7a52"),
    "Moderate": ("#FCF1D8", "#7a5b00"),
    "Weak": ("#FBE7DD", "#9a3412"),
    "Poor": ("#F8DADA", "#8a1c1c"),
}


def _quality_badge_html(quality):
    if not quality:
        return '<span style="font-family:{0};font-size:10.5px;color:#8A8F9C;">Not yet scored</span>'.format(SANS)
    bg, fg = QUALITY_BADGE_COLORS.get(quality["label"], ("#EDEAE2", "#3C4256"))
    return (
        f'<span style="display:inline-block;min-width:34px;text-align:center;padding:3px 8px;border-radius:10px;'
        f'background:{bg};color:{fg};font-family:{SANS};font-size:12px;font-weight:700;">{quality["score"]}</span>'
        f'<div style="margin-top:3px;font-family:{SANS};font-size:10px;color:{fg};">{html.escape(quality["label"])}</div>'
    )


def _entry_badge_html(entry_result):
    """Renders the Fresh / Retest / Near-breakout / No-setup badge, plus the
    entry zone/trigger, and the exact actionable entry price where the
    classifier has computed one (Fresh Breakout / Retest only)."""
    if not entry_result:
        return '<span style="font-family:{0};font-size:10.5px;color:#8A8F9C;">Not classified</span>'.format(SANS)

    bg, fg = ENTRY_STATE_BADGE_COLORS.get(entry_result.state, ("#EDEAE2", "#3C4256"))
    badge = (
        f'<span style="display:inline-block;padding:3px 8px;border-radius:10px;'
        f'background:{bg};color:{fg};font-family:{SANS};font-size:11px;font-weight:700;white-space:nowrap;">'
        f'{html.escape(entry_result.label())}</span>'
    )
    detail = ""
    if entry_result.exact_entry_price:
        detail += f'<div style="margin-top:3px;font-size:10.5px;color:#3C4256;font-weight:700;">Entry: ₹{entry_result.exact_entry_price:,.2f}</div>'
    elif entry_result.entry_zone:
        lo, hi = entry_result.entry_zone
        detail += f'<div style="margin-top:3px;font-size:10.5px;color:#3C4256;">Entry: ₹{lo:,.2f}&ndash;₹{hi:,.2f}</div>'
    elif entry_result.entry_trigger:
        detail += f'<div style="margin-top:3px;font-size:10.5px;color:#3C4256;">Trigger: ≥₹{entry_result.entry_trigger:,.2f}</div>'
    elif entry_result.state == EntryState.NONE:
        # No exact price, no zone, no trigger -- the price-structure
        # overlay found nothing to key an entry off (e.g. already
        # extended past resistance, or no usable trailing-high/retest
        # reference in this stock's history). Say so explicitly rather
        # than leaving Entry/Stop-Loss/Targets/R:R blank with no
        # explanation -- same "nothing shown without a reason" rule as
        # the rest of this report.
        detail += (
            f'<div style="margin-top:3px;font-size:10.5px;color:#8A8F9C;">'
            f'No clean entry structure found &mdash; Stop-Loss/Targets/R:R not available.</div>'
        )
    return badge + detail


def _stop_loss_cell_html(entry_result):
    """Dedicated Stop-Loss / Risk column. Only populated for FRESH_BREAKOUT
    and RETEST rows (the states with an exact entry price to measure risk
    against). Shows the final chosen stop, its basis (structural vs ATR),
    and the ATR-based alternative for reference."""
    if not entry_result or not entry_result.stop_loss:
        return '<span style="font-family:{0};font-size:10.5px;color:#8A8F9C;">&mdash;</span>'.format(SANS)

    lines = [
        f'<div style="font-family:{SANS};font-size:12px;font-weight:700;color:#8a1c1c;">₹{entry_result.stop_loss:,.2f}</div>',
        f'<div style="font-size:10.5px;color:#8a1c1c;">Risk: {entry_result.risk_pct:+.1f}%</div>',
        f'<div style="margin-top:2px;font-size:10px;color:#8A8F9C;">{html.escape(entry_result.stop_basis or "")}</div>',
    ]
    if entry_result.atr:
        lines.append(
            f'<div style="font-size:10px;color:#8A8F9C;">ATR(14) ₹{entry_result.atr:,.2f} &middot; '
            f'ATR stop ₹{entry_result.atr_stop:,.2f}</div>'
        )
    return "".join(lines)


TARGET_LABELS = {
    "target1": "T1 &middot; Prior resistance",
    "target2": "T2 &middot; Measured move",
    "target3": "T3 &middot; Fib ext / major level",
}


def _targets_cell_html(targets):
    """Dedicated Targets column. Only populated when the row's Entry
    classification produced an exact entry price (Fresh Breakout /
    Retest) -- see utils.breakout_targets.compute_targets. Each of the
    three targets is independent and separately labeled with its basis,
    same transparency convention as the Stop-Loss cell."""
    if not targets or not any(targets.values()):
        return '<span style="font-family:{0};font-size:10.5px;color:#8A8F9C;">&mdash;</span>'.format(SANS)

    lines = []
    for key in ("target1", "target2", "target3"):
        t = targets.get(key)
        label = TARGET_LABELS[key]
        if not t:
            lines.append(
                f'<div style="margin-top:2px;font-size:10px;color:#8A8F9C;">{label}: not available</div>'
            )
            continue
        lines.append(
            f'<div style="margin-top:3px;font-family:{SANS};font-size:11.5px;font-weight:700;color:#0f5132;">'
            f'₹{t["price"]:,.2f} <span style="font-weight:400;color:#3d7a52;">({t["gain_pct"]:+.1f}%)</span></div>'
            f'<div style="font-size:9.5px;color:#8A8F9C;">{label}</div>'
        )
    return "".join(lines)


RR_TARGET_LABELS = {"target1": "T1", "target2": "T2", "target3": "T3"}


def _risk_reward_cell_html(risk_reward):
    """Dedicated Risk:Reward column. This is a core filter, not a
    decoration -- see utils.breakout_targets.compute_risk_reward and
    scan_universe's filtered_low_rr split. Badge color mirrors the same
    RISK_REWARD_MIN_THRESHOLD / RISK_REWARD_PREFERRED_THRESHOLD used to
    decide whether a row is even shown here, so the color and the
    presence of the row in this table never disagree."""
    if not risk_reward:
        return '<span style="font-family:{0};font-size:10.5px;color:#8A8F9C;">Not available</span>'.format(SANS)

    ratio = risk_reward["ratio"]
    if ratio >= RISK_REWARD_PREFERRED_THRESHOLD:
        bg, fg = ("#DCEFE0", "#0f5132")
    elif ratio >= RISK_REWARD_MIN_THRESHOLD:
        bg, fg = ("#FCF1D8", "#7a5b00")
    else:
        bg, fg = ("#F8DADA", "#8a1c1c")

    target_label = RR_TARGET_LABELS.get(risk_reward["target_used"], "")
    badge = (
        f'<span style="display:inline-block;padding:3px 8px;border-radius:10px;'
        f'background:{bg};color:{fg};font-family:{SANS};font-size:12px;font-weight:700;white-space:nowrap;">'
        f'1 : {ratio:.2f}</span>'
    )
    detail = (
        f'<div style="margin-top:3px;font-size:10px;color:#3C4256;">'
        f'Risk ₹{risk_reward["risk"]:,.2f} &middot; Reward ₹{risk_reward["reward"]:,.2f} (vs {target_label})</div>'
    )
    return badge + detail


def _failure_risk_cell_html(failure_risk):
    """Dedicated Failure Risk column, shown on every row. Reports the
    historical probability THIS exact pattern has failed within
    utils.breakout_failure.FAILURE_WINDOW_DAYS days on THIS stock before
    -- informational context, not itself a gate, same convention as the
    Quality Score badge."""
    if not failure_risk or not failure_risk.backtest:
        return '<span style="font-family:{0};font-size:10.5px;color:#8A8F9C;">No historical sample yet</span>'.format(SANS)

    bt = failure_risk.backtest
    if bt.sample_size == 0 or bt.failure_rate is None:
        return '<span style="font-family:{0};font-size:10.5px;color:#8A8F9C;">No historical occurrences to check</span>'.format(SANS)

    if bt.elevated:
        color = "#8a1c1c"
    elif not bt.confirmed_sample:
        color = "#8A8F9C"
    else:
        color = "#0f5132"

    caveat = "" if bt.confirmed_sample else f' <span style="color:#8A8F9C;">(sample &lt;{MIN_SAMPLES_FOR_FAILURE_RATE})</span>'
    return (
        f'<div style="font-family:{SANS};font-size:12px;font-weight:700;color:{color};">'
        f'{bt.failure_rate*100:.0f}% failed within {FAILURE_WINDOW_DAYS}d{caveat}</div>'
        f'<div style="font-size:10px;color:#8A8F9C;">{bt.failure_count} of {bt.sample_size} past occurrences</div>'
    )


CONFIRMATION_SCORE_COLOR_BANDS = (
    (0.85, "#0f5132"),
    (0.6, "#7a5b00"),
    (0.0, "#8a1c1c"),
)


def _confirmation_score_color(score):
    avail = score.available_count
    if avail == 0:
        return "#8A8F9C"
    frac = score.passed_count / avail
    for threshold, color in CONFIRMATION_SCORE_COLOR_BANDS:
        if frac >= threshold:
            return color
    return CONFIRMATION_SCORE_COLOR_BANDS[-1][1]


def _confirmation_cell_html(score):
    """Dedicated Confirmation Score column -- see
    utils.breakout_confirmation module docstring for the 9-point
    checklist. Shows the "(passed)/(available)" tally plus which of the
    available checks failed, so the badge is never just a number with no
    backing evidence, same transparency convention as the Quality and
    Regime badges. Checks with no data (currently always Sector
    confirmation) are excluded from the tally rather than shown as a
    failure."""
    if not score or score.available_count == 0:
        return '<span style="font-family:{0};font-size:10.5px;color:#8A8F9C;">Not available</span>'.format(SANS)

    color = _confirmation_score_color(score)
    badge = (
        f'<span style="display:inline-block;padding:3px 8px;border-radius:10px;'
        f'background:#EDEAE2;color:{color};font-family:{SANS};font-size:12px;font-weight:700;white-space:nowrap;">'
        f'{score.label()}</span>'
    )
    failed = score.failed
    if failed:
        failed_html = "".join(
            f'<div style="font-size:10px;color:#8a1c1c;">&#10007; {html.escape(c.label)}</div>' for c in failed
        )
    else:
        failed_html = '<div style="font-size:10px;color:#0f5132;">All available checks passed.</div>'
    return badge + failed_html


SETUP_STRENGTH_BADGE_COLORS = {
    # label -> (background, text) -- reuses the same visual language as
    # the Quality badge so "strong" always means the same color across
    # this email.
    "Very Strong": ("#DCEFE0", "#0f5132"),
    "Strong": ("#E4F0E9", "#3d7a52"),
    "Moderate": ("#FCF1D8", "#7a5b00"),
    "Weak": ("#F8DADA", "#8a1c1c"),
}


def _setup_strength_cell_html(strength):
    """Dedicated Setup Strength column -- a single weighted-average read
    across the Quality, Risk:Reward, Failure Risk, Confirmation, and
    Fundamentals columns already in this row (see compute_setup_strength
    above). Shown as a score/label plus how many of the 5 underlying
    components were actually available, same transparency convention as
    every other score in this email -- and explicitly NOT framed as a
    guarantee or an override of the row's Confirmed/Watch/Filtered
    bucket; see the disclaimer in the report footer."""
    if not strength:
        return '<span style="font-family:{0};font-size:10.5px;color:#8A8F9C;">Not enough data</span>'.format(SANS)

    bg, fg = SETUP_STRENGTH_BADGE_COLORS.get(strength["label"], ("#EDEAE2", "#3C4256"))
    badge = (
        f'<span style="display:inline-block;min-width:34px;text-align:center;padding:3px 8px;border-radius:10px;'
        f'background:{bg};color:{fg};font-family:{SANS};font-size:12px;font-weight:700;">{strength["score"]}%</span>'
        f'<div style="margin-top:3px;font-family:{SANS};font-size:10px;color:{fg};">{html.escape(strength["label"])}</div>'
    )
    note = (
        f'<div style="margin-top:2px;font-size:9.5px;color:#8A8F9C;">'
        f'{strength["components_used"]}/{strength["components_total"]} factors available</div>'
    )
    return badge + note


def _fundamentals_note_html(fundamentals):
    """Sub-line under the Quality badge showing the fundamental-quality
    read (ROE / Debt-Equity / Earnings growth) for rows where it was
    computed -- currently only rows that cleared the technical bar for
    Confirmed. Other buckets pass row.get('fundamentals') == None and get
    nothing rendered here."""
    if fundamentals is None:
        return ""
    if not fundamentals["available"]:
        return (
            f'<div style="margin-top:3px;font-size:9.5px;color:#8A8F9C;">Fundamentals: data unavailable</div>'
        )
    color = "#0f5132" if fundamentals["passed"] else "#8a1c1c"
    mark = "&#10003;" if fundamentals["passed"] else "&#10007;"
    detail = " &middot; ".join(f"{html.escape(label)} {html.escape(val)}" for label, _, val in fundamentals["checks"])
    return (
        f'<div style="margin-top:3px;font-size:9.5px;color:{color};">{mark} Fundamentals: {detail}</div>'
    )


def _row_bg_for_symbol(symbol, default_bg):
    """Light-green override when this row's symbol is one of Bala's own
    stocks (constants.STOCKS) -- lets the personal holdings jump out
    from the wider NIFTY 500 scan at a glance. Falls back to the
    table's normal alternating background otherwise."""
    if symbol.upper() in MY_STOCK_SYMBOLS:
        return MY_STOCK_ROW_BG
    return default_bg


def _row_bg_for_row(row, default_bg):
    """Background override, checked in priority order:
      1. Best Execution Case (light sky blue) -- see is_best_execution_case.
         Takes priority over the own-stock shading below since it's the
         stronger, more actionable signal.
      2. Own-stock symbol (light green) -- see MY_STOCK_ROW_BG.
      3. The table's normal alternating background otherwise."""
    if row.get("best_execution"):
        return BEST_EXECUTION_ROW_BG
    return _row_bg_for_symbol(row["symbol"], default_bg)


def _signal_rows_html(rows, row_bg):
    if not rows:
        return '<tr><td style="padding:10px 12px;font-family:{0};font-size:12px;color:#8A8F9C;" colspan="12">None today.</td></tr>'.format(SANS)

    out = []
    for row in rows:
        emoji = PATTERN_STYLES.get(row["pattern"], "🔹")
        caution = ""
        if row["dq_notes"]:
            caution = (
                f'<div style="margin-top:3px;font-size:10.5px;color:#9a3412;">'
                f'⚠ {html.escape("; ".join(row["dq_notes"]))}</div>'
            )
        regime_note = ""
        if row.get("regime_downgraded"):
            regime_note = (
                f'<div style="margin-top:3px;font-size:10.5px;color:#7a5b00;">'
                f'\u2b07 Cleared the backtest, but held back from Confirmed by the 🔴 Bearish-regime filter '
                f'(needs Quality \u2265{BEAR_MARKET_MIN_QUALITY_SCORE} and preferred Risk:Reward).</div>'
            )
        fund_note = ""
        if row.get("fundamental_downgraded"):
            fund_note = (
                f'<div style="margin-top:3px;font-size:10.5px;color:#7a5b00;">'
                f'\u2b07 Cleared the backtest, but held back from Confirmed by the fundamentals gate '
                f'(needs ROE \u2265{FUNDAMENTAL_MIN_ROE*100:.0f}%, Debt/Equity \u2264{FUNDAMENTAL_MAX_DEBT_TO_EQUITY:.0f}%, '
                f'non-negative earnings growth -- see Quality column).</div>'
            )
        best_execution_note = ""
        if row.get("best_execution"):
            best_execution_note = (
                f'<div style="margin-top:3px;font-size:10.5px;font-weight:700;color:#0B4C7C;">'
                f'{BEST_EXECUTION_LABEL}</div>'
            )
        bg = _row_bg_for_row(row, row_bg)
        out.append(f"""
        <tr style="background:{bg};">
          <td data-label="Symbol" style="padding:9px 12px;font-family:{SANS};font-size:13px;font-weight:700;color:#1F2430;border-bottom:1px solid #EDEAE2;">{html.escape(row['symbol'])}{best_execution_note}</td>
          <td data-label="Pattern" style="padding:9px 12px;font-family:{SANS};font-size:12px;color:#3C4256;border-bottom:1px solid #EDEAE2;">{emoji} {html.escape(row['pattern'])}<div style="margin-top:2px;font-size:10.5px;color:#8A8F9C;">{html.escape(row['detail'])}</div>{caution}{regime_note}{fund_note}</td>
          <td data-label="Price" style="padding:9px 12px;font-family:{SANS};font-size:12px;color:#3C4256;border-bottom:1px solid #EDEAE2;text-align:right;">₹{row['signal_price']:,.2f}</td>
          <td data-label="Backtest" style="padding:9px 12px;font-family:{SANS};font-size:11.5px;color:#3C4256;border-bottom:1px solid #EDEAE2;">{_backtest_cell(row['backtest'])}</td>
          <td data-label="Quality" style="padding:9px 12px;border-bottom:1px solid #EDEAE2;text-align:center;">{_quality_badge_html(row['quality'])}{_fundamentals_note_html(row.get('fundamentals'))}</td>
          <td data-label="Entry" style="padding:9px 12px;border-bottom:1px solid #EDEAE2;">{_entry_badge_html(row.get('entry'))}</td>
          <td data-label="Stop-Loss" style="padding:9px 12px;border-bottom:1px solid #EDEAE2;">{_stop_loss_cell_html(row.get('entry'))}</td>
          <td data-label="Targets" style="padding:9px 12px;border-bottom:1px solid #EDEAE2;">{_targets_cell_html(row.get('targets'))}</td>
          <td data-label="Risk:Reward" style="padding:9px 12px;border-bottom:1px solid #EDEAE2;">{_risk_reward_cell_html(row.get('risk_reward'))}</td>
          <td data-label="Failure Risk" style="padding:9px 12px;border-bottom:1px solid #EDEAE2;">{_failure_risk_cell_html(row.get('failure_risk'))}</td>
          <td data-label="Confirmation" style="padding:9px 12px;border-bottom:1px solid #EDEAE2;">{_confirmation_cell_html(row.get('confirmation'))}</td>
          <td data-label="Setup Strength" style="padding:9px 12px;border-bottom:1px solid #EDEAE2;text-align:center;">{_setup_strength_cell_html(row.get('setup_strength'))}</td>
        </tr>
        """)
    return "".join(out)


def _near_breakout_rows_html(rows, row_bg):
    if not rows:
        return '<tr><td style="padding:10px 12px;font-family:{0};font-size:12px;color:#8A8F9C;" colspan="4">None today.</td></tr>'.format(SANS)

    out = []
    for row in rows:
        entry = row["entry"]
        caution = ""
        if row["dq_notes"]:
            caution = (
                f'<div style="margin-top:3px;font-size:10.5px;color:#9a3412;">'
                f'⚠ {html.escape("; ".join(row["dq_notes"]))}</div>'
            )
        distance = entry.distance_to_breakout_pct
        distance_str = f"{distance:.1f}% below trailing resistance" if distance is not None else "—"
        bg = _row_bg_for_symbol(row["symbol"], row_bg)
        out.append(f"""
        <tr style="background:{bg};">
          <td data-label="Symbol" style="padding:9px 12px;font-family:{SANS};font-size:13px;font-weight:700;color:#1F2430;border-bottom:1px solid #EDEAE2;">{html.escape(row['symbol'])}</td>
          <td data-label="Price" style="padding:9px 12px;font-family:{SANS};font-size:12px;color:#3C4256;border-bottom:1px solid #EDEAE2;text-align:right;">₹{row['current_price']:,.2f}</td>
          <td data-label="Distance" style="padding:9px 12px;font-family:{SANS};font-size:11.5px;color:#3C4256;border-bottom:1px solid #EDEAE2;">{html.escape(distance_str)}{caution}</td>
          <td data-label="Entry" style="padding:9px 12px;border-bottom:1px solid #EDEAE2;">{_entry_badge_html(entry)}</td>
        </tr>
        """)
    return "".join(out)


def _table_block(title, subtitle, rows, row_bg, accent):
    header = (
        f'<tr class="table-header-row"><td style="padding:6px 12px;font-family:{SANS};font-size:10px;font-weight:700;'
        f'color:#3C4256;border-bottom:1px solid #DAD5CB;">Symbol</td>'
        f'<td style="padding:6px 12px;font-family:{SANS};font-size:10px;font-weight:700;'
        f'color:#3C4256;border-bottom:1px solid #DAD5CB;">Pattern</td>'
        f'<td style="padding:6px 12px;font-family:{SANS};font-size:10px;font-weight:700;'
        f'color:#3C4256;border-bottom:1px solid #DAD5CB;text-align:right;">Price</td>'
        f'<td style="padding:6px 12px;font-family:{SANS};font-size:10px;font-weight:700;'
        f'color:#3C4256;border-bottom:1px solid #DAD5CB;">Backtest ({PRIMARY_HORIZON}d)</td>'
        f'<td style="padding:6px 12px;font-family:{SANS};font-size:10px;font-weight:700;'
        f'color:#3C4256;border-bottom:1px solid #DAD5CB;text-align:center;">Quality</td>'
        f'<td style="padding:6px 12px;font-family:{SANS};font-size:10px;font-weight:700;'
        f'color:#3C4256;border-bottom:1px solid #DAD5CB;">Entry</td>'
        f'<td style="padding:6px 12px;font-family:{SANS};font-size:10px;font-weight:700;'
        f'color:#3C4256;border-bottom:1px solid #DAD5CB;">Stop-Loss</td>'
        f'<td style="padding:6px 12px;font-family:{SANS};font-size:10px;font-weight:700;'
        f'color:#3C4256;border-bottom:1px solid #DAD5CB;">Targets</td>'
        f'<td style="padding:6px 12px;font-family:{SANS};font-size:10px;font-weight:700;'
        f'color:#3C4256;border-bottom:1px solid #DAD5CB;">Risk:Reward</td>'
        f'<td style="padding:6px 12px;font-family:{SANS};font-size:10px;font-weight:700;'
        f'color:#3C4256;border-bottom:1px solid #DAD5CB;">Failure Risk ({FAILURE_WINDOW_DAYS}d)</td>'
        f'<td style="padding:6px 12px;font-family:{SANS};font-size:10px;font-weight:700;'
        f'color:#3C4256;border-bottom:1px solid #DAD5CB;">Confirmation</td>'
        f'<td style="padding:6px 12px;font-family:{SANS};font-size:10px;font-weight:700;'
        f'color:#3C4256;border-bottom:1px solid #DAD5CB;text-align:center;">Setup Strength</td></tr>'
    )
    rows_html = _signal_rows_html(rows, row_bg)
    return f"""
    <tr>
      <td style="padding:0 28px 18px;" class="email-padding">
        <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="border-radius:6px;background:#FAF8F1;border:1px solid #E7DFC9;">
          <tr>
            <td style="padding:14px 16px 4px;">
              <div style="font-family:{SANS};font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:{accent};">{title}</div>
              <div style="margin:2px 0 8px;font-family:{SANS};font-size:11px;color:#8A8F9C;">{subtitle}</div>
              <table width="100%" cellpadding="0" cellspacing="0" role="presentation" class="stack-table" style="border-collapse:collapse;">
                {header}
                {rows_html}
              </table>
            </td>
          </tr>
        </table>
      </td>
    </tr>
    """


REGIME_BLOCK_COLORS = {
    # regime -> (background, border, accent text)
    MarketRegime.BULLISH: ("#EEF7F0", "#CDE7D4", "#0f5132"),
    MarketRegime.NEUTRAL: ("#FCF7EA", "#EFDFAF", "#7a5b00"),
    MarketRegime.BEARISH: ("#FBEEEC", "#F2C9C2", "#8a1c1c"),
}


def _regime_block(regime: RegimeResult):
    """NIFTY 500 market regime -- computed once per run by
    utils.market_regime.compute_market_regime() and used to gate what
    gets shown as Confirmed (see scan_universe's regime_downgraded logic).
    Renders the 5-check breakdown so the badge is never just a color with
    no backing evidence, same transparency convention as the Quality Score."""
    bg, border, accent = REGIME_BLOCK_COLORS.get(regime.regime, ("#FAF8F1", "#E7DFC9", "#3C4256"))
    d = regime.detail

    def _check_row(label):
        passed = regime.checks.get(label)
        mark = "✔" if passed else "✘"
        color = "#0f5132" if passed else "#8a1c1c"
        return (
            f'<span style="display:inline-block;margin:2px 10px 2px 0;font-family:{SANS};font-size:11px;color:{color};">'
            f'{mark} {html.escape(label)}</span>'
        )

    checks_html = "".join(_check_row(label) for label in REGIME_CHECK_LABELS)

    detail_bits = []
    if d.get("nifty_close") is not None:
        detail_bits.append(
            f'NIFTY 50 {d["nifty_close"]:,.0f} vs 50DMA {d["nifty_50dma"]:,.0f} / 200DMA {d["nifty_200dma"]:,.0f}'
        )
    if d.get("nifty_resistance") is not None:
        detail_bits.append(f'prior resistance {d["nifty_resistance"]:,.0f}')
    if d.get("breadth_pct") is not None:
        detail_bits.append(f'breadth {d["breadth_pct"]:.0f}% ({d["breadth_above"]}/{d["breadth_total"]} above 50DMA)')
    if d.get("vix") is not None:
        detail_bits.append(f'India VIX {d["vix"]:.1f}')
    detail_line = " &middot; ".join(detail_bits) if detail_bits else "Underlying index/breadth/VIX data unavailable this run."

    notes_html = ""
    if regime.notes:
        notes_html = (
            f'<div style="margin-top:6px;font-family:{SANS};font-size:10px;color:#8A8F9C;">'
            f'⚠ {html.escape(" ".join(regime.notes))}</div>'
        )

    return f"""
    <tr>
      <td style="padding:0 28px 18px;" class="email-padding">
        <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="border-radius:6px;background:{bg};border:1px solid {border};">
          <tr>
            <td style="padding:14px 16px;">
              <div style="font-family:{SANS};font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:{accent};">Market Regime &mdash; NIFTY 500</div>
              <div style="margin:4px 0 6px;font-family:{SERIF};font-size:17px;color:{accent};">{regime.label()} <span style="font-family:{SANS};font-size:11px;font-weight:400;color:#8A8F9C;">({regime.score}/5 checks)</span></div>
              <div>{checks_html}</div>
              <div style="margin-top:8px;font-family:{SANS};font-size:11px;color:#3C4256;">{html.escape(detail_line)}</div>
              <div style="margin-top:6px;font-family:{SANS};font-size:11.5px;color:{accent};">{html.escape(regime.interpretation())}</div>
              {notes_html}
            </td>
          </tr>
        </table>
      </td>
    </tr>
    """


def _near_breakout_block(rows):
    header = (
        f'<tr class="table-header-row"><td style="padding:6px 12px;font-family:{SANS};font-size:10px;font-weight:700;'
        f'color:#3C4256;border-bottom:1px solid #DAD5CB;">Symbol</td>'
        f'<td style="padding:6px 12px;font-family:{SANS};font-size:10px;font-weight:700;'
        f'color:#3C4256;border-bottom:1px solid #DAD5CB;text-align:right;">Price</td>'
        f'<td style="padding:6px 12px;font-family:{SANS};font-size:10px;font-weight:700;'
        f'color:#3C4256;border-bottom:1px solid #DAD5CB;">Distance</td>'
        f'<td style="padding:6px 12px;font-family:{SANS};font-size:10px;font-weight:700;'
        f'color:#3C4256;border-bottom:1px solid #DAD5CB;">Entry</td></tr>'
    )
    rows_html = _near_breakout_rows_html(rows, "#ffffff")
    return f"""
    <tr>
      <td style="padding:0 28px 18px;" class="email-padding">
        <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="border-radius:6px;background:#FAF8F1;border:1px solid #E7DFC9;">
          <tr>
            <td style="padding:14px 16px 4px;">
              <div style="font-family:{SANS};font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#1c4a8a;">🔵 Near Breakout &mdash; Watch</div>
              <div style="margin:2px 0 8px;font-family:{SANS};font-size:11px;color:#8A8F9C;">No pattern fired today, but price is within {ENTRY_CLASSIFIER_CONFIG.near_breakout_pct:.0f}% of trailing resistance -- not a buy yet, watch for the trigger price with volume confirmation.</div>
              <table width="100%" cellpadding="0" cellspacing="0" role="presentation" class="stack-table" style="border-collapse:collapse;">
                {header}
                {rows_html}
              </table>
            </td>
          </tr>
        </table>
      </td>
    </tr>
    """


def build_report_html(confirmed, watch_list, filtered_low_rr, near_breakout_watch, scan_stats):
    now_ist = dt.datetime.now(ZoneInfo("Asia/Kolkata"))
    date_str = get_date_with_suffix(now_ist)
    regime: RegimeResult = scan_stats["regime"]

    regime_block = _regime_block(regime)

    confirmed_subtitle = (
        f"Backtest cleared the bar: ≥{MIN_SAMPLES_FOR_CONFIDENCE} past occurrences and ≥{CONFIRM_HIT_RATE_THRESHOLD*100:.0f}% {PRIMARY_HORIZON}-day hit-rate, "
        f"Risk:Reward ≥ {RISK_REWARD_MIN_THRESHOLD:.1f}, and (where fundamentals data is available) ROE ≥{FUNDAMENTAL_MIN_ROE*100:.0f}%, "
        f"Debt/Equity ≤{FUNDAMENTAL_MAX_DEBT_TO_EQUITY:.0f}%, non-negative earnings growth -- high quality on both the chart and the balance sheet, not chart-only. "
        "Entry column shows whether this is chaseable now, better bought on a retest, or neither -- see disclaimer below."
    )
    if regime.only_strongest:
        confirmed_subtitle += (
            f" 🔴 Bearish regime is active, so this list is also restricted to Quality \u2265{BEAR_MARKET_MIN_QUALITY_SCORE} "
            f"and preferred Risk:Reward (\u2265{RISK_REWARD_PREFERRED_THRESHOLD:.1f}) -- see Market Regime above."
        )
    confirmed_subtitle += (
        f' Rows shaded <span style="background:{BEST_EXECUTION_ROW_BG};padding:0 3px;">light sky blue</span> are this '
        f"run's {BEST_EXECUTION_LABEL} rows -- Confirmed, Fresh Breakout/Retest (a concrete entry price today), "
        f"Setup Strength \u2265{BEST_EXECUTION_MIN_SETUP_STRENGTH}, Risk:Reward \u2265{RISK_REWARD_PREFERRED_THRESHOLD:.1f}, "
        "and Failure Risk not flagged elevated -- see disclaimer below for the full rule."
    )
    confirmed_block = _table_block(
        "✅ Confirmed Breakouts",
        confirmed_subtitle,
        confirmed[:MAX_CONFIRMED_ROWS], "#ffffff", "#0f5132",
    )
    watch_subtitle = (
        f"Pattern fired today but the historical sample was thin or the hit-rate was weak -- treat as a watch item, not a call. "
        f"(Still Risk:Reward ≥ {RISK_REWARD_MIN_THRESHOLD:.1f} where R:R data was available.)"
    )
    if regime.only_strongest:
        watch_subtitle += " Rows marked with ⬇ below cleared the backtest but were held back from Confirmed by the Bearish-regime filter."
    watch_subtitle += " Rows marked with ⬇ and 'fundamentals gate' cleared the backtest but were held back from Confirmed by weak/unavailable fundamentals."
    watch_block = _table_block(
        "👀 Unconfirmed / Watch List",
        watch_subtitle,
        watch_list[:MAX_WATCH_ROWS], "#ffffff", "#7a5b00",
    )
    filtered_rr_block = _table_block(
        "🚫 Filtered &mdash; Poor Risk/Reward",
        f"Would otherwise have qualified for Confirmed/Watch, but Risk:Reward against the nearest available target came in "
        f"below {RISK_REWARD_MIN_THRESHOLD:.1f} -- shown here, not dropped, so nothing disappears silently. "
        f"Setups with R:R &ge; {RISK_REWARD_PREFERRED_THRESHOLD:.1f} are the ones worth prioritizing in the tables above.",
        filtered_low_rr[:MAX_FILTERED_RR_ROWS], "#ffffff", "#8a1c1c",
    )
    near_breakout_block = _near_breakout_block(near_breakout_watch[:MAX_NEAR_BREAKOUT_ROWS])

    universe_note = (
        f"Scanned {scan_stats['universe_size']} NIFTY 500 symbols "
        f"({'live index list' if scan_stats['universe_is_live'] else 'fallback core list -- live NSE index fetch failed this run'}); "
        f"{scan_stats['history_count']} returned usable price history; "
        f"{scan_stats['skipped_count']} skipped by the data-quality gate; "
        f"{scan_stats['filtered_low_rr_count']} filtered out on Risk:Reward &lt; {RISK_REWARD_MIN_THRESHOLD:.1f}."
    )
    bhav_note = (
        f"Same-day NSE bhavcopy cross-check: active ({scan_stats['bhav_date']})."
        if scan_stats["bhav_is_live"]
        else "Same-day NSE bhavcopy cross-check: unavailable this run (NSE fetch failed) -- signals rely on yfinance data only."
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="color-scheme" content="light">
<meta name="supported-color-schemes" content="light">
<meta name="x-apple-disable-message-reformatting">
<meta name="format-detection" content="telephone=no,date=no,address=no,email=no,url=no">
<title>Daily Breakout Screener</title>
<style>
  body {{ margin:0; padding:0; background:#F2F0EC; }}
  table {{ border-collapse:collapse !important; }}
  /* Stop iOS/macOS Mail's auto dark-mode from repainting our own
     colors -- without this, backgrounds/text can invert unpredictably
     on an iPhone even though color-scheme/supported-color-schemes
     above say "light only". Belt-and-suspenders for iOS Mail. */
  @media (prefers-color-scheme: dark) {{
    body, .email-container {{ background:#F2F0EC !important; }}
  }}
  @media screen and (max-width:600px) {{
    .email-container {{ width:100% !important; max-width:100% !important; }}
    .email-padding {{ padding-left:14px !important; padding-right:14px !important; }}
    /* Wide data tables (11/6/4 columns) don't fit a phone screen -- on
       screens under 600px, drop the header row and stack every <td> into
       its own full-width block instead, with the column name (from
       data-label) printed above the value. Same information, no
       side-scrolling and no squashed columns. */
    .stack-table, .stack-table tbody, .stack-table tr, .stack-table td {{
      display:block !important;
      width:100% !important;
      box-sizing:border-box !important;
    }}
    .stack-table .table-header-row {{ display:none !important; }}
    .stack-table tr {{
      border-bottom:1px solid #DAD5CB !important;
      padding:10px 0 !important;
    }}
    .stack-table tr:last-child {{ border-bottom:none !important; }}
    .stack-table td {{
      border-bottom:none !important;
      padding:6px 14px !important;
      text-align:left !important;
    }}
    .stack-table td[data-label]:before {{
      content: attr(data-label);
      display:block;
      font-family:{SANS};
      font-size:9.5px;
      font-weight:700;
      letter-spacing:0.08em;
      text-transform:uppercase;
      color:#B0A98C;
      margin-bottom:3px;
    }}
    /* Bump up the smallest text a touch so it's still readable at arm's
       length on a phone, without changing the desktop email at all. */
    .stack-table td, .stack-table td div, .stack-table td span {{ font-size:13px !important; }}
    .stack-table td div[style*="font-size:9"],
    .stack-table td div[style*="font-size:10"] {{ font-size:11.5px !important; }}
    /* The closing legend/disclaimer paragraph is dense reference text --
       10px is too small to read comfortably on a phone. */
    .legend-text {{ font-size:12.5px !important; line-height:1.6 !important; }}
  }}
</style>
</head>
<body style="margin:0;padding:0;background:#F2F0EC;font-family:{SERIF};color:#1B2233;">
  <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="background:#F2F0EC;width:100%;">
    <tr>
      <td align="center" style="padding:20px 16px;" class="email-padding">
        <table width="100%" cellpadding="0" cellspacing="0" role="presentation" class="email-container" style="max-width:680px;min-width:280px;background:#ffffff;border:1px solid #DAD5CB;border-radius:4px;overflow:hidden;">
          <tr>
            <td style="background:#14213D;padding:26px 28px 22px;" class="email-padding">
              <div style="font-family:{SANS};font-size:10px;font-weight:700;letter-spacing:0.16em;text-transform:uppercase;color:#B08D57;">Equities &nbsp;&bull;&nbsp; Daily Screener</div>
              <h1 style="margin:8px 0 0;font-family:{SERIF};font-size:22px;font-weight:400;line-height:1.3;color:#ffffff;">Daily Breakout Screener &mdash; NIFTY 500</h1>
            </td>
          </tr>
          <tr><td style="height:3px;line-height:3px;font-size:0;background:linear-gradient(90deg,#B08D57,#D9C393 45%,#B08D57);">&nbsp;</td></tr>
          <tr>
            <td style="padding:14px 28px 4px;" class="email-padding">
              <p style="margin:0;font-family:{SANS};font-size:12px;color:#8A8F9C;">{date_str}</p>
            </td>
          </tr>
          <tr>
            <td style="padding:6px 28px 16px;border-bottom:1px solid #EDEAE2;" class="email-padding">
              <p style="margin:0;font-family:{SANS};font-size:12px;color:#4A5063;">{universe_note}</p>
              <p style="margin:4px 0 0;font-family:{SANS};font-size:11px;color:#8A8F9C;">{bhav_note}</p>
            </td>
          </tr>
          {regime_block}
          {confirmed_block}
          {watch_block}
          {filtered_rr_block}
          {near_breakout_block}
          <tr>
            <td style="padding:16px 28px;border-top:1px solid #EDEAE2;" class="email-padding">
              <p class="legend-text" style="margin:0;font-family:{SANS};font-size:10px;color:#8A8F9C;line-height:1.5;">
                Market Regime (top of this email) is a 5-check read on the NIFTY 500 as a whole -- NIFTY vs its
                50/200-day moving averages, NIFTY above its own trailing resistance, breadth (% of this run's
                scanned universe above their own 50-day average), and India VIX -- scored 0-5 and bucketed into
                🟢 Bullish / 🟡 Neutral / 🔴 Bearish. Bullish and Neutral use the normal Confirmed/Watch thresholds
                below; in a Bearish regime, Confirmed also requires a Strong/Excellent Quality score and preferred
                Risk:Reward, and every pattern's "Market trend supportive" Fresh-Breakout check (see Entry column)
                is withheld -- the detectors and backtests themselves never change, only how much of what fires
                gets called Confirmed.
                Every signal above is mechanical (price/volume/pattern rules only, no AI narrative) and every
                "Confirmed" signal has been backtested against that stock's own trailing history at run time --
                see the Backtest column. The Quality score (0-100) is a transparent weighted blend of that
                backtest's hit-rate (45%), sample size (20%), average return (15%), consistency across the
                5/10/20-day horizons (10%), and this run's data-quality notes (10%) -- not a prediction, a
                summary of how well this exact pattern has actually worked on this exact stock before.
                The Entry column is a separate, generic price-structure read (trailing-high resistance, volume,
                RSI, and market regime) on timing only: 🟢 Fresh Breakout means every chase-safety check passed
                today; 🟡 Retest means the stock pulled back to resistance and printed a bullish reversal candle
                -- typically better risk/reward than chasing; 🔵 Near Breakout means it hasn't broken out yet
                but is close, worth a watch, not a buy. A pattern being "Confirmed" and its Entry state being
                Fresh/Retest are independent checks -- read both before acting. For Fresh Breakout and Retest
                rows, the Stop-Loss column is not an arbitrary percentage: it compares a structural stop (below
                breakout support, the prior swing low, the pattern low, or the retest low -- whichever sits
                nearest to entry) against an ATR(14)-based stop, and uses whichever is more logical for that
                stock's own volatility -- falling back to the ATR stop if the structural level is too tight or
                too wide relative to ATR. Risk % is the distance from entry to that stop.
                The Targets column (Fresh Breakout / Retest rows only) shows three independent technical
                price objectives from that same entry price: Target 1 is the nearest prior swing-high
                resistance in the stock's own trailing history; Target 2 is the pattern's own measured move
                (e.g. cup depth or flagpole height projected from the breakout, ATR-based if the pattern has
                no natural base geometry); Target 3 is a 1.618x Fibonacci extension of that same base, or a
                further proven swing high, whichever sits farther out. These are transparent projections, not
                predictions -- "not available" means the underlying structure (a swing high, a pattern base)
                wasn't found in this stock's history, not that the target is zero.
                Risk:Reward compares that same stop-loss (risk) against the NEAREST available target (reward) --
                shown as e.g. "1 : 1.84" meaning ₹1.84 of reward for every ₹1 risked. This is a CORE FILTER: any
                setup with R:R below {RISK_REWARD_MIN_THRESHOLD:.1f} is pulled out of Confirmed/Watch into the
                "Filtered -- Poor Risk/Reward" section below, on the same nothing-silently-dropped principle as
                everything else in this email -- a strong backtest hit-rate doesn't matter if the reward on offer
                doesn't clear a sane multiple of the risk. R:R &ge; {RISK_REWARD_PREFERRED_THRESHOLD:.1f} (green badge)
                is worth prioritizing over the rest.
                The Failure Risk column, shown on every row, is a separate, historical number: how often this
                exact pattern has fallen back below its own breakout-day low within {FAILURE_WINDOW_DAYS} days on
                this exact stock before -- a base rate for context, not itself a gate, shown "(sample
                &lt;{MIN_SAMPLES_FOR_FAILURE_RATE})" when there isn't enough history yet to call it reliable.
                The Confirmation column is a 9-point checklist tally (price above resistance, volume &gt;2x average,
                a close near the day's high, RSI 55-75, price above both the 20- and 50-day averages, relative
                strength vs NIFTY 50, sector confirmation, the NIFTY regime read above, and no excessive extension
                past resistance) -- shown as "(passed)/(available)", e.g. "8/8". Sector confirmation is always N/A
                in this build -- {html.escape(SECTOR_DATA_NOTE)} -- so every score currently maxes out at 8/8, not
                9/9; unavailable checks are excluded from the tally rather than counted as a failure. This is
                additional context on a row whose Confirmed/Watch/Filtered bucket is already decided --
                it doesn't itself gate anything, the way Risk:Reward does.
                <b>Fundamentals gate (Confirmed Breakouts only):</b> clearing the technical backtest bar is necessary but no
                longer sufficient for the Confirmed section -- the underlying business also has to clear a classic
                "quality stock" screen, pulled from yfinance at run time: Return on Equity \u2265{FUNDAMENTAL_MIN_ROE*100:.0f}%,
                Debt/Equity \u2264{FUNDAMENTAL_MAX_DEBT_TO_EQUITY:.0f}%, and non-negative latest earnings growth. This is only
                fetched for rows that already cleared the technical bar, and it never gates the Watch List, Filtered, Failed,
                or Near-Breakout sections -- only Confirmed. Same "nothing silently dropped" rule as everything else here: a
                row that clears technicals but fails an available fundamentals check is pulled into Watch List with a note
                (⬇, see Pattern column), not discarded. If yfinance has none of the three metrics for a symbol, the gate is
                skipped rather than treated as a fail -- a data gap doesn't disqualify a signal. The Quality column shows
                which specific metrics were checked and whether each passed.
                Rows shaded <span style="background:#E9F7ED;padding:0 3px;">light green</span> are symbols already
                in your own direct-equity list (constants.STOCKS) -- called out purely to jump out from the wider
                NIFTY 500 scan, not a separate signal.
                <b>{html.escape(BEST_EXECUTION_LABEL)}</b> rows -- shaded <span style="background:{BEST_EXECUTION_ROW_BG};padding:0 3px;">light
                sky blue</span> in the Confirmed table above, and taking priority over the light-green own-stock
                shading when both would apply -- are a stricter, rule-based filter layered on top of everything else
                on this page, meant to surface only the handful of setups worth a first look TODAY rather than the
                full Confirmed list: (1) already in the Confirmed Breakouts bucket, (2) Entry State is Fresh Breakout
                or Retest, i.e. there's an exact entry price today, (3) Setup Strength
                \u2265{BEST_EXECUTION_MIN_SETUP_STRENGTH} ("Very Strong"), (4) Risk:Reward
                \u2265{RISK_REWARD_PREFERRED_THRESHOLD:.1f}, and (5) Failure Risk not flagged elevated on a reliable
                sample. It runs no new backtest and adds no new data -- it's a same-fail-soft, all-of-the-above pass
                over the columns already on this page, so a row missing any one component simply doesn't qualify
                rather than being marked bad. Like Setup Strength, this is <b>not</b> a probability of success and
                <b>not</b> investment advice -- verify every number on the row yourself before acting on it.
                <b>Setup Strength</b> is a single weighted summary of the columns to its left -- Quality Score (35%),
                Risk:Reward (25%, capped at 3:1), Failure Risk (20%), Confirmation Score (15%), and the Fundamentals
                gate where it ran (5%) -- so a row that's strong across the board stands out without reading five
                separate cells. It is a summary, not a new calculation: it backtests nothing on its own, and any
                component missing for a row (e.g. no R:R because there's no exact entry price) is dropped and the
                remaining weights renormalized, same fail-soft rule as everywhere else here -- the "X/5 factors
                available" note under the score shows how much of it a given row is actually resting on. It does
                <b>not</b> override or replace the Confirmed/Watch/Filtered/Failed bucket a row already landed in,
                and it is not a probability of the stock going up -- treat it as one more input to weigh alongside
                the columns it summarizes and your own judgment, not a stand-alone buy signal.
                Past pattern performance does not guarantee future results. Informational only -- not
                investment advice. Verify before acting.
              </p>
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>
"""


def send_report_email(report_html):
    now_ist = dt.datetime.now(ZoneInfo("Asia/Kolkata"))
    subject = f"Daily Breakout Screener - {get_date_with_suffix(now_ist)}"
    return email_service.send_email(subject=subject, html_body=report_html)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main(dry_run=False):
    log.info("Breakout Screener: starting run...")

    symbols, universe_is_live = get_nifty500_symbols()
    bhav_df, bhav_date, bhav_is_live = get_bhavcopy()

    histories = fetch_universe_history(symbols)

    # Regime is computed AFTER histories are in hand -- its breadth check
    # (utils.market_regime._compute_breadth) reuses this exact {symbol: df}
    # dict rather than pulling the universe a second time.
    regime = compute_market_regime(histories)
    log.info(f"Breakout Screener: market regime {regime.label()} (score {regime.score}/5) -- {regime.checks}")

    confirmed, watch_list, filtered_low_rr, near_breakout_watch, skipped_count = scan_universe(
        histories, bhav_df, regime
    )

    scan_stats = {
        "universe_size": len(symbols),
        "universe_is_live": universe_is_live,
        "history_count": len(histories),
        "skipped_count": skipped_count,
        "filtered_low_rr_count": len(filtered_low_rr),
        "bhav_is_live": bhav_is_live,
        "bhav_date": bhav_date.strftime("%d %b %Y") if bhav_date else None,
        "regime": regime,
    }
    log.info(
        f"Breakout Screener: {len(confirmed)} confirmed, {len(watch_list)} watch-list, "
        f"{len(filtered_low_rr)} filtered on Risk:Reward, {len(near_breakout_watch)} near-breakout signals "
        f"out of {len(histories)} symbols scanned ({skipped_count} skipped on data quality). "
        f"Market regime: {regime.label()}."
    )

    report_html = build_report_html(confirmed, watch_list, filtered_low_rr, near_breakout_watch, scan_stats)

    if dry_run:
        out_path = "breakout_report_preview.html"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report_html)
        log.info(f"Dry run -- wrote preview to {out_path} instead of sending email.")
        return

    success = send_report_email(report_html)
    if success:
        log.info("Breakout Screener: email sent successfully.")
    else:
        log.error("Breakout Screener: failed to send email.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan NIFTY 500 for chart-pattern breakouts and email the report.")
    parser.add_argument("--dry-run", action="store_true", help="Write the report to a local HTML file instead of emailing it.")
    args = parser.parse_args()
    main(dry_run=args.dry_run)