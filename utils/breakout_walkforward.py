"""
utils/breakout_walkforward.py

Out-of-sample / walk-forward validation for the breakout screener --
answers a DIFFERENT question than utils.breakout_backtest's live,
per-stock backtest does. That module asks "has THIS pattern worked on
THIS stock before, using all available history" -- which is exactly
right for a live signal, but it's also how thresholds like
CONFIRM_HIT_RATE_THRESHOLD (0.55) end up implicitly "tuned" against the
whole dataset if you ever eyeball full-history stats while picking them:
a threshold, weight, or filter that looks good on the full sample can
still be curve-fit noise that won't hold up on data it never saw. This
module exists to check that, by splitting history into folds and never
letting any stat from a later fold influence anything computed from an
earlier one.

THIS IS A SEPARATE, OFFLINE RESEARCH TOOL -- it is not imported by
breakout_controller.py's daily run and does not change any live
behavior. Run it manually / on a schedule to sanity-check that the
screener's fixed thresholds and pattern detectors still generalize, not
every day.

-----------------------------------------------------------------------
IMPORTANT PREREQUISITE -- read before running this
-----------------------------------------------------------------------
breakout_controller.HISTORY_PERIOD is currently "2y". A 2018-2026
walk-forward is NOT POSSIBLE on 2 years of data -- most folds below
would simply come back empty. This module fetches its OWN longer
history (see fetch_long_history()) rather than reusing the live
pipeline's 2y pull, specifically so the daily email job doesn't have to
pay for 8+ years of data on every run just to support this offline
check. You still need a symbol list and a detector map from your own
code (see the __main__ block at the bottom for how those plug in) --
this module doesn't hardcode either.

-----------------------------------------------------------------------
Design choices worth knowing before reading results
-----------------------------------------------------------------------
- Per-stock-per-pattern samples are almost always too thin to say
  anything on their own (this project's own MIN_SAMPLES_FOR_CONFIDENCE=5
  makes that explicit). Fold evaluation here therefore POOLS events
  across every symbol in the universe for a given pattern within a
  fold, and reports pooled trade stats -- "how did the Breakout pattern
  do, across the whole NIFTY 500, during 2024-2025" rather than
  per-symbol numbers, which is what you actually need to judge whether
  a PATTERN (and the thresholds gating it) generalizes.
- A signal's forward return/exit is allowed to use price data that
  falls AFTER a fold's end date (e.g. a signal on the last trading day
  of a Test fold still gets its full 10-day forward read from days that
  are technically past the fold boundary). This is intentional and NOT
  lookahead: the signal itself only used information up to its own
  date, and we're measuring what actually happened next, not feeding
  that into any decision. What's excluded is the reverse -- a fold
  never OFFERS UP events whose signal date falls outside [start, end],
  so an OOS fold can't be quietly padded with events from Train.
"""

import datetime as dt

import numpy as np
import pandas as pd

from utils.logger import log
from utils.breakout_backtest import (
    find_historical_events, simulate_trades, compute_trade_stats,
    FORWARD_HORIZONS, PRIMARY_HORIZON, CONFIRM_HIT_RATE_THRESHOLD, MIN_SAMPLES_FOR_CONFIDENCE,
)

HISTORY_PERIOD_FOR_WALKFORWARD = "10y"  # yfinance max practical lookback; adjust if you have a
                                          # longer/cleaner data source for 2018+ NSE history


def fetch_long_history(symbols, batch_size=50):
    """Separate, longer-lookback fetch for validation only -- deliberately
    NOT reusing breakout_controller.fetch_universe_history, which is tuned
    for the live daily run's speed/quota budget on HISTORY_PERIOD='2y'.
    Same batching approach; swap in your own data source here if 10y of
    clean yfinance NSE data proves unreliable for older symbols (thinly
    traded/renamed/delisted names often have gappy or missing history
    that far back -- expect some symbols to come back short or empty,
    which is itself informative: those are exactly the survivorship-bias
    casualties flagged in the item-15 review note)."""
    import yfinance as yf

    histories = {}
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        tickers = [f"{s}.NS" for s in batch]
        try:
            data = yf.download(
                tickers, period=HISTORY_PERIOD_FOR_WALKFORWARD, interval="1d",
                group_by="ticker", auto_adjust=True, progress=False, threads=True,
            )
        except Exception as e:
            log.warning(f"Walk-forward: batch fetch failed ({batch[0]}..{batch[-1]}): {e}")
            continue
        for symbol in batch:
            ticker = f"{symbol}.NS"
            try:
                df = data[ticker].dropna(how="all") if len(batch) > 1 else data.dropna(how="all")
            except Exception:
                continue
            if df is None or df.empty or len(df) < 260:  # need real multi-year runway
                continue
            histories[symbol] = df
        log.info(f"Walk-forward: fetched long history for {len(histories)}/{i + len(batch)} symbols so far...")
    return histories


# -----------------------------------------------------------------------
# Fold definitions
# -----------------------------------------------------------------------
def train_validation_oos_folds(train_end="2023-12-31", validation_end="2025-12-31", oos_end=None):
    """The simple three-way split from the review note:
      Train:      earliest available  -> train_end
      Validation: day after train_end -> validation_end
      Out-of-Sample: day after validation_end -> oos_end (default: today)
    Returns a list of (fold_name, start_date_or_None, end_date) tuples;
    start_date=None means "from the earliest data available" (resolved
    per-symbol at evaluation time, since different symbols have
    different amounts of history)."""
    train_end = pd.Timestamp(train_end)
    validation_end = pd.Timestamp(validation_end)
    oos_end = pd.Timestamp(oos_end) if oos_end else pd.Timestamp(dt.date.today())
    return [
        ("Train", None, train_end),
        ("Validation", train_end + pd.Timedelta(days=1), validation_end),
        ("Out-of-Sample", validation_end + pd.Timedelta(days=1), oos_end),
    ]


def rolling_walk_forward_folds(first_test_year=2022, last_test_year=2026, train_start_year=2018):
    """Expanding-window walk-forward, exactly as specified in the review
    note: Train always starts at train_start_year and grows by one year
    each fold; Test is always the single year immediately after Train's
    end. Returns a list of dicts:
      {"fold": "2022", "train_start": Timestamp, "train_end": Timestamp,
       "test_start": Timestamp, "test_end": Timestamp}
    """
    folds = []
    for test_year in range(first_test_year, last_test_year + 1):
        train_start = pd.Timestamp(f"{train_start_year}-01-01")
        train_end = pd.Timestamp(f"{test_year - 1}-12-31")
        test_start = pd.Timestamp(f"{test_year}-01-01")
        test_end = pd.Timestamp(f"{test_year}-12-31")
        folds.append({
            "fold": str(test_year),
            "train_start": train_start, "train_end": train_end,
            "test_start": test_start, "test_end": test_end,
        })
    return folds


# -----------------------------------------------------------------------
# Per-fold evaluation
# -----------------------------------------------------------------------
def _date_to_positional(df, target_date):
    """Positional index of the first bar at/after target_date, or None if
    target_date is past the end of df. target_date=None means 'no floor'
    (positional 0)."""
    if target_date is None:
        return 0
    idx = df.index.searchsorted(target_date)
    return int(idx) if idx < len(df) else None


def evaluate_pattern_in_window(df, detector_fn, window_start, window_end, min_gap_days=5):
    """Finds detector_fn's firings whose OWN signal date falls inside
    [window_start, window_end] (window_end=None means 'through the end
    of df'), simulates realistic trades for them (see
    utils.breakout_backtest.simulate_trades -- forward data past
    window_end is fine to use for MEASURING outcomes, see module
    docstring), and returns the trade list (not yet aggregated, so
    callers can pool across symbols before calling compute_trade_stats).
    Returns [] if there's no runway or nothing fired in-window."""
    scan_start_i = _date_to_positional(df, window_start)
    if scan_start_i is None:
        return []

    as_of_i = len(df) - 1
    if window_end is not None:
        end_i = _date_to_positional(df, window_end)
        # as_of_i is passed to find_historical_events as an EXCLUSIVE
        # upper bound net of max_horizon inside that function, so push
        # one bar past window_end to include window_end's own bar.
        as_of_i = min(len(df) - 1, (end_i + 1) if end_i is not None else len(df) - 1)

    event_indices = find_historical_events(
        df, detector_fn, as_of_i, min_gap_days=min_gap_days,
        max_events=10_000, scan_start_i=scan_start_i,
    )
    if not event_indices:
        return []
    return simulate_trades(df, event_indices)


def evaluate_universe_window(histories, detector_by_name, window_start, window_end, min_gap_days=5):
    """Runs evaluate_pattern_in_window for every (symbol, pattern) pair in
    `histories` x `detector_by_name`, POOLING trades across all symbols
    per pattern (see module docstring on why per-symbol samples are too
    thin to trust alone). Returns:
      {pattern_name: {"trade_stats": dict_or_None, "symbols_with_events": int}, ...}
    plus a special "_ALL_PATTERNS_POOLED" key pooling every pattern
    together, for a single top-level generalization check.
    """
    pooled_by_pattern = {name: [] for name in detector_by_name}
    symbols_with_events = {name: 0 for name in detector_by_name}
    all_trades = []

    for symbol, df in histories.items():
        for pattern_name, detector_fn in detector_by_name.items():
            try:
                trades = evaluate_pattern_in_window(df, detector_fn, window_start, window_end, min_gap_days)
            except Exception as e:
                log.warning(f"Walk-forward: {symbol}/{pattern_name} failed in window: {e}")
                continue
            if trades:
                pooled_by_pattern[pattern_name].extend(trades)
                symbols_with_events[pattern_name] += 1
                all_trades.extend(trades)

    result = {
        name: {
            "trade_stats": compute_trade_stats(trades),
            "symbols_with_events": symbols_with_events[name],
        }
        for name, trades in pooled_by_pattern.items()
    }
    result["_ALL_PATTERNS_POOLED"] = {
        "trade_stats": compute_trade_stats(all_trades),
        "symbols_with_events": len(histories),
    }
    return result


# -----------------------------------------------------------------------
# Drivers
# -----------------------------------------------------------------------
def run_train_validation_oos(histories, detector_by_name, train_end="2023-12-31", validation_end="2025-12-31", oos_end=None):
    """The simple 3-way split. Returns {fold_name: evaluate_universe_window(...) result}."""
    folds = train_validation_oos_folds(train_end, validation_end, oos_end)
    return {
        name: evaluate_universe_window(histories, detector_by_name, start, end)
        for name, start, end in folds
    }


def run_walk_forward(histories, detector_by_name, first_test_year=2022, last_test_year=2026, train_start_year=2018):
    """Expanding-window walk-forward. Returns a list of dicts:
      {"fold": "2022", "train": evaluate_universe_window(...) result,
       "test": evaluate_universe_window(...) result}
    one entry per test year."""
    folds = rolling_walk_forward_folds(first_test_year, last_test_year, train_start_year)
    results = []
    for f in folds:
        train_result = evaluate_universe_window(histories, detector_by_name, f["train_start"], f["train_end"])
        test_result = evaluate_universe_window(histories, detector_by_name, f["test_start"], f["test_end"])
        results.append({"fold": f["fold"], "train": train_result, "test": test_result})
    return results


# -----------------------------------------------------------------------
# Generalization diagnostics -- did what looked good in one fold hold up
# in the next?
# -----------------------------------------------------------------------
def compare_fold_generalization(fold_a, fold_b, fold_a_label="Train", fold_b_label="Test"):
    """
    fold_a / fold_b: results from evaluate_universe_window (same shape --
    {pattern_name: {"trade_stats": ..., ...}, "_ALL_PATTERNS_POOLED": ...}).

    For every pattern present with a usable sample in BOTH folds, flags:
      - hit-rate cleared CONFIRM_HIT_RATE_THRESHOLD in fold_a but not fold_b
        (the headline case the review note is about: a threshold picked
        by looking at one blend of history doesn't hold on unseen data)
      - expectancy sign flip (positive in fold_a, negative/zero in fold_b)
      - profit factor drops below 1.0 in fold_b having been above it in fold_a

    Returns a list of dicts, one per pattern evaluated:
      {"pattern": str, "fold_a_n": int, "fold_b_n": int,
       "fold_a_hit_rate": float, "fold_b_hit_rate": float,
       "fold_a_expectancy": float, "fold_b_expectancy": float,
       "fold_a_profit_factor": float, "fold_b_profit_factor": float,
       "flags": [str, ...]}   # empty list = generalized cleanly
    Patterns without a usable sample (< MIN_SAMPLES_FOR_CONFIDENCE) in
    EITHER fold are skipped -- not enough data in one fold to say
    anything about generalization either way.
    """
    rows = []
    pattern_names = [k for k in fold_a.keys() if k != "_ALL_PATTERNS_POOLED"]
    for name in pattern_names:
        a = fold_a.get(name, {}).get("trade_stats")
        b = fold_b.get(name, {}).get("trade_stats")
        if not a or not b or a["sample_size"] < MIN_SAMPLES_FOR_CONFIDENCE or b["sample_size"] < MIN_SAMPLES_FOR_CONFIDENCE:
            continue

        flags = []
        if a["win_rate"] >= CONFIRM_HIT_RATE_THRESHOLD and b["win_rate"] < CONFIRM_HIT_RATE_THRESHOLD:
            flags.append(
                f"hit-rate cleared the {CONFIRM_HIT_RATE_THRESHOLD*100:.0f}% Confirmed bar in {fold_a_label} "
                f"({a['win_rate']*100:.0f}%) but NOT in {fold_b_label} ({b['win_rate']*100:.0f}%) -- "
                f"threshold may be fit to {fold_a_label}, not a real edge"
            )
        if a["expectancy"] > 0 and b["expectancy"] <= 0:
            flags.append(
                f"expectancy flipped from positive in {fold_a_label} ({a['expectancy']*100:+.2f}%/trade) "
                f"to non-positive in {fold_b_label} ({b['expectancy']*100:+.2f}%/trade)"
            )
        if a["profit_factor"] >= 1.0 and b["profit_factor"] < 1.0:
            flags.append(
                f"profit factor dropped below 1.0 in {fold_b_label} ({b['profit_factor']:.2f}, "
                f"was {a['profit_factor']:.2f} in {fold_a_label})"
            )

        rows.append({
            "pattern": name,
            "fold_a_n": a["sample_size"], "fold_b_n": b["sample_size"],
            "fold_a_hit_rate": a["win_rate"], "fold_b_hit_rate": b["win_rate"],
            "fold_a_expectancy": a["expectancy"], "fold_b_expectancy": b["expectancy"],
            "fold_a_profit_factor": a["profit_factor"], "fold_b_profit_factor": b["profit_factor"],
            "flags": flags,
        })
    return rows


def format_generalization_report(rows, fold_a_label="Train", fold_b_label="Test"):
    """Plain-text summary table + flags, for console/log output -- not an
    HTML email block (this is an offline research tool, see module
    docstring)."""
    if not rows:
        return "No patterns had a usable sample (>= MIN_SAMPLES_FOR_CONFIDENCE) in both folds -- nothing to compare."

    lines = [f"{'Pattern':<28} {fold_a_label+' n':>9} {fold_b_label+' n':>8} "
             f"{fold_a_label+' hit%':>10} {fold_b_label+' hit%':>10} "
             f"{fold_a_label+' exp%':>10} {fold_b_label+' exp%':>10}  Flags"]
    for r in rows:
        lines.append(
            f"{r['pattern']:<28} {r['fold_a_n']:>9} {r['fold_b_n']:>8} "
            f"{r['fold_a_hit_rate']*100:>9.0f}% {r['fold_b_hit_rate']*100:>9.0f}% "
            f"{r['fold_a_expectancy']*100:>+9.2f}% {r['fold_b_expectancy']*100:>+9.2f}%  "
            f"{'; '.join(r['flags']) if r['flags'] else 'OK -- generalized'}"
        )
    n_flagged = sum(1 for r in rows if r["flags"])
    lines.append(f"\n{n_flagged}/{len(rows)} patterns flagged for a {fold_a_label}->{fold_b_label} generalization gap.")
    return "\n".join(lines)


if __name__ == "__main__":
    # Example wiring -- adjust imports to your actual module paths.
    # This block fetches long history itself (see the prerequisite note
    # at the top of this file) rather than reusing the live pipeline's 2y
    # pull, and is meant to be run manually, not scheduled alongside the
    # daily email job.
    import argparse
    from utils.nse_data import get_nifty500_symbols
    from utils.breakout_patterns import PATTERN_DETECTOR_BY_NAME

    parser = argparse.ArgumentParser(description="Walk-forward / out-of-sample validation for the breakout screener.")
    parser.add_argument("--mode", choices=["simple", "rolling"], default="simple")
    args = parser.parse_args()

    symbols, _ = get_nifty500_symbols()
    log.info(f"Walk-forward: fetching {HISTORY_PERIOD_FOR_WALKFORWARD} of history for {len(symbols)} symbols "
             f"(this is slow and is NOT the live daily pipeline's 2y fetch)...")
    histories = fetch_long_history(symbols)
    log.info(f"Walk-forward: {len(histories)}/{len(symbols)} symbols returned enough long-run history to use.")

    if args.mode == "simple":
        result = run_train_validation_oos(histories, PATTERN_DETECTOR_BY_NAME)
        print(format_generalization_report(
            compare_fold_generalization(result["Train"], result["Validation"], "Train", "Validation"),
            "Train", "Validation",
        ))
        print()
        print(format_generalization_report(
            compare_fold_generalization(result["Validation"], result["Out-of-Sample"], "Validation", "Out-of-Sample"),
            "Validation", "Out-of-Sample",
        ))
    else:
        for fold_result in run_walk_forward(histories, PATTERN_DETECTOR_BY_NAME):
            print(f"\n=== Test year {fold_result['fold']} ===")
            print(format_generalization_report(
                compare_fold_generalization(fold_result["train"], fold_result["test"], "Train", "Test"),
                "Train", "Test",
            ))