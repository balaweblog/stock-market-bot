"""
utils/breakout_backtest.py

Live, per-stock, per-pattern backtest -- run at scan time, not
pre-computed. For every pattern flagged TODAY for a stock, this replays
the SAME detector function across that stock's own trailing history to
see how that pattern actually performed the last time(s) it fired,
before the signal is allowed into the "Confirmed" section of the email.

No-lookahead discipline: for each historical day i where the detector
would have fired, only df.iloc[:i+1] existed at the time -- exactly what
the detector itself is restricted to (see breakout_patterns.py's module
docstring). Forward returns are then measured strictly AFTER i, using
data that would only have arrived later. This is what makes the
backtest's hit-rate meaningful instead of hindsight-biased.
"""

import numpy as np
from utils.logger import log

FORWARD_HORIZONS = (5, 10, 20)  # trading days
MIN_SAMPLES_FOR_CONFIDENCE = 5
CONFIRM_HIT_RATE_THRESHOLD = 0.55  # >=55% of past occurrences positive at the primary horizon
PRIMARY_HORIZON = 10


def backtest_pattern(df, detector_fn, as_of_i, min_gap_days=5, max_events=60):
    """
    Replays detector_fn across df[:as_of_i] (i.e. everything strictly
    BEFORE today's signal, so today's own occurrence never leaks into
    its own backtest sample), records forward returns at each horizon in
    FORWARD_HORIZONS for every historical firing, and returns a stats
    dict.

    min_gap_days de-duplicates overlapping/adjacent firings of the same
    setup (e.g. a resistance breakout that keeps re-triggering for a
    week straight) into one event, so the sample isn't artificially
    inflated by one prolonged move.

    Returns:
      {
        "sample_size": int,
        "horizons": {5: {"hit_rate": float, "avg_return": float, "median_return": float}, ...},
        "confirmed": bool,   # sample_size >= MIN_SAMPLES_FOR_CONFIDENCE
                              # and PRIMARY_HORIZON hit_rate >= CONFIRM_HIT_RATE_THRESHOLD
      }
    or None if there's not enough trailing history to attempt a
    backtest at all (as opposed to "attempted, zero historical events found").
    """
    max_horizon = max(FORWARD_HORIZONS)
    if as_of_i < 60:  # not enough runway to even try
        return None

    closes = df["Close"].values
    event_indices = []
    last_event_i = -min_gap_days - 1

    # Only scan up to as_of_i - max_horizon so every historical event has
    # a FULL forward window available to measure -- otherwise a signal
    # from 3 days ago would only have a 3-day-old forward return, not a
    # true 5/10/20-day one, and would bias hit-rates toward whatever
    # partial move happened to occur.
    scan_end = as_of_i - max_horizon
    if scan_end < 60:
        return {"sample_size": 0, "horizons": {}, "confirmed": False}

    for hist_i in range(60, scan_end):
        if hist_i - last_event_i < min_gap_days:
            continue
        try:
            fired = detector_fn(df, hist_i)
        except Exception:
            fired = None
        if fired:
            event_indices.append(hist_i)
            last_event_i = hist_i
            if len(event_indices) >= max_events:
                break

    if not event_indices:
        return {"sample_size": 0, "horizons": {}, "confirmed": False}

    horizon_returns = {h: [] for h in FORWARD_HORIZONS}
    for ei in event_indices:
        entry_price = closes[ei]
        if entry_price <= 0:
            continue
        for h in FORWARD_HORIZONS:
            exit_i = ei + h
            if exit_i < len(closes):
                horizon_returns[h].append((closes[exit_i] - entry_price) / entry_price)

    horizons = {}
    for h, rets in horizon_returns.items():
        if not rets:
            continue
        rets_arr = np.array(rets)
        horizons[h] = {
            "hit_rate": float((rets_arr > 0).mean()),
            "avg_return": float(rets_arr.mean()),
            "median_return": float(np.median(rets_arr)),
            "sample_size": int(len(rets_arr)),
        }

    sample_size = len(event_indices)
    primary = horizons.get(PRIMARY_HORIZON)
    confirmed = bool(
        sample_size >= MIN_SAMPLES_FOR_CONFIDENCE
        and primary is not None
        and primary["hit_rate"] >= CONFIRM_HIT_RATE_THRESHOLD
    )

    return {"sample_size": sample_size, "horizons": horizons, "confirmed": confirmed}


def backtest_signal(symbol, df, detector_fn, as_of_i):
    """Thin wrapper with logging -- used by the controller so a slow or
    failing backtest for one symbol/pattern never takes down the run."""
    try:
        return backtest_pattern(df, detector_fn, as_of_i)
    except Exception as e:
        log.warning(f"Breakout Screener: backtest failed for {symbol} ({detector_fn.__name__}): {e}")
        return None