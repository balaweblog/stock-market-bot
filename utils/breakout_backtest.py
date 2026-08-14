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

# -----------------------------------------------------------------------
# Realistic trade simulation -- see simulate_trades() / compute_trade_stats()
# below. The hit-rate/avg-return read above answers "did price end up
# higher N days later" -- it says nothing about the PATH price took to
# get there, what it would have cost to actually trade it, or whether a
# high hit-rate is being propped up by a few large losers. A strategy
# can post an 80% hit-rate and still lose money (small wins, rare fat
# losses); a 55% hit-rate strategy with a good win/loss ratio can be far
# more profitable. Expectancy -- (Win% x Avg Win) - (Loss% x Avg Loss) --
# is the number that actually answers "is this worth trading", so it
# and the metrics that feed it (profit factor, drawdown, MAE/MFE, Sharpe/
# Sortino) sit alongside the existing hit-rate stats rather than
# replacing them.
# -----------------------------------------------------------------------
ATR_PERIOD_TRADE_SIM = 14

# Stop-loss placed this many ATRs below entry, checked against each
# day's LOW during the holding window -- exits on the earliest day the
# stop is touched rather than waiting for the fixed horizon. Without
# this, "average return at 10 days" hides trades that were down 15% on
# day 3 and merely happened to recover by day 10; no real trader holds
# through that blind to a stop.
STOP_LOSS_ATR_MULT = 2.0

# Cost assumptions -- TUNE THESE to the actual broker/account being
# modeled; these are placeholder, conservative-ish defaults for NSE
# equity delivery via a discount broker, not a substitute for real
# figures. Both are round-trip unless noted.
SLIPPAGE_BPS = 10          # each side (entry AND exit) fills this much worse
                            # than the reference price -- 20bps round-trip total
BROKERAGE_TAX_BPS = 15     # brokerage + STT + exchange/SEBI charges + stamp duty
                            # + GST, combined, round-trip, as a fraction of
                            # trade value

# Trades are simulated using the SAME event set find_historical_events()
# already produces (no separate, possibly-inconsistent replay), held for
# up to this many trading days unless the ATR stop fires first. Reuses
# PRIMARY_HORIZON so the "realistic" read is asking about the same
# holding period as the headline hit-rate stat, not a different question.
TRADE_SIM_HORIZON = PRIMARY_HORIZON


def find_historical_events(df, detector_fn, as_of_i, min_gap_days=5, max_events=60, max_horizon=None, scan_start_i=None):
    """
    Replays detector_fn across df[:as_of_i] (i.e. everything strictly
    BEFORE today's signal, so today's own occurrence never leaks into
    its own sample) and returns the list of historical positional
    indices where it would have fired -- the SAME no-lookahead event
    set backtest_pattern() below uses for its hit-rate stats, extracted
    here so other modules (utils.breakout_failure's failure-rate
    backtest) can replay the identical historical occurrences rather
    than re-deriving a possibly-different set for what's meant to be
    the "same" pattern history.

    min_gap_days de-duplicates overlapping/adjacent firings of the same
    setup (e.g. a resistance breakout that keeps re-triggering for a
    week straight) into one event, so the sample isn't artificially
    inflated by one prolonged move.

    max_horizon: only scan up to as_of_i - max_horizon so every event
    has a FULL forward window available to whatever the caller measures
    -- otherwise a signal from 3 days ago would only have a 3-day-old
    forward read, not a true one, biasing results toward whatever partial
    move happened to occur. Defaults to max(FORWARD_HORIZONS) so callers
    that don't pass their own horizon still get a safe, conservative cutoff
    consistent with backtest_pattern's own scan.

    scan_start_i: positional floor for the scan, default 60 (the
    detector's own minimum-lookback requirement -- unrelated to and not
    a substitute for that floor, just an additional lower bound). Lets a
    caller bound the scan to a specific WINDOW of history rather than
    always starting at the beginning of df -- e.g. utils.breakout_walkforward
    uses this to keep each walk-forward fold's event set confined to
    that fold's own date range, so a "Train" fold can't see events that
    actually belong to "Test". Always clamped to >= 60 regardless of
    what's passed, since detectors need that much trailing data to run
    at all.

    Returns [] if there's not enough runway to scan at all, or if the
    pattern simply never fired historically -- callers distinguish "no
    runway" from "zero events" via as_of_i themselves if they need to.
    """
    if max_horizon is None:
        max_horizon = max(FORWARD_HORIZONS)
    if as_of_i < 60:
        return []

    scan_start = max(60, scan_start_i) if scan_start_i is not None else 60
    scan_end = as_of_i - max_horizon
    if scan_end < scan_start:
        return []

    event_indices = []
    last_event_i = scan_start - min_gap_days - 1
    for hist_i in range(scan_start, scan_end):
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
    return event_indices


def backtest_pattern(df, detector_fn, as_of_i, min_gap_days=5, max_events=60):
    """
    Builds forward-return stats from find_historical_events() above:
    records forward returns at each horizon in FORWARD_HORIZONS for
    every historical firing, and returns a stats dict.

    Returns:
      {
        "sample_size": int,
        "horizons": {5: {"hit_rate": float, "avg_return": float, "median_return": float}, ...},
        "confirmed": bool,   # sample_size >= MIN_SAMPLES_FOR_CONFIDENCE
                              # and PRIMARY_HORIZON hit_rate >= CONFIRM_HIT_RATE_THRESHOLD
        "trade_stats": dict or None,  # see compute_trade_stats() -- win_rate,
                              # avg_winner, avg_loser, expectancy, profit_factor,
                              # max_drawdown, sharpe, sortino, avg_holding_days,
                              # avg_mae, avg_mfe, avg_gap, stop_out_rate,
                              # gap_risk_rate, and the cost assumptions used
      }
    or None if there's not enough trailing history to attempt a
    backtest at all (as opposed to "attempted, zero historical events found").
    """
    max_horizon = max(FORWARD_HORIZONS)
    if as_of_i < 60:  # not enough runway to even try
        return None

    closes = df["Close"].values
    event_indices = find_historical_events(
        df, detector_fn, as_of_i, min_gap_days=min_gap_days, max_events=max_events, max_horizon=max_horizon,
    )

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

    # Additive, non-breaking: a realistic trade-level simulation (entry
    # next-bar-open, ATR stop, slippage/costs, MAE/MFE) alongside the
    # same-close/fixed-horizon hit-rate stats above -- see module
    # docstring and simulate_trades()/compute_trade_stats() for why hit-
    # rate alone can be a misleading thing to optimize. None if there
    # weren't enough valid next-bar entries to simulate anything.
    trades = simulate_trades(df, event_indices)
    trade_stats = compute_trade_stats(trades)

    return {
        "sample_size": sample_size,
        "horizons": horizons,
        "confirmed": confirmed,
        "trade_stats": trade_stats,
    }


def _atr_series(df, period=ATR_PERIOD_TRADE_SIM):
    """Plain rolling ATR (simple mean of True Range, not Wilder-smoothed --
    consistent with the ATR already computed elsewhere in this project).
    Returns a numpy array aligned to df's rows; entries before `period`
    prior closes exist are NaN. Purely historical inputs (High/Low/Close
    up to and including each row) -- safe to compute over the full df
    since callers only ever index into positions that were already
    no-lookahead-safe to begin with."""
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    n = len(closes)
    atr = np.full(n, np.nan)
    for i in range(period, n):
        trs = []
        for j in range(i - period + 1, i + 1):
            prior_close = closes[j - 1]
            tr = max(
                highs[j] - lows[j],
                abs(highs[j] - prior_close),
                abs(lows[j] - prior_close),
            )
            trs.append(tr)
        atr[i] = sum(trs) / len(trs)
    return atr


def simulate_trades(
    df,
    event_indices,
    horizon=None,
    stop_loss_atr_mult=STOP_LOSS_ATR_MULT,
    slippage_bps=SLIPPAGE_BPS,
    cost_bps=BROKERAGE_TAX_BPS,
):
    """
    Turns raw event indices (from find_historical_events) into simulated
    ROUND-TRIP TRADES instead of a same-close-to-N-days-later snapshot:

      Entry: NEXT bar's OPEN after the signal bar -- the earliest a real
        order could actually be placed, since the signal itself is only
        known once bar `ei` has closed. (backtest_pattern()'s own
        hit-rate stats above use closes[ei] as a same-day reference
        price for simplicity/backward-compat; this is the more
        realistic entry.)
      Exit: whichever comes FIRST of (a) an ATR-based stop-loss touched
        intraday (checked against each day's LOW), or (b) `horizon`
        trading days after entry, exiting at that day's close.
      Costs: slippage_bps applied against the trader on BOTH entry and
        exit (buy fills higher, sell fills lower); cost_bps (brokerage +
        taxes, round-trip) subtracted from the gross return.
      MAE / MFE: the worst and best mark-to-market excursion (using
        daily High/Low) at any point during the actual holding window --
        NOT the same as the exit return, and computed even for trades
        that exit at the horizon rather than the stop.
      Gap risk: the overnight gap (entry-day open vs. signal-day close)
        is recorded per trade -- this is the risk a fixed stop can't
        protect against, since a stock can open BELOW the stop.

    Returns a list of trade dicts (empty if no event produced a valid
    next-bar entry). Each dict:
      {event_i, entry_i, exit_i, entry_date, exit_date, entry_price,
       exit_price, gross_return, net_return, holding_days, exit_reason
       ("stop"|"horizon"), mae, mfe, gap}
    """
    if horizon is None:
        horizon = TRADE_SIM_HORIZON

    opens = df["Open"].values
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    dates = df.index
    atr = _atr_series(df)
    n = len(closes)

    trades = []
    for ei in event_indices:
        entry_i = ei + 1
        if entry_i >= n:
            continue  # signal fired too recently in this replay to have a next bar

        raw_entry = opens[entry_i]
        signal_close = closes[ei]
        if raw_entry is None or raw_entry <= 0 or signal_close is None or signal_close <= 0:
            continue

        gap = (raw_entry - signal_close) / signal_close
        entry_price = raw_entry * (1 + slippage_bps / 10000.0)  # buy slips up

        entry_atr = atr[ei]
        stop_price = (
            entry_price - stop_loss_atr_mult * entry_atr
            if entry_atr is not None and not np.isnan(entry_atr)
            else None
        )

        mae = 0.0  # most negative mark-to-market excursion, as a fraction (0 or negative)
        mfe = 0.0  # most positive mark-to-market excursion, as a fraction (0 or positive)
        exit_i = None
        exit_reason = "horizon"
        max_i = min(entry_i + horizon, n - 1)

        for j in range(entry_i, max_i + 1):
            day_low = lows[j]
            day_high = highs[j]
            mae = min(mae, (day_low - entry_price) / entry_price)
            mfe = max(mfe, (day_high - entry_price) / entry_price)
            if stop_price is not None and day_low <= stop_price:
                exit_i = j
                exit_reason = "stop"
                raw_exit = stop_price  # conservative: assumes fill at the stop itself,
                break                   # not the (possibly worse) day's actual low
        if exit_i is None:
            exit_i = max_i
            raw_exit = closes[exit_i]

        exit_price = raw_exit * (1 - slippage_bps / 10000.0)  # sell slips down
        gross_return = (exit_price - entry_price) / entry_price
        net_return = gross_return - (cost_bps / 10000.0)
        holding_days = exit_i - entry_i

        trades.append({
            "event_i": ei,
            "entry_i": entry_i,
            "exit_i": exit_i,
            "entry_date": dates[entry_i],
            "exit_date": dates[exit_i],
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "gross_return": float(gross_return),
            "net_return": float(net_return),
            "holding_days": int(holding_days),
            "exit_reason": exit_reason,
            "mae": float(mae),
            "mfe": float(mfe),
            "gap": float(gap),
        })

    return trades


def compute_trade_stats(trades):
    """
    Aggregates simulate_trades() output into the metrics that actually
    say whether a pattern is worth trading, not just whether it "worked":

      win_rate, avg_winner, avg_loser  -- avg_loser is the raw (negative)
        mean of losing trades, so it's a magnitude with sign attached.
      expectancy  -- (Win% x Avg Win) - (Loss% x Avg Loser-magnitude),
        computed here as win_rate*avg_winner + loss_rate*avg_loser since
        avg_loser is already negative; per-trade expected return, net of
        costs. THIS is the number that should drive whether a pattern is
        worth acting on -- a high hit-rate with negative expectancy is a
        losing strategy, and a mediocre hit-rate with positive
        expectancy can be a good one. See module docstring above.
      profit_factor  -- gross profit / gross loss (both positive
        magnitudes). >1 means the winners paid for the losers; below 1
        means they didn't, regardless of hit-rate. inf if there were
        zero losing trades.
      max_drawdown  -- peak-to-trough decline of an equity curve built
        by compounding trades IN CHRONOLOGICAL ORDER of entry, one after
        another. Simplification: this assumes trades are taken
        sequentially (one position in this pattern/stock at a time), not
        a portfolio running many symbols concurrently -- true portfolio-
        level drawdown depends on position sizing and overlap across the
        whole strategy, which this single-pattern/single-stock replay
        can't see.
      sharpe / sortino  -- computed on per-trade net returns (mean /
        std, and mean / downside-std respectively), then ANNUALIZED by
        scaling by sqrt(trades-per-year), where trades-per-year is
        estimated from the actual calendar span between first and last
        trade in the sample. With small samples this estimate is noisy --
        treat as directional, not precise, and prefer the per-trade
        (unannualized) figures on thin samples.
      avg_holding_days, avg_mae, avg_mfe, avg_gap  -- self-explanatory
        means across trades.
      stop_out_rate  -- fraction of trades that exited on the stop
        rather than surviving to the horizon.
      gap_risk  -- fraction of trades whose overnight entry gap moved
        AGAINST the position by more than 1% -- the risk a same-day stop
        can't protect against, since the stock can open through it.

    Returns None if `trades` is empty.
    """
    if not trades:
        return None

    rets = np.array([t["net_return"] for t in trades])
    wins = rets[rets > 0]
    losses = rets[rets <= 0]

    win_rate = float(len(wins) / len(rets))
    loss_rate = 1.0 - win_rate
    avg_winner = float(wins.mean()) if len(wins) else 0.0
    avg_loser = float(losses.mean()) if len(losses) else 0.0  # <= 0
    expectancy = win_rate * avg_winner + loss_rate * avg_loser

    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())  # positive magnitude
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = float("inf") if gross_profit > 0 else 0.0

    trades_sorted = sorted(trades, key=lambda t: t["entry_i"])
    equity = [1.0]
    for t in trades_sorted:
        equity.append(equity[-1] * (1 + t["net_return"]))
    equity = np.array(equity)
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / running_max
    max_drawdown = float(drawdowns.min())

    mean_ret = float(rets.mean())
    std_ret = float(rets.std(ddof=1)) if len(rets) > 1 else 0.0
    sharpe_per_trade = (mean_ret / std_ret) if std_ret > 0 else None

    downside = rets[rets < 0]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sortino_per_trade = (mean_ret / downside_std) if downside_std > 0 else None

    first_date, last_date = trades_sorted[0]["entry_date"], trades_sorted[-1]["entry_date"]
    span_days = (last_date - first_date).days if last_date != first_date else None
    trades_per_year = (len(trades) / (span_days / 365.25)) if span_days else None

    sharpe = (sharpe_per_trade * (trades_per_year ** 0.5)) if (sharpe_per_trade is not None and trades_per_year) else sharpe_per_trade
    sortino = (sortino_per_trade * (trades_per_year ** 0.5)) if (sortino_per_trade is not None and trades_per_year) else sortino_per_trade

    adverse_gap_count = sum(1 for t in trades if t["gap"] < -0.01)

    return {
        "sample_size": len(trades),
        "win_rate": win_rate,
        "avg_winner": avg_winner,
        "avg_loser": avg_loser,
        "expectancy": expectancy,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "sortino": sortino,
        "sharpe_per_trade": sharpe_per_trade,
        "sortino_per_trade": sortino_per_trade,
        "trades_per_year_est": trades_per_year,
        "avg_holding_days": float(np.mean([t["holding_days"] for t in trades])),
        "avg_mae": float(np.mean([t["mae"] for t in trades])),
        "avg_mfe": float(np.mean([t["mfe"] for t in trades])),
        "avg_gap": float(np.mean([t["gap"] for t in trades])),
        "stop_out_rate": float(np.mean([1.0 if t["exit_reason"] == "stop" else 0.0 for t in trades])),
        "gap_risk_rate": float(adverse_gap_count / len(trades)),
        "slippage_bps": SLIPPAGE_BPS,
        "cost_bps": BROKERAGE_TAX_BPS,
        "stop_loss_atr_mult": STOP_LOSS_ATR_MULT,
    }


def backtest_signal(symbol, df, detector_fn, as_of_i):
    """Thin wrapper with logging -- used by the controller so a slow or
    failing backtest for one symbol/pattern never takes down the run."""
    try:
        return backtest_pattern(df, detector_fn, as_of_i)
    except Exception as e:
        log.warning(f"Breakout Screener: backtest failed for {symbol} ({detector_fn.__name__}): {e}")
        return None


# -----------------------------------------------------------------------
# Breakout Quality Score
# -----------------------------------------------------------------------
# A single 0-100 number so signals can be ranked/skimmed at a glance,
# built ENTIRELY from numbers already computed above (plus the
# data-quality notes from utils.nse_data) -- no hidden inputs, no ML
# black box, so every score is explainable by pointing at its breakdown.
#
# Weights (sum to 100):
#   45  Historical hit-rate at PRIMARY_HORIZON -- the main question:
#       "when this exact pattern fired on this exact stock before, did
#       it actually work?"
#   20  Sample confidence -- a 90% hit-rate over 4 occurrences is a lot
#       less trustworthy than 65% over 25, so more history earns more score.
#   15  Average return magnitude at PRIMARY_HORIZON -- a positive hit-rate
#       on trivially small moves scores lower than one with real magnitude.
#   10  Cross-horizon consistency -- does the edge hold at 5/10/20 days,
#       or is it a one-horizon coincidence?
#   10  Data-quality -- caution notes (large single-day move, missing
#       bhavcopy cross-check, etc.) pull the score down since they mean
#       "trust this a bit less," even if the backtest itself looks good.
QUALITY_WEIGHTS = {
    "hit_rate": 45,
    "sample_confidence": 20,
    "return_magnitude": 15,
    "consistency": 10,
    "data_quality": 10,
}

# Hit-rate scoring band: at/below this is "no better than a coin flip",
# at/above this is full marks for this component.
HIT_RATE_FLOOR = 0.50
HIT_RATE_CEILING = 0.85

# Sample-size scoring band: full marks at this many historical occurrences.
SAMPLE_SIZE_FOR_FULL_CONFIDENCE = 20

# Return-magnitude scoring band (at PRIMARY_HORIZON): full marks at/above this.
RETURN_MAGNITUDE_CEILING = 0.05  # 5%

QUALITY_LABELS = (
    (80, "Excellent"),
    (65, "Strong"),
    (50, "Moderate"),
    (35, "Weak"),
    (0, "Poor"),
)


def _clamp01(x):
    return max(0.0, min(1.0, x))


def _quality_label(score):
    for threshold, label in QUALITY_LABELS:
        if score >= threshold:
            return label
    return QUALITY_LABELS[-1][1]


def compute_quality_score(bt, dq_notes=None):
    """
    bt: the dict returned by backtest_pattern (or None / zero-sample).
    dq_notes: the data-quality caution notes list for this symbol
      (utils.nse_data.data_quality_check), or None.

    Returns None if there's no historical sample at all (score would be
    meaningless -- the caller should show "No historical sample yet"
    instead of a fabricated number), otherwise:
      {"score": int 0-100, "label": str, "breakdown": {component: int, ...}}
    """
    if not bt or not bt.get("horizons") or bt.get("sample_size", 0) == 0:
        return None

    primary = bt["horizons"].get(PRIMARY_HORIZON)
    if not primary:
        return None

    # -- Hit-rate component
    hit_rate_frac = _clamp01((primary["hit_rate"] - HIT_RATE_FLOOR) / (HIT_RATE_CEILING - HIT_RATE_FLOOR))

    # -- Sample confidence component
    sample_frac = _clamp01(bt["sample_size"] / SAMPLE_SIZE_FOR_FULL_CONFIDENCE)

    # -- Return magnitude component (negative avg return scores 0, not negative)
    return_frac = _clamp01(primary["avg_return"] / RETURN_MAGNITUDE_CEILING)

    # -- Cross-horizon consistency component
    horizons_available = [h for h in FORWARD_HORIZONS if h in bt["horizons"]]
    horizons_positive = [h for h in horizons_available if bt["horizons"][h]["hit_rate"] >= HIT_RATE_FLOOR]
    consistency_frac = (len(horizons_positive) / len(horizons_available)) if horizons_available else 0.0

    # -- Data-quality component (each caution note costs 35 points of this sub-score, floor 0)
    dq_frac = _clamp01(1.0 - 0.35 * len(dq_notes or []))

    components = {
        "hit_rate": hit_rate_frac * QUALITY_WEIGHTS["hit_rate"],
        "sample_confidence": sample_frac * QUALITY_WEIGHTS["sample_confidence"],
        "return_magnitude": return_frac * QUALITY_WEIGHTS["return_magnitude"],
        "consistency": consistency_frac * QUALITY_WEIGHTS["consistency"],
        "data_quality": dq_frac * QUALITY_WEIGHTS["data_quality"],
    }
    score = round(sum(components.values()))

    return {
        "score": score,
        "label": _quality_label(score),
        "breakdown": {k: round(v, 1) for k, v in components.items()},
    }


# -----------------------------------------------------------------------
# Setup Score -- a calibrated blend of the FOUR separately-computed reads
# a row already carries by the time scan_universe() is done with it:
#   - Quality Score        (this module)      -- has this pattern worked
#                                                  on this stock before?
#   - Confirmation ratio    (utils.breakout_confirmation) -- how many of
#                                                  today's independent
#                                                  checks agree?
#   - Failure Risk          (utils.breakout_failure)      -- how often
#                                                  has this exact setup
#                                                  round-tripped within
#                                                  FAILURE_WINDOW_DAYS?
#   - Risk:Reward           (utils.breakout_targets)      -- worth
#                                                  taking even if right?
#
# Today these sit in four separate columns with no combined read -- a
# row with an 82 Quality Score but only 4/8 Confirmation checks and an
# elevated historical Failure Rate looks strong on the first column and
# weak on the other two, and nothing forces those to be weighed
# together. This is a logistic blend (a bounded weighted sum through a
# sigmoid, same shape as a simple logistic-regression score) rather
# than a linear average, so a genuinely bad read on ANY one component
# pulls the composite down hard instead of being diluted by three good
# ones -- a 95 Quality Score paired with a coin-flip Confirmation ratio
# should NOT still land the composite near "Excellent".
#
# PURELY INFORMATIONAL, same as Quality/Confirmation themselves -- this
# does not change cleared_backtest / meets_confirmed_bar / R:R filter
# logic in breakout_controller.scan_universe. It's a fifth, additive
# column, not a new gate.
# -----------------------------------------------------------------------
SETUP_SCORE_WEIGHTS = {
    "quality": 2.2,          # backtest-derived Quality Score, rescaled to [-1, 1]
    "confirmation": 1.6,     # today's checklist ratio, rescaled to [-1, 1]
    "failure_risk": 1.4,     # historical same-pattern failure rate, inverted
    "risk_reward": 1.0,      # R:R ratio, rescaled around the preferred bar
}
SETUP_SCORE_BIAS = -0.4      # slight pessimistic prior: an average row on every
                              # component (all rescaled inputs at 0) should score
                              # a bit below 50, since staying out of a mediocre
                              # setup is the cheaper mistake to make


def _sigmoid(x: float) -> float:
    # Guard against overflow on pathological inputs -- inputs here are
    # already bounded to roughly [-1, 1] per component before weighting,
    # so this is a safety margin, not a normal code path.
    x = max(-50.0, min(50.0, x))
    return 1.0 / (1.0 + np.exp(-x))


def compute_setup_score(quality, confirmation, failure_risk, risk_reward):
    """
    quality: dict from compute_quality_score() above, or None.
    confirmation: ConfirmationScore from utils.breakout_confirmation, or None.
    failure_risk: FailureAssessment from utils.breakout_failure, or None.
    risk_reward: dict from utils.breakout_targets.compute_risk_reward(), or None.

    Each component is rescaled to roughly [-1, +1] (0 = "neutral/unknown",
    positive = "supportive", negative = "concerning") before being
    weighted and passed through a sigmoid, so a missing component
    contributes 0 (no opinion) rather than being penalized as if it had
    failed -- the same "an unavailable check shouldn't manufacture a
    false negative" fail-soft convention the rest of this project uses.

    Returns None only if EVERY component is unavailable (nothing to
    score), otherwise {"score": int 0-100, "label": str, "components_used": int}.
    """
    terms = []

    if quality is not None:
        # Quality Score is already 0-100 with 50 as its own "coin flip"
        # anchor (see HIT_RATE_FLOOR/CEILING banding above) -- rescale
        # around that same midpoint.
        terms.append(SETUP_SCORE_WEIGHTS["quality"] * ((quality["score"] - 50.0) / 50.0))

    if confirmation is not None and confirmation.available_count > 0:
        frac = confirmation.passed_count / confirmation.available_count
        terms.append(SETUP_SCORE_WEIGHTS["confirmation"] * ((frac - 0.5) * 2))

    if failure_risk is not None and failure_risk.backtest is not None and failure_risk.backtest.failure_rate is not None:
        # Invert: a LOW historical failure rate is supportive (+),
        # a HIGH one is concerning (-). Centered on
        # HIGH_FAILURE_RATE_CEILING so "elevated" reads clearly negative.
        rate = failure_risk.backtest.failure_rate
        terms.append(SETUP_SCORE_WEIGHTS["failure_risk"] * ((HIGH_FAILURE_RATE_CEILING - rate) / HIGH_FAILURE_RATE_CEILING))
    if failure_risk is not None and failure_risk.same_day.state.value == "Caution":
        # Same-day Caution (2/4 bull-trap flags -- short of the 3/4 that
        # would have pulled the row into Failed Breakout and out of
        # scoring entirely) still deserves a small penalty here, since
        # it's a live same-day signal, not a historical base rate.
        terms.append(-0.5)

    if risk_reward is not None and risk_reward.get("ratio") is not None:
        from utils.breakout_targets import RISK_REWARD_MIN_THRESHOLD, RISK_REWARD_PREFERRED_THRESHOLD
        ratio = risk_reward["ratio"]
        # 0 at the MIN threshold (barely investable), +1 at/above the
        # PREFERRED threshold, negative below MIN (this row would
        # normally already be filtered out of Confirmed/Watch by R:R,
        # but the score stays honest if ever called on an unfiltered row).
        span = RISK_REWARD_PREFERRED_THRESHOLD - RISK_REWARD_MIN_THRESHOLD
        terms.append(SETUP_SCORE_WEIGHTS["risk_reward"] * max(-1.0, min(1.0, (ratio - RISK_REWARD_MIN_THRESHOLD) / span)))

    if not terms:
        return None

    raw = SETUP_SCORE_BIAS + sum(terms)
    score = round(_sigmoid(raw) * 100)
    label = _quality_label(score)  # reuse the same Excellent/Strong/Moderate/Weak/Poor bands
    return {"score": score, "label": label, "components_used": len(terms)}