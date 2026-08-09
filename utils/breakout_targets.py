"""
utils/breakout_targets.py

Technical price targets layered on top of an entry price (from
utils.entry_classification) and the pattern signal that fired
(from utils.breakout_patterns). This is what turns "average +5.0% at
10D" -- a useful but abstract backtest statistic -- into concrete
price levels a discretionary trader would actually plan around.

Three independent targets, each computed a different, explainable way
-- same "every number traces back to a specific calculation, no hidden
inputs" discipline as the rest of this project (see
utils.breakout_backtest's Quality Score docstring):

  Target 1 -- Previous resistance
      The nearest prior swing-high level of df["High"] that still sits
      above the entry price, found in the stock's OWN trailing history.
      Answers "what's the next overhead supply zone this stock has
      already respected before?" Only ever looks at df.iloc[:i] --
      strictly before today's signal bar -- the same no-lookahead
      discipline enforced in utils.breakout_patterns and
      utils.breakout_backtest.

  Target 2 -- Pattern-measured move
      The classic technical-analysis projection: take the *height* of
      whatever base/pole/band/cup produced today's breakout and project
      it forward from the breakout point (e.g. cup depth added to the
      rim, flagpole move added to the flag high). Each pattern detector
      in utils.breakout_patterns already computes this height as part
      of firing the signal and now stashes the raw numbers in its
      returned dict -- this module reuses them directly rather than
      re-deriving (and risking disagreeing with) what actually fired.
      Patterns with no natural geometry (Volume Surge, 52-Week High,
      Golden Cross) fall back to an ATR-multiple projection, clearly
      labeled as a fallback.

  Target 3 -- Fibonacci extension / major resistance
      A 1.618x extension of that same base height, projected from the
      breakout point -- OR the highest prior swing high still found in
      the lookback window, whichever is FURTHER. A "major" target
      shouldn't be reported as closer than a level the stock has
      already proven it can reach.

None of these are guarantees or predictions -- they're transparent,
replayable projections, surfaced alongside (not instead of) the
backtest's own hit-rate/avg-return numbers.

Risk/Reward
    compute_risk_reward() below turns a target into a decision input,
    not just a number to admire: it measures reward (nearest available
    target minus entry) against risk (entry minus the stop-loss from
    utils.entry_classification) and returns a single R:R ratio.
    breakout_controller.py treats this as a CORE FILTER, not decoration
    -- setups with R:R < RISK_REWARD_MIN_THRESHOLD are pulled out of
    Confirmed/Watch into their own labeled section rather than shown
    as if they were equally good, and R:R >= RISK_REWARD_PREFERRED_THRESHOLD
    is called out as the ones worth prioritizing.
"""

import numpy as np
from scipy.signal import argrelextrema

from utils.logger import log

# Risk/Reward thresholds -- shared with breakout_controller.py so the
# filter and the badge coloring in the email always agree with each other.
RISK_REWARD_MIN_THRESHOLD = 1.5        # below this, filtered out of Confirmed/Watch
RISK_REWARD_PREFERRED_THRESHOLD = 2.0  # at/above this, flagged as a priority setup


# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
class TargetsConfig:
    def __init__(
        self,
        prior_resistance_lookback: int = 252,   # ~1 trading year of history to scan for swing highs
        swing_order: int = 5,                   # local-max comparison window used to define a "swing high"
        measured_move_atr_fallback_mult: float = 3.0,
        fib_extension_ratio: float = 1.618,
    ):
        self.prior_resistance_lookback = prior_resistance_lookback
        self.swing_order = swing_order
        self.measured_move_atr_fallback_mult = measured_move_atr_fallback_mult
        self.fib_extension_ratio = fib_extension_ratio


# -----------------------------------------------------------------------
# Shared structural scan -- used by BOTH Target 1 and Target 3, so the
# two never disagree about what "the next swing high" actually is.
# -----------------------------------------------------------------------
def _find_swing_highs_above(df, i, entry_price, lookback, order):
    """Local maxima of df['High'] within df.iloc[:i] (strictly before
    today -- no lookahead) that sit above entry_price. Returns a sorted
    ascending list of distinct price levels (nearest overhead level
    first, highest/furthest last)."""
    start = max(0, i - lookback)
    window = df.iloc[start:i]
    if len(window) < order * 2 + 1:
        return []
    highs = window["High"].values
    idx = argrelextrema(highs, np.greater_equal, order=order)[0]
    levels = sorted({round(float(highs[j]), 2) for j in idx if highs[j] > entry_price})
    return levels


# -----------------------------------------------------------------------
# Per-pattern "height" -- the base/pole/cup/band whose size projects
# forward as the measured move. Pulled straight from the raw numbers
# each detector in breakout_patterns.py now stashes in its signal dict.
# -----------------------------------------------------------------------
def _pattern_swing(pattern_name, sig):
    """Returns (low, high, basis_label) describing the geometric base
    that produced this pattern's breakout, or None if this pattern has
    no natural measured-move geometry (Volume Surge, 52-Week High,
    Golden Cross -- these are momentum/trend signals, not base
    breakouts, so there's no "height" to project)."""
    try:
        if pattern_name == "Resistance Breakout":
            height = sig["resistance"] - sig["support"]
            return sig["support"], sig["resistance"], f"₹{height:.2f} range of the prior consolidation"
        if pattern_name == "Bollinger Squeeze Breakout":
            height = sig["band_upper"] - sig["band_lower"]
            return sig["band_lower"], sig["band_upper"], f"₹{height:.2f} band width at the squeeze"
        if pattern_name == "Bull Flag Breakout":
            lo, hi = sorted((sig["pole_start"], sig["pole_end"]))
            return lo, hi, f"₹{hi - lo:.2f} flagpole move"
        if pattern_name == "Triangle Breakout":
            hi = sig["upper_trendline"]
            lo = hi - sig["early_range"]
            return lo, hi, f"₹{sig['early_range']:.2f} widest part of the triangle"
        if pattern_name == "Cup and Handle Breakout":
            height = sig["rim_level"] - sig["cup_low"]
            return sig["cup_low"], sig["rim_level"], f"₹{height:.2f} cup depth"
    except KeyError:
        # Signal dict predates the raw-field additions, or a detector was
        # renamed/changed without updating this mapping -- fail soft into
        # the ATR fallback rather than raising.
        return None
    return None


# -----------------------------------------------------------------------
# Target 1 -- Previous resistance
# -----------------------------------------------------------------------
def _target1(levels, entry_price, cfg):
    if not levels:
        return None
    nearest = levels[0]
    return {
        "price": nearest,
        "gain_pct": round((nearest - entry_price) / entry_price * 100, 2),
        "basis": f"Nearest prior swing high in the trailing {cfg.prior_resistance_lookback}-day window.",
    }


# -----------------------------------------------------------------------
# Target 2 -- Pattern-measured move
# -----------------------------------------------------------------------
def _target2(pattern_name, sig, entry_price, atr, cfg):
    swing = _pattern_swing(pattern_name, sig)
    if swing:
        lo, hi, basis = swing
        height = hi - lo
        if height > 0:
            price = round(entry_price + height, 2)
            return {
                "price": price,
                "gain_pct": round((price - entry_price) / entry_price * 100, 2),
                "basis": f"Measured move: {basis}, projected from the breakout.",
            }
    if atr:
        height = cfg.measured_move_atr_fallback_mult * atr
        price = round(entry_price + height, 2)
        return {
            "price": price,
            "gain_pct": round((price - entry_price) / entry_price * 100, 2),
            "basis": f"{pattern_name} has no natural base geometry to measure -- fallback projection of "
                     f"{cfg.measured_move_atr_fallback_mult:.1f}x ATR(14).",
        }
    return None


# -----------------------------------------------------------------------
# Target 3 -- Fibonacci extension / major resistance
# -----------------------------------------------------------------------
def _target3(pattern_name, sig, entry_price, atr, levels, cfg):
    candidates = []

    swing = _pattern_swing(pattern_name, sig)
    height = None
    if swing:
        lo, hi, _basis = swing
        height = hi - lo
    elif atr:
        height = cfg.measured_move_atr_fallback_mult * atr

    if height and height > 0:
        fib_price = round(entry_price + cfg.fib_extension_ratio * height, 2)
        candidates.append((
            fib_price,
            f"{cfg.fib_extension_ratio:.3f}x Fibonacci extension of the ₹{height:.2f} pattern base, "
            f"projected from the breakout.",
        ))

    if len(levels) > 1:
        # levels[0] is Target 1 (nearest); levels[-1] is the highest swing
        # high still found in the window -- a further, already-proven level.
        far = levels[-1]
        candidates.append((
            far,
            f"Highest prior swing high still found in the trailing history (₹{far:.2f}) "
            f"-- a level the stock has already reached before.",
        ))

    if not candidates:
        return None

    # Take whichever candidate is FURTHER out -- Target 3 is meant to be
    # the extended objective, not a closer duplicate of Target 1/2.
    price, basis = max(candidates, key=lambda c: c[0])
    return {
        "price": price,
        "gain_pct": round((price - entry_price) / entry_price * 100, 2),
        "basis": basis,
    }


# -----------------------------------------------------------------------
# Orchestrator -- what breakout_controller.py calls
# -----------------------------------------------------------------------
def compute_targets(symbol, df, i, pattern_name, sig, entry_price, atr, cfg=None):
    """
    symbol, df, i: same convention as everywhere else in this project --
      df is the symbol's full OHLCV history, i is today's positional index.
    pattern_name, sig: the pattern name and signal dict from
      utils.breakout_patterns.scan_all_patterns (sig carries the raw
      structural numbers each detector stashed for this module to reuse).
    entry_price: the actionable entry price from
      utils.entry_classification (exact_entry_price -- only meaningful
      for Fresh Breakout / Retest rows; call site should skip this
      function entirely for Near Breakout / No Setup rows).
    atr: ATR(14) from utils.entry_classification's ClassificationResult,
      used as the fallback height for patterns without natural geometry.

    Returns {"target1": {...} or None, "target2": {...} or None,
             "target3": {...} or None}. Never raises -- one component
      failing (e.g. too little history for a swing-high scan) never
      blocks the other two or the row it's attached to, same fail-soft
      pattern as utils.breakout_backtest.backtest_signal.
    """
    cfg = cfg or TargetsConfig()
    if entry_price is None:
        return {"target1": None, "target2": None, "target3": None}

    try:
        levels = _find_swing_highs_above(df, i, entry_price, cfg.prior_resistance_lookback, cfg.swing_order)
    except Exception as e:
        log.warning(f"Breakout Screener: swing-high scan failed for {symbol}: {e}")
        levels = []

    try:
        t1 = _target1(levels, entry_price, cfg)
    except Exception as e:
        log.warning(f"Breakout Screener: Target 1 failed for {symbol}: {e}")
        t1 = None

    try:
        t2 = _target2(pattern_name, sig, entry_price, atr, cfg)
    except Exception as e:
        log.warning(f"Breakout Screener: Target 2 failed for {symbol}: {e}")
        t2 = None

    try:
        t3 = _target3(pattern_name, sig, entry_price, atr, levels, cfg)
    except Exception as e:
        log.warning(f"Breakout Screener: Target 3 failed for {symbol}: {e}")
        t3 = None

    return {"target1": t1, "target2": t2, "target3": t3}


# -----------------------------------------------------------------------
# Risk/Reward -- the core filter
# -----------------------------------------------------------------------
def compute_risk_reward(entry_price, stop_loss, targets):
    """
    entry_price: the actionable entry price (utils.entry_classification's
      exact_entry_price).
    stop_loss: entry_price's paired stop from utils.entry_classification
      (ClassificationResult.stop_loss -- already reconciled between the
      structural and ATR-based stop).
    targets: the dict returned by compute_targets() above,
      {"target1": {...} or None, "target2": {...} or None, "target3": {...} or None}.

    Reward is measured against the NEAREST available target -- Target 1
    (prior resistance) if it was found, Target 2 (measured move) if not,
    Target 3 (fib extension) only as a last resort. Using the nearest
    target keeps this a conservative "will I actually get paid before
    something else happens" number, rather than an inflated ratio that
    only looks good against the most optimistic, least-likely level.

    Returns None if entry_price or stop_loss is missing, if the stop
    isn't actually below entry (nothing valid to measure risk against),
    or if every target came back None (nothing to measure reward
    against) -- never a fabricated ratio. Otherwise:
      {
        "risk": float,              # entry - stop, in price terms
        "reward": float,            # target - entry, in price terms
        "ratio": float,             # reward / risk
        "target_used": "target1" | "target2" | "target3",
        "target_price": float,
        "meets_threshold": bool,    # ratio >= RISK_REWARD_MIN_THRESHOLD
        "preferred": bool,          # ratio >= RISK_REWARD_PREFERRED_THRESHOLD
      }
    """
    if entry_price is None or stop_loss is None or not targets:
        return None

    risk = entry_price - stop_loss
    if risk <= 0:
        return None  # stop at/above entry -- not a valid risk to measure against

    for key in ("target1", "target2", "target3"):
        t = targets.get(key)
        if not t:
            continue
        reward = t["price"] - entry_price
        if reward <= 0:
            continue  # a target at/below entry isn't a reward -- try the next one
        ratio = round(reward / risk, 2)
        return {
            "risk": round(risk, 2),
            "reward": round(reward, 2),
            "ratio": ratio,
            "target_used": key,
            "target_price": t["price"],
            "meets_threshold": ratio >= RISK_REWARD_MIN_THRESHOLD,
            "preferred": ratio >= RISK_REWARD_PREFERRED_THRESHOLD,
        }

    return None