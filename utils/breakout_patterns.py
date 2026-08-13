"""
utils/breakout_patterns.py

Chart-pattern breakout detectors for the daily screener
(controllers/breakout_controller.py).

Design rule that matters more than any individual pattern's cleverness:
every detector function below takes (df, i) and is only allowed to look
at df.iloc[:i+1] -- i.e. today (index i) and everything BEFORE it. This
is what lets utils/breakout_backtest.py replay the exact same detector
against history and get a trustworthy result: if a detector peeked at
future rows, its "backtest" would silently be cheating (lookahead bias)
and the win-rates in the email would be meaningless. Keep that contract
if you add a new detector.

df columns expected: Open, High, Low, Close, Volume (ascending by date,
plain RangeIndex or DatetimeIndex both fine -- only positional .iloc
access is used).

Each detector returns None (no signal) or a dict describing the signal.
Every returned dict has at minimum: {"pattern": str, "signal_price": float}.
"""

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------
def _window(df, i, lookback):
    start = max(0, i - lookback + 1)
    return df.iloc[start:i + 1]


def _swing_points(closes, order=3):
    """Local maxima/minima indices (positional, within the passed array)
    using a simple order-N comparison window -- good enough for
    trendline/shape heuristics without pulling in a heavier TA library."""
    highs_idx = argrelextrema(closes.values, np.greater_equal, order=order)[0]
    lows_idx = argrelextrema(closes.values, np.less_equal, order=order)[0]
    return highs_idx, lows_idx


def _linreg_slope(y):
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y))
    slope, _ = np.polyfit(x, y, 1)
    return slope


# ---------------------------------------------------------------------
# Guardrails -- shared pre-checks every detector runs before its own
# pattern-specific logic, all gated purely on df.iloc[:i+1] like every
# detector here (see the lookahead-bias note at the top of this file).
# These are baked into EACH detector function (not bolted on as a
# separate wrapper around scan_all_patterns) specifically so that
# utils/breakout_backtest.py's historical replay hits the exact same
# gate a live signal would. Left as a wrapper-only check, a corporate
# action, a data glitch, or an illiquid stretch buried somewhere in a
# stock's own trailing history would still count as a "past occurrence"
# in that stock's backtest -- quietly inflating or deflating the very
# hit-rate number the email calls "confirmed."
# ---------------------------------------------------------------------

# Liquidity / penny-stock gate.
LIQUIDITY_LOOKBACK = 20
MIN_AVG_PRICE = 20.0          # rupees -- below this, treat as a penny stock
MIN_AVG_TURNOVER = 1.0e7      # rupees/day (~₹1 crore), 20-day average Close*Volume

# Corporate-action gate -- flags an overnight gap that lines up with a
# common split/bonus/consolidation ratio rather than genuine trading.
CORP_ACTION_RATIOS = (0.5, 0.2, 0.25, 0.1, 1 / 3, 2.0, 3.0, 4.0, 5.0, 10.0)
CORP_ACTION_TOLERANCE = 0.04  # 4% band around a "clean" ratio counts as a match
CORP_ACTION_MIN_GAP = 0.15    # only worth checking once the overnight gap is already big

# Data-sanity gate -- how many trailing bars to sanity-check for feed glitches.
DATA_SANITY_CHECK_DAYS = 5


def _data_sanity_ok(df, i, check_days=DATA_SANITY_CHECK_DAYS):
    """Basic OHLCV integrity check over the last few bars. Catches feed
    glitches -- bad ticks, a frozen/stale bar repeated by a caching
    layer, zero/negative prices, or an impossible High/Low/Close
    relationship -- that would otherwise get read as a genuine breakout."""
    start = max(0, i - check_days + 1)
    recent = df.iloc[start:i + 1]
    if recent.empty:
        return False

    for col in ("Open", "High", "Low", "Close"):
        vals = recent[col]
        if vals.isna().any() or (vals <= 0).any():
            return False

    if (recent["High"] < recent["Low"]).any():
        return False
    if (recent["High"] < recent[["Open", "Close"]].max(axis=1) - 1e-6).any():
        return False
    if (recent["Low"] > recent[["Open", "Close"]].min(axis=1) + 1e-6).any():
        return False

    # A frozen/stale feed: identical OHLC repeated across the whole
    # window. A real market essentially never prints the exact same
    # four prices on three-plus consecutive sessions.
    if len(recent) >= 3 and recent[["Open", "High", "Low", "Close"]].nunique().eq(1).all():
        return False

    return True


def _is_liquid(df, i, lookback=LIQUIDITY_LOOKBACK, min_price=MIN_AVG_PRICE, min_turnover=MIN_AVG_TURNOVER):
    """Penny-stock / illiquid-stock gate. A pattern firing on a thinly
    traded name isn't really a tradable setup -- entering or exiting at
    any real size would move the price itself -- and its "past
    occurrences" in the backtest are just as unreliable for the same
    reason."""
    if i < lookback - 1:
        return False
    window = df.iloc[i - lookback + 1:i + 1]
    if window["Close"].isna().any() or window["Volume"].isna().any():
        return False
    avg_price = window["Close"].mean()
    avg_turnover = (window["Close"] * window["Volume"]).mean()
    return avg_price >= min_price and avg_turnover >= min_turnover


def _looks_like_corporate_action(df, i, ratios=CORP_ACTION_RATIOS, tol=CORP_ACTION_TOLERANCE, min_gap=CORP_ACTION_MIN_GAP):
    """Flags an overnight gap that lines up with a common split/bonus/
    consolidation ratio (1:2, 1:5, 1:10, 3:1, etc.) rather than genuine
    trading. Left unfiltered, a 1:5 split reads as an ~80% single-day
    drop that can masquerade as several patterns below, and a bonus
    issue can just as easily look like a volume-surge breakout -- purely
    a price-adjustment artifact, not a signal."""
    if i < 1:
        return False
    prev_close = df.iloc[i - 1]["Close"]
    today_open = df.iloc[i]["Open"]
    if prev_close <= 0 or today_open <= 0:
        return False
    gap = abs(today_open - prev_close) / prev_close
    if gap < min_gap:
        return False
    ratio = today_open / prev_close
    return any(abs(ratio - r) / r <= tol for r in ratios)


# Extension / exhaustion gate -- a stock that has already run too far
# from its own mean, or is already deeply overbought off an outsized
# single-day move, has less room left before mean-reversion/profit-taking
# than its raw hit-rate alone would suggest. E.g. a 4.9x-volume, +8.2%
# day (real example: NATIONALUM) can carry a strong historical hit-rate
# AND a meaningfully elevated near-term failure rate at the same time --
# these aren't contradictory, they're the same phenomenon (the move is
# already extended) showing up in two different numbers. Gating on
# extension keeps a screener signal from being "confirmed" purely on a
# stock's own backtest average, when TODAY's specific instance is
# already more stretched than the average historical occurrence was.
EXTENSION_ATR_LOOKBACK = 14
EXTENSION_MAX_ATR_MULTIPLE = 3.5   # today's Close vs. its 20-day SMA, in ATR(14) multiples
EXTENSION_RSI_CEILING = 80.0       # RSI(14) at/above this = deeply overbought
EXTENSION_MIN_DAY_MOVE = 0.06      # only let RSI-overbought gate a signal if today's move was itself already big


def _atr(df, i, period=EXTENSION_ATR_LOOKBACK):
    """Average True Range over `period` sessions ending at i, using only
    df.iloc[:i+1] -- standard True Range (max of high-low, |high-prev
    close|, |low-prev close|) averaged over the trailing window."""
    if i < period:
        return None
    highs = df["High"].iloc[:i + 1]
    lows = df["Low"].iloc[:i + 1]
    prev_close = df["Close"].iloc[:i + 1].shift(1)
    tr = pd.concat([
        highs - lows,
        (highs - prev_close).abs(),
        (lows - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.iloc[-period:].mean()
    return float(atr) if pd.notna(atr) else None


def _rsi(df, i, period=14):
    """Classic RSI(period) using only df.iloc[:i+1]."""
    if i < period:
        return None
    closes = df["Close"].iloc[:i + 1]
    delta = closes.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.iloc[-period:].mean()
    avg_loss = losses.iloc[-period:].mean()
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _is_too_extended(df, i):
    """True if today's bar is already stretched too far from its own
    mean to trust as a fresh signal -- either (a) Close is more than
    EXTENSION_MAX_ATR_MULTIPLE average-true-ranges away from its own
    20-day SMA, or (b) RSI(14) is already deeply overbought AND today's
    own move was itself large (so it's not just a slow grind that
    happens to be overbought, but a sharp spike on top of one)."""
    if i < 20:
        return False
    closes = df["Close"].iloc[:i + 1]
    ma20 = closes.iloc[-20:].mean()
    today_close = closes.iloc[-1]

    atr = _atr(df, i)
    if atr and atr > 0:
        extension_atr = abs(today_close - ma20) / atr
        if extension_atr > EXTENSION_MAX_ATR_MULTIPLE:
            return True

    rsi = _rsi(df, i)
    if rsi is not None and rsi >= EXTENSION_RSI_CEILING:
        prev_close = df.iloc[i - 1]["Close"]
        if prev_close > 0:
            day_move = (today_close - prev_close) / prev_close
            if day_move >= EXTENSION_MIN_DAY_MOVE:
                return True

    return False


def _passes_guardrails(df, i):
    """Single entry point every detector calls before its own
    pattern-specific logic. True only if today's bar is clean enough,
    liquid enough, not a corporate-action price artifact, and not
    already too extended to trust for pattern detection."""
    if not _data_sanity_ok(df, i):
        return False
    if not _is_liquid(df, i):
        return False
    if _looks_like_corporate_action(df, i):
        return False
    if _is_too_extended(df, i):
        return False
    return True


# ---------------------------------------------------------------------
# 1. Resistance / consolidation breakout
# ---------------------------------------------------------------------
def detect_resistance_breakout(df, i, lookback=20, min_range_days=10, buffer=0.002):
    if i < lookback:
        return None
    if not _passes_guardrails(df, i):
        return None
    prior = df.iloc[i - lookback:i]  # strictly before today
    today = df.iloc[i]

    resistance = prior["High"].max()
    support = prior["Low"].min()
    if resistance <= 0 or support <= 0:
        return None

    range_pct = (resistance - support) / support
    # A genuine consolidation, not a stock that's simply been trending --
    # cap how wide the prior range was allowed to be.
    if range_pct > 0.15:
        return None

    if today["Close"] > resistance * (1 + buffer) and today["Close"] > today["Open"]:
        return {
            "pattern": "Resistance Breakout",
            "signal_price": float(today["Close"]),
            "detail": f"Broke {lookback}-day range high of ₹{resistance:.2f} (range was {range_pct*100:.1f}% wide).",
            # Raw base geometry, stashed for utils.breakout_targets' measured-move
            # projection -- so it never has to re-derive (and risk disagreeing
            # with) the exact range that actually fired.
            "resistance": float(resistance),
            "support": float(support),
        }
    return None


# ---------------------------------------------------------------------
# 2. Volume-surge breakout
# ---------------------------------------------------------------------
def detect_volume_surge(df, i, lookback=20, vol_mult=2.0, min_price_change=0.02):
    if i < lookback:
        return None
    if not _passes_guardrails(df, i):
        return None
    prior = df.iloc[i - lookback:i]
    today = df.iloc[i]

    avg_vol = prior["Volume"].mean()
    if avg_vol <= 0:
        return None
    vol_ratio = today["Volume"] / avg_vol

    prev_close = df.iloc[i - 1]["Close"]
    if prev_close <= 0:
        return None
    price_change = (today["Close"] - prev_close) / prev_close

    if vol_ratio >= vol_mult and price_change >= min_price_change:
        return {
            "pattern": "Volume Surge Breakout",
            "signal_price": float(today["Close"]),
            "detail": f"Volume {vol_ratio:.1f}x the {lookback}-day average, price up {price_change*100:.1f}% on the day.",
        }
    return None


# ---------------------------------------------------------------------
# 3. 52-week high breakout
# ---------------------------------------------------------------------
def detect_52w_high(df, i, lookback=252, tolerance=0.003):
    if i < 60:  # allow this on shorter histories too, using whatever's available
        return None
    if not _passes_guardrails(df, i):
        return None
    prior = df.iloc[max(0, i - lookback):i]
    today = df.iloc[i]
    high_52w = prior["High"].max()
    if high_52w <= 0:
        return None
    if today["Close"] >= high_52w * (1 - tolerance):
        return {
            "pattern": "52-Week High Breakout",
            "signal_price": float(today["Close"]),
            "detail": f"Closed at/above the trailing {'52-week' if lookback >= 252 else f'{lookback}-day'} high of ₹{high_52w:.2f}.",
        }
    return None


# ---------------------------------------------------------------------
# 4. Moving-average crossover (golden cross / price reclaim)
# ---------------------------------------------------------------------
def detect_ma_crossover(df, i, fast=50, slow=200, confirm_days=3):
    if i < slow + confirm_days:
        return None
    if not _passes_guardrails(df, i):
        return None
    closes = df["Close"].iloc[:i + 1]
    fast_ma = closes.rolling(fast).mean()
    slow_ma = closes.rolling(slow).mean()

    if fast_ma.iloc[-1] <= slow_ma.iloc[-1]:
        return None
    # Crossed within the last `confirm_days` sessions (not a stale cross
    # from weeks ago), so the signal is actually "fresh".
    recent_diff = (fast_ma - slow_ma).iloc[-(confirm_days + 1):]
    if not (recent_diff.iloc[0] <= 0 and recent_diff.iloc[-1] > 0):
        return None

    return {
        "pattern": "Golden Cross (50/200 DMA)",
        "signal_price": float(df.iloc[i]["Close"]),
        "detail": f"{fast}-day MA (₹{fast_ma.iloc[-1]:.2f}) crossed above {slow}-day MA (₹{slow_ma.iloc[-1]:.2f}) within the last {confirm_days} sessions.",
    }


# ---------------------------------------------------------------------
# 5. Bollinger squeeze breakout
# ---------------------------------------------------------------------
def detect_bollinger_squeeze_breakout(df, i, window=20, squeeze_lookback=60, squeeze_pct=0.35, num_std=2.0):
    if i < window + squeeze_lookback:
        return None
    if not _passes_guardrails(df, i):
        return None
    closes = df["Close"].iloc[:i + 1]
    ma = closes.rolling(window).mean()
    std = closes.rolling(window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    bandwidth = (upper - lower) / ma

    recent_bw = bandwidth.iloc[-squeeze_lookback:]
    percentile_rank = (recent_bw < recent_bw.iloc[-2]).mean()  # was it tight relative to its own recent history, right before today?
    was_squeezed = percentile_rank <= squeeze_pct

    today = df.iloc[i]
    breaks_upper = today["Close"] > upper.iloc[-1]

    if was_squeezed and breaks_upper:
        return {
            "pattern": "Bollinger Squeeze Breakout",
            "signal_price": float(today["Close"]),
            "detail": f"Bandwidth was in its tightest {percentile_rank*100:.0f}th percentile over {squeeze_lookback} days, then closed above the upper band (₹{upper.iloc[-1]:.2f}).",
            # Raw band levels at the moment of breakout -- utils.breakout_targets
            # uses the band width as this pattern's measured-move height.
            "band_lower": float(lower.iloc[-1]),
            "band_upper": float(upper.iloc[-1]),
        }
    return None


# ---------------------------------------------------------------------
# 6. Classic chart shapes -- heuristic pivot/trendline based
# ---------------------------------------------------------------------
def detect_flag_pattern(df, i, pole_lookback=15, flag_lookback=8, min_pole_move=0.08):
    """Bull flag: a sharp up-move (the pole), then a shallow, tight,
    slightly-downward-or-flat drift (the flag), then a breakout above the
    flag's high on today's bar."""
    if i < pole_lookback + flag_lookback:
        return None
    if not _passes_guardrails(df, i):
        return None

    flag = df.iloc[i - flag_lookback:i]  # before today
    pole = df.iloc[i - pole_lookback - flag_lookback:i - flag_lookback]
    today = df.iloc[i]

    if pole.empty or flag.empty:
        return None

    pole_move = (pole["Close"].iloc[-1] - pole["Close"].iloc[0]) / max(pole["Close"].iloc[0], 1e-9)
    if pole_move < min_pole_move:
        return None

    flag_high = flag["High"].max()
    flag_low = flag["Low"].min()
    flag_range_pct = (flag_high - flag_low) / max(flag_low, 1e-9)
    flag_slope = _linreg_slope(flag["Close"].values)

    # Flag should be tight and flat/slightly down -- not itself a big
    # trending move, or this is just "still going up", not a flag.
    if flag_range_pct > 0.10 or flag_slope > 0:
        return None

    if today["Close"] > flag_high * 1.002:
        return {
            "pattern": "Bull Flag Breakout",
            "signal_price": float(today["Close"]),
            "detail": f"Pole move of {pole_move*100:.0f}% then a {flag_range_pct*100:.1f}%-wide flag, broke above flag high ₹{flag_high:.2f}.",
            # Raw pole endpoints -- utils.breakout_targets projects the pole's
            # own price move forward from the flag breakout (classic
            # "flagpole" measured move).
            "pole_start": float(pole["Close"].iloc[0]),
            "pole_end": float(pole["Close"].iloc[-1]),
            "flag_high": float(flag_high),
        }
    return None


def detect_triangle_pattern(df, i, lookback=40, min_touches=2):
    """Symmetrical/ascending triangle: converging highs and lows over the
    lookback window, breakout above the upper trendline today."""
    if i < lookback:
        return None
    if not _passes_guardrails(df, i):
        return None
    window = df.iloc[i - lookback:i]  # before today
    today = df.iloc[i]

    highs_idx, lows_idx = _swing_points(window["Close"], order=3)
    if len(highs_idx) < min_touches or len(lows_idx) < min_touches:
        return None

    high_slope = _linreg_slope(window["High"].values[highs_idx])
    low_slope = _linreg_slope(window["Low"].values[lows_idx])

    # Converging: highs flat-to-falling, lows flat-to-rising (or highs
    # falling faster than lows falling, for a descending-into-support
    # shape) -- broadly, the range must be narrowing.
    early_range = window["High"].iloc[:lookback // 3].max() - window["Low"].iloc[:lookback // 3].min()
    late_range = window["High"].iloc[-lookback // 3:].max() - window["Low"].iloc[-lookback // 3:].min()
    if early_range <= 0 or late_range >= early_range * 0.7:
        return None
    if high_slope > 0.5 * abs(low_slope) + 1e-6 and low_slope < 0:
        return None  # highs rising faster than lows -- not a converging triangle

    upper_trendline = window["High"].iloc[-5:].max()
    if today["Close"] > upper_trendline * 1.002:
        return {
            "pattern": "Triangle Breakout",
            "signal_price": float(today["Close"]),
            "detail": f"Range narrowed from ₹{early_range:.2f} to ₹{late_range:.2f} wide over {lookback} days, broke above ₹{upper_trendline:.2f}.",
            # Widest (early) part of the triangle -- utils.breakout_targets
            # projects that full range forward from the breakout, the
            # standard triangle measured-move convention.
            "early_range": float(early_range),
            "upper_trendline": float(upper_trendline),
        }
    return None


def detect_cup_and_handle(df, i, cup_lookback=90, handle_lookback=12, max_handle_pct=0.15, min_cup_depth=0.12):
    """Cup: a rounded U-shaped decline-and-recovery back near the prior
    high. Handle: a shallow pullback right after. Breakout: close above
    both the cup's prior high and the handle's high today."""
    if i < cup_lookback + handle_lookback:
        return None
    if not _passes_guardrails(df, i):
        return None

    cup = df.iloc[i - cup_lookback - handle_lookback:i - handle_lookback]
    handle = df.iloc[i - handle_lookback:i]
    today = df.iloc[i]
    if cup.empty or handle.empty:
        return None

    left_rim = cup["Close"].iloc[0]
    cup_low = cup["Close"].min()
    right_rim = cup["Close"].iloc[-1]
    if left_rim <= 0 or cup_low <= 0:
        return None

    depth = (left_rim - cup_low) / left_rim
    rims_aligned = abs(right_rim - left_rim) / left_rim < 0.08
    low_idx_in_cup = cup["Close"].values.argmin()
    low_roughly_centered = 0.25 * len(cup) < low_idx_in_cup < 0.85 * len(cup)  # rounded, not a V or a late dip

    if depth < min_cup_depth or not rims_aligned or not low_roughly_centered:
        return None

    handle_high = handle["High"].max()
    handle_low = handle["Low"].min()
    handle_pct = (handle_high - handle_low) / max(handle_low, 1e-9)
    if handle_pct > max_handle_pct or handle["Close"].iloc[-1] > handle_high:
        return None  # handle already broke out before today, or too deep/wide to count as a handle

    rim_level = max(left_rim, right_rim, handle_high)
    if today["Close"] > rim_level * 1.002:
        return {
            "pattern": "Cup and Handle Breakout",
            "signal_price": float(today["Close"]),
            "detail": f"{depth*100:.0f}% deep cup over {cup_lookback} days, {handle_pct*100:.1f}%-wide handle, broke above ₹{rim_level:.2f}.",
            # Cup depth -- utils.breakout_targets projects it forward from the
            # rim, the standard cup-and-handle measured-move convention.
            "cup_low": float(cup_low),
            "rim_level": float(rim_level),
        }
    return None


# ---------------------------------------------------------------------
# Master scan
# ---------------------------------------------------------------------
ALL_DETECTORS = [
    detect_resistance_breakout,
    detect_volume_surge,
    detect_52w_high,
    detect_ma_crossover,
    detect_bollinger_squeeze_breakout,
    detect_flag_pattern,
    detect_triangle_pattern,
    detect_cup_and_handle,
]

# pattern display name -> detector function. Lets the backtester replay
# the EXACT function that produced a given signal against history,
# without the controller needing to know the mapping itself.
PATTERN_DETECTOR_BY_NAME = {
    "Resistance Breakout": detect_resistance_breakout,
    "Volume Surge Breakout": detect_volume_surge,
    "52-Week High Breakout": detect_52w_high,
    "Golden Cross (50/200 DMA)": detect_ma_crossover,
    "Bollinger Squeeze Breakout": detect_bollinger_squeeze_breakout,
    "Bull Flag Breakout": detect_flag_pattern,
    "Triangle Breakout": detect_triangle_pattern,
    "Cup and Handle Breakout": detect_cup_and_handle,
}


def scan_all_patterns(df, i, detectors=None):
    """Runs every detector against df at position i, returns the list of
    triggered signal dicts (possibly empty, possibly more than one --
    e.g. a volume surge AND a resistance breakout on the same day is
    common and meaningful, not a bug)."""
    detectors = detectors or ALL_DETECTORS
    signals = []
    for fn in detectors:
        try:
            result = fn(df, i)
        except Exception:
            result = None  # a single detector's edge-case bug should never take down the whole scan
        if result:
            signals.append(result)
    return signals