"""
utils/entry_classification.py

Three-state breakout entry classifier, layered on top of whatever pattern
detector fired in utils.breakout_patterns. A pattern firing tells you a
breakout is happening; this module tells you whether it's safe to chase,
worth waiting for a retest on, or still approaching.

States
  A. FRESH_BREAKOUT   - Just cleared resistance, still inside the
                         "chaseable" zone on every check.
  B. RETEST           - Broke out earlier, pulled back to resistance,
                         showing a bullish reversal candle today.
                         Preferred entry -- better risk/reward than
                         chasing the initial move.
  C. NEAR_BREAKOUT    - Hasn't broken out yet, but close. Watch-list only,
                         with a concrete trigger price.
  NONE                - Doesn't qualify for any of the above (e.g.
                         extended too far past the breakout with no
                         valid retest).

The "resistance" level used here is a generic trailing-high proxy (see
`trailing_resistance()`), independent of whatever level any specific
pattern detector used internally -- this is meant as a buy-timing overlay
that applies the same way regardless of which pattern fired.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import pandas as pd


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------

class EntryState(Enum):
    FRESH_BREAKOUT = "Fresh Breakout"
    RETEST = "Breakout Retest"
    NEAR_BREAKOUT = "Near Breakout"
    NONE = "No Setup"


ENTRY_STATE_EMOJI = {
    EntryState.FRESH_BREAKOUT: "🟢",
    EntryState.RETEST: "🟡",
    EntryState.NEAR_BREAKOUT: "🔵",
    EntryState.NONE: "⚪",
}


@dataclass
class Bar:
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class ClassificationResult:
    symbol: str
    state: EntryState
    reasons: List[str] = field(default_factory=list)
    entry_zone: Optional[tuple] = None       # (low, high)
    entry_trigger: Optional[float] = None    # for NEAR_BREAKOUT
    exact_entry_price: Optional[float] = None  # single actionable price for FRESH_BREAKOUT / RETEST
    breakout_close: Optional[float] = None
    breakout_date: Optional[str] = None
    distance_to_breakout_pct: Optional[float] = None
    extension_above_breakout_pct: Optional[float] = None
    passed_checks: List[str] = field(default_factory=list)
    failed_checks: List[str] = field(default_factory=list)
    atr: Optional[float] = None
    atr_stop: Optional[float] = None
    structural_stop: Optional[float] = None
    structural_stop_basis: Optional[str] = None   # which structural level was used
    stop_candidates: dict = field(default_factory=dict)  # label -> price, all candidates considered
    stop_loss: Optional[float] = None             # final chosen stop (structural or ATR, whichever is more logical)
    stop_basis: Optional[str] = None              # "structural (<label>)" or "ATR (<reason>)"
    risk_pct: Optional[float] = None              # (entry - stop_loss) / entry * 100

    def label(self) -> str:
        return f"{ENTRY_STATE_EMOJI[self.state]} {self.state.value}"


@dataclass
class ClassifierConfig:
    volume_avg_lookback: int = 20
    min_breakout_pct: float = 1.0
    max_breakout_pct: float = 2.0
    min_volume_multiple: float = 1.5
    strong_close_position: float = 0.65
    rsi_overbought_ceiling: float = 72.0
    max_extension_pct: float = 6.0
    retest_lookback_days: int = 8
    retest_zone_pct: float = 1.5
    near_breakout_pct: float = 3.0
    reversal_close_position: float = 0.60
    resistance_lookback: int = 20   # trailing window used to derive resistance

    # --- stop-loss config ---
    atr_period: int = 14
    atr_stop_multiplier: float = 1.5     # entry - (multiplier x ATR)
    support_buffer_pct: float = 0.5      # cushion placed below a structural support level
    swing_low_lookback: int = 10         # bars searched *before* the breakout for a prior swing low
    min_stop_atr_mult: float = 0.5       # structural stop tighter than this (in ATR units) is too tight -- use ATR instead
    max_stop_atr_mult: float = 2.5       # structural stop wider than this is too loose -- use ATR instead


# ----------------------------------------------------------------------
# Indicators (self-contained -- no extra dependency beyond pandas)
# ----------------------------------------------------------------------

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)  # neutral default where undefined (early history)


def trailing_resistance(df: pd.DataFrame, lookback: int = 20, exclude_last: int = 1) -> Optional[float]:
    """Highest close over the trailing window, excluding the most recent
    `exclude_last` bars (so today's own move doesn't define its own
    resistance level)."""
    if len(df) < lookback + exclude_last + 1:
        return None
    window = df["Close"].iloc[-(lookback + exclude_last):-exclude_last]
    if window.empty:
        return None
    return float(window.max())


def bars_from_df(df: pd.DataFrame, lookback: int = 30) -> List[Bar]:
    """Adapts a yfinance-style OHLCV DataFrame (columns Open/High/Low/Close/Volume,
    DatetimeIndex) into the Bar list the classifier core works with."""
    tail = df.tail(lookback)
    bars = []
    for idx, row in tail.iterrows():
        bars.append(Bar(
            date=str(idx.date()) if hasattr(idx, "date") else str(idx),
            open=float(row["Open"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            close=float(row["Close"]),
            volume=float(row["Volume"]),
        ))
    return bars


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _avg_volume(bars: List[Bar], lookback: int, exclude_last: int = 1) -> float:
    window = bars[-(lookback + exclude_last):-exclude_last] if exclude_last else bars[-lookback:]
    if not window:
        return 0.0
    return sum(b.volume for b in window) / len(window)


def _close_position_in_range(bar: Bar) -> float:
    rng = bar.high - bar.low
    if rng <= 0:
        return 1.0
    return (bar.close - bar.low) / rng


def _is_bullish_reversal_candle(bar: Bar, prior_bar: Optional[Bar], cfg: ClassifierConfig) -> bool:
    bullish = bar.close > bar.open
    strong_close = _close_position_in_range(bar) >= cfg.reversal_close_position
    reclaimed = prior_bar is None or bar.close > prior_bar.close
    return bullish and strong_close and reclaimed


def compute_atr_from_bars(bars: List[Bar], period: int = 14) -> Optional[float]:
    """Wilder-smoothed average true range over the trailing bars -- same
    ewm(alpha=1/period, adjust=False) convention compute_rsi() above
    already uses, so the two indicators this module leans on (RSI for
    the overbought check, ATR for the stop-loss distance sanity check
    and the stop itself) don't disagree about how "trailing average" is
    defined. A plain unweighted mean of True Range is more sensitive to
    whichever single bar happens to age out of the window each day --
    Wilder smoothing decays older bars gradually instead, which is the
    textbook ATR definition and behaves more stably feeding the
    ATR-multiple stop bounds in resolve_stop_loss().

    Needs at least period+1 bars (one extra for the first bar's
    previous close) to seed the smoothing with `period` true-range
    observations; returns None otherwise."""
    if len(bars) < period + 1:
        return None
    window = bars[-(period + 1):]
    true_ranges = []
    for i in range(1, len(window)):
        bar, prev = window[i], window[i - 1]
        tr = max(
            bar.high - bar.low,
            abs(bar.high - prev.close),
            abs(bar.low - prev.close),
        )
        true_ranges.append(tr)
    tr_series = pd.Series(true_ranges)
    atr_series = tr_series.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    atr = atr_series.iloc[-1]
    return float(atr) if pd.notna(atr) else None


def resolve_stop_loss(
    bars: List[Bar],
    breakout_idx: int,
    resistance: float,
    entry_price: float,
    atr: Optional[float],
    cfg: ClassifierConfig,
    is_retest: bool,
) -> tuple:
    """
    Computes every structural stop candidate, picks the nearest one below
    entry as the primary structural stop, then reconciles it against the
    ATR-based stop -- falling back to ATR if the structural level sits
    outside a sane ATR-multiple range (too tight = noise risk, too wide =
    oversized risk).

    Returns (stop_loss, stop_basis, structural_stop, structural_basis,
             atr_stop, candidates_dict).
    """
    candidates = {}

    # 1. Breakout support -- the former resistance, now acting as support,
    #    with a small cushion below it.
    breakout_support = round(resistance * (1 - cfg.support_buffer_pct / 100), 2)
    candidates["breakout support"] = breakout_support

    # 2. Previous swing low -- lowest low in the window immediately before
    #    the breakout bar (the base the stock broke out of).
    pre_breakout_start = max(0, breakout_idx - cfg.swing_low_lookback)
    pre_breakout_window = bars[pre_breakout_start:breakout_idx]
    if pre_breakout_window:
        candidates["previous swing low"] = round(min(b.low for b in pre_breakout_window), 2)

    # 3. Pattern low -- lowest low from the breakout bar through today
    #    (covers the breakout + any pullback since).
    post_breakout_window = bars[breakout_idx:]
    if post_breakout_window:
        candidates["pattern low"] = round(min(b.low for b in post_breakout_window), 2)

    # 4. Retest low -- lowest low strictly after the breakout bar (only
    #    meaningful once a retest has actually happened).
    if is_retest:
        retest_window = bars[breakout_idx + 1:]
        if retest_window:
            candidates["retest low"] = round(min(b.low for b in retest_window), 2)

    # Only candidates that sit below entry are valid stops.
    valid = {label: price for label, price in candidates.items() if price < entry_price}

    structural_stop, structural_basis = None, None
    if valid:
        # Nearest to entry = tightest, least-risk structural level.
        structural_basis, structural_stop = max(valid.items(), key=lambda kv: kv[1])

    atr_stop = round(entry_price - cfg.atr_stop_multiplier * atr, 2) if atr else None

    if structural_stop is not None and atr:
        distance_in_atr = (entry_price - structural_stop) / atr
        if cfg.min_stop_atr_mult <= distance_in_atr <= cfg.max_stop_atr_mult:
            return structural_stop, f"structural ({structural_basis})", structural_stop, structural_basis, atr_stop, candidates
        else:
            reason = "too tight, noise risk" if distance_in_atr < cfg.min_stop_atr_mult else "too wide, oversized risk"
            return atr_stop, f"ATR (structural level {reason})", structural_stop, structural_basis, atr_stop, candidates

    if structural_stop is not None and not atr:
        return structural_stop, f"structural ({structural_basis})", structural_stop, structural_basis, atr_stop, candidates

    if atr_stop is not None:
        return atr_stop, "ATR (no structural candidate available)", structural_stop, structural_basis, atr_stop, candidates

    return None, None, structural_stop, structural_basis, atr_stop, candidates


# ----------------------------------------------------------------------
# Core classification (bar-list based -- reusable outside this codebase too)
# ----------------------------------------------------------------------

def classify_stock(
    symbol: str,
    bars: List[Bar],
    resistance: float,
    rsi: float,
    market_trend_supportive: bool,
    cfg: ClassifierConfig = ClassifierConfig(),
) -> ClassificationResult:
    if len(bars) < 3:
        return ClassificationResult(symbol, EntryState.NONE, reasons=["Insufficient history"])

    today = bars[-1]
    current = today.close

    # ---- Case C: NEAR BREAKOUT ----
    if current < resistance:
        distance_pct = (resistance - current) / resistance * 100
        if distance_pct <= cfg.near_breakout_pct:
            trigger = round(resistance * 1.001, 2)
            return ClassificationResult(
                symbol=symbol, state=EntryState.NEAR_BREAKOUT,
                reasons=[f"{distance_pct:.1f}% below resistance ₹{resistance:.2f}",
                         "Watch for breakout close with volume ≥1.5x avg"],
                entry_trigger=trigger,
                distance_to_breakout_pct=round(distance_pct, 2),
            )
        return ClassificationResult(symbol, EntryState.NONE,
                                     reasons=[f"{distance_pct:.1f}% below resistance — too far to watch"])

    # Above resistance -- find the breakout bar within the lookback window.
    breakout_idx = None
    for i in range(len(bars) - 1, max(len(bars) - cfg.retest_lookback_days - 1, 0), -1):
        prev_close = bars[i - 1].close if i > 0 else None
        if bars[i].close > resistance and prev_close is not None and prev_close <= resistance:
            breakout_idx = i
            break

    if breakout_idx is None:
        return ClassificationResult(symbol, EntryState.NONE,
                                     reasons=["Above resistance but no recent qualifying breakout bar found"])

    breakout_bar = bars[breakout_idx]
    breakout_pct = (breakout_bar.close - resistance) / resistance * 100
    days_since_breakout = (len(bars) - 1) - breakout_idx
    extension_pct = (current - breakout_bar.close) / breakout_bar.close * 100

    # ---- Case B: RETEST (checked before fresh -- it's the preferred entry) ----
    if days_since_breakout >= 1:
        pulled_back = any(
            resistance <= b.low <= resistance * (1 + cfg.retest_zone_pct / 100)
            for b in bars[breakout_idx + 1:-1]
        ) or (resistance <= today.low <= resistance * (1 + cfg.retest_zone_pct / 100))

        held_above_resistance = all(b.close >= resistance * 0.98 for b in bars[breakout_idx + 1:])
        prior_bar = bars[-2] if len(bars) >= 2 else None
        reversal_candle = _is_bullish_reversal_candle(today, prior_bar, cfg)

        if pulled_back and held_above_resistance and reversal_candle:
            zone_low = round(resistance * 1.0025, 2)
            zone_high = round(resistance * (1 + cfg.retest_zone_pct / 100), 2)
            entry_price = round(today.close, 2)
            atr = compute_atr_from_bars(bars, cfg.atr_period)
            stop_loss, stop_basis, structural_stop, structural_basis, atr_stop, candidates = resolve_stop_loss(
                bars, breakout_idx, resistance, entry_price, atr, cfg, is_retest=True,
            )
            risk_pct = round((entry_price - stop_loss) / entry_price * 100, 2) if stop_loss else None
            reasons = [f"Breakout on {breakout_bar.date} at ₹{breakout_bar.close:.2f}",
                       f"Retested near resistance ₹{resistance:.2f}, held support",
                       "Bullish reversal candle confirmed today"]
            if stop_loss:
                reasons.append(f"Stop-loss ₹{stop_loss:.2f} ({stop_basis}), risk {risk_pct:.1f}%")
            return ClassificationResult(
                symbol=symbol, state=EntryState.RETEST,
                reasons=reasons,
                entry_zone=(zone_low, zone_high),
                exact_entry_price=entry_price,
                breakout_close=breakout_bar.close, breakout_date=breakout_bar.date,
                atr=round(atr, 2) if atr else None,
                atr_stop=atr_stop, structural_stop=structural_stop, structural_stop_basis=structural_basis,
                stop_candidates=candidates, stop_loss=stop_loss, stop_basis=stop_basis, risk_pct=risk_pct,
            )

    # ---- Case A: FRESH BREAKOUT ----
    if days_since_breakout == 0:
        avg_vol = _avg_volume(bars, cfg.volume_avg_lookback, exclude_last=1)
        vol_multiple = (today.volume / avg_vol) if avg_vol else 0
        strong_candle = _close_position_in_range(today) >= cfg.strong_close_position

        checks = {
            "Close above resistance": current > resistance,
            f"Breakout by ≥{cfg.min_breakout_pct}%": breakout_pct >= cfg.min_breakout_pct,
            f"Volume ≥{cfg.min_volume_multiple}x avg": vol_multiple >= cfg.min_volume_multiple,
            "Strong closing candle": strong_candle,
            f"RSI not overbought (<{cfg.rsi_overbought_ceiling})": rsi < cfg.rsi_overbought_ceiling,
            f"Not >{cfg.max_extension_pct}% above breakout": extension_pct <= cfg.max_extension_pct,
            "Market trend supportive": market_trend_supportive,
        }
        passed = [k for k, v in checks.items() if v]
        failed = [k for k, v in checks.items() if not v]

        if not failed:
            zone_low = round(resistance * (1 + cfg.min_breakout_pct / 100 * 1.25), 2)
            zone_high = round(current * 1.0025, 2)
            entry_price = round(current, 2)
            atr = compute_atr_from_bars(bars, cfg.atr_period)
            stop_loss, stop_basis, structural_stop, structural_basis, atr_stop, candidates = resolve_stop_loss(
                bars, breakout_idx, resistance, entry_price, atr, cfg, is_retest=False,
            )
            risk_pct = round((entry_price - stop_loss) / entry_price * 100, 2) if stop_loss else None
            reasons = [f"Breakout {breakout_pct:.1f}% above resistance, "
                       f"volume {vol_multiple:.1f}x avg, RSI {rsi:.0f}"]
            if stop_loss:
                reasons.append(f"Stop-loss ₹{stop_loss:.2f} ({stop_basis}), risk {risk_pct:.1f}%")
            return ClassificationResult(
                symbol=symbol, state=EntryState.FRESH_BREAKOUT,
                reasons=reasons,
                entry_zone=(zone_low, zone_high),
                exact_entry_price=entry_price,
                breakout_close=breakout_bar.close, breakout_date=breakout_bar.date,
                extension_above_breakout_pct=round(extension_pct, 2),
                passed_checks=passed, failed_checks=failed,
                atr=round(atr, 2) if atr else None,
                atr_stop=atr_stop, structural_stop=structural_stop, structural_stop_basis=structural_basis,
                stop_candidates=candidates, stop_loss=stop_loss, stop_basis=stop_basis, risk_pct=risk_pct,
            )
        return ClassificationResult(
            symbol=symbol, state=EntryState.NONE,
            reasons=[f"Failed: {', '.join(failed)}"],
            breakout_close=breakout_bar.close, breakout_date=breakout_bar.date,
            passed_checks=passed, failed_checks=failed,
        )

    return ClassificationResult(
        symbol=symbol, state=EntryState.NONE,
        reasons=[f"Breakout {days_since_breakout}d ago, extended {extension_pct:.1f}% — "
                 "no valid retest and no longer fresh"],
        breakout_close=breakout_bar.close, breakout_date=breakout_bar.date,
        extension_above_breakout_pct=round(extension_pct, 2),
    )


# ----------------------------------------------------------------------
# DataFrame entry point -- what breakout_controller.py calls
# ----------------------------------------------------------------------

def classify_entry(
    symbol: str,
    df: pd.DataFrame,
    market_trend_supportive: bool,
    cfg: ClassifierConfig = ClassifierConfig(),
) -> Optional[ClassificationResult]:
    """Convenience wrapper: derives resistance + RSI from the OHLCV df itself
    and runs classify_stock(). Returns None if there isn't enough history to
    classify (mirrors the fail-soft style used elsewhere in this codebase)."""
    resistance = trailing_resistance(df, lookback=cfg.resistance_lookback)
    if resistance is None:
        return None

    rsi_series = compute_rsi(df["Close"])
    rsi_today = float(rsi_series.iloc[-1])

    bars = bars_from_df(df, lookback=cfg.retest_lookback_days + cfg.volume_avg_lookback + 2)
    return classify_stock(symbol, bars, resistance, rsi_today, market_trend_supportive, cfg)