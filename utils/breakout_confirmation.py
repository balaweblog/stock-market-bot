"""
utils/breakout_confirmation.py

Breakout Confirmation Score: a simple, transparent checklist tally
layered on top of data already computed elsewhere in this project
(utils.entry_classification's resistance/RSI conventions,
utils.market_regime's NIFTY regime read, the day's own OHLCV) -- no new
data source beyond one already-fetched NIFTY series, no new lookahead,
just a scorecard of independent yes/no confirmations so a fired
pattern's overall conviction can be read at a glance ("8/8") instead of
scanning every column individually.

Nine independent checks. Each is True, False, or None ("Not available"
-- excluded from BOTH the numerator and denominator, same "an
unavailable check shouldn't manufacture a false negative" fail-soft
convention used throughout this project):

  1. Price > resistance    Above the SAME trailing-resistance level
                            utils.entry_classification.trailing_resistance
                            uses for the Entry column, so the two never
                            disagree about what "resistance" means.
  2. Volume > 2x average   Today's volume vs its own trailing 20-day
                            average -- independent of, and a stricter
                            bar than, the 1.5x the Fresh-Breakout
                            classifier itself requires.
  3. Close near high       Closed in the top portion of today's own
                            high/low range.
  4. RSI 55-75             Momentum confirmed but not yet
                            overbought/euphoric -- a band, not just a
                            ceiling.
  5. Above 20/50 DMA       Price above both trailing moving averages --
                            short and medium-term trend agree.
  6. Relative strength     The stock's own trailing return beats NIFTY
                            50's over the same window -- reuses the
                            SAME NIFTY close series
                            utils.market_regime already fetched this run
                            (RegimeResult.nifty_closes), no extra fetch,
                            aligned positionally (bar-for-bar from the
                            end of each series) rather than by calendar
                            date, since the two indices don't always
                            share a holiday calendar.
  7. Sector confirmation   NOT AVAILABLE in this pipeline yet -- there's
                            no sector-mapping / sector-index data source
                            wired in anywhere in this project. Always
                            scores None ("N/A") rather than silently
                            guessing or defaulting to a fabricated pass.
                            Wiring this up needs a symbol->sector map and
                            per-sector index history, neither of which
                            utils.nse_data currently provides.
  8. NIFTY regime           Reuses
                            utils.market_regime.RegimeResult.entries_supportive
                            directly (Bullish/Neutral = pass, Bearish =
                            fail) -- never recomputed, same source of
                            truth this already feeds for the Entry
                            column's own "Market trend supportive" check.
  9. No excessive extension Price hasn't run too far past resistance to
                            still be a reasonable entry (checked against
                            the same resistance level as check 1).

Score is shown as "(passed) / (checks that had data)" -- e.g. "8/8"
until sector data exists, since check 7 is excluded from the
denominator until then, then "9/9" once it's wired in.

This is a CONVICTION READ, not a gate -- unlike Risk:Reward or
utils.breakout_failure's same-day bull-trap check, nothing is filtered
out of Confirmed/Watch by this score alone. It's additional context on
a row whose bucket (Confirmed/Watch/Filtered/Failed) is already decided
by the time this runs.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from utils.entry_classification import trailing_resistance, compute_rsi

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
RESISTANCE_LOOKBACK = 20          # matches utils.entry_classification.ClassifierConfig.resistance_lookback default
VOLUME_AVG_LOOKBACK = 20
VOLUME_CONFIRM_MULTIPLE = 2.0
CLOSE_NEAR_HIGH_MIN_POSITION = 0.65
RSI_BAND = (55.0, 75.0)
DMA_SHORT = 20
DMA_LONG = 50
RELATIVE_STRENGTH_LOOKBACK = 21   # ~1 trading month
EXTENSION_CEILING_PCT = 8.0       # price no more than this % above resistance

CONFIRMATION_CHECK_LABELS = (
    "Price > resistance",
    "Volume > 2x average",
    "Close near high",
    "RSI 55-75",
    "Above 20/50 DMA",
    "Relative strength vs NIFTY",
    "Sector confirmation",
    "NIFTY regime",
    "No excessive extension",
)

SECTOR_DATA_NOTE = (
    "Sector confirmation needs a symbol-to-sector map and sector-index history, "
    "neither of which this pipeline has yet -- always shown as N/A."
)


@dataclass
class ConfirmationCheck:
    label: str
    passed: Optional[bool]     # None = not available, excluded from the score
    detail: str = ""


@dataclass
class ConfirmationScore:
    checks: List[ConfirmationCheck] = field(default_factory=list)

    @property
    def passed_count(self) -> int:
        return sum(1 for c in self.checks if c.passed is True)

    @property
    def available_count(self) -> int:
        return sum(1 for c in self.checks if c.passed is not None)

    @property
    def failed(self) -> List[ConfirmationCheck]:
        return [c for c in self.checks if c.passed is False]

    def label(self) -> str:
        avail = self.available_count
        if avail == 0:
            return "No checks available"
        return f"{self.passed_count}/{avail}"


# -----------------------------------------------------------------------
# Small self-contained bar-geometry helpers -- deliberately not imported
# from utils.breakout_failure (which has its own, inverse-purpose close-
# position/volume helpers) so this module stays a standalone overlay,
# same "reimplemented standalone to keep this module dependency-free"
# convention utils.market_regime uses for its own resistance check.
# -----------------------------------------------------------------------
def _volume_ratio(df, i, lookback=VOLUME_AVG_LOOKBACK):
    if i < lookback:
        return None
    avg_vol = df["Volume"].iloc[i - lookback:i].mean()
    if avg_vol is None or avg_vol <= 0:
        return None
    return float(df["Volume"].iloc[i]) / float(avg_vol)


def _close_position(df, i):
    bar = df.iloc[i]
    h, l, c = float(bar["High"]), float(bar["Low"]), float(bar["Close"])
    rng = h - l
    if rng <= 0:
        return 1.0
    return (c - l) / rng


def _relative_strength(df, i, nifty_closes, lookback=RELATIVE_STRENGTH_LOOKBACK):
    """Stock's trailing `lookback`-bar return vs NIFTY 50's over the same
    number of bars, aligned POSITIONALLY (last N+1 bars of each series)
    rather than by calendar date -- the two calendars don't always match
    bar-for-bar (different holiday lists), and a positional read is
    close enough for a single yes/no confirmation check. Returns None if
    either series doesn't have enough history."""
    if nifty_closes is None or len(nifty_closes) < lookback + 1:
        return None
    if i < lookback:
        return None
    stock_closes = df["Close"].iloc[i - lookback:i + 1]
    if len(stock_closes) < lookback + 1:
        return None
    stock_start, stock_end = float(stock_closes.iloc[0]), float(stock_closes.iloc[-1])
    nifty_tail = nifty_closes.iloc[-(lookback + 1):]
    nifty_start, nifty_end = float(nifty_tail.iloc[0]), float(nifty_tail.iloc[-1])
    if stock_start <= 0 or nifty_start <= 0:
        return None
    stock_return = (stock_end - stock_start) / stock_start
    nifty_return = (nifty_end - nifty_start) / nifty_start
    return stock_return - nifty_return


# -----------------------------------------------------------------------
# Orchestrator -- what breakout_controller.py calls
# -----------------------------------------------------------------------
def compute_confirmation_score(symbol, df: pd.DataFrame, i: int, regime=None) -> ConfirmationScore:
    """
    symbol, df, i: same convention as everywhere else in this project --
      df is the symbol's full OHLCV history, i is today's positional
      index (only df.iloc[:i+1] is ever looked at, no lookahead).
    regime: the RegimeResult from utils.market_regime.compute_market_regime()
      for this run -- feeds check 8 (NIFTY regime) and check 6 (relative
      strength, via regime.nifty_closes). Both checks score None if
      regime is None or the underlying data wasn't available this run.

    Never raises -- each of the 9 checks is independently wrapped, so one
    check's data gap never takes down the other 8, same fail-soft pattern
    as utils.breakout_targets.compute_targets.
    """
    checks: List[ConfirmationCheck] = []
    today_close = float(df["Close"].iloc[i])

    # 1 & 9 share the same resistance level, computed once.
    try:
        resistance = trailing_resistance(df.iloc[:i + 1], lookback=RESISTANCE_LOOKBACK)
    except Exception:
        resistance = None

    if resistance is None:
        checks.append(ConfirmationCheck("Price > resistance", None, "Not enough trailing history for a resistance read."))
        extension_check = ConfirmationCheck("No excessive extension", None, "Not enough trailing history for a resistance read.")
    else:
        checks.append(ConfirmationCheck(
            "Price > resistance", today_close > resistance,
            f"Close ₹{today_close:,.2f} vs trailing resistance ₹{resistance:,.2f}.",
        ))
        extension_pct = (today_close - resistance) / resistance * 100
        extension_check = ConfirmationCheck(
            "No excessive extension", extension_pct <= EXTENSION_CEILING_PCT,
            f"{extension_pct:+.1f}% vs resistance (ceiling {EXTENSION_CEILING_PCT:.0f}%).",
        )

    # 2. Volume > 2x average
    try:
        vol_ratio = _volume_ratio(df, i)
    except Exception:
        vol_ratio = None
    if vol_ratio is None:
        checks.append(ConfirmationCheck("Volume > 2x average", None, "Not enough trailing history for a volume average."))
    else:
        checks.append(ConfirmationCheck(
            "Volume > 2x average", vol_ratio >= VOLUME_CONFIRM_MULTIPLE,
            f"Volume {vol_ratio:.1f}x the {VOLUME_AVG_LOOKBACK}-day average.",
        ))

    # 3. Close near high
    try:
        close_pos = _close_position(df, i)
        checks.append(ConfirmationCheck(
            "Close near high", close_pos >= CLOSE_NEAR_HIGH_MIN_POSITION,
            f"Closed at the {close_pos*100:.0f}th percentile of today's range.",
        ))
    except Exception:
        checks.append(ConfirmationCheck("Close near high", None, "Could not read today's bar."))

    # 4. RSI 55-75
    try:
        rsi_series = compute_rsi(df["Close"].iloc[:i + 1])
        rsi_today = float(rsi_series.iloc[-1])
        lo, hi = RSI_BAND
        checks.append(ConfirmationCheck(
            "RSI 55-75", lo <= rsi_today <= hi,
            f"RSI {rsi_today:.0f} (band {lo:.0f}-{hi:.0f}).",
        ))
    except Exception:
        checks.append(ConfirmationCheck("RSI 55-75", None, "RSI could not be computed."))

    # 5. Above 20/50 DMA
    try:
        closes = df["Close"].iloc[:i + 1]
        if len(closes) < DMA_LONG:
            checks.append(ConfirmationCheck("Above 20/50 DMA", None, f"Not enough history for a {DMA_LONG}-day average."))
        else:
            sma_short = float(closes.rolling(DMA_SHORT).mean().iloc[-1])
            sma_long = float(closes.rolling(DMA_LONG).mean().iloc[-1])
            checks.append(ConfirmationCheck(
                "Above 20/50 DMA", today_close > sma_short and today_close > sma_long,
                f"Close ₹{today_close:,.2f} vs {DMA_SHORT}DMA ₹{sma_short:,.2f} / {DMA_LONG}DMA ₹{sma_long:,.2f}.",
            ))
    except Exception:
        checks.append(ConfirmationCheck("Above 20/50 DMA", None, "Moving averages could not be computed."))

    # 6. Relative strength vs NIFTY
    nifty_closes = getattr(regime, "nifty_closes", None) if regime else None
    try:
        rs = _relative_strength(df, i, nifty_closes)
    except Exception:
        rs = None
    if rs is None:
        checks.append(ConfirmationCheck(
            "Relative strength vs NIFTY", None,
            "NIFTY history unavailable this run or not enough trailing history." if nifty_closes is None
            else "Not enough trailing history for a relative-strength read.",
        ))
    else:
        checks.append(ConfirmationCheck(
            "Relative strength vs NIFTY", rs > 0,
            f"{RELATIVE_STRENGTH_LOOKBACK}-day return {rs*100:+.1f}pp vs NIFTY 50 over the same window.",
        ))

    # 7. Sector confirmation -- not available in this pipeline yet
    checks.append(ConfirmationCheck("Sector confirmation", None, SECTOR_DATA_NOTE))

    # 8. NIFTY regime
    if regime is None:
        checks.append(ConfirmationCheck("NIFTY regime", None, "Market regime unavailable this run."))
    else:
        checks.append(ConfirmationCheck(
            "NIFTY regime", regime.entries_supportive,
            f"{regime.label()} ({regime.score}/5 checks).",
        ))

    checks.append(extension_check)

    return ConfirmationScore(checks=checks)