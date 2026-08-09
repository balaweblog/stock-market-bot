"""
utils/market_regime.py

NIFTY 500 market regime filter, computed once per screener run and used
to decide how permissive the rest of controllers/breakout_controller.py
is about what it calls "Confirmed". The 8 pattern detectors in
utils.breakout_patterns and the per-symbol backtest in
utils.breakout_backtest fire exactly the same way in every regime --
this module never touches detection itself. What changes is the bar a
fired signal has to clear to be shown as Confirmed vs pushed to Watch.

Five checks, same fail-soft convention used throughout this project
(controllers/breakout_controller.py's own bhavcopy and NIFTY-trend
fetches already do this: if a data source can't be reached, the
individual check it feeds defaults to True -- "a data hiccup shouldn't
block signals" -- and it's logged as a note so the email stays honest
about what it could and couldn't verify that run):

  1. NIFTY 50 close > its 50-day moving average
  2. 50-day moving average > 200-day moving average
  3. Breadth: >60% of THIS RUN'S scanned universe closing above their
     OWN trailing 50-day moving average -- reuses the same {symbol: df}
     history breakout_controller.py already pulled for the day's scan,
     no extra universe-wide fetch.
  4. NIFTY 50 close at/above its own trailing resistance -- i.e. the
     index itself is in a breakout structure, not just "up". Same
     trailing-high convention as utils.entry_classification.trailing_resistance,
     reimplemented standalone here to keep this module dependency-free.
  5. India VIX supportive -- at/below VIX_SUPPORTIVE_CEILING, i.e. no
     acute fear priced into the options market.

Score = count of checks that came back True (0-5):
  4-5  -> BULLISH  (breakout strategies fully enabled)
  2-3  -> NEUTRAL  (normal thresholds -- no extra leniency, no extra
                     restriction)
  0-1  -> BEARISH  (only the strongest breakouts are allowed through --
                     see breakout_controller.py's regime-gated Confirmed
                     bar: Strong/Excellent Quality AND Risk:Reward at the
                     PREFERRED threshold, not just the normal minimum)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

from utils.logger import log

NIFTY_TICKER = "^NSEI"
VIX_TICKER = "^INDIAVIX"

FAST_DMA = 50
SLOW_DMA = 200
RESISTANCE_LOOKBACK = 60       # trading days, excluding the most recent bar
BREADTH_LOOKBACK = 50          # each stock's own trailing MA for the breadth check
BREADTH_MIN_UNIVERSE = 30      # need at least this many eligible symbols for breadth to count at all
BREADTH_BULLISH_PCT = 60.0
VIX_SUPPORTIVE_CEILING = 20.0

BULLISH_MIN_SCORE = 4          # of 5
NEUTRAL_MIN_SCORE = 2          # of 5

# Regime-gated Confirmed bar for Bearish runs (see breakout_controller.py) --
# lives here since it's part of "what the regime means", read from there.
BEAR_MARKET_MIN_QUALITY_SCORE = 65   # Strong/Excellent on the 0-100 Quality Score scale


class MarketRegime(Enum):
    BULLISH = "Bullish"
    NEUTRAL = "Neutral"
    BEARISH = "Bearish"


REGIME_EMOJI = {
    MarketRegime.BULLISH: "🟢",
    MarketRegime.NEUTRAL: "🟡",
    MarketRegime.BEARISH: "🔴",
}

REGIME_INTERPRETATION = {
    MarketRegime.BULLISH: "Breakout strategies fully enabled -- normal Confirmed/Watch thresholds apply.",
    MarketRegime.NEUTRAL: "Normal thresholds apply -- no extra leniency, no extra restriction.",
    MarketRegime.BEARISH: (
        "Only the strongest breakouts are allowed through: Confirmed now also requires a "
        f"Strong/Excellent Quality score (\u226570) and Risk:Reward at the preferred bar, not just "
        "the normal minimum -- everything else that still passes the backtest is shown in Watch instead."
    ),
}

CHECK_LABELS = (
    "NIFTY > 50DMA",
    "50DMA > 200DMA",
    "Breadth > 60%",
    "NIFTY above prior resistance",
    "VIX supportive",
)


@dataclass
class RegimeResult:
    regime: MarketRegime
    score: int                             # 0-5, count of True checks
    checks: Dict[str, bool]                # check label -> True/False (fail-soft defaults already resolved)
    detail: Dict[str, object]              # raw numbers behind each check, for the email breakdown
    notes: List[str] = field(default_factory=list)   # fail-soft / data-availability caveats
    # NIFTY 50 close series from this run's fetch (see compute_market_regime),
    # stashed here so other overlay modules -- currently
    # utils.breakout_confirmation's relative-strength check -- can reuse the
    # SAME NIFTY history rather than re-fetching it. None if the NIFTY fetch
    # failed this run (checks 1/2/4 above will already be showing their
    # fail-soft defaults in that case too).
    nifty_closes: Optional[pd.Series] = None

    def label(self) -> str:
        return f"{REGIME_EMOJI[self.regime]} {self.regime.value}"

    def interpretation(self) -> str:
        return REGIME_INTERPRETATION[self.regime]

    @property
    def entries_supportive(self) -> bool:
        """Feeds utils.entry_classification's 'Market trend supportive'
        Fresh-Breakout check. Bearish regime withholds it; Neutral and
        Bullish both allow it -- Neutral isn't restrictive on its own,
        only Bearish is (see module docstring)."""
        return self.regime != MarketRegime.BEARISH

    @property
    def only_strongest(self) -> bool:
        """True only in Bearish -- breakout_controller.py uses this to
        raise the Confirmed bar rather than filtering it here, so every
        row's bucket assignment still lives in one place."""
        return self.regime == MarketRegime.BEARISH


def _fetch_index_history(ticker, period, label):
    try:
        df = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False)
        df = df.dropna(subset=["Close"])
        return df
    except Exception as e:
        log.warning(f"Market Regime: {label} fetch failed ({ticker}): {e}")
        return None


def _trailing_resistance(closes: pd.Series, lookback=RESISTANCE_LOOKBACK, exclude_last=1) -> Optional[float]:
    """Highest close over the trailing window, excluding the most recent
    `exclude_last` bars -- same convention as
    utils.entry_classification.trailing_resistance, reimplemented
    standalone so this module has no dependency on the per-stock
    classifier."""
    if len(closes) < lookback + exclude_last + 1:
        return None
    window = closes.iloc[-(lookback + exclude_last):-exclude_last]
    if window.empty:
        return None
    return float(window.max())


def _compute_breadth(histories, lookback=BREADTH_LOOKBACK):
    """% of this run's scanned universe (the same {symbol: df} the rest
    of the screener already pulled) closing above their OWN trailing
    `lookback`-day moving average. Returns (pct, above_count, total_count)
    -- pct is None if fewer than BREADTH_MIN_UNIVERSE symbols had enough
    history to even ask the question, so the caller can fail-soft it."""
    above, total = 0, 0
    for df in histories.values():
        if len(df) < lookback + 1:
            continue
        closes = df["Close"]
        sma = closes.rolling(lookback).mean().iloc[-1]
        if pd.isna(sma) or sma <= 0:
            continue
        total += 1
        if float(closes.iloc[-1]) > float(sma):
            above += 1
    if total < BREADTH_MIN_UNIVERSE:
        return None, above, total
    return (above / total) * 100.0, above, total


def compute_market_regime(histories) -> RegimeResult:
    """
    histories: {symbol: OHLCV df} -- the SAME dict
    breakout_controller.fetch_universe_history() already built for the
    day's scan, reused here for the breadth check. Only NIFTY 50 and
    India VIX are fetched fresh, both single-ticker calls -- call this
    AFTER fetch_universe_history() so breadth reflects the actual
    universe that was scanned, not a separate/stale pull.
    """
    notes: List[str] = []
    checks: Dict[str, bool] = {}
    detail: Dict[str, object] = {}
    nifty_closes: Optional[pd.Series] = None

    # ---- NIFTY 50: checks 1, 2, 4 ----
    nifty_df = _fetch_index_history(NIFTY_TICKER, period="2y", label="NIFTY 50")
    if nifty_df is None or len(nifty_df) < SLOW_DMA:
        checks["NIFTY > 50DMA"] = True
        checks["50DMA > 200DMA"] = True
        checks["NIFTY above prior resistance"] = True
        notes.append("NIFTY 50 fetch failed or had insufficient history -- defaulted its 3 checks to True.")
        detail["nifty_close"] = None
        detail["nifty_50dma"] = None
        detail["nifty_200dma"] = None
        detail["nifty_resistance"] = None
    else:
        close = nifty_df["Close"]
        nifty_closes = close
        last_close = float(close.iloc[-1])
        sma_fast = float(close.rolling(FAST_DMA).mean().iloc[-1])
        sma_slow = float(close.rolling(SLOW_DMA).mean().iloc[-1])
        resistance = _trailing_resistance(close)

        checks["NIFTY > 50DMA"] = last_close > sma_fast
        checks["50DMA > 200DMA"] = sma_fast > sma_slow
        if resistance is None:
            checks["NIFTY above prior resistance"] = True
            notes.append("Not enough NIFTY history for the prior-resistance check -- defaulted to True.")
        else:
            checks["NIFTY above prior resistance"] = last_close >= resistance

        detail["nifty_close"] = round(last_close, 2)
        detail["nifty_50dma"] = round(sma_fast, 2)
        detail["nifty_200dma"] = round(sma_slow, 2)
        detail["nifty_resistance"] = round(resistance, 2) if resistance is not None else None

    # ---- Breadth: check 3 ----
    breadth_pct, above, total = _compute_breadth(histories)
    if breadth_pct is None:
        checks["Breadth > 60%"] = True
        notes.append(
            f"Only {total} symbols had enough history for the breadth check "
            f"(need {BREADTH_MIN_UNIVERSE}+) -- defaulted to True."
        )
    else:
        checks["Breadth > 60%"] = breadth_pct > BREADTH_BULLISH_PCT
    detail["breadth_pct"] = round(breadth_pct, 1) if breadth_pct is not None else None
    detail["breadth_above"] = above
    detail["breadth_total"] = total

    # ---- India VIX: check 5 ----
    vix_df = _fetch_index_history(VIX_TICKER, period="6mo", label="India VIX")
    if vix_df is None or vix_df.empty:
        checks["VIX supportive"] = True
        notes.append("India VIX fetch failed -- defaulted to True.")
        detail["vix"] = None
    else:
        vix_last = float(vix_df["Close"].iloc[-1])
        checks["VIX supportive"] = vix_last <= VIX_SUPPORTIVE_CEILING
        detail["vix"] = round(vix_last, 2)

    score = sum(1 for v in checks.values() if v)

    if score >= BULLISH_MIN_SCORE:
        regime = MarketRegime.BULLISH
    elif score >= NEUTRAL_MIN_SCORE:
        regime = MarketRegime.NEUTRAL
    else:
        regime = MarketRegime.BEARISH

    return RegimeResult(regime=regime, score=score, checks=checks, detail=detail, notes=notes, nifty_closes=nifty_closes)