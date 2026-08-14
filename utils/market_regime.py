"""
utils/market_regime.py

NIFTY 500 market regime score, computed once per screener run and used
to decide how permissive the rest of controllers/breakout_controller.py
is about what it calls "Confirmed". The 8 pattern detectors in
utils.breakout_patterns and the per-symbol backtest in
utils.breakout_backtest fire exactly the same way in every regime --
this module never touches detection itself. What changes is the bar a
fired signal has to clear to be shown as Confirmed vs pushed to Watch.

WEIGHTED MODEL (0-100, 8 factors -- see FACTOR_WEIGHTS):
  NIFTY Trend               20   NIFTY 50 close > its own 50DMA
  NIFTY 50DMA > 200DMA      15   golden-cross structure
  Market Breadth            15   % of this run's scanned universe above their OWN 50DMA
  NIFTY Relative Strength   10   NIFTY's own RELATIVE_STRENGTH_LOOKBACK-day rate of change > 0
  India VIX                 10   at/below VIX_SUPPORTIVE_CEILING
  Sector Breadth            10   % of this run's synthetic sector indices above their OWN 50DMA
  Advance/Decline           10   % of the scanned universe closing up vs down today
  Volume Breadth            10   % of the scanned universe trading above its OWN 20-day avg volume

  NOTE on "NIFTY Relative Strength": relative strength needs a stated
  benchmark, and none was given, so this factor reads NIFTY's OWN
  short-term momentum (20-day rate of change) rather than NIFTY vs.
  some other index/asset. If you meant NIFTY vs. a specific benchmark
  (S&P 500, gold, US 10Y, etc.), say which and this factor swaps for
  that comparison instead.

MISSING DATA IS NEVER TREATED AS BULLISH. This is the one deliberate
reversal from this project's usual fail-soft convention (elsewhere,
e.g. this module's own NIFTY-trend/breadth/VIX fetches used to default
an unreachable check to True -- "a data hiccup shouldn't block
signals"). For the regime score specifically, that convention would
mean a bad data day quietly nudges every signal toward the permissive
end, which is exactly backwards for something whose whole job is to
tighten up when conditions are shaky. So here instead:
  - A factor that can't be computed (fetch failed, too little history,
    universe/sector sample too thin) is EXCLUDED, not defaulted True.
    Its weight is dropped from the denominator and the score is
    renormalized over whatever weight IS available, so the score stays
    on a 0-100 scale regardless of how many factors came in -- and a
    note is logged saying which factor was skipped and why.
  - If the factors that DID come in cover less than
    MIN_COVERAGE_FOR_UNCAPPED_REGIME points of weight, a computed
    Aggressive or Normal read is capped down to Selective -- a high
    score built on a fifth of the model isn't trustworthy enough to
    greenlight aggressive entries, even though the same weight-excluded
    math is honest about a fully-covered run.
  - If NOTHING could be computed at all this run, the regime defaults
    to Avoid (the most conservative band), not Aggressive/Normal --
    never guess bullish when you have nothing to go on.

Score bands (0-100, MarketRegime enum):
  80-100 -> AGGRESSIVE  (breakout strategies fully enabled)
  65-79  -> NORMAL      (normal thresholds -- no extra leniency, no extra
                          restriction)
  50-64  -> SELECTIVE   (only the strongest breakouts are allowed through --
                          see breakout_controller.py's regime-gated Confirmed
                          bar: Strong/Excellent Quality AND Risk:Reward at the
                          PREFERRED threshold, not just the normal minimum)
  0-49   -> AVOID       (same stricter Confirmed bar as Selective, PLUS
                          "Market trend supportive" is withheld from every
                          Fresh-Breakout entry classification -- see
                          entries_supportive below)
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
RELATIVE_STRENGTH_LOOKBACK = 20    # trading days -- NIFTY's own rate-of-change window (see module docstring)
BREADTH_LOOKBACK = 50              # each stock's own trailing MA for the Market Breadth factor
BREADTH_MIN_UNIVERSE = 30          # need at least this many eligible symbols for a breadth-style factor to count at all
BREADTH_BULLISH_PCT = 60.0
VOLUME_BREADTH_LOOKBACK = 20       # trading days, each stock's own trailing avg-volume window
VOLUME_BREADTH_BULLISH_PCT = 50.0
ADVANCE_DECLINE_BULLISH_PCT = 55.0
SECTOR_TREND_SMA_PERIOD = 50       # each synthetic sector index's own trailing MA for the Sector Breadth factor
SECTOR_BREADTH_BULLISH_PCT = 60.0
SECTOR_BREADTH_MIN_SECTORS = 4     # need at least this many sectors with a valid 50DMA read for the factor to count
VIX_SUPPORTIVE_CEILING = 20.0

# label -> weight, must sum to 100. Order here is also the display order
# (see CHECK_LABELS below).
FACTOR_WEIGHTS: Dict[str, int] = {
    "NIFTY Trend": 20,
    "NIFTY 50DMA > 200DMA": 15,
    "Market Breadth": 15,
    "NIFTY Relative Strength": 10,
    "India VIX": 10,
    "Sector Breadth": 10,
    "Advance/Decline": 10,
    "Volume Breadth": 10,
}
TOTAL_WEIGHT = sum(FACTOR_WEIGHTS.values())
assert TOTAL_WEIGHT == 100, f"FACTOR_WEIGHTS must sum to 100, got {TOTAL_WEIGHT}"

CHECK_LABELS = tuple(FACTOR_WEIGHTS.keys())

AGGRESSIVE_MIN_SCORE = 80
NORMAL_MIN_SCORE = 65
SELECTIVE_MIN_SCORE = 50
# Below this, an Aggressive/Normal READING is capped down to Selective --
# see module docstring's "missing data is never treated as bullish".
MIN_COVERAGE_FOR_UNCAPPED_REGIME = 50

# Regime-gated Confirmed bar for Selective/Avoid runs (see
# breakout_controller.py) -- lives here since it's part of "what the
# regime means", read from there.
STRICT_CONFIRMED_MIN_QUALITY_SCORE = 65   # Strong/Excellent on the 0-100 Quality Score scale


class MarketRegime(Enum):
    AGGRESSIVE = "Aggressive Breakout"
    NORMAL = "Normal"
    SELECTIVE = "Selective"
    AVOID = "Avoid New Breakouts"


REGIME_EMOJI = {
    MarketRegime.AGGRESSIVE: "🟢",
    MarketRegime.NORMAL: "🟡",
    MarketRegime.SELECTIVE: "🟠",
    MarketRegime.AVOID: "🔴",
}

REGIME_INTERPRETATION = {
    MarketRegime.AGGRESSIVE: "Breakout strategies fully enabled -- normal Confirmed/Watch thresholds apply.",
    MarketRegime.NORMAL: "Normal thresholds apply -- no extra leniency, no extra restriction.",
    MarketRegime.SELECTIVE: (
        "Only the strongest breakouts are allowed through: Confirmed now also requires a "
        f"Strong/Excellent Quality score (\u2265{STRICT_CONFIRMED_MIN_QUALITY_SCORE}) and Risk:Reward at the "
        "preferred bar, not just the normal minimum -- everything else that still passes the backtest is shown "
        "in Watch instead."
    ),
    MarketRegime.AVOID: (
        "Same stricter Confirmed bar as Selective (Quality \u2265"
        f"{STRICT_CONFIRMED_MIN_QUALITY_SCORE} and preferred Risk:Reward), PLUS 'Market trend supportive' is "
        "withheld from every Fresh-Breakout entry classification this run -- conditions are broadly working "
        "against new breakouts, not just thin on the strongest ones."
    ),
}


@dataclass
class RegimeResult:
    regime: MarketRegime
    score: int                             # 0-100, weighted % of AVAILABLE factors that passed (see module docstring)
    checks: Dict[str, Optional[bool]]      # factor label -> True/False, or None if excluded (data unavailable)
    detail: Dict[str, object]              # raw numbers behind each factor, for the email breakdown
    factors_available: int                 # how many of the 8 factors had usable data this run
    factors_total: int = len(FACTOR_WEIGHTS)
    notes: List[str] = field(default_factory=list)   # exclusion / data-availability / cap explanations
    # NIFTY 50 close series from this run's fetch (see compute_market_regime),
    # stashed here so other overlay modules -- currently
    # utils.breakout_confirmation's relative-strength check -- can reuse the
    # SAME NIFTY history rather than re-fetching it. None if the NIFTY fetch
    # failed this run (the NIFTY-derived factors will already be showing as
    # excluded in that case too).
    nifty_closes: Optional[pd.Series] = None

    def label(self) -> str:
        return f"{REGIME_EMOJI[self.regime]} {self.regime.value}"

    def interpretation(self) -> str:
        return REGIME_INTERPRETATION[self.regime]

    @property
    def entries_supportive(self) -> bool:
        """Feeds utils.entry_classification's 'Market trend supportive'
        Fresh-Breakout check. Withheld only in Avoid; Selective, Normal
        and Aggressive all allow it -- Selective is restrictive on the
        Confirmed bar (see only_strongest) but not on entries themselves."""
        return self.regime != MarketRegime.AVOID

    @property
    def only_strongest(self) -> bool:
        """True in Selective and Avoid -- breakout_controller.py uses
        this to raise the Confirmed bar rather than filtering it here,
        so every row's bucket assignment still lives in one place."""
        return self.regime in (MarketRegime.SELECTIVE, MarketRegime.AVOID)


def _fetch_index_history(ticker, period, label):
    try:
        df = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False)
        df = df.dropna(subset=["Close"])
        return df
    except Exception as e:
        log.warning(f"Market Regime: {label} fetch failed ({ticker}): {e}")
        return None


def _compute_breadth(histories, lookback=BREADTH_LOOKBACK):
    """% of this run's scanned universe (the same {symbol: df} the rest
    of the screener already pulled) closing above their OWN trailing
    `lookback`-day moving average. Returns (pct, above_count, total_count)
    -- pct is None if fewer than BREADTH_MIN_UNIVERSE symbols had enough
    history to even ask the question, so the caller excludes the factor."""
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


def _compute_advance_decline(histories):
    """% of this run's scanned universe closing up vs. down from the
    PRIOR bar (unchanged closes are excluded from both the numerator and
    denominator). Returns (pct_advancing, advancers, decliners, total)
    -- pct is None if fewer than BREADTH_MIN_UNIVERSE symbols had a
    valid up/down read."""
    up, down, total = 0, 0, 0
    for df in histories.values():
        if len(df) < 2:
            continue
        closes = df["Close"]
        last, prev = float(closes.iloc[-1]), float(closes.iloc[-2])
        if prev == 0 or last == prev:
            continue
        total += 1
        if last > prev:
            up += 1
        else:
            down += 1
    if total < BREADTH_MIN_UNIVERSE:
        return None, up, down, total
    return (up / total) * 100.0, up, down, total


def _compute_volume_breadth(histories, lookback=VOLUME_BREADTH_LOOKBACK):
    """% of this run's scanned universe trading above its OWN trailing
    `lookback`-day average volume today -- a participation read (is the
    move backed by broad volume, or a handful of names on light volume).
    Returns (pct, above_count, total_count) -- pct is None if fewer than
    BREADTH_MIN_UNIVERSE symbols had enough volume history."""
    above, total = 0, 0
    for df in histories.values():
        if "Volume" not in df.columns or len(df) < lookback + 1:
            continue
        vol = df["Volume"]
        avg = vol.rolling(lookback).mean().iloc[-1]
        if pd.isna(avg) or avg <= 0:
            continue
        total += 1
        if float(vol.iloc[-1]) > float(avg):
            above += 1
    if total < BREADTH_MIN_UNIVERSE:
        return None, above, total
    return (above / total) * 100.0, above, total


def _compute_sector_breadth(sector_context, sma_period=SECTOR_TREND_SMA_PERIOD, min_sectors=SECTOR_BREADTH_MIN_SECTORS):
    """% of this run's synthetic sector indices (from
    breakout_controller.build_sector_context -- {sector: {"index": pd.Series, ...}})
    trading above their OWN trailing `sma_period`-day average -- i.e. how
    many SECTORS are themselves in an uptrend, distinct from Market
    Breadth's stock-level read. Deliberately takes the already-built
    sector_context dict rather than raw histories/SECTOR_MAP so this
    module doesn't have to know how synthetic sector indices get built.
    Returns (pct, above_count, total_count) -- pct is None if sector_context
    is empty (SECTOR_MAP not populated) or fewer than min_sectors sectors
    had enough history for a 50DMA read."""
    if not sector_context:
        return None, 0, 0
    above, total = 0, 0
    for info in sector_context.values():
        idx = info.get("index") if isinstance(info, dict) else None
        if idx is None or idx.empty or len(idx) < sma_period:
            continue
        sma = float(idx.iloc[-sma_period:].mean())
        total += 1
        if float(idx.iloc[-1]) > sma:
            above += 1
    if total < min_sectors:
        return None, above, total
    return (above / total) * 100.0, above, total


def compute_market_regime(histories, sector_context=None) -> RegimeResult:
    """
    histories: {symbol: OHLCV df} -- the SAME dict
    breakout_controller.fetch_universe_history() already built for the
    day's scan, reused here for Market Breadth, Advance/Decline and
    Volume Breadth. Only NIFTY 50 and India VIX are fetched fresh, both
    single-ticker calls -- call this AFTER fetch_universe_history() so
    those three factors reflect the actual universe that was scanned,
    not a separate/stale pull.

    sector_context: {sector_name: {"index": pd.Series, ...}} from
    breakout_controller.build_sector_context(), or None/{} if SECTOR_MAP
    isn't populated yet. Feeds Sector Breadth only -- pass it in built
    from the SAME histories/run if you have it; the factor is simply
    excluded (not defaulted bullish) when it's missing.
    """
    notes: List[str] = []
    checks: Dict[str, Optional[bool]] = {}
    detail: Dict[str, object] = {}
    nifty_closes: Optional[pd.Series] = None

    # ---- NIFTY 50: Trend, 50DMA>200DMA, Relative Strength ----
    nifty_df = _fetch_index_history(NIFTY_TICKER, period="2y", label="NIFTY 50")
    if nifty_df is None or len(nifty_df) < SLOW_DMA:
        checks["NIFTY Trend"] = None
        checks["NIFTY 50DMA > 200DMA"] = None
        checks["NIFTY Relative Strength"] = None
        notes.append(
            "NIFTY 50 fetch failed or had insufficient history -- Trend, 50DMA>200DMA and Relative Strength "
            "factors excluded from this run's score (not defaulted bullish)."
        )
        detail["nifty_close"] = None
        detail["nifty_50dma"] = None
        detail["nifty_200dma"] = None
        detail["nifty_roc_pct"] = None
    else:
        close = nifty_df["Close"]
        nifty_closes = close
        last_close = float(close.iloc[-1])
        sma_fast = float(close.rolling(FAST_DMA).mean().iloc[-1])
        sma_slow = float(close.rolling(SLOW_DMA).mean().iloc[-1])

        checks["NIFTY Trend"] = last_close > sma_fast
        checks["NIFTY 50DMA > 200DMA"] = sma_fast > sma_slow
        detail["nifty_close"] = round(last_close, 2)
        detail["nifty_50dma"] = round(sma_fast, 2)
        detail["nifty_200dma"] = round(sma_slow, 2)

        if len(close) > RELATIVE_STRENGTH_LOOKBACK:
            prior = float(close.iloc[-RELATIVE_STRENGTH_LOOKBACK - 1])
            roc_pct = (last_close - prior) / prior * 100.0 if prior else None
            checks["NIFTY Relative Strength"] = (roc_pct > 0) if roc_pct is not None else None
            detail["nifty_roc_pct"] = round(roc_pct, 2) if roc_pct is not None else None
            if roc_pct is None:
                notes.append("NIFTY Relative Strength: prior close was zero/invalid -- factor excluded.")
        else:
            checks["NIFTY Relative Strength"] = None
            detail["nifty_roc_pct"] = None
            notes.append(
                f"Not enough NIFTY history for the {RELATIVE_STRENGTH_LOOKBACK}-day Relative Strength read -- "
                "factor excluded."
            )

    # ---- Market Breadth ----
    breadth_pct, above, total = _compute_breadth(histories)
    if breadth_pct is None:
        checks["Market Breadth"] = None
        notes.append(
            f"Only {total} symbols had enough history for the Market Breadth check "
            f"(need {BREADTH_MIN_UNIVERSE}+) -- factor excluded."
        )
    else:
        checks["Market Breadth"] = breadth_pct > BREADTH_BULLISH_PCT
    detail["breadth_pct"] = round(breadth_pct, 1) if breadth_pct is not None else None
    detail["breadth_above"] = above
    detail["breadth_total"] = total

    # ---- India VIX ----
    vix_df = _fetch_index_history(VIX_TICKER, period="6mo", label="India VIX")
    if vix_df is None or vix_df.empty:
        checks["India VIX"] = None
        notes.append("India VIX fetch failed -- factor excluded.")
        detail["vix"] = None
    else:
        vix_last = float(vix_df["Close"].iloc[-1])
        checks["India VIX"] = vix_last <= VIX_SUPPORTIVE_CEILING
        detail["vix"] = round(vix_last, 2)

    # ---- Sector Breadth ----
    sector_pct, sec_above, sec_total = _compute_sector_breadth(sector_context)
    if sector_pct is None:
        checks["Sector Breadth"] = None
        if not sector_context:
            notes.append("No sector-classification data available this run (SECTOR_MAP empty) -- Sector Breadth factor excluded.")
        else:
            notes.append(
                f"Only {sec_total} sectors had enough history for a trend read "
                f"(need {SECTOR_BREADTH_MIN_SECTORS}+) -- Sector Breadth factor excluded."
            )
    else:
        checks["Sector Breadth"] = sector_pct > SECTOR_BREADTH_BULLISH_PCT
    detail["sector_breadth_pct"] = round(sector_pct, 1) if sector_pct is not None else None
    detail["sector_breadth_above"] = sec_above
    detail["sector_breadth_total"] = sec_total

    # ---- Advance/Decline ----
    adv_pct, up, down, ad_total = _compute_advance_decline(histories)
    if adv_pct is None:
        checks["Advance/Decline"] = None
        notes.append(
            f"Only {ad_total} symbols had a valid prior close for the Advance/Decline read "
            f"(need {BREADTH_MIN_UNIVERSE}+) -- factor excluded."
        )
    else:
        checks["Advance/Decline"] = adv_pct > ADVANCE_DECLINE_BULLISH_PCT
    detail["advance_decline_pct"] = round(adv_pct, 1) if adv_pct is not None else None
    detail["advancers"] = up
    detail["decliners"] = down
    detail["advance_decline_total"] = ad_total

    # ---- Volume Breadth ----
    volb_pct, vb_above, vb_total = _compute_volume_breadth(histories)
    if volb_pct is None:
        checks["Volume Breadth"] = None
        notes.append(
            f"Only {vb_total} symbols had enough volume history for the Volume Breadth read "
            f"(need {BREADTH_MIN_UNIVERSE}+) -- factor excluded."
        )
    else:
        checks["Volume Breadth"] = volb_pct > VOLUME_BREADTH_BULLISH_PCT
    detail["volume_breadth_pct"] = round(volb_pct, 1) if volb_pct is not None else None
    detail["volume_breadth_above"] = vb_above
    detail["volume_breadth_total"] = vb_total

    # ---- Weighted aggregation -- missing data EXCLUDED, never counted as bullish ----
    available_weight = 0
    passed_weight = 0
    factors_available = 0
    for label, weight in FACTOR_WEIGHTS.items():
        result = checks.get(label)
        if result is None:
            continue
        available_weight += weight
        factors_available += 1
        if result:
            passed_weight += weight

    if available_weight == 0:
        # Nothing could be computed this run at all -- don't fabricate a
        # score. Default to the most conservative regime and say so loudly.
        score = 0
        regime = MarketRegime.AVOID
        notes.append(
            "No market-regime factors could be computed this run (data sources unreachable) -- "
            "defaulting to Avoid New Breakouts rather than guessing bullish."
        )
    else:
        score = round(passed_weight / available_weight * 100)
        if score >= AGGRESSIVE_MIN_SCORE:
            regime = MarketRegime.AGGRESSIVE
        elif score >= NORMAL_MIN_SCORE:
            regime = MarketRegime.NORMAL
        elif score >= SELECTIVE_MIN_SCORE:
            regime = MarketRegime.SELECTIVE
        else:
            regime = MarketRegime.AVOID

        if available_weight < MIN_COVERAGE_FOR_UNCAPPED_REGIME and regime in (MarketRegime.AGGRESSIVE, MarketRegime.NORMAL):
            notes.append(
                f"Only {available_weight}/{TOTAL_WEIGHT} weighted points of data were available this run "
                f"({factors_available}/{len(FACTOR_WEIGHTS)} factors) -- capped at Selective rather than trusting "
                f"a {regime.value} call built on that little coverage."
            )
            regime = MarketRegime.SELECTIVE

    detail["available_weight"] = available_weight
    detail["passed_weight"] = passed_weight

    return RegimeResult(
        regime=regime,
        score=score,
        checks=checks,
        detail=detail,
        factors_available=factors_available,
        factors_total=len(FACTOR_WEIGHTS),
        notes=notes,
        nifty_closes=nifty_closes,
    )