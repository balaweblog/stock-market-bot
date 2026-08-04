"""
swing_trade_regime.py

Market-regime (broad-index trend) filter, gating individual buy signals
behind the state of the overall market -- see the review this module
implements: "buying bullish setups during a broad market downtrend has a
materially worse hit rate; professionals gate individual signals behind
an index-level trend filter." Nothing in the rest of the pipeline
previously checked whether Nifty/broader breadth was itself in an uptrend
before green-lighting an individual stock's breakout.

Independent of the LLM, same as swing_trade_risk.py: this is computed
directly from real index price history via stockpredictor.fetch_data, not
from anything the model claims about "market conditions".
"""

import os
import pandas as pd

from utils.logger import log
from services.stock_fetcher import fetch_stock_data

# ^NSEI (Nifty 50) is the default regime index for an India-focused swing
# strategy. Override via env var if the universe being screened is
# US-listed or otherwise not NSE-anchored.
MARKET_REGIME_INDEX = os.getenv("MARKET_REGIME_INDEX", "^NSEI")

# Off by default is the WRONG default for a risk control the review
# specifically flags as missing -- so this defaults ON. Set to "false" to
# restore the old behavior of never checking market breadth at all.
REQUIRE_MARKET_REGIME_FILTER = os.getenv("REQUIRE_MARKET_REGIME_FILTER", "true").lower() == "true"

# How the regime is classified:
#   bullish  -- index above BOTH its 20-week and 50-week SMA.
#   caution  -- above one but not both (mixed breadth).
#   bearish  -- below both.
# Only "bullish" passes the gate by default; "caution" can optionally be
# allowed through via MARKET_REGIME_ALLOW_CAUTION, since a strict
# "bearish only blocks" is arguably too permissive but a strict "bullish
# only" can also produce long dry spells -- this is left tunable rather
# than hardcoded either way.
MARKET_REGIME_ALLOW_CAUTION = os.getenv("MARKET_REGIME_ALLOW_CAUTION", "false").lower() == "true"


def _weekly_index_trend(index_ticker, weeks=55):
    """
    Returns a dict with latest_close, sma20w, sma50w for the index, or
    None if history is unavailable/insufficient. Mirrors the weekly
    resampling approach already used for individual-stock technicals in
    swing_trade_advisor._fetch_weekly_technicals, for consistency.
    """
    try:
        df = fetch_stock_data(index_ticker)
        if df is None or len(df) < 30 or "close" not in df.columns:
            return None
        weekly = df["close"].resample("W").last().dropna()
        if len(weekly) < 21:
            return None
        sma20 = weekly.rolling(window=20).mean()
        latest_close = float(weekly.iloc[-1])
        latest_sma20 = float(sma20.iloc[-1])
        prev_sma20 = float(sma20.iloc[-2]) if len(sma20) >= 2 and pd.notna(sma20.iloc[-2]) else None
        pct_above_sma20 = round(((latest_close - latest_sma20) / latest_sma20) * 100, 2)
        sma20_slope_rising = (latest_sma20 > prev_sma20) if prev_sma20 is not None else True
        in_uptrend = (latest_close > latest_sma20) and sma20_slope_rising
        return {
            "index_ticker": index_ticker,
            "latest_close": round(latest_close, 2),
            "sma20w": round(latest_sma20, 2),
            "pct_above_sma20": pct_above_sma20,
            "sma20_slope_rising": sma20_slope_rising,
            "in_uptrend": in_uptrend,
        }
    except Exception as e:
        log.warning(f"Could not compute market-regime trend for '{index_ticker}': {e}")
        return None


def check_market_regime(index_ticker=None):
    """
    Returns (passes_gate: bool, detail: dict).

    detail always includes "classification" ("bullish"/"caution"/
    "bearish"/"unknown") and "index_ticker"; when data was available it
    also includes latest_close/sma20w/sma50w for display in the report.

    "unknown" (index data unavailable) does NOT block a run by default --
    a data-provider hiccup on the index feed shouldn't silently stop every
    individual stock signal that week. It's disclosed as unverified
    instead. Set MARKET_REGIME_FAIL_CLOSED=true to block on "unknown" too
    if you'd rather fail safe than fail open.
    """
    index_ticker = index_ticker or MARKET_REGIME_INDEX
    trend = _weekly_index_trend(index_ticker)

    if trend is None:
        fail_closed = os.getenv("MARKET_REGIME_FAIL_CLOSED", "false").lower() == "true"
        detail = {"classification": "unknown", "index_ticker": index_ticker}
        return (not fail_closed), detail

    above20 = trend["latest_close"] > trend["sma20w"]
    above50 = trend["latest_close"] > trend["sma50w"]

    if above20 and above50:
        classification = "bullish"
    elif above20 or above50:
        classification = "caution"
    else:
        classification = "bearish"

    detail = {"classification": classification, "index_ticker": index_ticker, **trend}

    if classification == "bullish":
        passes = True
    elif classification == "caution":
        passes = MARKET_REGIME_ALLOW_CAUTION
    else:
        passes = False

    return passes, detail


def regime_note_html(detail):
    """Small disclosure line for the email/no-qualifying-trade output
    explaining the regime gate's decision this run, using the same plain
    inline-styled fragment convention as the rest of the report."""
    import html as _html
    sans = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
    cls = detail.get("classification", "unknown")
    idx = _html.escape(detail.get("index_ticker", "?"))
    if cls == "unknown":
        text = f"Market-regime check: could not fetch trend data for {idx} this run -- proceeding without the regime gate (fail-open)."
        color = "#8A8F9C"
    elif cls == "bullish":
        text = (
            f"Market-regime check: {idx} is above both its 20-week ({detail.get('sma20w')}) "
            f"and 50-week ({detail.get('sma50w')}) SMA -- broad market regime is bullish, gate passed."
        )
        color = "#2F5233"
    elif cls == "caution":
        text = (
            f"Market-regime check: {idx} is above only one of its 20-week ({detail.get('sma20w')}) / "
            f"50-week ({detail.get('sma50w')}) SMA -- mixed breadth."
        )
        color = "#A6812F"
    else:
        text = (
            f"Market-regime check: {idx} is below both its 20-week ({detail.get('sma20w')}) and "
            f"50-week ({detail.get('sma50w')}) SMA -- broad market regime is bearish."
        )
        color = "#8B2E2E"
    return f'<div style="font-family:{sans};font-size:11px;color:{color};margin-top:6px;">{text}</div>'