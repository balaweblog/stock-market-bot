"""
swing_trade_risk.py

Position sizing, volatility-adjusted (ATR-based) stops/targets, and
sector-concentration control for swing_trade_advisor.py.

WHY THIS EXISTS (see the review this repo was built from):
- A flat % stop-loss treats a sleepy FMCG stock the same as a high-beta
  small-cap. Two stocks can carry the model's own "1:2 risk:reward" and
  still have wildly different actual dollar/rupee risk once you account
  for how much each one actually moves week to week.
- "Is this a good trade" is the wrong question in isolation -- a pro sizes
  every position so a stop-out costs a fixed, small slice of the
  portfolio, and caps how much of that portfolio can be riding on one
  sector/factor bet at the same time.

This module is deliberately independent of the LLM: every number here is
computed from real price history (via yfinance, same data source
stockpredictor.fetch_data already relies on), never from the model's
self-reported figures. It is ADDITIVE to the existing flat-% fields the
model already returns -- it doesn't delete them, it computes an
independent risk-managed alternative and the email shows both side by
side so a flat-% claim that's out of line with the stock's own volatility
is visible, not silently overwritten.
"""

import os
import pandas as pd

import stockpredictor  # reuses fetch_data + log, same as swing_trade_advisor.py

# -----------------------------
# Configurable knobs
# -----------------------------
def _env_float(name, default):
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw.strip())
    except ValueError:
        stockpredictor.log.warning(f"WARNING: env var {name}='{raw}' is not a valid number -- using default {default}.")
        return default


def _env_int(name, default):
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.strip())
    except ValueError:
        stockpredictor.log.warning(f"WARNING: env var {name}='{raw}' is not a valid integer -- using default {default}.")
        return default


# ATR multiple used for the stop distance. 1.5x weekly ATR is a common
# swing-trade default -- tight enough to cap risk, wide enough that normal
# weekly noise doesn't stop you out of a trade that's still working.
ATR_STOP_MULTIPLE = _env_float("ATR_STOP_MULTIPLE", 1.5)
# Target is expressed as a multiple of the ATR-based stop distance, tied
# to the same minimum risk:reward the rest of the strategy already
# enforces (MIN_RISK_REWARD lives in swing_trade_advisor.py; passed in by
# the caller rather than re-read here so this module has no import-order
# dependency on it).
DEFAULT_MIN_RISK_REWARD = _env_float("MIN_RISK_REWARD", 2.0)

# Fraction of total trading capital a pro risks on any single position if
# it hits its stop. 0.5-1% is the standard range; 1% is used as the
# default here. This is a PORTFOLIO-level number, so it only produces a
# meaningful share count if PORTFOLIO_VALUE is also set -- otherwise
# position sizing degrades gracefully to "risk-per-unit-capital %" only.
RISK_PCT_PER_TRADE = _env_float("RISK_PCT_PER_TRADE", 1.0)
PORTFOLIO_VALUE = _env_float("PORTFOLIO_VALUE", 0.0)  # 0 = not configured

# PORTFOLIO_VALUE is a single flat number with no currency attached to it,
# but this project trades both Indian (INR) and US (USD) tickers in the
# same run (see swing_trade_advisor._fetch_current_price's Rs./$ symbol
# selection via stockpredictor.classify_market). Without checking this,
# shares_for_1pct_risk/position_value_for_1pct_risk below would silently
# divide an INR portfolio value by a USD stop distance (or vice versa) for
# any ticker whose market doesn't match whichever currency the operator had
# in mind when they set PORTFOLIO_VALUE -- e.g. treating a Rs.10,00,000
# portfolio as if it were $10,00,000, over/under-sizing the real position
# by roughly the INR/USD exchange rate. Defaults to INR (this project's
# primary market); set PORTFOLIO_CURRENCY=USD if PORTFOLIO_VALUE was
# entered in dollars instead.
PORTFOLIO_CURRENCY = (os.getenv("PORTFOLIO_CURRENCY", "INR") or "INR").strip().upper()

# Sector/factor concentration cap: at most this many qualifying picks from
# the same sector get emailed in a single run. Multiple names from the same
# sector are frequently the same underlying bet (e.g. three IT names all
# riding a rupee-depreciation trade) wearing different tickers.
MAX_PICKS_PER_SECTOR = _env_int("MAX_PICKS_PER_SECTOR", 1)


def _weekly_atr(ticker, period_weeks=14):
    """
    Computes weekly Average True Range from real price history via
    stockpredictor.fetch_data -- independent of anything the model claimed.
    Returns (atr, latest_close) or (None, None) on any failure/insufficient
    history. Uses a simplified True Range (weekly high-low range plus gap
    vs prior close) resampled from daily data, since that's what
    stockpredictor.fetch_data already returns.
    """
    ticker = (ticker or "").strip()
    if not ticker:
        return None, None
    try:
        df = stockpredictor.fetch_data(ticker)
        if df is None or len(df) < 30:
            return None, None
        required = {"high", "low", "close"}
        if not required.issubset(set(df.columns)):
            return None, None
        if not isinstance(df.index, pd.DatetimeIndex):
            date_col = next((c for c in df.columns if c.lower() == "date"), None)
            if date_col is None:
                return None, None
            df = df.set_index(pd.to_datetime(df[date_col]))

        weekly = df.resample("W").agg({"high": "max", "low": "min", "close": "last"}).dropna()
        if len(weekly) < period_weeks + 1:
            return None, None

        prev_close = weekly["close"].shift(1)
        tr = pd.concat([
            weekly["high"] - weekly["low"],
            (weekly["high"] - prev_close).abs(),
            (weekly["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)

        atr = tr.rolling(period_weeks).mean().iloc[-1]
        latest_close = weekly["close"].iloc[-1]
        if pd.isna(atr) or pd.isna(latest_close) or latest_close == 0:
            return None, None
        return round(float(atr), 2), round(float(latest_close), 2)
    except Exception as e:
        stockpredictor.log.warning(f"Could not compute weekly ATR for '{ticker}': {e}")
        return None, None


def compute_volatility_adjusted_plan(ticker, min_risk_reward=None):
    """
    Returns a dict describing an ATR-based stop/target/position-size plan
    for `ticker`, or None if ATR couldn't be computed. This is the
    volatility-aware counterpart to the model's flat stop_loss_pct /
    target1_pct -- callers should show both, not replace one with the
    other silently, so a flat-% claim that's out of step with the stock's
    real volatility is visible in the report.

    Fields:
      atr_weekly, latest_close: raw inputs, for transparency in the report.
      stop_loss_pct: ATR_STOP_MULTIPLE * ATR expressed as a % of price.
      target1_pct: stop_loss_pct * min_risk_reward.
      risk_reward_ratio: "1 : {min_risk_reward}" (by construction).
      shares_for_1pct_risk: how many shares 1% of PORTFOLIO_VALUE buys you
        room to hold given this stop distance -- None if PORTFOLIO_VALUE
        isn't configured.
      risk_amount_per_share: absolute currency risk per share to the stop.
    """
    if min_risk_reward is None:
        min_risk_reward = DEFAULT_MIN_RISK_REWARD

    atr, price = _weekly_atr(ticker)
    if atr is None or price is None:
        return None

    stop_distance = ATR_STOP_MULTIPLE * atr
    stop_loss_pct = round((stop_distance / price) * 100, 2)
    target1_pct = round(stop_loss_pct * min_risk_reward, 2)

    plan = {
        "atr_weekly": atr,
        "latest_close": price,
        "stop_loss_pct": stop_loss_pct,
        "target1_pct": target1_pct,
        "risk_reward_ratio": f"1 : {_fmt(min_risk_reward)}",
        "risk_amount_per_share": round(stop_distance, 2),
        "shares_for_1pct_risk": None,
        "position_value_for_1pct_risk": None,
        "position_size_note": None,
    }

    if PORTFOLIO_VALUE > 0 and stop_distance > 0:
        ticker_currency = _ticker_currency(ticker)
        if ticker_currency is not None and ticker_currency != PORTFOLIO_CURRENCY:
            # Don't compute a share count off a currency mismatch (e.g.
            # dividing an INR PORTFOLIO_VALUE by a USD stop distance) --
            # that produces a confidently wrong number, not a missing one.
            plan["position_size_note"] = (
                f"Skipped: PORTFOLIO_VALUE is configured as {PORTFOLIO_CURRENCY} but this "
                f"ticker trades in {ticker_currency} -- set PORTFOLIO_CURRENCY={ticker_currency} "
                "(or size this position separately) rather than mixing currencies."
            )
        else:
            risk_budget = PORTFOLIO_VALUE * (RISK_PCT_PER_TRADE / 100.0)
            shares = int(risk_budget // stop_distance)
            plan["shares_for_1pct_risk"] = max(shares, 0)
            plan["position_value_for_1pct_risk"] = round(shares * price, 2)

    return plan


def _ticker_currency(ticker):
    """
    Best-effort currency for `ticker`, using the same market classification
    swing_trade_advisor.py already relies on for its Rs./$ display symbol.
    Returns "INR", "USD", or None if it can't be determined (e.g.
    classify_market isn't available) -- None means "unknown", not "no
    mismatch", but callers treat it as "skip the check" since PORTFOLIO_VALUE
    is opt-in and off by default anyway.
    """
    try:
        market = stockpredictor.classify_market(ticker)
    except Exception as e:
        stockpredictor.log.warning(f"Could not classify market for '{ticker}' -- skipping currency check: {e}")
        return None
    return "INR" if str(market).strip().lower() == "india" else "USD"


def _fmt(x):
    return f"{x:g}"


def apply_sector_concentration_cap(stocks, max_per_sector=None):
    """
    Given a list of qualifying stock dicts (each with a "sector" field,
    carried through from Stage 1), returns (kept, dropped_for_concentration).

    Ordering matters: stocks are expected to already be sorted best-first
    (e.g. by confidence_score_adjusted) by the caller, since ties are
    broken by "whichever came first in the list" -- keep the strongest
    idea per sector, not an arbitrary one.

    dropped_for_concentration entries get a "_concentration_note" field so
    the report can disclose *why* an otherwise-qualifying stock didn't make
    the final cut (this is a "too much of the same bet", not a "failed
    verification" rejection -- kept separate from all_rejected for that
    reason; see swing_trade_advisor.py's call site).
    """
    if max_per_sector is None:
        max_per_sector = MAX_PICKS_PER_SECTOR
    if max_per_sector <= 0:
        max_per_sector = 1

    seen_per_sector = {}
    kept, dropped = [], []
    for s in stocks:
        sector = (s.get("sector") or "Unknown").strip() or "Unknown"
        count = seen_per_sector.get(sector, 0)
        if count < max_per_sector:
            kept.append(s)
            seen_per_sector[sector] = count + 1
        else:
            s = dict(s)
            s["_concentration_note"] = (
                f"Dropped: this run already has {max_per_sector} pick(s) from the "
                f"'{sector}' sector -- multiple same-sector picks are frequently the "
                "same underlying factor bet (e.g. several IT names all riding one "
                "rupee move) rather than genuinely independent ideas. Raise "
                "MAX_PICKS_PER_SECTOR to allow more."
            )
            dropped.append(s)
    return kept, dropped