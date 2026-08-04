"""
swing_trade_backtest.py

Backtesting harness: runs the strategy's TECHNICAL entry rules (uptrend,
weekly RSI<70 and rising, bullish MACD crossover) plus the ATR-based
risk-managed exit (see swing_trade_risk.py) against real historical price
data, and reports the numbers a pro would actually want before trusting a
filter combination: win rate, average realized R:R, max drawdown, and
average time-in-trade.

HONEST LIMITATION, stated up front rather than glossed over: this
backtest covers the TECHNICAL leg only, not the FUNDAMENTALS leg (>=20%
YoY revenue/profit growth, low debt, high ROE). yfinance only exposes a
company's most recent ~4-8 quarters of financials -- it does not give you
point-in-time historical fundamentals as they would have been known on
an arbitrary past date. Backtesting the fundamentals filter properly
would require a real point-in-time fundamentals database (e.g. a paid
data vendor) so that the growth number used at each historical point is
the one that was ACTUALLY knowable then, not a number computed with the
benefit of hindsight. Running the growth filter with today's trailing
financials against price action from three years ago would silently
leak look-ahead bias into the result -- which is worse than not
backtesting the fundamentals leg at all, because it would produce a
false sense of confidence. If you have (or can get) a point-in-time
fundamentals source, extend `_passes_fundamentals_asof()` below to use it
and remove this caveat.

USAGE (from a shell with network + yfinance available -- this repo's
sandboxed dev environment does not have outbound network, so this can't
be run standalone here; run it wherever the rest of the pipeline runs):

    python swing_trade_backtest.py --tickers RELIANCE.NS,TCS.NS,LTIM.NS \\
        --start 2019-01-01 --end 2025-01-01 --horizon-days 150

    # Or from a file, one ticker per line:
    python swing_trade_backtest.py --tickers-file universe.txt \\
        --start 2019-01-01 --end 2025-01-01

Every trade the backtest takes is logged to backtest_trades.csv (one row
per trade) so results are auditable, not just a summary number pulled
out of a black box.
"""

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

# Reuse the same thresholds/constants the live pipeline uses, so a
# backtest run actually tests the rules currently in effect rather than a
# hand-copied second version that can drift out of sync.
from controllers import swing_controller as sta
from . import swing_trade_risk as risk

TRADES_LOG = "backtest_trades.csv"
TRADE_FIELDNAMES = [
    "ticker", "signal_date", "entry_date", "entry_price",
    "stop_loss_pct", "target1_pct", "exit_date", "exit_price",
    "exit_reason", "realized_pct", "days_in_trade",
]


def _fetch_history(ticker, start, end):
    if yf is None:
        raise RuntimeError("yfinance is required for backtesting -- pip install yfinance")
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df is None or df.empty:
        return None
    df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
    return df


def _weekly_technical_signals(daily_df):
    """
    Walks the full daily history and returns a list of signal dates
    (weekly bar close dates) where, AS OF THAT WEEK ONLY (no future data
    used), the technical entry rules fire:
      - price above 20-week AND 50-week SMA (if REQUIRE_UPTREND_FILTER)
      - weekly RSI(14) below MAX_RSI_OVERBOUGHT and rising vs 2 weeks ago
      - MACD line above its signal line (bullish crossover in effect)
    This mirrors swing_trade_advisor._verify_technicals' logic exactly,
    computed with only data up to and including that week -- no look-ahead.
    """
    weekly = daily_df.resample("W").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    if len(weekly) < 55:
        return []

    close = weekly["close"]
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()

    signals = []
    for i in range(55, len(weekly)):
        price = close.iloc[i]
        if sta.REQUIRE_UPTREND_FILTER:
            if not (price > sma20.iloc[i] and price > sma50.iloc[i]):
                continue
        rsi_now, rsi_prev = rsi.iloc[i], rsi.iloc[i - 2] if i >= 2 else None
        if pd.isna(rsi_now) or rsi_now >= sta.MAX_RSI_OVERBOUGHT:
            continue
        if rsi_prev is not None and not pd.isna(rsi_prev) and rsi_now < rsi_prev:
            continue
        if macd_line.iloc[i] <= signal_line.iloc[i]:
            continue
        signals.append(weekly.index[i])
    return signals


def _atr_stop_target(daily_df, as_of_date, atr_multiple, min_rr):
    window = daily_df[daily_df.index <= as_of_date]
    weekly = window.resample("W").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    if len(weekly) < 15:
        return None, None, None
    prev_close = weekly["close"].shift(1)
    tr = pd.concat([
        weekly["high"] - weekly["low"],
        (weekly["high"] - prev_close).abs(),
        (weekly["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    price = weekly["close"].iloc[-1]
    if pd.isna(atr) or price == 0:
        return None, None, None
    stop_pct = (atr_multiple * atr / price) * 100
    target_pct = stop_pct * min_rr
    return round(stop_pct, 2), round(target_pct, 2), round(float(price), 2)


def _simulate_trade(daily_df, signal_date, horizon_days, atr_multiple, min_rr):
    """
    Enters at the next available daily close after `signal_date`, walks
    forward bar by bar, and exits on whichever comes first: stop hit,
    target hit, or horizon expiry (exit at close on the horizon date).
    On a day where both stop and target would have been touched, the stop
    is assumed to have hit first (conservative -- a daily bar can't tell
    us the real intraday order).
    """
    future = daily_df[daily_df.index > signal_date]
    if future.empty:
        return None
    entry_date = future.index[0]
    entry_price = float(future["close"].iloc[0])

    stop_pct, target_pct, _ = _atr_stop_target(daily_df, signal_date, atr_multiple, min_rr)
    if stop_pct is None:
        return None
    stop_price = entry_price * (1 - stop_pct / 100.0)
    target_price = entry_price * (1 + target_pct / 100.0)

    horizon_end = entry_date + pd.Timedelta(days=horizon_days)
    trade_window = future[future.index <= horizon_end]
    if trade_window.empty:
        return None

    for ts, bar in trade_window.iterrows():
        if bar["low"] <= stop_price:
            return {
                "entry_date": entry_date, "entry_price": entry_price,
                "stop_loss_pct": stop_pct, "target1_pct": target_pct,
                "exit_date": ts, "exit_price": stop_price, "exit_reason": "stop",
                "realized_pct": -stop_pct,
                "days_in_trade": (ts - entry_date).days,
            }
        if bar["high"] >= target_price:
            return {
                "entry_date": entry_date, "entry_price": entry_price,
                "stop_loss_pct": stop_pct, "target1_pct": target_pct,
                "exit_date": ts, "exit_price": target_price, "exit_reason": "target",
                "realized_pct": target_pct,
                "days_in_trade": (ts - entry_date).days,
            }

    last = trade_window.iloc[-1]
    last_ts = trade_window.index[-1]
    realized = ((float(last["close"]) - entry_price) / entry_price) * 100
    return {
        "entry_date": entry_date, "entry_price": entry_price,
        "stop_loss_pct": stop_pct, "target1_pct": target_pct,
        "exit_date": last_ts, "exit_price": float(last["close"]), "exit_reason": "horizon_expiry",
        "realized_pct": round(realized, 2),
        "days_in_trade": (last_ts - entry_date).days,
    }


def _max_drawdown(equity_curve):
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for v in equity_curve:
        peak = max(peak, v)
        dd = (peak - v) / peak if peak else 0.0
        max_dd = max(max_dd, dd)
    return round(max_dd * 100, 2)


def backtest_ticker(ticker, start, end, horizon_days, atr_multiple, min_rr, min_gap_days=20):
    daily_df = _fetch_history(ticker, start, end)
    if daily_df is None or "close" not in daily_df.columns:
        return []
    signals = _weekly_technical_signals(daily_df)

    trades = []
    last_entry_end = None
    for sig_date in signals:
        # Don't stack overlapping trades on the same ticker from
        # back-to-back weekly signals -- wait for the prior simulated
        # trade to conclude (or min_gap_days, whichever is sooner to
        # evaluate) before taking a fresh signal.
        if last_entry_end is not None and sig_date <= last_entry_end:
            continue
        trade = _simulate_trade(daily_df, sig_date, horizon_days, atr_multiple, min_rr)
        if trade is None:
            continue
        trade["ticker"] = ticker
        trade["signal_date"] = sig_date
        trades.append(trade)
        last_entry_end = trade["exit_date"]
    return trades


def run_backtest(tickers, start, end, horizon_days=150, atr_multiple=None, min_rr=None):
    atr_multiple = atr_multiple if atr_multiple is not None else risk.ATR_STOP_MULTIPLE
    min_rr = min_rr if min_rr is not None else sta.MIN_RISK_REWARD

    all_trades = []
    for t in tickers:
        try:
            trades = backtest_ticker(t, start, end, horizon_days, atr_multiple, min_rr)
            all_trades.extend(trades)
            print(f"{t}: {len(trades)} trade(s)")
        except Exception as e:
            print(f"{t}: FAILED -- {e}", file=sys.stderr)

    if not all_trades:
        print("No trades generated -- nothing to summarize. Check ticker symbols and date range.")
        return {}

    with Path(TRADES_LOG).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRADE_FIELDNAMES)
        writer.writeheader()
        for t in all_trades:
            writer.writerow({k: t.get(k) for k in TRADE_FIELDNAMES})

    wins = [t for t in all_trades if t["exit_reason"] == "target"]
    losses = [t for t in all_trades if t["exit_reason"] == "stop"]
    flat = [t for t in all_trades if t["exit_reason"] == "horizon_expiry"]

    realized = [t["realized_pct"] for t in all_trades]
    equity_curve = []
    cum = 100.0
    for r in realized:
        cum *= (1 + r / 100.0)
        equity_curve.append(cum)

    summary = {
        "total_trades": len(all_trades),
        "wins": len(wins),
        "losses": len(losses),
        "expired_flat": len(flat),
        "win_rate_pct": round(100 * len(wins) / len(all_trades), 1),
        "avg_realized_pct": round(sum(realized) / len(realized), 2),
        "avg_days_in_trade": round(sum(t["days_in_trade"] for t in all_trades) / len(all_trades), 1),
        "max_drawdown_pct": _max_drawdown(equity_curve),
        "cumulative_return_pct": round(cum - 100.0, 1),
        "trades_logged_to": str(Path(TRADES_LOG).resolve()),
    }
    return summary


def _print_summary(summary):
    if not summary:
        return
    print("\n=== Backtest Summary (technical leg only -- see module docstring) ===")
    for k, v in summary.items():
        print(f"{k}: {v}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", help="Comma-separated ticker list, e.g. RELIANCE.NS,TCS.NS")
    parser.add_argument("--tickers-file", help="Path to a file with one ticker per line")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--horizon-days", type=int, default=150, help="Max days to hold before forced exit (default ~5 months)")
    parser.add_argument("--atr-multiple", type=float, default=None, help="Override ATR_STOP_MULTIPLE for this run")
    parser.add_argument("--min-rr", type=float, default=None, help="Override MIN_RISK_REWARD for this run")
    args = parser.parse_args()

    tickers = []
    if args.tickers:
        tickers.extend(t.strip() for t in args.tickers.split(",") if t.strip())
    if args.tickers_file:
        tickers.extend(
            line.strip() for line in Path(args.tickers_file).read_text().splitlines() if line.strip()
        )
    if not tickers:
        parser.error("Provide --tickers and/or --tickers-file")

    summary = run_backtest(
        tickers, args.start, args.end,
        horizon_days=args.horizon_days,
        atr_multiple=args.atr_multiple,
        min_rr=args.min_rr,
    )
    _print_summary(summary)


if __name__ == "__main__":
    main()
