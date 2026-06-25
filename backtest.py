import math
import pandas as pd
from config import STOCKS
from stock_fetcher import fetch_stock_data
from main import calculate_indicators, calculate_score, get_signal


def run_backtest(symbol, initial_cash=100000, verbose=False):
    df = fetch_stock_data(symbol)
    df = calculate_indicators(df)

    if df.empty or len(df) < 50:
        raise ValueError(f"Not enough historical data to backtest {symbol}")

    signals = []
    for i in range(len(df)):
        score, _ = calculate_score(df.iloc[: i + 1])
        signals.append(get_signal(score))

    cash = initial_cash
    shares = 0
    entry_price = 0
    trades = []
    portfolio_values = []

    for i in range(1, len(df)):
        today = df.iloc[i]
        yesterday_signal = signals[i - 1]
        current_signal = signals[i]
        price = today["open"] if not pd.isna(today["open"]) else today["close"]

        if price <= 0 or pd.isna(price):
            portfolio_values.append(cash + shares * today["close"])
            continue

        if shares == 0 and yesterday_signal.startswith("GREEN"):
            shares = math.floor(cash / price)
            entry_price = price
            cash -= shares * price

        if shares > 0 and current_signal.startswith("RED"):
            exit_price = price
            cash += shares * exit_price
            trades.append({
                "entry_price": entry_price,
                "exit_price": exit_price,
                "shares": shares,
                "return_pct": (exit_price - entry_price) / entry_price if entry_price else 0,
            })
            shares = 0
            entry_price = 0

        portfolio_values.append(cash + shares * today["close"])

    final_value = cash + shares * df.iloc[-1]["close"]
    returns = pd.Series(portfolio_values).pct_change().fillna(0)
    cumulative_return = final_value / initial_cash - 1

    if len(returns) > 0:
        annualized_return = (1 + cumulative_return) ** (252 / len(returns)) - 1
        annualized_vol = returns.std() * (252**0.5)
        sharpe = annualized_return / annualized_vol if annualized_vol > 0 else float('nan')
    else:
        annualized_return = 0
        annualized_vol = 0
        sharpe = float('nan')

    peak = pd.Series(portfolio_values).cummax()
    drawdown = (pd.Series(portfolio_values) - peak) / peak
    max_drawdown = drawdown.min()

    wins = [t for t in trades if t["return_pct"] > 0]
    win_rate = len(wins) / len(trades) if trades else 0

    result = {
        "symbol": symbol,
        "start_date": df.iloc[0]["date"].strftime("%Y-%m-%d") if "date" in df.columns else None,
        "end_date": df.iloc[-1]["date"].strftime("%Y-%m-%d") if "date" in df.columns else None,
        "initial_cash": initial_cash,
        "final_value": round(final_value, 2),
        "total_return": round(cumulative_return * 100, 2),
        "annual_return": round(annualized_return * 100, 2),
        "annual_volatility": round(annualized_vol * 100, 2),
        "sharpe_ratio": round(sharpe, 2) if not math.isnan(sharpe) else None,
        "max_drawdown": round(max_drawdown * 100, 2),
        "trades": len(trades),
        "win_rate": round(win_rate * 100, 2),
    }

    if verbose:
        print(f"Backtest {symbol}: {result}")

    return result


def run_all_backtests():
    results = []
    for name, ticker in STOCKS.items():
        try:
            print(f"Backtesting {name} ({ticker})...")
            results.append(run_backtest(ticker, verbose=False))
        except Exception as exc:
            print(f"Skipping {ticker}: {exc}")
    return pd.DataFrame(results)


if __name__ == "__main__":
    summary = run_all_backtests()
    print(summary.to_string(index=False))
