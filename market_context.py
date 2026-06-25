import yfinance as yf
import pandas as pd
from datetime import timedelta


def fetch_index_context(index_symbol="^NSEI", period="180d"):
    df = yf.download(index_symbol, period=period, interval="1d", auto_adjust=True, progress=False)
    if df.empty:
        return {
            "index_symbol": index_symbol,
            "trend": "unknown",
            "return_20d": 0.0,
            "return_50d": 0.0,
        }

    df.reset_index(inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        cols = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    else:
        cols = list(df.columns)

    normalized = []
    for col in cols:
        name = str(col).lower().strip()
        name = name.replace(" ", "").replace("-", "").replace("_", "")
        normalized.append(name)

    df.columns = normalized
    if "close" not in df.columns:
        source_close = None
        for candidate in df.columns:
            if "close" in candidate:
                source_close = candidate
                break
        if source_close is not None:
            df["close"] = df[source_close]

    if "close" not in df.columns:
        print(
            f"Index data for {index_symbol} missing 'close' after normalization. "
            f"Available columns: {', '.join(df.columns)}"
        )
        return {
            "index_symbol": index_symbol,
            "trend": "unknown",
            "return_20d": 0.0,
            "return_50d": 0.0,
        }

    df["return_20d"] = df["close"].pct_change(20)
    df["return_50d"] = df["close"].pct_change(50)
    trend = "bullish" if df["return_20d"].iloc[-1] > 0 and df["return_50d"].iloc[-1] > 0 else "bearish"

    return {
        "index_symbol": index_symbol,
        "trend": trend,
        "return_20d": round(float(df["return_20d"].iloc[-1] or 0.0), 4),
        "return_50d": round(float(df["return_50d"].iloc[-1] or 0.0), 4),
    }


def build_market_context(symbol):
    ticker = yf.Ticker(symbol)
    info = ticker.info
    sector = info.get("sector")
    industry = info.get("industry")
    context = fetch_index_context()
    context["sector"] = sector or "unknown"
    context["industry"] = industry or "unknown"
    return context
