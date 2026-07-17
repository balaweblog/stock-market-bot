import yfinance as yf
import pandas as pd
from datetime import timedelta
import requests

def get_resilient_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive"
    })
    return session


def fetch_index_context(index_symbol="^NSEI", period="180d"):
    try:
        session = get_resilient_session()
        df = yf.download(index_symbol, period=period, interval="1d", auto_adjust=True, progress=False, session=session)
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
    except Exception:
        return {
            "index_symbol": index_symbol,
            "trend": "unknown",
            "return_20d": 0.0,
            "return_50d": 0.0,
        }


def classify_market(ticker):
    """
    Classifies a ticker as India or US based on its exchange suffix.
    Indian tickers pulled via yfinance carry a '.NS' (NSE) or '.BO' (BSE)
    suffix; US tickers (e.g. AAPL, GOOG, AMZN, QQQ) have no suffix.
    Mirrors main.py's classify_market so this module stays self-contained.
    """
    upper_ticker = str(ticker or "").upper()
    if upper_ticker.endswith(".NS") or upper_ticker.endswith(".BO"):
        return "India"
    return "US"


# Benchmark index used for the "market trend" comparison, keyed by market.
# Previously fetch_index_context() was always called with no argument,
# which silently defaulted to "^NSEI" (Nifty) for every stock -- including
# US stocks. Since the Nifty has been broadly trending up, this made the
# "Trend" field in the email show "bullish" for nearly every stock,
# regardless of that stock's actual market. Each market now gets its own
# benchmark index.
BENCHMARK_INDEX_BY_MARKET = {
    "India": "^NSEI",   # Nifty 50
    "US": "^GSPC",       # S&P 500
}


def build_market_context(symbol):
    try:
        session = get_resilient_session()
        ticker = yf.Ticker(symbol, session=session)
        info = ticker.info
        if info is None:
            info = {}
        sector = info.get("sector", "unknown") if isinstance(info, dict) else "unknown"
        industry = info.get("industry", "unknown") if isinstance(info, dict) else "unknown"
    except Exception:
        sector = "unknown"
        industry = "unknown"

    market = classify_market(symbol)
    benchmark_symbol = BENCHMARK_INDEX_BY_MARKET.get(market, "^NSEI")

    context = fetch_index_context(index_symbol=benchmark_symbol)
    context["sector"] = sector or "unknown"
    context["industry"] = industry or "unknown"
    return context