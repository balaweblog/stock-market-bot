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
        r20 = df["return_20d"].iloc[-1]
        r50 = df["return_50d"].iloc[-1]

        # Require a meaningful move (>1%) in both windows before calling it
        # bullish/bearish -- previously any positive-vs-positive combo (even
        # +0.01%) counted as "bullish", so nearly every symbol landed there.
        # NaN returns (not enough history) now report "unknown" instead of
        # silently falling into "bearish".
        NEUTRAL_BAND = 0.01
        if pd.isna(r20) or pd.isna(r50):
            trend = "unknown"
        elif r20 > NEUTRAL_BAND and r50 > NEUTRAL_BAND:
            trend = "bullish"
        elif r20 < -NEUTRAL_BAND and r50 < -NEUTRAL_BAND:
            trend = "bearish"
        else:
            trend = "neutral"

        return {
            "index_symbol": index_symbol,
            "trend": trend,
            "return_20d": round(float(r20), 4) if pd.notna(r20) else 0.0,
            "return_50d": round(float(r50), 4) if pd.notna(r50) else 0.0,
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

    # PREVIOUS BUG: this called fetch_index_context(benchmark_symbol) only,
    # so every stock in a market inherited the *index's* trend (Nifty for
    # all India stocks, S&P for all US stocks) instead of its own. Since
    # both indices have been broadly trending up, "Trend" showed "bullish"
    # for almost every stock regardless of how that stock itself was doing.
    # Now we compute the trend from the stock's own price history, and keep
    # the benchmark's trend as separate, clearly-labeled context.
    context = fetch_index_context(index_symbol=symbol)
    benchmark_context = fetch_index_context(index_symbol=benchmark_symbol)

    context["sector"] = sector or "unknown"
    context["industry"] = industry or "unknown"
    context["benchmark_index"] = benchmark_symbol
    context["benchmark_trend"] = benchmark_context.get("trend")
    context["benchmark_return_20d"] = benchmark_context.get("return_20d")
    context["benchmark_return_50d"] = benchmark_context.get("return_50d")
    return context