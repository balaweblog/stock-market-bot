import pandas as pd
import yfinance as yf

try:
    # Reuse the same resilient session (custom User-Agent/headers) that
    # market_context.py already needed to add to stop Yahoo from blocking
    # requests from this box. Without it, this module was the only yfinance
    # caller in the codebase still using a bare, unauthenticated session.
    from market_context import get_resilient_session
except ImportError:
    def get_resilient_session():
        return None


def _empty_result():
    return {
        "revenue_growth": None,
        "net_income": None,
        "operating_cash_flow": None,
        "current_ratio": None,
        "quick_ratio": None,
        "debt_to_equity": None,
        "free_cash_flow": None,
    }


def fetch_advanced_fundamentals(symbol):
    # PREVIOUS BUG: none of the yfinance calls below were wrapped in a
    # try/except, unlike every other data-fetching module in this codebase
    # (market_context.py, commodity_tracker.py all defensively catch
    # exceptions and fall back to safe defaults). A single bad/delisted
    # ticker, a network hiccup, or a Yahoo rate-limit response would raise
    # and crash the whole scoring pipeline instead of just zero-scoring
    # that one symbol.
    try:
        session = get_resilient_session()
        ticker = yf.Ticker(symbol, session=session) if session else yf.Ticker(symbol)
        info = ticker.info if hasattr(ticker, "info") else {}
        if not isinstance(info, dict):
            info = {}
        financials = ticker.financials if hasattr(ticker, "financials") else None
        cashflow = ticker.cashflow if hasattr(ticker, "cashflow") else None
    except Exception:
        return _empty_result()

    revenue_growth = None
    net_income = None
    operating_cash_flow = None

    try:
        if financials is not None and "Total Revenue" in financials.index:
            revenue = financials.loc["Total Revenue"]
            if len(revenue) >= 2:
                latest = revenue.iloc[0]
                prior = revenue.iloc[1]
                # PREVIOUS BUG: `if prior and prior != 0` treats NaN as valid
                # since bool(nan) is True and nan != 0 is also True in
                # Python. A NaN prior-year revenue (missing data, common for
                # newer listings) silently produced revenue_growth = NaN,
                # which then poisoned score_advanced_fundamentals in an
                # order-dependent way (max(0, nan) can return nan instead of
                # 0 depending on argument order). Use pd.notna() so missing
                # data is treated as "unknown" (None), not silently NaN.
                if pd.notna(latest) and pd.notna(prior) and prior != 0:
                    revenue_growth = float((latest - prior) / prior)

        if financials is not None and "Net Income" in financials.index:
            ni = financials.loc["Net Income"].iloc[0]
            if pd.notna(ni):
                net_income = float(ni)

        if cashflow is not None and "Operating Cash Flow" in cashflow.index:
            ocf = cashflow.loc["Operating Cash Flow"].iloc[0]
            if pd.notna(ocf):
                operating_cash_flow = float(ocf)
    except Exception:
        # Financials/cashflow parsing failed (unexpected shape, etc.) --
        # fall through with whatever ratio-based info fields we still have.
        pass

    return {
        "revenue_growth": revenue_growth,
        "net_income": net_income,
        "operating_cash_flow": operating_cash_flow,
        "current_ratio": info.get("currentRatio"),
        "quick_ratio": info.get("quickRatio"),
        "debt_to_equity": info.get("debtToEquity"),
        "free_cash_flow": info.get("freeCashflow"),
    }


def score_advanced_fundamentals(data):
    score = 0

    revenue_growth = data.get("revenue_growth")
    if revenue_growth is not None and pd.notna(revenue_growth):
        score += min(20, max(0, revenue_growth * 100))
    if data.get("net_income") is not None and data["net_income"] > 0:
        score += 20
    if data.get("operating_cash_flow") is not None and data["operating_cash_flow"] > 0:
        score += 15
    if data.get("current_ratio") is not None and data["current_ratio"] > 1.2:
        score += 10
    if data.get("quick_ratio") is not None and data["quick_ratio"] > 0.8:
        score += 10
    if data.get("debt_to_equity") is not None and data["debt_to_equity"] < 100:
        score += 15
    if data.get("free_cash_flow") is not None and data["free_cash_flow"] > 0:
        score += 10

    return min(100, score)