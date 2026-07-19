import yfinance as yf


def fetch_advanced_fundamentals(symbol):
    ticker = yf.Ticker(symbol)
    info = ticker.info if hasattr(ticker, "info") else {}
    financials = ticker.financials if hasattr(ticker, "financials") else None
    cashflow = ticker.cashflow if hasattr(ticker, "cashflow") else None

    revenue_growth = None
    net_income = None
    operating_cash_flow = None

    if financials is not None and "Total Revenue" in financials.index:
        revenue = financials.loc["Total Revenue"]
        if len(revenue) >= 2:
            latest = revenue.iloc[0]
            prior = revenue.iloc[1]
            if prior and prior != 0:
                revenue_growth = (latest - prior) / prior

    if financials is not None and "Net Income" in financials.index:
        net_income = financials.loc["Net Income"].iloc[0]

    if cashflow is not None and "Operating Cash Flow" in cashflow.index:
        operating_cash_flow = cashflow.loc["Operating Cash Flow"].iloc[0]

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

    if data.get("revenue_growth") is not None:
        score += min(20, max(0, data["revenue_growth"] * 100))
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