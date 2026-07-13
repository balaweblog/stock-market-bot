from datetime import date, datetime

import yfinance as yf
import pandas as pd


def format_event_date(value):
    if value in (None, "", "Not available"):
        return "Not available"

    if isinstance(value, (list, tuple)):
        for item in value:
            formatted = format_event_date(item)
            if formatted != "Not available":
                return formatted
        return "Not available"

    if isinstance(value, dict):
        for key in ("date", "value", "raw"):
            if key in value:
                formatted = format_event_date(value[key])
                if formatted != "Not available":
                    return formatted
        return "Not available"

    if isinstance(value, (datetime, date)):
        return value.strftime("%d %b %Y")

    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value)).strftime("%d %b %Y")
        except (OverflowError, ValueError, OSError):
            return "Not available"

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return "Not available"
        try:
            return pd.Timestamp(text).strftime("%d %b %Y")
        except Exception:
            return text

    return str(value)


def build_upcoming_event_summary(ticker, info=None):
    if info is None:
        info = {}

    today = pd.Timestamp.today().normalize()
    max_date = today + pd.Timedelta(days=60)

    def pick(*keys):
        for key in keys:
            value = info.get(key)
            if value not in (None, "", [], {}):
                return value
        return None

    def parse(value):
        if value is None:
            return None

        if isinstance(value, (list, tuple)):
            dates = [parse(v) for v in value]
            dates = [d for d in dates if d is not None]
            return min(dates) if dates else None

        if isinstance(value, dict):
            for k in ("date", "value", "raw"):
                if k in value:
                    return parse(value[k])

        if isinstance(value, (datetime, date)):
            return pd.Timestamp(value).normalize()

        if isinstance(value, (int, float)):
            try:
                return pd.to_datetime(value, unit="s").normalize()
            except Exception:
                return None

        try:
            return pd.Timestamp(value).normalize()
        except Exception:
            return None

    earnings_date = None
    dividend_date = None

    try:
        cal = ticker.calendar

        if isinstance(cal, pd.DataFrame) and "Value" in cal.columns:

            if "Earnings Date" in cal.index:
                earnings_date = parse(cal.loc["Earnings Date", "Value"])

            if "Ex-Dividend Date" in cal.index:
                dividend_date = parse(cal.loc["Ex-Dividend Date", "Value"])

    except Exception:
        pass

    if earnings_date is None:
        earnings_date = parse(
            pick(
                "earningsDate",
                "earningsTimestamp",
                "nextEarningsDate",
            )
        )

    if dividend_date is None:
        dividend_date = parse(
            pick(
                "exDividendDate",
                "dividendDate",
            )
        )

    # Only keep events within next 60 days

    if earnings_date is None or earnings_date < today or earnings_date > max_date:
        earnings_date = None

    if dividend_date is None or dividend_date < today or dividend_date > max_date:
        dividend_date = None

    dividend_display = (
        dividend_date.strftime("%d %b %Y")
        if dividend_date is not None
        else "NA"
    )

    earnings_display = (
        earnings_date.strftime("%d %b %Y")
        if earnings_date is not None
        else "NA"
    )

    events = []

    if dividend_date is not None:
        events.append(("Dividend Record", dividend_date))

    if earnings_date is not None:
        events.append(("Results Announcement", earnings_date))

    if events:

        label, dt = min(events, key=lambda x: x[1])

        return {
            "has_event": True,
            "dividend_record_date": dividend_display,
            "results_announcement_date": earnings_display,
            "next_upcoming_event_label": label,
            "next_upcoming_event_date": dt.strftime("%d %b %Y"),
        }

    return {
        "has_event": False,
        "dividend_record_date": "NA",
        "results_announcement_date": "NA",
        "next_upcoming_event_label": "",
        "next_upcoming_event_date": "",
    }

def fetch_stock_data(symbol):
    df = yf.download(
        symbol,
        period="300d",
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    df.reset_index(inplace=True)

    if isinstance(df.columns, pd.MultiIndex):
        cols = [c[0] if isinstance(c, tuple) else c for c in df.columns]
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
        raise Exception(
            f"Missing required 'close' column after normalization for {symbol}. "
            f"Available columns: {', '.join(df.columns)}"
        )

    return df


def fetch_fundamentals(symbol):
    ticker = yf.Ticker(symbol)
    info = ticker.info

    return {
        "pe": info.get("trailingPE"),
        "pb": info.get("priceToBook"),
        "marketCap": info.get("marketCap"),
        "roe": info.get("returnOnEquity"),
        "debtToEquity": info.get("debtToEquity"),
        "dividendYield": info.get("dividendYield"),
        "beta": info.get("beta") if info.get("beta") is not None else info.get("beta3Year"),
        "upcomingEvents": build_upcoming_event_summary(ticker,info),
    }
