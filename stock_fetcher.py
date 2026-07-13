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


def build_upcoming_event_summary(info):
    if not info:
        return {
            "dividend_record_date": "Not available",
            "results_announcement_date": "Not available",
            "next_upcoming_event_label": "Upcoming Event",
            "next_upcoming_event_date": "Not available",
        }

    def _pick(*keys):
        for key in keys:
            value = info.get(key)
            if value in (None, ""):
                continue
            return value
        return None

    def _parse_date(value):
        if value in (None, "", "Not available"):
            return None
        try:
            return pd.Timestamp(value)
        except Exception:
            return None

    dividend_value = _pick("dividendDate", "exDividendDate", "lastDividendDate")
    results_value = _pick("earningsDate", "earningsTimestamp", "nextEarningsDate")

    candidates = []
    dividend_date = _parse_date(dividend_value)
    if dividend_date is not None:
        candidates.append(("Dividend Record", dividend_date, format_event_date(dividend_value)))

    results_date = _parse_date(results_value)
    if results_date is not None:
        candidates.append(("Results Announcement", results_date, format_event_date(results_value)))

    today = pd.Timestamp(datetime.now().date())
    upcoming = [item for item in candidates if item[1].normalize() >= today.normalize()]
    if upcoming:
        next_event = min(upcoming, key=lambda item: item[1])
        return {
            "dividend_record_date": format_event_date(dividend_value),
            "results_announcement_date": format_event_date(results_value),
            "next_upcoming_event_label": next_event[0],
            "next_upcoming_event_date": next_event[2],
        }

    if candidates:
        next_event = min(candidates, key=lambda item: item[1])
        return {
            "dividend_record_date": format_event_date(dividend_value),
            "results_announcement_date": format_event_date(results_value),
            "next_upcoming_event_label": next_event[0],
            "next_upcoming_event_date": next_event[2],
        }

    return {
        "dividend_record_date": format_event_date(dividend_value),
        "results_announcement_date": format_event_date(results_value),
        "next_upcoming_event_label": "Upcoming Event",
        "next_upcoming_event_date": "Not available",
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
        "marketCap": info.get("marketCap"),
        "roe": info.get("returnOnEquity"),
        "debtToEquity": info.get("debtToEquity"),
        "dividendYield": info.get("dividendYield"),
        "upcomingEvents": build_upcoming_event_summary(info),
    }