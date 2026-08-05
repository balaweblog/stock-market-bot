from datetime import date, datetime

import yfinance as yf
import pandas as pd

from utils.yf_throttle import get_shared_session, call_with_retries, throttle as _yf_pace

# NOTE: this used to create its own module-level requests.Session, separate
# from the one models/market_context.py and controllers/stock_controller.py
# each built independently -- three uncoordinated sessions (so three
# separate cookie/crumb negotiations) and three uncoordinated request
# streams hitting Yahoo Finance, which is why fundamentals lookups here
# could get 429'd even in a run where fetch_data() itself was behaving.
# _yf_session now points at the single process-wide shared session so the
# crumb is negotiated once and every module's request-rate throttle is the
# same clock (see utils/yf_throttle.py).
_yf_session = get_shared_session()
# Inject the shared session into yfinance's internal request handling too,
# for the one call below (fetch_stock_data) that doesn't pass a session
# explicitly.
yf.utils._requests = _yf_session


def _with_retries(func, symbol, max_attempts=5):
    # Delegates to the shared, rate-limit-aware retry helper so this
    # module's backoff behavior (and the shared throttle applied before
    # every attempt) matches every other module that talks to yfinance.
    return call_with_retries(func, symbol, max_attempts=max_attempts)


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
    if info is None and isinstance(ticker, dict):
        info = ticker
        ticker = None
    elif info is None:
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
    df = call_with_retries(
        yf.download,
        symbol,
        period="300d",
        interval="1d",
        auto_adjust=True,
        progress=False,
        session=_yf_session,
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


def _fetch_fundamentals_raw(symbol):
    # Use custom session with realistic User-Agent for Yahoo Finance requests
    ticker = yf.Ticker(symbol, session=_yf_session)
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

def fetch_fundamentals(symbol):
    return _with_retries(_fetch_fundamentals_raw, symbol)

def _fetch_advanced_fundamentals_raw(symbol):
    ticker = yf.Ticker(symbol, session=_yf_session)
    info = ticker.info

    return {
        "forwardPE": info.get("forwardPE"),
        "pegRatio": info.get("pegRatio"),
        "priceToSales": info.get("priceToSalesTrailing12Months"),
        "ebitdaMargins": info.get("ebitdaMargins"),
        "profitMargins": info.get("profitMargins"),
        "operatingMargins": info.get("operatingMargins"),
        "currentRatio": info.get("currentRatio"),
        "quickRatio": info.get("quickRatio"),
        "freeCashflow": info.get("freeCashflow"),
        "operatingCashflow": info.get("operatingCashflow"),
        "revenueGrowth": info.get("revenueGrowth"),
        "earningsGrowth": info.get("earningsGrowth"),
    }

def fetch_advanced_fundamentals(symbol):
    return _with_retries(_fetch_advanced_fundamentals_raw, symbol)


def _safe_pct(value):
    """Return float percentage (0-100 scale) or None."""
    if value is None:
        return None
    try:
        v = float(value)
        # yfinance returns fractions (0.12 = 12%) for major_holders breakdown
        return round(v * 100 if v <= 1.0 else v, 2)
    except (TypeError, ValueError):
        return None


def _direction_arrow(current, previous):
    """Return (arrow, color) comparing two percentage values."""
    if current is None or previous is None:
        return "→", "#64748b"
    diff = current - previous
    if diff > 0.3:
        return "↑", "#16a34a"
    if diff < -0.3:
        return "↓", "#dc2626"
    return "→", "#64748b"


def _fetch_ownership_activity_raw(symbol):
    """
    Fetches institutional, mutual-fund, insider ownership data from yfinance.

    Returns a dict with:
      institutional_pct      – % held by FII/institutional investors (latest)
      institutional_prev_pct – % held previous quarter (for direction)
      institutional_arrow    – ↑ / ↓ / →
      institutional_color
      mutualfund_pct         – % held by domestic MF/DII (latest)
      mutualfund_prev_pct
      mutualfund_arrow
      mutualfund_color
      insider_pct            – % held by insiders/promoters
      insider_arrow
      insider_color
      net_insider_activity   – "Net Buy", "Net Sell", or "Neutral" from netSharePurchaseActivity
      net_insider_shares     – raw net shares int or None
      available              – bool, False if all data missing
    """
    result = {
        "institutional_pct": None,
        "institutional_prev_pct": None,
        "institutional_arrow": "→",
        "institutional_color": "#64748b",
        "mutualfund_pct": None,
        "mutualfund_prev_pct": None,
        "mutualfund_arrow": "→",
        "mutualfund_color": "#64748b",
        "insider_pct": None,
        "insider_arrow": "→",
        "insider_color": "#64748b",
        "net_insider_activity": "Neutral",
        "net_insider_shares": None,
        "available": False,
    }

    try:
        ticker = yf.Ticker(symbol, session=_yf_session)
        # This function makes several separate yfinance HTTP calls
        # (info, institutional_holders, mutualfund_holders,
        # insider_purchases) back to back. call_with_retries()/_with_retries
        # only throttles once at the top of the whole function, so each of
        # these internal calls needs its own throttle() to keep them spaced
        # out too -- otherwise they'd fire as an uncoordinated burst.
        _yf_pace()
        info = ticker.info or {}

        # ── Insider / Promoter % ──────────────────────────────────────────
        insider_pct = _safe_pct(info.get("heldPercentInsiders"))
        result["insider_pct"] = insider_pct

        # ── Institutional (FII proxy) % ───────────────────────────────────
        _yf_pace()
        ih = ticker.institutional_holders
        if ih is not None and not ih.empty:
            # Sum pctHeld across all institutional holders for latest two report dates
            pct_col = next((c for c in ih.columns if "pct" in c.lower() or "percent" in c.lower()), None)
            date_col = next((c for c in ih.columns if "date" in c.lower()), None)

            if pct_col and date_col:
                ih = ih.sort_values(date_col, ascending=False)
                dates = ih[date_col].dropna().unique()
                if len(dates) >= 1:
                    latest_rows = ih[ih[date_col] == dates[0]]
                    result["institutional_pct"] = round(
                        float(latest_rows[pct_col].apply(
                            lambda x: float(x) * 100 if float(x) <= 1.0 else float(x)
                        ).sum()), 2
                    )
                if len(dates) >= 2:
                    prev_rows = ih[ih[date_col] == dates[1]]
                    result["institutional_prev_pct"] = round(
                        float(prev_rows[pct_col].apply(
                            lambda x: float(x) * 100 if float(x) <= 1.0 else float(x)
                        ).sum()), 2
                    )
            else:
                # Fallback: use info field
                result["institutional_pct"] = _safe_pct(info.get("heldPercentInstitutions"))

        else:
            result["institutional_pct"] = _safe_pct(info.get("heldPercentInstitutions"))

        arrow, color = _direction_arrow(
            result["institutional_pct"], result["institutional_prev_pct"]
        )
        result["institutional_arrow"] = arrow
        result["institutional_color"] = color

        # ── Mutual Fund / DII % ───────────────────────────────────────────
        _yf_pace()
        mf = ticker.mutualfund_holders
        if mf is not None and not mf.empty:
            pct_col = next((c for c in mf.columns if "pct" in c.lower() or "percent" in c.lower()), None)
            date_col = next((c for c in mf.columns if "date" in c.lower()), None)

            if pct_col and date_col:
                mf = mf.sort_values(date_col, ascending=False)
                dates = mf[date_col].dropna().unique()
                if len(dates) >= 1:
                    latest_rows = mf[mf[date_col] == dates[0]]
                    result["mutualfund_pct"] = round(
                        float(latest_rows[pct_col].apply(
                            lambda x: float(x) * 100 if float(x) <= 1.0 else float(x)
                        ).sum()), 2
                    )
                if len(dates) >= 2:
                    prev_rows = mf[mf[date_col] == dates[1]]
                    result["mutualfund_prev_pct"] = round(
                        float(prev_rows[pct_col].apply(
                            lambda x: float(x) * 100 if float(x) <= 1.0 else float(x)
                        ).sum()), 2
                    )

        arrow, color = _direction_arrow(
            result["mutualfund_pct"], result["mutualfund_prev_pct"]
        )
        result["mutualfund_arrow"] = arrow
        result["mutualfund_color"] = color

        # ── Net insider purchase activity ─────────────────────────────────
        net_shares = None
        activity = "Neutral"
        _yf_pace()
        purchases = ticker.insider_purchases
        if purchases is not None and not purchases.empty:
            if "Net Shares Purchased (Sold)" in purchases.columns:
                try:
                    net_shares = int(purchases["Net Shares Purchased (Sold)"].iloc[0])
                    if net_shares > 0:
                        activity = "Net Buy"
                    elif net_shares < 0:
                        activity = "Net Sell"
                except (ValueError, TypeError, IndexError):
                    pass

        result["net_insider_activity"] = activity
        result["net_insider_shares"] = net_shares

        # Insider direction arrow based on net activity
        if result["net_insider_activity"] == "Net Buy":
            result["insider_arrow"] = "↑"
            result["insider_color"] = "#16a34a"
        elif result["net_insider_activity"] == "Net Sell":
            result["insider_arrow"] = "↓"
            result["insider_color"] = "#dc2626"
        else:
            result["insider_arrow"] = "→"
            result["insider_color"] = "#64748b"

        # Mark available if we got at least one meaningful data point
        result["available"] = any([
            result["institutional_pct"] is not None,
            result["mutualfund_pct"] is not None,
            result["insider_pct"] is not None,
        ])

    except Exception:
        pass  # Return default result with available=False

    return result


def fetch_ownership_activity(symbol):
    return _with_retries(_fetch_ownership_activity_raw, symbol)