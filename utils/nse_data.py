"""
utils/nse_data.py

Data access + data-quality layer for the daily breakout screener
(controllers/breakout_controller.py).

Two independent data sources, used for different jobs:
  - NSE Bhavcopy (official end-of-day CSV for the whole exchange) -- used
    ONLY as a same-day cross-check: does today's close/volume for a symbol
    roughly agree with what yfinance says? This catches the two failure
    modes that would otherwise silently produce a wrong "breakout":
    (1) yfinance serving a stale/adjusted price that doesn't match the
    real market close, and (2) a corporate action (split/bonus) on the
    scan date that distorts the price series.
  - yfinance -- used for full OHLCV history per symbol (needed for
    pattern detection + backtesting, which bhavcopy alone can't provide
    since it's a single day's snapshot).

Everything here is fail-soft by design: a bhavcopy fetch failure does not
stop the scan, it just removes the same-day cross-check and the report
notes that. A single symbol's yfinance failure just drops that symbol
from the scan (logged), it never crashes the run.
"""

import io
import zipfile
import datetime as dt

import requests
import pandas as pd

from utils.logger import log

_NSE_HEADERS = {
    # NSE's archive endpoints reject requests without a browser-like
    # User-Agent (and sometimes want an initial cookie-setting hit to
    # nseindia.com first) -- both handled below.
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/csv,application/csv,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}

NIFTY500_LIST_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"

# Fallback, hand-maintained snapshot used ONLY if the live NIFTY 500 list
# fetch fails outright. This is intentionally NOT the full 500 -- it's a
# liquid, stable core so the scan degrades to "smaller universe" instead
# of "no run at all" on a bad day. Update occasionally; it is not meant
# to track index reconstitutions precisely.
NIFTY500_FALLBACK_CORE = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "SBIN", "ITC",
    "LT", "BHARTIARTL", "HINDUNILVR", "BAJFINANCE", "KOTAKBANK", "AXISBANK",
    "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO", "NTPC", "ONGC", "COALINDIA",
    "TATASTEEL", "TATAMOTORS", "WIPRO", "HCLTECH", "POWERGRID", "M&M",
    "ADANIENT", "ADANIPORTS", "ASIANPAINT", "BAJAJFINSV", "BEL", "BPCL",
    "CIPLA", "DIVISLAB", "DRREDDY", "EICHERMOT", "GRASIM", "HDFCLIFE",
    "HEROMOTOCO", "HINDALCO", "INDUSINDBK", "JSWSTEEL", "NESTLEIND",
    "SBILIFE", "SHREECEM", "TATACONSUM", "TECHM", "UPL", "APOLLOHOSP",
    "BAJAJ-AUTO", "BRITANNIA",
]


def _nse_session():
    """A requests session that first hits the NSE homepage to pick up the
    cookies NSE's archive endpoints expect -- a plain GET without this
    frequently gets a 401/403 even with a browser User-Agent."""
    session = requests.Session()
    session.headers.update(_NSE_HEADERS)
    try:
        session.get("https://www.nseindia.com", timeout=8)
    except Exception as e:
        log.warning(f"NSE session warm-up failed (continuing anyway): {e}")
    return session


def get_nifty500_symbols():
    """
    Returns (symbols, is_live):
      symbols: sorted list of NSE trading symbols (e.g. "RELIANCE"),
        WITHOUT the ".NS" suffix -- callers append that for yfinance.
      is_live: True if this came from the live NSE index list, False if
        it fell back to NIFTY500_FALLBACK_CORE.
    """
    try:
        session = _nse_session()
        resp = session.get(NIFTY500_LIST_URL, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        symbol_col = next((c for c in df.columns if c.strip().lower() == "symbol"), None)
        if not symbol_col:
            raise ValueError(f"Unexpected NIFTY 500 CSV columns: {list(df.columns)}")
        symbols = sorted(set(df[symbol_col].dropna().astype(str).str.strip()))
        if len(symbols) < 400:
            raise ValueError(f"NIFTY 500 list fetch returned only {len(symbols)} symbols -- suspicious, treating as failed.")
        log.info(f"Breakout Screener: fetched {len(symbols)} live NIFTY 500 symbols.")
        return symbols, True
    except Exception as e:
        log.warning(f"Breakout Screener: live NIFTY 500 list fetch failed ({e}) -- using {len(NIFTY500_FALLBACK_CORE)}-symbol fallback core.")
        return sorted(NIFTY500_FALLBACK_CORE), False


def _bhavcopy_udiff_url(date):
    # Current (post-2024) NSE "UDiFF" common bhavcopy final file.
    return (
        "https://nsearchives.nseindia.com/content/cm/"
        f"BhavCopy_NSE_CM_0_0_0_{date.strftime('%Y%m%d')}_F_0000.csv.zip"
    )


def _bhavcopy_legacy_url(date):
    # Older per-day bhavcopy path, kept as a second attempt in case UDiFF
    # is briefly unavailable or the endpoint changes again.
    mon = date.strftime("%b").upper()
    return (
        f"https://archives.nseindia.com/content/historical/EQUITIES/"
        f"{date.year}/{mon}/cm{date.strftime('%d')}{mon}{date.year}bhav.csv.zip"
    )


def _parse_bhavcopy_zip(content):
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        with zf.open(csv_name) as f:
            df = pd.read_csv(f)
    df.columns = [c.strip() for c in df.columns]
    return df


def _normalize_bhavcopy_columns(df):
    """UDiFF and legacy bhavcopy files use different column names for the
    same fields -- normalize both down to a fixed set of columns so the
    rest of the code doesn't care which format was fetched."""
    rename_map = {
        # UDiFF style
        "TckrSymb": "SYMBOL", "SctySrs": "SERIES", "ClsPric": "CLOSE",
        "OpnPric": "OPEN", "HghPric": "HIGH", "LwPric": "LOW",
        "TtlTradgVol": "VOLUME", "TtlTrfVal": "TURNOVER",
        # Legacy style
        "SYMBOL": "SYMBOL", "SERIES": "SERIES", "CLOSE": "CLOSE",
        "OPEN": "OPEN", "HIGH": "HIGH", "LOW": "LOW",
        "TOTTRDQTY": "VOLUME", "TOTTRDVAL": "TURNOVER",
    }
    cols = {c: rename_map[c] for c in df.columns if c in rename_map}
    df = df.rename(columns=cols)
    keep = [c for c in ["SYMBOL", "SERIES", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "TURNOVER"] if c in df.columns]
    df = df[keep]
    if "SERIES" in df.columns:
        # EQ = normal equity series -- excludes SME, ETFs, debt instruments
        # etc. that would otherwise pollute the cross-check.
        df = df[df["SERIES"].astype(str).str.strip() == "EQ"]
    return df.reset_index(drop=True)


def get_bhavcopy(as_of_date=None, max_days_back=6):
    """
    Returns (df, actual_date, is_live):
      df: normalized bhavcopy DataFrame (SYMBOL, OPEN, HIGH, LOW, CLOSE,
        VOLUME, TURNOVER) for the most recent trading day at or before
        as_of_date, or None if every attempt failed.
      actual_date: the date the returned data is actually for.
      is_live: False (and df=None) if no source produced usable data --
        callers should treat this as "cross-check unavailable this run"
        rather than fail the whole scan.

    Walks backward day by day (skipping weekends) up to max_days_back
    times, since as_of_date may be a market holiday with no bhavcopy.
    """
    if as_of_date is None:
        as_of_date = dt.date.today()

    session = _nse_session()
    date = as_of_date
    attempts = 0
    while attempts < max_days_back:
        if date.weekday() >= 5:  # Saturday/Sunday -- no bhavcopy, skip without counting as an attempt
            date -= dt.timedelta(days=1)
            continue
        for url_fn in (_bhavcopy_udiff_url, _bhavcopy_legacy_url):
            try:
                resp = session.get(url_fn(date), timeout=20)
                if resp.status_code == 200 and resp.content[:2] == b"PK":  # valid zip signature
                    df = _normalize_bhavcopy_columns(_parse_bhavcopy_zip(resp.content))
                    if len(df) > 500:  # sanity floor -- a real bhavcopy has thousands of EQ rows
                        log.info(f"Breakout Screener: bhavcopy loaded for {date} ({len(df)} EQ rows) via {url_fn.__name__}.")
                        return df, date, True
            except Exception as e:
                log.warning(f"Breakout Screener: bhavcopy fetch failed for {date} via {url_fn.__name__}: {e}")
        attempts += 1
        date -= dt.timedelta(days=1)

    log.warning(f"Breakout Screener: no usable bhavcopy found within {max_days_back} trading days back from {as_of_date} -- same-day cross-check disabled this run.")
    return None, None, False


# -----------------------------------------------------------------------
# Data-quality gate applied per symbol, before any pattern is evaluated.
# This is the "enhanced conditions/caution" layer the screener relies on
# to avoid flagging breakouts on bad data.
# -----------------------------------------------------------------------
def data_quality_check(symbol, hist_df, bhav_df, min_history_days=260, price_cross_check_pct=0.03):
    """
    hist_df: yfinance OHLCV history for this symbol (ascending by date).
    bhav_df: normalized bhavcopy DataFrame for the scan date, or None if
      the bhavcopy fetch failed this run.

    Returns (ok: bool, notes: list[str]). ok=False means "skip this
    symbol entirely this run" (e.g. not enough history to trust a
    pattern/backtest). Non-fatal issues are returned as caution notes
    with ok still True, so the caller can attach them to any signal it
    finds for this symbol instead of hiding it silently.
    """
    notes = []

    if hist_df is None or hist_df.empty:
        return False, ["No price history returned."]

    if len(hist_df) < min_history_days:
        return False, [f"Only {len(hist_df)} trading days of history (<{min_history_days}) -- too little for a reliable pattern/backtest read."]

    last = hist_df.iloc[-1]

    if last.get("Volume", 0) in (0, None) or pd.isna(last.get("Volume")):
        return False, ["Zero/missing volume on the scan date -- likely not traded, illiquid, or suspended."]

    # Flat-line day (open == high == low == close): common for suspended
    # or circuit-locked-with-no-trade symbols, and pattern math on a flat
    # day is meaningless.
    if last["Open"] == last["High"] == last["Low"] == last["Close"]:
        return False, ["Open/High/Low/Close identical on scan date -- likely no real trading activity."]

    # Extreme single-day move: not disqualifying (real breakouts can gap),
    # but flagged so a reader knows to check for a split/bonus/corporate
    # action before trusting the signal.
    prev_close = hist_df.iloc[-2]["Close"] if len(hist_df) >= 2 else None
    if prev_close and prev_close > 0:
        day_move = abs(last["Close"] - prev_close) / prev_close
        if day_move >= 0.20:
            notes.append(f"Large single-day move ({day_move*100:.0f}%) -- verify this isn't a split/bonus/corporate action before acting.")

    # Cross-check against bhavcopy, if we have it this run.
    if bhav_df is not None:
        row = bhav_df.loc[bhav_df["SYMBOL"] == symbol]
        if row.empty:
            notes.append("Symbol not found in today's NSE bhavcopy EQ series -- cross-check skipped.")
        else:
            bhav_close = float(row.iloc[0]["CLOSE"])
            if bhav_close > 0:
                diff_pct = abs(last["Close"] - bhav_close) / bhav_close
                if diff_pct > price_cross_check_pct:
                    return False, [
                        f"yfinance close (₹{last['Close']:.2f}) vs NSE bhavcopy close "
                        f"(₹{bhav_close:.2f}) differ by {diff_pct*100:.1f}% -- data mismatch, skipping to avoid a false signal."
                    ]

    return True, notes