"""
Support / resistance levels for the equity report.

Two complementary views, since a single-day pivot alone isn't that useful
for a multi-week swing/position-holding report:

  1. Classic floor-trader pivot points (P, R1-R3, S1-S3), computed from the
     most recently COMPLETED trading day's high/low/close. Standard
     formula used by every retail terminal (Zerodha Kite, Sharekhan, etc).
     Useful for gauging the very next session's likely turning points.

  2. Rolling swing high/low zones (20-day, 50-day), which matter more for
     a report whose stated recommendations run over weeks, not a single
     session -- these show the broader range price has actually
     respected recently, not just yesterday's bar.

Both are computed straight from the OHLC dataframe stock_fetcher.py
already returns (normalized lower-case column names: open/high/low/close).
"""

import pandas as pd


def compute_pivot_levels(df):
    """Classic (floor-trader) pivot points from the previous completed
    session's H/L/C. Returns None if there isn't at least 2 rows of data
    or the previous row's OHLC is incomplete."""
    if df is None or len(df) < 2:
        return None

    prev = df.iloc[-2]
    high, low, close = prev.get("high"), prev.get("low"), prev.get("close")

    if any(v is None or pd.isna(v) for v in (high, low, close)):
        return None

    high, low, close = float(high), float(low), float(close)
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)

    basis_date = prev.get("date")
    basis_date_str = None
    if basis_date is not None:
        try:
            basis_date_str = pd.Timestamp(basis_date).strftime("%d %b %Y")
        except Exception:
            basis_date_str = str(basis_date)

    return {
        "pivot": round(pivot, 2),
        "r1": round(r1, 2), "r2": round(r2, 2), "r3": round(r3, 2),
        "s1": round(s1, 2), "s2": round(s2, 2), "s3": round(s3, 2),
        "basis_date": basis_date_str,
    }


def compute_swing_zones(df, windows=(20, 50)):
    """Rolling high/low range over each window (in trading days). Reuses
    the high20/low20 columns calculate_indicators() already adds when
    present, otherwise computes them fresh."""
    if df is None or df.empty:
        return {}

    zones = {}
    for window in windows:
        label = f"{window}d"
        high_col, low_col = f"high{window}", f"low{window}"
        if high_col in df.columns and low_col in df.columns:
            h, l = df[high_col].iloc[-1], df[low_col].iloc[-1]
        elif "high" in df.columns and "low" in df.columns and len(df) >= window:
            h = df["high"].rolling(window).max().iloc[-1]
            l = df["low"].rolling(window).min().iloc[-1]
        else:
            continue
        if pd.notna(h) and pd.notna(l):
            zones[label] = {"high": round(float(h), 2), "low": round(float(l), 2)}
    return zones


def nearest_levels(current_price, pivot_levels):
    """Returns (nearest_support, nearest_resistance) from the pivot ladder,
    i.e. the closest level below and above the current price."""
    if not pivot_levels or current_price is None:
        return None, None
    all_levels = sorted(
        pivot_levels[k] for k in ("s3", "s2", "s1", "pivot", "r1", "r2", "r3")
    )
    supports = [lvl for lvl in all_levels if lvl <= current_price]
    resistances = [lvl for lvl in all_levels if lvl > current_price]
    nearest_support = max(supports) if supports else None
    nearest_resistance = min(resistances) if resistances else None
    return nearest_support, nearest_resistance


def build_support_resistance_html(pivot_levels, swing_zones, current_price):
    """Small two-row addition to the per-stock parameter table: the pivot
    ladder on one row, 20D/50D swing ranges on the next. Returns "" if
    there's nothing to show (keeps a data gap silent rather than a row of
    dashes)."""
    if not pivot_levels and not swing_zones:
        return ""

    sans = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
    rows_html = ""

    if pivot_levels:
        nearest_support, nearest_resistance = nearest_levels(current_price, pivot_levels)
        bias_note = ""
        if current_price is not None:
            if current_price >= pivot_levels["pivot"]:
                bias_note = '<span style="color:#16a34a;font-weight:700;">Trading above pivot</span>'
            else:
                bias_note = '<span style="color:#dc2626;font-weight:700;">Trading below pivot</span>'

        ladder = (
            f'S3 {pivot_levels["s3"]} &nbsp;&middot;&nbsp; S2 {pivot_levels["s2"]} &nbsp;&middot;&nbsp; '
            f'S1 {pivot_levels["s1"]} &nbsp;&middot;&nbsp; <strong>P {pivot_levels["pivot"]}</strong> '
            f'&nbsp;&middot;&nbsp; R1 {pivot_levels["r1"]} &nbsp;&middot;&nbsp; R2 {pivot_levels["r2"]} '
            f'&nbsp;&middot;&nbsp; R3 {pivot_levels["r3"]}'
        )
        basis = f' (basis {pivot_levels["basis_date"]})' if pivot_levels.get("basis_date") else ""
        rows_html += f"""
        <tr>
            <td colspan="2" style="padding:10px 0 4px;">
                <strong style="font-family:{sans};font-size:10px;text-transform:uppercase;letter-spacing:0.06em;color:#8A8F9C;">Pivot Support / Resistance{basis}</strong>
                <div style="margin-top:5px;font-family:{sans};font-size:12px;color:#0f172a;">{ladder}</div>
                <div style="margin-top:4px;font-family:{sans};font-size:11px;">{bias_note}{' &nbsp;&middot;&nbsp; Nearest support ' + str(nearest_support) if nearest_support is not None else ''}{' / resistance ' + str(nearest_resistance) if nearest_resistance is not None else ''}</div>
            </td>
        </tr>
        """

    if swing_zones:
        cells = ""
        for label in ("20d", "50d"):
            zone = swing_zones.get(label)
            if not zone:
                continue
            cells += (
                f'<td style="padding:6px 0;width:50%;"><strong>{label.upper()} Range</strong>'
                f'<div style="color:#0f172a;margin-top:4px;">{zone["low"]} - {zone["high"]}</div></td>'
            )
        if cells:
            rows_html += f"<tr>{cells}</tr>"

    return rows_html