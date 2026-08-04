"""
swing_trade_outcomes.py

Outcome-tracking feedback loop: logs every stock this system actually
recommended (emailed), then -- run periodically, e.g. weekly, separately
from the main scan -- checks real price history since to see whether each
pick hit its target, its stop, or neither within the horizon.

WHY: the existing rejection-history log (REJECTION_HISTORY_LOG in
swing_trade_advisor.py) tracks candidates that DIDN'T pass, for threshold
tuning. Nothing tracked what happened to the ones that DID pass and got
emailed. Without that, there's no way to ever measure whether this system
actually works -- you're flying blind on your own track record. This
module is that missing feedback loop.

Two entry points:
  log_recommendation(stock, today_str)      -- call this once per emailed
                                                 pick, right when the email
                                                 is sent (see
                                                 swing_trade_advisor.run()).
  update_and_summarize_outcomes()            -- run this on a schedule
                                                 (e.g. its own weekly cron,
                                                 separate from the scan) to
                                                 fetch real price history
                                                 for every logged pick and
                                                 fill in what happened.

Also runnable directly: `python swing_trade_outcomes.py` updates the log
and prints a plain-text summary (win rate, average realized R:R, hit
rate by strategy_type) to stdout -- suitable for its own cron entry that
emails you the output, or for eyeballing locally.
"""

import csv
import sys
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from utils.logger import log
from services.stock_fetcher import fetch_stock_data

OUTCOMES_LOG = "swing_trade_outcomes_log.csv"

FIELDNAMES = [
    "date_recommended", "ticker", "name", "sector", "strategy_type",
    "entry_price", "stop_loss_pct", "target1_pct", "target2_pct",
    "exit_date_expected", "confidence_score_adjusted",
    "outcome",            # "pending" | "hit_target1" | "hit_target2" | "hit_stop" | "expired_no_hit" | "data_unavailable"
    "outcome_price", "outcome_date", "realized_pct", "days_in_trade",
    "date_outcome_checked",
]


def log_recommendation(stock, today_str_iso=None):
    """
    Appends one row for a stock that was actually emailed as a
    recommendation this run. `today_str_iso` should be an ISO date string
    (YYYY-MM-DD); defaults to "now" in IST if not supplied.

    Best-effort and silent-on-failure, same convention as
    swing_trade_advisor._log_rejection_history -- a broken log write must
    never abort or change the outcome of an actual scan run.
    """
    if today_str_iso is None:
        today_str_iso = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d")

    try:
        entry_price = None
        price_display = stock.get("current_price_display")
        if price_display:
            digits = "".join(c for c in price_display if c.isdigit() or c == ".")
            entry_price = float(digits) if digits else None

        row = {
            "date_recommended": today_str_iso,
            "ticker": (stock.get("ticker") or "").strip(),
            "name": stock.get("name") or "",
            "sector": stock.get("sector") or "",
            "strategy_type": stock.get("strategy_type") or "",
            "entry_price": entry_price,
            "stop_loss_pct": stock.get("stop_loss_pct"),
            "target1_pct": stock.get("target1_pct"),
            "target2_pct": stock.get("target2_pct"),
            "exit_date_expected": stock.get("exit_date") or "",
            "confidence_score_adjusted": stock.get("confidence_score_adjusted"),
            "outcome": "pending",
            "outcome_price": "",
            "outcome_date": "",
            "realized_pct": "",
            "days_in_trade": "",
            "date_outcome_checked": "",
        }
        path = Path(OUTCOMES_LOG)
        write_header = not path.exists()
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        log.warning(f"Could not write outcome log entry: {e}")


def _load_rows(log_path=OUTCOMES_LOG):
    path = Path(log_path)
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _save_rows(rows, log_path=OUTCOMES_LOG):
    with Path(log_path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _parse_pct(value):
    try:
        digits = "".join(c for c in str(value) if c.isdigit() or c in ".-")
        return float(digits) if digits else None
    except (ValueError, TypeError):
        return None


HORIZON_DAYS_IF_NO_EXIT_DATE = 150  # ~5 months, matches the strategy's stated horizon


def _check_one_outcome(row, today):
    """
    Fetches real daily price history for `row["ticker"]` since
    date_recommended and determines whether the stop or either target was
    hit first, walking forward day by day (stop-loss checked before
    target on any day both would have been touched intraday, since a
    daily OHLC bar can't tell us the actual intraday sequence and assuming
    the worse outcome first is the conservative choice for a track record).

    Returns an updated copy of `row`. Leaves "outcome" as "pending" if the
    horizon hasn't elapsed yet and neither level has been hit.
    """
    ticker = row.get("ticker", "").strip()
    entry_price = _parse_pct(row.get("entry_price"))
    # stop_loss_pct/target1_pct/target2_pct are unsigned magnitudes
    # everywhere else in this codebase (see swing_trade_advisor.py's
    # _parse_first_number, which strips sign entirely, and
    # swing_trade_risk.py's ATR plan, which always computes a positive
    # %). _parse_pct above deliberately keeps a leading "-" (useful for
    # other numeric fields), so take abs() here specifically -- otherwise
    # a model that happened to write stop_loss_pct as "-8%" would flip
    # the computed stop_price to ABOVE entry_price instead of below it,
    # silently corrupting the very outcome-tracking feedback loop this
    # module exists to keep honest.
    stop_pct = _parse_pct(row.get("stop_loss_pct"))
    stop_pct = abs(stop_pct) if stop_pct is not None else None
    t1_pct = _parse_pct(row.get("target1_pct"))
    t1_pct = abs(t1_pct) if t1_pct is not None else None
    t2_pct = _parse_pct(row.get("target2_pct"))
    t2_pct = abs(t2_pct) if t2_pct is not None else None

    if not ticker or entry_price is None or stop_pct is None or t1_pct is None:
        row["outcome"] = "data_unavailable"
        row["date_outcome_checked"] = today.strftime("%Y-%m-%d")
        return row

    try:
        rec_date = datetime.strptime(row["date_recommended"], "%Y-%m-%d")
    except (ValueError, KeyError):
        row["outcome"] = "data_unavailable"
        row["date_outcome_checked"] = today.strftime("%Y-%m-%d")
        return row

    horizon_end = rec_date + timedelta(days=HORIZON_DAYS_IF_NO_EXIT_DATE)

    try:
        df = fetch_stock_data(ticker)
        if df is None or "close" not in df.columns:
            row["outcome"] = "data_unavailable"
            row["date_outcome_checked"] = today.strftime("%Y-%m-%d")
            return row
        if not isinstance(df.index, pd.DatetimeIndex):
            date_col = next((c for c in df.columns if c.lower() == "date"), None)
            if date_col is None:
                row["outcome"] = "data_unavailable"
                row["date_outcome_checked"] = today.strftime("%Y-%m-%d")
                return row
            df = df.set_index(pd.to_datetime(df[date_col]))

        window = df[df.index >= rec_date]
        if window.empty:
            row["outcome"] = "pending"
            return row

        stop_price = entry_price * (1 - stop_pct / 100.0)
        t1_price = entry_price * (1 + t1_pct / 100.0)
        t2_price = entry_price * (1 + t2_pct / 100.0) if t2_pct is not None else None

        low_col = "low" if "low" in window.columns else "close"
        high_col = "high" if "high" in window.columns else "close"

        for ts, bar in window.iterrows():
            if bar[low_col] <= stop_price:
                row["outcome"] = "hit_stop"
                row["outcome_price"] = round(stop_price, 2)
                row["outcome_date"] = ts.strftime("%Y-%m-%d")
                row["realized_pct"] = round(-stop_pct, 2)
                row["days_in_trade"] = (ts.to_pydatetime().replace(tzinfo=None) - rec_date).days
                row["date_outcome_checked"] = today.strftime("%Y-%m-%d")
                return row
            if t2_price is not None and bar[high_col] >= t2_price:
                row["outcome"] = "hit_target2"
                row["outcome_price"] = round(t2_price, 2)
                row["outcome_date"] = ts.strftime("%Y-%m-%d")
                row["realized_pct"] = round(t2_pct, 2)
                row["days_in_trade"] = (ts.to_pydatetime().replace(tzinfo=None) - rec_date).days
                row["date_outcome_checked"] = today.strftime("%Y-%m-%d")
                return row
            if bar[high_col] >= t1_price:
                row["outcome"] = "hit_target1"
                row["outcome_price"] = round(t1_price, 2)
                row["outcome_date"] = ts.strftime("%Y-%m-%d")
                row["realized_pct"] = round(t1_pct, 2)
                row["days_in_trade"] = (ts.to_pydatetime().replace(tzinfo=None) - rec_date).days
                row["date_outcome_checked"] = today.strftime("%Y-%m-%d")
                return row

        if today >= horizon_end:
            last_close = float(window["close"].iloc[-1])
            realized = round(((last_close - entry_price) / entry_price) * 100, 2)
            row["outcome"] = "expired_no_hit"
            row["outcome_price"] = round(last_close, 2)
            row["outcome_date"] = window.index[-1].strftime("%Y-%m-%d")
            row["realized_pct"] = realized
            row["days_in_trade"] = (window.index[-1].to_pydatetime().replace(tzinfo=None) - rec_date).days
            row["date_outcome_checked"] = today.strftime("%Y-%m-%d")
        else:
            row["outcome"] = "pending"
        return row
    except Exception as e:
        log.warning(f"Could not check outcome for '{ticker}': {e}")
        row["outcome"] = "data_unavailable"
        row["date_outcome_checked"] = today.strftime("%Y-%m-%d")
        return row


def update_and_summarize_outcomes(log_path=OUTCOMES_LOG):
    """
    Re-checks every "pending" row (only -- resolved rows are never
    re-evaluated) and returns a summary dict of win rate, average realized
    %, and a breakdown by strategy_type across all RESOLVED rows (pending
    ones are excluded from the stats, since they haven't happened yet).
    """
    rows = _load_rows(log_path)
    if not rows:
        return {"total_logged": 0, "resolved": 0, "message": "No recommendations logged yet."}

    today = datetime.now(ZoneInfo("Asia/Kolkata")).replace(tzinfo=None)
    updated_rows = []
    for row in rows:
        if row.get("outcome") == "pending" or not row.get("outcome"):
            row = _check_one_outcome(row, today)
        updated_rows.append(row)
    _save_rows(updated_rows, log_path)

    resolved = [r for r in updated_rows if r.get("outcome") in
                ("hit_target1", "hit_target2", "hit_stop", "expired_no_hit")]
    wins = [r for r in resolved if r.get("outcome") in ("hit_target1", "hit_target2")]
    losses = [r for r in resolved if r.get("outcome") == "hit_stop"]
    flat = [r for r in resolved if r.get("outcome") == "expired_no_hit"]

    realized_values = [_parse_pct(r.get("realized_pct")) for r in resolved]
    realized_values = [v for v in realized_values if v is not None]

    by_strategy = {}
    for r in resolved:
        st = r.get("strategy_type") or "Unknown"
        by_strategy.setdefault(st, {"n": 0, "wins": 0})
        by_strategy[st]["n"] += 1
        if r.get("outcome") in ("hit_target1", "hit_target2"):
            by_strategy[st]["wins"] += 1

    return {
        "total_logged": len(updated_rows),
        "resolved": len(resolved),
        "pending": len(updated_rows) - len(resolved),
        "wins": len(wins),
        "losses": len(losses),
        "expired_flat": len(flat),
        "win_rate_pct": round(100 * len(wins) / len(resolved), 1) if resolved else None,
        "avg_realized_pct": round(sum(realized_values) / len(realized_values), 2) if realized_values else None,
        "by_strategy_type": {
            k: {"n": v["n"], "win_rate_pct": round(100 * v["wins"] / v["n"], 1) if v["n"] else None}
            for k, v in by_strategy.items()
        },
    }


def _print_summary(summary):
    print("=== Swing Trade Advisor -- Outcome Tracking Summary ===")
    for k, v in summary.items():
        if k == "by_strategy_type":
            print("by_strategy_type:")
            for st, stats in v.items():
                print(f"  {st}: n={stats['n']}, win_rate={stats['win_rate_pct']}%")
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    summary = update_and_summarize_outcomes()
    _print_summary(summary)
    if summary.get("resolved", 0) == 0:
        sys.exit(0)