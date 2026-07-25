"""
analyze_rejection_history.py

Reads the CSV that swing_trade_advisor.py's _log_rejection_history builds up
over successive runs (default: swing_trade_rejection_history.csv, same
directory the workflow runs in -- override with REJECTION_HISTORY_LOG or a
CLI path argument) and reports, per threshold, how often it blocked a
candidate and by how much.

The point: don't relax MIN_GROWTH_YOY_PCT / MAX_DEBT_TO_EQUITY_PCT / etc.
because ONE run's candidates happened to fall just short. Relax a threshold
only once there's a real pattern across several DIFFERENT runs and tickers --
this script is what tells you whether that pattern exists yet.

Usage:
    python analyze_rejection_history.py [path-to-csv]
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


def load(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["threshold"] = float(r["threshold"])
            r["actual_value"] = float(r["actual_value"])
            r["margin_missed_by"] = float(r["margin_missed_by"])
            rows.append(r)
    return rows


def summarize(rows):
    by_metric = defaultdict(list)
    for r in rows:
        by_metric[r["metric"]].append(r)

    dates = sorted(set(r["date"] for r in rows))
    print(f"Rejection history: {len(dates)} distinct run date(s) logged, {len(rows)} threshold-miss(es) total\n")

    for metric, entries in sorted(by_metric.items(), key=lambda kv: -len(kv[1])):
        margins = [e["margin_missed_by"] for e in entries]
        threshold = entries[0]["threshold"]
        run_dates = set(e["date"] for e in entries)
        tickers = set(e["ticker"] for e in entries)
        # "near-miss" = missed by no more than 25% of the threshold's own size
        near_misses = [e for e in entries if 0 < e["margin_missed_by"] <= abs(threshold) * 0.25]

        print(f"--- {metric}  (threshold in effect: {threshold}) ---")
        print(f"  Blocked {len(entries)} candidate-instance(s) across {len(run_dates)} run(s), {len(tickers)} distinct ticker(s)")
        print(f"  Miss margin -- avg: {mean(margins):.2f}, median: {median(margins):.2f}, smallest: {min(margins):.2f}, largest: {max(margins):.2f}")
        print(f"  Near-misses (within 25% of threshold): {len(near_misses)}")
        if near_misses:
            worst = sorted(near_misses, key=lambda e: e["margin_missed_by"])[:5]
            for e in worst:
                print(f"    {e['date']}  {e['name']} ({e['ticker']}): actual={e['actual_value']}, missed by {e['margin_missed_by']}")

        if len(run_dates) >= 3 and len(near_misses) >= max(2, len(entries) // 2):
            print("  -> Recurring near-misses across multiple runs -- this threshold may genuinely be tighter than needed.")
        elif len(run_dates) <= 1:
            print("  -> Only one run so far -- not enough history yet to tell coincidence from a real pattern.")
        else:
            print("  -> Misses aren't consistently close -- no strong case yet for relaxing this one.")
        print()

    print(
        "Read this as: a metric with recurring near-misses spread across several\n"
        "DIFFERENT runs AND different tickers is a real signal the bar may be\n"
        "tighter than your strategy needs -- worth relaxing deliberately. A metric\n"
        "that only shows up once, or whose misses are large and scattered, is more\n"
        "likely that run's coincidence, not something to change."
    )


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "swing_trade_rejection_history.csv"
    if not Path(path).exists():
        print(f"No history file found at {path} yet -- run swing_trade_advisor.py a few times first to build one up.")
        sys.exit(0)
    rows = load(path)
    if not rows:
        print(f"{path} exists but has no rows yet.")
        sys.exit(0)
    summarize(rows)