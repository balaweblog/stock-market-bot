"""
controllers/breakout_controller.py

Daily chart-pattern breakout screener over the NIFTY 500 universe.
Separate from wealth_controller.py's monthly SIP checkup and from
stock_controller.py's daily briefing on the fixed watchlist -- this one
scans a much wider universe (NIFTY 500) purely on price/volume
mechanics, no AI narrative involved, and only surfaces a signal after
backtesting that exact pattern against the stock's own history.

Pipeline, per run:
  1. Get today's NIFTY 500 symbol list (utils.nse_data).
  2. Get today's NSE bhavcopy for a same-day price/volume cross-check
     (fail-soft -- proceeds without it if NSE is unreachable, and says
     so in the email).
  3. Pull ~2 years of daily OHLCV per symbol from yfinance, in batches.
  4. Per symbol: data-quality gate (utils.nse_data.data_quality_check) ->
     skip if it fails outright, else carry any caution notes forward.
  5. Run every pattern detector against today's bar
     (utils.breakout_patterns.scan_all_patterns).
  6. For every pattern that fired, replay that SAME detector across the
     symbol's own trailing history (utils.breakout_backtest) to get a
     live hit-rate/avg-return read -- this is what "confirmed" means
     below, never a canned/pre-baked statistic.
  7. Split into "Confirmed Breakouts" (backtest cleared the bar) and
     "Unconfirmed / Watch List" (pattern fired today but backtest sample
     was too thin or the historical hit-rate was weak) -- nothing is
     silently dropped, everything shown is labeled for what it is.
  8. Render + send the email, in the same visual style as the other
     reports in this project.

This is deliberately NOT wired to llm_backend -- every number in the
email traces back to a specific, replayable calculation, not a model's
narrative gloss on the numbers.
"""

import argparse
import datetime as dt
import html
import time

import pandas as pd
import yfinance as yf
from zoneinfo import ZoneInfo

from utils.config import get_date_with_suffix
from utils.logger import log
from utils import email_service
from utils.nse_data import get_nifty500_symbols, get_bhavcopy, data_quality_check
from utils.breakout_patterns import scan_all_patterns, PATTERN_DETECTOR_BY_NAME
from utils.breakout_backtest import backtest_signal, PRIMARY_HORIZON, MIN_SAMPLES_FOR_CONFIDENCE, CONFIRM_HIT_RATE_THRESHOLD

SERIF = "Georgia, 'Times New Roman', serif"
SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"

HISTORY_PERIOD = "2y"
BATCH_SIZE = 40          # symbols per yfinance batch download
BATCH_PAUSE_SECONDS = 1  # be polite to the endpoint between batches
MAX_CONFIRMED_ROWS = 25  # keep the email skimmable even on a big breakout day
MAX_WATCH_ROWS = 15

PATTERN_STYLES = {
    # pattern -> emoji, used purely for quick visual scanning of the table
    "Resistance Breakout": "📈",
    "Volume Surge Breakout": "🔊",
    "52-Week High Breakout": "🚀",
    "Golden Cross (50/200 DMA)": "✳️",
    "Bollinger Squeeze Breakout": "🎯",
    "Bull Flag Breakout": "🚩",
    "Triangle Breakout": "🔺",
    "Cup and Handle Breakout": "☕",
}


# -----------------------------------------------------------------------
# Data pull
# -----------------------------------------------------------------------
def _fetch_history_batch(tickers):
    """One yfinance batch download. Returns {symbol: df} for symbols that
    came back with usable data -- silently omits ones that didn't
    (logged), since a handful of delisted/renamed symbols in any index
    list is normal and shouldn't stop the run."""
    out = {}
    try:
        data = yf.download(
            tickers, period=HISTORY_PERIOD, interval="1d",
            group_by="ticker", auto_adjust=False, threads=True, progress=False,
        )
    except Exception as e:
        log.warning(f"Breakout Screener: batch download failed for {len(tickers)} tickers: {e}")
        return out

    for t in tickers:
        try:
            if len(tickers) == 1:
                df = data
            else:
                df = data[t]
            df = df.dropna(subset=["Close"])
            if not df.empty:
                out[t] = df
        except Exception:
            continue
    return out


def fetch_universe_history(symbols):
    """symbols: NSE symbols without ".NS". Returns {symbol: OHLCV df}."""
    tickers = [f"{s}.NS" for s in symbols]
    symbol_by_ticker = {f"{s}.NS": s for s in symbols}
    result = {}

    for start in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[start:start + BATCH_SIZE]
        batch_data = _fetch_history_batch(batch)
        for ticker, df in batch_data.items():
            result[symbol_by_ticker[ticker]] = df
        log.info(f"Breakout Screener: fetched history for {len(result)}/{len(symbols)} symbols so far...")
        if start + BATCH_SIZE < len(tickers):
            time.sleep(BATCH_PAUSE_SECONDS)

    missing = len(symbols) - len(result)
    if missing:
        log.warning(f"Breakout Screener: {missing} symbols returned no usable yfinance history (delisted/renamed/thin -- skipped).")
    return result


# -----------------------------------------------------------------------
# Scan
# -----------------------------------------------------------------------
def scan_universe(histories, bhav_df):
    """
    histories: {symbol: OHLCV df}
    Returns (confirmed, watch_list, skipped_count):
      confirmed: list of signal dicts that cleared the backtest bar
      watch_list: list of signal dicts that fired today but didn't clear
        the bar (thin sample or weak historical hit-rate)
      skipped_count: symbols dropped entirely by the data-quality gate
    """
    confirmed, watch_list = [], []
    skipped_count = 0

    for symbol, df in histories.items():
        ok, dq_notes = data_quality_check(symbol, df, bhav_df)
        if not ok:
            skipped_count += 1
            continue

        i = len(df) - 1
        today_signals = scan_all_patterns(df, i)
        if not today_signals:
            continue

        for sig in today_signals:
            detector_fn = PATTERN_DETECTOR_BY_NAME.get(sig["pattern"])
            bt = backtest_signal(symbol, df, detector_fn, i) if detector_fn else None

            row = {
                "symbol": symbol,
                "pattern": sig["pattern"],
                "signal_price": sig["signal_price"],
                "detail": sig["detail"],
                "dq_notes": dq_notes,
                "backtest": bt,
            }
            if bt and bt.get("confirmed"):
                confirmed.append(row)
            else:
                watch_list.append(row)

    def _rank_key(row):
        bt = row["backtest"] or {}
        primary = (bt.get("horizons") or {}).get(PRIMARY_HORIZON, {})
        return (primary.get("hit_rate", 0), bt.get("sample_size", 0))

    confirmed.sort(key=_rank_key, reverse=True)
    watch_list.sort(key=_rank_key, reverse=True)
    return confirmed, watch_list, skipped_count


# -----------------------------------------------------------------------
# Email rendering -- same visual language as wealth_controller.py's report
# -----------------------------------------------------------------------
def _backtest_cell(bt):
    if not bt or not bt.get("horizons"):
        return '<span style="color:#8A8F9C;">No historical sample yet</span>'
    primary = bt["horizons"].get(PRIMARY_HORIZON)
    if not primary:
        return '<span style="color:#8A8F9C;">No historical sample yet</span>'
    return (
        f'{primary["hit_rate"]*100:.0f}% hit-rate over {bt["sample_size"]} past occurrences '
        f'(avg {primary["avg_return"]*100:+.1f}% at {PRIMARY_HORIZON}d)'
    )


def _signal_rows_html(rows, row_bg):
    if not rows:
        return '<tr><td style="padding:10px 12px;font-family:{0};font-size:12px;color:#8A8F9C;" colspan="4">None today.</td></tr>'.format(SANS)

    out = []
    for row in rows:
        emoji = PATTERN_STYLES.get(row["pattern"], "🔹")
        caution = ""
        if row["dq_notes"]:
            caution = (
                f'<div style="margin-top:3px;font-size:10.5px;color:#9a3412;">'
                f'⚠ {html.escape("; ".join(row["dq_notes"]))}</div>'
            )
        out.append(f"""
        <tr style="background:{row_bg};">
          <td style="padding:9px 12px;font-family:{SANS};font-size:13px;font-weight:700;color:#1F2430;border-bottom:1px solid #EDEAE2;">{html.escape(row['symbol'])}</td>
          <td style="padding:9px 12px;font-family:{SANS};font-size:12px;color:#3C4256;border-bottom:1px solid #EDEAE2;">{emoji} {html.escape(row['pattern'])}<div style="margin-top:2px;font-size:10.5px;color:#8A8F9C;">{html.escape(row['detail'])}</div>{caution}</td>
          <td style="padding:9px 12px;font-family:{SANS};font-size:12px;color:#3C4256;border-bottom:1px solid #EDEAE2;text-align:right;">₹{row['signal_price']:,.2f}</td>
          <td style="padding:9px 12px;font-family:{SANS};font-size:11.5px;color:#3C4256;border-bottom:1px solid #EDEAE2;">{_backtest_cell(row['backtest'])}</td>
        </tr>
        """)
    return "".join(out)


def _table_block(title, subtitle, rows, row_bg, accent):
    header = (
        f'<tr><td style="padding:6px 12px;font-family:{SANS};font-size:10px;font-weight:700;'
        f'color:#3C4256;border-bottom:1px solid #DAD5CB;">Symbol</td>'
        f'<td style="padding:6px 12px;font-family:{SANS};font-size:10px;font-weight:700;'
        f'color:#3C4256;border-bottom:1px solid #DAD5CB;">Pattern</td>'
        f'<td style="padding:6px 12px;font-family:{SANS};font-size:10px;font-weight:700;'
        f'color:#3C4256;border-bottom:1px solid #DAD5CB;text-align:right;">Price</td>'
        f'<td style="padding:6px 12px;font-family:{SANS};font-size:10px;font-weight:700;'
        f'color:#3C4256;border-bottom:1px solid #DAD5CB;">Backtest ({PRIMARY_HORIZON}d)</td></tr>'
    )
    rows_html = _signal_rows_html(rows, row_bg)
    return f"""
    <tr>
      <td style="padding:0 28px 18px;" class="email-padding">
        <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="border-radius:6px;background:#FAF8F1;border:1px solid #E7DFC9;">
          <tr>
            <td style="padding:14px 16px 4px;">
              <div style="font-family:{SANS};font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:{accent};">{title}</div>
              <div style="margin:2px 0 8px;font-family:{SANS};font-size:11px;color:#8A8F9C;">{subtitle}</div>
              <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="border-collapse:collapse;">
                {header}
                {rows_html}
              </table>
            </td>
          </tr>
        </table>
      </td>
    </tr>
    """


def build_report_html(confirmed, watch_list, scan_stats):
    now_ist = dt.datetime.now(ZoneInfo("Asia/Kolkata"))
    date_str = get_date_with_suffix(now_ist)

    confirmed_block = _table_block(
        "✅ Confirmed Breakouts",
        f"Backtest cleared the bar: ≥{MIN_SAMPLES_FOR_CONFIDENCE} past occurrences and ≥{CONFIRM_HIT_RATE_THRESHOLD*100:.0f}% {PRIMARY_HORIZON}-day hit-rate.",
        confirmed[:MAX_CONFIRMED_ROWS], "#ffffff", "#0f5132",
    )
    watch_block = _table_block(
        "👀 Unconfirmed / Watch List",
        "Pattern fired today but the historical sample was thin or the hit-rate was weak -- treat as a watch item, not a call.",
        watch_list[:MAX_WATCH_ROWS], "#ffffff", "#7a5b00",
    )

    universe_note = (
        f"Scanned {scan_stats['universe_size']} NIFTY 500 symbols "
        f"({'live index list' if scan_stats['universe_is_live'] else 'fallback core list -- live NSE index fetch failed this run'}); "
        f"{scan_stats['history_count']} returned usable price history; "
        f"{scan_stats['skipped_count']} skipped by the data-quality gate."
    )
    bhav_note = (
        f"Same-day NSE bhavcopy cross-check: active ({scan_stats['bhav_date']})."
        if scan_stats["bhav_is_live"]
        else "Same-day NSE bhavcopy cross-check: unavailable this run (NSE fetch failed) -- signals rely on yfinance data only."
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="color-scheme" content="light">
<title>Daily Breakout Screener</title>
<style>
  body {{ margin:0; padding:0; background:#F2F0EC; }}
  table {{ border-collapse:collapse !important; }}
  @media screen and (max-width:600px) {{
    .email-container {{ width:100% !important; max-width:100% !important; }}
    .email-padding {{ padding-left:14px !important; padding-right:14px !important; }}
  }}
</style>
</head>
<body style="margin:0;padding:0;background:#F2F0EC;font-family:{SERIF};color:#1B2233;">
  <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="background:#F2F0EC;width:100%;">
    <tr>
      <td align="center" style="padding:20px 16px;" class="email-padding">
        <table width="100%" cellpadding="0" cellspacing="0" role="presentation" class="email-container" style="max-width:680px;min-width:280px;background:#ffffff;border:1px solid #DAD5CB;border-radius:4px;overflow:hidden;">
          <tr>
            <td style="background:#14213D;padding:26px 28px 22px;" class="email-padding">
              <div style="font-family:{SANS};font-size:10px;font-weight:700;letter-spacing:0.16em;text-transform:uppercase;color:#B08D57;">Equities &nbsp;&bull;&nbsp; Daily Screener</div>
              <h1 style="margin:8px 0 0;font-family:{SERIF};font-size:22px;font-weight:400;line-height:1.3;color:#ffffff;">Daily Breakout Screener &mdash; NIFTY 500</h1>
            </td>
          </tr>
          <tr><td style="height:3px;line-height:3px;font-size:0;background:linear-gradient(90deg,#B08D57,#D9C393 45%,#B08D57);">&nbsp;</td></tr>
          <tr>
            <td style="padding:14px 28px 4px;" class="email-padding">
              <p style="margin:0;font-family:{SANS};font-size:12px;color:#8A8F9C;">{date_str}</p>
            </td>
          </tr>
          <tr>
            <td style="padding:6px 28px 16px;border-bottom:1px solid #EDEAE2;" class="email-padding">
              <p style="margin:0;font-family:{SANS};font-size:12px;color:#4A5063;">{universe_note}</p>
              <p style="margin:4px 0 0;font-family:{SANS};font-size:11px;color:#8A8F9C;">{bhav_note}</p>
            </td>
          </tr>
          {confirmed_block}
          {watch_block}
          <tr>
            <td style="padding:16px 28px;border-top:1px solid #EDEAE2;" class="email-padding">
              <p style="margin:0;font-family:{SANS};font-size:10px;color:#8A8F9C;line-height:1.5;">
                Every signal above is mechanical (price/volume/pattern rules only, no AI narrative) and every
                "Confirmed" signal has been backtested against that stock's own trailing history at run time --
                see the Backtest column. Past pattern performance does not guarantee future results.
                Informational only -- not investment advice. Verify before acting.
              </p>
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>
"""


def send_report_email(report_html):
    now_ist = dt.datetime.now(ZoneInfo("Asia/Kolkata"))
    subject = f"Daily Breakout Screener - {get_date_with_suffix(now_ist)}"
    return email_service.send_email(subject=subject, html_body=report_html)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main(dry_run=False):
    log.info("Breakout Screener: starting run...")

    symbols, universe_is_live = get_nifty500_symbols()
    bhav_df, bhav_date, bhav_is_live = get_bhavcopy()

    histories = fetch_universe_history(symbols)
    confirmed, watch_list, skipped_count = scan_universe(histories, bhav_df)

    scan_stats = {
        "universe_size": len(symbols),
        "universe_is_live": universe_is_live,
        "history_count": len(histories),
        "skipped_count": skipped_count,
        "bhav_is_live": bhav_is_live,
        "bhav_date": bhav_date.strftime("%d %b %Y") if bhav_date else None,
    }
    log.info(
        f"Breakout Screener: {len(confirmed)} confirmed, {len(watch_list)} watch-list signals "
        f"out of {len(histories)} symbols scanned ({skipped_count} skipped on data quality)."
    )

    report_html = build_report_html(confirmed, watch_list, scan_stats)

    if dry_run:
        out_path = "breakout_report_preview.html"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report_html)
        log.info(f"Dry run -- wrote preview to {out_path} instead of sending email.")
        return

    success = send_report_email(report_html)
    if success:
        log.info("Breakout Screener: email sent successfully.")
    else:
        log.error("Breakout Screener: failed to send email.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan NIFTY 500 for chart-pattern breakouts and email the report.")
    parser.add_argument("--dry-run", action="store_true", help="Write the report to a local HTML file instead of emailing it.")
    args = parser.parse_args()
    main(dry_run=args.dry_run)