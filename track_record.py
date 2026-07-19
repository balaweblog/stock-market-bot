"""
Track-record bookkeeping for the equity report.

Every run already recomputes target/stop-loss fresh from the CURRENT price
(see position_sizing.apply_risk_management), so those numbers drift run to
run and can't be compared to each other directly. To know whether a call
actually "worked", we have to freeze the target/stop at the moment a
directional call (STRONG BUY / BUY / HOLD) first appears, then watch
later runs' prices against that frozen pair until one is hit or the
signal changes -- at which point the call is "closed" and logged.

This module owns that state machine plus the HTML panel that reports the
resulting hit-rate to the reader. It's intentionally separate from the
existing prior/summary "stocks" history in main.py (which only ever
remembers the single most recent run and is used for signal-change /
breach badges) -- this module accumulates a real history across many runs.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

SANS = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
SERIF = "Georgia,'Times New Roman',serif"

# Only these signals represent an actual directional trade recommendation
# with a real target/stop (see position_sizing.apply_risk_management) --
# "SELL" gets size=0 and target==entry, so it isn't a call worth scoring.
TRACKED_SIGNALS = {"STRONG BUY", "BUY / HOLD"}

MAX_CLOSED_HISTORY = 200


def _now_iso():
    return datetime.now(ZoneInfo("Asia/Kolkata")).isoformat()


def process_ticker_track_record(ticker, stock_name, signal, current_price, target, stop_loss, prior_open):
    """
    Advances the state machine for one ticker by one run.

    prior_open: the existing open-call dict for this ticker (or None),
        as previously returned by this same function.

    Returns (new_open_or_None, closed_record_or_None).
    """
    closed = None
    working_open = prior_open

    if working_open and current_price is not None:
        frozen_target = working_open.get("frozen_target")
        frozen_stop = working_open.get("frozen_stop_loss")
        entry_price = working_open.get("entry_price")

        outcome = None
        if frozen_target is not None and current_price >= frozen_target:
            outcome = "Target Hit"
        elif frozen_stop is not None and current_price <= frozen_stop:
            outcome = "Stopped Out"
        elif signal != working_open.get("signal"):
            outcome = "Signal Changed"

        if outcome:
            return_pct = None
            if entry_price:
                return_pct = round(((current_price - entry_price) / entry_price) * 100, 2)
            closed = {
                "ticker": ticker,
                "stock_name": stock_name,
                "signal": working_open.get("signal"),
                "entry_price": entry_price,
                "opened_at": working_open.get("opened_at"),
                "target": frozen_target,
                "stop_loss": frozen_stop,
                "exit_price": current_price,
                "closed_at": _now_iso(),
                "outcome": outcome,
                "return_pct": return_pct,
            }
            working_open = None

    new_open = working_open
    if signal in TRACKED_SIGNALS and new_open is None:
        new_open = {
            "ticker": ticker,
            "stock_name": stock_name,
            "signal": signal,
            "entry_price": current_price,
            "opened_at": _now_iso(),
            "frozen_target": target,
            "frozen_stop_loss": stop_loss,
        }
    elif signal not in TRACKED_SIGNALS and new_open is prior_open:
        # Signal is no longer directional (e.g. dropped to SELL/HOLD-adjacent)
        # and nothing closed it above -- stop tracking it as an open call
        # rather than leaving a stale entry with no exit condition.
        new_open = None

    return new_open, closed


def update_track_record(track_record_state, ticker, stock_name, signal, current_price, target, stop_loss):
    """
    Convenience wrapper: mutates and returns track_record_state
    ({"open": {...}, "closed": [...]}) for one ticker in one call.
    """
    prior_open = (track_record_state.get("open") or {}).get(ticker)
    new_open, closed = process_ticker_track_record(
        ticker, stock_name, signal, current_price, target, stop_loss, prior_open
    )

    open_map = track_record_state.setdefault("open", {})
    if new_open:
        open_map[ticker] = new_open
    else:
        open_map.pop(ticker, None)

    if closed:
        closed_list = track_record_state.setdefault("closed", [])
        closed_list.insert(0, closed)
        del closed_list[MAX_CLOSED_HISTORY:]

    return track_record_state


def carry_forward_ticker(track_record_state, ticker):
    """Called when a stock's fetch failed this run -- keeps its open call
    alive rather than silently dropping it because of a data outage."""
    return  # open_map already has it; nothing to do, kept for symmetry/clarity


def _stat_row(label, value, color="#0f172a"):
    return (
        f'<td style="padding:10px 14px;text-align:center;">'
        f'<div style="font-family:{SANS};font-size:10px;text-transform:uppercase;'
        f'letter-spacing:0.06em;color:#8A8F9C;">{label}</div>'
        f'<div style="margin-top:4px;font-family:{SERIF};font-size:20px;color:{color};">{value}</div>'
        f'</td>'
    )


def build_track_record_html(track_record_state, recent_n=8):
    """
    Renders the accountability panel: win-rate stats derived from closed
    calls, plus a short table of the most recent closed calls. Returns ""
    if there isn't at least one closed call yet (a brand-new deployment
    has nothing to show, and an empty/zero panel would just look broken).
    """
    closed = (track_record_state or {}).get("closed") or []
    if not closed:
        return ""

    wins = sum(1 for c in closed if c["outcome"] == "Target Hit")
    losses = sum(1 for c in closed if c["outcome"] == "Stopped Out")
    decided = wins + losses  # "Signal Changed" exits are excluded from win-rate
    win_rate = round((wins / decided) * 100, 1) if decided else None

    returns = [c["return_pct"] for c in closed if c.get("return_pct") is not None]
    avg_return = round(sum(returns) / len(returns), 2) if returns else None

    win_rate_color = "#16a34a" if (win_rate is not None and win_rate >= 50) else "#dc2626"
    avg_return_color = "#16a34a" if (avg_return is not None and avg_return >= 0) else "#dc2626"

    stats_html = "".join([
        _stat_row("Closed Calls", len(closed)),
        _stat_row("Win Rate", f"{win_rate}%" if win_rate is not None else "n/a", win_rate_color),
        _stat_row("Avg Return / Call", f"{avg_return:+.2f}%" if avg_return is not None else "n/a", avg_return_color),
        _stat_row("Open Calls", len((track_record_state or {}).get("open") or {})),
    ])

    def outcome_style(outcome):
        if outcome == "Target Hit":
            return "#16a34a", "🎯"
        if outcome == "Stopped Out":
            return "#dc2626", "🛑"
        return "#8A8F9C", "↔"

    rows_html = ""
    for c in closed[:recent_n]:
        color, icon = outcome_style(c["outcome"])
        ret = c.get("return_pct")
        ret_text = f"{ret:+.2f}%" if ret is not None else "n/a"
        closed_date = (c.get("closed_at") or "")[:10]
        rows_html += (
            '<tr>'
            f'<td style="padding:6px 10px;border-top:1px solid #EDEAE2;font-family:{SANS};font-size:12px;color:#14213D;">{c.get("stock_name") or c.get("ticker")}</td>'
            f'<td style="padding:6px 10px;border-top:1px solid #EDEAE2;font-family:{SANS};font-size:12px;color:#4A5063;">{c.get("signal")}</td>'
            f'<td style="padding:6px 10px;border-top:1px solid #EDEAE2;font-family:{SANS};font-size:12px;color:#4A5063;">{c.get("entry_price")} &rarr; {c.get("exit_price")}</td>'
            f'<td style="padding:6px 10px;border-top:1px solid #EDEAE2;font-family:{SANS};font-size:12px;font-weight:700;color:{color};">{icon} {c["outcome"]}</td>'
            f'<td style="padding:6px 10px;border-top:1px solid #EDEAE2;font-family:{SANS};font-size:12px;color:{color};">{ret_text}</td>'
            f'<td style="padding:6px 10px;border-top:1px solid #EDEAE2;font-family:{SANS};font-size:11px;color:#8A8F9C;">{closed_date}</td>'
            '</tr>'
        )

    return f"""
    <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="margin:14px 20px;border:1px solid #E7E4DC;border-radius:4px;overflow:hidden;">
      <tr>
        <td style="background:#14213D;padding:9px 14px;">
          <span style="font-family:{SANS};font-size:12px;font-weight:700;color:#ffffff;">Track Record</span>
          <span style="font-family:{SANS};font-size:11px;color:#B7BEC9;margin-left:8px;">How past calls from this model actually played out</span>
        </td>
      </tr>
      <tr>
        <td>
          <table width="100%" cellpadding="0" cellspacing="0" role="presentation"><tr>{stats_html}</tr></table>
        </td>
      </tr>
      <tr>
        <td style="padding:0 4px 8px;">
          <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="border-collapse:collapse;">
            <tr>
              <td style="padding:6px 10px;font-family:{SANS};font-size:10px;text-transform:uppercase;letter-spacing:0.05em;color:#8A8F9C;">Stock</td>
              <td style="padding:6px 10px;font-family:{SANS};font-size:10px;text-transform:uppercase;letter-spacing:0.05em;color:#8A8F9C;">Call</td>
              <td style="padding:6px 10px;font-family:{SANS};font-size:10px;text-transform:uppercase;letter-spacing:0.05em;color:#8A8F9C;">Entry &rarr; Exit</td>
              <td style="padding:6px 10px;font-family:{SANS};font-size:10px;text-transform:uppercase;letter-spacing:0.05em;color:#8A8F9C;">Outcome</td>
              <td style="padding:6px 10px;font-family:{SANS};font-size:10px;text-transform:uppercase;letter-spacing:0.05em;color:#8A8F9C;">Return</td>
              <td style="padding:6px 10px;font-family:{SANS};font-size:10px;text-transform:uppercase;letter-spacing:0.05em;color:#8A8F9C;">Closed</td>
            </tr>
            {rows_html}
          </table>
        </td>
      </tr>
    </table>
    """