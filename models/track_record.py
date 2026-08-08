"""
Track-record bookkeeping for the equity report.

Every run already recomputes target/stop-loss fresh from the CURRENT price
(see position_sizing.apply_risk_management), so those numbers drift run to
run and can't be compared to each other directly. To know whether a call
actually "worked", we have to freeze the target/stop (and, now, the
horizon/probability/benchmark) at the moment a directional call first
appears, then watch later runs' prices against that frozen snapshot until
one is hit or the signal changes -- at which point the call is "closed"
and logged.

This module owns that state machine plus the HTML panels that report the
resulting hit-rate / "Model Performance" numbers to the reader. It's
intentionally separate from the existing prior/summary "stocks" history in
main.py (which only ever remembers the single most recent run and is used
for signal-change / breach badges) -- this module accumulates a real
history across many runs.

v2 additions over the original BUY-only tracker:
  - SELL-side tracking (a directional call is scored whether it says BUY
    or SELL, not just BUY) via a decline/rise threshold instead of a
    target/stop (SELL rows don't get a real target -- see
    TRACKED_SELL_SIGNALS below).
  - horizon_days / probability_pct frozen alongside target/stop, so a
    closed call can report "Horizon: 3 months, Probability: 62%" the way
    a real prediction log would.
  - Optional benchmark tracking (index price frozen at open, compared at
    close) so a closed call's return can be reported alongside its Alpha
    versus Nifty/S&P, not just its raw return.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

SANS = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
SERIF = "Georgia,'Times New Roman',serif"

# Only these signals represent an actual directional trade recommendation
# with a real target/stop (see position_sizing.apply_risk_management).
TRACKED_BUY_SIGNALS = {"STRONG BUY", "BUY / HOLD"}

# SELL-side calls don't get a real target/stop (SELL gets size=0 and
# target==entry in position_sizing), so they're scored differently: did
# the price actually decline after the call, rather than "did it hit a
# frozen target". SELL_DECLINE_CONFIRM_PCT / SELL_RISE_INVALIDATE_PCT
# below are the two outcome thresholds for that.
TRACKED_SELL_SIGNALS = {"SELL", "STRONG SELL", "SELL / REDUCE"}

# A SELL call is scored "Decline Confirmed" once price is down this much
# from the call price, and "Call Invalidated" if price instead rises this
# much -- mirrors a target/stop pair without needing a real short position.
SELL_DECLINE_CONFIRM_PCT = 5.0
SELL_RISE_INVALIDATE_PCT = 5.0

MAX_CLOSED_HISTORY = 200

# Horizon bucket (in days) used for the "N-month hit rate" stat -- calls
# are grouped into the nearest bucket below by their frozen horizon_days.
HORIZON_BUCKETS_DAYS = [30, 90, 180]


def _now_iso():
    return datetime.now(ZoneInfo("Asia/Kolkata")).isoformat()


def _is_buy_signal(signal):
    return signal in TRACKED_BUY_SIGNALS


def _is_sell_signal(signal):
    return signal in TRACKED_SELL_SIGNALS


def process_ticker_track_record(ticker, stock_name, signal, current_price, target, stop_loss,
                                 prior_open, horizon_days=None, probability_pct=None,
                                 benchmark_ticker=None, benchmark_price=None):
    """
    Advances the state machine for one ticker by one run.

    prior_open: the existing open-call dict for this ticker (or None),
        as previously returned by this same function.
    horizon_days / probability_pct: frozen onto the call when it's first
        opened (ignored on later runs of an already-open call -- the
        prediction doesn't get to move its own goalposts).
    benchmark_ticker / benchmark_price: the index (e.g. "^NSEI"/"^GSPC")
        and its current price, used to freeze a benchmark entry price at
        open and compute alpha at close. Optional -- a call opened without
        one just reports no alpha.

    Returns (new_open_or_None, closed_record_or_None).
    """
    closed = None
    working_open = prior_open
    is_buy = _is_buy_signal(signal)
    is_sell = _is_sell_signal(signal)

    if working_open and current_price is not None:
        call_side = working_open.get("side", "buy")
        outcome = None

        if call_side == "buy":
            frozen_target = working_open.get("frozen_target")
            frozen_stop = working_open.get("frozen_stop_loss")
            if frozen_target is not None and current_price >= frozen_target:
                outcome = "Target Hit"
            elif frozen_stop is not None and current_price <= frozen_stop:
                outcome = "Stopped Out"
            elif signal != working_open.get("signal"):
                outcome = "Signal Changed"
        else:  # call_side == "sell"
            entry_price = working_open.get("entry_price")
            if entry_price:
                move_pct = ((current_price - entry_price) / entry_price) * 100
                if move_pct <= -SELL_DECLINE_CONFIRM_PCT:
                    outcome = "Decline Confirmed"
                elif move_pct >= SELL_RISE_INVALIDATE_PCT:
                    outcome = "Call Invalidated"
            if outcome is None and signal != working_open.get("signal"):
                outcome = "Signal Changed"

        if outcome:
            entry_price = working_open.get("entry_price")
            return_pct = None
            if entry_price:
                raw_pct = ((current_price - entry_price) / entry_price) * 100
                # For a SELL call, a genuine decline is the "win" -- flip
                # the sign so return_pct is positive for a correct call on
                # either side, matching how Alpha/win-rate below read it.
                return_pct = round(-raw_pct if call_side == "sell" else raw_pct, 2)

            benchmark_entry_price = working_open.get("benchmark_entry_price")
            benchmark_return_pct = None
            alpha_pct = None
            if benchmark_entry_price and benchmark_price:
                benchmark_return_pct = round(
                    ((benchmark_price - benchmark_entry_price) / benchmark_entry_price) * 100, 2
                )
                if return_pct is not None:
                    bench_signed = -benchmark_return_pct if call_side == "sell" else benchmark_return_pct
                    alpha_pct = round(return_pct - bench_signed, 2)

            closed = {
                "ticker": ticker,
                "stock_name": stock_name,
                "signal": working_open.get("signal"),
                "side": call_side,
                "entry_price": entry_price,
                "opened_at": working_open.get("opened_at"),
                "target": working_open.get("frozen_target"),
                "stop_loss": working_open.get("frozen_stop_loss"),
                "horizon_days": working_open.get("horizon_days"),
                "probability_pct": working_open.get("probability_pct"),
                "exit_price": current_price,
                "closed_at": _now_iso(),
                "outcome": outcome,
                "return_pct": return_pct,
                "benchmark_ticker": working_open.get("benchmark_ticker"),
                "benchmark_entry_price": benchmark_entry_price,
                "benchmark_exit_price": benchmark_price,
                "benchmark_return_pct": benchmark_return_pct,
                "alpha_pct": alpha_pct,
            }
            working_open = None

    new_open = working_open
    if new_open is None and (is_buy or is_sell):
        new_open = {
            "ticker": ticker,
            "stock_name": stock_name,
            "signal": signal,
            "side": "buy" if is_buy else "sell",
            "entry_price": current_price,
            "opened_at": _now_iso(),
            "frozen_target": target if is_buy else None,
            "frozen_stop_loss": stop_loss if is_buy else None,
            "horizon_days": horizon_days,
            "probability_pct": probability_pct,
            "benchmark_ticker": benchmark_ticker,
            "benchmark_entry_price": benchmark_price,
        }
    elif new_open is prior_open and not is_buy and not is_sell:
        # Signal is no longer directional and nothing closed it above --
        # stop tracking it as an open call rather than leaving a stale
        # entry with no exit condition.
        new_open = None

    return new_open, closed


def update_track_record(track_record_state, ticker, stock_name, signal, current_price, target, stop_loss,
                         horizon_days=None, probability_pct=None, benchmark_ticker=None, benchmark_price=None):
    """
    Convenience wrapper: mutates and returns track_record_state
    ({"open": {...}, "closed": [...]}) for one ticker in one call.
    """
    prior_open = (track_record_state.get("open") or {}).get(ticker)
    new_open, closed = process_ticker_track_record(
        ticker, stock_name, signal, current_price, target, stop_loss, prior_open,
        horizon_days=horizon_days, probability_pct=probability_pct,
        benchmark_ticker=benchmark_ticker, benchmark_price=benchmark_price,
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


def _nearest_horizon_bucket(horizon_days):
    if horizon_days is None:
        return None
    best = min(HORIZON_BUCKETS_DAYS, key=lambda b: abs(b - horizon_days))
    return best


def compute_model_performance(track_record_state):
    """
    Rolls the closed-call log into the "Model Performance" numbers: per-side
    accuracy, target/stop hit rates, average winner/loser, realized R:R,
    average alpha versus benchmark, and a hit rate for the most common
    horizon bucket. Every stat is None (not 0/crash) when there isn't
    enough closed history to compute it honestly.
    """
    closed = (track_record_state or {}).get("closed") or []

    buy_closed = [c for c in closed if c.get("side", "buy") == "buy"]
    sell_closed = [c for c in closed if c.get("side") == "sell"]

    def _rate(subset, win_outcomes, decided_outcomes):
        decided = [c for c in subset if c["outcome"] in decided_outcomes]
        wins = [c for c in decided if c["outcome"] in win_outcomes]
        return round((len(wins) / len(decided)) * 100, 1) if decided else None

    # "Accuracy" is the broader, directional question -- across EVERY
    # closed call on that side (including ones exited early on a
    # "Signal Changed", not just clean Target-Hit/Stopped-Out closes), did
    # the stock actually end up moving in the called direction (return_pct
    # positive; return_pct is already sign-flipped for sell calls so
    # positive always means "the call was right")? This is deliberately a
    # different, broader question than target_hit_rate/stop_loss_hit_rate
    # below, which only look at the subset that ran all the way to a
    # target/stop and says nothing about calls exited early.
    def _accuracy(subset):
        decided = [c for c in subset if c.get("return_pct") is not None]
        wins = [c for c in decided if c["return_pct"] > 0]
        return round((len(wins) / len(decided)) * 100, 1) if decided else None

    buy_accuracy = _accuracy(buy_closed)
    sell_accuracy = _accuracy(sell_closed)

    buy_decided = [c for c in buy_closed if c["outcome"] in {"Target Hit", "Stopped Out"}]
    target_hit_rate = _rate(buy_closed, {"Target Hit"}, {"Target Hit", "Stopped Out"})
    stop_loss_hit_rate = round(
        (sum(1 for c in buy_decided if c["outcome"] == "Stopped Out") / len(buy_decided)) * 100, 1
    ) if buy_decided else None

    winners = [c["return_pct"] for c in closed if c["outcome"] in {"Target Hit", "Decline Confirmed"} and c.get("return_pct") is not None]
    losers = [c["return_pct"] for c in closed if c["outcome"] in {"Stopped Out", "Call Invalidated"} and c.get("return_pct") is not None]
    avg_winner = round(sum(winners) / len(winners), 2) if winners else None
    avg_loser = round(sum(losers) / len(losers), 2) if losers else None
    avg_rr_realized = round(abs(avg_winner / avg_loser), 2) if (avg_winner and avg_loser) else None

    alphas = [c["alpha_pct"] for c in closed if c.get("alpha_pct") is not None]
    avg_alpha = round(sum(alphas) / len(alphas), 2) if alphas else None

    # Horizon-bucketed hit rate: bucket every closed, decided call by its
    # nearest horizon bucket, then report the hit rate for whichever
    # bucket has the most closed calls (usually the report's default
    # horizon) -- reporting all buckets with too few calls each would be
    # noisier than useful.
    horizon_hit_rate = None
    horizon_hit_rate_label = None
    horizon_groups = {}
    for c in closed:
        if c["outcome"] not in {"Target Hit", "Stopped Out", "Decline Confirmed", "Call Invalidated"}:
            continue
        bucket = _nearest_horizon_bucket(c.get("horizon_days"))
        if bucket is None:
            continue
        horizon_groups.setdefault(bucket, []).append(c)
    if horizon_groups:
        best_bucket = max(horizon_groups, key=lambda b: len(horizon_groups[b]))
        group = horizon_groups[best_bucket]
        wins = sum(1 for c in group if c["outcome"] in {"Target Hit", "Decline Confirmed"})
        horizon_hit_rate = round((wins / len(group)) * 100, 1)
        horizon_hit_rate_label = f"{best_bucket}-Day Hit Rate" if best_bucket != 90 else "3-Month Hit Rate"

    return {
        "closed_count": len(closed),
        "open_count": len((track_record_state or {}).get("open") or {}),
        "buy_accuracy": buy_accuracy,
        "sell_accuracy": sell_accuracy,
        "target_hit_rate": target_hit_rate,
        "stop_loss_hit_rate": stop_loss_hit_rate,
        "avg_winner": avg_winner,
        "avg_loser": avg_loser,
        "avg_rr_realized": avg_rr_realized,
        "avg_alpha": avg_alpha,
        "horizon_hit_rate": horizon_hit_rate,
        "horizon_hit_rate_label": horizon_hit_rate_label,
    }


def _stat_row(label, value, color="#0f172a"):
    return (
        f'<td style="padding:10px 14px;text-align:center;">'
        f'<div style="font-family:{SANS};font-size:10px;text-transform:uppercase;'
        f'letter-spacing:0.06em;color:#8A8F9C;">{label}</div>'
        f'<div style="margin-top:4px;font-family:{SERIF};font-size:20px;color:{color};">{value}</div>'
        f'</td>'
    )


def _pct_or_na(value):
    return f"{value}%" if value is not None else "n/a"


def _signed_pct_or_na(value):
    return f"{value:+.2f}%" if value is not None else "n/a"


def build_track_record_html(track_record_state, recent_n=8):
    """
    Renders the full accountability panel: a "Model Performance" stat grid
    derived from ALL closed calls (buy accuracy, sell accuracy, target/stop
    hit rates, average winner/loser, realized R:R, average alpha, and a
    horizon-bucketed hit rate), plus a table of the most recent closed
    calls with their frozen Horizon/Probability and realized Alpha. Returns
    "" if there isn't at least one closed call yet (a brand-new deployment
    has nothing to show, and an empty/zero panel would just look broken).
    """
    closed = (track_record_state or {}).get("closed") or []
    if not closed:
        return ""

    perf = compute_model_performance(track_record_state)

    def _color_for_rate(value):
        return "#16a34a" if (value is not None and value >= 50) else ("#dc2626" if value is not None else "#0f172a")

    def _color_for_signed(value):
        return "#16a34a" if (value is not None and value >= 0) else ("#dc2626" if value is not None else "#0f172a")

    stats_row_1 = "".join([
        _stat_row("Buy Accuracy", _pct_or_na(perf["buy_accuracy"]), _color_for_rate(perf["buy_accuracy"])),
        _stat_row("Sell Accuracy", _pct_or_na(perf["sell_accuracy"]), _color_for_rate(perf["sell_accuracy"])),
        _stat_row("Target Hit Rate", _pct_or_na(perf["target_hit_rate"]), _color_for_rate(perf["target_hit_rate"])),
        _stat_row("Stop-Loss Hit Rate", _pct_or_na(perf["stop_loss_hit_rate"]),
                  "#dc2626" if (perf["stop_loss_hit_rate"] or 0) >= 50 else "#0f172a"),
    ])
    stats_row_2 = "".join([
        _stat_row("Avg Winner", _signed_pct_or_na(perf["avg_winner"]), "#16a34a" if perf["avg_winner"] else "#0f172a"),
        _stat_row("Avg Loser", _signed_pct_or_na(perf["avg_loser"]), "#dc2626" if perf["avg_loser"] else "#0f172a"),
        _stat_row("Avg R/R Realized", f"{perf['avg_rr_realized']}" if perf["avg_rr_realized"] is not None else "n/a"),
        _stat_row("Avg Alpha", _signed_pct_or_na(perf["avg_alpha"]), _color_for_signed(perf["avg_alpha"])),
    ])
    stats_row_3 = "".join([
        _stat_row("Closed Calls", perf["closed_count"]),
        _stat_row("Open Calls", perf["open_count"]),
        _stat_row(perf["horizon_hit_rate_label"] or "Horizon Hit Rate", _pct_or_na(perf["horizon_hit_rate"]),
                  _color_for_rate(perf["horizon_hit_rate"])),
    ])

    def outcome_style(outcome):
        if outcome in ("Target Hit", "Decline Confirmed"):
            return "#16a34a", "🎯"
        if outcome in ("Stopped Out", "Call Invalidated"):
            return "#dc2626", "🛑"
        return "#8A8F9C", "↔"

    rows_html = ""
    for c in closed[:recent_n]:
        color, icon = outcome_style(c["outcome"])
        ret = c.get("return_pct")
        ret_text = f"{ret:+.2f}%" if ret is not None else "n/a"
        alpha = c.get("alpha_pct")
        alpha_text = f"{alpha:+.2f}%" if alpha is not None else "n/a"
        horizon = c.get("horizon_days")
        horizon_text = f"{horizon}d" if horizon is not None else "—"
        prob = c.get("probability_pct")
        prob_text = f"{prob}%" if prob is not None else "—"
        closed_date = (c.get("closed_at") or "")[:10]
        rows_html += (
            '<tr>'
            f'<td style="padding:6px 10px;border-top:1px solid #EDEAE2;font-family:{SANS};font-size:12px;color:#14213D;">{c.get("stock_name") or c.get("ticker")}</td>'
            f'<td style="padding:6px 10px;border-top:1px solid #EDEAE2;font-family:{SANS};font-size:12px;color:#4A5063;">{c.get("signal")}</td>'
            f'<td style="padding:6px 10px;border-top:1px solid #EDEAE2;font-family:{SANS};font-size:12px;color:#4A5063;">{c.get("entry_price")} &rarr; {c.get("exit_price")}</td>'
            f'<td style="padding:6px 10px;border-top:1px solid #EDEAE2;font-family:{SANS};font-size:12px;color:#4A5063;">{horizon_text} / {prob_text}</td>'
            f'<td style="padding:6px 10px;border-top:1px solid #EDEAE2;font-family:{SANS};font-size:12px;font-weight:700;color:{color};">{icon} {c["outcome"]}</td>'
            f'<td style="padding:6px 10px;border-top:1px solid #EDEAE2;font-family:{SANS};font-size:12px;color:{color};">{ret_text}</td>'
            f'<td style="padding:6px 10px;border-top:1px solid #EDEAE2;font-family:{SANS};font-size:12px;color:{"#16a34a" if (alpha or 0) >= 0 else "#dc2626"};">{alpha_text}</td>'
            f'<td style="padding:6px 10px;border-top:1px solid #EDEAE2;font-family:{SANS};font-size:11px;color:#8A8F9C;">{closed_date}</td>'
            '</tr>'
        )

    return f"""
    <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="margin:14px 20px;border:1px solid #E7E4DC;border-radius:4px;overflow:hidden;">
      <tr>
        <td style="background:#14213D;padding:9px 14px;">
          <span style="font-family:{SANS};font-size:12px;font-weight:700;color:#ffffff;">Model Performance</span>
          <span style="font-family:{SANS};font-size:11px;color:#B7BEC9;margin-left:8px;">Prediction vs Reality &mdash; how past calls from this model actually played out</span>
        </td>
      </tr>
      <tr>
        <td>
          <table width="100%" cellpadding="0" cellspacing="0" role="presentation"><tr>{stats_row_1}</tr></table>
          <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="border-top:1px solid #F4F2ED;"><tr>{stats_row_2}</tr></table>
          <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="border-top:1px solid #F4F2ED;"><tr>{stats_row_3}</tr></table>
        </td>
      </tr>
      <tr>
        <td style="padding:0 4px 8px;">
          <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="border-collapse:collapse;">
            <tr>
              <td style="padding:6px 10px;font-family:{SANS};font-size:10px;text-transform:uppercase;letter-spacing:0.05em;color:#8A8F9C;">Stock</td>
              <td style="padding:6px 10px;font-family:{SANS};font-size:10px;text-transform:uppercase;letter-spacing:0.05em;color:#8A8F9C;">Call</td>
              <td style="padding:6px 10px;font-family:{SANS};font-size:10px;text-transform:uppercase;letter-spacing:0.05em;color:#8A8F9C;">Entry &rarr; Exit</td>
              <td style="padding:6px 10px;font-family:{SANS};font-size:10px;text-transform:uppercase;letter-spacing:0.05em;color:#8A8F9C;">Horizon / Prob.</td>
              <td style="padding:6px 10px;font-family:{SANS};font-size:10px;text-transform:uppercase;letter-spacing:0.05em;color:#8A8F9C;">Outcome</td>
              <td style="padding:6px 10px;font-family:{SANS};font-size:10px;text-transform:uppercase;letter-spacing:0.05em;color:#8A8F9C;">Return</td>
              <td style="padding:6px 10px;font-family:{SANS};font-size:10px;text-transform:uppercase;letter-spacing:0.05em;color:#8A8F9C;">Alpha</td>
              <td style="padding:6px 10px;font-family:{SANS};font-size:10px;text-transform:uppercase;letter-spacing:0.05em;color:#8A8F9C;">Closed</td>
            </tr>
            {rows_html}
          </table>
        </td>
      </tr>
    </table>
    """