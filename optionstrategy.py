"""
optionstrategy.py

Standalone companion to main.py, structured the same way as
swing_trade_advisor.py. Runs a single "recommend the best risk-defined
Nifty options strategy across Weekly / Monthly / Quarterly / Annual
horizons" prompt against whichever free LLM backend main.py already knows
how to set up (Groq free tier -> Gemini 2.5 Flash free tier -> local
Qwen2.5-1.5B fallback), then emails the result to the same recipients
configured for the stock report (EMAIL_TO / EMAIL_CC in config.py / the
workflow yaml's env vars).

This deliberately does NOT re-implement the LLM-selection / live-search
fallback chain -- it reuses swing_trade_advisor.generate_analysis(), which
is already generic (it just takes a prompt string and returns
(text, sources, used_live_search)), so both scripts stay in sync with
whatever provider/search path is configured. Email credential plumbing is
reused from main.py the same way swing_trade_advisor.py does it.

LIVE DATA: this script does not call the NSE options-chain API directly --
there is no clean free/official one, and scraping it reliably is its own
project. Instead it leans on the LLM's own live-search path (see
swing_trade_advisor.generate_analysis -- groq/compound's built-in web
search, then Tavily+Groq, then Gemini+Google Search grounding) to pull
Nifty spot, India VIX, options-chain levels, FII/DII activity, GIFT Nifty
pre-market, and global cues at generation time. REQUIRE_LIVE_DATA=true
(the default) refuses to email a run that never actually reached live
data, exactly like swing_trade_advisor.py.

CAVEAT: this is not a verified real-time trading signal. Web search
results can be a few hours stale, incomplete, or misread by the model.
Every strike, level, and Greek-adjacent figure below is a starting
hypothesis to verify against your broker's live options chain (e.g. Kite
Connect / Upstox API, or NSE's site directly) before placing any order.
Not investment advice.
"""

import os
import re
import sys
import json
import html
import traceback
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

import smtplib
from email.mime.text import MIMEText

import main                 # LLM init + email config/credentials (via config.py)
import swing_trade_advisor as swing  # reuses generate_analysis() + helpers, no duplication

# -----------------------------
# Config (env-overridable capital caps, per the prompt's constraint #3)
# -----------------------------
PER_HORIZON_CAP_PCT = float(os.getenv("OPTIONS_PER_HORIZON_CAP_PCT", "5"))
AGGREGATE_CAP_PCT = float(os.getenv("OPTIONS_AGGREGATE_CAP_PCT", "15"))

HORIZON_ORDER = ["Weekly", "Monthly", "Quarterly", "Annual"]


def _market_session_label():
    """
    NSE regular trading session is 09:15-15:30 IST. This run is allowed to
    fire at any point (constraint #3 in the prompt), but the email should
    say plainly whether it was generated during a live session (options
    chain likely fresh) or outside one (options chain will reflect the
    last close, not live quotes) -- that materially changes how much to
    trust the strike levels below.
    """
    now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
    is_weekday = now_ist.weekday() < 5  # Mon-Fri
    in_session = is_weekday and dtime(9, 15) <= now_ist.time() <= dtime(15, 30)
    if in_session:
        return "Live NSE trading session", True
    return "Outside NSE trading hours (data reflects last close, not live quotes)", False


# -----------------------------
# Prompt
# -----------------------------
def build_prompt():
    today_str = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%d %B %Y")
    session_label, in_session = _market_session_label()

    return f"""Core Objective: You are an expert at analysing Nifty options-related data. Using the most current, 
    Rebuild Nifty options strategies with live market data. Use current spot, futures, IV, and OI levels.live market data available to you (via web search), recommend the best risk-defined strategy -- or combination of strategies -- for the current moment, independently across FOUR time horizons:

1. Nifty Weekly -- Buy, Sell, or a Combination of both (current/nearest weekly expiry)
2. Nifty Monthly -- Buy, Sell, or a Combination of both (current monthly expiry)
3. Nifty Quarterly -- Buy, Sell, or a Combination of both (nearest quarterly expiry on the Mar/Jun/Sep/Dec cycle)
4. Nifty Annual -- Buy, Sell, or a Combination of both (farthest available annual/LEAPS-style expiry)

Horizons are independent -- they do not need to agree with each other. For example Weekly can be a bearish credit spread while Monthly is a bullish debit spread, if the data supports it.

NON-NEGOTIABLE CONSTRAINTS

1. RISK-DEFINED ONLY. Every recommended strategy must have a mathematically capped maximum loss known before entry (e.g. Bull/Bear Call/Put Spreads, Iron Condors, Iron Butterflies, Calendar Spreads, Ratio Spreads with a protective wing, Debit/Credit Spreads). NEVER recommend a naked/undefined-risk position (naked short call, naked short put, uncovered short straddle/strangle) under any circumstance, for any horizon.

2. GAP PROTECTION. Every recommendation must explicitly address overnight/weekend gap risk:
   - State the max loss if Nifty gaps up or down beyond the current day's expected move (derive the expected move from India VIX for that horizon's time-to-expiry).
   - Prefer structures whose short strikes sit outside the current expected-move band (VIX-implied 1 standard deviation) for that horizon.
   - If global cues (GIFT Nifty pre-market, US markets overnight, Asian markets) suggest elevated gap risk right now, explicitly flag it and bias the recommendation toward tighter risk-defined structures or reduced size -- never toward removing a hedge leg.

3. CAPITAL PROTECTION RULES.
   - Max capital at risk per horizon must not exceed {PER_HORIZON_CAP_PCT:.0f}% of a hypothetical total options-trading capital pool.
   - Aggregate capital at risk across all four horizons combined must not exceed {AGGREGATE_CAP_PCT:.0f}%.
   - State the stop-loss / adjustment trigger for each recommendation (e.g. "exit if Nifty closes beyond X level" or "adjust if short strike is breached").

4. This is a stateless, any-time request -- today is {today_str}, and this run is being generated during: {session_label}. Produce a full, self-contained recommendation using only live data you can find right now; do not assume access to any prior recommendation.

REQUIRED LIVE DATA TO SEARCH FOR AND USE (do not fabricate any of this -- if you cannot find a real current figure, say so for that field rather than inventing one):
- Nifty 50 spot price (current)
- Nifty futures price (weekly/monthly, for basis)
- Options chain levels for the four relevant expiries: key strikes, OI concentration, IV, PCR by OI
- India VIX (current level + recent change) -- for expected-move sizing
- FII/DII cash + index-options net positioning (most recent session)
- GIFT Nifty pre-market level (if available) and prior-session US market close (S&P 500, Nasdaq), for gap risk
- Any major near-term event risk (RBI policy, US Fed meeting, major earnings cluster, budget/election dates) that falls within each horizon's window

OUTPUT FORMAT -- respond with ONLY raw JSON matching the schema below, and nothing else (no markdown, no code fences, no commentary before or after). Keep every field to plain text/numbers only (no HTML):

{{
  "horizons": [
    {{
      "horizon": "Weekly",
      "expiry_date": "The actual expiry date used, e.g. '24 Jul 2026'",
      "bias": "One of: Bullish / Bearish / Neutral / Range-bound",
      "bias_reason": "One sentence grounded in the live data you found",
      "strategy_name": "e.g. 'Bear Call Spread' or 'Iron Condor'",
      "legs": "Full leg structure, e.g. 'Sell 25000 CE, Buy 25200 CE (1 lot)'",
      "max_loss": "Rupee figure, e.g. '₹9,500 per lot'",
      "max_loss_pct_capital": "% of allocated capital for this horizon, must be <= {PER_HORIZON_CAP_PCT:.0f}",
      "max_profit": "Rupee figure",
      "max_profit_pct_capital": "% of allocated capital for this horizon",
      "breakeven": "Breakeven level(s)",
      "gap_risk": "Explicit statement of loss if Nifty gaps beyond the expected move before next session, per constraint #2",
      "adjustment_trigger": "The specific exit/adjust rule, per constraint #3",
      "confidence": "One of: High / Medium / Low",
      "data_status": "One of: 'live' (found real current data), 'partial' (some fields estimated), 'stale' (data may be several hours old or unavailable) -- be honest here"
    }},
    {{ "horizon": "Monthly", ... same fields ... }},
    {{ "horizon": "Quarterly", ... same fields ... }},
    {{ "horizon": "Annual", ... same fields ... }}
  ],
  "aggregate_capital_at_risk_pct": "Sum of all four horizons' max_loss_pct_capital, must be <= {AGGREGATE_CAP_PCT:.0f}",
  "portfolio_view": "One paragraph: are you net long or net short gamma across all four horizons combined, does the combined structure over- or under-hedge overall market gap risk, and does the aggregate capital at risk stay within the {AGGREGATE_CAP_PCT:.0f}% cap"
}}
"""


# -----------------------------
# Parsing (mirrors swing_trade_advisor._parse_analysis_json, different key)
# -----------------------------
def _parse_analysis_json(text):
    """Parses the JSON the LLM was asked to return. Models sometimes still
    wrap it in ```json fences or add stray text despite instructions --
    handle both. Returns (horizons_list, aggregate_pct, portfolio_view), or
    (None, None, None) if nothing usable could be parsed."""
    cleaned = swing._strip_code_fences(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return None, None, None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None, None, None
    if not isinstance(data, dict):
        return None, None, None
    horizons = data.get("horizons")
    if not horizons or not isinstance(horizons, list):
        return None, None, None
    return horizons, data.get("aggregate_capital_at_risk_pct"), data.get("portfolio_view")


# -----------------------------
# HTML rendering
# -----------------------------
_BIAS_STYLE = {
    "bullish": ("#2F5233", "#E7EEE4"),
    "bearish": ("#8B2E2E", "#FBEAEA"),
    "neutral": ("#8A6D3B", "#F3ECDD"),
    "range-bound": ("#8A6D3B", "#F3ECDD"),
    "range bound": ("#8A6D3B", "#F3ECDD"),
}

_CONFIDENCE_STYLE = {
    "high": ("#2F5233", "#E7EEE4"),
    "medium": ("#A6812F", "#FDF3D9"),
    "low": ("#8B2E2E", "#FBEAEA"),
}

_DATA_STATUS_STYLE = {
    "live": ("#2F5233", "#E7EEE4", "Live"),
    "partial": ("#A6812F", "#FDF3D9", "Partial"),
    "stale": ("#8B2E2E", "#FBEAEA", "Stale"),
}


def _esc(v):
    v = "" if v is None else str(v).strip()
    return html.escape(v) if v else "—"


def _badge(text, color, bg):
    return (
        f'<span style="display:inline-block;padding:3px 10px;border-radius:3px;'
        f'font-size:11px;font-weight:700;color:{color};background:{bg};">{html.escape(str(text))}</span>'
    )


def _bias_badge(bias):
    key = str(bias or "").strip().lower()
    color, bg = _BIAS_STYLE.get(key, ("#8A8F9C", "#F4F2ED"))
    return _badge(bias or "—", color, bg)


def _confidence_badge(conf):
    key = str(conf or "").strip().lower()
    color, bg = _CONFIDENCE_STYLE.get(key, ("#8A8F9C", "#F4F2ED"))
    return _badge(conf or "—", color, bg)


def _data_status_badge(status):
    key = str(status or "").strip().lower()
    color, bg, label = _DATA_STATUS_STYLE.get(key, ("#8A8F9C", "#F4F2ED", status or "—"))
    return _badge(label, color, bg)


def _horizon_card_html(h, sans, serif):
    """One risk-defined-strategy card per horizon (Weekly/Monthly/
    Quarterly/Annual), styled consistently with the swing-trade note's
    per-stock parameter tables."""
    name = _esc(h.get("horizon"))
    expiry = _esc(h.get("expiry_date"))

    def row(label, key, value_color="#14213D", bold=False):
        weight = "font-weight:700;" if bold else ""
        return (
            f'<tr><td style="padding:6px 10px;font-size:12px;font-family:{sans};'
            f'color:#4A5063;border-top:1px solid #EDEAE2;width:38%;">{label}</td>'
            f'<td style="padding:6px 10px;font-size:12px;{weight}font-family:{sans};'
            f'color:{value_color};border-top:1px solid #EDEAE2;">{_esc(h.get(key))}</td></tr>'
        )

    def raw_row(label, cell_html):
        return (
            f'<tr><td style="padding:6px 10px;font-size:12px;font-family:{sans};'
            f'color:#4A5063;border-top:1px solid #EDEAE2;width:38%;">{label}</td>'
            f'<td style="padding:6px 10px;font-size:12px;font-family:{sans};'
            f'color:#14213D;border-top:1px solid #EDEAE2;">{cell_html or "—"}</td></tr>'
        )

    rows = "".join([
        row("Strategy", "strategy_name", bold=True),
        row("Legs", "legs"),
        raw_row("Directional Bias", _bias_badge(h.get("bias"))),
        row("Bias Rationale", "bias_reason"),
        row("Max Loss", "max_loss", value_color="#8B2E2E", bold=True),
        row("Max Loss (% of horizon capital)", "max_loss_pct_capital", value_color="#8B2E2E"),
        row("Max Profit", "max_profit", value_color="#2F5233", bold=True),
        row("Max Profit (% of horizon capital)", "max_profit_pct_capital", value_color="#2F5233"),
        row("Breakeven", "breakeven"),
        row("Gap Risk", "gap_risk"),
        row("Adjustment / Exit Trigger", "adjustment_trigger"),
        raw_row("Confidence", _confidence_badge(h.get("confidence"))),
        raw_row("Data Freshness", _data_status_badge(h.get("data_status"))),
    ])

    return f"""
<div style="margin-top:18px;border:1px solid #E7E4DC;border-radius:4px;overflow:hidden;">
  <div style="background:#14213D;padding:9px 12px;">
    <span style="font-family:{sans};font-size:12px;font-weight:700;color:#ffffff;">{name}</span>
    <span style="font-family:{sans};font-size:11px;color:#B08D57;margin-left:8px;">Expiry: {expiry}</span>
  </div>
  <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="border-collapse:collapse;">
    {rows}
  </table>
</div>
"""


def render_horizons_html(horizons, aggregate_pct, portfolio_view):
    sans = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
    serif = "Georgia,'Times New Roman',serif"

    # Render in the fixed Weekly -> Monthly -> Quarterly -> Annual order
    # regardless of what order the model returned them in.
    by_name = {}
    for h in horizons:
        key = str(h.get("horizon") or "").strip()
        if key:
            by_name[key] = h

    cards = ""
    for label in HORIZON_ORDER:
        h = by_name.get(label)
        if h:
            cards += _horizon_card_html(h, sans, serif)
        else:
            cards += (
                f'<div style="margin-top:18px;padding:12px;border:1px solid #EDEAE2;'
                f'border-radius:4px;background:#F4F2ED;font-family:{sans};font-size:12px;'
                f'color:#8A8F9C;">{html.escape(label)}: not returned by the model this run.</div>'
            )

    agg_display = _esc(aggregate_pct)
    agg_color = "#14213D"
    try:
        if float(str(aggregate_pct).replace("%", "").strip()) > AGGREGATE_CAP_PCT:
            agg_color = "#8B2E2E"  # flag if the model's own math breached the cap
    except (TypeError, ValueError):
        pass

    portfolio_html = (
        f'<div style="margin-top:16px;padding:12px 14px;background:#FAF9F6;'
        f'border:1px solid #EDEAE2;border-left:3px solid #B08D57;border-radius:4px;">'
        f'<div style="font-family:{sans};font-size:11px;font-weight:700;color:#14213D;'
        f'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">Portfolio View '
        f'&nbsp;&middot;&nbsp; Aggregate Capital at Risk: '
        f'<span style="color:{agg_color};">{agg_display}%</span> '
        f'(cap {AGGREGATE_CAP_PCT:.0f}%)</div>'
        f'<div style="font-family:{sans};font-size:12px;color:#4A5063;line-height:1.65;">{_esc(portfolio_view)}</div>'
        f'</div>'
    )

    return cards + portfolio_html


# -----------------------------
# Email
# -----------------------------
def build_email_html(horizons_html, today_str, sources, used_live_search, session_label):
    if used_live_search:
        disclaimer = (
            "Generated using a live-web-search-capable model -- see \"Sources checked\" above for what it "
            "actually looked at. Options-chain levels, IV, and OI figures can still be a few minutes to "
            "hours stale, incomplete, or misread by the model. Verify every strike, premium, and Greek "
            "against your broker's live options chain before placing any order. Not investment advice."
        )
    else:
        disclaimer = (
            "Generated by an LLM with no live market/internet access for this run -- every price, strike, "
            "IV, and OI figure above is model output and is NOT verified against a live feed. Do not trade "
            "off this run; confirm every figure against a real-time options chain first. Not investment advice."
        )

    sources_html = swing._build_sources_html(sources)
    sans = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
    serif = "Georgia,'Times New Roman',serif"
    live_tag = (
        '<span style="color:#B08D57;">&nbsp;&middot;&nbsp; Live web search used</span>'
        if used_live_search else ""
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Nifty Options Strategy Note</title>
<style>
  body {{ margin:0; padding:0; background:#F2F0EC; }}
  table {{ border-collapse:collapse !important; }}
  @media screen and (max-width:600px) {{
    .email-container {{ width:100% !important; max-width:100% !important; border-radius:0 !important; }}
    .email-padding {{ padding-left:16px !important; padding-right:16px !important; }}
  }}
</style>
</head>
<body style="margin:0;padding:0;background:#F2F0EC;font-family:{serif};color:#1B2233;">
  <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="background:#F2F0EC;width:100%;">
    <tr>
      <td align="center" style="padding:20px 16px;" class="email-padding">
        <table width="100%" cellpadding="0" cellspacing="0" role="presentation" class="email-container" style="max-width:680px;min-width:280px;background:#ffffff;border:1px solid #DAD5CB;border-radius:4px;overflow:hidden;">
          <tr>
            <td style="background:#14213D;padding:26px 28px 22px;" class="email-padding">
              <div style="font-family:{sans};font-size:10px;font-weight:700;letter-spacing:0.16em;text-transform:uppercase;color:#B08D57;">Market Intelligence &nbsp;&bull;&nbsp; Derivatives Desk</div>
              <h1 style="margin:8px 0 0;font-family:{serif};font-weight:400;font-size:23px;line-height:1.3;color:#ffffff;letter-spacing:0.01em;">Nifty Options Strategy Note</h1>
              <p style="margin:6px 0 0;font-family:{sans};font-size:12px;color:#B7BEC9;">Weekly &middot; Monthly &middot; Quarterly &middot; Annual &mdash; Risk-Defined Only</p>
            </td>
          </tr>
          <tr>
            <td style="height:3px;line-height:3px;font-size:0;background:linear-gradient(90deg,#B08D57,#D9C393 45%,#B08D57);">&nbsp;</td>
          </tr>
          <tr>
            <td style="padding:16px 28px 4px;" class="email-padding">
              <p style="margin:0;font-family:{sans};font-size:12px;color:#8A8F9C;">Prepared {today_str} at {datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%I:%M %p IST")} &nbsp;&middot;&nbsp; {session_label}{live_tag}</p>
            </td>
          </tr>
          <tr>
            <td style="padding:14px 28px 18px;" class="email-padding">
              {horizons_html}
              {sources_html}
            </td>
          </tr>
          <tr>
            <td style="padding:16px 28px 22px;border-top:1px solid #EDEAE2;" class="email-padding">
              <p style="margin:0;font-family:{sans};font-size:11px;line-height:1.6;color:#9AA0AC;">{disclaimer}</p>
              <p style="margin:10px 0 0;font-family:{sans};font-size:10px;letter-spacing:0.04em;color:#B9BEC7;">&copy; Portfolio Research Desk</p>
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>
"""


def send_option_strategy_email(html_body):
    if not all([main.EMAIL_FROM, main.EMAIL_PASSWORD, main.EMAIL_TO]):
        main.log.error(
            "Email credentials not found. Please set EMAIL_FROM, EMAIL_PASSWORD, "
            "and EMAIL_TO (the same env vars main.py uses)."
        )
        return False

    to_recipients = main.parse_email_list(main.EMAIL_TO)
    cc_recipients = main.parse_email_list(getattr(main, "EMAIL_CC", "") or "")

    if not to_recipients:
        main.log.error("No valid TO recipients found in EMAIL_TO.")
        return False

    now_ist = datetime.now(ZoneInfo("UTC")).astimezone(ZoneInfo("Asia/Kolkata"))
    time_str = now_ist.strftime("%I:%M %p IST")
    subject = f"Nifty Options Strategy Note — {main.get_date_with_suffix(now_ist)} · {time_str}"

    msg = MIMEText(html_body, "html")
    msg["Subject"] = subject
    msg["From"] = main.EMAIL_FROM
    msg["To"] = ", ".join(to_recipients)
    if cc_recipients:
        msg["Cc"] = ", ".join(cc_recipients)

    all_recipients = to_recipients + cc_recipients

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(main.EMAIL_FROM, main.EMAIL_PASSWORD)
        server.sendmail(main.EMAIL_FROM, all_recipients, msg.as_string())
        server.quit()
        main.log.info("Nifty options strategy email sent successfully.")
        return True
    except smtplib.SMTPAuthenticationError:
        main.log.error(
            "SMTP Authentication Error: check EMAIL_FROM/EMAIL_PASSWORD "
            "(use a Gmail App Password, not the account password)."
        )
    except Exception as e:
        main.log.error(f"Failed to send Nifty options strategy email: {e}")
        traceback.print_exc()
    return False


# -----------------------------
# Entry point
# -----------------------------
def run():
    today_str = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%d %B %Y")
    session_label, _in_session = _market_session_label()
    prompt = build_prompt()

    # Reuses swing_trade_advisor's Groq(compound/compound-mini) -> Tavily+Groq
    # -> Gemini+Search -> plain Groq -> local model chain unmodified -- this
    # function is generic over the prompt text, so nothing options-specific
    # needs duplicating here.
    analysis, sources, used_live_search = swing.generate_analysis(prompt)
    if not analysis:
        main.log.error(
            "No LLM backend produced output (no GROQ_API_KEY/GOOGLE_API_KEY set "
            "and local model unavailable/failed). Aborting without sending an email."
        )
        sys.exit(1)

    if not used_live_search and os.getenv("REQUIRE_LIVE_DATA", "true").lower() == "true":
        # Same rationale as swing_trade_advisor.py: without live search this
        # is pure training-data reasoning about option strikes/IV/OI, which
        # is exactly the stale, unverified output this run exists to avoid.
        main.log.error(
            "Live web search was not used for this run, so option-chain levels, "
            "IV, VIX, and OI figures would only reflect stale training-data -- "
            "not current strikes. Aborting without sending an email. Set "
            "REQUIRE_LIVE_DATA=false to override and allow a clearly-labeled "
            "stale-data email instead."
        )
        sys.exit(1)

    horizons, aggregate_pct, portfolio_view = _parse_analysis_json(analysis)
    if horizons:
        horizons_html = render_horizons_html(horizons, aggregate_pct, portfolio_view)
    else:
        main.log.error(
            "Could not parse JSON from LLM output; falling back to raw text display."
        )
        sans = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
        horizons_html = (
            f'<div style="font-family:{sans};font-size:12px;color:#8B2E2E;margin-bottom:8px;">'
            f"Note: the model's response could not be parsed as structured data; showing raw output below.</div>"
            f'<pre style="white-space:pre-wrap;font-family:{sans};font-size:12px;color:#14213D;">{html.escape(swing._strip_code_fences(analysis))}</pre>'
        )

    email_html = build_email_html(horizons_html, today_str, sources, used_live_search, session_label)

    if os.getenv("DRY_RUN", "false").lower() == "true":
        with open("option_strategy_report.html", "w") as f:
            f.write(email_html)
        main.log.info("DRY_RUN enabled -- wrote option_strategy_report.html instead of emailing.")
        return

    send_option_strategy_email(email_html)


if __name__ == "__main__":
    run()

# -----------------------------
# Real-time grounding status -- see swing_trade_advisor.py's own footer
# comment block for the full detail on each fallback tier (groq/compound,
# groq/compound-mini, Tavily+plain-Groq, Gemini+Google Search grounding,
# plain Groq, local Qwen2.5-1.5B). This module reuses that chain as-is via
# swing.generate_analysis(prompt) rather than re-implementing it.
# -----------------------------