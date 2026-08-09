"""
wealth_controller.py

Monthly SIP / wealth-building portfolio report. Separate from
stock_controller.py's daily equity briefing -- this one runs once a
month, looks at the FULL recurring-contribution picture (PPF, mutual
fund SIPs, NPS, direct equity SIPs, gold, silver, EPF, US stocks) in
one place, and asks the LLM for a short "keep / increase / reduce /
exit / at risk" read per instrument plus a few portfolio-level
diversification notes and suggestions for next month.

Deliberately kept simple:
  - One flat table (instrument, category, monthly amount, AI verdict,
    reason) -- no scoring engine, no charts, no PDF.
  - One short "Portfolio Take" block underneath (headline + a couple of
    diversification notes + a couple of suggestions).
  - Uses llm_backend.generate_analysis() exactly like
    stock_controller.generate_ai_stocks_story() -- same live-search-first
    fallback chain, same fail-soft behaviour, no new AI wiring.

Data source: utils/constants.py -> SIP_PORTFOLIO (edit that list to add,
remove, or re-amount an instrument -- nothing here needs to change).
"""

import re
import json
import html
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo

from utils.config import get_date_with_suffix
from utils.constants import SIP_PORTFOLIO, WEALTH_USD_TO_INR_ESTIMATE
from utils.prompt_loader import load_prompt
from utils.logger import log
from utils import email_service
from llm import llm_backend

# How many diversification-note / suggestion bullets to ask the LLM for.
# Kept small on purpose -- this is a one-minute monthly checkup, not a
# research report.
AI_DIVERSIFICATION_POINTS = 3
AI_SUGGESTION_POINTS = 3

VERDICT_STYLES = {
    "Continue": ("#0f5132", "#d1f2e0"),
    "Increase": ("#0f5132", "#d1f2e0"),
    "Reduce": ("#7a5b00", "#fdf0cc"),
    "At Risk": ("#7a5b00", "#fdf0cc"),
    "Exit": ("#8a1c1c", "#fbdada"),
}
DEFAULT_VERDICT_STYLE = ("#4A5063", "#EDEAE2")


# -----------------------------------------------------------------------
# Holdings helpers
# -----------------------------------------------------------------------
def _instrument_monthly_inr(entry):
    """Returns an instrument's monthly contribution normalized to INR."""
    if "amount_usd" in entry:
        return entry["amount_usd"] * WEALTH_USD_TO_INR_ESTIMATE
    return entry.get("amount_inr", 0)


def _format_amount(entry):
    if "amount_usd" in entry:
        return f"${entry['amount_usd']:,.0f}/mo (~₹{_instrument_monthly_inr(entry):,.0f})"
    return f"₹{entry.get('amount_inr', 0):,.0f}/mo"


def _build_holdings_block(portfolio):
    lines = []
    for entry in portfolio:
        line = f"- {entry['instrument']} [{entry['category']}]: {_format_amount(entry)}"
        if entry.get("notes"):
            line += f" ({entry['notes']})"
        lines.append(line)
    return "\n".join(lines)


def _portfolio_totals(portfolio):
    total_inr = sum(_instrument_monthly_inr(e) for e in portfolio)
    return total_inr


# -----------------------------------------------------------------------
# Prompt + AI call
# -----------------------------------------------------------------------
def _build_wealth_prompt(holdings_block, today_str):
    return load_prompt(
        "wealth/monthly_report",
        context_text="",
        today_str=today_str,
        ai_diversification_points=AI_DIVERSIFICATION_POINTS,
        ai_suggestion_points=AI_SUGGESTION_POINTS,
        holdings_block=holdings_block,
    )


def _strip_code_fences(text):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def _parse_wealth_report_json(text, instrument_names):
    cleaned = _strip_code_fences(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return {}, None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}, None

    raw_verdicts = data.get("instrument_verdicts") if isinstance(data, dict) else None
    raw_take = data.get("portfolio_take") if isinstance(data, dict) else None

    verdicts = {}
    if isinstance(raw_verdicts, dict):
        name_lookup = {n.strip().lower(): n for n in instrument_names}
        for key, val in raw_verdicts.items():
            matched_name = name_lookup.get(str(key).strip().lower(), key)
            if isinstance(val, dict):
                verdict = str(val.get("verdict", "")).strip() or "Continue"
                reason = str(val.get("reason", "")).strip()
            else:
                verdict, reason = "Continue", str(val).strip()
            verdicts[matched_name] = {"verdict": verdict, "reason": reason}

    portfolio_take = None
    if isinstance(raw_take, dict):
        headline = raw_take.get("headline")
        headline = headline.strip() if isinstance(headline, str) and headline.strip() else None
        notes = [str(n).strip() for n in raw_take.get("diversification_notes", []) if str(n).strip()]
        suggestions = [str(s).strip() for s in raw_take.get("suggestions", []) if str(s).strip()]
        if headline or notes or suggestions:
            portfolio_take = {
                "headline": headline,
                "diversification_notes": notes[:AI_DIVERSIFICATION_POINTS],
                "suggestions": suggestions[:AI_SUGGESTION_POINTS],
            }

    return verdicts, portfolio_take


def _fallback_verdicts(instrument_names):
    return {
        name: {"verdict": "Continue", "reason": "AI analysis unavailable this run -- no change flagged."}
        for name in instrument_names
    }


def _fallback_portfolio_take():
    return {
        "headline": "Portfolio Commentary Unavailable This Run",
        "diversification_notes": [],
        "suggestions": [
            "AI read was unavailable this run -- re-check next month's report before acting.",
        ],
    }


def generate_wealth_report(portfolio):
    """
    Single AI call covering the whole SIP portfolio at once (same
    live-search-first fallback chain as stock_controller's AI Stocks
    Story, via llm_backend.generate_analysis). Returns
    (verdicts, portfolio_take, used_live_search):
      verdicts: {instrument_name: {"verdict": str, "reason": str}} --
        falls back to a generic "Continue" per instrument if every AI
        tier fails, so the table is never left blank.
      portfolio_take: {"headline", "diversification_notes", "suggestions"}
      used_live_search: True if the tier that produced the result was
        genuinely grounded in live search results.
    """
    instrument_names = [e["instrument"] for e in portfolio]
    today_str = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%d %B %Y")
    holdings_block = _build_holdings_block(portfolio)
    prompt = _build_wealth_prompt(holdings_block, today_str)

    def _validate(text):
        verdicts, _take = _parse_wealth_report_json(text, instrument_names)
        return bool(verdicts)

    text, _sources, used_live = llm_backend.generate_analysis(
        prompt,
        max_tokens=1800,
        validate_fn=_validate,
        log_label="Monthly Wealth Report",
    )

    if text:
        verdicts, portfolio_take = _parse_wealth_report_json(text, instrument_names)
        if verdicts:
            # Fill in any instrument the model skipped so the table is
            # always complete, rather than silently missing a row.
            for name in instrument_names:
                verdicts.setdefault(name, {"verdict": "Continue", "reason": "Not flagged by AI this run."})
            return verdicts, portfolio_take or _fallback_portfolio_take(), used_live

    log.error("Monthly Wealth Report: every AI tier failed or returned nothing usable -- using fallback verdicts.")
    return _fallback_verdicts(instrument_names), _fallback_portfolio_take(), False


# -----------------------------------------------------------------------
# HTML rendering (kept deliberately simple -- one table, one takeaways box)
# -----------------------------------------------------------------------
SANS = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
SERIF = "Georgia,'Times New Roman',Times,serif"


def _build_table_html(portfolio, verdicts):
    rows = []
    for entry in portfolio:
        name = entry["instrument"]
        v = verdicts.get(name, {"verdict": "Continue", "reason": ""})
        text_color, bg_color = VERDICT_STYLES.get(v["verdict"], DEFAULT_VERDICT_STYLE)
        badge = (
            f'<span style="display:inline-block;padding:2px 8px;border-radius:10px;'
            f'font-size:11px;font-weight:700;color:{text_color};background:{bg_color};">'
            f'{html.escape(v["verdict"])}</span>'
        )
        rows.append(f"""
            <tr>
              <td style="padding:8px 10px;border-bottom:1px solid #EDEAE2;font-family:{SANS};font-size:12px;color:#1B2233;">{html.escape(name)}</td>
              <td style="padding:8px 10px;border-bottom:1px solid #EDEAE2;font-family:{SANS};font-size:11px;color:#8A8F9C;">{html.escape(entry["category"])}</td>
              <td style="padding:8px 10px;border-bottom:1px solid #EDEAE2;font-family:{SANS};font-size:12px;color:#1B2233;text-align:right;white-space:nowrap;">{html.escape(_format_amount(entry))}</td>
              <td style="padding:8px 10px;border-bottom:1px solid #EDEAE2;text-align:center;">{badge}</td>
              <td style="padding:8px 10px;border-bottom:1px solid #EDEAE2;font-family:{SANS};font-size:11px;color:#4A5063;">{html.escape(v["reason"])}</td>
            </tr>""")

    header = f"""
            <tr>
              <th style="padding:8px 10px;text-align:left;font-family:{SANS};font-size:10px;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;color:#8A8F9C;border-bottom:2px solid #14213D;">Instrument</th>
              <th style="padding:8px 10px;text-align:left;font-family:{SANS};font-size:10px;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;color:#8A8F9C;border-bottom:2px solid #14213D;">Category</th>
              <th style="padding:8px 10px;text-align:right;font-family:{SANS};font-size:10px;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;color:#8A8F9C;border-bottom:2px solid #14213D;">Monthly SIP</th>
              <th style="padding:8px 10px;text-align:center;font-family:{SANS};font-size:10px;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;color:#8A8F9C;border-bottom:2px solid #14213D;">Verdict</th>
              <th style="padding:8px 10px;text-align:left;font-family:{SANS};font-size:10px;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;color:#8A8F9C;border-bottom:2px solid #14213D;">Why</th>
            </tr>"""

    return f"""<table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="border-collapse:collapse;">{header}{''.join(rows)}</table>"""


def _build_portfolio_take_html(portfolio_take, used_live_search):
    headline = portfolio_take.get("headline")
    notes = portfolio_take.get("diversification_notes") or []
    suggestions = portfolio_take.get("suggestions") or []
    live_tag = " &nbsp;&middot;&nbsp; LIVE-GROUNDED" if used_live_search else ""

    headline_html = ""
    if headline:
        headline_html = (
            f'<div style="margin:6px 0 10px;font-family:{SERIF};font-size:17px;'
            f'font-weight:700;line-height:1.3;color:#1F2430;">{html.escape(headline)}</div>'
        )

    def _bullet_list(title, items):
        if not items:
            return ""
        lis = "".join(f'<li style="margin:0 0 4px;">{html.escape(i)}</li>' for i in items)
        return (
            f'<div style="font-family:{SANS};font-size:10px;font-weight:700;letter-spacing:0.06em;'
            f'text-transform:uppercase;color:#B08D57;margin:10px 0 4px;">{title}</div>'
            f'<ul style="margin:0;padding-left:16px;font-family:{SANS};font-size:12px;line-height:1.5;color:#3C4256;">{lis}</ul>'
        )

    return f"""
    <tr>
      <td style="padding:0 28px 18px;" class="email-padding">
        <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="border-radius:6px;background:#FAF8F1;border:1px solid #E7DFC9;">
          <tr>
            <td style="padding:14px 16px 16px;">
              <div style="font-family:{SANS};font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#B08D57;">🤖 Portfolio Take{live_tag}</div>
              {headline_html}
              {_bullet_list("Diversification Notes", notes)}
              {_bullet_list("Suggestions For Next Month", suggestions)}
            </td>
          </tr>
        </table>
      </td>
    </tr>
    """


def build_report_html(portfolio, verdicts, portfolio_take, used_live_search):
    total_inr = _portfolio_totals(portfolio)
    now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
    date_str = get_date_with_suffix(now_ist)
    table_html = _build_table_html(portfolio, verdicts)
    take_html = _build_portfolio_take_html(portfolio_take, used_live_search)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="color-scheme" content="light">
<title>Monthly Wealth &amp; SIP Report</title>
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
              <div style="font-family:{SANS};font-size:10px;font-weight:700;letter-spacing:0.16em;text-transform:uppercase;color:#B08D57;">Wealth Management &nbsp;&bull;&nbsp; Monthly Briefing</div>
              <h1 style="margin:8px 0 0;font-family:{SERIF};font-size:22px;font-weight:400;line-height:1.3;color:#ffffff;">Monthly Wealth &amp; SIP Report</h1>
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
              <p style="margin:0;font-family:{SANS};font-size:13px;color:#4A5063;">Total recurring monthly outflow across all instruments: <strong>₹{total_inr:,.0f}</strong></p>
            </td>
          </tr>
          {take_html}
          <tr>
            <td style="padding:0 28px 24px;" class="email-padding">
              {table_html}
            </td>
          </tr>
          <tr>
            <td style="padding:16px 28px;border-top:1px solid #EDEAE2;" class="email-padding">
              <p style="margin:0;font-family:{SANS};font-size:10px;color:#8A8F9C;line-height:1.5;">
                Generated automatically from your configured SIP list (utils/constants.py) plus an AI market read.
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
    now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
    subject = f"Monthly Wealth & SIP Report - {get_date_with_suffix(now_ist)}"
    return email_service.send_email(subject=subject, html_body=report_html)


def main(dry_run=False):
    llm_backend.init_llm_generator()
    portfolio = SIP_PORTFOLIO
    log.info(f"Generating monthly wealth report for {len(portfolio)} instruments...")

    verdicts, portfolio_take, used_live = generate_wealth_report(portfolio)
    report_html = build_report_html(portfolio, verdicts, portfolio_take, used_live)

    if dry_run:
        out_path = "wealth_report_preview.html"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report_html)
        log.info(f"Dry run -- wrote preview to {out_path} instead of sending email.")
        return

    success = send_report_email(report_html)
    if success:
        log.info("Monthly wealth report email sent successfully.")
    else:
        log.error("Failed to send monthly wealth report email.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and email the monthly wealth/SIP report.")
    parser.add_argument("--dry-run", action="store_true", help="Write the report to a local HTML file instead of emailing it.")
    args = parser.parse_args()
    main(dry_run=args.dry_run)