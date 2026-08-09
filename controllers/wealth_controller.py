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
  - One flat table (instrument, category, monthly amount, AI recommendation,
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
import difflib
import argparse
import requests
from datetime import datetime
from zoneinfo import ZoneInfo

from utils.config import get_date_with_suffix
from utils.constants import SIP_PORTFOLIO, WEALTH_USD_TO_INR_FALLBACK
from utils.prompt_loader import load_prompt
from utils.logger import log
from utils import email_service
from llm import llm_backend

# How many diversification-note / suggestion bullets to ask the LLM for.
# Kept small on purpose -- this is a one-minute monthly checkup, not a
# research report.
AI_DIVERSIFICATION_POINTS = 3
AI_SUGGESTION_POINTS = 3

RECOMMENDATION_STYLES = {
    "Continue": ("#0f5132", "#d1f2e0"),
    "Increase": ("#0f5132", "#d1f2e0"),
    "Reduce": ("#7a5b00", "#fdf0cc"),
    "At Risk": ("#7a5b00", "#fdf0cc"),
    "Exit": ("#8a1c1c", "#fbdada"),
    "Unavailable": ("#4A5063", "#EDEAE2"),
}
DEFAULT_RECOMMENDATION_STYLE = ("#4A5063", "#EDEAE2")

# Portfolio Snapshot -- a compact scorecard (capacity + a handful of
# AI-rated dimensions) giving the complete picture at a glance, separate
# from the per-instrument table and the narrative Portfolio Take box.
# Band -> (emoji, text color, background color). Reuses the same palette
# as RECOMMENDATION_STYLES for Green/Yellow/Red so the report stays
# visually consistent; "Orange" is a new, distinct step between the two.
SNAPSHOT_BAND_STYLES = {
    "Green": ("🟢", "#0f5132", "#d1f2e0"),
    "Yellow": ("🟡", "#7a5b00", "#fdf0cc"),
    "Orange": ("🟠", "#9a3412", "#ffe4cc"),
    "Red": ("🔴", "#8a1c1c", "#fbdada"),
}
DEFAULT_SNAPSHOT_BAND_STYLE = ("⚪", "#4A5063", "#EDEAE2")

# Fixed (key, display label) pairs for the AI-rated snapshot rows, in
# display order. Keys must match what the prompt template asks the LLM
# to return under "portfolio_snapshot" -- keep the two in sync.
SNAPSHOT_FIELDS = [
    ("equity_orientation", "Equity orientation"),
    ("retirement_discipline", "Retirement discipline"),
    ("debt_fixed_income_backing", "Debt/fixed-income backing"),
    ("diversification_by_products", "Diversification by products"),
    ("diversification_by_underlying_risk", "Diversification by underlying risk"),
    ("international_diversification", "International diversification"),
    ("precious_metal_allocation", "Precious-metal allocation"),
    ("direct_stock_concentration", "Direct-stock concentration"),
    ("fund_overlap", "Fund overlap"),
]


# -----------------------------------------------------------------------
# Live USD -> INR rate (free, no API key). Two independent providers tried
# in order -- both free/keyless -- before giving up and using the fixed
# fallback constant, so a single provider being down doesn't silently
# fall back to a stale hardcoded number on every run.
# -----------------------------------------------------------------------
def _fetch_rate_frankfurter():
    resp = requests.get("https://api.frankfurter.app/latest", params={"from": "USD", "to": "INR"}, timeout=6)
    resp.raise_for_status()
    return float(resp.json()["rates"]["INR"])


def _fetch_rate_open_er_api():
    resp = requests.get("https://open.er-api.com/v6/latest/USD", timeout=6)
    resp.raise_for_status()
    data = resp.json()
    if data.get("result") != "success":
        raise ValueError(f"open.er-api.com returned non-success result: {data.get('result')}")
    return float(data["rates"]["INR"])


def fetch_live_usd_inr_rate():
    """
    Returns (rate, is_live): the current USD->INR rate from a free,
    keyless FX API, or (WEALTH_USD_TO_INR_FALLBACK, False) if every
    provider fails -- so a live-fetch outage degrades to a labeled
    estimate instead of crashing the whole report.
    """
    for fetch_fn, provider_name in ((_fetch_rate_frankfurter, "frankfurter.app"), (_fetch_rate_open_er_api, "open.er-api.com")):
        try:
            rate = fetch_fn()
            if rate and rate > 0:
                log.info(f"USD/INR live rate: {rate:.2f} (via {provider_name})")
                return rate, True
        except Exception as e:
            log.warning(f"USD/INR live rate fetch failed via {provider_name}: {e}")

    log.warning(f"USD/INR live rate: every provider failed -- using fallback estimate {WEALTH_USD_TO_INR_FALLBACK}.")
    return WEALTH_USD_TO_INR_FALLBACK, False


# -----------------------------------------------------------------------
# Holdings helpers
# -----------------------------------------------------------------------
def _instrument_monthly_inr(entry, usd_inr_rate):
    """Returns an instrument's monthly contribution normalized to INR."""
    if "amount_usd" in entry:
        return entry["amount_usd"] * usd_inr_rate
    return entry.get("amount_inr", 0)


def _format_amount(entry, usd_inr_rate):
    if "amount_usd" in entry:
        return f"${entry['amount_usd']:,.0f}/mo (~₹{_instrument_monthly_inr(entry, usd_inr_rate):,.0f})"
    return f"₹{entry.get('amount_inr', 0):,.0f}/mo"


def _build_holdings_block(portfolio, usd_inr_rate):
    lines = []
    for entry in portfolio:
        line = f"- {entry['instrument']} [{entry['category']}]: {_format_amount(entry, usd_inr_rate)}"
        if entry.get("notes"):
            line += f" ({entry['notes']})"
        lines.append(line)
    return "\n".join(lines)


def _portfolio_totals(portfolio, usd_inr_rate):
    total_inr = sum(_instrument_monthly_inr(e, usd_inr_rate) for e in portfolio)
    return total_inr


def _format_lakh(amount_inr):
    """Formats an INR amount in lakhs (₹1,00,000 = 1 lakh), as commonly
    used for Indian investment-capacity figures."""
    return f"₹{amount_inr / 100000:,.2f} lakh"


# -----------------------------------------------------------------------
# Prompt + AI call
# -----------------------------------------------------------------------
# context_text is intentionally left empty -- the accuracy rules and
# fund/ETF-specific, rupee-anchored suggestion requirements now live
# directly in the wealth/monthly_report template itself, so they don't
# need to be duplicated (and risk drifting out of sync) here. Keep this
# hook available for genuinely dynamic, per-run context in future (e.g.
# injecting a live-search digest) rather than static instructions.
def _build_wealth_prompt(holdings_block, today_str):
    return load_prompt(
        "wealth/monthly_report",
        context_text="",
        today_str=today_str,
        ai_diversification_points=AI_DIVERSIFICATION_POINTS,
        ai_suggestion_points=AI_SUGGESTION_POINTS,
        holdings_block=holdings_block,
    )


# Words that carry no matching signal (fund-name boilerplate, tier
# labels, etc.) -- stripped before comparing an LLM-returned key against
# our configured instrument names, so e.g. "SBI Small Cap SIP" still
# matches "SBI Small Cap Fund" and "L&T" still matches "Larsen & Toubro".
_NOISE_WORDS = {
    "fund", "scheme", "sip", "the", "ltd", "limited", "direct", "growth",
    "mutual", "monthly", "tier", "i", "and", "of", "pension", "management",
}


def _normalize_name(s):
    s = s.lower().replace("&", " and ").replace("l&t", "larsen and toubro")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    tokens = [t for t in s.split() if t and t not in _NOISE_WORDS]
    return " ".join(tokens) if tokens else s.strip()


# Shorthand the LLM tends to use that's too short/abbreviated for the
# substring/fuzzy checks below to catch reliably, but is unambiguous
# against THIS portfolio's instrument list (unlike e.g. "SBI" or "ICICI"
# alone, which are genuinely ambiguous between the bank stock and the
# same-house mutual fund -- those are intentionally left unmatched
# rather than risk mis-assigning a recommendation to the wrong instrument).
_UNAMBIGUOUS_ALIASES = {
    "lt": "Larsen & Toubro",
    "l t": "Larsen & Toubro",  # "L&T" normalizes to this ("and" is a noise word)
    "bel": "Bharat Electronics (BEL)",
    "nps": "NPS (HDFC Pension Fund Mgmt)",  # only one NPS instrument -- unambiguous, unlike SBI/ICICI
    "epf": "Employee Provident Fund",  # only one EPF instrument -- unambiguous shorthand
    "ppf": "Public Provident Fund",  # only one PPF instrument -- unambiguous shorthand
}


def _match_instrument_name(raw_key, normalized_lookup):
    """
    Maps an LLM-returned key back to the exact configured instrument name.
    Tries, in order: exact normalized match, a known unambiguous alias,
    substring containment either direction (handles things like "SBI
    Small Cap Fund SIP" -> "SBI Small Cap Fund"), then a fuzzy ratio
    match. Falls back to the raw key (unmatched) only if nothing clears
    the similarity bar -- an unmatched key just means that instrument
    keeps its default "Unavailable" row rather than getting a wrong recommendation.
    """
    key_norm = _normalize_name(str(raw_key))
    if key_norm in normalized_lookup:
        return normalized_lookup[key_norm]

    if key_norm in _UNAMBIGUOUS_ALIASES:
        aliased = _UNAMBIGUOUS_ALIASES[key_norm]
        if _normalize_name(aliased) in normalized_lookup:
            return aliased

    # Short single-token keys (e.g. "SBI", "ICICI") are genuinely
    # ambiguous against this portfolio -- "sbi" is a substring of both
    # "State Bank of India" (spelled out, so no textual "sbi") and "SBI
    # Small Cap Fund" (which literally starts with it), and a loose
    # containment/fuzzy check below would happily match the wrong one.
    # Only an explicit alias above should resolve these; otherwise skip
    # straight to "unmatched" rather than risk a wrong-instrument recommendation.
    if " " not in key_norm and len(key_norm) <= 5:
        return raw_key

    for norm, orig in normalized_lookup.items():
        if norm and (norm in key_norm or key_norm in norm):
            return orig

    close = difflib.get_close_matches(key_norm, list(normalized_lookup.keys()), n=1, cutoff=0.6)
    if close:
        return normalized_lookup[close[0]]

    return raw_key


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

    raw_recommendations = data.get("instrument_recommendations") if isinstance(data, dict) else None
    raw_take = data.get("portfolio_take") if isinstance(data, dict) else None
    raw_snapshot = data.get("portfolio_snapshot") if isinstance(data, dict) else None

    recommendations = {}
    if isinstance(raw_recommendations, dict):
        normalized_lookup = {_normalize_name(n): n for n in instrument_names}
        for key, val in raw_recommendations.items():
            matched_name = _match_instrument_name(key, normalized_lookup)
            if matched_name not in instrument_names:
                log.warning(f"Monthly Wealth Report: AI returned recommendation for unrecognized instrument {key!r} -- ignoring.")
                continue
            if isinstance(val, dict):
                recommendation = str(val.get("recommendation", "")).strip() or "Continue"
                reason = str(val.get("reason", "")).strip()
            else:
                recommendation, reason = "Continue", str(val).strip()
            recommendations[matched_name] = {"recommendation": recommendation, "reason": reason}
            if str(key) != matched_name:
                log.info(f"Monthly Wealth Report: matched AI key {key!r} -> configured instrument {matched_name!r}.")

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

    portfolio_snapshot = None
    if isinstance(raw_snapshot, dict):
        rows = {}
        complete = True
        for key, _label in SNAPSHOT_FIELDS:
            entry = raw_snapshot.get(key)
            band = entry.get("band") if isinstance(entry, dict) else None
            label = entry.get("label") if isinstance(entry, dict) else None
            band = str(band).strip().title() if isinstance(band, str) and band.strip() else None
            label = str(label).strip() if isinstance(label, str) and label.strip() else None
            if band not in SNAPSHOT_BAND_STYLES or not label:
                complete = False
                continue
            rows[key] = {"band": band, "label": label}
        raw_score = raw_snapshot.get("overall_portfolio_health")
        score = None
        try:
            score = int(round(float(raw_score)))
        except (TypeError, ValueError):
            complete = False
        if score is not None:
            score = max(0, min(100, score))
        # Require every dimension plus a valid score -- a partial scorecard
        # would misrepresent the portfolio (e.g. showing 8 of 9 ratings
        # with no indication one is missing), so treat any gap as "the
        # whole snapshot is unavailable this run" rather than render it
        # incomplete.
        if complete and len(rows) == len(SNAPSHOT_FIELDS) and score is not None:
            portfolio_snapshot = {"rows": rows, "overall_portfolio_health": score}
        else:
            log.warning("Monthly Wealth Report: AI portfolio_snapshot was incomplete or malformed -- omitting scorecard this run.")

    return recommendations, portfolio_take, portfolio_snapshot


def _fallback_recommendations(instrument_names):
    return {
        name: {"recommendation": "Unavailable", "reason": "AI analysis unavailable this run -- no change flagged."}
        for name in instrument_names
    }


def _fallback_portfolio_take():
    return {
        "headline": "Portfolio Commentary Unavailable This Run",
        "note": "AI read was unavailable this run -- re-check next month's report before acting.",
        "diversification_notes": [],
        "suggestions": [],
    }


def generate_wealth_report(portfolio, usd_inr_rate):
    """
    Single AI call covering the whole SIP portfolio at once (same
    live-search-first fallback chain as stock_controller's AI Stocks
    Story, via llm_backend.generate_analysis). Returns
    (recommendations, portfolio_take, portfolio_snapshot, used_live_search):
      recommendations: {instrument_name: {"recommendation": str, "reason": str}} --
        falls back to a generic "Unavailable" per instrument if every AI
        tier fails, so the table is never left blank.
      portfolio_take: {"headline", "diversification_notes", "suggestions"}
      portfolio_snapshot: {"rows": {field_key: {"band", "label"}, ...},
        "overall_portfolio_health": int} or None if the AI didn't return a
        complete scorecard this run -- the report falls back to showing
        just the computed capacity figures in that case, rather than a
        partial or fabricated set of ratings.
      used_live_search: True if the tier that produced the result was
        genuinely grounded in live search results.
    """
    instrument_names = [e["instrument"] for e in portfolio]
    today_str = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%d %B %Y")
    holdings_block = _build_holdings_block(portfolio, usd_inr_rate)
    prompt = _build_wealth_prompt(holdings_block, today_str)

    def _validate(text):
        recommendations, _take, _snapshot = _parse_wealth_report_json(text, instrument_names)
        return bool(recommendations)

    text, _sources, used_live = llm_backend.generate_analysis(
        prompt,
        # Bumped from 1800 -- WEALTH_TAKE_GUIDANCE asks for named
        # funds/ETFs/sectors and exact ₹ amounts in every bullet, which
        # runs longer per point than the old percentage-only phrasing.
        max_tokens=2400,
        validate_fn=_validate,
        log_label="Monthly Wealth Report",
    )

    if text:
        recommendations, portfolio_take, portfolio_snapshot = _parse_wealth_report_json(text, instrument_names)
        if recommendations:
            # Fill in any instrument the model skipped so the table is
            # always complete, rather than silently missing a row. Before
            # giving up and defaulting to "Unavailable", retry once for just
            # the missing instruments -- most of the time a partial miss
            # is recoverable, so this should make the generic fallback
            # text rare rather than routine.
            missing = [name for name in instrument_names if name not in recommendations]
            if missing:
                log.warning(
                    f"Monthly Wealth Report: AI response covered {len(recommendations)}/{len(instrument_names)} "
                    f"instruments -- retrying for: {', '.join(missing)}. "
                    f"Check above for 'unrecognized instrument' warnings -- if the AI returned a recommendation "
                    f"under a name that didn't match, it's counted there instead of here."
                )
                retry_recommendations = _retry_missing_recommendations(missing, portfolio, usd_inr_rate)
                if retry_recommendations:
                    recommendations.update(retry_recommendations)
                    log.info(
                        f"Monthly Wealth Report: retry recovered AI recommendations for: {', '.join(retry_recommendations.keys())}."
                    )
                still_missing = [name for name in instrument_names if name not in recommendations]
                if still_missing:
                    log.warning(
                        f"Monthly Wealth Report: still no AI recommendation after retry for: {', '.join(still_missing)} "
                        f"-- defaulting to 'Unavailable'."
                    )
            for name in instrument_names:
                recommendations.setdefault(name, {"recommendation": "Unavailable", "reason": "Not flagged by AI this run."})
            return recommendations, portfolio_take or _fallback_portfolio_take(), portfolio_snapshot, used_live

    log.error("Monthly Wealth Report: every AI tier failed or returned nothing usable -- using fallback recommendations.")
    return _fallback_recommendations(instrument_names), _fallback_portfolio_take(), None, False


def _retry_missing_recommendations(missing_names, portfolio, usd_inr_rate):
    """
    One-shot retry for instruments the first AI call skipped entirely --
    asks again, but only for those instruments, so a partial miss doesn't
    fall straight through to the generic "Not flagged by AI this run"
    filler. Uses the same prompt/parsing path as the main call, just
    scoped to a smaller holdings block. Returns whatever subset of
    recommendations it manages to recover (possibly empty if the retry also
    fails) -- anything still missing after this is backfilled by the
    caller as before.
    """
    missing_entries = [e for e in portfolio if e["instrument"] in missing_names]
    if not missing_entries:
        return {}

    today_str = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%d %B %Y")
    holdings_block = _build_holdings_block(missing_entries, usd_inr_rate)
    prompt = _build_wealth_prompt(holdings_block, today_str)

    def _validate(text):
        v, _take, _snapshot = _parse_wealth_report_json(text, missing_names)
        return bool(v)

    text, _sources, _used_live = llm_backend.generate_analysis(
        prompt,
        max_tokens=600,
        validate_fn=_validate,
        log_label="Monthly Wealth Report (retry - missing instruments)",
    )
    if not text:
        return {}

    recommendations, _take, _snapshot = _parse_wealth_report_json(text, missing_names)
    return recommendations


# -----------------------------------------------------------------------
# HTML rendering (kept deliberately simple -- one table, one takeaways box)
# -----------------------------------------------------------------------
SANS = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
SERIF = "Georgia,'Times New Roman',Times,serif"


# Category strings that should all display as one "Mutual Funds" group in
# the report, regardless of how SIP_PORTFOLIO tags the underlying fund
# type (equity/debt/index/ELSS/hybrid/etc.) -- so the table shows a single
# consolidated mutual-fund bucket instead of splitting it by sub-type.
# Matched as a substring against the lowercased category, so e.g. an
# entry categorized "Equity Mutual Fund" or "Debt Fund" both collapse in.
# Deliberately does NOT match "NPS"/"PPF"/"EPF" categories -- those stay
# separate or fold into "Retirement" below. "Gold"/"Silver" fold into
# "Commodities" below instead of staying standalone. Extend this list if
# SIP_PORTFOLIO uses category labels not covered here.
_MUTUAL_FUND_CATEGORY_KEYWORDS = (
    "mutual fund", "elss", "index fund", "flexicap", "multicap",
    "midcap fund", "smallcap fund", "small cap fund", "largecap fund",
    "large cap fund", "hybrid fund", "liquid fund", "debt fund",
    "sectoral fund", "thematic fund",
)


# Category strings that should all display as one "Retirement" group,
# while each instrument (NPS, EPF, etc.) still shows as its own row within
# that group -- only the category header consolidates, not the rows.
# Deliberately excludes "PPF" (Public Provident Fund) -- kept as its own
# category since it wasn't asked to be folded in here.
_RETIREMENT_CATEGORY_KEYWORDS = ("nps", "epf", "retirement", "pension fund")

# Category strings that should all display as one "Commodities" group --
# Gold and Silver instruments (SGBs, ETFs, physical, digital) collapse
# into this header while each instrument still shows as its own row.
_COMMODITY_CATEGORY_KEYWORDS = ("gold", "silver", "commodit")


def _display_category(category):
    cat_lower = category.lower()
    if any(kw in cat_lower for kw in _MUTUAL_FUND_CATEGORY_KEYWORDS):
        return "Mutual Funds"
    if any(kw in cat_lower for kw in _RETIREMENT_CATEGORY_KEYWORDS):
        return "Retirement"
    if any(kw in cat_lower for kw in _COMMODITY_CATEGORY_KEYWORDS):
        return "Commodities"
    return category


def _group_by_category(portfolio):
    """Groups instruments by display category, preserving each category's
    first order of appearance in SIP_PORTFOLIO -- no separate category-order
    list to maintain when the portfolio changes. Category strings are passed
    through _display_category first, so mutual-fund sub-types collapse into
    "Mutual Funds", NPS/EPF collapse into "Retirement", and Gold/Silver
    collapse into "Commodities" regardless of how SIP_PORTFOLIO tags them;
    entries themselves are untouched and still render as separate rows --
    this only consolidates the category header, not the underlying data."""
    groups = {}
    order = []
    for entry in portfolio:
        cat = _display_category(entry["category"])
        if cat not in groups:
            groups[cat] = []
            order.append(cat)
        groups[cat].append(entry)
    return [(cat, groups[cat]) for cat in order]


def _build_table_html(portfolio, recommendations, usd_inr_rate):
    grouped = _group_by_category(portfolio)
    body_parts = []

    grand_total = _portfolio_totals(portfolio, usd_inr_rate)

    for category, entries in grouped:
        category_total = sum(_instrument_monthly_inr(e, usd_inr_rate) for e in entries)
        category_pct = (category_total / grand_total * 100) if grand_total else 0
        body_parts.append(f"""
            <tr>
              <td colspan="4" style="padding:10px 10px 6px;font-family:{SANS};font-size:10px;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;color:#B08D57;background:#FAF8F1;border-top:1px solid #E7DFC9;border-bottom:1px solid #E7DFC9;">
                {html.escape(category)} &nbsp;&middot;&nbsp; ₹{category_total:,.0f}/mo &nbsp;({category_pct:,.1f}% of overall)
              </td>
            </tr>""")
        for entry in entries:
            name = entry["instrument"]
            v = recommendations.get(name, {"recommendation": "Unavailable", "reason": ""})
            text_color, bg_color = RECOMMENDATION_STYLES.get(v["recommendation"], DEFAULT_RECOMMENDATION_STYLE)
            badge = (
                f'<span style="display:inline-block;padding:2px 8px;border-radius:10px;'
                f'font-size:11px;font-weight:700;color:{text_color};background:{bg_color};">'
                f'{html.escape(v["recommendation"])}</span>'
            )
            body_parts.append(f"""
            <tr>
              <td style="padding:8px 10px;border-bottom:1px solid #EDEAE2;font-family:{SANS};font-size:12px;color:#1B2233;">{html.escape(name)}</td>
              <td style="padding:8px 10px;border-bottom:1px solid #EDEAE2;font-family:{SANS};font-size:12px;color:#1B2233;text-align:right;white-space:nowrap;">{html.escape(_format_amount(entry, usd_inr_rate))}</td>
              <td style="padding:8px 10px;border-bottom:1px solid #EDEAE2;text-align:center;">{badge}</td>
              <td style="padding:8px 10px;border-bottom:1px solid #EDEAE2;font-family:{SANS};font-size:11px;color:#4A5063;">{html.escape(v["reason"])}</td>
            </tr>""")

    body_parts.append(f"""
            <tr>
              <td style="padding:10px 10px;font-family:{SANS};font-size:12px;font-weight:700;color:#14213D;border-top:2px solid #14213D;">Total</td>
              <td style="padding:10px 10px;font-family:{SANS};font-size:12px;font-weight:700;color:#14213D;text-align:right;white-space:nowrap;border-top:2px solid #14213D;">₹{grand_total:,.0f}/mo</td>
              <td style="padding:10px 10px;border-top:2px solid #14213D;" colspan="2"></td>
            </tr>""")

    header = f"""
            <tr>
              <th style="padding:8px 10px;text-align:left;font-family:{SANS};font-size:10px;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;color:#8A8F9C;border-bottom:2px solid #14213D;">Instrument</th>
              <th style="padding:8px 10px;text-align:right;font-family:{SANS};font-size:10px;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;color:#8A8F9C;border-bottom:2px solid #14213D;">Monthly SIP</th>
              <th style="padding:8px 10px;text-align:center;font-family:{SANS};font-size:10px;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;color:#8A8F9C;border-bottom:2px solid #14213D;">Recommendation</th>
              <th style="padding:8px 10px;text-align:left;font-family:{SANS};font-size:10px;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;color:#8A8F9C;border-bottom:2px solid #14213D;">Why</th>
            </tr>"""

    return f"""<table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="border-collapse:collapse;">{header}{''.join(body_parts)}</table>"""


def _score_band(score):
    """Maps the 0-100 overall_portfolio_health score to the same
    Green/Yellow/Orange/Red bands used for the individual rows, so the
    score's color is always consistent with how a rating of that severity
    would otherwise be shown."""
    if score >= 80:
        return "Green"
    if score >= 60:
        return "Yellow"
    if score >= 40:
        return "Orange"
    return "Red"


def _build_portfolio_snapshot_html(portfolio_snapshot, grand_total_inr):
    """
    Renders the Portfolio Snapshot scorecard: two always-available,
    code-computed capacity rows (monthly/annual investment capacity, from
    the actual configured portfolio -- no AI involved), followed by the
    AI-rated dimension rows and an overall health score when the AI
    returned a complete scorecard this run. If it didn't (portfolio_snapshot
    is None), only the capacity rows show, plus a short status note --
    same "don't fabricate, just say it's unavailable" approach as the
    Portfolio Take fallback.
    """
    monthly_cap = _format_lakh(grand_total_inr)
    annual_cap = _format_lakh(grand_total_inr * 12)

    def _plain_row(label, value):
        return f"""
            <tr>
              <td style="padding:6px 10px;font-family:{SANS};font-size:12px;color:#3C4256;">{html.escape(label)}</td>
              <td style="padding:6px 10px;font-family:{SANS};font-size:12px;font-weight:700;color:#1F2430;text-align:right;">{html.escape(value)}</td>
            </tr>"""

    def _band_row(label, band, badge_label):
        emoji, text_color, bg_color = SNAPSHOT_BAND_STYLES.get(band, DEFAULT_SNAPSHOT_BAND_STYLE)
        badge = (
            f'<span style="display:inline-block;padding:2px 8px;border-radius:10px;'
            f'font-size:11px;font-weight:700;color:{text_color};background:{bg_color};">'
            f'{emoji} {html.escape(badge_label)}</span>'
        )
        return f"""
            <tr>
              <td style="padding:6px 10px;font-family:{SANS};font-size:12px;color:#3C4256;">{html.escape(label)}</td>
              <td style="padding:6px 10px;text-align:right;">{badge}</td>
            </tr>"""

    rows_html = [
        _plain_row("Monthly investment capacity", monthly_cap),
        _plain_row("Annual investment capacity", annual_cap),
    ]

    note_html = ""
    if portfolio_snapshot:
        rows = portfolio_snapshot["rows"]
        for key, label in SNAPSHOT_FIELDS:
            entry = rows[key]
            rows_html.append(_band_row(label, entry["band"], entry["label"]))
        score = portfolio_snapshot["overall_portfolio_health"]
        score_band = _score_band(score)
        rows_html.append(_band_row("Overall portfolio health", score_band, f"{score}/100"))
    else:
        note_html = (
            f'<div style="margin:8px 0 0;font-family:{SANS};font-size:12px;'
            f'font-style:italic;line-height:1.5;color:#8A8F9C;">Portfolio ratings unavailable this run -- re-check next month\'s report.</div>'
        )

    table_html = (
        f'<table width="100%" cellpadding="0" cellspacing="0" role="presentation" '
        f'style="border-collapse:collapse;">{"".join(rows_html)}</table>'
    )

    return f"""
    <tr>
      <td style="padding:0 28px 18px;" class="email-padding">
        <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="border-radius:6px;background:#FAF8F1;border:1px solid #E7DFC9;">
          <tr>
            <td style="padding:14px 16px 16px;">
              <div style="font-family:{SANS};font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#B08D57;">📊 Portfolio Snapshot</div>
              {note_html}
              {table_html}
            </td>
          </tr>
        </table>
      </td>
    </tr>
    """


def _build_portfolio_take_html(portfolio_take, used_live_search):
    headline = portfolio_take.get("headline")
    note = portfolio_take.get("note")
    notes = portfolio_take.get("diversification_notes") or []
    suggestions = portfolio_take.get("suggestions") or []
    live_tag = " &nbsp;&middot;&nbsp; LIVE-GROUNDED" if used_live_search else ""

    headline_html = ""
    if headline:
        headline_html = (
            f'<div style="margin:6px 0 10px;font-family:{SERIF};font-size:17px;'
            f'font-weight:700;line-height:1.3;color:#1F2430;">{html.escape(headline)}</div>'
        )

    # Status note (e.g. "AI read was unavailable this run") -- only set on
    # the fallback path, so it never competes with real AI-generated
    # suggestions. Rendered as a plain line, not a bulleted "suggestion",
    # since it's a run-status message rather than portfolio advice.
    note_html = ""
    if note:
        note_html = (
            f'<div style="margin:6px 0 0;font-family:{SANS};font-size:12px;'
            f'font-style:italic;line-height:1.5;color:#8A8F9C;">{html.escape(note)}</div>'
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
              {note_html}
              {_bullet_list("Diversification Notes", notes)}
              {_bullet_list("Suggestions For Next Month", suggestions)}
            </td>
          </tr>
        </table>
      </td>
    </tr>
    """


def build_report_html(portfolio, recommendations, portfolio_take, portfolio_snapshot, used_live_search, usd_inr_rate, rate_is_live):
    total_inr = _portfolio_totals(portfolio, usd_inr_rate)
    now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
    date_str = get_date_with_suffix(now_ist)
    table_html = _build_table_html(portfolio, recommendations, usd_inr_rate)
    snapshot_html = _build_portfolio_snapshot_html(portfolio_snapshot, total_inr)
    take_html = _build_portfolio_take_html(portfolio_take, used_live_search)
    fx_tag = "live rate" if rate_is_live else "fallback estimate -- live fetch failed"

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
              <p style="margin:4px 0 0;font-family:{SANS};font-size:11px;color:#8A8F9C;">USD/INR: ₹{usd_inr_rate:,.2f} ({fx_tag})</p>
            </td>
          </tr>
          {snapshot_html}
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

    usd_inr_rate, rate_is_live = fetch_live_usd_inr_rate()
    recommendations, portfolio_take, portfolio_snapshot, used_live = generate_wealth_report(portfolio, usd_inr_rate)
    report_html = build_report_html(portfolio, recommendations, portfolio_take, portfolio_snapshot, used_live, usd_inr_rate, rate_is_live)

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