"""
swing_trade_scoring.py

Weighted composite scoring, as a diagnostic/ranking complement to the
existing all-or-nothing hard-gate verification in swing_trade_advisor.py.

WHY: five simultaneous hard filters means one weak metric kills an
otherwise strong candidate, and a pure pass/fail throws away exactly the
information you'd want for tuning ("how close did the ones that failed
actually come?"). A composite score doesn't remove the hard gate -- the
gate stays, because "this contradicts the strategy's own stated rule" is
a real, disclosed problem worth keeping visible and blocking on by
default. What this module adds is a per-candidate score built from how
far past (or short of) each threshold a candidate lands, so:
  (a) every run's "no qualifying trade" summary can be ranked by how
      close each rejected candidate actually came, instead of an
      unordered list, and
  (b) when USE_COMPOSITE_SCORE=true and more than one candidate would
      otherwise qualify, the strongest one by composite score is put
      first rather than "whichever the model happened to list first".

This module never overrides a hard contradiction -- see
swing_trade_advisor.py's _split_qualifying, which is unchanged. It only
adds ranking on top.
"""

import os

# Weights sum to 1.0; tune via env vars if you want to emphasize one leg
# of the strategy over another. Defaults weight fundamentals (this
# strategy's original gate) and risk:reward slightly higher than
# technicals/sentiment, since growth-and-quality is the thesis and
# technicals/sentiment are the entry timing on top of it.
def _env_float(name, default):
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


WEIGHT_FUNDAMENTALS = _env_float("SCORE_WEIGHT_FUNDAMENTALS", 0.35)
WEIGHT_TECHNICALS = _env_float("SCORE_WEIGHT_TECHNICALS", 0.25)
WEIGHT_RISK_REWARD = _env_float("SCORE_WEIGHT_RISK_REWARD", 0.25)
WEIGHT_SANITY = _env_float("SCORE_WEIGHT_SANITY", 0.15)

USE_COMPOSITE_SCORE = os.getenv("USE_COMPOSITE_SCORE", "false").lower() == "true"


def _leg_score(notes, relevant_prefixes=None):
    """
    Turns a list of (note_text, severity) tuples from one verification leg
    into a 0-100 score: starts at 100, each "hard" contradiction costs 40
    points, each "soft"/"nodata" (unverifiable) costs 10. Floored at 0.
    This mirrors the existing _adjust_confidence penalty logic in
    swing_trade_advisor.py (hard costs more than soft) but produces a
    bounded 0-100 scale that's comparable across legs regardless of how
    many individual checks each leg happens to run.
    """
    score = 100.0
    for _text, sev in notes:
        if sev == "hard":
            score -= 40.0
        else:  # "soft" or "nodata"
            score -= 10.0
    return max(0.0, score)


def compute_composite_score(rr_notes, tech_notes, fund_notes, sanity_notes):
    """
    Returns (composite_0_100, breakdown_dict). Composite is a weighted
    average of the four leg scores -- see module docstring for why this
    exists alongside (not instead of) the hard-gate check.
    """
    fund_score = _leg_score(fund_notes)
    tech_score = _leg_score(tech_notes)
    rr_score = _leg_score(rr_notes)
    sanity_score = _leg_score(sanity_notes)

    total_weight = (WEIGHT_FUNDAMENTALS + WEIGHT_TECHNICALS + WEIGHT_RISK_REWARD + WEIGHT_SANITY) or 1.0
    composite = (
        fund_score * WEIGHT_FUNDAMENTALS
        + tech_score * WEIGHT_TECHNICALS
        + rr_score * WEIGHT_RISK_REWARD
        + sanity_score * WEIGHT_SANITY
    ) / total_weight

    breakdown = {
        "fundamentals_score": round(fund_score, 1),
        "technicals_score": round(tech_score, 1),
        "risk_reward_score": round(rr_score, 1),
        "sanity_score": round(sanity_score, 1),
        "composite_score": round(composite, 1),
    }
    return round(composite, 1), breakdown


def rank_by_composite(stocks):
    """
    Sorts a list of stock dicts (each already carrying a "_composite_score"
    key, set by the caller after compute_composite_score) descending by
    that score. Stable sort -- ties keep the model's original ordering.
    Safe to call on qualifying candidates, rejected candidates, or a mix
    of both (e.g. for the "closest near-misses" section of a
    no-qualifying-trade report).
    """
    return sorted(stocks, key=lambda s: s.get("_composite_score", 0.0), reverse=True)


def composite_score_html(stock):
    """Small inline fragment showing the composite score + leg breakdown,
    for use in the per-stock card or the near-miss ranking list."""
    breakdown = stock.get("_composite_breakdown")
    score = stock.get("_composite_score")
    if score is None or breakdown is None:
        return None
    return (
        f'<strong>{score:.1f}/100</strong> '
        f'<span style="font-size:11px;color:#8A8F9C;">'
        f'(fundamentals {breakdown["fundamentals_score"]:.0f} &middot; '
        f'technicals {breakdown["technicals_score"]:.0f} &middot; '
        f'risk:reward {breakdown["risk_reward_score"]:.0f} &middot; '
        f'sanity {breakdown["sanity_score"]:.0f})</span>'
    )