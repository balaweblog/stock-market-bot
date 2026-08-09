"""
utils/breakout_backtest.py

Live, per-stock, per-pattern backtest -- run at scan time, not
pre-computed. For every pattern flagged TODAY for a stock, this replays
the SAME detector function across that stock's own trailing history to
see how that pattern actually performed the last time(s) it fired,
before the signal is allowed into the "Confirmed" section of the email.

No-lookahead discipline: for each historical day i where the detector
would have fired, only df.iloc[:i+1] existed at the time -- exactly what
the detector itself is restricted to (see breakout_patterns.py's module
docstring). Forward returns are then measured strictly AFTER i, using
data that would only have arrived later. This is what makes the
backtest's hit-rate meaningful instead of hindsight-biased.
"""

import numpy as np
from utils.logger import log

FORWARD_HORIZONS = (5, 10, 20)  # trading days
MIN_SAMPLES_FOR_CONFIDENCE = 5
CONFIRM_HIT_RATE_THRESHOLD = 0.55  # >=55% of past occurrences positive at the primary horizon
PRIMARY_HORIZON = 10


def find_historical_events(df, detector_fn, as_of_i, min_gap_days=5, max_events=60, max_horizon=None):
    """
    Replays detector_fn across df[:as_of_i] (i.e. everything strictly
    BEFORE today's signal, so today's own occurrence never leaks into
    its own sample) and returns the list of historical positional
    indices where it would have fired -- the SAME no-lookahead event
    set backtest_pattern() below uses for its hit-rate stats, extracted
    here so other modules (utils.breakout_failure's failure-rate
    backtest) can replay the identical historical occurrences rather
    than re-deriving a possibly-different set for what's meant to be
    the "same" pattern history.

    min_gap_days de-duplicates overlapping/adjacent firings of the same
    setup (e.g. a resistance breakout that keeps re-triggering for a
    week straight) into one event, so the sample isn't artificially
    inflated by one prolonged move.

    max_horizon: only scan up to as_of_i - max_horizon so every event
    has a FULL forward window available to whatever the caller measures
    -- otherwise a signal from 3 days ago would only have a 3-day-old
    forward read, not a true one, biasing results toward whatever partial
    move happened to occur. Defaults to max(FORWARD_HORIZONS) so callers
    that don't pass their own horizon still get a safe, conservative cutoff
    consistent with backtest_pattern's own scan.

    Returns [] if there's not enough runway to scan at all, or if the
    pattern simply never fired historically -- callers distinguish "no
    runway" from "zero events" via as_of_i themselves if they need to.
    """
    if max_horizon is None:
        max_horizon = max(FORWARD_HORIZONS)
    if as_of_i < 60:
        return []

    scan_end = as_of_i - max_horizon
    if scan_end < 60:
        return []

    event_indices = []
    last_event_i = -min_gap_days - 1
    for hist_i in range(60, scan_end):
        if hist_i - last_event_i < min_gap_days:
            continue
        try:
            fired = detector_fn(df, hist_i)
        except Exception:
            fired = None
        if fired:
            event_indices.append(hist_i)
            last_event_i = hist_i
            if len(event_indices) >= max_events:
                break
    return event_indices


def backtest_pattern(df, detector_fn, as_of_i, min_gap_days=5, max_events=60):
    """
    Builds forward-return stats from find_historical_events() above:
    records forward returns at each horizon in FORWARD_HORIZONS for
    every historical firing, and returns a stats dict.

    Returns:
      {
        "sample_size": int,
        "horizons": {5: {"hit_rate": float, "avg_return": float, "median_return": float}, ...},
        "confirmed": bool,   # sample_size >= MIN_SAMPLES_FOR_CONFIDENCE
                              # and PRIMARY_HORIZON hit_rate >= CONFIRM_HIT_RATE_THRESHOLD
      }
    or None if there's not enough trailing history to attempt a
    backtest at all (as opposed to "attempted, zero historical events found").
    """
    max_horizon = max(FORWARD_HORIZONS)
    if as_of_i < 60:  # not enough runway to even try
        return None

    closes = df["Close"].values
    event_indices = find_historical_events(
        df, detector_fn, as_of_i, min_gap_days=min_gap_days, max_events=max_events, max_horizon=max_horizon,
    )

    if not event_indices:
        return {"sample_size": 0, "horizons": {}, "confirmed": False}

    horizon_returns = {h: [] for h in FORWARD_HORIZONS}
    for ei in event_indices:
        entry_price = closes[ei]
        if entry_price <= 0:
            continue
        for h in FORWARD_HORIZONS:
            exit_i = ei + h
            if exit_i < len(closes):
                horizon_returns[h].append((closes[exit_i] - entry_price) / entry_price)

    horizons = {}
    for h, rets in horizon_returns.items():
        if not rets:
            continue
        rets_arr = np.array(rets)
        horizons[h] = {
            "hit_rate": float((rets_arr > 0).mean()),
            "avg_return": float(rets_arr.mean()),
            "median_return": float(np.median(rets_arr)),
            "sample_size": int(len(rets_arr)),
        }

    sample_size = len(event_indices)
    primary = horizons.get(PRIMARY_HORIZON)
    confirmed = bool(
        sample_size >= MIN_SAMPLES_FOR_CONFIDENCE
        and primary is not None
        and primary["hit_rate"] >= CONFIRM_HIT_RATE_THRESHOLD
    )

    return {"sample_size": sample_size, "horizons": horizons, "confirmed": confirmed}


def backtest_signal(symbol, df, detector_fn, as_of_i):
    """Thin wrapper with logging -- used by the controller so a slow or
    failing backtest for one symbol/pattern never takes down the run."""
    try:
        return backtest_pattern(df, detector_fn, as_of_i)
    except Exception as e:
        log.warning(f"Breakout Screener: backtest failed for {symbol} ({detector_fn.__name__}): {e}")
        return None


# -----------------------------------------------------------------------
# Breakout Quality Score
# -----------------------------------------------------------------------
# A single 0-100 number so signals can be ranked/skimmed at a glance,
# built ENTIRELY from numbers already computed above (plus the
# data-quality notes from utils.nse_data) -- no hidden inputs, no ML
# black box, so every score is explainable by pointing at its breakdown.
#
# Weights (sum to 100):
#   45  Historical hit-rate at PRIMARY_HORIZON -- the main question:
#       "when this exact pattern fired on this exact stock before, did
#       it actually work?"
#   20  Sample confidence -- a 90% hit-rate over 4 occurrences is a lot
#       less trustworthy than 65% over 25, so more history earns more score.
#   15  Average return magnitude at PRIMARY_HORIZON -- a positive hit-rate
#       on trivially small moves scores lower than one with real magnitude.
#   10  Cross-horizon consistency -- does the edge hold at 5/10/20 days,
#       or is it a one-horizon coincidence?
#   10  Data-quality -- caution notes (large single-day move, missing
#       bhavcopy cross-check, etc.) pull the score down since they mean
#       "trust this a bit less," even if the backtest itself looks good.
QUALITY_WEIGHTS = {
    "hit_rate": 45,
    "sample_confidence": 20,
    "return_magnitude": 15,
    "consistency": 10,
    "data_quality": 10,
}

# Hit-rate scoring band: at/below this is "no better than a coin flip",
# at/above this is full marks for this component.
HIT_RATE_FLOOR = 0.50
HIT_RATE_CEILING = 0.85

# Sample-size scoring band: full marks at this many historical occurrences.
SAMPLE_SIZE_FOR_FULL_CONFIDENCE = 20

# Return-magnitude scoring band (at PRIMARY_HORIZON): full marks at/above this.
RETURN_MAGNITUDE_CEILING = 0.05  # 5%

QUALITY_LABELS = (
    (80, "Excellent"),
    (65, "Strong"),
    (50, "Moderate"),
    (35, "Weak"),
    (0, "Poor"),
)


def _clamp01(x):
    return max(0.0, min(1.0, x))


def _quality_label(score):
    for threshold, label in QUALITY_LABELS:
        if score >= threshold:
            return label
    return QUALITY_LABELS[-1][1]


def compute_quality_score(bt, dq_notes=None):
    """
    bt: the dict returned by backtest_pattern (or None / zero-sample).
    dq_notes: the data-quality caution notes list for this symbol
      (utils.nse_data.data_quality_check), or None.

    Returns None if there's no historical sample at all (score would be
    meaningless -- the caller should show "No historical sample yet"
    instead of a fabricated number), otherwise:
      {"score": int 0-100, "label": str, "breakdown": {component: int, ...}}
    """
    if not bt or not bt.get("horizons") or bt.get("sample_size", 0) == 0:
        return None

    primary = bt["horizons"].get(PRIMARY_HORIZON)
    if not primary:
        return None

    # -- Hit-rate component
    hit_rate_frac = _clamp01((primary["hit_rate"] - HIT_RATE_FLOOR) / (HIT_RATE_CEILING - HIT_RATE_FLOOR))

    # -- Sample confidence component
    sample_frac = _clamp01(bt["sample_size"] / SAMPLE_SIZE_FOR_FULL_CONFIDENCE)

    # -- Return magnitude component (negative avg return scores 0, not negative)
    return_frac = _clamp01(primary["avg_return"] / RETURN_MAGNITUDE_CEILING)

    # -- Cross-horizon consistency component
    horizons_available = [h for h in FORWARD_HORIZONS if h in bt["horizons"]]
    horizons_positive = [h for h in horizons_available if bt["horizons"][h]["hit_rate"] >= HIT_RATE_FLOOR]
    consistency_frac = (len(horizons_positive) / len(horizons_available)) if horizons_available else 0.0

    # -- Data-quality component (each caution note costs 35 points of this sub-score, floor 0)
    dq_frac = _clamp01(1.0 - 0.35 * len(dq_notes or []))

    components = {
        "hit_rate": hit_rate_frac * QUALITY_WEIGHTS["hit_rate"],
        "sample_confidence": sample_frac * QUALITY_WEIGHTS["sample_confidence"],
        "return_magnitude": return_frac * QUALITY_WEIGHTS["return_magnitude"],
        "consistency": consistency_frac * QUALITY_WEIGHTS["consistency"],
        "data_quality": dq_frac * QUALITY_WEIGHTS["data_quality"],
    }
    score = round(sum(components.values()))

    return {
        "score": score,
        "label": _quality_label(score),
        "breakdown": {k: round(v, 1) for k, v in components.items()},
    }


# -----------------------------------------------------------------------
# Setup Score -- a calibrated blend of the FOUR separately-computed reads
# a row already carries by the time scan_universe() is done with it:
#   - Quality Score        (this module)      -- has this pattern worked
#                                                  on this stock before?
#   - Confirmation ratio    (utils.breakout_confirmation) -- how many of
#                                                  today's independent
#                                                  checks agree?
#   - Failure Risk          (utils.breakout_failure)      -- how often
#                                                  has this exact setup
#                                                  round-tripped within
#                                                  FAILURE_WINDOW_DAYS?
#   - Risk:Reward           (utils.breakout_targets)      -- worth
#                                                  taking even if right?
#
# Today these sit in four separate columns with no combined read -- a
# row with an 82 Quality Score but only 4/8 Confirmation checks and an
# elevated historical Failure Rate looks strong on the first column and
# weak on the other two, and nothing forces those to be weighed
# together. This is a logistic blend (a bounded weighted sum through a
# sigmoid, same shape as a simple logistic-regression score) rather
# than a linear average, so a genuinely bad read on ANY one component
# pulls the composite down hard instead of being diluted by three good
# ones -- a 95 Quality Score paired with a coin-flip Confirmation ratio
# should NOT still land the composite near "Excellent".
#
# PURELY INFORMATIONAL, same as Quality/Confirmation themselves -- this
# does not change cleared_backtest / meets_confirmed_bar / R:R filter
# logic in breakout_controller.scan_universe. It's a fifth, additive
# column, not a new gate.
# -----------------------------------------------------------------------
SETUP_SCORE_WEIGHTS = {
    "quality": 2.2,          # backtest-derived Quality Score, rescaled to [-1, 1]
    "confirmation": 1.6,     # today's checklist ratio, rescaled to [-1, 1]
    "failure_risk": 1.4,     # historical same-pattern failure rate, inverted
    "risk_reward": 1.0,      # R:R ratio, rescaled around the preferred bar
}
SETUP_SCORE_BIAS = -0.4      # slight pessimistic prior: an average row on every
                              # component (all rescaled inputs at 0) should score
                              # a bit below 50, since staying out of a mediocre
                              # setup is the cheaper mistake to make


def _sigmoid(x: float) -> float:
    # Guard against overflow on pathological inputs -- inputs here are
    # already bounded to roughly [-1, 1] per component before weighting,
    # so this is a safety margin, not a normal code path.
    x = max(-50.0, min(50.0, x))
    return 1.0 / (1.0 + np.exp(-x))


def compute_setup_score(quality, confirmation, failure_risk, risk_reward):
    """
    quality: dict from compute_quality_score() above, or None.
    confirmation: ConfirmationScore from utils.breakout_confirmation, or None.
    failure_risk: FailureAssessment from utils.breakout_failure, or None.
    risk_reward: dict from utils.breakout_targets.compute_risk_reward(), or None.

    Each component is rescaled to roughly [-1, +1] (0 = "neutral/unknown",
    positive = "supportive", negative = "concerning") before being
    weighted and passed through a sigmoid, so a missing component
    contributes 0 (no opinion) rather than being penalized as if it had
    failed -- the same "an unavailable check shouldn't manufacture a
    false negative" fail-soft convention the rest of this project uses.

    Returns None only if EVERY component is unavailable (nothing to
    score), otherwise {"score": int 0-100, "label": str, "components_used": int}.
    """
    terms = []

    if quality is not None:
        # Quality Score is already 0-100 with 50 as its own "coin flip"
        # anchor (see HIT_RATE_FLOOR/CEILING banding above) -- rescale
        # around that same midpoint.
        terms.append(SETUP_SCORE_WEIGHTS["quality"] * ((quality["score"] - 50.0) / 50.0))

    if confirmation is not None and confirmation.available_count > 0:
        frac = confirmation.passed_count / confirmation.available_count
        terms.append(SETUP_SCORE_WEIGHTS["confirmation"] * ((frac - 0.5) * 2))

    if failure_risk is not None and failure_risk.backtest is not None and failure_risk.backtest.failure_rate is not None:
        # Invert: a LOW historical failure rate is supportive (+),
        # a HIGH one is concerning (-). Centered on
        # HIGH_FAILURE_RATE_CEILING so "elevated" reads clearly negative.
        rate = failure_risk.backtest.failure_rate
        terms.append(SETUP_SCORE_WEIGHTS["failure_risk"] * ((HIGH_FAILURE_RATE_CEILING - rate) / HIGH_FAILURE_RATE_CEILING))
    if failure_risk is not None and failure_risk.same_day.state.value == "Caution":
        # Same-day Caution (2/4 bull-trap flags -- short of the 3/4 that
        # would have pulled the row into Failed Breakout and out of
        # scoring entirely) still deserves a small penalty here, since
        # it's a live same-day signal, not a historical base rate.
        terms.append(-0.5)

    if risk_reward is not None and risk_reward.get("ratio") is not None:
        from utils.breakout_targets import RISK_REWARD_MIN_THRESHOLD, RISK_REWARD_PREFERRED_THRESHOLD
        ratio = risk_reward["ratio"]
        # 0 at the MIN threshold (barely investable), +1 at/above the
        # PREFERRED threshold, negative below MIN (this row would
        # normally already be filtered out of Confirmed/Watch by R:R,
        # but the score stays honest if ever called on an unfiltered row).
        span = RISK_REWARD_PREFERRED_THRESHOLD - RISK_REWARD_MIN_THRESHOLD
        terms.append(SETUP_SCORE_WEIGHTS["risk_reward"] * max(-1.0, min(1.0, (ratio - RISK_REWARD_MIN_THRESHOLD) / span)))

    if not terms:
        return None

    raw = SETUP_SCORE_BIAS + sum(terms)
    score = round(_sigmoid(raw) * 100)
    label = _quality_label(score)  # reuse the same Excellent/Strong/Moderate/Weak/Poor bands
    return {"score": score, "label": label, "components_used": len(terms)}