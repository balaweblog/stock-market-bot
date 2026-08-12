"""
utils/breakout_failure.py

False-breakout ("bull trap") detection, layered on top of whatever
pattern detector fired in utils.breakout_patterns -- same relationship
utils.entry_classification and utils.breakout_targets already have to
that module. A pattern firing says a breakout happened; utils.breakout_backtest
says how that pattern has historically paid off; this module says how
likely THIS specific occurrence is to simply fail -- get rejected and
handed straight back -- rather than follow through.

Two independent reads, on purpose (same separation-of-concerns as the
rest of this project):

  1. Same-day characteristics (assess_same_day_risk)
     Today's own bar, checked for the classic bull-trap fingerprints:
     weak/below-average volume (nothing actually confirmed the move),
     a long upper wick (price pushed higher intraday and was rejected),
     a weak close within the day's range, and giving back most of the
     day's high-to-open gain. No lookahead is even possible here --
     there's no "tomorrow" yet -- so this is a same-day risk read, not
     a prediction of tomorrow. Enough of these flags together earns the
     🔴 Failed Breakout label the same run the pattern fired.

  2. Historical failure-rate backtest (backtest_failure_rate)
     Replays the SAME detector across the stock's own trailing history
     that utils.breakout_backtest already does (reusing
     utils.breakout_backtest.find_historical_events so the two backtests
     are counting the exact same historical occurrences), and asks a
     different question of each one: not "did it make money by day 10",
     but "did price fall back below the breakout bar's own low within
     FAILURE_WINDOW_DAYS" -- i.e. gave the whole breakout back. That
     probability is what actually answers "how likely is THIS pattern,
     on THIS stock, to be a trap" -- a specific, replayable number
     instead of a generic "false breakouts happen sometimes" caveat.

Both reads are fail-soft, same convention as the rest of this project:
a data hiccup or too-thin a sample defaults to "not flagged" rather than
manufacturing a false alarm, and always says why in the reasons/notes.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from utils.breakout_backtest import find_historical_events, HIGH_FAILURE_RATE_CEILING
from utils.logger import log

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
LOW_VOLUME_CEILING = 1.0          # today's volume below this multiple of the trailing average = "nothing confirmed it"
VOLUME_AVG_LOOKBACK = 20
UPPER_WICK_MIN_RATIO = 0.35       # upper wick / day's range at/above this = "pushed higher and was rejected"
WEAK_CLOSE_CEILING = 0.40         # close position in the day's range at/below this = "closed weak"
GIVE_BACK_CEILING = 0.60          # (high-close)/(high-open) at/above this = "gave back most of the day's gain"
BULL_TRAP_MIN_FLAGS = 3           # of 4 same-day flags -- fires 🔴 Failed Breakout
CAUTION_MIN_FLAGS = 2             # of 4 -- fires ⚠️ Caution, short of the full bull-trap label

FAILURE_WINDOW_DAYS = 3           # "did it fail within N days" -- the window this module backtests
MIN_SAMPLES_FOR_FAILURE_RATE = 5  # below this, the historical rate is shown but not called reliable


class FailureRisk(Enum):
    CLEAN = "Clean"
    CAUTION = "Caution"
    FAILED_BREAKOUT = "Failed Breakout"


FAILURE_RISK_EMOJI = {
    FailureRisk.CLEAN: "🟢",
    FailureRisk.CAUTION: "⚠️",
    FailureRisk.FAILED_BREAKOUT: "🔴",
}

SAME_DAY_FLAG_LABELS = (
    "Below-average volume",
    "Long upper wick",
    "Weak close in range",
    "Gave back most of the day's gain",
)


@dataclass
class SameDayRisk:
    state: FailureRisk
    flags: Dict[str, bool] = field(default_factory=dict)   # label -> True/False
    score: int = 0                                          # count of True flags (0-4)
    detail: Dict[str, object] = field(default_factory=dict)  # raw numbers behind each flag
    reasons: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def label(self) -> str:
        return f"{FAILURE_RISK_EMOJI[self.state]} {self.state.value}"


@dataclass
class FailureBacktest:
    sample_size: int
    failure_count: int
    failure_rate: Optional[float]   # None if sample_size == 0
    confirmed_sample: bool          # sample_size >= MIN_SAMPLES_FOR_FAILURE_RATE
    elevated: bool                  # confirmed_sample and failure_rate >= HIGH_FAILURE_RATE_CEILING


@dataclass
class FailureAssessment:
    same_day: SameDayRisk
    backtest: Optional[FailureBacktest]   # None if there wasn't enough runway to attempt one

    @property
    def state(self) -> FailureRisk:
        return self.same_day.state

    def label(self) -> str:
        return self.same_day.label()

    @property
    def is_failed_breakout(self) -> bool:
        """True only on today's same-day characteristics -- this is what
        breakout_controller.py uses to pull a row out of Confirmed/Watch
        entirely, same CORE FILTER treatment as Risk:Reward. The
        historical failure-rate backtest is informational context shown
        alongside surviving rows, not itself a gate -- see module
        docstring."""
        return self.same_day.state == FailureRisk.FAILED_BREAKOUT


# -----------------------------------------------------------------------
# Same-day bull-trap characteristics
# -----------------------------------------------------------------------
def _volume_ratio(df, i, lookback=VOLUME_AVG_LOOKBACK):
    if i < lookback:
        return None
    avg_vol = df["Volume"].iloc[i - lookback:i].mean()
    if avg_vol is None or avg_vol <= 0:
        return None
    return float(df["Volume"].iloc[i]) / float(avg_vol)


def _bar_geometry(df, i):
    bar = df.iloc[i]
    o, h, l, c = float(bar["Open"]), float(bar["High"]), float(bar["Low"]), float(bar["Close"])
    day_range = h - l
    upper_wick_ratio = ((h - max(o, c)) / day_range) if day_range > 0 else 0.0
    close_position = ((c - l) / day_range) if day_range > 0 else 1.0
    give_back = ((h - c) / (h - o)) if h > o else 0.0
    return {
        "open": o, "high": h, "low": l, "close": c,
        "upper_wick_ratio": upper_wick_ratio,
        "close_position": close_position,
        "give_back": give_back,
    }


def assess_same_day_risk(symbol, df, i) -> SameDayRisk:
    """
    Checks TODAY's bar (df.iloc[i]) plus the trailing volume average
    (df.iloc[i-lookback:i], strictly before today) for the classic
    bull-trap fingerprints. No forward-looking data is possible or used
    here -- this is a same-day read, not a next-day prediction; "does
    this breakout already look like it's failing" is answerable the
    moment the bar closes.

    Returns a SameDayRisk with state CLEAN / CAUTION / FAILED_BREAKOUT
    based on how many of the 4 flags fired (see BULL_TRAP_MIN_FLAGS /
    CAUTION_MIN_FLAGS). Fails soft to CLEAN with a note if there isn't
    enough trailing history for the volume check -- the other 3 flags
    can still fire normally off today's own bar.
    """
    notes: List[str] = []
    detail: Dict[str, object] = {}

    geom = _bar_geometry(df, i)
    detail.update(geom)

    vol_ratio = _volume_ratio(df, i)
    if vol_ratio is None:
        notes.append(f"Not enough trailing history for the {VOLUME_AVG_LOOKBACK}-day volume average -- volume flag defaulted to False.")
        low_volume = False
    else:
        low_volume = vol_ratio < LOW_VOLUME_CEILING
    detail["volume_ratio"] = round(vol_ratio, 2) if vol_ratio is not None else None

    flags = {
        "Below-average volume": low_volume,
        "Long upper wick": geom["upper_wick_ratio"] >= UPPER_WICK_MIN_RATIO,
        "Weak close in range": geom["close_position"] <= WEAK_CLOSE_CEILING,
        "Gave back most of the day's gain": geom["give_back"] >= GIVE_BACK_CEILING,
    }
    score = sum(1 for v in flags.values() if v)

    if score >= BULL_TRAP_MIN_FLAGS:
        state = FailureRisk.FAILED_BREAKOUT
    elif score >= CAUTION_MIN_FLAGS:
        state = FailureRisk.CAUTION
    else:
        state = FailureRisk.CLEAN

    reasons = [label for label, fired in flags.items() if fired]
    if vol_ratio is not None:
        reasons_detail = [f"Volume {vol_ratio:.1f}x the {VOLUME_AVG_LOOKBACK}-day average"] if low_volume else []
    else:
        reasons_detail = []

    return SameDayRisk(
        state=state, flags=flags, score=score, detail=detail,
        reasons=reasons if reasons else ["No bull-trap characteristics on today's bar."],
        notes=notes,
    )


# -----------------------------------------------------------------------
# Historical failure-rate backtest
# -----------------------------------------------------------------------
def backtest_failure_rate(symbol, df, detector_fn, as_of_i, window_days=FAILURE_WINDOW_DAYS) -> Optional[FailureBacktest]:
    """
    Replays detector_fn across the stock's own trailing history via
    utils.breakout_backtest.find_historical_events() -- the SAME event
    set (same min_gap_days/max_events/max_horizon defaults) that
    feeds the hit-rate backtest shown alongside this in the email, so
    "of the occurrences reported above, X% failed within N days" is
    literally about the same occurrences, not a separately-sampled set.

    "Failed" here means: any close in the window (event_i, event_i +
    window_days] fell below that event's own breakout-bar LOW -- i.e.
    gave back the entire breakout day, not just a shallow pullback.
    This is a single, pattern-agnostic invalidation level that works
    the same way across all 8 detectors in utils.breakout_patterns
    without needing each one's own idea of "resistance".

    Returns None if there wasn't enough runway to even attempt this
    (mirrors utils.breakout_backtest.backtest_pattern's convention),
    otherwise a FailureBacktest with sample_size 0 if the pattern
    simply never fired historically.
    """
    if as_of_i < 60:
        return None

    try:
        event_indices = find_historical_events(df, detector_fn, as_of_i)
    except Exception as e:
        log.warning(f"Breakout Screener: failure-rate event scan failed for {symbol} ({detector_fn.__name__}): {e}")
        return None

    if not event_indices:
        return FailureBacktest(sample_size=0, failure_count=0, failure_rate=None, confirmed_sample=False, elevated=False)

    closes = df["Close"].values
    lows = df["Low"].values
    n = len(df)

    failure_count = 0
    counted = 0
    for ei in event_indices:
        breakout_low = lows[ei]
        if breakout_low <= 0:
            continue
        window_end = min(ei + window_days, n - 1)
        if window_end <= ei:
            continue
        window_closes = closes[ei + 1:window_end + 1]
        if len(window_closes) == 0:
            continue
        counted += 1
        if (window_closes < breakout_low).any():
            failure_count += 1

    if counted == 0:
        return FailureBacktest(sample_size=0, failure_count=0, failure_rate=None, confirmed_sample=False, elevated=False)

    failure_rate = failure_count / counted
    confirmed_sample = counted >= MIN_SAMPLES_FOR_FAILURE_RATE
    elevated = confirmed_sample and failure_rate >= HIGH_FAILURE_RATE_CEILING

    return FailureBacktest(
        sample_size=counted, failure_count=failure_count, failure_rate=failure_rate,
        confirmed_sample=confirmed_sample, elevated=elevated,
    )


# -----------------------------------------------------------------------
# Orchestrator -- what breakout_controller.py calls
# -----------------------------------------------------------------------
def evaluate_failure_risk(symbol, df, i, detector_fn) -> FailureAssessment:
    """
    symbol, df, i: same convention as everywhere else in this project.
    detector_fn: the exact detector function that fired today (from
      utils.breakout_patterns.PATTERN_DETECTOR_BY_NAME), so the
      historical replay is guaranteed to be the same pattern, not a
      re-derived approximation of it.

    Never raises -- a failure in either half never blocks the row it's
    attached to, same fail-soft pattern as utils.breakout_backtest.backtest_signal
    and utils.breakout_targets.compute_targets.
    """
    try:
        same_day = assess_same_day_risk(symbol, df, i)
    except Exception as e:
        log.warning(f"Breakout Screener: same-day failure-risk check failed for {symbol}: {e}")
        same_day = SameDayRisk(state=FailureRisk.CLEAN, notes=["Same-day risk check failed -- defaulted to Clean."])

    bt = None
    if detector_fn is not None:
        try:
            bt = backtest_failure_rate(symbol, df, detector_fn, i)
        except Exception as e:
            log.warning(f"Breakout Screener: failure-rate backtest failed for {symbol} ({detector_fn.__name__}): {e}")
            bt = None

    return FailureAssessment(same_day=same_day, backtest=bt)