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
     (utils.breakout_patterns.scan_all_patterns). Each detector first
     runs a shared set of guardrails (utils.breakout_patterns._passes_guardrails)
     before its own pattern-specific logic -- a data-sanity check (bad
     ticks, frozen/stale bars, impossible OHLC), a liquidity/penny-stock
     floor (min average price and turnover), and a corporate-action gap
     check (flags overnight gaps that line up with a common split/bonus
     ratio). These are baked into each detector function itself, not a
     wrapper around scan_all_patterns, so step 6's historical replay
     gets the identical gate a live signal would -- otherwise a stale
     feed, an illiquid stretch, or a split buried in a stock's own past
     would silently distort that stock's own backtest.
  6. For every pattern that fired, replay that SAME detector across the
     symbol's own trailing history (utils.breakout_backtest) to get a
     live hit-rate/avg-return read -- this is what "confirmed" means
     below, never a canned/pre-baked statistic. On top of that shared
     module's own Confirmed bar, this controller layers one more,
     stricter, TIERED bar (CONFIRMATION_TIERS below, Tier A/B/C by
     occurrence count with a hit-rate requirement that rises as sample
     size falls) -- a row has to clear the shared bar AND land in one of
     the tiers for a signal to reach Confirmed; a row that clears only
     the shared bar, or whose occurrence count is too thin to qualify
     for any tier ("Experimental"), is downgraded to Watch with a note
     (row["confirmation_tier_downgraded"]).
  7. Split into "Confirmed Breakouts" (backtest cleared the bar) and
     "Unconfirmed / Watch List" (pattern fired today but backtest sample
     was too thin or the historical hit-rate was weak) -- nothing is
     silently dropped, everything shown is labeled for what it is.
  8. Layer an Entry Classification on every signal (utils.entry_classification):
     not every breakout is an immediate buy. Each row is tagged Fresh
     Breakout, Retest, or (for symbols that haven't broken out yet but are
     close) Near Breakout -- with a concrete entry zone or trigger price.
     Retest is flagged as the preferred entry where it applies. This is a
     generic price-structure overlay (trailing-high resistance), independent
     of whichever specific pattern detector fired.
  9. For any row with an exact entry price (Fresh Breakout / Retest),
     compute three technical price targets (utils.breakout_targets):
     Target 1 (nearest prior swing-high resistance), Target 2 (the
     pattern's own measured move, e.g. cup depth or flagpole height
     projected from the breakout), and Target 3 (Fibonacci extension of
     that same base, or a further proven swing high, whichever is
     farther). This turns the backtest's average-return number into
     concrete price levels.
  10. Compute Risk/Reward for every row that has an entry price, a stop
      loss, and at least one target: reward is measured against the
      NEAREST available target, risk against the stop from step 8. This
      is a CORE FILTER, not decoration -- rows with R:R below
      RISK_REWARD_MIN_THRESHOLD (1.5) are pulled out of Confirmed/Watch
      into their own "Filtered -- Poor Risk/Reward" section rather than
      shown as if they were equally good, and R:R at/above
      RISK_REWARD_PREFERRED_THRESHOLD (2.0) is called out as worth
      prioritizing. Rows with no R:R data (missing entry/stop/target)
      are left alone rather than penalized for a gap elsewhere in the
      pipeline.
  11. Historical Failure Risk read (utils.breakout_failure): how often
      this exact pattern has fallen back below its own breakout-day low
      within utils.breakout_failure.FAILURE_WINDOW_DAYS days on this
      exact stock before -- informational context on every row, not a
      gate. (Earlier versions of this pipeline also ran a same-day
      bull-trap check -- below-average volume, a long upper wick, a weak
      close, giving back most of the day's gain -- and pulled flagged
      rows into a dedicated "🔴 Failed Breakout" section. That section
      has been removed; rows are no longer segregated on same-day
      characteristics and are bucketed by the normal
      backtest/Risk:Reward rules below like everything else.)
  12a. Fundamental quality gate, Confirmed Breakouts ONLY: for any row
      still on track for Confirmed after steps 10-11, fetch yfinance's
      .info for that symbol (lazily -- only for rows about to qualify,
      not the whole universe) and require ROE, Debt/Equity, and latest
      earnings growth to each clear a classic quality-stock bar where
      that metric is available. A row that clears the technical bar but
      fails an available fundamentals check is pulled into Watch List
      with a note, never dropped; a symbol with none of the three metrics
      available is left ungated (data gap, not a fail). This makes
      "Confirmed" mean high quality on the chart AND the balance sheet,
      not chart mechanics alone.
  12b. Extension Score / chase-risk gate (compute_extension_score, this
      controller): scores how far price has ALREADY moved today --
      <2% +10, 2-4% +8, 4-6% +5, 6-8% +2, 8-10% -5, >10% -15 -- and
      surfaces distance from the 20/50-day averages, distance above the
      breakout level, and ATR(14) multiples above that breakout level as
      supporting context on every fired signal. Confirmed Breakouts ONLY:
      a row is pulled into Watch List (never dropped, row["extension_downgraded"])
      if today's volume is >=2x its trailing 20-day average AND today's
      move already scores below 0 AND price sits at/above the breakout
      level -- the NATIONALUM-style case (strong backtest, but already
      extended too far on heavy volume to be a controlled entry) that
      this gate exists to catch. Also folded into Setup Strength (step
      12d) as one more weighted input.
  12b2. Extreme-volume / weak-close gate (also this controller, also
      Confirmed Breakouts ONLY, separate from 12b above): raw volume
      multiple alone isn't a reliability signal -- KPIL fired on 28.3x
      average volume with only a 62% hit-rate, +1.5% average return, and
      a 19% 3-day failure rate on that exact pattern/stock backtest.
      Extreme volume (>=5x avg -- see VOLUME_BANDS, distinct from the
      1.5-3x "Healthy" band) can mean institutional accumulation, news,
      a block deal, short covering, panic buying, or same-day
      exhaustion, and a threshold alone can't tell those apart, so it
      needs price/structure confirmation. A Confirmed row is pulled into
      Watch List (never dropped, row["extreme_volume_downgraded"]) if
      today's volume is >=5x its trailing 20-day average AND today's CLV
      (Close Location Value -- where the close sits in today's own
      high-low range) is below 0.50, i.e. the close gave back most of
      the day's range rather than confirming the move. A row with no
      intraday range to compute CLV from is left ungated (data gap, not
      a fail). Every row with volume data also shows its Volume band
      (Normal/Healthy/Elevated/Extreme) regardless of whether this gate
      fires, so a raw multiple never reads as unqualified strength.
  12c. Breakout Confirmation Score (utils.breakout_confirmation): a
      9-point checklist tally (price vs resistance, volume, close
      position, RSI band, moving averages, relative strength vs NIFTY,
      sector confirmation, the NIFTY regime read from step 6, and
      extension past resistance) attached to every fired signal as
      "(passed)/(available)" context -- e.g. "8/8" while sector data
      remains unavailable. Purely informational, same as the Quality
      Score -- doesn't change which bucket a row lands in.
  13. Render + send the email, in the same visual style as the other
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
from utils.breakout_backtest import (
    backtest_signal, compute_quality_score,
    PRIMARY_HORIZON, MIN_SAMPLES_FOR_CONFIDENCE, CONFIRM_HIT_RATE_THRESHOLD,
)
from utils.entry_classification import classify_entry, EntryState, ClassifierConfig
from utils.breakout_targets import (
    compute_targets, TargetsConfig, compute_risk_reward,
    RISK_REWARD_MIN_THRESHOLD, RISK_REWARD_PREFERRED_THRESHOLD,
)
from utils.market_regime import (
    compute_market_regime, MarketRegime, RegimeResult,
    CHECK_LABELS as REGIME_CHECK_LABELS, BEAR_MARKET_MIN_QUALITY_SCORE,
)
from utils.breakout_failure import (
    evaluate_failure_risk, FailureRisk,
    FAILURE_WINDOW_DAYS, MIN_SAMPLES_FOR_FAILURE_RATE, HIGH_FAILURE_RATE_CEILING,
)
from utils.breakout_confirmation import compute_confirmation_score, SECTOR_DATA_NOTE
from utils.constants import STOCKS

## Tiered statistical bar for Confirmed, layered ON TOP of
# utils.breakout_backtest's own MIN_SAMPLES_FOR_CONFIDENCE /
# CONFIRM_HIT_RATE_THRESHOLD bar (imported above). This is deliberately a
# SEPARATE and additional check, not a replacement -- other callers of
# the shared backtest module still get its own default bar; only this
# controller's Confirmed bucket also has to clear one of the tiers below.
#
# A single flat "≥8 samples, ≥60%" bar (the previous version of this)
# still let a threadbare-but-lucky sample (e.g. 10/15 = 67%) sit right
# next to a deep, proven one (e.g. 40/60 = 67%) with no way to tell them
# apart at a glance -- and a hit-rate bar alone doesn't compensate for
# how much noisier a small sample is. Tiering the hit-rate requirement
# UP as the sample size goes DOWN (stricter hit-rate demanded from a
# thinner sample) directly addresses that: a thin sample has to be
# unambiguously strong to earn Confirmed, not just clear the same bar a
# 40-occurrence sample would. Below CONFIRMATION_TIER_MIN_SAMPLES there's
# no tier that qualifies at all -- that occurrence count is grouped under
# "Experimental" and can only ever reach Watch List, regardless of how
# high its raw hit-rate happens to be (this is what stops a 100%/1-occurrence
# row from reading as stronger than a 79%/14-occurrence one).
#
# Evaluated in listed order; a row qualifies for the FIRST tier whose
# sample-size band it falls into, and then has to clear THAT tier's own
# hit-rate bar (see classify_confirmation_tier below).
CONFIRMATION_TIERS = (
    # (label, min_samples, max_samples_or_None, min_hit_rate)
    ("Tier A", 30, None, 0.60),
    ("Tier B", 15, 29, 0.60),
    ("Tier C", 10, 14, 0.65),
)
CONFIRMATION_TIER_MIN_SAMPLES = min(t[1] for t in CONFIRMATION_TIERS)  # 10 -- floor of the lowest tier
EXPERIMENTAL_LABEL = f"Experimental (<{CONFIRMATION_TIER_MIN_SAMPLES} occurrences)"


def classify_confirmation_tier(sample_size, hit_rate):
    """Returns the tier label ('Tier A' / 'Tier B' / 'Tier C') a row's
    backtest occurrence count and hit-rate qualify for, or None if it
    doesn't clear any tier -- either because sample_size falls below
    CONFIRMATION_TIER_MIN_SAMPLES (Experimental), or because the
    hit-rate doesn't clear the bar for the tier its sample size falls
    into. A row is evaluated against exactly ONE tier (the one its
    sample size falls into), not a ladder it has to climb -- a 12-sample,
    70% row is Tier C on its own merits, it doesn't also need to clear
    Tier A/B's higher sample-size floors."""
    if sample_size is None or hit_rate is None:
        return None
    for label, min_n, max_n, min_hr in CONFIRMATION_TIERS:
        if sample_size >= min_n and (max_n is None or sample_size <= max_n) and hit_rate >= min_hr:
            return label
    return None

SERIF = "Georgia, 'Times New Roman', serif"
SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"

HISTORY_PERIOD = "2y"
BATCH_SIZE = 40          # symbols per yfinance batch download
BATCH_PAUSE_SECONDS = 1  # be polite to the endpoint between batches
MAX_CONFIRMED_ROWS = 25  # keep the email skimmable even on a big breakout day
MAX_WATCH_ROWS = 15
MAX_NEAR_BREAKOUT_ROWS = 15
MAX_RETEST_ROWS = 15
MAX_FILTERED_RR_ROWS = 15

ENTRY_CLASSIFIER_CONFIG = ClassifierConfig()
TARGETS_CONFIG = TargetsConfig()

# -----------------------------------------------------------------------
# Fundamental quality gate -- Confirmed Breakouts ONLY.
#
# The rest of this pipeline is purely technical (price/volume/pattern
# mechanics, backtested against the stock's own history). This gate adds
# one more requirement before a row is allowed to sit in the "Confirmed"
# bucket: the underlying business has to look financially sound too, not
# just the chart. It is fetched from yfinance's .info (P/E, ROE,
# debt/equity, earnings growth, margins) -- the same "classic quality
# stock" screen: low debt, strong return on equity, growing earnings.
#
# Fail-soft, same convention as every other filter in this file:
#   - Fetched LAZILY, only for rows that already cleared the technical
#     backtest bar (i.e. rows that would otherwise land in Confirmed) --
#     not for the whole NIFTY 500 universe, to keep the run fast.
#   - If yfinance has NONE of the three metrics for a symbol, the gate is
#     treated as unavailable and does NOT block the row -- a data gap
#     elsewhere in the pipeline never silently disqualifies a signal.
#   - If AT LEAST ONE metric is available and it fails the bar, the row
#     is pulled out of Confirmed into Watch List (never dropped), same as
#     the Bearish-regime downgrade -- see row["fundamental_downgraded"].
FUNDAMENTAL_MIN_ROE = 0.15                 # Return on Equity >= 15%
FUNDAMENTAL_MAX_DEBT_TO_EQUITY = 100.0     # Debt/Equity <= 100 (yfinance reports this as a percentage, e.g. 45.2 == 45.2%)
FUNDAMENTAL_MIN_EARNINGS_GROWTH = 0.0      # Latest earnings growth must be non-negative, not shrinking


def _fetch_fundamentals_info(symbol, cache):
    """Raw yfinance .info dict for one symbol, cached per run. Fail-soft --
    returns {} (not None) on any fetch error so callers don't need a
    separate None-check path; an empty dict is treated the same as "no
    metrics available" by evaluate_fundamental_quality."""
    if symbol in cache:
        return cache[symbol]
    info = {}
    try:
        info = yf.Ticker(f"{symbol}.NS").info or {}
    except Exception as e:
        log.warning(f"Breakout Screener: fundamentals fetch failed for {symbol}: {e}")
    cache[symbol] = info
    return info


def evaluate_fundamental_quality(symbol, cache):
    """Returns a dict:
      available: True if yfinance returned at least one of the three
        metrics for this symbol.
      passed: True only if available AND every metric that WAS returned
        clears its threshold (missing individual metrics don't count
        against the row, same fail-soft principle as Risk:Reward).
      checks: list of (label, passed, display_value) for the metrics that
        were available, for the email to show its work.
    """
    info = _fetch_fundamentals_info(symbol, cache)
    roe = info.get("returnOnEquity")
    debt_to_equity = info.get("debtToEquity")
    earnings_growth = info.get("earningsGrowth")
    if earnings_growth is None:
        earnings_growth = info.get("earningsQuarterlyGrowth")

    checks = []
    if roe is not None:
        checks.append(("ROE", roe >= FUNDAMENTAL_MIN_ROE, f"{roe*100:.1f}%"))
    if debt_to_equity is not None:
        checks.append(("Debt/Equity", debt_to_equity <= FUNDAMENTAL_MAX_DEBT_TO_EQUITY, f"{debt_to_equity:.0f}%"))
    if earnings_growth is not None:
        checks.append(("Earnings growth", earnings_growth >= FUNDAMENTAL_MIN_EARNINGS_GROWTH, f"{earnings_growth*100:+.1f}%"))

    available = bool(checks)
    passed = available and all(ok for _, ok, _ in checks)
    return {"available": available, "passed": passed, "checks": checks}

# -----------------------------------------------------------------------
# Extension Score -- "how far has price already moved today" penalty.
#
# Two rows can carry identical Quality/Confirmation/Risk:Reward reads and
# still be very different trades: one is still sitting right at its
# trigger, the other has already run 8-10% intraday on the news. The
# second is a chase, not a breakout entry, no matter how clean the
# backtest looks -- e.g. NATIONALUM: strong hit-rate on paper, but
# already extended enough that entering AT the signal price meant paying
# up near the day's high with little room left before the next pullback.
#
# This is computed directly off the stock's own OHLCV history -- no
# dependency on any other util module -- and is entirely self-contained
# and fail-soft: any row missing enough history for a given sub-metric
# just gets None for that piece rather than blocking the rest.
#
# EXTENSION_SCORE_BANDS: today's Close-over-Close % move -> points.
# First band whose upper bound the move falls under wins; the final
# (None, ...) entry is the catch-all for anything above the highest
# named bound.
EXTENSION_SCORE_BANDS = (
    (2.0, 10),
    (4.0, 8),
    (6.0, 5),
    (8.0, 2),
    (10.0, -5),
    (None, -15),
)
# EXTENSION_LABELS: points -> label, highest matching lower-bound wins.
EXTENSION_LABELS = (
    (10, "Fresh"),
    (8, "Controlled"),
    (5, "Extended"),
    (2, "Highly Extended"),
    (-5, "Chase Risk"),
    (float("-inf"), "Severe Chase Risk"),
)
EXTENSION_ATR_PERIOD = 14
EXTENSION_MA_PERIODS = (20, 50)  # distance-from-DMA context, not separately scored

# CLV_BANDS: Close Location Value -> label. CLV = (Close - Low) / (High -
# Low), i.e. where today's close sits within today's own range (1.0 =
# closed AT the high, 0.0 = closed AT the low). A breakout that closes
# near the day's high is a much more convincing signal than one that
# gives back most of the intraday move before the close -- the quantified
# version of the "close near high" read the confirmation checklist does
# qualitatively (see utils.breakout_confirmation, not this file). Highest
# matching lower-bound wins.
CLV_BANDS = (
    (0.80, "Excellent"),
    (0.65, "Good"),
    (0.50, "Neutral"),
    (float("-inf"), "Weak"),
)

# VOLUME_BANDS: today's volume vs trailing 20-day average -> label.
# Raw volume multiple alone doesn't tell you whether a breakout is
# stronger -- a KPIL-style 28.3x day can still carry a mediocre backtest
# (62% hit-rate, +1.5% avg return, 19% 3-day failure), because extreme
# volume is ambiguous on its own: it can mean institutional accumulation,
# a genuine news-driven re-rating, a block deal, short covering, panic
# buying, or same-day exhaustion/distribution -- and a raw multiple can't
# tell those apart. This labels the multiple so a reader isn't reading
# "28.3x" as automatically bullish; EXTREME_VOLUME_MULTIPLE below turns
# it into an actual gate, gated on CLV (price/structure), not the
# multiple alone. Highest matching lower-bound wins.
VOLUME_BANDS = (
    (5.0, "Extreme"),
    (3.0, "Elevated"),
    (1.5, "Healthy"),
    (float("-inf"), "Normal"),
)


# -----------------------------------------------------------------------
# VWAP / Anchored VWAP -- "price above resistance" doesn't say whether
# the volume actually transacting in this stock is, on average, paying
# UP for it or not. VWAP answers that.
#
# IMPORTANT LIMITATION: this pipeline pulls DAILY OHLCV from yfinance
# (see HISTORY_PERIOD above), not intraday tick/volume data, so there is
# no true intraday SESSION VWAP here -- a classic day-trading VWAP
# resets every session and needs intraday bars to build. What's below
# is two DAILY-bar VWAP reads instead, both legitimate, both commonly
# used by swing/position traders on a daily chart, but neither is "the"
# session VWAP a day-trading platform would show:
#   trailing_vwap  -- a VWAP_TRAILING_PERIOD-day volume-weighted average
#     of the daily typical price ((H+L+C)/3), i.e. "has recent volume
#     been transacting above or below this stock's own recent average
#     price" -- a rolling context read, not scored, shown as supporting
#     detail only.
#   anchored_vwap  -- the SAME calculation, but anchored from the
#     breakout base rather than a fixed rolling window: the ANCHOR bar
#     is the most recent day price closed back above breakout_level,
#     found by scanning back up to VWAP_ANCHOR_LOOKBACK_DAYS bars (falls
#     back to the oldest bar in that window if price never closed below
#     breakout_level within it, and collapses to TODAY itself on a Fresh
#     Breakout, where "since the breakout" correctly IS today -- see
#     compute_vwap_context below for the exact anchor-detection logic).
#     This is the "Anchored VWAP from breakout base" read, and the one
#     folded into Trade Quality / Best Execution Case below -- price
#     holding above the volume-weighted average cost paid since the
#     breakout is a much stronger read than price above resistance
#     alone, especially on a fresh breakout where there's barely any
#     trading history above the level yet to weigh against.
# Anchor detection is derived purely from this stock's own daily bars
# and breakout_level -- same "nothing guessed, everything replayable"
# convention as the rest of this file -- NOT read from
# utils.entry_classification's own internal breakout-date bookkeeping
# (not exposed to this controller), so on a Retest that pulled back
# without ever closing below breakout_level, the anchor may land a few
# bars later than the pattern's true first breakout bar. Documented
# rather than hidden -- see anchor_index/anchor_date in the returned dict.
VWAP_TRAILING_PERIOD = 20
VWAP_ANCHOR_LOOKBACK_DAYS = 60  # how far back to search for the breakout-base anchor bar

# VWAP_LABEL_COLORS: label -> (background, text) -- same greener-is-
# better badge language as Extension/RS/Quality. "Above Both" is the
# strong read this section exists to catch (price holding above BOTH
# its own recent average cost AND its volume-weighted average cost
# since the breakout); "Below Both" is the weak read (volume has, on
# net, been paying UP for a lower price than today's).
VWAP_LABEL_COLORS = {
    "Above Both": ("#DCEFE0", "#0f5132"),
    "Above Anchored Only": ("#E4F0E9", "#3d7a52"),
    "Above Trailing Only": ("#FCF1D8", "#7a5b00"),
    "Below Both": ("#F8DADA", "#8a1c1c"),
    # Single-figure labels -- used when only one of the two VWAP reads
    # was available for this row (e.g. too little history yet for the
    # trailing window), so the badge never claims more than it knows.
    "Above Anchored": ("#E4F0E9", "#3d7a52"),
    "Below Anchored": ("#F8DADA", "#8a1c1c"),
    "Above Trailing": ("#FCF1D8", "#7a5b00"),
    "Below Trailing": ("#F8DADA", "#8a1c1c"),
}


# -----------------------------------------------------------------------
# Relative Strength vs NIFTY -- "is the stock breaking resistance?" is
# only half the question; "is it outperforming the index while it does
# it?" is the other half. A +5% breakout on a day NIFTY is flat is a
# genuinely strong stock. A +5% breakout on a day NIFTY is +4.8% is just
# the whole market moving -- the stock isn't showing any real leadership,
# it's along for the ride.
#
# Computed as each stock's own trailing % return over a lookback window
# MINUS NIFTY's own % return over the SAME calendar window (not the same
# row-position window -- see _nifty_close_asof below), at 20-day and
# 50-day lookbacks. Positive = outperforming, negative = lagging.
# Self-contained aside from one shared input: nifty_series, a single
# {date: Close} series fetched ONCE per run (see fetch_nifty_index_history)
# and passed to every symbol's read, same one-fetch-many-reads pattern as
# `regime` above -- not a per-symbol yfinance call.
NIFTY_INDEX_TICKER = "^NSEI"
# (lookback_days, blend_weight) -- 20d weighted higher than 50d since a
# breakout is a near-term timing signal; RS should mostly answer "is it
# outperforming RIGHT NOW", with the 50d read as slower-moving context.
RS_LOOKBACKS = (
    (20, 60),
    (50, 40),
)
# RS_BANDS: blended (weighted-average) outperformance in percentage
# points vs NIFTY over the lookbacks above -> label. Highest matching
# lower-bound wins. Deliberately NOT zero-centered at a hair-trigger --
# the whole point of this factor is catching stocks that are only
# "breaking out" because the index is; a couple points of daily/weekly
# noise either side of flat isn't real leadership OR real laggardness.
RS_BANDS = (
    (5.0, "Strong Outperformance"),
    (1.5, "Outperforming"),
    (-1.5, "In-Line With NIFTY"),
    (-5.0, "Underperforming"),
    (float("-inf"), "Weak / Lagging"),
)

# -----------------------------------------------------------------------
# Sector Confirmation -- "the stock broke out" is a weaker signal than
# "the stock broke out AND its sector is confirming": a breakout on a
# stock whose peers are all going nowhere is much more likely to be a
# single-name news pop that fades once the reason for it does, versus a
# breakout that's part of the sector actually turning. Four checks:
#   Sector > 20DMA / Sector > 50DMA -- is the sector's own trend healthy?
#   Sector relative strength        -- is the SECTOR outperforming NIFTY
#                                       (same idea as the stock-level
#                                       Relative Strength above, one level
#                                       up)?
#   Sector breadth                  -- what fraction of the sector's own
#                                       constituents are individually
#                                       trending healthy (above their own
#                                       20DMA) right now, i.e. is this a
#                                       broad move or a couple of names
#                                       dragging the average up?
#
# REQUIRES SECTOR_MAP: {symbol: sector_name}, one entry per NIFTY 500
# symbol. NOT populated in this build -- there is no sector-classification
# data anywhere in this codebase to draw from (this is the exact gap
# SECTOR_DATA_NOTE, imported from utils.breakout_confirmation, already
# flags for the OTHER confirmation checklist elsewhere in this report).
# Populate SECTOR_MAP below (e.g. from NSE's own sector/industry
# classification, or wherever utils.constants.STOCKS itself was sourced)
# and every check below turns on automatically -- nothing else needs to
# change. Until then every row's sector_confirmation comes back None and
# the report says so, same fail-soft "N/A, and it says why" convention as
# the rest of this module.
#
# Deliberately NOT an extra yfinance fetch per sector -- the sector
# "index" here is a synthetic, equal-weighted, rebased-to-100 average of
# whatever stocks in SECTOR_MAP land in `histories` this run (built once
# in build_sector_context, below), reusing data already being fetched
# for the main scan rather than pulling NSE's official sector indices
# separately. Good enough for a directional read; if you'd rather use the
# real NSE sector indices (e.g. NIFTY BANK, NIFTY IT), say so and this
# swaps for per-sector yfinance tickers instead.
SECTOR_MAP = {
    # "RELIANCE": "Oil & Gas", "TCS": "IT", "HDFCBANK": "Banking", ...
    # symbol -> sector name, one entry per NIFTY 500 symbol.
}
SECTOR_DATA_NOTE_LOCAL = (
    "no sector-classification map is loaded (SECTOR_MAP is empty) -- "
    "add a {symbol: sector} entry per NIFTY 500 symbol to turn this on"
)
SECTOR_MIN_CONSTITUENTS = 3   # a sector index built from fewer names than this is too noisy to trust -- dropped
SECTOR_BREADTH_SMA_PERIOD = 20  # "trending healthy" for breadth purposes = above its own 20DMA
SECTOR_RS_LOOKBACK = 20         # matches Relative Strength's own 20d window, for a consistent read
SECTOR_CONFIRMATION_WEIGHTS = {
    "above_20dma": 25,
    "above_50dma": 25,
    "relative_strength": 30,
    "breadth": 20,
}

# Chase-risk flag: the specific combination this section exists to catch
# -- a pattern fired on well-above-average volume AND price is already
# deep in a penalty band (score < CHASE_RISK_MAX_SCORE, i.e. the
# 8-10%/>10% bands) AND price is at/above the breakout level itself. All
# three have to hold together; any one alone is normal breakout
# behavior, not a chase signal. A flagged row is pulled OUT of Confirmed
# into Watch List, same fail-soft convention as the fundamentals/regime
# gates above -- never silently dropped, always labeled why
# (row["extension_downgraded"]).
CHASE_RISK_VOLUME_MULTIPLE = 2.0   # today's volume vs trailing 20-day average
CHASE_RISK_MAX_SCORE = 0           # today's move score must be BELOW this to flag chase risk

# Extreme-volume / weak-close gate: a SEPARATE gate from chase-risk above,
# catching a different failure mode. Chase-risk is about a move that's
# already run too far to enter today; this one is about whether truly
# extreme volume (>=5x, see VOLUME_BANDS) actually represents demand, or
# distribution/exhaustion dressed up as a breakout -- the KPIL case: heavy
# volume on the day, but the close gave back most of the range, and the
# backtest on that exact pattern/stock combo was mediocre (62% hit-rate,
# +1.5% avg, 19% 3-day failure). Raw volume multiple can't tell "real
# demand" apart from "panic buying that reverses" or "someone selling into
# the pop" -- CLV (Close Location Value, see CLV_BANDS) is the
# price/structure read that can: a close that gives back most of the day's
# range on extreme volume is the classic distribution/exhaustion signature,
# not confirmation. A row is only flagged if CLV is actually available for
# the day (today's High > Low) -- a missing CLV is a data gap, not a fail,
# same fail-soft convention as the fundamentals gate.
EXTREME_VOLUME_MULTIPLE = 5.0       # today's volume vs trailing 20-day average
EXTREME_VOLUME_MIN_CLV = 0.50       # CLV must clear this (Neutral-or-better close) to confirm extreme volume


def compute_extension_score(symbol, df, i, breakout_level):
    """Extension read for one fired signal. Returns a dict:
      today_move_pct: today's Close-over-Close % change
      score: points from EXTENSION_SCORE_BANDS for that move (+10..-15)
      label: EXTENSION_LABELS bucket for that score
      dist_20dma_pct / dist_50dma_pct: % distance of today's close from
        the trailing 20/50-day SMA (None if not enough history yet)
      dist_breakout_pct: % distance of today's close above breakout_level
        (None if breakout_level wasn't available -- e.g. no entry price
        AND no signal price, which shouldn't happen in practice but is
        handled rather than assumed)
      atr: 14-day ATR in price terms (None if not enough history)
      atr_multiple_above_breakout: (close - breakout_level) / atr, i.e.
        how many average-true-ranges above the breakout level today's
        close already sits (None if either input is unavailable)
      volume_multiple: today's volume vs trailing 20-day average volume
        (None if not enough history)
      volume_label: VOLUME_BANDS bucket for that multiple (None if
        volume_multiple is None) -- Normal / Healthy / Elevated / Extreme
      clv: Close Location Value, (Close - Low) / (High - Low) for
        today's bar -- 1.0 means today closed AT the high, 0.0 means it
        closed AT the low. None if today's High == Low (e.g. a halted
        or illiquid session with no range to locate the close within).
      clv_label: CLV_BANDS bucket for that value (None if clv is None)
      chase_risk: see CHASE_RISK_* above
    Returns None only if there isn't even a previous close to measure
    today's move against (i.e. i < 1)."""
    if i < 1 or i >= len(df):
        return None

    close = float(df["Close"].iloc[i])
    prev_close = float(df["Close"].iloc[i - 1])
    if prev_close == 0:
        return None
    today_move_pct = (close - prev_close) / prev_close * 100.0

    score = next(pts for upper, pts in EXTENSION_SCORE_BANDS if upper is None or today_move_pct < upper)
    label = next(lbl for lower, lbl in EXTENSION_LABELS if score >= lower)

    def _dist_from_sma(period):
        if i + 1 < period:
            return None
        sma = float(df["Close"].iloc[i - period + 1:i + 1].mean())
        if sma == 0:
            return None
        return (close - sma) / sma * 100.0

    dist_20dma_pct = _dist_from_sma(20)
    dist_50dma_pct = _dist_from_sma(50)

    dist_breakout_pct = None
    if breakout_level:
        dist_breakout_pct = (close - breakout_level) / breakout_level * 100.0

    atr = None
    if i >= EXTENSION_ATR_PERIOD:
        highs = df["High"].iloc[i - EXTENSION_ATR_PERIOD + 1:i + 1]
        lows = df["Low"].iloc[i - EXTENSION_ATR_PERIOD + 1:i + 1]
        prior_closes = df["Close"].iloc[i - EXTENSION_ATR_PERIOD:i]
        true_ranges = [
            max(
                float(highs.iloc[j]) - float(lows.iloc[j]),
                abs(float(highs.iloc[j]) - float(prior_closes.iloc[j])),
                abs(float(lows.iloc[j]) - float(prior_closes.iloc[j])),
            )
            for j in range(len(highs))
        ]
        if true_ranges:
            atr = sum(true_ranges) / len(true_ranges)

    atr_multiple_above_breakout = None
    if atr and breakout_level:
        atr_multiple_above_breakout = (close - breakout_level) / atr

    volume_multiple = None
    volume_label = None
    if "Volume" in df.columns and i >= 20:
        avg_vol = float(df["Volume"].iloc[i - 20:i].mean())
        today_vol = float(df["Volume"].iloc[i])
        if avg_vol > 0:
            volume_multiple = today_vol / avg_vol
            volume_label = next(lbl for lower, lbl in VOLUME_BANDS if volume_multiple >= lower)

    today_high = float(df["High"].iloc[i])
    today_low = float(df["Low"].iloc[i])
    clv = None
    clv_label = None
    if today_high > today_low:
        clv = (close - today_low) / (today_high - today_low)
        clv_label = next(lbl for lower, lbl in CLV_BANDS if clv >= lower)

    chase_risk = bool(
        volume_multiple is not None and volume_multiple >= CHASE_RISK_VOLUME_MULTIPLE
        and score < CHASE_RISK_MAX_SCORE
        and (breakout_level is None or close >= breakout_level)
    )

    return {
        "today_move_pct": today_move_pct,
        "score": score,
        "label": label,
        "dist_20dma_pct": dist_20dma_pct,
        "dist_50dma_pct": dist_50dma_pct,
        "dist_breakout_pct": dist_breakout_pct,
        "atr": atr,
        "atr_multiple_above_breakout": atr_multiple_above_breakout,
        "volume_multiple": volume_multiple,
        "volume_label": volume_label,
        "clv": clv,
        "clv_label": clv_label,
        "chase_risk": chase_risk,
    }


def _volume_weighted_avg_price(df, start_idx, end_idx):
    """Volume-weighted average of the daily typical price ((H+L+C)/3)
    over df.iloc[start_idx:end_idx+1] (inclusive both ends). Returns
    None if the slice is empty or its total volume is 0 (e.g. a halted
    stretch) -- never divides by zero."""
    if start_idx > end_idx or start_idx < 0 or end_idx >= len(df):
        return None
    segment = df.iloc[start_idx:end_idx + 1]
    volume = segment["Volume"].astype(float)
    total_volume = float(volume.sum())
    if total_volume <= 0:
        return None
    typical_price = (
        segment["High"].astype(float) + segment["Low"].astype(float) + segment["Close"].astype(float)
    ) / 3.0
    return float((typical_price * volume).sum() / total_volume)


def compute_vwap_context(symbol, df, i, breakout_level):
    """VWAP / Anchored VWAP read for one fired signal -- see the VWAP
    block comment above (VWAP_TRAILING_PERIOD/VWAP_ANCHOR_LOOKBACK_DAYS)
    for the daily-bar-vs-intraday-session caveat. Returns a dict:
      trailing_vwap: VWAP_TRAILING_PERIOD-day volume-weighted typical
        price (None if not enough history yet).
      anchored_vwap: volume-weighted typical price from the detected
        breakout-base anchor bar through today (None if breakout_level
        wasn't available, same as Extension Score's own breakout_level
        handling).
      anchor_index / anchor_date: which bar in `df` was used as the
        anchor for anchored_vwap (None if anchored_vwap is None).
      price_above_trailing_vwap / price_above_anchored_vwap: bool|None,
        None wherever the corresponding VWAP figure itself is None.
      label: VWAP_LABEL_COLORS bucket -- "Above Both" / "Above Anchored
        Only" / "Above Trailing Only" / "Below Both" when BOTH figures
        are available; "Above Anchored" / "Below Anchored" / "Above
        Trailing" / "Below Trailing" when only ONE is (never overstates
        "Both" on partial data); None if NEITHER figure was available.
    Returns None only if there isn't a valid bar to evaluate (i < 0 or
    i >= len(df)) -- unlike Extension Score this does NOT require i>=1,
    since a Fresh Breakout's anchor collapses to day i itself."""
    if i < 0 or i >= len(df):
        return None

    close = float(df["Close"].iloc[i])

    trailing_vwap = None
    if i + 1 >= VWAP_TRAILING_PERIOD:
        trailing_vwap = _volume_weighted_avg_price(df, i - VWAP_TRAILING_PERIOD + 1, i)

    anchored_vwap = None
    anchor_index = None
    if breakout_level:
        # Scan backward for the most recent bar that closed BELOW
        # breakout_level -- the anchor is the bar immediately after it
        # (the first bar that closed back above, i.e. the breakout bar
        # itself). Bounded to VWAP_ANCHOR_LOOKBACK_DAYS so a stock that
        # has sat above an old, stale resistance level for a long time
        # doesn't anchor absurdly far back.
        lookback_floor = max(0, i - VWAP_ANCHOR_LOOKBACK_DAYS)
        anchor_index = lookback_floor  # default: never dropped below within the window
        for j in range(i - 1, lookback_floor - 1, -1):
            if float(df["Close"].iloc[j]) < breakout_level:
                anchor_index = j + 1
                break
        anchored_vwap = _volume_weighted_avg_price(df, anchor_index, i)
        if anchored_vwap is None:
            anchor_index = None

    price_above_trailing_vwap = (close > trailing_vwap) if trailing_vwap is not None else None
    price_above_anchored_vwap = (close > anchored_vwap) if anchored_vwap is not None else None

    # "Both" labels only when BOTH figures are actually available -- a
    # row with only one figure available (e.g. too little history for
    # the trailing window) gets a single-figure label instead, rather
    # than "Below Both" overstating what's actually known.
    label = None
    if price_above_anchored_vwap is not None and price_above_trailing_vwap is not None:
        if price_above_anchored_vwap and price_above_trailing_vwap:
            label = "Above Both"
        elif price_above_anchored_vwap:
            label = "Above Anchored Only"
        elif price_above_trailing_vwap:
            label = "Above Trailing Only"
        else:
            label = "Below Both"
    elif price_above_anchored_vwap is not None:
        label = "Above Anchored" if price_above_anchored_vwap else "Below Anchored"
    elif price_above_trailing_vwap is not None:
        label = "Above Trailing" if price_above_trailing_vwap else "Below Trailing"

    return {
        "trailing_vwap": trailing_vwap,
        "anchored_vwap": anchored_vwap,
        "anchor_index": anchor_index,
        "anchor_date": df.index[anchor_index] if anchor_index is not None else None,
        "price_above_trailing_vwap": price_above_trailing_vwap,
        "price_above_anchored_vwap": price_above_anchored_vwap,
        "label": label,
    }


def _nifty_close_asof(nifty_series, date):
    """NIFTY's Close on `date`, or the most recent prior trading date if
    `date` itself isn't in the series (index/stock trading calendars can
    differ by a session here and there). Date-based lookup, not
    positional -- deliberately NOT just df.index[i - period] on the
    NIFTY series, since a handful of calendar mismatches over a 20-50 day
    window would silently misalign stock and index returns. Returns None
    if there's nothing on or before `date` (e.g. date predates the
    fetched history, or nifty_series itself is None/empty)."""
    if nifty_series is None or nifty_series.empty:
        return None
    try:
        val = nifty_series.asof(date)
    except Exception:
        return None
    if val is None or (isinstance(val, float) and val != val):  # NaN check w/o importing math
        return None
    return float(val)


def compute_relative_strength(symbol, df, i, nifty_series):
    """Relative Strength read for one fired signal -- see the
    Relative-Strength-vs-NIFTY block comment above (RS_LOOKBACKS/
    RS_BANDS) for the reasoning. Returns a dict:
      periods: {20: {...}, 50: {...}} -- one entry per RS_LOOKBACKS
        lookback, each {stock_pct, nifty_pct, rel_pct} (any of which can
        be None if there isn't enough history yet, or the NIFTY
        benchmark wasn't available for that date)
      score: blended rel_pct across whichever periods have data,
        weighted per RS_LOOKBACKS (renormalized over just the available
        periods, same fail-soft convention as Setup Strength's own
        sub-score blending) -- None if NEITHER period has data
      label: RS_BANDS bucket for that blended score (None if score is None)
    Returns None only if there isn't even a previous close to measure
    today's move against (i.e. i < 1), or nifty_series wasn't available
    at all this run."""
    if i < 1 or i >= len(df) or nifty_series is None:
        return None

    close = float(df["Close"].iloc[i])
    date_today = df.index[i]
    nifty_today = _nifty_close_asof(nifty_series, date_today)

    periods = {}
    weighted_sum = 0.0
    weight_total = 0.0
    for lookback, weight in RS_LOOKBACKS:
        if i - lookback < 0:
            periods[lookback] = {"stock_pct": None, "nifty_pct": None, "rel_pct": None}
            continue
        prior_close = float(df["Close"].iloc[i - lookback])
        if prior_close == 0:
            periods[lookback] = {"stock_pct": None, "nifty_pct": None, "rel_pct": None}
            continue
        stock_pct = (close - prior_close) / prior_close * 100.0

        nifty_prior = _nifty_close_asof(nifty_series, df.index[i - lookback])
        if nifty_today is None or nifty_prior is None or nifty_prior == 0:
            periods[lookback] = {"stock_pct": stock_pct, "nifty_pct": None, "rel_pct": None}
            continue
        nifty_pct = (nifty_today - nifty_prior) / nifty_prior * 100.0
        rel_pct = stock_pct - nifty_pct
        periods[lookback] = {"stock_pct": stock_pct, "nifty_pct": nifty_pct, "rel_pct": rel_pct}
        weighted_sum += rel_pct * weight
        weight_total += weight

    if weight_total == 0:
        return {"periods": periods, "score": None, "label": None}

    score = weighted_sum / weight_total
    label = next(lbl for lower, lbl in RS_BANDS if score >= lower)
    return {"periods": periods, "score": score, "label": label}


def build_sector_context(histories, sector_map, nifty_series):
    """Precomputed ONCE per run (not per-signal) -- groups every symbol
    in `histories` by sector_map, builds one equal-weighted synthetic
    sector index per sector (each constituent rebased to start at 100 on
    its own first available date, then averaged across the sector -- see
    the Sector Confirmation block comment above for why this is
    synthetic rather than a real fetched index), plus today's breadth
    read for that sector. Returns {sector_name: {"index": pd.Series,
    "breadth_today": float 0-1|None, "constituent_count": int}}.
    Sectors with fewer than SECTOR_MIN_CONSTITUENTS constituents actually
    present in this run's histories are dropped entirely -- too thin a
    sample for a meaningful sector read, same reasoning as the
    thin-sample handling elsewhere in this module (see
    WILSON_CONFIDENCE_Z / confidence_adjusted_hit_rate). Returns {} if
    sector_map is empty (SECTOR_MAP not yet populated -- see above) or
    no sector cleared the minimum-constituent bar."""
    if not sector_map:
        return {}

    by_sector = {}
    for symbol, df in histories.items():
        sector = sector_map.get(symbol)
        if not sector or df.empty:
            continue
        by_sector.setdefault(sector, []).append((symbol, df))

    context = {}
    for sector, members in by_sector.items():
        if len(members) < SECTOR_MIN_CONSTITUENTS:
            continue

        rebased_series = []
        breadth_eligible = 0
        above_sma_count = 0
        for symbol, df in members:
            close = df["Close"].dropna()
            if close.empty or float(close.iloc[0]) == 0:
                continue
            rebased_series.append((close / float(close.iloc[0]) * 100.0).rename(symbol))

            if len(close) >= SECTOR_BREADTH_SMA_PERIOD:
                sma = float(close.iloc[-SECTOR_BREADTH_SMA_PERIOD:].mean())
                breadth_eligible += 1
                if float(close.iloc[-1]) > sma:
                    above_sma_count += 1

        if not rebased_series:
            continue

        sector_index = pd.concat(rebased_series, axis=1).mean(axis=1).dropna()
        breadth_today = (above_sma_count / breadth_eligible) if breadth_eligible else None

        context[sector] = {
            "index": sector_index,
            "breadth_today": breadth_today,
            "constituent_count": len(members),
        }
    return context


def compute_sector_confirmation(symbol, sector_map, sector_context, nifty_series):
    """Sector Confirmation for one fired signal -- see the block comment
    above build_sector_context. Returns {
      sector: str,
      above_20dma / above_50dma: bool|None -- is the synthetic sector
        index above its own trailing 20/50-day average?
      relative_strength_pct: float|None -- the sector's own
        SECTOR_RS_LOOKBACK-day return minus NIFTY's return over the SAME
        calendar window (positive = sector outperforming NIFTY)
      breadth: float|None -- fraction (0-1) of this sector's
        constituents in this run individually trading above their own
        20DMA today
      passed_count / available_count: same "(passed)/(available)"
        convention as the Confirmation column
      score: 0-100 fail-soft blend of the 4 checks (SECTOR_CONFIRMATION_WEIGHTS)
    } or None if this symbol has no sector mapped yet, or its sector
    didn't clear SECTOR_MIN_CONSTITUENTS this run."""
    sector = sector_map.get(symbol)
    if not sector or sector not in sector_context:
        return None

    idx = sector_context[sector]["index"]
    if idx.empty:
        return None
    close_today = float(idx.iloc[-1])

    def _above_sma(period):
        if len(idx) < period:
            return None
        return close_today > float(idx.iloc[-period:].mean())

    above_20dma = _above_sma(20)
    above_50dma = _above_sma(50)

    relative_strength_pct = None
    if len(idx) > SECTOR_RS_LOOKBACK and nifty_series is not None:
        prior_sector = float(idx.iloc[-SECTOR_RS_LOOKBACK - 1])
        nifty_today = _nifty_close_asof(nifty_series, idx.index[-1])
        nifty_prior = _nifty_close_asof(nifty_series, idx.index[-SECTOR_RS_LOOKBACK - 1])
        if prior_sector and nifty_today is not None and nifty_prior:
            sector_pct = (close_today - prior_sector) / prior_sector * 100.0
            nifty_pct = (nifty_today - nifty_prior) / nifty_prior * 100.0
            relative_strength_pct = sector_pct - nifty_pct

    breadth = sector_context[sector]["breadth_today"]

    components = []
    available_count = 0
    passed_count = 0
    for val, weight in ((above_20dma, SECTOR_CONFIRMATION_WEIGHTS["above_20dma"]),
                        (above_50dma, SECTOR_CONFIRMATION_WEIGHTS["above_50dma"])):
        if val is None:
            continue
        available_count += 1
        passed_count += 1 if val else 0
        components.append((weight, 1.0 if val else 0.0))

    if relative_strength_pct is not None:
        available_count += 1
        passed_count += 1 if relative_strength_pct > 0 else 0
        # -5pp..+5pp rescaled to 0..1, same span as the stock-level RS_BANDS midpoint,
        # clamped at the edges rather than let an extreme reading blow the blend out.
        rs_frac = max(0.0, min(1.0, (relative_strength_pct + 5.0) / 10.0))
        components.append((SECTOR_CONFIRMATION_WEIGHTS["relative_strength"], rs_frac))

    if breadth is not None:
        available_count += 1
        passed_count += 1 if breadth >= 0.5 else 0
        components.append((SECTOR_CONFIRMATION_WEIGHTS["breadth"], breadth))

    score = _fail_soft_blend(components)

    return {
        "sector": sector,
        "above_20dma": above_20dma,
        "above_50dma": above_50dma,
        "relative_strength_pct": relative_strength_pct,
        "breadth": breadth,
        "passed_count": passed_count,
        "available_count": available_count,
        "score": score,
    }


# Bare NSE symbols (constants.STOCKS keys, e.g. "BEL", "SBIN", "ITC") from
# Bala's own direct-equity list -- NOT constants.WATCHLIST, which holds
# full company names in a different format. Uppercased for a
# case-insensitive match against the screener's own symbol strings.
MY_STOCK_SYMBOLS = {s.upper() for s in STOCKS.keys()}
MY_STOCK_ROW_BG = "#E9F7ED"  # light green -- "this is one of your own stocks"

ENTRY_STATE_BADGE_COLORS = {
    # EntryState -> (background, text)
    EntryState.FRESH_BREAKOUT: ("#DCEFE0", "#0f5132"),
    EntryState.RETEST: ("#FCF1D8", "#7a5b00"),
    EntryState.NEAR_BREAKOUT: ("#DCE8F5", "#1c4a8a"),
    EntryState.NONE: ("#EDEAE2", "#8A8F9C"),
}

# -----------------------------------------------------------------------
# Strategy labels -- purely presentational names for the two entry states
# that carry an actionable price, mapped onto the classic "chase the
# breakout" vs "wait for the retest" framing:
#   Strategy A (Aggressive) = Fresh Breakout -- buy the breakout bar
#     itself once every chase-safety check passes (close above
#     resistance, volume, close position, market trend -- see
#     utils.entry_classification, not recomputed here).
#   Strategy B (Conservative) = Retest -- price pulled back to the
#     breakout level, held as support, and confirmed with a bullish
#     candle and rising volume before the entry is offered. This is the
#     SAME retest logic already driving EntryState.RETEST above (this
#     label doesn't add a new detector, it names an existing one) --
#     surfaced explicitly here, plus its own report section below,
#     because a retest entry usually locks in a materially tighter stop
#     and better R:R than chasing the breakout bar (buying closer to the
#     level that has to hold, not several percent above it).
# Near Breakout has no entry price yet (nothing to buy), so it isn't
# labeled a strategy -- it's a watch item, not a trade.
# -----------------------------------------------------------------------
STRATEGY_LABEL_BY_ENTRY_STATE = {
    EntryState.FRESH_BREAKOUT: "Strategy A \u2014 Aggressive",
    EntryState.RETEST: "Strategy B \u2014 Conservative",
}
STRATEGY_LABEL_COLOR_BY_ENTRY_STATE = {
    EntryState.FRESH_BREAKOUT: "#7a5b00",
    EntryState.RETEST: "#0f5132",
}

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


def fetch_nifty_index_history():
    """One-time fetch of NIFTY 50 index history (^NSEI), used as the
    benchmark for every symbol's Relative Strength read (see
    compute_relative_strength below). Returns a pandas Series of Close
    prices indexed by date, or None if the fetch failed/came back empty
    -- fail-soft, same convention as the rest of this module: a missing
    benchmark just means every row's relative_strength comes back None
    rather than blocking the run."""
    try:
        data = yf.download(
            NIFTY_INDEX_TICKER, period=HISTORY_PERIOD, interval="1d",
            auto_adjust=False, progress=False,
        )
    except Exception as e:
        log.warning(f"Breakout Screener: NIFTY index history fetch failed: {e}")
        return None

    if data is None or data.empty or "Close" not in data.columns:
        log.warning("Breakout Screener: NIFTY index history came back empty -- Relative Strength will be unavailable this run.")
        return None

    close = data["Close"]
    if isinstance(close, pd.DataFrame):  # yfinance sometimes returns a 1-col frame for a single ticker
        close = close.iloc[:, 0]
    return close.dropna()


# -----------------------------------------------------------------------
# Setup Strength -- FIVE separate sub-scores, each answering a distinct
# question, blended into one final number. Earlier versions of this
# blended six flat components (Quality/R:R/Failure Risk/Extension/
# Confirmation/Fundamentals) into a single average with no structure --
# that buries the difference between "this pattern statistically works"
# and "now is a good moment to get in", which are genuinely different
# questions a reader needs answered separately, not pre-mixed. Splitting
# them out:
#
#   Score 1 -- Breakout Quality  (30%): "Will the breakout continue?"
#     Backtest Quality Score, today's Confirmation checklist, inverse
#     Failure Risk, and (Confirmed-only) Fundamentals -- everything about
#     whether THIS PATTERN, on THIS STOCK, has a track record of playing
#     out. Nothing here is about timing the entry.
#   Score 2 -- Trade Quality     (20%): "Is the entry attractive right now?"
#     Entry State (Retest preferred > Fresh Breakout > Near Breakout), the
#     Extension/chase-risk read, and price vs. Anchored VWAP (see
#     compute_vwap_context) -- purely about TODAY's price action relative
#     to the breakout, independent of whether the pattern itself is
#     statistically sound.
#   Score 3 -- Risk:Reward       (15%): "Is the payoff worth the risk?"
#     The R:R ratio on its own, capped at SETUP_STRENGTH_RR_CAP.
#   Score 4 -- Market Regime     (15%): "Is this the right environment?"
#     The same NIFTY-regime read used to gate Confirmed in a Bearish
#     tape (see BEAR_MARKET_MIN_QUALITY_SCORE) -- identical for every row
#     in a given run, since it's a market-wide read, not a per-stock one.
#   Score 5 -- Sector Confirmation (20%): "Is the sector confirming, or
#     is this stock breaking out alone?" See compute_sector_confirmation
#     above -- above sector 20/50DMA, sector relative strength vs NIFTY,
#     sector breadth. Weighted on par with Trade Quality deliberately --
#     this was flagged as a high-priority addition, not a minor tiebreak.
#     Comes back None (and is dropped/renormalized like any other missing
#     sub-score) until SECTOR_MAP is populated -- see the Sector
#     Confirmation block comment above.
#
# Each of the 5 sub-scores is itself fail-soft-averaged across its own
# components (same convention as before -- a missing component is
# dropped and that sub-score's remaining weights renormalized), and then
# the 5 sub-scores are blended at the weights above; if an ENTIRE
# sub-score is unavailable for a row (e.g. no R:R because there's no
# exact entry price), it's dropped and the remaining ones are
# renormalized. Explicitly NOT a probability or a guarantee, and does
# NOT override the Confirmed/Watch/Filtered/Failed bucket a row already
# landed in -- see build_report_html's disclaimer.
# -----------------------------------------------------------------------
BREAKOUT_QUALITY_WEIGHTS = {
    "backtest": 45,        # Quality Score -- backtest hit-rate/sample/consistency, already blended
    "confirmation": 25,    # today's 9-point technical checklist
    "failure_risk": 20,    # inverse of this pattern's historical failure rate
    "fundamentals": 10,    # ROE/Debt-Equity/earnings growth, Confirmed-only
}
TRADE_QUALITY_WEIGHTS = {
    "entry_state": 35,     # how actionable/preferred the entry classification is right now
    "extension": 40,       # inverse of today's chase-risk -- see compute_extension_score
    "vwap": 25,             # price above Anchored VWAP from the breakout base -- see compute_vwap_context
}
SETUP_STRENGTH_WEIGHTS = {
    "breakout_quality": 30,
    "trade_quality": 20,
    "risk_reward": 15,
    "market_regime": 15,
    "sector_confirmation": 20,
}
SETUP_STRENGTH_RR_CAP = 3.0  # R:R at/above this is treated as fully maxed out on that component
# compute_extension_score's score range (see EXTENSION_SCORE_BANDS) is
# +10 (barely moved) down to -15 (severely extended) -- normalized to a
# 0-1 fraction the same way Quality/Confirmation already are, so it
# blends into the same weighted average.
EXTENSION_SCORE_MIN = -15
EXTENSION_SCORE_MAX = 10

# How "attractive to enter right now" each Entry State is, for the Trade
# Quality sub-score -- Retest is this project's preferred entry (buying a
# pullback to confirmed support beats chasing the breakout bar itself),
# Fresh Breakout is actionable but by definition already moving, and Near
# Breakout hasn't triggered yet so it's scored low, not zero. NONE
# (no exact entry price at all) is intentionally absent -- there's
# nothing to score, so it's excluded rather than penalized, same
# fail-soft rule as everywhere else.
ENTRY_STATE_TRADE_QUALITY_FRACTIONS = {
    EntryState.RETEST: 1.0,
    EntryState.FRESH_BREAKOUT: 0.75,
    EntryState.NEAR_BREAKOUT: 0.35,
}

SETUP_STRENGTH_BANDS = (
    (80, "Very Strong"),
    (65, "Strong"),
    (50, "Moderate"),
    (0, "Weak"),
)


def _fail_soft_blend(components):
    """Shared fail-soft weighted-average helper: components is a list of
    (weight, fraction 0-1) pairs. Returns an int 0-100, or None if
    components is empty. Missing inputs are simply absent from the list
    (never a stand-in zero), and the present weights are renormalized so
    a data gap never silently drags the score down or up."""
    if not components:
        return None
    total_weight = sum(w for w, _ in components)
    return round(sum(w * frac for w, frac in components) / total_weight * 100)


def compute_breakout_quality_score(row):
    """Score 1 -- 'Will the breakout continue?' Returns {"score": int
    0-100, "components_used": int, "components_total": int} or None if
    none of the underlying components were available."""
    components = []

    quality = row.get("quality")
    if quality:
        components.append((BREAKOUT_QUALITY_WEIGHTS["backtest"], quality["score"] / 100.0))

    confirmation = row.get("confirmation")
    if confirmation and confirmation.available_count:
        components.append((
            BREAKOUT_QUALITY_WEIGHTS["confirmation"],
            confirmation.passed_count / confirmation.available_count,
        ))

    failure_risk = row.get("failure_risk")
    if failure_risk and failure_risk.backtest and failure_risk.backtest.failure_rate is not None:
        components.append((BREAKOUT_QUALITY_WEIGHTS["failure_risk"], 1.0 - failure_risk.backtest.failure_rate))

    fundamentals = row.get("fundamentals")
    if fundamentals and fundamentals["available"] and fundamentals["checks"]:
        passed_frac = sum(1 for _, ok, _ in fundamentals["checks"] if ok) / len(fundamentals["checks"])
        components.append((BREAKOUT_QUALITY_WEIGHTS["fundamentals"], passed_frac))

    score = _fail_soft_blend(components)
    if score is None:
        return None
    return {"score": score, "components_used": len(components), "components_total": len(BREAKOUT_QUALITY_WEIGHTS)}


def compute_trade_quality_score(row):
    """Score 2 -- 'Is the entry attractive right now?' Returns {"score":
    int 0-100, "components_used": int, "components_total": int} or None
    if neither an Entry State nor an Extension read was available."""
    components = []

    entry = row.get("entry")
    if entry and entry.state in ENTRY_STATE_TRADE_QUALITY_FRACTIONS:
        components.append((
            TRADE_QUALITY_WEIGHTS["entry_state"],
            ENTRY_STATE_TRADE_QUALITY_FRACTIONS[entry.state],
        ))

    extension = row.get("extension")
    if extension:
        ext_frac = (extension["score"] - EXTENSION_SCORE_MIN) / (EXTENSION_SCORE_MAX - EXTENSION_SCORE_MIN)
        ext_frac = max(0.0, min(1.0, ext_frac))
        components.append((TRADE_QUALITY_WEIGHTS["extension"], ext_frac))

    vwap = row.get("vwap")
    if vwap and vwap["price_above_anchored_vwap"] is not None:
        components.append((TRADE_QUALITY_WEIGHTS["vwap"], 1.0 if vwap["price_above_anchored_vwap"] else 0.0))

    score = _fail_soft_blend(components)
    if score is None:
        return None
    return {"score": score, "components_used": len(components), "components_total": len(TRADE_QUALITY_WEIGHTS)}


def compute_risk_reward_setup_score(row):
    """Score 3 -- 'Is the payoff worth the risk?' Just the R:R ratio,
    capped at SETUP_STRENGTH_RR_CAP and rescaled to 0-100. Returns None
    if there's no R:R data for this row (no exact entry price)."""
    risk_reward = row.get("risk_reward")
    if not risk_reward:
        return None
    frac = min(risk_reward["ratio"] / SETUP_STRENGTH_RR_CAP, 1.0)
    return {"score": round(frac * 100)}


def compute_market_regime_setup_score(regime):
    """Score 4 -- 'Is this the right environment for breakouts?' The same
    market-wide regime read used to gate Confirmed in a Bearish tape --
    identical for every row in a given run. Returns None only if regime
    itself wasn't computed (shouldn't happen in normal operation)."""
    if regime is None or regime.score is None:
        return None
    frac = max(0.0, min(1.0, regime.score / 5.0))
    return {"score": round(frac * 100), "label": regime.label()}


def compute_sector_setup_score(row):
    """Score 5 -- 'Is the sector confirming, or is this stock breaking
    out alone?' Just row['sector_confirmation']['score'] carried through
    -- that dict already IS a 0-100 fail-soft blend of the 4 sector
    checks (see compute_sector_confirmation). Returns None if no sector
    confirmation was available for this row (no SECTOR_MAP entry, or its
    sector didn't clear SECTOR_MIN_CONSTITUENTS)."""
    sector_confirmation = row.get("sector_confirmation")
    if not sector_confirmation or sector_confirmation["score"] is None:
        return None
    return {"score": sector_confirmation["score"], "sector": sector_confirmation["sector"]}


def compute_setup_strength(row, regime):
    """Returns {"score": int 0-100, "label": str, "components_used": int,
    "components_total": int, "breakout_quality": {...}|None,
    "trade_quality": {...}|None, "risk_reward": {...}|None,
    "market_regime": {...}|None, "sector_confirmation": {...}|None} or
    None if NONE of the five sub-scores could be computed for this row."""
    breakout_quality = compute_breakout_quality_score(row)
    trade_quality = compute_trade_quality_score(row)
    risk_reward = compute_risk_reward_setup_score(row)
    market_regime = compute_market_regime_setup_score(regime)
    sector_confirmation = compute_sector_setup_score(row)

    components = []
    if breakout_quality:
        components.append((SETUP_STRENGTH_WEIGHTS["breakout_quality"], breakout_quality["score"] / 100.0))
    if trade_quality:
        components.append((SETUP_STRENGTH_WEIGHTS["trade_quality"], trade_quality["score"] / 100.0))
    if risk_reward:
        components.append((SETUP_STRENGTH_WEIGHTS["risk_reward"], risk_reward["score"] / 100.0))
    if market_regime:
        components.append((SETUP_STRENGTH_WEIGHTS["market_regime"], market_regime["score"] / 100.0))
    if sector_confirmation:
        components.append((SETUP_STRENGTH_WEIGHTS["sector_confirmation"], sector_confirmation["score"] / 100.0))

    score = _fail_soft_blend(components)
    if score is None:
        return None
    label = next(lbl for threshold, lbl in SETUP_STRENGTH_BANDS if score >= threshold)
    return {
        "score": score,
        "label": label,
        "components_used": len(components),
        "components_total": len(SETUP_STRENGTH_WEIGHTS),
        "breakout_quality": breakout_quality,
        "trade_quality": trade_quality,
        "risk_reward": risk_reward,
        "market_regime": market_regime,
        "sector_confirmation": sector_confirmation,
    }


# -----------------------------------------------------------------------
# Confidence-adjusted hit rate -- a raw hit-rate treats "100% over 1
# occurrence" (ZYDUSLIFE/COFORGE-style) as equal to or better than "79%
# over 14 occurrences" (a NATIONALUM-style deep sample), when the first
# number is actually far less trustworthy: a single coin-flip landing
# heads tells you almost nothing about the coin. Wilson's score interval
# answers "given n tries and this hit rate, what's a defensible LOWER
# bound on the true hit rate at ~95% confidence" -- it shrinks
# aggressively toward 0 at tiny n and converges to the raw hit rate as n
# grows, which is exactly the shape wanted here. This is used (a) to
# de-fang thin-sample rows in the ranking below, so a 1/1 or 2/2 result
# can no longer out-rank a deep, merely-good sample, and (b) surfaced
# directly in the report next to the raw number so nothing about the
# adjustment is hidden from the reader -- same "show your work"
# convention as everything else in this file.
# -----------------------------------------------------------------------
WILSON_CONFIDENCE_Z = 1.96  # ~95% confidence

# Bands for the plain-language confidence label shown next to a
# confidence-adjusted hit rate. Keyed off the SAME sample-size bands as
# CONFIRMATION_TIERS above, so the label lines up with the tier a reader
# has already seen on Confirmed/Watch rows rather than inventing a
# separate set of thresholds.
CONFIDENCE_LABEL_BANDS = (
    (30, "High"),           # Tier A band
    (15, "Moderate"),        # Tier B band
    (CONFIRMATION_TIER_MIN_SAMPLES, "Low"),  # Tier C band
    (0, "Very Low"),         # Experimental
)


def wilson_lower_bound(hit_rate, n, z=WILSON_CONFIDENCE_Z):
    """95%-confidence lower bound on the true hit rate, given an observed
    hit_rate (0-1) over n occurrences. Returns 0.0 for n<=0. This is the
    standard Wilson score interval lower bound -- unlike a raw hit rate,
    it accounts for how little n tells you: e.g. 100% over 1 occurrence
    collapses to roughly a 21% lower bound, while 91% over 11 occurrences
    only comes down to roughly 65%, and a deep, consistent sample barely
    moves at all."""
    if n <= 0:
        return 0.0
    phat = max(0.0, min(1.0, hit_rate))
    z2 = z * z
    denominator = 1.0 + z2 / n
    center = phat + z2 / (2 * n)
    margin = z * ((phat * (1 - phat) / n + z2 / (4 * n * n)) ** 0.5)
    return max(0.0, (center - margin) / denominator)


def confidence_adjusted_hit_rate(hit_rate, sample_size):
    """Returns {"raw": hit_rate, "adjusted": wilson-lower-bound hit rate,
    "sample_size": sample_size, "confidence": label} for display and
    ranking use, or None if hit_rate or sample_size is missing.
    `adjusted` is what ranking should use in place of raw hit-rate;
    `confidence` is a plain-language label for the report so a reader can
    see AT A GLANCE why a thin-sample row isn't being weighted like a
    deep one, without doing the math themselves."""
    if hit_rate is None or not sample_size:
        return None
    adjusted = wilson_lower_bound(hit_rate, sample_size)
    label = next(lbl for threshold, lbl in CONFIDENCE_LABEL_BANDS if sample_size >= threshold)
    return {"raw": hit_rate, "adjusted": adjusted, "sample_size": sample_size, "confidence": label}


# -----------------------------------------------------------------------
# Risk-adjusted ranking -- ranking rows on raw hit-rate alone rewards a
# signal like NATIONALUM (79% 10-day hit-rate, but 14% chance of falling
# back within 3 days) over one like INDUSTOWER (71% hit-rate, 0% 3-day
# failure in-sample), even though INDUSTOWER is the cleaner trade once
# the odds of an early failure are priced in. It also rewards a 1/1 or
# 2/2 thin sample over a deep, merely-good one, which is just as much of
# a distortion -- see confidence_adjusted_hit_rate above. This computes
# the primary horizon's expected value using the Wilson-adjusted (not
# raw) hit rate times avg_return, and scales that down by the probability
# of failing within the failure window, so ranking reflects "expected
# return per unit of risk AND sample confidence taken", not "how often
# did this eventually work out at the 10-day mark, on however few tries".
# -----------------------------------------------------------------------
def compute_risk_adjusted_score(row):
    """Returns a float (can be negative) or None if the primary-horizon
    backtest stats aren't available. Never guesses at a missing
    failure-rate -- if it's not available (thin sample), the raw
    (un-adjusted) expected value is used as-is rather than penalized."""
    bt = row.get("backtest")
    if not bt or not bt.get("horizons"):
        return None
    primary = bt["horizons"].get(PRIMARY_HORIZON)
    if not primary or primary.get("hit_rate") is None or primary.get("avg_return") is None:
        return None

    hit_rate_conf = confidence_adjusted_hit_rate(primary["hit_rate"], bt.get("sample_size"))
    # hit_rate_conf can only be None here if sample_size is falsy, which
    # shouldn't happen alongside a populated hit_rate -- fall back to the
    # raw hit-rate rather than fail the whole score in that edge case.
    effective_hit_rate = hit_rate_conf["adjusted"] if hit_rate_conf else primary["hit_rate"]

    raw_ev = effective_hit_rate * primary["avg_return"]

    failure_risk = row.get("failure_risk")
    failure_rate = None
    if failure_risk and failure_risk.backtest and failure_risk.backtest.failure_rate is not None:
        failure_rate = failure_risk.backtest.failure_rate

    if failure_rate is None:
        return raw_ev
    # Failing within the failure window effectively forfeits the trade's
    # upside before the primary horizon plays out -- scale expected value
    # down by the odds of that happening.
    return raw_ev * (1.0 - failure_rate)


# -----------------------------------------------------------------------
# Best Execution Case -- a rule-based read across signals ALREADY computed
# for a row (which bucket it landed in, Entry State, Setup Strength,
# Risk:Reward, Failure Risk). Not a new backtest, not a probability, and
# not a replacement for Setup Strength -- it's a stricter, all-of-the-above
# filter on top of it, meant to flag only the handful of rows an
# at-a-glance read would actually be comfortable acting on TODAY, out of
# everything that fired. Same fail-soft convention as the rest of this
# file: a row missing a needed component simply doesn't qualify -- it's
# never flagged as bad, just not called out as a best case.
#
# A row qualifies ONLY when ALL of the following hold:
#   1. It landed in the Confirmed Breakouts bucket (checked by the caller
#      -- Watch/Filtered/Near-Breakout rows never qualify, no matter how
#      high their component scores are).
#   2. Entry State is Fresh Breakout or Retest -- there's a concrete entry
#      price today, not just a pattern fire with no actionable zone.
#   3. Setup Strength >= BEST_EXECUTION_MIN_SETUP_STRENGTH (the composite
#      already blends Breakout Quality / Trade Quality / Risk:Reward /
#      Market Regime -- see compute_setup_strength above).
#   4. Risk:Reward is at/above RISK_REWARD_PREFERRED_THRESHOLD, not just
#      the bare minimum that keeps a row out of the Filtered section.
#   5. Failure Risk isn't flagged elevated on a reliable sample (reuses
#      utils.breakout_failure's own "elevated" read rather than
#      re-deriving a second threshold here).
#   6. Price is above Anchored VWAP from the breakout base (see
#      compute_vwap_context) WHEREVER that figure was available for this
#      row -- if it wasn't (e.g. breakout_level itself was unavailable),
#      this check is skipped rather than failing the row on a data gap,
#      same fail-soft convention as everything else here.
BEST_EXECUTION_MIN_SETUP_STRENGTH = 80  # matches the "Very Strong" Setup Strength band
BEST_EXECUTION_ROW_BG = "#87CEFA"       # light sky blue -- "best case for execution today"
BEST_EXECUTION_LABEL = "\U0001F3AF Best Execution Case"


def is_best_execution_case(row):
    """row is expected to already carry entry/setup_strength/risk_reward/
    failure_risk (i.e. called after those are set on the row, and only
    for rows about to be placed in the Confirmed bucket)."""
    entry = row.get("entry")
    if not entry or entry.state not in (EntryState.FRESH_BREAKOUT, EntryState.RETEST):
        return False

    strength = row.get("setup_strength")
    if not strength or strength["score"] < BEST_EXECUTION_MIN_SETUP_STRENGTH:
        return False

    risk_reward = row.get("risk_reward")
    if not risk_reward or risk_reward["ratio"] < RISK_REWARD_PREFERRED_THRESHOLD:
        return False

    failure_risk = row.get("failure_risk")
    if failure_risk and failure_risk.backtest and failure_risk.backtest.elevated:
        return False

    vwap = row.get("vwap")
    if vwap and vwap["price_above_anchored_vwap"] is False:
        return False

    return True


# -----------------------------------------------------------------------
# Scan
# -----------------------------------------------------------------------
def scan_universe(histories, bhav_df, regime: RegimeResult, nifty_series=None, sector_context=None):
    """
    histories: {symbol: OHLCV df}
    regime: RegimeResult from utils.market_regime.compute_market_regime(),
        computed once per run from the SAME histories dict. Two effects:
          - regime.entries_supportive feeds every fresh-breakout entry
            classification's "Market trend supportive" check (True in
            Bullish/Neutral, False in Bearish).
          - regime.only_strongest (Bearish only) raises the Confirmed bar
            below: a signal must ALSO carry a Strong/Excellent Quality
            score and Risk:Reward at the PREFERRED threshold, not just
            clear the backtest -- see REGIME_INTERPRETATION in
            utils.market_regime. Rows that clear the backtest but not
            this stricter bar are downgraded to Watch, never dropped.
    nifty_series: pandas Series of NIFTY 50 Close indexed by date, from
        fetch_nifty_index_history() -- fetched ONCE per run and reused
        for every symbol's Relative Strength read (see
        compute_relative_strength above). None is handled fail-soft:
        every row's row["relative_strength"] just comes back None.
    sector_context: {sector_name: {...}} from build_sector_context(),
        built ONCE per run from the SAME histories dict plus SECTOR_MAP
        -- reused for every symbol's Sector Confirmation read (see
        compute_sector_confirmation above). None/empty is handled
        fail-soft: every row's row["sector_confirmation"] just comes
        back None (see SECTOR_MAP -- not populated in this build).
    Returns (confirmed, watch_list, filtered_low_rr, near_breakout_watch, skipped_count):
      confirmed: list of signal dicts that cleared the backtest bar AND
        (where R:R data was available) cleared RISK_REWARD_MIN_THRESHOLD
        AND, in a Bearish regime, the stricter regime bar above
      watch_list: signals that fired today but didn't clear the backtest
        bar (thin sample or weak historical hit-rate), OR that cleared it
        but were downgraded by the Bearish-regime bar (row["regime_downgraded"]
        marks these so the email can label them honestly)
      filtered_low_rr: signals that would otherwise have qualified for
        Confirmed/Watch but were pulled out because R:R < RISK_REWARD_MIN_THRESHOLD
        -- shown separately, never silently dropped, per this project's
        "everything shown is labeled for what it is" convention
      near_breakout_watch: symbols that haven't broken out yet but are
        close, per utils.entry_classification -- surfaced even though no
        pattern fired today, so the screener is useful before the breakout too
      skipped_count: symbols dropped entirely by the data-quality gate

      Every row (in every bucket above) also carries row["failure_risk"].backtest
      -- the historical probability THIS exact pattern has failed within
      {FAILURE_WINDOW_DAYS} days on THIS stock before, for context. There is
      no longer a same-day bull-trap gate that segregates rows into their own
      section -- rows are bucketed purely on the backtest/R:R rules above.
    """
    confirmed, watch_list, filtered_low_rr, near_breakout_watch = [], [], [], []
    skipped_count = 0
    market_trend_supportive = regime.entries_supportive
    fundamentals_cache = {}  # per-run cache, symbol -> yfinance .info dict
    sector_context = sector_context or {}

    for symbol, df in histories.items():
        ok, dq_notes = data_quality_check(symbol, df, bhav_df)
        if not ok:
            skipped_count += 1
            continue

        i = len(df) - 1
        entry_result = classify_entry(symbol, df, market_trend_supportive, ENTRY_CLASSIFIER_CONFIG)

        today_signals = scan_all_patterns(df, i)
        if not today_signals:
            # No pattern fired today -- still worth surfacing if price
            # structure alone says we're approaching a breakout.
            if entry_result and entry_result.state == EntryState.NEAR_BREAKOUT:
                near_breakout_watch.append({
                    "symbol": symbol,
                    "entry": entry_result,
                    "dq_notes": dq_notes,
                    "current_price": float(df["Close"].iloc[-1]),
                })
            continue

        for sig in today_signals:
            detector_fn = PATTERN_DETECTOR_BY_NAME.get(sig["pattern"])
            bt = backtest_signal(symbol, df, detector_fn, i) if detector_fn else None
            quality = compute_quality_score(bt, dq_notes)
            failure_risk = evaluate_failure_risk(symbol, df, i, detector_fn)
            confirmation = compute_confirmation_score(symbol, df, i, regime)

            # Extension Score -- how far price has already moved today
            # relative to the breakout level. Uses the exact entry price
            # where the classifier found one (Fresh Breakout / Retest),
            # falling back to the pattern's own signal price otherwise --
            # every fired signal gets a breakout_level to measure against.
            # See compute_extension_score above.
            breakout_level = None
            if entry_result and entry_result.exact_entry_price:
                breakout_level = entry_result.exact_entry_price
            elif sig.get("signal_price"):
                breakout_level = sig["signal_price"]
            extension = compute_extension_score(symbol, df, i, breakout_level)

            # VWAP / Anchored VWAP -- is price holding above its own
            # recent volume-weighted average cost, and above the
            # volume-weighted average cost paid since the breakout base?
            # Same breakout_level input as Extension Score above. See
            # compute_vwap_context and the VWAP block comment for the
            # daily-bar-vs-intraday-session caveat.
            vwap = compute_vwap_context(symbol, df, i, breakout_level)

            # Relative Strength vs NIFTY -- is this stock actually
            # showing leadership, or just moving because the whole
            # market is? See compute_relative_strength above.
            relative_strength = compute_relative_strength(symbol, df, i, nifty_series)

            # Sector Confirmation -- did the sector confirm, or is this
            # stock breaking out alone? See compute_sector_confirmation
            # above. None until SECTOR_MAP is populated (see comment
            # there) -- fail-soft, same as everything else.
            sector_confirmation = compute_sector_confirmation(symbol, SECTOR_MAP, sector_context, nifty_series)

            # Targets need an actual actionable entry price to project
            # from -- only Fresh Breakout / Retest rows have one. Near
            # Breakout hasn't triggered yet and No Setup isn't a buy, so
            # there's nothing to project a target from in either case.
            targets = None
            risk_reward = None
            if entry_result and entry_result.exact_entry_price:
                targets = compute_targets(
                    symbol, df, i, sig["pattern"], sig,
                    entry_result.exact_entry_price, entry_result.atr,
                    TARGETS_CONFIG,
                )
                risk_reward = compute_risk_reward(
                    entry_result.exact_entry_price, entry_result.stop_loss, targets,
                )

            # Cleared the backtest bar on its own merits (the ONLY bar
            # outside a Bearish regime) -- AND clears the stricter,
            # controller-local TIERED bar (CONFIRMATION_TIERS above) on
            # top of it. Both have to pass; a row can clear the shared
            # module's own (looser) bar and still get pulled back to
            # Watch here if its occurrence count is too thin to reach any
            # tier, or its hit-rate doesn't clear the bar for the tier its
            # sample size falls into.
            confirmation_tier = None
            if bt and bt.get("horizons"):
                primary_stats = bt["horizons"].get(PRIMARY_HORIZON)
                if primary_stats:
                    confirmation_tier = classify_confirmation_tier(
                        bt.get("sample_size"), primary_stats.get("hit_rate")
                    )
            strict_bar_cleared = confirmation_tier is not None
            shared_bar_cleared = bool(bt and bt.get("confirmed"))
            # Cleared the shared bar but not this controller's stricter
            # tiered one -- surfaced on the row so Watch List can explain
            # why (see "confirmation_tier_downgraded" below), same
            # convention as regime_downgraded/fundamental_downgraded.
            confirmation_tier_downgraded = shared_bar_cleared and not strict_bar_cleared
            cleared_backtest = shared_bar_cleared and strict_bar_cleared

            # In a Bearish regime, clearing the backtest is necessary but
            # no longer sufficient -- "only the strongest breakouts are
            # allowed through" also means a Strong/Excellent Quality score
            # and Risk:Reward at the PREFERRED bar, not just the normal
            # minimum (which the filtered_low_rr gate below already enforces
            # regardless of regime).
            if regime.only_strongest:
                meets_confirmed_bar = (
                    cleared_backtest
                    and quality is not None and quality["score"] >= BEAR_MARKET_MIN_QUALITY_SCORE
                    and risk_reward is not None and risk_reward.get("preferred")
                )
            else:
                meets_confirmed_bar = cleared_backtest

            # Fundamental quality gate -- Confirmed Breakouts only (see
            # FUNDAMENTAL_MIN_ROE et al. above). Only bother fetching
            # fundamentals for rows that are, on technicals alone, about
            # to qualify for Confirmed -- keeps this an occasional
            # per-signal yfinance call rather than a 500-symbol sweep.
            fundamentals = None
            fundamental_downgraded = False
            if meets_confirmed_bar:
                fundamentals = evaluate_fundamental_quality(symbol, fundamentals_cache)
                if fundamentals["available"] and not fundamentals["passed"]:
                    meets_confirmed_bar = False
                    fundamental_downgraded = True

            # Extension / chase-risk gate -- Confirmed Breakouts only,
            # same fail-soft convention as the fundamentals gate directly
            # above: a row that clears every other bar but is flagged
            # chase_risk (see compute_extension_score / CHASE_RISK_*) is
            # pulled out of Confirmed into Watch List, never dropped.
            extension_downgraded = False
            if meets_confirmed_bar and extension and extension["chase_risk"]:
                meets_confirmed_bar = False
                extension_downgraded = True

            # Extreme-volume / weak-close gate -- Confirmed Breakouts
            # only, same fail-soft convention as the gates above: a row
            # that clears every other bar but fired on Extreme volume
            # (see VOLUME_BANDS/EXTREME_VOLUME_MULTIPLE) WITHOUT a
            # confirming close (CLV below EXTREME_VOLUME_MIN_CLV) is
            # pulled out of Confirmed into Watch List, never dropped. A
            # row with no CLV available (no intraday range) is left
            # alone rather than penalized for a data gap.
            extreme_volume_downgraded = False
            if (
                meets_confirmed_bar and extension
                and extension["volume_label"] == "Extreme"
                and extension["clv"] is not None
                and extension["clv"] < EXTREME_VOLUME_MIN_CLV
            ):
                meets_confirmed_bar = False
                extreme_volume_downgraded = True

            row = {
                "symbol": symbol,
                "pattern": sig["pattern"],
                "signal_price": sig["signal_price"],
                "detail": sig["detail"],
                "dq_notes": dq_notes,
                "backtest": bt,
                "quality": quality,
                "entry": entry_result,
                "targets": targets,
                "risk_reward": risk_reward,
                "failure_risk": failure_risk,
                "confirmation": confirmation,
                "fundamentals": fundamentals,
                "extension": extension,
                "vwap": vwap,
                "relative_strength": relative_strength,
                "sector_confirmation": sector_confirmation,
                # Cleared the backtest but was pulled out of Confirmed
                # specifically by the Bearish-regime bar (not by R:R,
                # fundamentals, or the extension/chase-risk gate, which
                # each have their own flag/section) -- lets the email say
                # WHY this row is sitting in Watch instead of leaving it
                # unexplained.
                "regime_downgraded": bool(
                    regime.only_strongest and cleared_backtest and not meets_confirmed_bar
                    and not fundamental_downgraded and not extension_downgraded
                    and not extreme_volume_downgraded
                ),
                # Cleared backtest AND the regime bar, but pulled out of
                # Confirmed specifically by weak/high-debt/shrinking
                # fundamentals -- see evaluate_fundamental_quality above.
                "fundamental_downgraded": fundamental_downgraded,
                # Cleared the shared backtest module's own Confirmed bar,
                # but not this controller's stricter TIERED bar -- see
                # CONFIRMATION_TIERS / classify_confirmation_tier above.
                "strict_bar_downgraded": confirmation_tier_downgraded,
                # Which tier (Tier A/B/C) a Confirmed row actually
                # cleared, or None for a row that isn't Confirmed (or
                # cleared the shared bar only) -- shown on the row so a
                # reader can see AT A GLANCE how deep/strong the sample
                # behind a Confirmed signal actually is, not just that it
                # passed some single opaque bar.
                "confirmation_tier": confirmation_tier,
                # Cleared backtest, regime, and fundamentals, but pulled
                # out of Confirmed specifically by the extension/chase-risk
                # gate (above-average volume + already deep in a
                # negative-scoring extension band + at/above breakout) --
                # see compute_extension_score / CHASE_RISK_* above.
                "extension_downgraded": extension_downgraded,
                # Cleared backtest, regime, fundamentals, and the
                # extension/chase-risk gate, but pulled out of Confirmed
                # specifically by Extreme volume (>=5x avg) paired with a
                # weak/reversing close -- see EXTREME_VOLUME_* above.
                "extreme_volume_downgraded": extreme_volume_downgraded,
            }
            # Computed last -- needs quality/risk_reward/failure_risk/
            # confirmation/fundamentals/entry/extension to already be in
            # the row dict above, since it's a blend of those 4 sub-scores,
            # not an independent calculation. See compute_setup_strength.
            row["setup_strength"] = compute_setup_strength(row, regime)
            # See compute_risk_adjusted_score above -- needs backtest and
            # failure_risk already on the row, same ordering requirement
            # as setup_strength.
            row["risk_adjusted_score"] = compute_risk_adjusted_score(row)
            # Confidence-adjusted hit rate for DISPLAY (see
            # confidence_adjusted_hit_rate above) -- risk_adjusted_score
            # already folds the Wilson-adjusted number into ranking; this
            # separate field is just so the report can show the raw and
            # adjusted numbers side by side rather than hiding the math.
            bt = row.get("backtest") or {}
            primary_stats = bt.get("horizons", {}).get(PRIMARY_HORIZON) if bt.get("horizons") else None
            row["hit_rate_confidence"] = (
                confidence_adjusted_hit_rate(primary_stats["hit_rate"], bt.get("sample_size"))
                if primary_stats and primary_stats.get("hit_rate") is not None
                else None
            )

            # Risk/Reward is a CORE FILTER, not decoration: a row with
            # computed R:R below threshold gets pulled into its own
            # section regardless of how good the backtest/quality looked,
            # since a great hit-rate on a bad R:R setup still loses money
            # over time. Rows with no R:R data (missing entry/stop/target)
            # are left in their normal bucket -- don't punish a row for a
            # gap elsewhere in the pipeline.
            if risk_reward and not risk_reward["meets_threshold"]:
                row["best_execution"] = False
                filtered_low_rr.append(row)
            elif meets_confirmed_bar:
                # Best Execution Case is only ever evaluated for rows
                # about to land in Confirmed -- see is_best_execution_case.
                row["best_execution"] = is_best_execution_case(row)
                confirmed.append(row)
            else:
                row["best_execution"] = False
                watch_list.append(row)

    def _rank_key(row):
        # Primary sort: Best Execution Case rows lead every list they're
        # eligible for (see is_best_execution_case -- Confirmed only).
        # Next, Risk-Adjusted Score (see compute_risk_adjusted_score) --
        # expected value at the primary horizon, scaled down by the odds
        # of an early failure, so a high-hit-rate-but-fragile signal
        # (e.g. NATIONALUM: 79% hit-rate, 14% 3-day failure) doesn't
        # automatically outrank a lower-hit-rate-but-clean one (e.g.
        # INDUSTOWER: 71% hit-rate, 0% 3-day failure) the way raw
        # hit-rate comparison would. None (no backtest stats at all)
        # ranks last, not first/zero, since it's the least-trustworthy
        # row to lead with, not a neutral one. Then Setup Strength score
        # (same "missing ranks last" rule), then the same ties as before:
        # Breakout Quality Score, Risk/Reward ratio, backtest sample size.
        best_execution = 1 if row.get("best_execution") else 0
        risk_adjusted = row.get("risk_adjusted_score")
        risk_adjusted_score = risk_adjusted if risk_adjusted is not None else float("-inf")
        strength = row.get("setup_strength")
        strength_score = strength["score"] if strength else -1
        quality_score = row["quality"]["score"] if row["quality"] else -1
        rr = row.get("risk_reward")
        rr_ratio = rr["ratio"] if rr else 0
        bt = row["backtest"] or {}
        return (best_execution, risk_adjusted_score, strength_score, quality_score, rr_ratio, bt.get("sample_size", 0))

    confirmed.sort(key=_rank_key, reverse=True)
    watch_list.sort(key=_rank_key, reverse=True)
    filtered_low_rr.sort(key=lambda r: (r["risk_reward"] or {}).get("ratio", 0), reverse=True)
    near_breakout_watch.sort(key=lambda r: r["entry"].distance_to_breakout_pct or 999)

    # One row per symbol across the whole report. A single stock can fire
    # several independent chart patterns on the same day (e.g. a Resistance
    # Breakout AND a Triangle Breakout), and each pattern is backtested and
    # scored on its own merits -- see the per-pattern loop above. Left alone
    # that means the same symbol can legitimately land in more than one
    # section (e.g. Confirmed on one pattern, Watch on another), which reads
    # as a duplicate/bug even though the underlying scoring is correct. To
    # keep the report to one row per symbol, we keep only that symbol's
    # single best-ranked pattern and drop the rest, in section-priority
    # order: Confirmed > Watch > Filtered (low R:R) > Near-Breakout. Each
    # bucket is already sorted best-first above, so keeping the first
    # occurrence of a symbol per bucket keeps its strongest pattern.
    confirmed, watch_list, filtered_low_rr, near_breakout_watch = _dedupe_rows_by_symbol(
        confirmed, watch_list, filtered_low_rr, near_breakout_watch
    )

    return confirmed, watch_list, filtered_low_rr, near_breakout_watch, skipped_count


def _dedupe_rows_by_symbol(*buckets):
    """Keep only the first row per symbol, scanning buckets in the order
    passed in. Each bucket must already be sorted best-first. Returns a
    tuple of deduped buckets in the same order/count as the input."""
    seen_symbols = set()
    deduped = []
    for bucket in buckets:
        kept = []
        for row in bucket:
            symbol = row["symbol"]
            if symbol in seen_symbols:
                continue
            seen_symbols.add(symbol)
            kept.append(row)
        deduped.append(kept)
    return tuple(deduped)


# -----------------------------------------------------------------------
# Email rendering -- same visual language as wealth_controller.py's report
# -----------------------------------------------------------------------
def _backtest_cell(row):
    bt = row.get("backtest") if isinstance(row, dict) else row
    if not bt or not bt.get("horizons"):
        return '<span style="color:#8A8F9C;">No historical sample yet</span>'
    primary = bt["horizons"].get(PRIMARY_HORIZON)
    if not primary:
        return '<span style="color:#8A8F9C;">No historical sample yet</span>'
    cell = (
        f'{primary["hit_rate"]*100:.0f}% hit-rate over {bt["sample_size"]} past occurrences '
        f'(avg {primary["avg_return"]*100:+.1f}% at {PRIMARY_HORIZON}d)'
    )
    if isinstance(row, dict):
        tier = row.get("confirmation_tier")
        if tier:
            cell += (
                f'<div style="margin-top:3px;font-size:10.5px;color:#0f5132;">'
                f'{html.escape(tier)} sample</div>'
            )
        hit_rate_conf = row.get("hit_rate_confidence")
        if hit_rate_conf is not None:
            # Only worth calling out when the adjustment actually moves the
            # number meaningfully -- on a deep sample the Wilson-adjusted
            # rate is within a point or two of the raw one, so surfacing it
            # there would just be noise. Below CONFIRMATION_TIER_MIN_SAMPLES it's the
            # opposite problem: the raw number alone is the misleading one.
            if hit_rate_conf["sample_size"] < CONFIRMATION_TIER_MIN_SAMPLES:
                cell += (
                    f'<div style="margin-top:3px;font-size:10.5px;color:#9a3412;">'
                    f'Confidence: {html.escape(hit_rate_conf["confidence"])} '
                    f'(only {hit_rate_conf["sample_size"]} occurrence{"s" if hit_rate_conf["sample_size"] != 1 else ""} -- '
                    f'95%-confidence floor on hit-rate is {hit_rate_conf["adjusted"]*100:.0f}%, not {hit_rate_conf["raw"]*100:.0f}%)</div>'
                )
        risk_adjusted = row.get("risk_adjusted_score")
        if risk_adjusted is not None:
            cell += (
                f'<div style="margin-top:3px;font-size:10.5px;color:#8A8F9C;">'
                f'Risk-adj. EV: {risk_adjusted*100:+.1f}% (Wilson-adjusted hit-rate &times; avg return, scaled down by 3d failure odds)</div>'
            )
    return cell


QUALITY_BADGE_COLORS = {
    # label -> (background, text)
    "Excellent": ("#DCEFE0", "#0f5132"),
    "Strong": ("#E4F0E9", "#3d7a52"),
    "Moderate": ("#FCF1D8", "#7a5b00"),
    "Weak": ("#FBE7DD", "#9a3412"),
    "Poor": ("#F8DADA", "#8a1c1c"),
}


def _quality_badge_html(quality):
    if not quality:
        return '<span style="font-family:{0};font-size:10.5px;color:#8A8F9C;">Not yet scored</span>'.format(SANS)
    bg, fg = QUALITY_BADGE_COLORS.get(quality["label"], ("#EDEAE2", "#3C4256"))
    return (
        f'<span style="display:inline-block;min-width:34px;text-align:center;padding:3px 8px;border-radius:10px;'
        f'background:{bg};color:{fg};font-family:{SANS};font-size:12px;font-weight:700;">{quality["score"]}</span>'
        f'<div style="margin-top:3px;font-family:{SANS};font-size:10px;color:{fg};">{html.escape(quality["label"])}</div>'
    )


def _entry_badge_html(entry_result):
    """Renders the Fresh / Retest / Near-breakout / No-setup badge, plus the
    entry zone/trigger, and the exact actionable entry price where the
    classifier has computed one (Fresh Breakout / Retest only)."""
    if not entry_result:
        return '<span style="font-family:{0};font-size:10.5px;color:#8A8F9C;">Not classified</span>'.format(SANS)

    bg, fg = ENTRY_STATE_BADGE_COLORS.get(entry_result.state, ("#EDEAE2", "#3C4256"))
    badge = (
        f'<span style="display:inline-block;padding:3px 8px;border-radius:10px;'
        f'background:{bg};color:{fg};font-family:{SANS};font-size:11px;font-weight:700;white-space:nowrap;">'
        f'{html.escape(entry_result.label())}</span>'
    )
    strategy_label = STRATEGY_LABEL_BY_ENTRY_STATE.get(entry_result.state)
    if strategy_label:
        strategy_color = STRATEGY_LABEL_COLOR_BY_ENTRY_STATE.get(entry_result.state, "#8A8F9C")
        badge += (
            f'<div style="margin-top:2px;font-size:9.5px;color:{strategy_color};">'
            f'{html.escape(strategy_label)}</div>'
        )
    detail = ""
    if entry_result.exact_entry_price:
        detail += f'<div style="margin-top:3px;font-size:10.5px;color:#3C4256;font-weight:700;">Entry: ₹{entry_result.exact_entry_price:,.2f}</div>'
    elif entry_result.entry_zone:
        lo, hi = entry_result.entry_zone
        detail += f'<div style="margin-top:3px;font-size:10.5px;color:#3C4256;">Entry: ₹{lo:,.2f}&ndash;₹{hi:,.2f}</div>'
    elif entry_result.entry_trigger:
        detail += f'<div style="margin-top:3px;font-size:10.5px;color:#3C4256;">Trigger: ≥₹{entry_result.entry_trigger:,.2f}</div>'
    elif entry_result.state == EntryState.NONE:
        # No exact price, no zone, no trigger -- the price-structure
        # overlay found nothing to key an entry off (e.g. already
        # extended past resistance, or no usable trailing-high/retest
        # reference in this stock's history). Say so explicitly rather
        # than leaving Entry/Stop-Loss/Targets/R:R blank with no
        # explanation -- same "nothing shown without a reason" rule as
        # the rest of this report.
        detail += (
            f'<div style="margin-top:3px;font-size:10.5px;color:#8A8F9C;">'
            f'No clean entry structure found &mdash; Stop-Loss/Targets/R:R not available.</div>'
        )
    return badge + detail


def _stop_loss_cell_html(entry_result):
    """Dedicated Stop-Loss / Risk column. Only populated for FRESH_BREAKOUT
    and RETEST rows (the states with an exact entry price to measure risk
    against). Shows the final chosen stop, its basis (structural vs ATR),
    and the ATR-based alternative for reference."""
    if not entry_result or not entry_result.stop_loss:
        return '<span style="font-family:{0};font-size:10.5px;color:#8A8F9C;">&mdash;</span>'.format(SANS)

    lines = [
        f'<div style="font-family:{SANS};font-size:12px;font-weight:700;color:#8a1c1c;">₹{entry_result.stop_loss:,.2f}</div>',
        f'<div style="font-size:10.5px;color:#8a1c1c;">Risk: {entry_result.risk_pct:+.1f}%</div>',
        f'<div style="margin-top:2px;font-size:10px;color:#8A8F9C;">{html.escape(entry_result.stop_basis or "")}</div>',
    ]
    if entry_result.atr:
        lines.append(
            f'<div style="font-size:10px;color:#8A8F9C;">ATR(14) ₹{entry_result.atr:,.2f} &middot; '
            f'ATR stop ₹{entry_result.atr_stop:,.2f}</div>'
        )
    return "".join(lines)


TARGET_LABELS = {
    "target1": "T1 &middot; Prior resistance",
    "target2": "T2 &middot; Measured move",
    "target3": "T3 &middot; Fib ext / major level",
}


def _targets_cell_html(targets):
    """Dedicated Targets column. Only populated when the row's Entry
    classification produced an exact entry price (Fresh Breakout /
    Retest) -- see utils.breakout_targets.compute_targets. Each of the
    three targets is independent and separately labeled with its basis,
    same transparency convention as the Stop-Loss cell."""
    if not targets or not any(targets.values()):
        return '<span style="font-family:{0};font-size:10.5px;color:#8A8F9C;">&mdash;</span>'.format(SANS)

    lines = []
    for key in ("target1", "target2", "target3"):
        t = targets.get(key)
        label = TARGET_LABELS[key]
        if not t:
            lines.append(
                f'<div style="margin-top:2px;font-size:10px;color:#8A8F9C;">{label}: not available</div>'
            )
            continue
        lines.append(
            f'<div style="margin-top:3px;font-family:{SANS};font-size:11.5px;font-weight:700;color:#0f5132;">'
            f'₹{t["price"]:,.2f} <span style="font-weight:400;color:#3d7a52;">({t["gain_pct"]:+.1f}%)</span></div>'
            f'<div style="font-size:9.5px;color:#8A8F9C;">{label}</div>'
        )
    return "".join(lines)


RR_TARGET_LABELS = {"target1": "T1", "target2": "T2", "target3": "T3"}


def _risk_reward_cell_html(risk_reward):
    """Dedicated Risk:Reward column. This is a core filter, not a
    decoration -- see utils.breakout_targets.compute_risk_reward and
    scan_universe's filtered_low_rr split. Badge color mirrors the same
    RISK_REWARD_MIN_THRESHOLD / RISK_REWARD_PREFERRED_THRESHOLD used to
    decide whether a row is even shown here, so the color and the
    presence of the row in this table never disagree."""
    if not risk_reward:
        return '<span style="font-family:{0};font-size:10.5px;color:#8A8F9C;">Not available</span>'.format(SANS)

    ratio = risk_reward["ratio"]
    if ratio >= RISK_REWARD_PREFERRED_THRESHOLD:
        bg, fg = ("#DCEFE0", "#0f5132")
    elif ratio >= RISK_REWARD_MIN_THRESHOLD:
        bg, fg = ("#FCF1D8", "#7a5b00")
    else:
        bg, fg = ("#F8DADA", "#8a1c1c")

    target_label = RR_TARGET_LABELS.get(risk_reward["target_used"], "")
    badge = (
        f'<span style="display:inline-block;padding:3px 8px;border-radius:10px;'
        f'background:{bg};color:{fg};font-family:{SANS};font-size:12px;font-weight:700;white-space:nowrap;">'
        f'1 : {ratio:.2f}</span>'
    )
    detail = (
        f'<div style="margin-top:3px;font-size:10px;color:#3C4256;">'
        f'Risk ₹{risk_reward["risk"]:,.2f} &middot; Reward ₹{risk_reward["reward"]:,.2f} (vs {target_label})</div>'
    )
    return badge + detail


def _failure_risk_cell_html(failure_risk):
    """Dedicated Failure Risk column, shown on every row. Reports the
    historical probability THIS exact pattern has failed within
    utils.breakout_failure.FAILURE_WINDOW_DAYS days on THIS stock before
    -- informational context, not itself a gate, same convention as the
    Quality Score badge."""
    if not failure_risk or not failure_risk.backtest:
        return '<span style="font-family:{0};font-size:10.5px;color:#8A8F9C;">No historical sample yet</span>'.format(SANS)

    bt = failure_risk.backtest
    if bt.sample_size == 0 or bt.failure_rate is None:
        return '<span style="font-family:{0};font-size:10.5px;color:#8A8F9C;">No historical occurrences to check</span>'.format(SANS)

    if bt.elevated:
        color = "#8a1c1c"
    elif not bt.confirmed_sample:
        color = "#8A8F9C"
    else:
        color = "#0f5132"

    caveat = "" if bt.confirmed_sample else f' <span style="color:#8A8F9C;">(sample &lt;{MIN_SAMPLES_FOR_FAILURE_RATE})</span>'
    return (
        f'<div style="font-family:{SANS};font-size:12px;font-weight:700;color:{color};">'
        f'{bt.failure_rate*100:.0f}% failed within {FAILURE_WINDOW_DAYS}d{caveat}</div>'
        f'<div style="font-size:10px;color:#8A8F9C;">{bt.failure_count} of {bt.sample_size} past occurrences</div>'
    )


CONFIRMATION_SCORE_COLOR_BANDS = (
    (0.85, "#0f5132"),
    (0.6, "#7a5b00"),
    (0.0, "#8a1c1c"),
)


def _confirmation_score_color(score):
    avail = score.available_count
    if avail == 0:
        return "#8A8F9C"
    frac = score.passed_count / avail
    for threshold, color in CONFIRMATION_SCORE_COLOR_BANDS:
        if frac >= threshold:
            return color
    return CONFIRMATION_SCORE_COLOR_BANDS[-1][1]


def _confirmation_cell_html(score):
    """Dedicated Confirmation Score column -- see
    utils.breakout_confirmation module docstring for the 9-point
    checklist. Shows the "(passed)/(available)" tally plus which of the
    available checks failed, so the badge is never just a number with no
    backing evidence, same transparency convention as the Quality and
    Regime badges. Checks with no data (currently always Sector
    confirmation) are excluded from the tally rather than shown as a
    failure."""
    if not score or score.available_count == 0:
        return '<span style="font-family:{0};font-size:10.5px;color:#8A8F9C;">Not available</span>'.format(SANS)

    color = _confirmation_score_color(score)
    badge = (
        f'<span style="display:inline-block;padding:3px 8px;border-radius:10px;'
        f'background:#EDEAE2;color:{color};font-family:{SANS};font-size:12px;font-weight:700;white-space:nowrap;">'
        f'{score.label()}</span>'
    )
    failed = score.failed
    if failed:
        failed_html = "".join(
            f'<div style="font-size:10px;color:#8a1c1c;">&#10007; {html.escape(c.label)}</div>' for c in failed
        )
    else:
        failed_html = '<div style="font-size:10px;color:#0f5132;">All available checks passed.</div>'
    return badge + failed_html


EXTENSION_LABEL_COLORS = {
    # label -> (background, text) -- greener the fresher, redder the
    # more extended, same visual language as the Quality badge.
    "Fresh": ("#DCEFE0", "#0f5132"),
    "Controlled": ("#E4F0E9", "#3d7a52"),
    "Extended": ("#FCF1D8", "#7a5b00"),
    "Highly Extended": ("#FBE7DD", "#9a3412"),
    "Chase Risk": ("#F8DADA", "#8a1c1c"),
    "Severe Chase Risk": ("#F8DADA", "#8a1c1c"),
}

CLV_LABEL_COLORS = {
    # label -> text color only -- CLV rides inline in the Extension
    # detail line rather than getting its own badge, so just the text
    # color changes; same greener-is-better language as everything else.
    "Excellent": "#0f5132",
    "Good": "#3d7a52",
    "Neutral": "#7a5b00",
    "Weak": "#8a1c1c",
}

VOLUME_LABEL_COLORS = {
    # label -> text color only, same inline-line convention as CLV above.
    # Deliberately NOT greener-is-better: Extreme isn't "worse" than
    # Healthy on its own, it's just ambiguous until CLV confirms it (see
    # VOLUME_BANDS / EXTREME_VOLUME_* above) -- neutral gray/blue for
    # Normal/Healthy/Elevated, amber for Extreme as a "look closer" cue,
    # not a verdict.
    "Normal": "#8A8F9C",
    "Healthy": "#3d7a52",
    "Elevated": "#3C4256",
    "Extreme": "#9a3412",
}

RELATIVE_STRENGTH_BADGE_COLORS = {
    # label -> (background, text) -- same greener-is-better language as
    # the Extension/Quality badges.
    "Strong Outperformance": ("#DCEFE0", "#0f5132"),
    "Outperforming": ("#E4F0E9", "#3d7a52"),
    "In-Line With NIFTY": ("#EDEAE2", "#3C4256"),
    "Underperforming": ("#FCF1D8", "#7a5b00"),
    "Weak / Lagging": ("#F8DADA", "#8a1c1c"),
}


def _relative_strength_cell_html(relative_strength):
    """Dedicated Relative Strength column -- see compute_relative_strength
    above. Leads with the blended (20d/50d-weighted) outperformance-vs-
    NIFTY badge, then the two lookback periods broken out individually
    (stock % vs NIFTY %) so the badge is never just a number with no
    backing evidence, same transparency convention as every other score
    in this email."""
    if not relative_strength or relative_strength["score"] is None:
        return '<span style="font-family:{0};font-size:10.5px;color:#8A8F9C;">Not available</span>'.format(SANS)

    bg, fg = RELATIVE_STRENGTH_BADGE_COLORS.get(relative_strength["label"], ("#EDEAE2", "#3C4256"))
    badge = (
        f'<span style="display:inline-block;padding:3px 8px;border-radius:10px;'
        f'background:{bg};color:{fg};font-family:{SANS};font-size:12px;font-weight:700;white-space:nowrap;">'
        f'{relative_strength["score"]:+.1f}pp vs NIFTY &middot; {html.escape(relative_strength["label"])}</span>'
    )
    detail_bits = []
    for lookback, _ in RS_LOOKBACKS:
        p = relative_strength["periods"].get(lookback)
        if p and p["stock_pct"] is not None and p["nifty_pct"] is not None:
            detail_bits.append(
                f'{lookback}d: {p["stock_pct"]:+.1f}% vs NIFTY {p["nifty_pct"]:+.1f}%'
            )
        elif p and p["stock_pct"] is not None:
            detail_bits.append(f'{lookback}d: {p["stock_pct"]:+.1f}% (NIFTY unavailable)')
    detail = (
        f'<div style="margin-top:3px;font-size:10px;color:#8A8F9C;">{" &middot; ".join(detail_bits)}</div>'
        if detail_bits else ""
    )
    return badge + detail


SECTOR_CONFIRMATION_BADGE_COLORS = {
    # (passed_count/available_count fraction range) -> (background, text)
    # -- same greener-is-better language as everything else, bucketed
    # coarsely since sector confirmation is a 4-point tally, not a
    # continuous score, for badge-color purposes.
    "strong": ("#DCEFE0", "#0f5132"),    # 4/4 or 3/4 (or 3/3, 2/3...) passed
    "mixed": ("#FCF1D8", "#7a5b00"),     # roughly half passed
    "weak": ("#F8DADA", "#8a1c1c"),      # 0-1 of the available checks passed
}


def _sector_confirmation_cell_html(sector_confirmation):
    """Dedicated Sector Confirmation column -- see
    compute_sector_confirmation above. Leads with the sector name and
    (passed)/(available) tally -- same convention as the Confirmation
    column -- then breaks out each of the 4 checks individually so the
    badge is never just a number with no backing evidence."""
    if not sector_confirmation:
        return (
            f'<span style="font-family:{SANS};font-size:10.5px;color:#8A8F9C;">'
            f'N/A &mdash; {html.escape(SECTOR_DATA_NOTE_LOCAL)}</span>'
        )

    available = sector_confirmation["available_count"]
    passed = sector_confirmation["passed_count"]
    if available == 0:
        return '<span style="font-family:{0};font-size:10.5px;color:#8A8F9C;">Not available</span>'.format(SANS)

    frac = passed / available
    tier = "strong" if frac >= 0.75 else "weak" if frac <= 0.25 else "mixed"
    bg, fg = SECTOR_CONFIRMATION_BADGE_COLORS[tier]
    badge = (
        f'<span style="display:inline-block;padding:3px 8px;border-radius:10px;'
        f'background:{bg};color:{fg};font-family:{SANS};font-size:12px;font-weight:700;white-space:nowrap;">'
        f'{html.escape(sector_confirmation["sector"])} &middot; {passed}/{available}</span>'
    )

    def _mark(val):
        if val is None:
            return None
        return "&#10003;" if val else "&#10007;"

    detail_bits = []
    m = _mark(sector_confirmation["above_20dma"])
    if m:
        detail_bits.append(f'{m} &gt;20DMA')
    m = _mark(sector_confirmation["above_50dma"])
    if m:
        detail_bits.append(f'{m} &gt;50DMA')
    rs = sector_confirmation["relative_strength_pct"]
    if rs is not None:
        m = "&#10003;" if rs > 0 else "&#10007;"
        detail_bits.append(f'{m} RS {rs:+.1f}pp vs NIFTY')
    breadth = sector_confirmation["breadth"]
    if breadth is not None:
        m = "&#10003;" if breadth >= 0.5 else "&#10007;"
        detail_bits.append(f'{m} breadth {breadth*100:.0f}%')
    detail = (
        f'<div style="margin-top:3px;font-size:10px;color:#8A8F9C;">{" &middot; ".join(detail_bits)}</div>'
        if detail_bits else ""
    )
    return badge + detail


def _extension_cell_html(extension):
    """Dedicated Extension column -- see compute_extension_score above.
    Leads with today's own Close-over-Close move and the Extension Score
    badge/label, then the supporting distance reads (20/50DMA, breakout
    level, ATR) so the badge is never just a number with no backing
    evidence, same transparency convention as every other score in this
    email. Volume band (Normal/Healthy/Elevated/Extreme -- see
    VOLUME_BANDS) is shown on every row with volume data, not just
    flagged ones, so a raw multiple never reads as unqualified strength.
    A ⚠ CHASE RISK line appears only when the row was actually pulled out
    of Confirmed by that gate (see extension_downgraded, and
    row["extension"]["chase_risk"]); a ⚠ EXTREME VOLUME, WEAK CLOSE line
    appears only when the row was pulled out by the separate extreme-
    volume gate (see extreme_volume_downgraded, EXTREME_VOLUME_*)."""
    if not extension:
        return '<span style="font-family:{0};font-size:10.5px;color:#8A8F9C;">Not available</span>'.format(SANS)

    bg, fg = EXTENSION_LABEL_COLORS.get(extension["label"], ("#EDEAE2", "#3C4256"))
    badge = (
        f'<span style="display:inline-block;padding:3px 8px;border-radius:10px;'
        f'background:{bg};color:{fg};font-family:{SANS};font-size:12px;font-weight:700;white-space:nowrap;">'
        f'{extension["today_move_pct"]:+.1f}% today &middot; {html.escape(extension["label"])}</span>'
    )
    detail_bits = []
    if extension["dist_20dma_pct"] is not None:
        detail_bits.append(f'20DMA {extension["dist_20dma_pct"]:+.1f}%')
    if extension["dist_50dma_pct"] is not None:
        detail_bits.append(f'50DMA {extension["dist_50dma_pct"]:+.1f}%')
    if extension["dist_breakout_pct"] is not None:
        detail_bits.append(f'vs breakout {extension["dist_breakout_pct"]:+.1f}%')
    if extension["atr_multiple_above_breakout"] is not None:
        detail_bits.append(f'{extension["atr_multiple_above_breakout"]:.1f}&times; ATR above breakout')
    detail = (
        f'<div style="margin-top:3px;font-size:10px;color:#8A8F9C;">{" &middot; ".join(detail_bits)}</div>'
        if detail_bits else ""
    )
    clv_line = ""
    if extension["clv"] is not None:
        clv_color = CLV_LABEL_COLORS.get(extension["clv_label"], "#8A8F9C")
        clv_line = (
            f'<div style="margin-top:3px;font-size:10px;color:{clv_color};">'
            f'CLV {extension["clv"]:.2f} &middot; {html.escape(extension["clv_label"])} close</div>'
        )
    # Volume band shown on every row with volume data -- not just chase-risk
    # or extreme-volume-downgraded rows -- so a raw multiple never reads
    # unlabeled/unqualified (see VOLUME_BANDS).
    volume_line = ""
    if extension["volume_multiple"] is not None:
        vol_color = VOLUME_LABEL_COLORS.get(extension["volume_label"], "#8A8F9C")
        volume_line = (
            f'<div style="margin-top:3px;font-size:10px;color:{vol_color};">'
            f'Volume {extension["volume_multiple"]:.1f}&times; avg &middot; {html.escape(extension["volume_label"])}</div>'
        )
    chase_note = ""
    if extension["chase_risk"]:
        vol_txt = f'{extension["volume_multiple"]:.1f}&times;' if extension["volume_multiple"] is not None else "elevated"
        chase_note = (
            f'<div style="margin-top:3px;font-size:10px;font-weight:700;color:#8a1c1c;">'
            f'\u26a0 CHASE RISK &middot; {vol_txt} avg volume</div>'
        )
    extreme_volume_note = ""
    if extension["volume_label"] == "Extreme" and extension["clv"] is not None and extension["clv"] < EXTREME_VOLUME_MIN_CLV:
        extreme_volume_note = (
            f'<div style="margin-top:3px;font-size:10px;font-weight:700;color:#8a1c1c;">'
            f'\u26a0 EXTREME VOLUME, WEAK CLOSE &middot; needs structure confirmation</div>'
        )
    return badge + detail + clv_line + volume_line + chase_note + extreme_volume_note


def _vwap_cell_html(vwap):
    """Dedicated VWAP column -- see compute_vwap_context above and the
    VWAP block comment for the daily-bar-vs-intraday-session caveat.
    Leads with the Above/Below badge, then the two underlying figures
    (Trailing, Anchored) so the badge is never just a checkmark with no
    number behind it, same transparency convention as every other score
    in this email."""
    if not vwap or vwap["label"] is None:
        return '<span style="font-family:{0};font-size:10.5px;color:#8A8F9C;">Not available</span>'.format(SANS)

    bg, fg = VWAP_LABEL_COLORS.get(vwap["label"], ("#EDEAE2", "#3C4256"))
    badge = (
        f'<span style="display:inline-block;padding:3px 8px;border-radius:10px;'
        f'background:{bg};color:{fg};font-family:{SANS};font-size:12px;font-weight:700;white-space:nowrap;">'
        f'{html.escape(vwap["label"])}</span>'
    )
    detail_bits = []
    if vwap["anchored_vwap"] is not None:
        mark = "&#10003;" if vwap["price_above_anchored_vwap"] else "&#10007;"
        detail_bits.append(f'{mark} Anchored \u20b9{vwap["anchored_vwap"]:,.2f}')
    if vwap["trailing_vwap"] is not None:
        mark = "&#10003;" if vwap["price_above_trailing_vwap"] else "&#10007;"
        detail_bits.append(f'{mark} {VWAP_TRAILING_PERIOD}d \u20b9{vwap["trailing_vwap"]:,.2f}')
    detail = (
        f'<div style="margin-top:3px;font-size:10px;color:#8A8F9C;">{" &middot; ".join(detail_bits)}</div>'
        if detail_bits else ""
    )
    return badge + detail


SETUP_STRENGTH_BADGE_COLORS = {
    # label -> (background, text) -- reuses the same visual language as
    # the Quality badge so "strong" always means the same color across
    # this email.
    "Very Strong": ("#DCEFE0", "#0f5132"),
    "Strong": ("#E4F0E9", "#3d7a52"),
    "Moderate": ("#FCF1D8", "#7a5b00"),
    "Weak": ("#F8DADA", "#8a1c1c"),
}


def _setup_strength_cell_html(strength):
    """Dedicated Setup Strength column -- FOUR sub-scores (Breakout
    Quality, Trade Quality, Risk:Reward, Market Regime -- see
    compute_setup_strength above) blended into one headline number, with
    each sub-score also shown so a reader can see WHICH question is
    dragging the row down instead of just a single opaque percentage.
    Same transparency convention as every other score in this email --
    and explicitly NOT framed as a guarantee or an override of the row's
    Confirmed/Watch/Filtered bucket; see the disclaimer in the report
    footer."""
    if not strength:
        return '<span style="font-family:{0};font-size:10.5px;color:#8A8F9C;">Not enough data</span>'.format(SANS)

    bg, fg = SETUP_STRENGTH_BADGE_COLORS.get(strength["label"], ("#EDEAE2", "#3C4256"))
    badge = (
        f'<span style="display:inline-block;min-width:34px;text-align:center;padding:3px 8px;border-radius:10px;'
        f'background:{bg};color:{fg};font-family:{SANS};font-size:12px;font-weight:700;">{strength["score"]}%</span>'
        f'<div style="margin-top:3px;font-family:{SANS};font-size:10px;color:{fg};">{html.escape(strength["label"])}</div>'
    )
    note = (
        f'<div style="margin-top:2px;font-size:9.5px;color:#8A8F9C;">'
        f'{strength["components_used"]}/{strength["components_total"]} scores available</div>'
    )
    # Sub-score breakdown -- abbreviated labels so this fits in a narrow
    # table cell; "--" for a sub-score that wasn't available for this row
    # (e.g. no Risk:Reward because there's no exact entry price) rather
    # than a misleading 0.
    def _sub(label, sub):
        val = f'{sub["score"]}%' if sub else '--'
        return f'{label} {val}'

    breakdown = " &middot; ".join([
        _sub("BQ", strength.get("breakout_quality")),
        _sub("TQ", strength.get("trade_quality")),
        _sub("R:R", strength.get("risk_reward")),
        _sub("Regime", strength.get("market_regime")),
        _sub("Sector", strength.get("sector_confirmation")),
    ])
    note += (
        f'<div style="margin-top:2px;font-size:9.5px;color:#8A8F9C;">{breakdown}</div>'
    )
    return badge + note


def _fundamentals_note_html(fundamentals):
    """Sub-line under the Quality badge showing the fundamental-quality
    read (ROE / Debt-Equity / Earnings growth) for rows where it was
    computed -- currently only rows that cleared the technical bar for
    Confirmed. Other buckets pass row.get('fundamentals') == None and get
    nothing rendered here."""
    if fundamentals is None:
        return ""
    if not fundamentals["available"]:
        return (
            f'<div style="margin-top:3px;font-size:9.5px;color:#8A8F9C;">Fundamentals: data unavailable</div>'
        )
    color = "#0f5132" if fundamentals["passed"] else "#8a1c1c"
    mark = "&#10003;" if fundamentals["passed"] else "&#10007;"
    detail = " &middot; ".join(f"{html.escape(label)} {html.escape(val)}" for label, _, val in fundamentals["checks"])
    return (
        f'<div style="margin-top:3px;font-size:9.5px;color:{color};">{mark} Fundamentals: {detail}</div>'
    )


def _row_bg_for_symbol(symbol, default_bg):
    """Light-green override when this row's symbol is one of Bala's own
    stocks (constants.STOCKS) -- lets the personal holdings jump out
    from the wider NIFTY 500 scan at a glance. Falls back to the
    table's normal alternating background otherwise."""
    if symbol.upper() in MY_STOCK_SYMBOLS:
        return MY_STOCK_ROW_BG
    return default_bg


def _row_bg_for_row(row, default_bg):
    """Background override, checked in priority order:
      1. Best Execution Case (light sky blue) -- see is_best_execution_case.
         Takes priority over the own-stock shading below since it's the
         stronger, more actionable signal.
      2. Own-stock symbol (light green) -- see MY_STOCK_ROW_BG.
      3. The table's normal alternating background otherwise."""
    if row.get("best_execution"):
        return BEST_EXECUTION_ROW_BG
    return _row_bg_for_symbol(row["symbol"], default_bg)


def _signal_rows_html(rows, row_bg):
    if not rows:
        return '<tr><td style="padding:10px 12px;font-family:{0};font-size:12px;color:#8A8F9C;" colspan="15">None today.</td></tr>'.format(SANS)

    out = []
    for row in rows:
        emoji = PATTERN_STYLES.get(row["pattern"], "🔹")
        caution = ""
        if row["dq_notes"]:
            caution = (
                f'<div style="margin-top:3px;font-size:10.5px;color:#9a3412;">'
                f'⚠ {html.escape("; ".join(row["dq_notes"]))}</div>'
            )
        regime_note = ""
        if row.get("regime_downgraded"):
            regime_note = (
                f'<div style="margin-top:3px;font-size:10.5px;color:#7a5b00;">'
                f'\u2b07 Cleared the backtest, but held back from Confirmed by the 🔴 Bearish-regime filter '
                f'(needs Quality \u2265{BEAR_MARKET_MIN_QUALITY_SCORE} and preferred Risk:Reward).</div>'
            )
        fund_note = ""
        if row.get("fundamental_downgraded"):
            fund_note = (
                f'<div style="margin-top:3px;font-size:10.5px;color:#7a5b00;">'
                f'\u2b07 Cleared the backtest, but held back from Confirmed by the fundamentals gate '
                f'(needs ROE \u2265{FUNDAMENTAL_MIN_ROE*100:.0f}%, Debt/Equity \u2264{FUNDAMENTAL_MAX_DEBT_TO_EQUITY:.0f}%, '
                f'non-negative earnings growth -- see Quality column).</div>'
            )
        strict_bar_note = ""
        if row.get("strict_bar_downgraded"):
            strict_bar_note = (
                f'<div style="margin-top:3px;font-size:10.5px;color:#7a5b00;">'
                f'\u2b07 Cleared the standard backtest bar, but held back from Confirmed by the tiered sample/hit-rate bar '
                f'(needs Tier A/B/C -- see disclaimer below for the full table).</div>'
            )
        extension_note = ""
        if row.get("extension_downgraded"):
            extension_note = (
                f'<div style="margin-top:3px;font-size:10.5px;color:#7a5b00;">'
                f'\u2b07 Cleared the backtest, but held back from Confirmed by the extension/chase-risk gate '
                f'(\u2265{CHASE_RISK_VOLUME_MULTIPLE:.0f}&times; avg volume on an already-extended move -- see Extension column).</div>'
            )
        extreme_volume_note = ""
        if row.get("extreme_volume_downgraded"):
            extreme_volume_note = (
                f'<div style="margin-top:3px;font-size:10.5px;color:#7a5b00;">'
                f'\u2b07 Cleared the backtest, but held back from Confirmed by the extreme-volume gate '
                f'(\u2265{EXTREME_VOLUME_MULTIPLE:.0f}&times; avg volume without a confirming close, CLV &lt;{EXTREME_VOLUME_MIN_CLV:.2f} '
                f'-- see Extension column).</div>'
            )
        best_execution_note = ""
        if row.get("best_execution"):
            best_execution_note = (
                f'<div style="margin-top:3px;font-size:10.5px;font-weight:700;color:#0B4C7C;">'
                f'{BEST_EXECUTION_LABEL}</div>'
            )
        bg = _row_bg_for_row(row, row_bg)
        out.append(f"""
        <tr style="background:{bg};">
          <td data-label="Symbol" style="padding:9px 12px;font-family:{SANS};font-size:13px;font-weight:700;color:#1F2430;border-bottom:1px solid #EDEAE2;">{html.escape(row['symbol'])}{best_execution_note}</td>
          <td data-label="Pattern" style="padding:9px 12px;font-family:{SANS};font-size:12px;color:#3C4256;border-bottom:1px solid #EDEAE2;">{emoji} {html.escape(row['pattern'])}<div style="margin-top:2px;font-size:10.5px;color:#8A8F9C;">{html.escape(row['detail'])}</div>{caution}{regime_note}{fund_note}{strict_bar_note}{extension_note}{extreme_volume_note}</td>
          <td data-label="Price" style="padding:9px 12px;font-family:{SANS};font-size:12px;color:#3C4256;border-bottom:1px solid #EDEAE2;text-align:right;">₹{row['signal_price']:,.2f}</td>
          <td data-label="Backtest" style="padding:9px 12px;font-family:{SANS};font-size:11.5px;color:#3C4256;border-bottom:1px solid #EDEAE2;">{_backtest_cell(row)}</td>
          <td data-label="Quality" style="padding:9px 12px;border-bottom:1px solid #EDEAE2;text-align:center;">{_quality_badge_html(row['quality'])}{_fundamentals_note_html(row.get('fundamentals'))}</td>
          <td data-label="Entry" style="padding:9px 12px;border-bottom:1px solid #EDEAE2;">{_entry_badge_html(row.get('entry'))}</td>
          <td data-label="Stop-Loss" style="padding:9px 12px;border-bottom:1px solid #EDEAE2;">{_stop_loss_cell_html(row.get('entry'))}</td>
          <td data-label="Targets" style="padding:9px 12px;border-bottom:1px solid #EDEAE2;">{_targets_cell_html(row.get('targets'))}</td>
          <td data-label="Risk:Reward" style="padding:9px 12px;border-bottom:1px solid #EDEAE2;">{_risk_reward_cell_html(row.get('risk_reward'))}</td>
          <td data-label="Failure Risk" style="padding:9px 12px;border-bottom:1px solid #EDEAE2;">{_failure_risk_cell_html(row.get('failure_risk'))}</td>
          <td data-label="Extension" style="padding:9px 12px;border-bottom:1px solid #EDEAE2;">{_extension_cell_html(row.get('extension'))}</td>
          <td data-label="VWAP" style="padding:9px 12px;border-bottom:1px solid #EDEAE2;">{_vwap_cell_html(row.get('vwap'))}</td>
          <td data-label="Relative Strength" style="padding:9px 12px;border-bottom:1px solid #EDEAE2;">{_relative_strength_cell_html(row.get('relative_strength'))}</td>
          <td data-label="Confirmation" style="padding:9px 12px;border-bottom:1px solid #EDEAE2;">{_confirmation_cell_html(row.get('confirmation'))}</td>
          <td data-label="Sector" style="padding:9px 12px;border-bottom:1px solid #EDEAE2;">{_sector_confirmation_cell_html(row.get('sector_confirmation'))}</td>
          <td data-label="Setup Strength" style="padding:9px 12px;border-bottom:1px solid #EDEAE2;text-align:center;">{_setup_strength_cell_html(row.get('setup_strength'))}</td>
        </tr>
        """)
    return "".join(out)


def _near_breakout_rows_html(rows, row_bg):
    if not rows:
        return '<tr><td style="padding:10px 12px;font-family:{0};font-size:12px;color:#8A8F9C;" colspan="4">None today.</td></tr>'.format(SANS)

    out = []
    for row in rows:
        entry = row["entry"]
        caution = ""
        if row["dq_notes"]:
            caution = (
                f'<div style="margin-top:3px;font-size:10.5px;color:#9a3412;">'
                f'⚠ {html.escape("; ".join(row["dq_notes"]))}</div>'
            )
        distance = entry.distance_to_breakout_pct
        distance_str = f"{distance:.1f}% below trailing resistance" if distance is not None else "—"
        bg = _row_bg_for_symbol(row["symbol"], row_bg)
        out.append(f"""
        <tr style="background:{bg};">
          <td data-label="Symbol" style="padding:9px 12px;font-family:{SANS};font-size:13px;font-weight:700;color:#1F2430;border-bottom:1px solid #EDEAE2;">{html.escape(row['symbol'])}</td>
          <td data-label="Price" style="padding:9px 12px;font-family:{SANS};font-size:12px;color:#3C4256;border-bottom:1px solid #EDEAE2;text-align:right;">₹{row['current_price']:,.2f}</td>
          <td data-label="Distance" style="padding:9px 12px;font-family:{SANS};font-size:11.5px;color:#3C4256;border-bottom:1px solid #EDEAE2;">{html.escape(distance_str)}{caution}</td>
          <td data-label="Entry" style="padding:9px 12px;border-bottom:1px solid #EDEAE2;">{_entry_badge_html(entry)}</td>
        </tr>
        """)
    return "".join(out)


def _table_block(title, subtitle, rows, row_bg, accent):
    header = (
        f'<tr class="table-header-row"><td style="padding:6px 12px;font-family:{SANS};font-size:10px;font-weight:700;'
        f'color:#3C4256;border-bottom:1px solid #DAD5CB;">Symbol</td>'
        f'<td style="padding:6px 12px;font-family:{SANS};font-size:10px;font-weight:700;'
        f'color:#3C4256;border-bottom:1px solid #DAD5CB;">Pattern</td>'
        f'<td style="padding:6px 12px;font-family:{SANS};font-size:10px;font-weight:700;'
        f'color:#3C4256;border-bottom:1px solid #DAD5CB;text-align:right;">Price</td>'
        f'<td style="padding:6px 12px;font-family:{SANS};font-size:10px;font-weight:700;'
        f'color:#3C4256;border-bottom:1px solid #DAD5CB;">Backtest ({PRIMARY_HORIZON}d)</td>'
        f'<td style="padding:6px 12px;font-family:{SANS};font-size:10px;font-weight:700;'
        f'color:#3C4256;border-bottom:1px solid #DAD5CB;text-align:center;">Quality</td>'
        f'<td style="padding:6px 12px;font-family:{SANS};font-size:10px;font-weight:700;'
        f'color:#3C4256;border-bottom:1px solid #DAD5CB;">Entry</td>'
        f'<td style="padding:6px 12px;font-family:{SANS};font-size:10px;font-weight:700;'
        f'color:#3C4256;border-bottom:1px solid #DAD5CB;">Stop-Loss</td>'
        f'<td style="padding:6px 12px;font-family:{SANS};font-size:10px;font-weight:700;'
        f'color:#3C4256;border-bottom:1px solid #DAD5CB;">Targets</td>'
        f'<td style="padding:6px 12px;font-family:{SANS};font-size:10px;font-weight:700;'
        f'color:#3C4256;border-bottom:1px solid #DAD5CB;">Risk:Reward</td>'
        f'<td style="padding:6px 12px;font-family:{SANS};font-size:10px;font-weight:700;'
        f'color:#3C4256;border-bottom:1px solid #DAD5CB;">Failure Risk ({FAILURE_WINDOW_DAYS}d)</td>'
        f'<td style="padding:6px 12px;font-family:{SANS};font-size:10px;font-weight:700;'
        f'color:#3C4256;border-bottom:1px solid #DAD5CB;">Extension</td>'
        f'<td style="padding:6px 12px;font-family:{SANS};font-size:10px;font-weight:700;'
        f'color:#3C4256;border-bottom:1px solid #DAD5CB;">VWAP</td>'
        f'<td style="padding:6px 12px;font-family:{SANS};font-size:10px;font-weight:700;'
        f'color:#3C4256;border-bottom:1px solid #DAD5CB;">Relative Strength</td>'
        f'<td style="padding:6px 12px;font-family:{SANS};font-size:10px;font-weight:700;'
        f'color:#3C4256;border-bottom:1px solid #DAD5CB;">Confirmation</td>'
        f'<td style="padding:6px 12px;font-family:{SANS};font-size:10px;font-weight:700;'
        f'color:#3C4256;border-bottom:1px solid #DAD5CB;">Sector</td>'
        f'<td style="padding:6px 12px;font-family:{SANS};font-size:10px;font-weight:700;'
        f'color:#3C4256;border-bottom:1px solid #DAD5CB;text-align:center;">Setup Strength</td></tr>'
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
              <table width="100%" cellpadding="0" cellspacing="0" role="presentation" class="stack-table" style="border-collapse:collapse;">
                {header}
                {rows_html}
              </table>
            </td>
          </tr>
        </table>
      </td>
    </tr>
    """


REGIME_BLOCK_COLORS = {
    # regime -> (background, border, accent text)
    MarketRegime.BULLISH: ("#EEF7F0", "#CDE7D4", "#0f5132"),
    MarketRegime.NEUTRAL: ("#FCF7EA", "#EFDFAF", "#7a5b00"),
    MarketRegime.BEARISH: ("#FBEEEC", "#F2C9C2", "#8a1c1c"),
}


def _regime_block(regime: RegimeResult):
    """NIFTY 500 market regime -- computed once per run by
    utils.market_regime.compute_market_regime() and used to gate what
    gets shown as Confirmed (see scan_universe's regime_downgraded logic).
    Renders the 5-check breakdown so the badge is never just a color with
    no backing evidence, same transparency convention as the Quality Score."""
    bg, border, accent = REGIME_BLOCK_COLORS.get(regime.regime, ("#FAF8F1", "#E7DFC9", "#3C4256"))
    d = regime.detail

    def _check_row(label):
        passed = regime.checks.get(label)
        mark = "✔" if passed else "✘"
        color = "#0f5132" if passed else "#8a1c1c"
        return (
            f'<span style="display:inline-block;margin:2px 10px 2px 0;font-family:{SANS};font-size:11px;color:{color};">'
            f'{mark} {html.escape(label)}</span>'
        )

    checks_html = "".join(_check_row(label) for label in REGIME_CHECK_LABELS)

    detail_bits = []
    if d.get("nifty_close") is not None:
        detail_bits.append(
            f'NIFTY 50 {d["nifty_close"]:,.0f} vs 50DMA {d["nifty_50dma"]:,.0f} / 200DMA {d["nifty_200dma"]:,.0f}'
        )
    if d.get("nifty_resistance") is not None:
        detail_bits.append(f'prior resistance {d["nifty_resistance"]:,.0f}')
    if d.get("breadth_pct") is not None:
        detail_bits.append(f'breadth {d["breadth_pct"]:.0f}% ({d["breadth_above"]}/{d["breadth_total"]} above 50DMA)')
    if d.get("vix") is not None:
        detail_bits.append(f'India VIX {d["vix"]:.1f}')
    detail_line = " &middot; ".join(detail_bits) if detail_bits else "Underlying index/breadth/VIX data unavailable this run."

    notes_html = ""
    if regime.notes:
        notes_html = (
            f'<div style="margin-top:6px;font-family:{SANS};font-size:10px;color:#8A8F9C;">'
            f'⚠ {html.escape(" ".join(regime.notes))}</div>'
        )

    return f"""
    <tr>
      <td style="padding:0 28px 18px;" class="email-padding">
        <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="border-radius:6px;background:{bg};border:1px solid {border};">
          <tr>
            <td style="padding:14px 16px;">
              <div style="font-family:{SANS};font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:{accent};">Market Regime &mdash; NIFTY 500</div>
              <div style="margin:4px 0 6px;font-family:{SERIF};font-size:17px;color:{accent};">{regime.label()} <span style="font-family:{SANS};font-size:11px;font-weight:400;color:#8A8F9C;">({regime.score}/5 checks)</span></div>
              <div>{checks_html}</div>
              <div style="margin-top:8px;font-family:{SANS};font-size:11px;color:#3C4256;">{html.escape(detail_line)}</div>
              <div style="margin-top:6px;font-family:{SANS};font-size:11.5px;color:{accent};">{html.escape(regime.interpretation())}</div>
              {notes_html}
            </td>
          </tr>
        </table>
      </td>
    </tr>
    """


def _near_breakout_block(rows):
    header = (
        f'<tr class="table-header-row"><td style="padding:6px 12px;font-family:{SANS};font-size:10px;font-weight:700;'
        f'color:#3C4256;border-bottom:1px solid #DAD5CB;">Symbol</td>'
        f'<td style="padding:6px 12px;font-family:{SANS};font-size:10px;font-weight:700;'
        f'color:#3C4256;border-bottom:1px solid #DAD5CB;text-align:right;">Price</td>'
        f'<td style="padding:6px 12px;font-family:{SANS};font-size:10px;font-weight:700;'
        f'color:#3C4256;border-bottom:1px solid #DAD5CB;">Distance</td>'
        f'<td style="padding:6px 12px;font-family:{SANS};font-size:10px;font-weight:700;'
        f'color:#3C4256;border-bottom:1px solid #DAD5CB;">Entry</td></tr>'
    )
    rows_html = _near_breakout_rows_html(rows, "#ffffff")
    return f"""
    <tr>
      <td style="padding:0 28px 18px;" class="email-padding">
        <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="border-radius:6px;background:#FAF8F1;border:1px solid #E7DFC9;">
          <tr>
            <td style="padding:14px 16px 4px;">
              <div style="font-family:{SANS};font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#1c4a8a;">🔵 Near Breakout &mdash; Watch</div>
              <div style="margin:2px 0 8px;font-family:{SANS};font-size:11px;color:#8A8F9C;">No pattern fired today, but price is within {ENTRY_CLASSIFIER_CONFIG.near_breakout_pct:.0f}% of trailing resistance -- not a buy yet, watch for the trigger price with volume confirmation.</div>
              <table width="100%" cellpadding="0" cellspacing="0" role="presentation" class="stack-table" style="border-collapse:collapse;">
                {header}
                {rows_html}
              </table>
            </td>
          </tr>
        </table>
      </td>
    </tr>
    """


def build_report_html(confirmed, watch_list, filtered_low_rr, near_breakout_watch, scan_stats):
    now_ist = dt.datetime.now(ZoneInfo("Asia/Kolkata"))
    date_str = get_date_with_suffix(now_ist)
    regime: RegimeResult = scan_stats["regime"]

    regime_block = _regime_block(regime)

    confirmed_subtitle = (
        f"Backtest cleared the bar: reached Tier A (\u226530 occurrences, \u226560% {PRIMARY_HORIZON}-day hit-rate), "
        f"Tier B (15-29 occurrences, \u226560% hit-rate), or Tier C (10-14 occurrences, \u226565% hit-rate) -- "
        f"see disclaimer below for why the hit-rate bar rises as the sample thins out. Also: "
        f"Risk:Reward \u2265 {RISK_REWARD_MIN_THRESHOLD:.1f}, (where fundamentals data is available) ROE \u2265{FUNDAMENTAL_MIN_ROE*100:.0f}%, "
        f"Debt/Equity \u2264{FUNDAMENTAL_MAX_DEBT_TO_EQUITY:.0f}%, non-negative earnings growth, and not flagged Chase Risk on "
        f"\u2265{CHASE_RISK_VOLUME_MULTIPLE:.0f}&times; avg volume with an already-extended move (see Extension column), and not "
        f"flagged on \u2265{EXTREME_VOLUME_MULTIPLE:.0f}&times; avg volume without a confirming close (CLV \u2265{EXTREME_VOLUME_MIN_CLV:.2f}) -- "
        "high quality on the chart and the balance sheet, not a chase or an unconfirmed volume spike. "
        "Entry column shows whether this is chaseable now, better bought on a retest, or neither -- see disclaimer below."
    )
    if regime.only_strongest:
        confirmed_subtitle += (
            f" 🔴 Bearish regime is active, so this list is also restricted to Quality \u2265{BEAR_MARKET_MIN_QUALITY_SCORE} "
            f"and preferred Risk:Reward (\u2265{RISK_REWARD_PREFERRED_THRESHOLD:.1f}) -- see Market Regime above."
        )
    confirmed_subtitle += (
        f' Rows shaded <span style="background:{BEST_EXECUTION_ROW_BG};padding:0 3px;">light sky blue</span> are this '
        f"run's {BEST_EXECUTION_LABEL} rows -- Confirmed, Fresh Breakout/Retest (a concrete entry price today), "
        f"Setup Strength \u2265{BEST_EXECUTION_MIN_SETUP_STRENGTH}, Risk:Reward \u2265{RISK_REWARD_PREFERRED_THRESHOLD:.1f}, "
        "and Failure Risk not flagged elevated -- see disclaimer below for the full rule."
    )
    confirmed_block = _table_block(
        "✅ Confirmed Breakouts",
        confirmed_subtitle,
        confirmed[:MAX_CONFIRMED_ROWS], "#ffffff", "#0f5132",
    )
    # -----------------------------------------------------------------
    # Retest Candidates -- Strategy B (Conservative) rows pulled out of
    # Confirmed Breakouts into their own highlighted section, since this
    # is generally the better risk/reward entry (buying close to the
    # level that has to hold, not several percent above it after chasing
    # the breakout bar) -- see STRATEGY_LABEL_BY_ENTRY_STATE above and
    # the disclaimer below for the Aggressive/Conservative framing. This
    # is purely a re-presentation, not a new filter: every row here is
    # ALREADY in the Confirmed Breakouts table above with the same
    # numbers -- it's surfaced a second time, alone, so a reader who
    # specifically wants "wait for the retest, don't chase" candidates
    # doesn't have to hunt for the 🟡 Retest badge row-by-row.
    # -----------------------------------------------------------------
    retest_candidates = [r for r in confirmed if r.get("entry") and r["entry"].state == EntryState.RETEST]
    retest_block = ""
    if retest_candidates:
        retest_block = _table_block(
            "🎯 Retest Candidates (Strategy B \u2014 Conservative)",
            "Confirmed Breakouts that pulled back to the breakout level, held it as support, and confirmed with a "
            "bullish candle and rising volume -- usually a tighter stop and better Risk:Reward than buying the "
            "breakout bar itself. Duplicated from Confirmed Breakouts above for visibility, not a separate list.",
            retest_candidates[:MAX_RETEST_ROWS], "#ffffff", "#7a5b00",
        )
    watch_subtitle = (
        f"Pattern fired today but the historical sample was thin or the hit-rate was weak -- treat as a watch item, not a call. "
        f"(Still Risk:Reward ≥ {RISK_REWARD_MIN_THRESHOLD:.1f} where R:R data was available.)"
    )
    if regime.only_strongest:
        watch_subtitle += " Rows marked with ⬇ below cleared the backtest but were held back from Confirmed by the Bearish-regime filter."
    watch_subtitle += " Rows marked with ⬇ and 'fundamentals gate' cleared the backtest but were held back from Confirmed by weak/unavailable fundamentals."
    watch_subtitle += f" Rows marked with ⬇ and 'tiered sample/hit-rate bar' cleared the standard backtest bar but not this screener's tiered Confirmed requirement (Tier A/B/C by occurrence count -- see disclaimer below)."
    watch_subtitle += " Rows marked with ⬇ and 'extension/chase-risk gate' cleared every other bar but were held back from Confirmed because today's move is already deep in a chase-risk band on well-above-average volume -- see Extension column."
    watch_subtitle += f" Rows marked with ⬇ and 'extreme-volume gate' fired on \u2265{EXTREME_VOLUME_MULTIPLE:.0f}\u00d7 avg volume but closed weak (CLV below {EXTREME_VOLUME_MIN_CLV:.2f}) -- extreme volume without a confirming close, held back pending price/structure confirmation."
    watch_block = _table_block(
        "👀 Unconfirmed / Watch List",
        watch_subtitle,
        watch_list[:MAX_WATCH_ROWS], "#ffffff", "#7a5b00",
    )
    filtered_rr_block = _table_block(
        "🚫 Filtered &mdash; Poor Risk/Reward",
        f"Would otherwise have qualified for Confirmed/Watch, but Risk:Reward against the nearest available target came in "
        f"below {RISK_REWARD_MIN_THRESHOLD:.1f} -- shown here, not dropped, so nothing disappears silently. "
        f"Setups with R:R &ge; {RISK_REWARD_PREFERRED_THRESHOLD:.1f} are the ones worth prioritizing in the tables above.",
        filtered_low_rr[:MAX_FILTERED_RR_ROWS], "#ffffff", "#8a1c1c",
    )
    near_breakout_block = _near_breakout_block(near_breakout_watch[:MAX_NEAR_BREAKOUT_ROWS])

    universe_note = (
        f"Scanned {scan_stats['universe_size']} NIFTY 500 symbols "
        f"({'live index list' if scan_stats['universe_is_live'] else 'fallback core list -- live NSE index fetch failed this run'}); "
        f"{scan_stats['history_count']} returned usable price history; "
        f"{scan_stats['skipped_count']} skipped by the data-quality gate; "
        f"{scan_stats['filtered_low_rr_count']} filtered out on Risk:Reward &lt; {RISK_REWARD_MIN_THRESHOLD:.1f}."
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
<meta name="supported-color-schemes" content="light">
<meta name="x-apple-disable-message-reformatting">
<meta name="format-detection" content="telephone=no,date=no,address=no,email=no,url=no">
<title>Daily Breakout Screener</title>
<style>
  body {{ margin:0; padding:0; background:#F2F0EC; }}
  table {{ border-collapse:collapse !important; }}
  /* Stop iOS/macOS Mail's auto dark-mode from repainting our own
     colors -- without this, backgrounds/text can invert unpredictably
     on an iPhone even though color-scheme/supported-color-schemes
     above say "light only". Belt-and-suspenders for iOS Mail. */
  @media (prefers-color-scheme: dark) {{
    body, .email-container {{ background:#F2F0EC !important; }}
  }}
  @media screen and (max-width:600px) {{
    .email-container {{ width:100% !important; max-width:100% !important; }}
    .email-padding {{ padding-left:14px !important; padding-right:14px !important; }}
    /* Wide data tables (13/4 columns) don't fit a phone screen -- on
       screens under 600px, drop the header row and stack every <td> into
       its own full-width block instead, with the column name (from
       data-label) printed above the value. Same information, no
       side-scrolling and no squashed columns. */
    .stack-table, .stack-table tbody, .stack-table tr, .stack-table td {{
      display:block !important;
      width:100% !important;
      box-sizing:border-box !important;
    }}
    .stack-table .table-header-row {{ display:none !important; }}
    .stack-table tr {{
      border-bottom:1px solid #DAD5CB !important;
      padding:10px 0 !important;
    }}
    .stack-table tr:last-child {{ border-bottom:none !important; }}
    .stack-table td {{
      border-bottom:none !important;
      padding:6px 14px !important;
      text-align:left !important;
    }}
    .stack-table td[data-label]:before {{
      content: attr(data-label);
      display:block;
      font-family:{SANS};
      font-size:9.5px;
      font-weight:700;
      letter-spacing:0.08em;
      text-transform:uppercase;
      color:#B0A98C;
      margin-bottom:3px;
    }}
    /* Bump up the smallest text a touch so it's still readable at arm's
       length on a phone, without changing the desktop email at all. */
    .stack-table td, .stack-table td div, .stack-table td span {{ font-size:13px !important; }}
    .stack-table td div[style*="font-size:9"],
    .stack-table td div[style*="font-size:10"] {{ font-size:11.5px !important; }}
    /* The closing legend/disclaimer paragraph is dense reference text --
       10px is too small to read comfortably on a phone. */
    .legend-text {{ font-size:12.5px !important; line-height:1.6 !important; }}
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
          {regime_block}
          {confirmed_block}
          {retest_block}
          {watch_block}
          {filtered_rr_block}
          {near_breakout_block}
          <tr>
            <td style="padding:16px 28px;border-top:1px solid #EDEAE2;" class="email-padding">
              <p class="legend-text" style="margin:0;font-family:{SANS};font-size:10px;color:#8A8F9C;line-height:1.5;">
                Market Regime (top of this email) is a 5-check read on the NIFTY 500 as a whole -- NIFTY vs its
                50/200-day moving averages, NIFTY above its own trailing resistance, breadth (% of this run's
                scanned universe above their own 50-day average), and India VIX -- scored 0-5 and bucketed into
                🟢 Bullish / 🟡 Neutral / 🔴 Bearish. Bullish and Neutral use the normal Confirmed/Watch thresholds
                below; in a Bearish regime, Confirmed also requires a Strong/Excellent Quality score and preferred
                Risk:Reward, and every pattern's "Market trend supportive" Fresh-Breakout check (see Entry column)
                is withheld -- the detectors and backtests themselves never change, only how much of what fires
                gets called Confirmed.
                Every signal above is mechanical (price/volume/pattern rules only, no AI narrative) and every
                "Confirmed" signal has been backtested against that stock's own trailing history at run time --
                see the Backtest column. The Quality score (0-100) is a transparent weighted blend of that
                backtest's hit-rate (45%), sample size (20%), average return (15%), consistency across the
                5/10/20-day horizons (10%), and this run's data-quality notes (10%) -- not a prediction, a
                summary of how well this exact pattern has actually worked on this exact stock before.
                The Entry column is a separate, generic price-structure read (trailing-high resistance, volume,
                RSI, and market regime) on timing only: 🟢 Fresh Breakout means every chase-safety check passed
                today; 🟡 Retest means the stock pulled back to resistance and printed a bullish reversal candle
                -- typically better risk/reward than chasing; 🔵 Near Breakout means it hasn't broken out yet
                but is close, worth a watch, not a buy. A pattern being "Confirmed" and its Entry state being
                Fresh/Retest are independent checks -- read both before acting.
                <b>Two strategies, both already in this report:</b> Fresh Breakout and Retest map onto the classic
                "chase it" vs "wait for it" framing -- labeled directly under the Entry badge as
                <b>Strategy A &mdash; Aggressive</b> (buy the breakout bar itself once it clears resistance on
                strong volume and closes near the day's high -- see Confirmation column for the exact checks) or
                <b>Strategy B &mdash; Conservative</b> (wait for price to pull back to the breakout level, hold as
                support, and confirm with a bullish candle and rising volume before buying). A retest entry is
                usually the better trade mechanically: buying near ₹505 on a pullback to a ₹500 breakout level
                carries a tighter, more logical stop (e.g. ₹490, just under the retest low) than chasing the
                ₹515 breakout bar itself with the same ₹490 stop -- same risk in rupees, more reward available to
                the same target. The 🎯 Retest Candidates section above pulls out exactly the Confirmed rows
                currently offering that Strategy B entry, so it doesn't have to be found row-by-row in the main
                table; every row in it is also already in Confirmed Breakouts, this isn't a separate filter or a
                new signal. Neither strategy label changes which bucket (Confirmed/Watch/Filtered) a row sits in
                or how it's scored -- it's a naming/re-presentation layer over the existing Entry classification,
                not a new gate.
                For Fresh Breakout and Retest
                rows, the Stop-Loss column is not an arbitrary percentage: it compares a structural stop (below
                breakout support, the prior swing low, the pattern low, or the retest low -- whichever sits
                nearest to entry) against an ATR(14)-based stop, and uses whichever is more logical for that
                stock's own volatility -- falling back to the ATR stop if the structural level is too tight or
                too wide relative to ATR. Risk % is the distance from entry to that stop.
                The Targets column (Fresh Breakout / Retest rows only) shows three independent technical
                price objectives from that same entry price: Target 1 is the nearest prior swing-high
                resistance in the stock's own trailing history; Target 2 is the pattern's own measured move
                (e.g. cup depth or flagpole height projected from the breakout, ATR-based if the pattern has
                no natural base geometry); Target 3 is a 1.618x Fibonacci extension of that same base, or a
                further proven swing high, whichever sits farther out. These are transparent projections, not
                predictions -- "not available" means the underlying structure (a swing high, a pattern base)
                wasn't found in this stock's history, not that the target is zero.
                Risk:Reward compares that same stop-loss (risk) against the NEAREST available target (reward) --
                shown as e.g. "1 : 1.84" meaning ₹1.84 of reward for every ₹1 risked. This is a CORE FILTER: any
                setup with R:R below {RISK_REWARD_MIN_THRESHOLD:.1f} is pulled out of Confirmed/Watch into the
                "Filtered -- Poor Risk/Reward" section below, on the same nothing-silently-dropped principle as
                everything else in this email -- a strong backtest hit-rate doesn't matter if the reward on offer
                doesn't clear a sane multiple of the risk. R:R &ge; {RISK_REWARD_PREFERRED_THRESHOLD:.1f} (green badge)
                is worth prioritizing over the rest.
                The Failure Risk column, shown on every row, is a separate, historical number: how often this
                exact pattern has fallen back below its own breakout-day low within {FAILURE_WINDOW_DAYS} days on
                this exact stock before -- a base rate for context, not itself a gate, shown "(sample
                &lt;{MIN_SAMPLES_FOR_FAILURE_RATE})" when there isn't enough history yet to call it reliable.
                The Extension column is an "extension penalty" on how far price has already moved TODAY,
                independent of everything else in this row: two signals can carry the same Quality/R:R/Confirmation
                reads and still be very different trades if one is still near its trigger and the other has already
                run 8-10% intraday -- the second is a chase, not a breakout entry. It scores today's Close-over-Close
                move -- &lt;2% +10, 2-4% +8, 4-6% +5, 6-8% +2, 8-10% -5, &gt;10% -15 -- and labels the row Fresh
                &#8594; Controlled &#8594; Extended &#8594; Highly Extended &#8594; Chase Risk &#8594; Severe Chase
                Risk accordingly, alongside supporting context: distance from the 20- and 50-day averages, distance
                above the breakout level itself, and how many ATR(14)s above that breakout level today's close
                already sits. <b>Extension/chase-risk gate (Confirmed Breakouts only):</b> a row that clears every
                other Confirmed bar is still pulled into Watch List, with a note (⬇ "extension/chase-risk gate",
                see Pattern column), if today's volume is \u2265{CHASE_RISK_VOLUME_MULTIPLE:.0f}&times; its trailing
                20-day average AND today's move already scores below {CHASE_RISK_MAX_SCORE} (the 8-10%/&gt;10% bands)
                AND price is at/above the breakout level -- e.g. the NATIONALUM-style case this gate exists to catch:
                a strong-looking backtest paired with a breakout that's already run too far on heavy volume to be a
                controlled entry. Same "nothing silently dropped" rule as the fundamentals gate above.
                Every row with volume data also shows a <b>Volume band</b> underneath the Extension detail line --
                Normal, Healthy (1.5-3x avg), Elevated (3-5x avg), or Extreme (&gt;5x avg) -- because a raw volume
                multiple alone doesn't tell you a breakout is stronger: KPIL is the cautionary example, 28.3x avg
                volume but only a 62% hit-rate, +1.5% average return, and 19% 3-day failure rate on that exact
                pattern/stock history. Extreme volume is genuinely ambiguous -- it can mean institutional
                accumulation, a news-driven re-rating, a block deal, short covering, panic buying, or same-day
                exhaustion/distribution -- so it needs price/structure confirmation, not just the threshold.
                <b>Extreme-volume gate (Confirmed Breakouts only, separate from the chase-risk gate above):</b> a
                row that clears every other Confirmed bar is still pulled into Watch List, with a note (⬇
                "extreme-volume gate", see Pattern column), if today's volume is
                \u2265{EXTREME_VOLUME_MULTIPLE:.0f}&times; its trailing 20-day average AND today's CLV (Close
                Location Value -- where the close sits within today's own high-low range) is below
                {EXTREME_VOLUME_MIN_CLV:.2f} -- i.e. the close gave back most of the day's range, the
                distribution/exhaustion signature rather than confirmed demand. A row with no intraday range to
                compute CLV from is left alone (data gap, not a fail), same convention as everywhere else.
                The Confirmation column is a 9-point checklist tally (price above resistance, volume &gt;2x average,
                a close near the day's high, RSI 55-75, price above both the 20- and 50-day averages, relative
                strength vs NIFTY 50, sector confirmation, the NIFTY regime read above, and no excessive extension
                past resistance) -- shown as "(passed)/(available)", e.g. "8/8". Sector confirmation is always N/A
                in this build -- {html.escape(SECTOR_DATA_NOTE)} -- so every score currently maxes out at 8/8, not
                9/9; unavailable checks are excluded from the tally rather than counted as a failure. This is
                additional context on a row whose Confirmed/Watch/Filtered bucket is already decided --
                it doesn't itself gate anything, the way Risk:Reward does. (Its own "extension past resistance"
                checklist item is a simple pass/fail from the same underlying idea as the Extension column, kept for
                backward compatibility with the existing 9-point tally -- the Extension column is the detailed,
                scored read.)
                <b>Statistical bar (Confirmed Breakouts only):</b> two things are stacked here, not one. The shared
                backtest module's own bar is \u2265{MIN_SAMPLES_FOR_CONFIDENCE} past occurrences and
                \u2265{CONFIRM_HIT_RATE_THRESHOLD*100:.0f}% {PRIMARY_HORIZON}-day hit-rate; on top of that, this screener
                requires the row to also land in one of three TIERS, chosen so the hit-rate bar rises as the sample
                thins out rather than staying flat -- a thin sample has to be unambiguously strong to reach Confirmed,
                not just clear the same number a much deeper sample would:
                <br>&nbsp;&nbsp;\u2022 <b>Tier A:</b> \u226530 occurrences, \u226560% hit-rate
                <br>&nbsp;&nbsp;\u2022 <b>Tier B:</b> 15-29 occurrences, \u226560% hit-rate
                <br>&nbsp;&nbsp;\u2022 <b>Tier C:</b> 10-14 occurrences, \u226565% hit-rate
                <br>Below {CONFIRMATION_TIER_MIN_SAMPLES} occurrences ("Experimental") no tier qualifies, however high the
                raw hit-rate -- this is what stops a 100%-over-1-occurrence read from ever outranking a 79%-over-14 one.
                A row has to clear both the shared bar and one of the tiers above -- in practice, the tiered bar is the
                binding one. A signal that clears the shared bar but not a tier is pulled into Watch List with a note
                (⬇ "tiered sample/hit-rate bar", see Pattern column), never dropped. A Confirmed row's Backtest column
                shows which tier it actually cleared.
                <b>Confidence-adjusted hit rate:</b> the tiers above gate WHICH bucket a row lands in; ranking within a
                bucket (Risk-adj. EV, above) goes a step further and uses a Wilson-score 95%-confidence LOWER BOUND on
                hit-rate rather than the raw number, so a thin sample can't out-rank a deep, merely-good one even within
                the same tier. Any row below {CONFIRMATION_TIER_MIN_SAMPLES} occurrences shows both numbers in the
                Backtest column -- the raw hit-rate as fired, and the confidence floor actually used for ranking -- so
                nothing about the adjustment is hidden.
                <b>Fundamentals gate (Confirmed Breakouts only):</b> clearing the technical backtest bar is necessary but no
                longer sufficient for the Confirmed section -- the underlying business also has to clear a classic
                "quality stock" screen, pulled from yfinance at run time: Return on Equity \u2265{FUNDAMENTAL_MIN_ROE*100:.0f}%,
                Debt/Equity \u2264{FUNDAMENTAL_MAX_DEBT_TO_EQUITY:.0f}%, and non-negative latest earnings growth. This is only
                fetched for rows that already cleared the technical bar, and it never gates the Watch List, Filtered, Failed,
                or Near-Breakout sections -- only Confirmed. Same "nothing silently dropped" rule as everything else here: a
                row that clears technicals but fails an available fundamentals check is pulled into Watch List with a note
                (⬇, see Pattern column), not discarded. If yfinance has none of the three metrics for a symbol, the gate is
                skipped rather than treated as a fail -- a data gap doesn't disqualify a signal. The Quality column shows
                which specific metrics were checked and whether each passed.
                Rows shaded <span style="background:#E9F7ED;padding:0 3px;">light green</span> are symbols already
                in your own direct-equity list (constants.STOCKS) -- called out purely to jump out from the wider
                NIFTY 500 scan, not a separate signal.
                <b>{html.escape(BEST_EXECUTION_LABEL)}</b> rows -- shaded <span style="background:{BEST_EXECUTION_ROW_BG};padding:0 3px;">light
                sky blue</span> in the Confirmed table above, and taking priority over the light-green own-stock
                shading when both would apply -- are a stricter, rule-based filter layered on top of everything else
                on this page, meant to surface only the handful of setups worth a first look TODAY rather than the
                full Confirmed list: (1) already in the Confirmed Breakouts bucket, (2) Entry State is Fresh Breakout
                or Retest, i.e. there's an exact entry price today, (3) Setup Strength
                \u2265{BEST_EXECUTION_MIN_SETUP_STRENGTH} ("Very Strong"), (4) Risk:Reward
                \u2265{RISK_REWARD_PREFERRED_THRESHOLD:.1f}, (5) Failure Risk not flagged elevated on a reliable
                sample, and (6) price above Anchored VWAP from the breakout base wherever that figure was available
                (see the VWAP column). It runs no new backtest and adds no new data -- it's a same-fail-soft, all-of-the-above pass
                over the columns already on this page, so a row missing any one component simply doesn't qualify
                rather than being marked bad. Like Setup Strength, this is <b>not</b> a probability of success and
                <b>not</b> investment advice -- verify every number on the row yourself before acting on it.
                <b>Setup Strength</b> deliberately separates breakout QUALITY from EXECUTION quality rather than
                blending everything into one flat average -- it's four sub-scores, each answering a distinct
                question, blended at the weights below:
                <br>&nbsp;&nbsp;\u2022 <b>Breakout Quality (35%)</b> -- "Will the breakout continue?" Backtest Quality
                Score, today's Confirmation checklist, inverse Failure Risk, and (Confirmed-only) the Fundamentals
                gate -- everything about whether this pattern, on this stock, has a track record of playing out.
                <br>&nbsp;&nbsp;\u2022 <b>Trade Quality (25%)</b> -- "Is the entry attractive right now?" Entry State
                (Retest scores highest as the preferred entry, then Fresh Breakout, then Near Breakout), the
                Extension/chase-risk read, and price vs. Anchored VWAP from the breakout base -- purely about today's
                price action, independent of the pattern's own track record.
                <br>&nbsp;&nbsp;\u2022 <b>Risk:Reward (20%)</b> -- "Is the payoff worth the risk?" The R:R ratio
                itself, capped at 3:1.
                <br>&nbsp;&nbsp;\u2022 <b>Market Regime (20%)</b> -- "Is this the right environment for breakouts?"
                The same NIFTY-regime read shown above -- identical for every row in a given run, since it's a
                market-wide read, not a per-stock one.
                <br>Each sub-score is itself a fail-soft average of ITS OWN components (e.g. Trade Quality uses
                Entry State, Extension, and VWAP); if an entire sub-score is unavailable for a row (e.g. no Risk:Reward
                because there's no exact entry price), it's dropped and the remaining sub-scores' weights
                renormalized, same convention as everywhere else here. The Setup Strength cell shows all four
                sub-scores (BQ / TQ / R:R / Regime) alongside the headline number, and the "X/4 scores available"
                note shows how many of the four a given row is actually resting on. It is a summary, not a new
                calculation -- it backtests nothing on its own. It does <b>not</b> override or replace the
                Confirmed/Watch/Filtered/Failed bucket a row already landed in, and it is not a probability of the
                stock going up -- treat it as one more input to weigh alongside the columns it summarizes and your
                own judgment, not a stand-alone buy signal.
                Past pattern performance does not guarantee future results. Informational only -- not
                investment advice. Verify before acting.
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

    # Regime is computed AFTER histories are in hand -- its breadth check
    # (utils.market_regime._compute_breadth) reuses this exact {symbol: df}
    # dict rather than pulling the universe a second time.
    regime = compute_market_regime(histories)
    log.info(f"Breakout Screener: market regime {regime.label()} (score {regime.score}/5) -- {regime.checks}")

    # One-time NIFTY 50 index fetch, reused for every symbol's Relative
    # Strength read -- see fetch_nifty_index_history / compute_relative_strength.
    nifty_series = fetch_nifty_index_history()
    if nifty_series is None:
        log.warning("Breakout Screener: proceeding without a Relative Strength benchmark this run.")

    # One-time sector-index build, reused for every symbol's Sector
    # Confirmation read -- see build_sector_context / compute_sector_confirmation.
    # Empty until SECTOR_MAP is populated (see comment there).
    sector_context = build_sector_context(histories, SECTOR_MAP, nifty_series)
    if not sector_context:
        log.warning(f"Breakout Screener: Sector Confirmation unavailable this run -- {SECTOR_DATA_NOTE_LOCAL}.")

    confirmed, watch_list, filtered_low_rr, near_breakout_watch, skipped_count = scan_universe(
        histories, bhav_df, regime, nifty_series, sector_context
    )

    scan_stats = {
        "universe_size": len(symbols),
        "universe_is_live": universe_is_live,
        "history_count": len(histories),
        "skipped_count": skipped_count,
        "filtered_low_rr_count": len(filtered_low_rr),
        "bhav_is_live": bhav_is_live,
        "bhav_date": bhav_date.strftime("%d %b %Y") if bhav_date else None,
        "regime": regime,
    }
    log.info(
        f"Breakout Screener: {len(confirmed)} confirmed, {len(watch_list)} watch-list, "
        f"{len(filtered_low_rr)} filtered on Risk:Reward, {len(near_breakout_watch)} near-breakout signals "
        f"out of {len(histories)} symbols scanned ({skipped_count} skipped on data quality). "
        f"Market regime: {regime.label()}."
    )

    report_html = build_report_html(confirmed, watch_list, filtered_low_rr, near_breakout_watch, scan_stats)

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