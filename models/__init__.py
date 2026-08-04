"""
models package: Domain Models, Financial Metrics, Scoring Engines & Backtesting
"""
from .fundamentals import score_fundamentals
from .advanced_fundamentals import fetch_advanced_fundamentals, score_advanced_fundamentals
from .scorer import final_score, decision
from .recommendation_logic import choose_stock_entry, derive_commodity_buy_levels
from .position_sizing import calculate_position_size, apply_risk_management
from .support_resistance import compute_pivot_levels, compute_swing_zones, nearest_levels, build_support_resistance_html
from .market_context import get_resilient_session, fetch_index_context, classify_market, build_market_context
from .swing_trade_scoring import compute_composite_score, rank_by_composite, composite_score_html
from .swing_trade_risk import apply_sector_concentration_cap, compute_volatility_adjusted_plan
from .swing_trade_regime import check_market_regime, regime_note_html
from .swing_trade_universe import tickers_for_sectors, all_tickers, ticker_count_by_sector
from .swing_trade_outcomes import log_recommendation, update_and_summarize_outcomes
from .swing_trade_backtest import run_backtest
from .track_record import update_track_record, build_track_record_html
