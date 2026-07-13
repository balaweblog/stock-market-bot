import unittest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from commodity_tracker import CommodityTracker
from position_sizing import apply_risk_management
from recommendation_logic import derive_commodity_buy_levels
from main import build_fundamentals_html, build_quick_summary, calculate_52_week_range, calculate_risk_meter


class RecommendationLogicTests(unittest.TestCase):
    def test_build_quick_summary_collects_buy_watch_caution_and_event_items(self):
        dividend_record_date = (datetime.now(ZoneInfo("Asia/Kolkata")).date() + timedelta(days=2)).strftime("%d %b %Y")
        rows = [
            {
                "stock_name": "SBIN",
                "signal": "BUY / HOLD",
                "recommended_buy_level": 1030,
                "current_price": 1028,
                "ema20": 1045,
                "upcoming_events": {
                    "has_event": False,
                },
            },
            {
                "stock_name": "ICICI Bank",
                "signal": "HOLD",
                "recommended_buy_level": 1200,
                "current_price": 1210,
                "ema20": 1215,
                "upcoming_events": {
                    "has_event": True,
                    "results_announcement_date": "18 Jul 2026",
                    "next_upcoming_event_label": "Results",
                    "next_upcoming_event_date": "18 Jul 2026",
                },
            },
            {
                "stock_name": "ITC",
                "signal": "SELL",
                "recommended_buy_level": 400,
                "current_price": 390,
                "ema20": 395,
                "upcoming_events": {
                    "has_event": False,
                },
            },
            {
                "stock_name": "TCS",
                "signal": "HOLD",
                "recommended_buy_level": 3500,
                "current_price": 3498,
                "ema20": 3502,
                "upcoming_events": {
                    "has_event": True,
                    "dividend_record_date": dividend_record_date,
                    "results_announcement_date": "NA",
                    "next_upcoming_event_label": "Dividend Record",
                    "next_upcoming_event_date": dividend_record_date,
                },
            },
        ]

        summary = build_quick_summary(rows)

        self.assertIn("✅ Buy: SBIN below ₹1030", summary)
        self.assertIn("✅ Watch: ICICI Bank results on 18 Jul", summary)
        self.assertIn("⚠ ITC below EMA20—avoid adding", summary)
        self.assertIn("📅 TCS dividend record in 2 days", summary)

    def test_stock_buy_levels_adjust_for_live_price_distance(self):
        entry_context = {
            "price_vs_ema20_pct": -6,
            "price_vs_ema50_pct": -8,
        }
        risk_data = apply_risk_management(
            "BUY / HOLD",
            72,
            cash=100000,
            price=100,
            entry_context=entry_context,
        )

        self.assertLess(risk_data["buy_levels"]["patient_entry"], 96)
        self.assertLess(risk_data["buy_levels"]["optimal_entry"], 100)
        self.assertGreater(risk_data["buy_levels"]["aggressive_entry"], 100)

    def test_commodity_buy_levels_follow_current_price(self):
        history = [
            {"price": 100, "change": -2.0},
            {"price": 98, "change": -1.5},
            {"price": 96, "change": -2.2},
        ]
        levels = derive_commodity_buy_levels(94.5, history)

        self.assertLess(levels["patient_entry"], 94.5)
        self.assertLess(levels["optimal_entry"], 94.5)
        self.assertGreater(levels["aggressive_entry"], 94.5)

    def test_trade_plan_builds_actionable_levels(self):
        tracker = CommodityTracker()
        levels = {
            "patient_entry": 95.0,
            "optimal_entry": 97.0,
            "aggressive_entry": 100.0,
        }
        history = [
            {"price": 96.0, "change": -0.5},
            {"price": 97.0, "change": 0.2},
            {"price": 98.0, "change": 1.1},
        ]

        plan = tracker.build_trade_plan(98.0, history, levels)

        self.assertEqual(plan["entry_low"], 95.0)
        self.assertEqual(plan["entry_high"], 100.0)
        self.assertGreater(plan["stop_loss"], 0)
        self.assertGreater(plan["target"], 98.0)
        self.assertIn(plan["bias"], {"Bullish", "Bearish", "Neutral"})

    def test_risk_meter_uses_atr_volatility_beta_and_adx(self):
        low_close = [100 * (1.001 ** index) for index in range(30)]
        low_df = pd.DataFrame({"close": low_close})
        low_latest = low_df.iloc[-1].copy()
        low_latest["atr"] = 1.0
        low_latest["adx"] = 32.0

        high_close = [100, 112, 96, 118, 92, 121, 89, 125, 87, 130, 85, 128, 83, 132, 81, 135, 79, 138, 77, 140, 75, 142, 73, 145, 71, 148, 69, 150, 67, 152]
        high_df = pd.DataFrame({"close": high_close})
        high_latest = high_df.iloc[-1].copy()
        high_latest["atr"] = 7.0
        high_latest["adx"] = 14.0

        low_meter = calculate_risk_meter(low_df, low_latest, beta=0.8)
        high_meter = calculate_risk_meter(high_df, high_latest, beta=1.7)

        self.assertEqual(low_meter["label"], "Low")
        self.assertEqual(high_meter["label"], "High")
        self.assertIn("🟢", low_meter["emoji"])
        self.assertIn("🔴", high_meter["emoji"])

    def test_52_week_range_calculates_distance_from_high_and_low(self):
        df = pd.DataFrame({
            "high": [1200, 1550, 1500],
            "low": [1000, 1080, 1100],
            "close": [1200, 1500, 1411],
        })
        latest = df.iloc[-1]

        range_data = calculate_52_week_range(df, latest)

        self.assertEqual(range_data["high_52w"], 1550)
        self.assertEqual(range_data["low_52w"], 1000)
        self.assertEqual(range_data["high_distance_text"], "↓ 9.0% below high")
        self.assertEqual(range_data["low_distance_text"], "↑ 41.1% above low")

    def test_fundamentals_html_shows_key_metrics(self):
        html = build_fundamentals_html({
            "pe": 21.84,
            "pb": 3.25,
            "dividendYield": 0.023,
            "roe": 0.185,
            "debtToEquity": 42.67,
        }, 70)

        self.assertIn("Score 70", html)
        self.assertIn("PE", html)
        self.assertIn("21.8", html)
        self.assertIn("PB", html)
        self.assertIn("3.2", html)
        self.assertIn("Dividend Yield", html)
        self.assertIn("2.3%", html)
        self.assertIn("ROE", html)
        self.assertIn("18.5%", html)
        self.assertIn("Debt/Equity", html)
        self.assertIn("42.7", html)


if __name__ == "__main__":
    unittest.main()
