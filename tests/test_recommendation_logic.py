import unittest

from commodity_tracker import CommodityTracker
from position_sizing import apply_risk_management
from recommendation_logic import derive_commodity_buy_levels
from main import build_quick_summary


class RecommendationLogicTests(unittest.TestCase):
    def test_build_quick_summary_collects_buy_watch_caution_and_event_items(self):
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
                    "dividend_record_date": "15 Jul 2026",
                    "results_announcement_date": "NA",
                    "next_upcoming_event_label": "Dividend Record",
                    "next_upcoming_event_date": "15 Jul 2026",
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


if __name__ == "__main__":
    unittest.main()
