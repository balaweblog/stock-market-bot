import unittest

from position_sizing import apply_risk_management
from recommendation_logic import derive_commodity_buy_levels


class RecommendationLogicTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
