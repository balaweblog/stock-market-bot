import unittest

from optionstrategy import _normalize_weekly_recommendations, build_prompt


class OptionStrategyTests(unittest.TestCase):
    def test_normalize_weekly_recommendations_expands_weekly_view(self):
        horizon = {
            "horizon": "Weekly",
            "bias": "Bullish",
            "strategy_name": "Bull Call Spread",
            "legs": "Buy 24000 CE, Sell 24200 CE",
        }

        recommendations = _normalize_weekly_recommendations(horizon)

        self.assertGreaterEqual(len(recommendations), 2)
        self.assertEqual(recommendations[0]["label"], "Primary")
        self.assertTrue(any("Alternative" in rec["label"] for rec in recommendations))

    def test_build_prompt_only_targets_weekly_horizon(self):
        prompt = build_prompt(live_data={})

        self.assertIn("Nifty Weekly", prompt)
        self.assertNotIn("Nifty Monthly", prompt)


if __name__ == "__main__":
    unittest.main()
