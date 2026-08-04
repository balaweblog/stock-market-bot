import unittest

from controllers.nifty_stock_controller import _format_weekly_return_display, render_stock_cards


class StockMarketAdvisorTests(unittest.TestCase):
    def test_format_weekly_return_display_supports_common_field_names(self):
        self.assertEqual(_format_weekly_return_display({"weekly_return": "3.7"}), "3.7%")
        self.assertEqual(_format_weekly_return_display({"weekly_return_percent": "-1.25"}), "-1.25%")
        self.assertEqual(_format_weekly_return_display({"return_pct": "2.0%"}), "2.0%")
        self.assertEqual(_format_weekly_return_display({"weekly_return_pct": "null"}), "n/a")
        self.assertEqual(_format_weekly_return_display({}), "n/a")

    def test_render_stock_cards_uses_snapshot_return_value(self):
        html = render_stock_cards([
            {
                "stock_name": "Test Stock",
                "news_timeline": [],
                "corporate_actions": "No changes",
                "weekly_return_pct": "2.4",
                "analyst_view": "In line",
                "assessment": "Positive",
                "recommendation": "Buy",
            }
        ])

        self.assertIn("2.4%", html)

    def test_render_stock_cards_shows_decision_signal(self):
        html = render_stock_cards([
            {
                "stock_name": "Test Stock",
                "news_timeline": [],
                "corporate_actions": "No changes",
                "weekly_return_pct": "2.4",
                "analyst_view": "In line",
                "assessment": "Positive",
                "recommendation": "Buy",
            }
        ])

        self.assertIn("Decision Signal", html)
        self.assertIn("Buy", html)

    def test_render_stock_cards_includes_action_now_and_quality_status(self):
        html = render_stock_cards([
            {
                "stock_name": "Test Stock",
                "news_timeline": [],
                "corporate_actions": "No changes",
                "weekly_return_pct": "2.4",
                "analyst_view": "In line",
                "assessment": "Positive",
                "recommendation": "Buy",
                "decision_note": "Strong setup with clear catalyst.",
            }
        ])

        self.assertIn("Action now", html)
        self.assertIn("Verified", html)


if __name__ == "__main__":
    unittest.main()
