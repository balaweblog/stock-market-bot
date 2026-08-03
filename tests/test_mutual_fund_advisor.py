import unittest

from mutual_fund_advisor import _format_monthly_return_display, render_fund_cards


class MutualFundAdvisorTests(unittest.TestCase):
    def test_format_monthly_return_display_supports_common_field_names(self):
        self.assertEqual(_format_monthly_return_display({"monthly_return": "3.7"}), "3.7%")
        self.assertEqual(_format_monthly_return_display({"monthly_return_percent": "-1.25"}), "-1.25%")
        self.assertEqual(_format_monthly_return_display({"return_pct": "2.0%"}), "2.0%")
        self.assertEqual(_format_monthly_return_display({"monthly_return_pct": "null"}), "n/a")
        self.assertEqual(_format_monthly_return_display({}), "n/a")

    def test_render_fund_cards_wraps_long_portfolio_changes(self):
        html = render_fund_cards([
            {
                "fund_name": "Test Fund",
                "news_timeline": [],
                "portfolio_changes": "No material disclosed changes found this window; no AMC press release, factsheet update, or reputable news item dated between 02 July 2026 and 01 August 2026 could be independently verified.",
                "recommendation": "Hold",
                "assessment": "Neutral",
                "monthly_return_pct": "2.4",
                "benchmark_comparison": "In line",
                "short_term_outlook": "Stable",
                "long_term_outlook": "Stable",
            }
        ])

        self.assertIn("No material disclosed changes found this window", html)
        self.assertIn("white-space:pre-wrap", html)
        self.assertIn("word-break:break-word", html)

    def test_render_fund_cards_includes_snapshot_metrics(self):
        html = render_fund_cards([
            {
                "fund_name": "Test Fund",
                "news_timeline": [],
                "portfolio_changes": "No changes",
                "recommendation": "Continue SIP",
                "assessment": "Positive",
                "monthly_return_pct": "2.4",
                "benchmark_comparison": "Ahead of benchmark",
                "fund_category": "Large & Mid Cap",
                "benchmark": "Nifty LargeMidcap 250",
                "nav_latest": "177.24",
                "aum_cr": "44048",
                "expense_ratio_pct": "0.84",
                "one_year_return_pct": "2.77",
                "three_year_return_pct": "13.82",
                "risk_level": "Medium",
                "decision_note": "Good fit for long-term SIPs with moderate risk tolerance.",
                "short_term_outlook": "Stable",
                "long_term_outlook": "Stable",
            }
        ])

        self.assertIn("Snapshot", html)
        self.assertIn("Large &amp; Mid Cap", html)
        self.assertIn("Expense", html)
        self.assertIn("Decision note", html)


if __name__ == "__main__":
    unittest.main()
