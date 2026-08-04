import unittest

from controllers.swing_controller import _choose_analysis_html


class SwingTradeAdvisorTests(unittest.TestCase):
    def test_no_qualifying_trade_uses_no_pick_message_when_candidates_fail(self):
        html = _choose_analysis_html(
            qualifying=[],
            candidates=[{"name": "Example", "ticker": "EXAMPLE"}],
            rejected=[{"name": "Example", "ticker": "EXAMPLE"}],
            require_qualifying_stock=True,
        )

        self.assertIn("No qualifying trade found for this run", html)


if __name__ == "__main__":
    unittest.main()
