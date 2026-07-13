import unittest
from datetime import date

from stock_fetcher import build_upcoming_event_summary, format_event_date


class StockFetcherTests(unittest.TestCase):
    def test_build_upcoming_event_summary_uses_available_fields(self):
        payload = {
            "dividendDate": "2026-07-20",
            "dividendPayDate": "2026-07-24",
            "earningsDate": "2026-07-22",
        }

        result = build_upcoming_event_summary(payload)

        self.assertEqual(result["dividend_record_date"], "20 Jul 2026")
        self.assertEqual(result["dividend_deposit_date"], "24 Jul 2026")
        self.assertEqual(result["results_announcement_date"], "22 Jul 2026")

    def test_format_event_date_handles_missing_values(self):
        self.assertEqual(format_event_date(None), "Not available")
        self.assertEqual(format_event_date(date(2026, 7, 20)), "20 Jul 2026")


if __name__ == "__main__":
    unittest.main()
