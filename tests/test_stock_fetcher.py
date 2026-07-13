import unittest
from datetime import date, timedelta

from stock_fetcher import build_upcoming_event_summary, format_event_date


class StockFetcherTests(unittest.TestCase):
    def test_build_upcoming_event_summary_uses_only_dates_within_next_60_days(self):
        today = date.today()
        payload = {
            "dividendDate": (today + timedelta(days=20)).strftime("%Y-%m-%d"),
            "earningsDate": (today + timedelta(days=80)).strftime("%Y-%m-%d"),
        }

        result = build_upcoming_event_summary(payload)

        self.assertEqual(result["dividend_record_date"], (today + timedelta(days=20)).strftime("%d %b %Y"))
        self.assertEqual(result["results_announcement_date"], "NA")

    def test_format_event_date_handles_missing_values(self):
        self.assertEqual(format_event_date(None), "Not available")
        self.assertEqual(format_event_date(date(2026, 7, 20)), "20 Jul 2026")


if __name__ == "__main__":
    unittest.main()
