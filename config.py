import os
import re

EMAIL_FROM = os.getenv("EMAIL_FROM")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_TO = os.getenv("EMAIL_TO")
EMAIL_CC = os.getenv("EMAIL_CC")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

EMAIL_REGEX = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"


def parse_email_list(value):
    if not value:
        return []

    cleaned_value = value.replace("\n", "").replace("\r", "")
    emails = [email.strip() for email in cleaned_value.split(",") if email.strip()]
    return [email for email in emails if re.match(EMAIL_REGEX, email)]

STOCKS_CSV = os.getenv("STOCKS")


NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Base stock list now lives in constants.py instead of the STOCKS_JSON
# environment variable. STOCKS_CSV (below) can still be used to add/override
# entries on top of this base list without touching constants.py.
from constants import STOCKS as _CONSTANTS_STOCKS

STOCKS = dict(_CONSTANTS_STOCKS)


def _parse_stocks_csv(csv_value):
    parsed = {}
    pairs = [item.strip() for item in csv_value.split(",") if item.strip()]
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Each STOCKS entry must be name=ticker, got: '{pair}'")
        name, ticker = pair.split("=", 1)
        parsed[name.strip()] = ticker.strip()
    return parsed


# STOCKS_CSV, if set, merges on top of the constants.py base list -- lets
# you add or override individual name=ticker entries without editing
# constants.py. On a name collision, the STOCKS_CSV entry wins since it's
# applied last.
if STOCKS_CSV:
    try:
        STOCKS.update(_parse_stocks_csv(STOCKS_CSV))
    except Exception as exc:
        raise ValueError(f"Invalid STOCKS environment variable: {exc}")