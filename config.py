import os
import json
import re

EMAIL_FROM = os.getenv("EMAIL_FROM")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_TO = os.getenv("EMAIL_TO")
EMAIL_CC = os.getenv("EMAIL_CC")

EMAIL_REGEX = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"


def parse_email_list(value):
    if not value:
        return []

    cleaned_value = value.replace("\n", "").replace("\r", "")
    emails = [email.strip() for email in cleaned_value.split(",") if email.strip()]
    return [email for email in emails if re.match(EMAIL_REGEX, email)]

STOCKS_JSON = os.getenv("STOCKS_JSON")
STOCKS_CSV = os.getenv("STOCKS")


NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

STOCKS = {}

if STOCKS_JSON:
    try:
        STOCKS = json.loads(STOCKS_JSON)
        if not isinstance(STOCKS, dict):
            raise ValueError("STOCKS_JSON must decode to a dict of name:ticker pairs")
    except Exception as exc:
        raise ValueError(f"Invalid STOCKS_JSON environment variable: {exc}")
elif STOCKS_CSV:
    try:
        pairs = [item.strip() for item in STOCKS_CSV.split(",") if item.strip()]
        for pair in pairs:
            if "=" not in pair:
                raise ValueError("Each STOCKS entry must be name=ticker")
            name, ticker = pair.split("=", 1)
            STOCKS[name.strip()] = ticker.strip()
    except Exception as exc:
        raise ValueError(f"Invalid STOCKS environment variable: {exc}")
