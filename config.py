import os
import json
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

STOCKS_JSON = os.getenv("STOCKS_JSON")
STOCKS_CSV = os.getenv("STOCKS")


NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

STOCKS = {}


def _parse_stocks_csv(csv_value):
    parsed = {}
    pairs = [item.strip() for item in csv_value.split(",") if item.strip()]
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Each STOCKS entry must be name=ticker, got: '{pair}'")
        name, ticker = pair.split("=", 1)
        parsed[name.strip()] = ticker.strip()
    return parsed


# NOTE: these are merged (not if/elif). Previously STOCKS_JSON, when set,
# silently took over and STOCKS_CSV was ignored entirely -- if a user kept
# e.g. Indian tickers only in STOCKS (CSV) and US tickers in STOCKS_JSON,
# the Indian half would vanish from every report with no error or warning.
# Merging means both env vars can be used together; on a name collision the
# STOCKS_CSV entry wins since it's applied last.
if STOCKS_JSON:
    try:
        parsed_json = json.loads(STOCKS_JSON)
        if not isinstance(parsed_json, dict):
            raise ValueError("STOCKS_JSON must decode to a dict of name:ticker pairs")
        STOCKS.update(parsed_json)
    except Exception as exc:
        raise ValueError(f"Invalid STOCKS_JSON environment variable: {exc}")

if STOCKS_CSV:
    try:
        STOCKS.update(_parse_stocks_csv(STOCKS_CSV))
    except Exception as exc:
        raise ValueError(f"Invalid STOCKS environment variable: {exc}")