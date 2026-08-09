# Static name -> ticker mapping for the stock reports.
#
# Previously this was supplied at runtime via the STOCKS_JSON environment
# variable (a JSON-encoded dict, e.g. exported as
#   export STOCKS_JSON='{"BEL":"BEL.NS", ...}'
# ). That's now hardcoded here instead, so the stock list lives in version
# control rather than in shell/CI environment config. To add, remove, or
# rename a stock, edit this dict directly.
STOCKS = {
    "BEL": "BEL.NS",
    "COALINDIA": "COALINDIA.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "SUNPHARMA": "SUNPHARMA.NS",
    "SBIN": "SBIN.NS",
    "ITC": "ITC.NS",
    "LT": "LT.NS",
    "TCS": "TCS.NS",
    "AAPL": "AAPL",
    "Goog": "GOOG",
    "AMZN": "AMZN",
    "QQQ": "QQQ",
    "KALYANJEWEL": "KALYANKJIL.NS",
    "CHENNPETRO": "CHENNPETRO.NS",
    "BAJFINANCE": "BAJFINANCE.NS"
}

# Stock watchlist for stock_market_advisor.py.
#
# Previously supplied at runtime via the STOCK_WATCHLIST_JSON environment
# variable (a JSON-encoded list of stock names). Now hardcoded here instead.
# To add, remove, or rename a watchlist entry, edit this list directly.
WATCHLIST = [
    "Sun Pharmaceutical Industries Ltd.",
    "State Bank of India",
    "ITC Ltd.",
    "Tata Consultancy Services Ltd.",
    "ICICI Bank Ltd.",
    "Larsen & Toubro Ltd.",
    "Coal India Ltd.",
    "Bharat Electronics Ltd.",
    "Apple Inc.",
    "Amazon.com, Inc.",
    "Alphabet Inc.",
    "Kalyan Jewellers India Ltd.",
    "Chennai Petroleum Corporation Ltd.",
    "Bajaj Finance Ltd."
]

# Mutual fund portfolio for mutual_fund_advisor.py.
#
# Previously supplied at runtime via the MF_PORTFOLIO_JSON environment
# variable (a JSON-encoded list of fund names). Now hardcoded here instead.
# To add, remove, or rename a fund, edit this list directly.
MF_PORTFOLIO = [
    "Mirae Asset Large & Midcap Fund - Direct Growth",
    "Parag Parikh Flexi Cap Fund - Direct Growth",
    "SBI Small Cap Fund - Direct Growth",
    "DSP Multi Asset Fund - Direct Growth",
    "ICICI Prudential Manufacturing Fund",
    "Nippon India Gold Savings Fund"
]

# -----------------------------------------------------------------------
# Monthly SIP / wealth-building portfolio for wealth_controller.py.
#
# Every recurring monthly contribution Bala makes -- PPF, mutual fund
# SIPs, NPS, direct equity SIPs, physical gold/silver, EPF, and US
# stocks -- in one place so wealth_controller.py can build the monthly
# report table and hand the whole picture to the LLM for a diversification
# / add-reduce-exit read. To add, remove, or re-amount an instrument, edit
# this list directly; "amount_inr" or "amount_usd" (not both) drives the
# per-instrument line -- USD entries are converted to INR for the total
# using WEALTH_USD_TO_INR_ESTIMATE below.
#
# category is used only for grouping/diversification context shown to the
# LLM and in the report table -- it doesn't drive any calculation.
# -----------------------------------------------------------------------
SIP_PORTFOLIO = [
    {"instrument": "PPF", "category": "Government / Debt", "amount_inr": 5000},
    {"instrument": "Mirae Asset Large & Midcap Fund", "category": "Mutual Fund - Large & Midcap", "amount_inr": 7000},
    {"instrument": "Parag Parikh Flexi Cap Fund", "category": "Mutual Fund - Flexi Cap", "amount_inr": 7000},
    {"instrument": "DSP Multi Asset Fund", "category": "Mutual Fund - Multi Asset", "amount_inr": 7000},
    {"instrument": "SBI Small Cap Fund", "category": "Mutual Fund - Small Cap", "amount_inr": 7000},
    {
        "instrument": "NPS (HDFC Pension Fund Mgmt)",
        "category": "Retirement - NPS Tier I",
        "amount_inr": 20565,
        "notes": "Scheme E 50% / Scheme C 25% / Scheme G 25%",
    },
    {"instrument": "ITC", "category": "Direct Equity", "amount_inr": 8000},
    {"instrument": "State Bank of India", "category": "Direct Equity", "amount_inr": 9000},
    {"instrument": "Larsen & Toubro", "category": "Direct Equity", "amount_inr": 13000},
    {"instrument": "ICICI Bank", "category": "Direct Equity", "amount_inr": 9000},
    {"instrument": "Coal India", "category": "Direct Equity", "amount_inr": 7000},
    {"instrument": "Sun Pharma", "category": "Direct Equity", "amount_inr": 9000},
    {"instrument": "Bharat Electronics (BEL)", "category": "Direct Equity", "amount_inr": 5000},
    {"instrument": "Chennai Petroleum", "category": "Direct Equity", "amount_inr": 8000},
    {"instrument": "Bajaj Finance", "category": "Direct Equity", "amount_inr": 8000},
    {"instrument": "Kalyan Jewellers", "category": "Direct Equity", "amount_inr": 7000},
    {"instrument": "Gold (Tanishq)", "category": "Physical Gold", "amount_inr": 10000},
    {"instrument": "Silver (Bhima Jewellers)", "category": "Physical Silver", "amount_inr": 5000},
    {"instrument": "ICICI Prudential Manufacturing Fund", "category": "Mutual Fund - Thematic", "amount_inr": 3000},
    {"instrument": "Nippon India Gold Savings Fund", "category": "Mutual Fund - Gold", "amount_inr": 1000},
    {"instrument": "EPF", "category": "Retirement - EPF", "amount_inr": 33000},
    {"instrument": "US Stocks", "category": "International Equity", "amount_usd": 100},
]

# Fixed, approximate USD->INR rate used ONLY to fold the US Stocks SIP line
# into the portfolio's total-monthly-outflow figure shown in the report.
# It is a rough constant, not a live rate -- update occasionally, or wire
# up a live FX fetch later if the estimate drifts too far.
WEALTH_USD_TO_INR_ESTIMATE = 88