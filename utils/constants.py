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