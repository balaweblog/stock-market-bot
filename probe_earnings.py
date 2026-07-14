import yfinance as yf, pandas as pd

t = yf.Ticker("INFY.NS")

print("=== earnings_dates ===")
try:
    ed = t.earnings_dates
    print(type(ed))
    if ed is not None:
        print(ed.head(8))
        print("Columns:", list(ed.columns))
    else:
        print("None")
except Exception as e:
    print("earnings_dates error:", e)

print("\n=== quarterly_earnings ===")
try:
    qe = t.quarterly_earnings
    print(type(qe))
    print(qe if qe is not None else "None")
except Exception as e:
    print("quarterly_earnings error:", e)

print("\n=== earnings_history ===")
try:
    eh = t.earnings_history
    print(type(eh))
    if eh is not None:
        print(eh)
        if hasattr(eh, "columns"):
            print("Columns:", list(eh.columns))
    else:
        print("None")
except Exception as e:
    print("earnings_history error:", e)
