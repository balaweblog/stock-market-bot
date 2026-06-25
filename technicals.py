import ta

def calculate_technicals(df):
    df["ema20"] = df["close"].ewm(span=20).mean()
    df["ema200"] = df["close"].ewm(span=200).mean()

    rsi = ta.momentum.RSIIndicator(close=df["close"], window=14)
    df["rsi"] = rsi.rsi()

    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    df["vol_avg"] = df["volume"].rolling(20).mean()

    return df