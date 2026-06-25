def technical_score(df):
    latest = df.iloc[-1]
    score = 0

    if latest["close"] > latest["ema200"]:
        score += 25

    if latest["close"] > latest["ema20"]:
        score += 25

    if latest["macd"] > latest["macd_signal"]:
        score += 20

    rsi = latest["rsi"]

    if 45 <= rsi <= 70:
        score += 20

    if latest["volume"] > latest["vol_avg"]:
        score += 10

    return score


def final_score(technical, fundamentals, sentiment):
    score = (
        technical * 0.4 +
        fundamentals * 0.35 +
        sentiment * 0.25
    )

    return round(score, 2)


def decision(score):
    if score >= 80:
        return "STRONG BUY"
    elif score >= 65:
        return "BUY / HOLD"
    elif score >= 50:
        return "HOLD"
    else:
        return "SELL"