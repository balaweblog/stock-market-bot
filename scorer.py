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