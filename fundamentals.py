def score_fundamentals(f):
    score = 0

    pe = f.get("pe")
    roe = f.get("roe")
    debt = f.get("debtToEquity")

    if pe and 0 < pe < 30:
        score += 30

    if roe and roe > 0.15:
        score += 40

    if debt and 0 <= debt < 150:
        score += 30

    return score