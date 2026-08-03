def score_fundamentals(f):
    score = 0

    pe = f.get("pe")
    roe = f.get("roe")
    debt = f.get("debtToEquity")

    if pe and 0 < pe < 30:
        score += 30

    if roe and roe > 0.15:
        score += 40

    # BUG FIX: `if debt and ...` evaluates bool(0) as False, meaning a
    # completely debt-free company (debt=0) incorrectly scored 0 points here.
    # Use explicit `is not None` so zero debt correctly awards full points.
    if debt is not None and 0 <= debt < 150:
        score += 30

    return score