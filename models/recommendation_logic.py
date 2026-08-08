def choose_stock_entry(signal, total_score, latest, market_context, entry_context):
    """Choose a live, price-aware entry style for stock recommendations."""
    trend_text = str(market_context.get("trend", "")).lower()
    bullish = any(token in trend_text for token in ["up", "positive", "bull", "strong", "rise", "recovery"])
    bearish = any(token in trend_text for token in ["down", "negative", "bear", "weak", "decline"])

    if signal.lower() in {"sell"} or total_score < 55:
        return "patient_entry"

    if bearish:
        return "patient_entry"

    adx = latest.get("adx")
    rsi = latest.get("rsi")
    volume_ratio = entry_context.get("volume_vs_avg_pct", 0) or 0
    rr = entry_context.get("risk_reward_ratio", 0) or 0
    price_vs_ema20 = entry_context.get("price_vs_ema20_pct", 0) or 0
    price_vs_ema50 = entry_context.get("price_vs_ema50_pct", 0) or 0

    scores = {
        "patient_entry": 0,
        "optimal_entry": 0,
        "aggressive_entry": 0,
    }

    scores["patient_entry"] += max(0, 70 - total_score) * 0.25
    scores["optimal_entry"] += min(25, total_score * 0.25)
    scores["aggressive_entry"] += min(25, total_score * 0.28)

    if bullish:
        scores["optimal_entry"] += 8
        scores["aggressive_entry"] += 10
    else:
        scores["patient_entry"] += 12

    if adx is not None:
        if adx >= 25:
            scores["aggressive_entry"] += 14
            scores["optimal_entry"] += 8
        elif adx >= 20:
            scores["optimal_entry"] += 10
        else:
            scores["patient_entry"] += 8

    if rsi is not None:
        if 45 <= rsi <= 70:
            scores["optimal_entry"] += 8
            scores["aggressive_entry"] += 4
        elif rsi > 70:
            scores["patient_entry"] += 8
        else:
            scores["patient_entry"] += 4

    if volume_ratio >= 12:
        scores["aggressive_entry"] += 12
        scores["optimal_entry"] += 6
    elif volume_ratio >= 6:
        scores["optimal_entry"] += 8
    else:
        scores["patient_entry"] += 8

    if rr >= 1.8:
        scores["aggressive_entry"] += 12
        scores["optimal_entry"] += 6
    elif rr >= 1.2:
        scores["optimal_entry"] += 8
    else:
        scores["patient_entry"] += 10

    if price_vs_ema20 >= 0 and price_vs_ema50 >= 0:
        scores["aggressive_entry"] += 8
        scores["optimal_entry"] += 4
    elif price_vs_ema20 >= -3 and price_vs_ema50 >= -4:
        scores["optimal_entry"] += 8
    else:
        scores["patient_entry"] += 10

    if price_vs_ema20 < -3 or price_vs_ema50 < -4:
        scores["patient_entry"] += 12
        scores["optimal_entry"] -= 2
    elif price_vs_ema20 > 2 and price_vs_ema50 > 2 and total_score >= 80 and bullish:
        scores["aggressive_entry"] += 10
        scores["optimal_entry"] -= 2
    elif abs(price_vs_ema20) <= 2 and abs(price_vs_ema50) <= 4:
        scores["optimal_entry"] += 8

    if total_score < 80 or (rsi is not None and rsi > 75) or volume_ratio < 8 or rr < 1.5:
        scores["aggressive_entry"] -= 15

    if total_score < 65 or price_vs_ema20 < -5 or price_vs_ema50 < -6:
        scores["optimal_entry"] -= 8

    return max(scores, key=scores.get)


def derive_commodity_buy_levels(current_price, history):
    """Create dynamic buy levels for gold/silver based on the live current price."""
    if not history or len(history) < 3:
        return {
            "patient_entry": round(current_price * 0.98, 2),
            "optimal_entry": round(current_price, 2),
            "aggressive_entry": round(current_price * 1.02, 2),
            "recommended_entry": "optimal_entry",
            "recommended_entry_label": "Optimal Entry",
            "recommended_buy_level": round(current_price, 2),
        }

    recent_changes = [row["change"] for row in history[-3:]]
    recent_prices = [row["price"] for row in history[-3:]]
    latest_change = recent_changes[-1]
    prev_change = recent_changes[-2]
    older_change = recent_changes[-3]
    avg_recent = sum(recent_prices) / len(recent_prices) if recent_prices else current_price
    price_vs_recent_avg_pct = ((current_price - avg_recent) / avg_recent) * 100 if avg_recent else 0

    score = 0
    if latest_change <= -1.5:
        score += 2
    if latest_change <= -2.5:
        score += 2
    if prev_change <= -1.0:
        score += 1
    if older_change <= -1.0:
        score += 1
    if latest_change < prev_change:
        score += 1
    if latest_change < 0:
        score += 1

    if price_vs_recent_avg_pct < -2.5:
        score += 2
    elif price_vs_recent_avg_pct > 2.5:
        score -= 1

    if score >= 6:
        recommended = "patient_entry"
        patient_discount = 0.04
        optimal_discount = 0.02
        aggressive_premium = 0.015
    elif score >= 4:
        recommended = "optimal_entry"
        patient_discount = 0.03
        optimal_discount = 0.01
        aggressive_premium = 0.012
    else:
        recommended = "aggressive_entry"
        patient_discount = 0.02
        optimal_discount = 0.005
        aggressive_premium = 0.008

    if price_vs_recent_avg_pct < -2.5:
        patient_discount += 0.01
        optimal_discount += 0.005
    elif price_vs_recent_avg_pct > 2.5:
        patient_discount -= 0.005
        optimal_discount -= 0.003
        aggressive_premium -= 0.003

    buy_levels = {
        "patient_entry": round(current_price * (1 - patient_discount), 2),
        "optimal_entry": round(current_price * (1 - optimal_discount), 2),
        "aggressive_entry": round(current_price * (1 + aggressive_premium), 2),
    }
    buy_levels["recommended_entry"] = recommended
    buy_levels["recommended_entry_label"] = recommended.replace("_", " ").title()
    buy_levels["recommended_buy_level"] = buy_levels[recommended]
    return buy_levels


def resolve_recommended_commodity_entry(buy_levels, risk_reward_by_entry, min_acceptable_rr=1.5):
    """Second pass over derive_commodity_buy_levels()'s recommendation, run
    once the stop-loss/target are known (build_trade_plan derives those from
    the buy_levels this function receives, so it can't happen any earlier).

    derive_commodity_buy_levels() above picks a tier from recent price
    momentum alone, with no visibility into risk/reward -- and because the
    stop and target it's blind to are shared across all three tiers,
    a higher (aggressive) entry structurally has less room to the target and
    more room to the stop than a lower (patient) one. That combination can
    -- and in production did -- produce a recommended tier with a materially
    worse risk/reward than a more conservative tier sitting right next to it
    on the same report. This overrides the recommendation in that case,
    mirroring how choose_stock_entry factors risk_reward_ratio into its own
    scoring for stocks instead of choosing on momentum alone.
    """
    recommended = buy_levels.get("recommended_entry")
    recommended_rr = risk_reward_by_entry.get(recommended)

    if recommended_rr is not None and recommended_rr >= min_acceptable_rr:
        return buy_levels

    # Prefer the most conservative tier (patient over optimal over
    # aggressive) among any tier that actually clears the bar, rather than
    # just the single best RR -- consistent with derive_commodity_buy_levels'
    # own tie-breaking toward caution.
    tier_order = ["patient_entry", "optimal_entry", "aggressive_entry"]
    better_tiers = [
        tier for tier in tier_order
        if tier != recommended
        and risk_reward_by_entry.get(tier) is not None
        and risk_reward_by_entry[tier] >= min_acceptable_rr
    ]

    if not better_tiers:
        # Nothing clears the bar -- fall back to whichever tier has the
        # best available RR rather than silently keeping the worst one.
        candidates = {
            tier: rr for tier, rr in risk_reward_by_entry.items() if rr is not None
        }
        if not candidates or max(candidates.values()) <= (recommended_rr or float("-inf")):
            return buy_levels
        better_tiers = [max(candidates, key=candidates.get)]

    new_recommended = better_tiers[0]
    buy_levels["recommended_entry"] = new_recommended
    buy_levels["recommended_entry_label"] = new_recommended.replace("_", " ").title()
    buy_levels["recommended_buy_level"] = buy_levels[new_recommended]
    return buy_levels