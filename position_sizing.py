def calculate_position_size(cash, confidence, risk_per_trade=0.01, stop_loss_pct=None, max_position_pct=1.0):
    """
    Returns the dollar notional to allocate to a trade, sized so that a move
    to the stop-loss costs exactly risk_per_trade% of cash, regardless of
    signal confidence.

    Confidence still gates whether a trade is sized at all (<=0 -> 0) and
    still selects the fallback stop_loss_pct tier below when the caller
    doesn't supply the real stop distance -- but it no longer scales the
    dollar amount risked. (Previously the result was also multiplied by
    confidence, so the dollar risk at the stop was risk_per_trade *
    confidence -- e.g. 0.65% of cash at confidence 0.65, not the full 1% --
    which meant risk_per_trade didn't actually describe the risk taken.)

    max_position_pct caps the returned notional at that fraction of cash
    (default 100%) -- without this cap, a tight stop_loss_pct (e.g. a
    caller passing an ATR-based 0.5% stop instead of relying on the
    fallback tiers below) can produce a notional bigger than the account,
    implying leverage this risk_per_trade-based sizer was never meant to imply.
    """
    if confidence <= 0 or cash <= 0:
        return 0

    # Guard against zero/negative stop distances -- not just "unset". A
    # stop_loss_pct <= 0 would previously divide by zero (or flip the sign
    # and return a bogus negative-turned-positive size via the max(0, ...)
    # floor), instead of falling back to the confidence-tiered default.
    if stop_loss_pct is None or stop_loss_pct <= 0:
        # Fallback tiers, only used when the caller doesn't supply the actual
        # stop distance. Kept in sync with the thresholds in
        # apply_risk_management()'s target/stop block (0.8 / 0.65) -- these
        # previously used 0.85 / 0.65, a different cutoff, so a confidence in
        # [0.80, 0.85) sized the position off a 5% stop while the stop-loss
        # actually shown to the user was 4% away. Whenever possible, prefer
        # passing stop_loss_pct explicitly instead of relying on this fallback.
        if confidence >= 0.8:
            stop_loss_pct = 0.04
        elif confidence >= 0.65:
            stop_loss_pct = 0.05
        else:
            stop_loss_pct = 0.06

    risk_amount = cash * risk_per_trade
    position_size = risk_amount / stop_loss_pct

    # Never size a position larger than the cash actually available.
    max_position = cash * max_position_pct
    return max(0, min(position_size, max_position))


def apply_risk_management(signal, total_score, cash, price, entry_context=None):
    confidence = min(1.0, max(0.0, total_score / 100))
    entry_context = entry_context or {}
    price_vs_ema20_pct = entry_context.get("price_vs_ema20_pct", 0) or 0
    price_vs_ema50_pct = entry_context.get("price_vs_ema50_pct", 0) or 0
    volume_ratio = entry_context.get("volume_vs_avg_pct", 0) or 0
    rr = entry_context.get("risk_reward_ratio", 0) or 0

    if signal in ("SELL", "RED -> SELL / EXIT"):
        return {
            "confidence": round(confidence, 2),
            "size": 0,
            "target": round(price, 2),
            "stop_loss": round(price * 0.95, 2),
            "buy_levels": {
                "patient_entry": round(price * 0.95, 2),
                "optimal_entry": round(price, 2),
                "aggressive_entry": round(price * 1.02, 2),
            },
        }

    if signal == "STRONG BUY":
        patient_discount = 0.03 if confidence >= 0.8 else 0.04
        optimal_discount = 0.01 if confidence >= 0.8 else 0.015
        aggressive_premium = 0.012 if confidence >= 0.8 else 0.015
    elif signal == "BUY / HOLD":
        patient_discount = 0.04
        optimal_discount = 0.0
        aggressive_premium = 0.018
    else:
        patient_discount = 0.05
        optimal_discount = 0.02
        aggressive_premium = 0.025

    if confidence < 0.6:
        patient_discount += 0.01
        optimal_discount += 0.01
        aggressive_premium += 0.01

    if price_vs_ema20_pct < -3 or price_vs_ema50_pct < -4:
        patient_discount += 0.01
        optimal_discount += 0.008
    elif price_vs_ema20_pct > 2 and price_vs_ema50_pct > 2:
        patient_discount -= 0.005
        optimal_discount -= 0.003
        aggressive_premium -= 0.003

    if volume_ratio >= 12:
        aggressive_premium -= 0.002
    if rr >= 1.5:
        aggressive_premium -= 0.001

    # Floors so the stacked +/- adjustments above can never collapse an
    # entry level onto (or past) the current price -- e.g. a very cheap
    # aggressive_premium after several negative adjustments should still
    # sit above current price, not at or below it.
    patient_discount = max(patient_discount, 0.0)
    optimal_discount = max(optimal_discount, 0.0)
    aggressive_premium = max(aggressive_premium, 0.001)

    buy_levels = {
        "patient_entry": round(price * (1 - patient_discount), 2),
        "optimal_entry": round(price * (1 - optimal_discount), 2),
        "aggressive_entry": round(price * (1 + aggressive_premium), 2),
    }

    if confidence >= 0.8:
        target_pct = 0.10
        stop_loss_pct = 0.04
    elif confidence >= 0.65:
        target_pct = 0.08
        stop_loss_pct = 0.05
    else:
        target_pct = 0.07
        stop_loss_pct = 0.06

    target = price * (1 + target_pct)
    stop_loss = price * (1 - stop_loss_pct)

    # Size the position off the SAME stop_loss_pct used for the stop-loss
    # shown above, so "risk_per_trade% of cash" actually matches the real
    # distance to the stop instead of a different, independently-derived tier.
    size = calculate_position_size(cash, confidence, stop_loss_pct=stop_loss_pct)

    return {
        "confidence": round(confidence, 2),
        "size": round(size, 2),
        "target": round(target, 2),
        "stop_loss": round(stop_loss, 2),
        "buy_levels": buy_levels,
    }