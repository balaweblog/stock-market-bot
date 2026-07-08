def calculate_position_size(cash, confidence, risk_per_trade=0.01, stop_loss_pct=None):
    if confidence <= 0:
        return 0

    if stop_loss_pct is None:
        if confidence >= 0.85:
            stop_loss_pct = 0.04
        elif confidence >= 0.65:
            stop_loss_pct = 0.05
        else:
            stop_loss_pct = 0.06

    risk_amount = cash * risk_per_trade
    position_size = risk_amount / (stop_loss_pct * confidence)
    return max(0, position_size)


def apply_risk_management(signal, total_score, cash, price):
    confidence = min(1.0, max(0.0, total_score / 100))

    if signal in ("SELL", "RED -> SELL / EXIT"):
        return {
            "confidence": round(confidence, 2),
            "size": 0,
            "target": round(price * 0.95, 2),
            "stop_loss": round(price * 0.95, 2),
            "buy_levels": {
                "patient_entry": round(price * 0.95, 2),
                "optimal_entry": round(price, 2),
                "aggressive_entry": round(price * 1.02, 2),
            },
        }

    size = calculate_position_size(cash, confidence)

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

    return {
        "confidence": round(confidence, 2),
        "size": round(size, 2),
        "target": round(target, 2),
        "stop_loss": round(stop_loss, 2),
        "buy_levels": buy_levels,
    }
