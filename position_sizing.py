def calculate_position_size(cash, confidence, risk_per_trade=0.01, stop_loss_pct=0.05):
    if confidence <= 0:
        return 0
    risk_amount = cash * risk_per_trade
    position_size = risk_amount / (stop_loss_pct * confidence)
    return max(0, position_size)


def apply_risk_management(signal, total_score, cash, price):
    confidence = min(1.0, max(0.0, total_score / 100))
    size = calculate_position_size(cash, confidence)
    target = price * 1.1
    stop_loss = price * 0.95

    if signal in ("SELL", "RED -> SELL / EXIT"):
        size = 0

    return {
        "confidence": round(confidence, 2),
        "size": round(size, 2),
        "target": round(target, 2),
        "stop_loss": round(stop_loss, 2),
    }
