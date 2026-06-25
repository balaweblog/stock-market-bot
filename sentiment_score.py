from transformers import pipeline
from datetime import datetime

sentiment_pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert")


def score_headlines(headlines):
    if not headlines:
        return {"score": 50.0, "label": "Neutral", "weighted_score": 50.0}

    total_score = 0.0
    total_weight = 0.0
    scored = []

    for idx, headline in enumerate(headlines[:8]):
        result = sentiment_pipe(headline)[0]
        label = result["label"].lower()
        confidence = result["score"]
        weight = max(1.0, 3.0 - idx * 0.3)

        value = 0.0
        if label == "positive":
            value = 50 + confidence * 50
        elif label == "negative":
            value = 50 - confidence * 50
        else:
            value = 50

        total_score += value * weight
        total_weight += weight
        scored.append({"headline": headline, "label": label, "confidence": confidence, "weight": weight, "value": value})

    weighted = total_score / total_weight if total_weight else 50.0
    label = "Positive" if weighted > 60 else "Negative" if weighted < 40 else "Neutral"

    return {"score": round(weighted, 2), "label": label, "details": scored}
