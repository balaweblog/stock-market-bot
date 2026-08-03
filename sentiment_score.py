from datetime import datetime

# Lazy-load the FinBERT pipeline -- loading at import time (the previous
# behaviour) caused the entire system to crash on import if `transformers` was
# unavailable, the model couldn't be downloaded (offline/CI), or HuggingFace
# was rate-limiting. Sentinel is set on first successful load; any subsequent
# import of this module reuses the already-loaded pipeline without reloading.
_sentiment_pipe = None
_sentiment_unavailable = False  # set True once we know loading failed


def _get_pipeline():
    """Return the FinBERT pipeline, loading it lazily on first call."""
    global _sentiment_pipe, _sentiment_unavailable
    if _sentiment_pipe is not None:
        return _sentiment_pipe
    if _sentiment_unavailable:
        return None
    try:
        from transformers import pipeline as _pipeline
        _sentiment_pipe = _pipeline("sentiment-analysis", model="ProsusAI/finbert")
        return _sentiment_pipe
    except Exception:
        _sentiment_unavailable = True
        return None


def score_headlines(headlines, available=True):
    """
    available=False means the news fetch itself failed (see
    news_engine.get_news) -- every source errored out, so `headlines` is
    empty for lack-of-data reasons, not because there's genuinely no news.

    In that case we still return a numeric score of 50.0 so any downstream
    math (calculate_combined_score, etc.) keeps working exactly as before,
    but the label is "Data Unavailable" rather than "Neutral" and
    available=False is passed through -- that's the flag the report should
    key its display off of, so a failed fetch doesn't get shown to the
    reader as if it were a real neutral sentiment reading.
    """
    if not available:
        return {
            "score": 50.0,
            "label": "Data Unavailable",
            "weighted_score": 50.0,
            "details": [],
            "available": False,
        }

    if not headlines:
        return {
            "score": 50.0,
            "label": "Neutral",
            "weighted_score": 50.0,
            "details": [],
            "available": True,
        }

    pipe = _get_pipeline()
    if pipe is None:
        # FinBERT unavailable (not installed, offline, etc.) -- return a neutral
        # fallback so the rest of the scoring pipeline still works.
        return {
            "score": 50.0,
            "label": "Neutral (FinBERT unavailable)",
            "weighted_score": 50.0,
            "details": [],
            "available": True,
        }

    total_score = 0.0
    total_weight = 0.0
    scored = []

    for idx, headline in enumerate(headlines[:8]):
        result = pipe(headline)[0]
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

    return {
        "score": round(weighted, 2),
        "label": label,
        "details": scored,
        "available": True,
    }