from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert"
)

def analyze_sentiment(headlines):
    if not headlines:
        return 50, "Neutral"

    score = 0

    for headline in headlines:
        result = classifier(headline)[0]

        label = result["label"]
        confidence = result["score"]

        if label == "positive":
            score += 10 * confidence
        elif label == "negative":
            score -= 10 * confidence

    final = 50 + score

    if final > 60:
        label = "Positive"
    elif final < 40:
        label = "Negative"
    else:
        label = "Neutral"

    return max(0, min(100, final)), label