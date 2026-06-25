import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def build_training_data(records):
    features = []
    labels = []
    for rec in records:
        features.append([
            rec.get("tech_score", 0),
            rec.get("fund_score", 0),
            rec.get("sentiment_score", 0),
            rec.get("revenue_growth", 0) or 0,
            rec.get("debt_to_equity", 0) or 0,
            rec.get("free_cash_flow", 0) or 0,
        ])
        labels.append(1 if rec.get("total_return", 0) > 0 else 0)
    return np.array(features), np.array(labels)


def train_model(records):
    X, y = build_training_data(records)
    if len(y) < 10 or len(np.unique(y)) < 2:
        raise ValueError("Not enough training data or no label variance")

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    scores = cross_val_score(model, X, y, cv=3, scoring="accuracy")
    model.fit(X, y)
    return model, scores.mean()


def predict_signal(model, features):
    return model.predict_proba([features])[0][1]
