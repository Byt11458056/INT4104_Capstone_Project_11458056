"""
Next-day direction prediction using a Random Forest classifier.

The model is intentionally simple — it exists to demonstrate the pipeline, not
to serve as a production trading signal.
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

FEATURE_COLS = ["returns", "rsi", "ma_5", "ma_20", "volatility"]

_MIN_ROWS = 10  # absolute minimum to attempt training


def train_price_model(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    test_size: float = 0.2,
) -> tuple[RandomForestClassifier, float, float]:
    """Train a Random Forest on *df* and return (model, train_acc, test_acc).

    Automatically clamps *test_size* so that both train and test sets contain
    at least one sample, even for very short lookback periods.
    """
    feature_cols = feature_cols or FEATURE_COLS

    X = df[feature_cols].values
    y = df["target"].values
    n = len(X)

    if n < 2:
        raise ValueError(
            f"Only {n} row(s) after feature engineering — need at least "
            f"{_MIN_ROWS}.  Try a longer lookback period."
        )

    max_test = max(1, n - 2)
    actual_test = max(1, min(int(n * test_size), max_test))
    effective_test_size = actual_test / n

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=effective_test_size, shuffle=False,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    return model, train_acc, test_acc


def predict_next_day(model: RandomForestClassifier, df: pd.DataFrame) -> tuple[int, float]:
    """Return (predicted_direction, probability_of_up) for the latest row."""
    feature_cols = FEATURE_COLS
    latest = df[feature_cols].iloc[[-1]].values
    prob = model.predict_proba(latest)[0]

    up_prob = float(prob[1]) if len(prob) > 1 else float(prob[0])
    direction = 1 if up_prob >= 0.5 else 0
    return direction, up_prob
