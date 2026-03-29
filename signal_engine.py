"""
Investment signal engine.

Blends the ML model's directional probability with the news sentiment score
into a single composite score, then maps it to BUY / HOLD / SELL.
Thresholds and weights adapt to the active trader profile.
"""

from __future__ import annotations

from utils import normalize, TraderProfile


def generate_signal(
    price_prediction_prob: float,
    sentiment_score: float,
    profile: TraderProfile | None = None,
    *,
    model_weight: float | None = None,
    sentiment_weight: float | None = None,
    buy_threshold: float | None = None,
    sell_threshold: float | None = None,
) -> dict:
    """Produce a trading signal from model output and sentiment.

    When a *profile* is supplied its weights/thresholds are used as defaults.
    Explicit keyword arguments override the profile values.
    """
    mw = model_weight if model_weight is not None else (profile.model_weight if profile else 0.60)
    sw = sentiment_weight if sentiment_weight is not None else (profile.sentiment_weight if profile else 0.40)
    bt = buy_threshold if buy_threshold is not None else (profile.buy_threshold if profile else 0.60)
    st = sell_threshold if sell_threshold is not None else (profile.sell_threshold if profile else 0.40)

    norm_sentiment = normalize(sentiment_score, min_val=-1.0, max_val=1.0)

    score = mw * price_prediction_prob + sw * norm_sentiment

    if score > bt:
        signal = "BUY"
    elif score < st:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        "score": round(score, 4),
        "signal": signal,
        "components": {
            "price_prob": round(price_prediction_prob, 4),
            "norm_sentiment": round(norm_sentiment, 4),
        },
    }
