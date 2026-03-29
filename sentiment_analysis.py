"""
LLM-based sentiment analysis for news articles.

Uses the same LLM provider configured by the user to score headlines,
replacing the heavy FinBERT model.  Falls back to keyword-based scoring
when no LLM is available.
"""

from __future__ import annotations

import json
import logging
import re

import numpy as np
from openai import OpenAI

log = logging.getLogger(__name__)

_BULLISH = {
    "surge": 0.7, "surges": 0.7, "soar": 0.8, "soars": 0.8,
    "rally": 0.6, "rallies": 0.6, "jump": 0.5, "jumps": 0.5,
    "gain": 0.4, "gains": 0.4, "rise": 0.3, "rises": 0.3,
    "beat": 0.5, "beats": 0.5, "record": 0.4, "upgrade": 0.6,
    "outperform": 0.5, "bullish": 0.6, "strong": 0.3,
    "growth": 0.3, "profit": 0.3, "boom": 0.6,
    "breakthrough": 0.5, "positive": 0.3, "optimistic": 0.4,
}

_BEARISH = {
    "crash": -0.8, "crashes": -0.8, "plunge": -0.7, "plunges": -0.7,
    "drop": -0.4, "drops": -0.4, "fall": -0.3, "falls": -0.3,
    "decline": -0.4, "declines": -0.4, "loss": -0.4, "losses": -0.4,
    "miss": -0.5, "misses": -0.5, "cut": -0.4, "downgrade": -0.6,
    "underperform": -0.5, "bearish": -0.6, "weak": -0.3,
    "risk": -0.2, "concern": -0.3, "fear": -0.4,
    "sell-off": -0.6, "selloff": -0.6, "recession": -0.5,
    "layoff": -0.4, "layoffs": -0.4, "negative": -0.3,
    "warning": -0.3, "tariff": -0.3, "sanctions": -0.3,
}

_SENTIMENT_PROMPT = (
    "You are a financial sentiment analyst. Score each headline on a scale "
    "from -1.0 (extremely bearish) to +1.0 (extremely bullish). 0.0 is neutral.\n\n"
    "Consider financial context: earnings beats are positive, layoffs negative, "
    "regulatory actions depend on context, macro tariffs/sanctions are negative.\n\n"
    "Return ONLY a JSON array with one object per headline:\n"
    '[{"index": 0, "score": 0.5, "label": "positive"}, ...]\n\n'
    'Labels: "positive" (score > 0.05), "negative" (score < -0.05), "neutral" otherwise.\n\n'
    "Headlines:\n"
)


def _keyword_score(text: str) -> float:
    text_lower = text.lower()
    scores = []
    for word, score in {**_BULLISH, **_BEARISH}.items():
        if word in text_lower:
            scores.append(score)
    return float(np.mean(scores)) if scores else 0.0


def _parse_llm_scores(response_text: str, count: int) -> list[dict] | None:
    try:
        match = re.search(r"\[.*\]", response_text, re.DOTALL)
        if not match:
            return None
        data = json.loads(match.group())
        if isinstance(data, list):
            return data[:count]
        return None
    except (json.JSONDecodeError, Exception):
        return None


def _llm_sentiment(
    texts: list[str],
    api_key: str,
    base_url: str,
    model: str,
    extra_headers: dict | None = None,
) -> list[dict]:
    headlines_block = "\n".join(f"[{i}] {t[:200]}" for i, t in enumerate(texts))
    prompt = _SENTIMENT_PROMPT + headlines_block

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers=extra_headers or {},
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a financial sentiment scoring system. Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=1500,
    )
    result_text = response.choices[0].message.content or ""
    parsed = _parse_llm_scores(result_text, len(texts))
    if parsed is None:
        raise ValueError("Could not parse LLM sentiment response")
    return parsed


def analyze_sentiment(
    articles: list[dict],
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    extra_headers: dict | None = None,
) -> dict:
    """Score each article and return aggregate statistics.

    Uses LLM when credentials are provided, falls back to keyword scoring.

    Returns ``{"headlines": [...], "avg_score": float}``.
    """
    if not articles:
        return {"headlines": [], "avg_score": 0.0}

    items: list[dict] = []
    for item in articles:
        if isinstance(item, str):
            items.append({"title": item, "url": "", "source": "", "content": ""})
        else:
            items.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "source": item.get("source", ""),
                "content": item.get("content", ""),
            })

    texts: list[str] = []
    for it in items:
        text = it["title"]
        if it["content"] and len(it["content"]) > len(text):
            text = it["content"][:500]
        if text:
            texts.append(text)

    if not texts:
        return {"headlines": [], "avg_score": 0.0}

    llm_scores = None
    if api_key and base_url and model:
        try:
            llm_scores = _llm_sentiment(texts, api_key, base_url, model, extra_headers)
        except Exception as exc:
            log.warning("LLM sentiment failed, falling back to keywords: %s", exc)

    scored: list[dict] = []
    raw_scores: list[float] = []

    for i, it in enumerate(items):
        if i >= len(texts):
            break

        if llm_scores and i < len(llm_scores):
            entry = llm_scores[i]
            score = float(entry.get("score", 0.0))
            score = max(-1.0, min(1.0, score))
            label = entry.get("label", "neutral")
        else:
            score = _keyword_score(texts[i])
            label = "positive" if score > 0.05 else ("negative" if score < -0.05 else "neutral")

        scored.append({
            "text": it["title"],
            "url": it["url"],
            "source": it["source"],
            "label": label,
            "score": round(score, 4),
        })
        raw_scores.append(score)

    return {
        "headlines": scored,
        "avg_score": float(np.mean(raw_scores)) if raw_scores else 0.0,
    }
