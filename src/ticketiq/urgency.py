"""Interpretable urgency scoring: lexicon + shouting + punctuation + category prior."""

import re
from .preprocess import caps_ratio

LEXICON = {
    "urgent": 3.0, "asap": 3.0, "immediately": 2.5, "unacceptable": 2.5,
    "frustrated": 2.0, "angry": 2.0, "furious": 3.0, "nobody": 1.5, "ignored": 2.0,
    "emergency": 3.0, "critical": 2.5, "deadline": 2.0, "twice": 1.0, "lost": 1.0,
    "crash": 1.0, "crashes": 1.0, "locked": 1.5, "damaged": 1.0, "refund": 1.0,
}
CATEGORY_PRIOR = {
    "billing": 0.6, "login_access": 0.8, "bug_report": 0.5,
    "shipping": 0.5, "cancellation": 0.3, "feature_request": -0.8,
}


def urgency_score(text: str, category: str = "") -> float:
    """Score in [0, 1]; higher = more urgent."""
    words = re.findall(r"[a-z']+", text.lower())
    lex = sum(LEXICON.get(w, 0.0) for w in words)
    shout = 1.5 * caps_ratio(text) if len(text) > 15 else 0.0
    bangs = min(text.count("!"), 3) * 0.5
    raw = lex + shout + bangs + CATEGORY_PRIOR.get(category, 0.0)
    return float(1 / (1 + pow(2.718281828, -(raw - 2.5))))  # logistic squash


def priority_label(score: float) -> str:
    return "P1-critical" if score >= 0.75 else "P2-high" if score >= 0.5 else "P3-normal" if score >= 0.25 else "P4-low"
