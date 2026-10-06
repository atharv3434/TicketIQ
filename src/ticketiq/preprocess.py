"""Lightweight text normalisation (no external corpora required)."""

import re

_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\S+@\S+\.\S+")
_NUM = re.compile(r"\b\d+(?:[.,]\d+)*\b")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, mask URLs/emails/numbers, collapse whitespace."""
    text = text.lower()
    text = _URL.sub(" <url> ", text)
    text = _EMAIL.sub(" <email> ", text)
    text = _NUM.sub(" <num> ", text)
    return _WS.sub(" ", text).strip()


def caps_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum(c.isupper() for c in letters) / len(letters) if letters else 0.0
