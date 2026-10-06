"""RAKE-style keyphrase extraction, implemented from scratch."""

import re
from collections import defaultdict
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_STOP = set(ENGLISH_STOP_WORDS) | {"please", "hi", "hello", "thanks", "thank", "advise", "update", "team", "need", "want", "would", "like"}
_SPLIT = re.compile(r"[^a-zA-Z0-9'\-\$\.]+|[.!?,;:]+(?:\s|$)")


def extract_keyphrases(text: str, top_k: int = 5):
    """Return [(phrase, score)] ranked by RAKE degree/frequency score."""
    tokens = re.findall(r"[A-Za-z0-9'\-\$]+(?:\.\d+)?|[.!?,;:]", text.lower())
    phrases, cur = [], []
    for tok in tokens:
        if tok in _STOP or re.fullmatch(r"[.!?,;:]", tok):
            if cur:
                phrases.append(cur)
                cur = []
        else:
            cur.append(tok)
    if cur:
        phrases.append(cur)

    freq, degree = defaultdict(int), defaultdict(int)
    for ph in phrases:
        for w in ph:
            freq[w] += 1
            degree[w] += len(ph)
    scored = {}
    for ph in phrases:
        if len(ph) > 4:
            continue
        scored[" ".join(ph)] = sum(degree[w] / freq[w] for w in ph)
    return sorted(scored.items(), key=lambda kv: -kv[1])[:top_k]
