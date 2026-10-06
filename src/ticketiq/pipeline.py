"""End-to-end triage orchestrator."""

from dataclasses import dataclass, asdict
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from .model import build_classifier, top_terms
from .keyphrases import extract_keyphrases
from .urgency import urgency_score, priority_label
from .similarity import DuplicateFinder

ROUTING = {
    "billing": "Finance Team", "login_access": "Identity & Access", "bug_report": "Engineering",
    "feature_request": "Product", "shipping": "Logistics", "cancellation": "Retention",
}


@dataclass
class TriageResult:
    category: str
    confidence: float
    queue: str
    urgency: float
    priority: str
    keyphrases: list
    why: list
    similar: list
    low_confidence: bool

    def to_dict(self):
        return asdict(self)


class TicketTriager:
    def __init__(self, low_conf_threshold: float = 0.55):
        self.clf = build_classifier()
        self.dups = DuplicateFinder()
        self.low_conf_threshold = low_conf_threshold

    def fit(self, texts, labels):
        self.clf.fit(texts, labels)
        self.dups.fit(texts)
        return self

    def triage(self, text: str, top_k_similar: int = 3) -> TriageResult:
        proba = self.clf.predict_proba([text])[0]
        i = proba.argmax()
        cat, conf = self.clf.classes_[i], float(proba[i])
        u = urgency_score(text, cat)
        return TriageResult(
            category=cat, confidence=round(conf, 3), queue=ROUTING.get(cat, "General"),
            urgency=round(u, 3), priority=priority_label(u),
            keyphrases=[p for p, _ in extract_keyphrases(text)],
            why=[(t, round(s, 3)) for t, s in top_terms(self.clf, text, cat)],
            similar=[(t, round(s, 3)) for t, s in self.dups.query(text, top_k_similar)],
            low_confidence=conf < self.low_conf_threshold,
        )

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "TicketTriager":
        return joblib.load(path)


def load_split(csv_path, test_size=0.2, seed=42):
    df = pd.read_csv(csv_path)
    return train_test_split(df, test_size=test_size, random_state=seed, stratify=df["category"])
