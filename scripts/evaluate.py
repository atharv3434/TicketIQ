import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from ticketiq.pipeline import TicketTriager, load_split
from ticketiq.urgency import urgency_score

_, test = load_split(ROOT / "data" / "tickets.csv")
triager = TicketTriager.load(ROOT / "models" / "triager.joblib")

pred = triager.clf.predict(test["text"])
print("=== Intent classification ===")
print(classification_report(test["category"], pred, digits=3))
labels = sorted(test["category"].unique())
print("Confusion matrix (rows=true):", labels)
print(confusion_matrix(test["category"], pred, labels=labels))

scores = [urgency_score(t, c) for t, c in zip(test["text"], pred)]
print(f"\n=== Urgency heuristic ===\nROC-AUC vs. is_urgent label: {roc_auc_score(test['is_urgent'], scores):.3f}")

conf = triager.clf.predict_proba(test["text"]).max(axis=1)
print(f"\nMean confidence: {conf.mean():.3f} | flagged low-confidence: {(conf < triager.low_conf_threshold).mean():.1%}")
