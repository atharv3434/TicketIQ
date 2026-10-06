"""Stress test on hand-written paraphrases the synthetic templates never contained."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
from ticketiq.pipeline import TicketTriager

df = pd.read_csv(ROOT / "data" / "holdout_paraphrases.csv")
t = TicketTriager.load(ROOT / "models" / "triager.joblib")
res = [t.triage(x) for x in df["text"]]
df["pred"] = [r.category for r in res]
df["conf"] = [r.confidence for r in res]
df["flagged"] = [r.low_confidence for r in res]
print(df[["text", "category", "pred", "conf", "flagged"]].to_string(index=False, max_colwidth=50))
print(f"\nHoldout accuracy: {(df.category == df.pred).mean():.1%} on {len(df)} unseen paraphrases")
print(f"Wrong answers flagged low-confidence: {df[df.category != df.pred].flagged.mean():.0%}" if (df.category != df.pred).any() else "No errors.")
