import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ticketiq.pipeline import TicketTriager, load_split

train, test = load_split(ROOT / "data" / "tickets.csv")
triager = TicketTriager().fit(train["text"].tolist(), train["category"].tolist())
out = ROOT / "models" / "triager.joblib"
triager.save(out)
print(f"Trained on {len(train)} tickets, saved -> {out}")
