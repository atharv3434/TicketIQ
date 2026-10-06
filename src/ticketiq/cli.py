"""Command-line interface: `ticketiq triage "..."`, `ticketiq repl`, `ticketiq batch file.csv`."""

import argparse
import json
from pathlib import Path
import pandas as pd
from .pipeline import TicketTriager

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "triager.joblib"


def _load(path):
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Model not found at {p}. Run `make data train` first.")
    return TicketTriager.load(p)


def _pretty(text, r):
    flag = "  (LOW CONFIDENCE - route to human)" if r.low_confidence else ""
    print(f"\nTicket   : {text}")
    print(f"Category : {r.category}  ({r.confidence:.0%}){flag}")
    print(f"Queue    : {r.queue}")
    print(f"Priority : {r.priority}  (urgency {r.urgency:.2f})")
    print(f"Keyphrase: {', '.join(r.keyphrases)}")
    print(f"Because  : {', '.join(f'{t}({s:+.2f})' for t, s in r.why)}")
    print("Similar  :")
    for t, s in r.similar:
        print(f"   {s:.2f}  {t[:90]}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ticketiq")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("triage"); t.add_argument("text"); t.add_argument("--json", action="store_true")
    sub.add_parser("repl")
    b = sub.add_parser("batch"); b.add_argument("csv"); b.add_argument("--col", default="text"); b.add_argument("--out", default="triaged.csv")
    a = ap.parse_args(argv)
    triager = _load(a.model)

    if a.cmd == "triage":
        r = triager.triage(a.text)
        if a.json:
            print(json.dumps(r.to_dict(), indent=2))
        else:
            _pretty(a.text, r)
    elif a.cmd == "repl":
        print("TicketIQ REPL - type a ticket, or 'quit'.")
        while True:
            try:
                s = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if s.lower() in {"quit", "exit", ""}:
                break
            _pretty(s, triager.triage(s))
    elif a.cmd == "batch":
        df = pd.read_csv(a.csv)
        res = [triager.triage(x) for x in df[a.col].astype(str)]
        df["category"] = [r.category for r in res]
        df["confidence"] = [r.confidence for r in res]
        df["priority"] = [r.priority for r in res]
        df["queue"] = [r.queue for r in res]
        df.to_csv(a.out, index=False)
        print(f"Triaged {len(df)} tickets -> {a.out}")


if __name__ == "__main__":
    main()
