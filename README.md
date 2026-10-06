# TicketIQ - Support Ticket Triage Engine

An intermediate, **offline** NLP project (pure scikit-learn, no model downloads) that turns a raw
support message into an actionable triage decision.

| Capability | Technique |
|---|---|
| Intent classification (6 classes) | Word + char n-gram TF-IDF -> Logistic Regression |
| Explainability | Per-prediction top contributing n-grams (coef x tf-idf) |
| Urgency / priority (P1-P4) | Interpretable lexicon + SHOUTING + punctuation + category prior, logistic-squashed |
| Keyphrase extraction | RAKE implemented from scratch |
| Duplicate / similar tickets | TF-IDF -> Truncated SVD (LSA) -> cosine similarity |
| Human-in-the-loop | Low-confidence predictions are flagged for manual routing |

## Quick start
```bash
pip install -r requirements.txt && pip install -e .
make data train evaluate test      # or: make all
ticketiq triage "URGENT!!! I was charged twice and nobody is replying"
ticketiq triage "App crashes on my iPhone" --json
ticketiq repl                      # interactive
ticketiq batch data/tickets.csv --out triaged.csv
python scripts/evaluate_holdout.py # honest stress test
```

## Layout
```
data/generate_data.py        synthetic noisy ticket generator (typos, shouting, urgency)
data/holdout_paraphrases.csv hand-written unseen phrasings for stress testing
src/ticketiq/                preprocess, keyphrases, urgency, similarity, model, pipeline, cli
scripts/                     train.py, evaluate.py, evaluate_holdout.py
tests/test_core.py           unit + end-to-end tests
```

## Read this before trusting the metrics
The synthetic train/test split shares templates, so `evaluate.py` reports ~100% accuracy - that is
**leakage, not skill**. On the 12 hand-written paraphrases the model scores ~75%, and about two thirds of its
errors carry low confidence, which is why the triage step flags them. That gap is the point of the project's
extension exercises.

## Extension ideas
1. Replace TF-IDF with sentence embeddings (`sentence-transformers`) and compare on the holdout set.
2. Calibrate probabilities (`CalibratedClassifierCV`) and tune `low_conf_threshold` for a target review rate.
3. Learn urgency from labels (`is_urgent` column) with a classifier, then compare against the heuristic.
4. Grow the holdout set to 200+ real tickets and add k-fold group splits by template.
5. Wrap `TicketTriager` in a FastAPI endpoint and add active learning from flagged tickets.
