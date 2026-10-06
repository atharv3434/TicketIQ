.PHONY: install data train evaluate demo test all
install:
	pip install -r requirements.txt && pip install -e .
data:
	python data/generate_data.py
train:
	python scripts/train.py
evaluate:
	python scripts/evaluate.py
demo:
	ticketiq triage "URGENT!!! I was charged twice for my subscription and nobody is replying"
test:
	pytest -q
all: data train evaluate test
