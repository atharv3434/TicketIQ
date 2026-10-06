"""Generate a synthetic, noisy support-ticket dataset (no downloads needed)."""
import random
from pathlib import Path
import pandas as pd

SEED = 42
N_PER_CLASS = 400

TEMPLATES = {
    "billing": [
        "I was charged twice for my {plan} subscription this month",
        "My invoice shows the wrong amount, I need a refund for {amt}",
        "Why was my credit card billed after I downgraded to {plan}?",
        "The payment failed but money left my account, please check the {thing}",
        "I need a copy of my receipt for the {plan} plan",
        "There is an unexpected charge of {amt} on my statement",
    ],
    "login_access": [
        "I can't log in, the password reset email never arrives",
        "My account is locked after too many attempts",
        "Two-factor code is not working on my {device}",
        "I forgot my username and the recovery link is broken",
        "Getting 'access denied' when opening the {thing} page",
        "Single sign-on keeps redirecting me back to the login screen",
    ],
    "bug_report": [
        "The app crashes whenever I open the {thing} on my {device}",
        "Export to CSV produces an empty file",
        "Clicking save on the {thing} throws an error 500",
        "Notifications show up twice and the badge count is wrong",
        "The {thing} page freezes and never finishes loading",
        "Dark mode makes the text unreadable in the {thing}",
    ],
    "feature_request": [
        "It would be great to have a {thing} integration with Slack",
        "Please add bulk editing to the {thing}",
        "Can you support exporting to PDF with custom branding?",
        "We would love an API endpoint for the {thing}",
        "Any plans for a native {device} app?",
        "Suggestion: allow scheduling reports from the {thing}",
    ],
    "shipping": [
        "My order has not arrived and tracking has not updated in {n} days",
        "The package was delivered to the wrong address",
        "I received a damaged item, the box was crushed",
        "Where is my parcel? It was supposed to arrive {n} days ago",
        "The courier marked my order as delivered but I got nothing",
        "I need to change the delivery address for my order",
    ],
    "cancellation": [
        "Please cancel my {plan} subscription effective immediately",
        "How do I close my account and delete my data?",
        "I want to stop my membership before the next renewal",
        "Cancel my order, I no longer need it",
        "I am switching providers, please terminate my contract",
        "Turn off auto-renew for my {plan} plan",
    ],
}
FILL = {
    "plan": ["Pro", "Basic", "Enterprise", "Team", "Starter"],
    "amt": ["$19.99", "$49", "$120.50", "$9.99", "$299"],
    "thing": ["dashboard", "report builder", "settings", "checkout", "billing page", "profile"],
    "device": ["iPhone", "Android phone", "laptop", "iPad", "Windows PC"],
    "n": ["3", "5", "7", "10", "12"],
}
URGENT = ["URGENT!!!", "ASAP.", "This is unacceptable!", "Need help immediately!", "Very frustrated,"]
POLITE = ["Hi team,", "Hello,", "Thanks in advance!", "Good morning,", "Hope you can help."]
CLOSERS = ["Please advise.", "Thank you.", "Any update?", "", "", ""]


def typo(word, rng):
    if len(word) > 4 and rng.random() < 0.5:
        i = rng.randrange(1, len(word) - 2)
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    return word


def noisy(text, rng, p=0.06):
    return " ".join(typo(w, rng) if rng.random() < p else w for w in text.split())


def make_ticket(cat, rng):
    tpl = rng.choice(TEMPLATES[cat])
    text = tpl.format(**{k: rng.choice(v) for k, v in FILL.items()})
    urgent = rng.random() < 0.3
    prefix = rng.choice(URGENT) if urgent else rng.choice(POLITE)
    text = f"{prefix} {text}. {rng.choice(CLOSERS)}".strip()
    if urgent and rng.random() < 0.5:
        text = text.upper()
    return noisy(text, rng), int(urgent)


def main():
    rng = random.Random(SEED)
    rows = [(*make_ticket(c, rng), c) for c in TEMPLATES for _ in range(N_PER_CLASS)]
    df = pd.DataFrame(rows, columns=["text", "is_urgent", "category"]).sample(frac=1, random_state=SEED)
    out = Path(__file__).parent / "tickets.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} tickets -> {out}")


if __name__ == "__main__":
    main()
