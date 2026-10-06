from ticketiq.preprocess import normalize, caps_ratio
from ticketiq.keyphrases import extract_keyphrases
from ticketiq.urgency import urgency_score, priority_label
from ticketiq.similarity import DuplicateFinder
from ticketiq.pipeline import TicketTriager


def test_normalize_masks_entities():
    out = normalize("Email me at a@b.com or see https://x.io, paid 49.99!")
    assert "<email>" in out and "<url>" in out and "<num>" in out


def test_caps_ratio():
    assert caps_ratio("ABC") == 1.0 and caps_ratio("abc") == 0.0 and caps_ratio("123") == 0.0


def test_keyphrases_found():
    kp = [p for p, _ in extract_keyphrases("My credit card was charged twice for the premium subscription")]
    assert any("credit card" in p for p in kp)


def test_urgency_ordering():
    calm = urgency_score("Hello, would be nice to have dark mode.", "feature_request")
    hot = urgency_score("URGENT!!! Account locked, nobody is replying, unacceptable!", "login_access")
    assert hot > calm and priority_label(hot) in {"P1-critical", "P2-high"}


def test_duplicate_finder_returns_self():
    texts = ["refund my double charge", "app crashes on launch", "reset password email missing"] * 3
    d = DuplicateFinder(n_components=4).fit(texts)
    assert d.query("refund double charge", 1)[0][0] == "refund my double charge"


def test_pipeline_end_to_end():
    X = ["charged twice refund invoice", "invoice amount wrong refund", "payment charged card billing",
         "cannot login password reset", "login locked password", "password reset email login"] * 4
    y = ["billing"] * 3 + ["login_access"] * 3
    y = y * 4
    t = TicketTriager().fit(X, y)
    r = t.triage("I was charged and want a refund")
    assert r.category == "billing" and 0 <= r.urgency <= 1 and r.queue == "Finance Team"
