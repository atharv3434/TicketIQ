"""Intent classifier: word + char n-gram TF-IDF -> calibrated-ish Logistic Regression."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from .preprocess import normalize


def build_classifier() -> Pipeline:
    feats = FeatureUnion([
        ("word", TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(preprocessor=normalize, analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)),
    ])
    return Pipeline([("features", feats), ("clf", LogisticRegression(C=5.0, max_iter=1000))])


def top_terms(pipeline: Pipeline, text: str, label: str, k: int = 5):
    """Explain a prediction: highest-weight word features present in the text."""
    word_vec = pipeline.named_steps["features"].transformer_list[0][1]
    clf = pipeline.named_steps["clf"]
    n_word = len(word_vec.vocabulary_)
    row = list(clf.classes_).index(label)
    coefs = clf.coef_[row][:n_word]
    vocab = {i: t for t, i in word_vec.vocabulary_.items()}
    x = word_vec.transform([text]).tocoo()
    contrib = sorted(((vocab[j], float(v * coefs[j])) for j, v in zip(x.col, x.data)), key=lambda kv: -kv[1])
    return contrib[:k]
