"""Near-duplicate ticket search with TF-IDF + truncated SVD (LSA)."""

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as l2norm
from .preprocess import normalize


class DuplicateFinder:
    def __init__(self, n_components: int = 64):
        self.vec = TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.n_components = n_components
        self.svd = None
        self.texts = []
        self.emb = None

    def fit(self, texts):
        self.texts = list(texts)
        X = self.vec.fit_transform(self.texts)
        k = max(2, min(self.n_components, X.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=0)
        self.emb = l2norm(self.svd.fit_transform(X))
        return self

    def query(self, text: str, top_k: int = 3, min_sim: float = 0.0):
        q = l2norm(self.svd.transform(self.vec.transform([text])))
        sims = (self.emb @ q.T).ravel()
        idx = np.argsort(-sims)[:top_k]
        return [(self.texts[i], float(sims[i])) for i in idx if sims[i] >= min_sim]
