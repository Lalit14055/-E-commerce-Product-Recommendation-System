"""Collaborative filtering (user-based kNN) and matrix factorization (SVD)."""
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity


class _BaseRecommender:
    matrix: pd.DataFrame

    def predict_scores(self, user_id) -> pd.Series:  # pragma: no cover - interface
        raise NotImplementedError

    def recommend(self, user_id, k: int = 10, exclude_seen: bool = True) -> pd.Series:
        scores = self.predict_scores(user_id)
        if exclude_seen and user_id in self.matrix.index:
            seen = self.matrix.loc[user_id]
            scores = scores.drop(seen[seen > 0].index.intersection(scores.index))
        return scores.sort_values(ascending=False).head(k)


class UserBasedCF(_BaseRecommender):
    """Recommends products liked by the most similar users (cosine similarity)."""

    def __init__(self, n_neighbors: int = 30):
        self.n_neighbors = n_neighbors

    def fit(self, matrix: pd.DataFrame) -> "UserBasedCF":
        self.matrix = matrix
        self._values = matrix.to_numpy(dtype=float)
        self._sim = cosine_similarity(self._values)
        np.fill_diagonal(self._sim, 0.0)
        self._user_pos = {u: i for i, u in enumerate(matrix.index)}
        return self

    def predict_scores(self, user_id) -> pd.Series:
        if user_id not in self._user_pos:
            return pd.Series(0.0, index=self.matrix.columns)
        i = self._user_pos[user_id]
        sims = self._sim[i]
        top = np.argsort(sims)[::-1][: self.n_neighbors]
        w = sims[top]
        if w.sum() == 0:
            return pd.Series(0.0, index=self.matrix.columns)
        scores = (w @ self._values[top]) / w.sum()
        return pd.Series(scores, index=self.matrix.columns)

    def predict_rating(self, user_id, product_id) -> float:
        scores = self.predict_scores(user_id)
        return float(np.clip(scores.get(product_id, 0.0), 1, 5))


class SVDRecommender(_BaseRecommender):
    """Matrix factorization with truncated SVD on a mean-centred rating matrix.

    For rating prediction (RMSE) fit it on an explicit-ratings-only matrix; for
    top-N ranking the blended implicit/explicit `score` matrix works well.
    """

    def __init__(self, n_factors: int = 20):
        self.n_factors = n_factors

    def fit(self, matrix: pd.DataFrame) -> "SVDRecommender":
        self.matrix = matrix
        R = matrix.to_numpy(dtype=float)
        mask = R > 0
        counts = mask.sum(axis=1, keepdims=True)
        global_mean = R[mask].mean() if mask.any() else 3.0
        # Users with no signal fall back to the global mean.
        self._user_means = np.where(counts > 0, (R * mask).sum(axis=1, keepdims=True) / np.maximum(counts, 1), global_mean)
        centred = np.where(mask, R - self._user_means, 0.0)
        k = int(min(self.n_factors, min(R.shape) - 1))
        U, s, Vt = svds(centred, k=k)
        self.user_factors = U * np.sqrt(s)                 # latent user preferences
        self.item_factors = (np.sqrt(s)[:, None] * Vt).T   # latent product features
        self._pred = self.user_factors @ self.item_factors.T + self._user_means
        self._user_pos = {u: i for i, u in enumerate(matrix.index)}
        self._item_pos = {p: j for j, p in enumerate(matrix.columns)}
        return self

    def predict_scores(self, user_id) -> pd.Series:
        if user_id not in self._user_pos:
            return pd.Series(0.0, index=self.matrix.columns)
        return pd.Series(self._pred[self._user_pos[user_id]], index=self.matrix.columns)

    def predict_rating(self, user_id, product_id) -> float:
        if user_id not in self._user_pos or product_id not in self._item_pos:
            return 3.0
        return float(np.clip(self._pred[self._user_pos[user_id], self._item_pos[product_id]], 1, 5))
