"""Hybrid recommender: blends collaborative and content-based scores."""
import pandas as pd


def _minmax(s: pd.Series) -> pd.Series:
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng > 0 else s * 0.0


def popular_products(interactions: pd.DataFrame, k: int = 10, min_votes: int = 5) -> pd.Series:
    """Bayesian-weighted popularity (works for brand-new users / cold start)."""
    stats = interactions.groupby("product_id")["score"].agg(["mean", "count"])
    global_mean = interactions["score"].mean()
    v = stats["count"]
    weighted = (v / (v + min_votes)) * stats["mean"] + (min_votes / (v + min_votes)) * global_mean
    return weighted.sort_values(ascending=False).head(k)


class HybridRecommender:
    """score = alpha * CF + (1 - alpha) * content-based (each min-max normalised)."""

    def __init__(self, cf_model, content_model, interactions: pd.DataFrame, alpha: float = 0.7):
        self.cf = cf_model
        self.content = content_model
        self.interactions = interactions
        self.alpha = alpha

    def has_history(self, user_id) -> bool:
        return user_id in self.cf.matrix.index and (self.cf.matrix.loc[user_id] > 0).any()

    def predict_scores(self, user_id) -> pd.Series:
        if not self.has_history(user_id):
            return popular_products(self.interactions, k=len(self.cf.matrix.columns))
        user_row = self.cf.matrix.loc[user_id]
        cf_scores = _minmax(self.cf.predict_scores(user_id))
        cb_scores = _minmax(self.content.user_profile_scores(user_row)).reindex(cf_scores.index).fillna(0.0)
        return self.alpha * cf_scores + (1 - self.alpha) * cb_scores

    def recommend(self, user_id, k: int = 10, exclude_seen: bool = True) -> pd.Series:
        scores = self.predict_scores(user_id)
        if exclude_seen and self.has_history(user_id):
            seen = self.cf.matrix.loc[user_id]
            scores = scores.drop(seen[seen > 0].index.intersection(scores.index))
        return scores.sort_values(ascending=False).head(k)
