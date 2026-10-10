"""Content-based filtering with TF-IDF and cosine similarity on product text."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    def __init__(self, max_features: int = 5000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features, ngram_range=ngram_range)

    def fit(self, products: pd.DataFrame) -> "ContentBasedRecommender":
        self.products = products.reset_index(drop=True)
        text = (
            (self.products["product_name"] + " ") * 2
            + (self.products["category"] + " ") * 2
            + self.products["description"]
        )
        self.tfidf = self.vectorizer.fit_transform(text)
        self.similarity = cosine_similarity(self.tfidf)
        self.product_ids = self.products["product_id"].to_numpy()
        self._pos = {pid: i for i, pid in enumerate(self.product_ids)}
        return self

    def similar_products(self, product_id, k: int = 10) -> pd.Series:
        """Top-k most similar products to a given product."""
        i = self._pos[product_id]
        sims = pd.Series(self.similarity[i], index=self.product_ids).drop(product_id)
        return sims.sort_values(ascending=False).head(k)

    def user_profile_scores(self, user_row: pd.Series, like_threshold: float = 3.0) -> pd.Series:
        """Score every product by similarity to the user's liked products.

        `user_row` is a Series of scores indexed by product_id (0 = unseen).
        """
        liked = user_row[user_row >= like_threshold]
        liked = liked[liked.index.isin(self._pos)]
        if liked.empty:
            return pd.Series(0.0, index=self.product_ids)
        idx = [self._pos[p] for p in liked.index]
        weights = liked.to_numpy(dtype=float)
        profile = np.asarray(self.tfidf[idx].T @ weights).ravel() / weights.sum()
        scores = cosine_similarity(profile.reshape(1, -1), self.tfidf).ravel()
        return pd.Series(scores, index=self.product_ids)

    def recommend(self, user_row: pd.Series, k: int = 10, exclude_seen: bool = True) -> pd.Series:
        scores = self.user_profile_scores(user_row)
        if exclude_seen:
            seen = user_row[user_row > 0].index
            scores = scores.drop(seen.intersection(scores.index))
        return scores.sort_values(ascending=False).head(k)
