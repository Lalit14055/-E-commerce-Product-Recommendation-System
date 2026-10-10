"""Ranking and rating-prediction metrics."""
from typing import Callable, Dict, Iterable

import numpy as np
import pandas as pd


def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    return len(set(recommended[:k]) & relevant) / k if k else 0.0


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    return len(set(recommended[:k]) & relevant) / len(relevant) if relevant else 0.0


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    dcg = sum(1.0 / np.log2(i + 2) for i, p in enumerate(recommended[:k]) if p in relevant)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true, y_pred = np.asarray(list(y_true), float), np.asarray(list(y_pred), float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2))) if len(y_true) else float("nan")


def evaluate_ranking(
    recommend_fn: Callable, test: pd.DataFrame, k: int = 10, relevance_threshold: float = 3.0
) -> Dict[str, float]:
    """Average Precision@K / Recall@K / NDCG@K across users in the test set.

    `recommend_fn(user_id, k)` must return a Series indexed by product_id
    (already excluding the user's training items).
    """
    p, r, n = [], [], []
    for user_id, grp in test.groupby("user_id"):
        relevant = set(grp.loc[grp["score"] >= relevance_threshold, "product_id"])
        if not relevant:
            continue
        recs = list(recommend_fn(user_id, k).index)
        p.append(precision_at_k(recs, relevant, k))
        r.append(recall_at_k(recs, relevant, k))
        n.append(ndcg_at_k(recs, relevant, k))
    return {f"Precision@{k}": float(np.mean(p)), f"Recall@{k}": float(np.mean(r)), f"NDCG@{k}": float(np.mean(n))}


def evaluate_rmse(predict_rating_fn: Callable, test: pd.DataFrame) -> float:
    """RMSE on held-out interactions that carry an explicit rating."""
    rated = test.dropna(subset=["rating"])
    preds = [predict_rating_fn(u, p) for u, p in zip(rated["user_id"], rated["product_id"])]
    return rmse(rated["rating"], preds)
