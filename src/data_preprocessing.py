"""Loading, cleaning and preparing the product and interaction data."""
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Implicit-feedback strength used when an explicit rating is missing.
IMPLICIT_WEIGHTS = {"click": 1.0, "add_to_cart": 3.0, "purchase": 4.0}


def load_data(data_dir: Path = DATA_DIR) -> Tuple[pd.DataFrame, pd.DataFrame]:
    products = pd.read_csv(Path(data_dir) / "products.csv")
    interactions = pd.read_csv(Path(data_dir) / "interactions.csv")
    return products, interactions


def clean_data(products: pd.DataFrame, interactions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Handle missing values, remove duplicates and add a unified `score` column."""
    products = products.drop_duplicates(subset="product_id").copy()
    for col in ["product_name", "category", "description"]:
        products[col] = products[col].fillna("").astype(str)
    products["price"] = pd.to_numeric(products["price"], errors="coerce")
    products["price"] = products["price"].fillna(products["price"].median())

    inter = interactions.drop_duplicates().dropna(subset=["user_id", "product_id"]).copy()
    inter = inter[inter["product_id"].isin(products["product_id"])]
    inter["interaction_type"] = inter["interaction_type"].fillna("click")
    inter["rating"] = pd.to_numeric(inter["rating"], errors="coerce")

    implicit = inter["interaction_type"].map(IMPLICIT_WEIGHTS).fillna(1.0)
    inter["score"] = inter["rating"].fillna(implicit).clip(1, 5)

    # One row per (user, product): keep the strongest signal.
    inter = (
        inter.sort_values("score")
        .groupby(["user_id", "product_id"], as_index=False)
        .agg(rating=("rating", "max"), interaction_type=("interaction_type", "last"), score=("score", "max"))
    )
    return products.reset_index(drop=True), inter.reset_index(drop=True)


def build_user_item_matrix(interactions: pd.DataFrame, product_ids=None, user_ids=None) -> pd.DataFrame:
    """Pivot interactions into a users x products matrix of scores (0 = unseen)."""
    matrix = interactions.pivot_table(index="user_id", columns="product_id", values="score", aggfunc="max")
    if product_ids is not None:
        matrix = matrix.reindex(columns=product_ids)
    if user_ids is not None:
        matrix = matrix.reindex(index=user_ids)
    return matrix.fillna(0.0)


def train_test_split_interactions(
    interactions: pd.DataFrame, test_size: float = 0.2, min_interactions: int = 5, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-user hold-out split. Users with few interactions stay entirely in train."""
    rng = np.random.default_rng(seed)
    train_parts, test_parts = [], []
    for _, grp in interactions.groupby("user_id"):
        if len(grp) < min_interactions:
            train_parts.append(grp)
            continue
        n_test = max(1, int(round(len(grp) * test_size)))
        test_idx = rng.choice(grp.index.to_numpy(), size=n_test, replace=False)
        test_parts.append(grp.loc[test_idx])
        train_parts.append(grp.drop(index=test_idx))
    return pd.concat(train_parts), pd.concat(test_parts)


def load_and_prepare(data_dir: Path = DATA_DIR) -> Tuple[pd.DataFrame, pd.DataFrame]:
    products, interactions = load_data(data_dir)
    return clean_data(products, interactions)
