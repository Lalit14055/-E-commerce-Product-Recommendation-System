"""Generate a synthetic e-commerce dataset (products.csv + interactions.csv).

Replace the output files with a real dataset (e.g. Amazon Reviews, Kaggle) that
follows the same column layout to use your own data.

Usage:
    python src/generate_sample_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

CATEGORIES = {
    "Electronics": {
        "nouns": ["Headphones", "Speaker", "Charger", "Smartwatch", "Power Bank", "Webcam"],
        "words": ["wireless", "bluetooth", "battery", "portable", "usb", "smart", "display",
                  "noise", "cancelling", "fast", "charging", "compact", "rechargeable"],
    },
    "Home & Kitchen": {
        "nouns": ["Blender", "Air Fryer", "Cookware Set", "Coffee Maker", "Knife Set", "Toaster"],
        "words": ["stainless", "steel", "kitchen", "cooking", "nonstick", "dishwasher", "safe",
                  "brew", "family", "easy", "clean", "durable", "countertop"],
    },
    "Fitness": {
        "nouns": ["Yoga Mat", "Dumbbell Set", "Resistance Bands", "Jump Rope", "Foam Roller", "Water Bottle"],
        "words": ["workout", "training", "strength", "exercise", "grip", "nonslip", "gym",
                  "flexible", "lightweight", "recovery", "muscle", "cardio", "home"],
    },
    "Books": {
        "nouns": ["Novel", "Cookbook", "Biography", "Science Guide", "Mystery Thriller", "Self-Help Book"],
        "words": ["bestselling", "story", "author", "chapters", "readers", "paperback", "insightful",
                  "gripping", "narrative", "guide", "illustrated", "edition", "classic"],
    },
    "Fashion": {
        "nouns": ["Sneakers", "Backpack", "Jacket", "Sunglasses", "Denim Jeans", "Leather Wallet"],
        "words": ["stylish", "comfortable", "cotton", "casual", "fit", "premium", "breathable",
                  "wardrobe", "fashion", "everyday", "lightweight", "design", "wear"],
    },
    "Beauty": {
        "nouns": ["Face Serum", "Moisturizer", "Shampoo", "Lipstick Set", "Sunscreen", "Hair Dryer"],
        "words": ["skin", "hydrating", "natural", "gentle", "glow", "dermatologist", "tested",
                  "vitamin", "daily", "care", "nourishing", "formula", "radiant"],
    },
    "Toys & Games": {
        "nouns": ["Puzzle", "Board Game", "Building Blocks", "RC Car", "Plush Toy", "Art Kit"],
        "words": ["kids", "fun", "creative", "learning", "play", "family", "educational",
                  "colorful", "pieces", "ages", "imagination", "safe", "interactive"],
    },
    "Outdoors": {
        "nouns": ["Tent", "Hiking Backpack", "Sleeping Bag", "Camping Lantern", "Trekking Poles", "Cooler Box"],
        "words": ["camping", "hiking", "waterproof", "rugged", "adventure", "trail", "outdoor",
                  "insulated", "packable", "weather", "durable", "mountain", "travel"],
    },
}
ADJECTIVES = ["Pro", "Ultra", "Classic", "Essential", "Premium", "Eco", "Max", "Lite", "Elite", "Prime"]


def generate_products(rng: np.random.Generator, per_category: int = 30) -> pd.DataFrame:
    rows, pid = [], 1
    for category, spec in CATEGORIES.items():
        for _ in range(per_category):
            noun = rng.choice(spec["nouns"])
            adj = rng.choice(ADJECTIVES)
            words = rng.choice(spec["words"], size=8, replace=False)
            description = (
                f"{adj} {noun.lower()} for {category.lower()} lovers. "
                f"Features: {', '.join(words[:4])}. Ideal for {', '.join(words[4:])} needs."
            )
            rows.append({
                "product_id": pid,
                "product_name": f"{adj} {noun} {rng.integers(100, 999)}",
                "category": category,
                "description": description,
                "price": round(float(rng.uniform(5, 250)), 2),
            })
            pid += 1
    return pd.DataFrame(rows)


def generate_interactions(rng: np.random.Generator, products: pd.DataFrame, n_users: int = 300) -> pd.DataFrame:
    categories = products["category"].unique()
    cat_arr = products["category"].to_numpy()
    rows = []
    for user_id in range(1, n_users + 1):
        favourites = rng.choice(categories, size=rng.integers(1, 3), replace=False)
        is_fav = np.isin(cat_arr, favourites)
        weights = np.where(is_fav, 4.0, 1.0)
        weights /= weights.sum()
        n = int(rng.integers(15, 40))
        chosen = rng.choice(len(products), size=n, replace=False, p=weights)
        for idx in chosen:
            fav = is_fav[idx]
            probs = [0.40, 0.25, 0.35] if fav else [0.80, 0.15, 0.05]
            itype = rng.choice(["click", "add_to_cart", "purchase"], p=probs)
            rating = np.nan
            if (itype == "purchase" and rng.random() < 0.9) or (itype == "add_to_cart" and rng.random() < 0.3):
                mean, sd = (4.2, 0.7) if fav else (2.8, 1.0)
                rating = float(np.clip(round(rng.normal(mean, sd)), 1, 5))
            rows.append({
                "user_id": user_id,
                "product_id": int(products.iloc[idx]["product_id"]),
                "rating": rating,
                "interaction_type": itype,
            })
    return pd.DataFrame(rows)


def main(seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    DATA_DIR.mkdir(exist_ok=True)
    products = generate_products(rng)
    interactions = generate_interactions(rng, products)
    products.to_csv(DATA_DIR / "products.csv", index=False)
    interactions.to_csv(DATA_DIR / "interactions.csv", index=False)
    print(f"Saved {len(products)} products and {len(interactions)} interactions to {DATA_DIR}")


if __name__ == "__main__":
    main()
