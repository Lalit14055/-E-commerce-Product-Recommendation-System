"""Streamlit dashboard for the E-commerce Product Recommendation System.

Run with:  streamlit run app.py
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.collaborative_filtering import SVDRecommender, UserBasedCF
from src.content_based import ContentBasedRecommender
from src.data_preprocessing import (
    DATA_DIR,
    build_user_item_matrix,
    load_and_prepare,
    train_test_split_interactions,
)
from src.evaluation import evaluate_ranking
from src.hybrid_recommender import HybridRecommender, popular_products

st.set_page_config(page_title="Product Recommender", page_icon="🛍️", layout="wide")


@st.cache_resource(show_spinner="Training recommendation models...")
def load_system():
    if not (DATA_DIR / "products.csv").exists():
        st.error("Dataset not found. Run `python src/generate_sample_data.py` or add "
                 "`products.csv` and `interactions.csv` to the `data/` folder.")
        st.stop()
    products, interactions = load_and_prepare()
    pids = products["product_id"].tolist()
    users = sorted(interactions["user_id"].unique())
    matrix = build_user_item_matrix(interactions, product_ids=pids, user_ids=users)
    return {
        "products": products,
        "interactions": interactions,
        "matrix": matrix,
        "cf": UserBasedCF(n_neighbors=30).fit(matrix),
        "svd": SVDRecommender(n_factors=20).fit(matrix),
        "content": ContentBasedRecommender().fit(products),
    }


def show_products(scores: pd.Series, products: pd.DataFrame, score_label: str = "score"):
    df = products.set_index("product_id").loc[scores.index, ["product_name", "category", "price", "description"]]
    df.insert(0, score_label, scores.round(3).values)
    df["price"] = df["price"].map("${:,.2f}".format)
    st.dataframe(df.reset_index(), width="stretch", hide_index=True)


sysm = load_system()
products, interactions, matrix = sysm["products"], sysm["interactions"], sysm["matrix"]
name_of = products.set_index("product_id")["product_name"]

st.title("🛍️ E-commerce Product Recommendation System")
page = st.sidebar.radio(
    "Explore", ["🏠 Overview", "👤 Recommendations for a User", "🔎 Similar Products", "🔥 Popular Products", "📈 Model Evaluation"]
)

# ------------------------------------------------------------------ Overview
if page == "🏠 Overview":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Users", f"{interactions['user_id'].nunique():,}")
    c2.metric("Products", f"{len(products):,}")
    c3.metric("Interactions", f"{len(interactions):,}")
    sparsity = 1 - (matrix.to_numpy() > 0).mean()
    c4.metric("Matrix sparsity", f"{sparsity:.1%}")

    left, right = st.columns(2)
    cat_counts = interactions.merge(products[["product_id", "category"]], on="product_id")["category"].value_counts()
    left.plotly_chart(
        px.bar(cat_counts, title="Interactions by category", labels={"value": "Interactions", "category": ""}),
        width="stretch",
    )
    type_counts = interactions["interaction_type"].value_counts()
    right.plotly_chart(px.pie(values=type_counts.values, names=type_counts.index, title="Interaction types"),
                       width="stretch")
    st.plotly_chart(px.histogram(interactions, x="score", nbins=5, title="Distribution of preference scores"),
                    width="stretch")

# ---------------------------------------------------- Recommendations for user
elif page == "👤 Recommendations for a User":
    col1, col2, col3 = st.columns(3)
    user_id = col1.selectbox("User ID", list(matrix.index))
    k = col2.slider("Number of recommendations", 3, 20, 10)
    alpha = col3.slider("Collaborative weight (α)", 0.0, 1.0, 0.7, 0.05,
                        help="1.0 = purely collaborative, 0.0 = purely content-based")
    cf_choice = st.radio("Collaborative model", ["Matrix factorization (SVD)", "User-based CF"], horizontal=True)
    cf_model = sysm["svd"] if cf_choice.startswith("Matrix") else sysm["cf"]
    hybrid = HybridRecommender(cf_model, sysm["content"], interactions, alpha=alpha)

    st.subheader(f"Top {k} hybrid recommendations for user {user_id}")
    show_products(hybrid.recommend(user_id, k), products, "hybrid score")

    with st.expander("What has this user interacted with?"):
        hist = interactions[interactions["user_id"] == user_id].sort_values("score", ascending=False)
        hist = hist.merge(products[["product_id", "product_name", "category"]], on="product_id")
        st.dataframe(hist[["product_name", "category", "interaction_type", "rating", "score"]],
                     width="stretch", hide_index=True)

# ------------------------------------------------------------ Similar products
elif page == "🔎 Similar Products":
    product_id = st.selectbox("Choose a product", products["product_id"],
                              format_func=lambda p: f"{p} – {name_of[p]}")
    k = st.slider("Number of similar products", 3, 20, 8)
    st.caption(products.set_index("product_id").loc[product_id, "description"])
    st.subheader("Similar products (TF-IDF + cosine similarity)")
    show_products(sysm["content"].similar_products(product_id, k), products, "similarity")

# ------------------------------------------------------------ Popular products
elif page == "🔥 Popular Products":
    st.write("Best option for **new users** with no history (cold start).")
    cats = ["All"] + sorted(products["category"].unique())
    cat = st.selectbox("Category", cats)
    k = st.slider("How many", 3, 20, 10)
    pop = popular_products(interactions, k=len(products))
    if cat != "All":
        pop = pop[pop.index.isin(products.loc[products["category"] == cat, "product_id"])]
    show_products(pop.head(k), products, "popularity score")

# ---------------------------------------------------------- Model evaluation
else:
    k = st.slider("K", 3, 20, 10)
    if st.button("Run evaluation"):
        train, test = train_test_split_interactions(interactions)
        pids = products["product_id"].tolist()
        m = build_user_item_matrix(train, product_ids=pids, user_ids=list(matrix.index))
        ucf, svd = UserBasedCF().fit(m), SVDRecommender().fit(m)
        cb = ContentBasedRecommender().fit(products)
        hyb = HybridRecommender(ucf, cb, train, alpha=0.7)
        models = {
            "Popularity": lambda u, kk: popular_products(train, kk),
            "User-based CF": ucf.recommend,
            "SVD": svd.recommend,
            "Content-based": lambda u, kk: cb.recommend(m.loc[u], kk),
            "Hybrid (User CF + content)": hyb.recommend,
        }
        with st.spinner("Evaluating..."):
            results = pd.DataFrame({n: evaluate_ranking(fn, test, k) for n, fn in models.items()}).T
        st.dataframe(results.style.format("{:.3f}").highlight_max(axis=0), width="stretch")
        st.plotly_chart(px.bar(results.reset_index().melt(id_vars="index"), x="index", y="value",
                               color="variable", barmode="group", labels={"index": "", "value": "score"}),
                        width="stretch")
