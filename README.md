# 🛍️ E-commerce Product Recommendation System

## 📌 Project Overview

The **E-commerce Product Recommendation System** is a machine learning project that provides personalized product recommendations based on user preferences, purchase history, ratings, and interactions.

The system analyzes user-item interaction data to identify similar users and products. It uses collaborative filtering and matrix factorization to recommend relevant products. An optional NLP-based content filtering component analyzes product descriptions to improve recommendations, especially when user interaction data is limited.

## 🎯 Objectives

- Recommend personalized products to users.
- Analyze user-item interactions such as ratings, clicks, and purchases.
- Implement collaborative filtering and matrix factorization.
- Use NLP to understand product descriptions.
- Develop a hybrid recommendation model combining collaborative and content-based filtering.
- Evaluate recommendation quality using appropriate metrics.

## ✨ Key Features

- **Personalized Recommendations:** Suggest products based on individual user preferences.
- **Collaborative Filtering:** Identify patterns among users with similar interests.
- **Matrix Factorization:** Learn latent user and product preferences.
- **Content-Based Filtering:** Recommend products with descriptions similar to previously liked products.
- **Hybrid Recommendation:** Combine collaborative and content-based recommendations.
- **Popular Products:** Recommend highly rated or frequently purchased products to new users.
- **Interactive Dashboard:** Display recommended products and relevant product information.

## 🧠 Recommendation Techniques

### 1. Collaborative Filtering
User-based k-nearest-neighbour filtering with cosine similarity (`src/collaborative_filtering.py`, `UserBasedCF`).

### 2. Matrix Factorization
Truncated SVD on a mean-centred user-item matrix to learn latent user and product vectors (`SVDRecommender`).

### 3. Content-Based Filtering with NLP
TF-IDF vectorization of product titles, descriptions and categories plus cosine similarity (`src/content_based.py`).

### 4. Hybrid Recommendation Model
`score = α · CF + (1 − α) · content`, each min-max normalised. Brand-new users fall back to Bayesian-weighted popular products (`src/hybrid_recommender.py`).

## 🛠️ Tech Stack

- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy, SciPy
- **Machine Learning:** Scikit-learn
- **NLP:** TF-IDF, Cosine Similarity
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Web Interface:** Streamlit
- **Development Tools:** Jupyter Notebook, VS Code, Git, GitHub

## 📂 Project Structure

```text
ecommerce-recommender-system/
│
├── data/
│   ├── products.csv
│   └── interactions.csv
│
├── notebooks/
│   └── exploratory_data_analysis.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── collaborative_filtering.py
│   ├── content_based.py
│   ├── hybrid_recommender.py
│   ├── evaluation.py
│   └── generate_sample_data.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 📊 Dataset

A **synthetic sample dataset** (240 products, 300 users, ~8,000 interactions) is included in `data/` so the project runs immediately. Regenerate it with `python src/generate_sample_data.py`.

**Product dataset (`products.csv`):** `product_id`, `product_name`, `category`, `description`, `price`

**Interaction dataset (`interactions.csv`):** `user_id`, `product_id`, `rating`, `interaction_type`

Interaction types: `click`, `add_to_cart`, `purchase`. When `rating` is empty, an implicit score is used (click = 1, add_to_cart = 3, purchase = 4).

To use real data (e.g. the Amazon Reviews dataset or a Kaggle e-commerce dataset), convert it to this column layout and overwrite the two CSV files.

## ⚙️ Workflow

1. **Data Collection:** Obtain product information and user-item interaction records.
2. **Data Preprocessing:** Handle missing values, remove duplicates, and prepare interaction data.
3. **Exploratory Data Analysis:** Analyze popular products, user activity, ratings, and product categories.
4. **Model Development:** Implement collaborative filtering and matrix factorization.
5. **NLP Processing:** Convert product descriptions into TF-IDF vectors and calculate cosine similarity.
6. **Hybrid Model:** Combine recommendation scores from collaborative and content-based models.
7. **Evaluation:** Measure recommendation performance using Precision@K, Recall@K, and NDCG@K.
8. **Deployment:** Build a Streamlit application where users can select a user ID or product and explore recommendations.

## 📈 Evaluation Metrics

- **Precision@K:** Measures how many recommended items are relevant among the top K recommendations.
- **Recall@K:** Measures how many relevant items are retrieved in the top K recommendations.
- **NDCG@K:** Evaluates the ranking quality of recommended products.
- **RMSE:** Measures rating prediction error when explicit ratings are available.

Evaluation uses a per-user hold-out split; a held-out item counts as relevant when its score is ≥ 3. See the notebook for a model-by-model comparison.

## 🚀 Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/Lalit14055/-E-commerce-Product-Recommendation-System
cd -E-commerce-Product-Recommendation-System
### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On macOS / Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

Ensure the dataset is available in the expected directory before starting the application (`python src/generate_sample_data.py` creates one).

### Optional: explore the notebook

```bash
jupyter notebook notebooks/exploratory_data_analysis.ipynb
```

## 🔮 Future Enhancements

- Integrate real-time user interactions.
- Add deep learning-based recommendation models.
- Use product images and multimodal embeddings.
- Introduce session-based recommendations.
- Add cold-start strategies for new users and products.
- Deploy the application to a cloud platform.
- Build an API using FastAPI for integration with an e-commerce website.

## 🎓 Learning Outcomes

- Practical implementation of recommendation algorithms.
- Understanding user-item interaction matrices and latent factors.
- Applying NLP to product descriptions.
- Evaluating personalized ranking systems.
- Developing and deploying an end-to-end machine learning application.

## 👨‍💻 Author

Lalit Patil 

GitHub: https://github.com/Lalit14055

---

*This project is developed for educational and portfolio purposes, demonstrating machine learning, recommendation algorithms, NLP, and application development.*
