# Quick Commerce Recommendation Engine

Seven recommendation strategies, built and benchmarked head-to-head on 3.4M real grocery
orders — from a popularity baseline to collaborative filtering, content-based filtering, a
hybrid ensemble, and a supervised ranking model trained on engineered behavioral features.

**Best result: 29.5% Precision@10, a 4x lift over a non-personalized baseline.**

Dataset: [Instacart Online Grocery Basket Analysis](https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset/data) — 206K customers, 3.4M orders, 49.7K products.

---

## Results

| Model | Precision@10 | Recall@10 |
|---|---:|---:|
| **Ranking Model** (gradient-boosted, supervised) | **29.5%** | **35.0%** |
| Personalized Frequency | 28.4% | 33.0% |
| Hybrid (frequency + collaborative + content) | 27.6% | 33.0% |
| Popularity baseline | 7.3% | 7.0% |
| Reorder-weighted popularity | 7.2% | 7.0% |
| Collaborative Filtering (ALS, implicit feedback) | 6.6% | 9.8% |
| Content-Based (TF-IDF similarity) | 0.3% | 0.3% |

Every model is scored identically: given a customer's order history, rank the top 10 products
they're most likely to buy next, then check against their actual next order.

The gap between the top three models and the rest isn't a modeling artifact — it reflects a
real property of the data: 59% of all purchases are reorders. Models that exploit that
directly land in one tier; collaborative and content-based signals, built for discovering new
products rather than predicting repeats, land in another. That distinction, and what it implies
for how the two families of models should be deployed, is covered in depth in the [business
report](./Recommendation_Engine_Business_Report.pdf).

## Engineering Highlights

- **Full-scale data validation before any modeling** — referential integrity checks across
  all 3.4M orders (orphaned foreign keys, null patterns, train/test split consistency) rather
  than assuming a public dataset is clean.
- **Memory-conscious pipeline design** — dtype downcasting and sparse matrix construction
  throughout; the 206K × 49.7K user-item matrix is built directly from transaction triples,
  never through a dense intermediate.
- **Leakage-checked supervised setup** — candidate generation and feature engineering are
  strictly sourced from pre-purchase history; the train/validation split is by customer, not
  by row, so no customer's behavior leaks across the split.
- **Diagnosed and fixed a real modeling bug mid-project** — an early version of the
  reorder-weighted baseline collapsed to near-zero precision because a fixed purchase-count
  threshold let low-volume products dominate the ranking. Root-caused via the actual purchase-count
  distribution, fixed by deriving the threshold from data (99th percentile) instead of a
  guessed constant.
- **ALS tuned, not just run** — swept the confidence-weighting parameter (`alpha`) across three
  values before drawing conclusions, rather than reporting a single untuned run.
- **Every model shares one evaluation harness** — Precision@K / Recall@K computed the same
  way for all seven models, so results are directly comparable, not cherry-picked.

## Project Structure

```
quick-commerce-recommender/
│
├── data/
│   ├── raw/              # Instacart CSVs (not included — see Setup)
│   └── processed/        # Generated artifacts: transactions.parquet, model results
│
├── notebooks/
│   ├── 01_data_validation.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_baseline_models.ipynb          # Popularity, reorder-weighted, personalized frequency
│   ├── 04_collaborative_filtering.ipynb  # ALS, implicit feedback
│   ├── 05_content_based.ipynb            # TF-IDF product similarity
│   ├── 06_hybrid_model.ipynb
│   ├── 07_feature_engineering.ipynb      # Candidate generation + feature table
│   ├── 08_ranking_model.ipynb            # Gradient-boosted classifier
│   └── 09_evaluation.ipynb               # Consolidated leaderboard
│
├── src/
│   ├── data_processing.py
│   ├── eda.py
│   ├── baseline_models.py
│   ├── collaborative_filtering.py
│   ├── content_based.py
│   ├── hybrid_model.py
│   ├── candidate_generation.py
│   ├── feature_engineering.py
│   └── ranking_model.py
│
├── Recommendation_Engine_Business_Report.pdf
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Download the [dataset from Kaggle](https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset/data)
and place the six CSVs in `data/raw/`:

```
orders.csv, products.csv, aisles.csv, departments.csv,
order_products__prior.csv, order_products__train.csv
```

Run the notebooks in order, 01 → 09. Each stage persists its output to `data/processed/`, so
later notebooks load prior results rather than recomputing them.

---

## Author

**Shubham Gupta** 

[![GitHub](https://img.shields.io/badge/GitHub-theShubhmGupta-black?logo=github)](https://github.com/theShubhmGupta)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-theshubhamguptaa-blue?logo=linkedin)](https://linkedin.com/in/theshubhamguptaa)
[![Tableau](https://img.shields.io/badge/Tableau-Public-blue?logo=tableau)](https://public.tableau.com/app/profile/shubham.gupta2025)
---
