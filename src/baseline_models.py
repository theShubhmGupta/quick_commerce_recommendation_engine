import pandas as pd
import numpy as np


def get_eval_users(txn: pd.DataFrame, sample_n: int | None = None, random_state: int = 42) -> pd.DataFrame:
    """
    Ground truth for evaluation: each user's actual train-order basket.

    Only users assigned to `eval_set == 'train'` have a labeled next basket
    available for offline evaluation. Users assigned to `eval_set == 'test'`
    correspond to Kaggle's holdout set, where the next basket is not observable,
    so they are excluded from offline evaluation.
    """
    truth = (
        txn[txn["eval_set"] == "train"]
        .groupby("user_id")["product_id"]
        .apply(set)
        .reset_index()
        .rename(columns={"product_id": "true_products"})
    )
    if sample_n is not None:
        truth = truth.sample(n=sample_n, random_state=random_state).reset_index(drop=True)
    return truth


def precision_recall_at_k(recommended: list[int], relevant: set[int], k: int) -> tuple[float, float]:
    """Standard Precision@K / Recall@K for one user given their ranked recommendation list."""
    top_k = recommended[:k]
    if len(top_k) == 0 or len(relevant) == 0:
        return 0.0, 0.0
    hits = len(set(top_k) & relevant)
    precision = hits / len(top_k)
    recall = hits / len(relevant)
    return precision, recall


def evaluate_model(recommend_fn, eval_users: pd.DataFrame, k: int = 10) -> dict:
    """
    Runs recommend_fn(user_id) -> ranked list of product_ids for every user
    in eval_users, scores against their actual train basket, and averages.
    Same harness gets reused for every baseline below, and later for CF/hybrid,
    so results are directly comparable across models.
    """
    precisions, recalls = [], []
    for _, row in eval_users.iterrows():
        recs = recommend_fn(row["user_id"])
        p, r = precision_recall_at_k(recs, row["true_products"], k)
        precisions.append(p)
        recalls.append(r)

    return {
        "k": k,
        "n_users_evaluated": len(eval_users),
        "precision_at_k": float(np.mean(precisions)),
        "recall_at_k": float(np.mean(recalls)),
    }


class PopularityModel:
    """
    Simplest possible baseline: recommend the same globally most-purchased
    products to everyone, regardless of who they are. No personalization at
    all — this is the floor every other model needs to beat, and it's also
    the fallback for cold-start users with little or no order history.
    """

    def __init__(self, top_n: int = 50):
        self.top_n = top_n
        self.ranked_products_: list[int] = []

    def fit(self, txn: pd.DataFrame) -> "PopularityModel":
        prior_txn = txn[txn["eval_set"] == "prior"]
        counts = prior_txn["product_id"].value_counts()
        self.ranked_products_ = counts.head(self.top_n).index.tolist()
        return self

    def recommend(self, user_id: int) -> list[int]:
        # ignores user_id entirely — same list for everyone
        return self.ranked_products_


class ReorderPopularityModel:
    """
    Rank products using a combination of purchase volume and reorder rate, subject
    to a minimum purchase threshold. Reorder rate alone can be misleading for
    low-volume products, where a small number of repeat purchases can produce an
    artificially high rate. Requiring both sufficient purchase volume and strong
    reorder behavior prioritizes products with consistent demand while reducing
    noise from sparsely purchased items.

    The purchase threshold should be derived from the observed distribution of
    product purchase counts rather than hardcoded, so that it reflects the actual
    catalog distribution.
    """

    def __init__(self, top_n: int = 50, min_purchases: int = 100):
        self.top_n = top_n
        self.min_purchases = min_purchases
        self.ranked_products_: list[int] = []

    def fit(self, txn: pd.DataFrame) -> "ReorderPopularityModel":
        prior_txn = txn[txn["eval_set"] == "prior"]
        stats = prior_txn.groupby("product_id").agg(
            n_purchases=("order_id", "count"),
            reorder_rate=("reordered", "mean"),
        )
        stats = stats[stats["n_purchases"] >= self.min_purchases]
        stats["reorder_score"] = stats["n_purchases"] * stats["reorder_rate"]
        self.ranked_products_ = (
            stats.sort_values("reorder_score", ascending=False).head(self.top_n).index.tolist()
        )
        return self

    def recommend(self, user_id: int) -> list[int]:
        return self.ranked_products_


class PersonalizedFrequencyModel:
    """
    First genuinely personalized baseline: recommend each user their own
    most-frequently-bought products, ranked by purchase count (ties broken
    by most recent order_number so a recent one-off doesn't outrank a
    long-standing habit). Falls back to the global popularity list for
    users with no prior history — new users can't have a personal history
    to rank, so the global ranking is the only sane fallback.
    """

    def __init__(self, top_n: int = 50):
        self.top_n = top_n
        self.user_products_: dict[int, list[int]] = {}
        self.fallback_ = PopularityModel(top_n=top_n)

    def fit(self, txn: pd.DataFrame) -> "PersonalizedFrequencyModel":
        self.fallback_.fit(txn)

        prior_txn = txn[txn["eval_set"] == "prior"]
        stats = (
            prior_txn.groupby(["user_id", "product_id"])
            .agg(n_purchases=("order_id", "count"), last_order_number=("order_number", "max"))
            .reset_index()
        )
        stats = stats.sort_values(
            ["user_id", "n_purchases", "last_order_number"], ascending=[True, False, False]
        )

        self.user_products_ = (
            stats.groupby("user_id")["product_id"]
            .apply(lambda s: s.head(self.top_n).tolist())
            .to_dict()
        )
        return self

    def recommend(self, user_id: int) -> list[int]:
        return self.user_products_.get(user_id, self.fallback_.recommend(user_id))
