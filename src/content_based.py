import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_product_text(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combines product_name, aisle, and department into one text field per
    product. Name alone captures fine-grained product identity but misses
    category grouping; aisle/department alone is too coarse (134/21 total
    categories across ~50k products). Combining the three gives TF-IDF
    enough vocabulary to distinguish similar products while still picking
    up on category-level similarity.
    """
    products = data["products"].merge(
        data["aisles"], on="aisle_id", how="left"
    ).merge(
        data["departments"], on="department_id", how="left"
    )

    products["aisle"] = products["aisle"].fillna("")
    products["department"] = products["department"].fillna("")
    products["text"] = (
        products["product_name"] + " " + products["aisle"] + " " + products["department"]
    )

    return products[["product_id", "text"]]


class ContentBasedModel:
    """
    Recommends products by similarity to a user's purchase history, using
    TF-IDF over product name/aisle/department text rather than purchase
    patterns. A user's profile is the average TF-IDF vector of everything
    they've bought in the prior set; products are ranked by cosine
    similarity to that profile.

    Similarity is computed one user at a time against the full sparse
    product matrix, never as a materialized product-by-product matrix —
    at ~50k products that matrix would be roughly 2.5 billion cells dense,
    the same class of problem as building the CF matrix densely.
    """

    def __init__(self, top_n: int = 50):
        self.top_n = top_n
        self.tfidf_matrix = None
        self.product_ids: np.ndarray = np.array([])
        self.product_idx: dict[int, int] = {}
        self.user_purchases_: dict[int, list[int]] = {}
        self.vectorizer = TfidfVectorizer(stop_words="english")

    def fit(self, txn: pd.DataFrame, data: dict[str, pd.DataFrame]) -> "ContentBasedModel":
        product_text = build_product_text(data)

        self.tfidf_matrix = self.vectorizer.fit_transform(product_text["text"])
        self.product_ids = product_text["product_id"].to_numpy()
        self.product_idx = {pid: i for i, pid in enumerate(self.product_ids)}

        prior_txn = txn[txn["eval_set"] == "prior"]
        self.user_purchases_ = (
            prior_txn.groupby("user_id")["product_id"].apply(list).to_dict()
        )

        return self

    def _user_profile(self, user_id: int):
        purchased = self.user_purchases_.get(user_id, [])
        if not purchased:
            return None

        idxs = [self.product_idx[p] for p in purchased if p in self.product_idx]
        if not idxs:
            return None

        return self.tfidf_matrix[idxs].mean(axis=0)

    def recommend(self, user_id: int) -> list[int]:
        profile = self._user_profile(user_id)
        if profile is None:
            return []

        profile = np.asarray(profile)
        sims = cosine_similarity(profile, self.tfidf_matrix).flatten()

        # exclude items already purchased so this behaves as a discovery
        # signal on top of the reorder-focused models, not a duplicate
        # of Personalized Frequency
        purchased_idxs = {
            self.product_idx[p] for p in self.user_purchases_.get(user_id, [])
            if p in self.product_idx
        }
        for idx in purchased_idxs:
            sims[idx] = -1

        top_idxs = sims.argsort()[::-1][: self.top_n]
        return [int(self.product_ids[i]) for i in top_idxs]
