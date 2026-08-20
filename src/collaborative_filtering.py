import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def build_user_item_matrix(txn: pd.DataFrame):
    """
    Builds the sparse user-item matrix ALS trains on, restricted to prior
    purchases (the actual history available at prediction time). Matrix
    values are raw purchase counts, which implicit treats as confidence
    weights rather than ratings — a product bought 5 times by a user
    carries more confidence than one bought once, not a "higher score."

    user_id/product_id in the raw data aren't contiguous integers starting
    at 0, which is what a sparse matrix needs for its row/column indices,
    so this also returns the mappings between raw IDs and matrix positions.
    Every downstream lookup (recommend, evaluate) has to go through these.
    """
    prior_txn = txn[txn["eval_set"] == "prior"]

    user_ids = np.sort(prior_txn["user_id"].unique())
    product_ids = np.sort(prior_txn["product_id"].unique())

    user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    product_to_idx = {pid: i for i, pid in enumerate(product_ids)}
    idx_to_product = {i: pid for pid, i in product_to_idx.items()}

    counts = (
        prior_txn.groupby(["user_id", "product_id"])
        .size()
        .reset_index(name="count")
    )

    rows = counts["user_id"].map(user_to_idx).to_numpy()
    cols = counts["product_id"].map(product_to_idx).to_numpy()
    values = counts["count"].to_numpy(dtype=np.float32)

    matrix = csr_matrix(
        (values, (rows, cols)), shape=(len(user_ids), len(product_ids))
    )

    return matrix, user_to_idx, product_to_idx, idx_to_product


class ALSModel:
    """
    Implicit-feedback matrix factorization via implicit's ALS
    (Hu, Koren & Volinsky, 2008). There are no ratings in this data, only
    purchase counts, so this treats count as a confidence signal rather
    than a preference score — the library's fit() call expects exactly
    that framing (see build_user_item_matrix).

    filter_already_liked_items is set to False when generating
    recommendations, deliberately overriding the library default. The
    default filters out anything a user has already purchased, which
    would be the wrong call here: the overall reorder rate sits at 0.59,
    meaning most of what shows up in a user's next basket is something
    they've bought before. Filtering those out would actively work
    against the task this model is being evaluated on.
    """

    def __init__(self, factors: int = 50, regularization: float = 0.01,
                 alpha: float = 40.0, iterations: int = 15, random_state: int = 42):
        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha
        self.iterations = iterations
        self.random_state = random_state
        self.model = None
        self.user_item_matrix = None
        self.user_to_idx = {}
        self.product_to_idx = {}
        self.idx_to_product = {}

    def fit(self, txn: pd.DataFrame) -> "ALSModel":
        import implicit

        matrix, user_to_idx, product_to_idx, idx_to_product = build_user_item_matrix(txn)
        self.user_item_matrix = matrix
        self.user_to_idx = user_to_idx
        self.product_to_idx = product_to_idx
        self.idx_to_product = idx_to_product

        self.model = implicit.als.AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            alpha=self.alpha,
            iterations=self.iterations,
            random_state=self.random_state,
        )
        self.model.fit(self.user_item_matrix)
        return self

    def recommend(self, user_id: int, n: int = 50) -> list[int]:
        if user_id not in self.user_to_idx:
            return []

        user_idx = self.user_to_idx[user_id]
        user_row = self.user_item_matrix[user_idx]

        item_idxs, _scores = self.model.recommend(
            user_idx, user_row, N=n, filter_already_liked_items=False
        )

        return [self.idx_to_product[int(i)] for i in item_idxs]
