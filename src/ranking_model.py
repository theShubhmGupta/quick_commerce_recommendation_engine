import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

FEATURE_COLUMNS = [
    "n_orders", "avg_basket_size", "avg_days_between_orders", "reorder_rate",
    "n_purchases", "product_reorder_rate", "avg_add_to_cart_position",
    "up_purchase_count", "up_last_order_number", "up_avg_cart_position",
    "user_max_order_number", "orders_since_last_purchase",
]


def split_by_user(featured: pd.DataFrame, val_fraction: float = 0.2, random_state: int = 42):
    """
    Splits the training table by user_id, not by row. A row-level split
    would let the same user appear in both train and validation with
    different candidate products — the model could pick up user-specific
    patterns from that user's train rows and get credit for them on the
    same user's validation rows, inflating the validation score in a way
    that wouldn't hold up on genuinely unseen users.
    """
    rng = np.random.RandomState(random_state)
    unique_users = featured["user_id"].unique()
    n_val = int(len(unique_users) * val_fraction)
    val_users = set(rng.choice(unique_users, size=n_val, replace=False))

    val_mask = featured["user_id"].isin(val_users)
    return featured[~val_mask].copy(), featured[val_mask].copy()


class RankingModel:
    """
    Gradient-boosted classifier over the Phase 6 feature table, predicting
    the probability that a candidate product belongs in the user's next
    basket. HistGradientBoostingClassifier is used rather than a plain
    logistic model because the features here are mixed-scale counts and
    rates that don't need pre-scaling for a tree-based model, and rather
    than XGBoost/LightGBM to avoid an extra dependency for a model that's
    already in scikit-learn core.

    class_weight='balanced' accounts for the label skew found in feature
    engineering (roughly 93%/7% negative/positive on the full data) —
    without it the model could get a deceptively low loss by just
    predicting negative for everything.
    """

    def __init__(self, random_state: int = 42, **hgb_kwargs):
        self.random_state = random_state
        self.model = HistGradientBoostingClassifier(
            class_weight="balanced", random_state=random_state, **hgb_kwargs
        )
        self.candidates: dict[int, set[int]] = {}

    def fit(self, train_table: pd.DataFrame, candidates: dict[int, set[int]]) -> "RankingModel":
        X = train_table[FEATURE_COLUMNS]
        y = train_table["label"]
        self.model.fit(X, y)
        # candidates are what recommend() scores at inference time — the
        # ranking model only ever ranks within a user's existing shortlist,
        # it doesn't generate new candidates itself
        self.candidates = candidates
        return self

    def score_table(self, table: pd.DataFrame) -> pd.Series:
        X = table[FEATURE_COLUMNS]
        return pd.Series(self.model.predict_proba(X)[:, 1], index=table.index)

    def recommend(self, user_id: int, featured_lookup: dict[int, pd.DataFrame], n: int = 10) -> list[int]:
        """
        featured_lookup maps user_id -> that user's already-featured
        candidate rows (product_id + feature columns), so scoring doesn't
        require recomputing features per call — the notebook builds this
        once from the full featured table, grouped by user.
        """
        user_rows = featured_lookup.get(user_id)
        if user_rows is None or user_rows.empty:
            return []

        scores = self.score_table(user_rows)
        ranked = user_rows.loc[scores.sort_values(ascending=False).index, "product_id"]
        return ranked.head(n).tolist()
