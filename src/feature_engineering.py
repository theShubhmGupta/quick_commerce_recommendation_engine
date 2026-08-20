import pandas as pd
from eda import customer_behavior, product_behavior


def build_training_table(candidates: dict[int, set[int]], txn: pd.DataFrame,
                          eval_users: pd.DataFrame) -> pd.DataFrame:
    """
    Flattens the per-user candidate sets into one row per (user, product)
    pair with a binary label: 1 if that product is actually in the user's
    train basket, 0 otherwise. This is the table the ranking model trains
    on directly.

    Only eval_users are included, since only they have a labeled train
    basket to check candidates against — the same restriction the Phase 4
    evaluation harness uses, for the same reason.
    """
    truth_lookup = eval_users.set_index("user_id")["true_products"].to_dict()

    rows = []
    for user_id, candidate_set in candidates.items():
        true_products = truth_lookup.get(user_id, set())
        for product_id in candidate_set:
            rows.append({
                "user_id": user_id,
                "product_id": product_id,
                "label": 1 if product_id in true_products else 0,
            })

    return pd.DataFrame(rows)


def add_features(table: pd.DataFrame, txn: pd.DataFrame) -> pd.DataFrame:
    """
    Joins user-level, product-level, and user-product interaction features
    onto the (user, product, label) table. Every feature here is computed
    from eval_set == 'prior' only — never from 'train' — since 'train' is
    where the label comes from, and letting it leak into the features
    would make the model's offline score meaningless.
    """
    prior_txn = txn[txn["eval_set"] == "prior"]

    user_features = customer_behavior(txn)
    product_features = product_behavior(txn)[
        ["product_id", "n_purchases", "reorder_rate", "avg_add_to_cart_position"]
    ].rename(columns={"reorder_rate": "product_reorder_rate"})

    up_features = (
        prior_txn.groupby(["user_id", "product_id"])
        .agg(
            up_purchase_count=("order_id", "count"),
            up_last_order_number=("order_number", "max"),
            up_avg_cart_position=("add_to_cart_order", "mean"),
        )
        .reset_index()
    )

    max_order_number = (
        prior_txn.groupby("user_id")["order_number"].max().rename("user_max_order_number")
    )

    result = table.merge(user_features, on="user_id", how="left")
    result = result.merge(product_features, on="product_id", how="left")
    result = result.merge(up_features, on=["user_id", "product_id"], how="left")
    result = result.merge(max_order_number, on="user_id", how="left")

    # every numeric feature is filled to 0 rather than dropped. A missing
    # user_features/product_features join means a candidate or user with
    # no prior-set history at all — rare (e.g. a user whose prior orders
    # weren't captured in this data cut, or a candidate product nobody in
    # the prior set bought), but real, and dropping those rows would
    # silently shrink the candidate set for exactly the users a
    # production system would still need to serve something to.
    feature_cols = [
        "n_orders", "avg_basket_size", "avg_days_between_orders", "reorder_rate",
        "n_purchases", "product_reorder_rate", "avg_add_to_cart_position",
        "up_purchase_count", "up_last_order_number", "up_avg_cart_position",
        "user_max_order_number",
    ]
    result[feature_cols] = result[feature_cols].fillna(0)

    # orders since this product was last bought — meaningful only for
    # candidates the user has actually purchased before; for a candidate
    # never bought there's nothing to measure "since" from, and
    # up_purchase_count == 0 already flags that case, so this is set to 0
    # rather than left to compute a misleading large gap
    result["orders_since_last_purchase"] = 0.0
    ever_purchased = result["up_purchase_count"] > 0
    result.loc[ever_purchased, "orders_since_last_purchase"] = (
        result.loc[ever_purchased, "user_max_order_number"]
        - result.loc[ever_purchased, "up_last_order_number"]
    )

    return result
