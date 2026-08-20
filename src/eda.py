import pandas as pd
from data_processing import load_raw_data, save_processed


def build_transactions(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Flatten prior and train order-product data into a single row-per-(order, product)
    table, enriched with order metadata and product, aisle, and department names.
    This serves as the primary analytical table used throughout the EDA.
    """ 
    order_products = pd.concat([data["prior"], data["train"]], ignore_index=True)

    txn = order_products.merge(
        data["orders"][
            ["order_id", "user_id", "eval_set", "order_number",
             "order_dow", "order_hour_of_day", "days_since_prior_order"]
        ],
        on="order_id",
        how="left",
    )

    txn = txn.merge(
        data["products"][["product_id", "product_name", "aisle_id", "department_id"]],
        on="product_id",
        how="left",
    )

    txn = txn.merge(data["aisles"], on="aisle_id", how="left")
    txn = txn.merge(data["departments"], on="department_id", how="left")

    return txn


def customer_behavior(txn: pd.DataFrame) -> pd.DataFrame:
    """
    Per-user summary of purchase behavior. Restricted to eval_set == 'prior'
    since that's the actual purchase history — train/test orders are the
    thing we're trying to predict, not signal to describe a user by.
    """
    prior_txn = txn[txn["eval_set"] == "prior"]

    orders_per_user = prior_txn.groupby("user_id")["order_id"].nunique()
    basket_size = prior_txn.groupby(["user_id", "order_id"]).size().groupby("user_id").mean()
    avg_days_between = prior_txn.groupby("user_id")["days_since_prior_order"].mean()
    reorder_rate = prior_txn.groupby("user_id")["reordered"].mean()

    return pd.DataFrame({
        "n_orders": orders_per_user,
        "avg_basket_size": basket_size,
        "avg_days_between_orders": avg_days_between,
        "reorder_rate": reorder_rate,
    }).reset_index()


def product_behavior(txn: pd.DataFrame) -> pd.DataFrame:
    """Per-product purchase volume and reorder rate, sorted by popularity."""
    prior_txn = txn[txn["eval_set"] == "prior"]

    summary = prior_txn.groupby(["product_id", "product_name", "aisle", "department"]).agg(
        n_purchases=("order_id", "count"),
        n_unique_users=("user_id", "nunique"),
        reorder_rate=("reordered", "mean"),
        avg_add_to_cart_position=("add_to_cart_order", "mean"),
    ).reset_index()

    return summary.sort_values("n_purchases", ascending=False)


def reorder_behavior(txn: pd.DataFrame) -> dict:
    """Overall reorder rate plus breakdowns by department and by order number."""
    prior_txn = txn[txn["eval_set"] == "prior"]

    return {
        "overall_reorder_rate": float(prior_txn["reordered"].mean()),
        "reorder_rate_by_department": prior_txn.groupby("department")["reordered"]
            .mean().sort_values(ascending=False).to_dict(),
        # useful for checking whether reorder rate climbs as a user places
        # more orders, i.e. whether loyalty/habit builds over time
        "reorder_rate_by_order_number": prior_txn.groupby("order_number")["reordered"]
            .mean().to_dict(),
    }


def basket_analysis(txn: pd.DataFrame) -> dict:
    """Basket size distribution plus the departments/aisles driving most volume."""
    prior_txn = txn[txn["eval_set"] == "prior"]
    basket_sizes = prior_txn.groupby("order_id").size()

    return {
        "basket_size_summary": basket_sizes.describe().to_dict(),
        "top_departments_by_volume": prior_txn["department"].value_counts().head(10).to_dict(),
        "top_aisles_by_volume": prior_txn["aisle"].value_counts().head(10).to_dict(),
    }


def time_behavior(txn: pd.DataFrame) -> dict:
    """Order volume by day-of-week and hour-of-day — useful for spotting peak demand windows."""
    prior_txn = txn[txn["eval_set"] == "prior"]

    return {
        "orders_by_dow": prior_txn.groupby("order_dow")["order_id"].nunique().to_dict(),
        "orders_by_hour": prior_txn.groupby("order_hour_of_day")["order_id"].nunique().to_dict(),
    }
