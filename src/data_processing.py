from pathlib import Path
import pandas as pd

# src/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

#Default pandas dtypes (int64, object) can be unnecessarily memory-intensive for these tables. 
#Since the integer values do not require the full int64 range, and string/object columns can consume significant memory at 34M rows, 
#explicitly choosing more appropriate dtypes can substantially reduce the overall memory footprint and improve processing performance.

DTYPES = {
    "orders": {
        "order_id": "int32",
        "user_id": "int32",
        "eval_set": "category",
        "order_number": "int16",
        "order_dow": "int8",
        "order_hour_of_day": "int8",
        # days_since_prior_order has NaNs (first order per user), so it can't be int
        "days_since_prior_order": "float32",
    },
    "order_products": {
        "order_id": "int32",
        "product_id": "int32",
        "add_to_cart_order": "int16",
        "reordered": "int8",
    },
    "products": {
        "product_id": "int32",
        "aisle_id": "int16",
        "department_id": "int8",
    },
    "aisles": {"aisle_id": "int16"},
    "departments": {"department_id": "int8"},
}


def load_raw_data(raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    """Load the six raw Instacart CSVs into a dict keyed by table name,
with memory-efficient dtypes to reduce the overall memory footprint."""
    files = {
        "orders": ("orders.csv", DTYPES["orders"]),
        "prior": ("order_products__prior.csv", DTYPES["order_products"]),
        "train": ("order_products__train.csv", DTYPES["order_products"]),
        "products": ("products.csv", DTYPES["products"]),
        "aisles": ("aisles.csv", DTYPES["aisles"]),
        "departments": ("departments.csv", DTYPES["departments"]),
    }
    data = {}
    for key, (filename, dtype) in files.items():
        path = raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"{filename} not found at {path} — check data/raw/")
        data[key] = pd.read_csv(path, dtype=dtype)
    return data


def validate_data(data: dict[str, pd.DataFrame]) -> dict:
    """
    Referential integrity + sanity checks on the raw tables. Catches the kind
    of thing that quietly breaks downstream joins if it slips through —
    orphaned order_ids, product_ids with no matching product row, etc.
    """
    orders, prior, train = data["orders"], data["prior"], data["train"]
    products, aisles, departments = data["products"], data["aisles"], data["departments"]

    report = {}

    report["shapes"] = {k: v.shape for k, v in data.items()}

    report["nulls"] = {
        "orders": orders.isnull().sum().to_dict(),
        "prior": prior.isnull().sum().to_dict(),
        "train": train.isnull().sum().to_dict(),
        "products": products.isnull().sum().to_dict(),
    }

    report["eval_set_counts"] = orders["eval_set"].value_counts().to_dict()

    report["n_unique_users"] = int(orders["user_id"].nunique())
    report["order_number_range"] = [
        int(orders["order_number"].min()),
        int(orders["order_number"].max()),
    ]
    report["duplicate_order_ids"] = int(orders["order_id"].duplicated().sum())

    # every order_id referenced in order_products should exist in orders.csv
    orders_ids = set(orders["order_id"])
    report["prior_order_ids_missing_from_orders"] = len(set(prior["order_id"]) - orders_ids)
    report["train_order_ids_missing_from_orders"] = len(set(train["order_id"]) - orders_ids)

    # cross-check eval_set counts against the actual order_products files —
    # these should line up exactly if the data is well-formed
    report["orders_marked_prior"] = int((orders.eval_set == "prior").sum())
    report["unique_orders_in_prior_file"] = int(prior["order_id"].nunique())
    report["orders_marked_train"] = int((orders.eval_set == "train").sum())
    report["unique_orders_in_train_file"] = int(train["order_id"].nunique())
    report["orders_marked_test"] = int((orders.eval_set == "test").sum())

    # same idea for product_id
    product_ids = set(products["product_id"])
    report["prior_product_ids_missing_from_products"] = len(set(prior["product_id"]) - product_ids)
    report["train_product_ids_missing_from_products"] = len(set(train["product_id"]) - product_ids)

    report["aisle_ids_in_products_not_in_aisles"] = len(
        set(products["aisle_id"]) - set(aisles["aisle_id"])
    )
    report["dept_ids_in_products_not_in_departments"] = len(
        set(products["department_id"]) - set(departments["department_id"])
    )

    report["prior_reordered_value_counts"] = prior["reordered"].value_counts().to_dict()
    report["train_reordered_value_counts"] = train["reordered"].value_counts().to_dict()

    report["add_to_cart_order_min"] = int(prior["add_to_cart_order"].min())

    # days_since_prior_order is null exactly when it's a user's first order —
    # there's nothing to measure "days since" against. Anything else is a red flag.
    null_dspo = orders[orders["days_since_prior_order"].isnull()]
    report["days_since_prior_null_count"] = len(null_dspo)
    report["days_since_prior_null_all_are_first_order"] = bool(
        (null_dspo["order_number"] == 1).all()
    )

    # a user's most recent order should always be their train or test order —
    # if it's still "prior", the split is broken for that user
    last_orders = orders.sort_values("order_number").groupby("user_id").tail(1)
    report["last_order_eval_set_breakdown"] = last_orders["eval_set"].value_counts().to_dict()

    return report


def save_processed(df: pd.DataFrame, name: str, processed_dir: Path = PROCESSED_DIR) -> Path:
    """Write a processed DataFrame to data/processed/ as parquet."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / f"{name}.parquet"
    df.to_parquet(out_path, index=False)
    return out_path
