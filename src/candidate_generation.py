import pandas as pd


def generate_candidates(user_id: int, txn: pd.DataFrame, pf_model, als_model, n_each: int = 50) -> set[int]:
    """
    Shortlist of plausible next-basket products for one user: everything
    they've bought before, plus each model's top-N picks. Scoring all
    ~50k products per user for the ranking step below would be wasteful
    and mostly pointless — the vast majority of products have no realistic
    chance of appearing in any given user's next basket, so candidate
    generation's job is narrowing the field to a set worth actually
    scoring, not doing the ranking itself.

    Reuses the Phase 5 models rather than building a separate candidate
    heuristic from scratch — Personalized Frequency and ALS already do a
    reasonable job of surfacing relevant products, so their outputs are
    a sensible starting shortlist for the ranking model to refine.
    """
    prior_txn = txn[(txn["eval_set"] == "prior") & (txn["user_id"] == user_id)]
    purchased = set(prior_txn["product_id"].unique())

    pf_candidates = set(pf_model.recommend(user_id)[:n_each])
    als_candidates = set(als_model.recommend(user_id, n=n_each))

    return purchased | pf_candidates | als_candidates


def generate_candidates_bulk(eval_users: pd.DataFrame, txn: pd.DataFrame,
                              pf_model, als_model, n_each: int = 50) -> dict[int, set[int]]:
    """
    Same as generate_candidates, but avoids re-filtering the full prior
    transaction table once per user — that's the expensive part at scale,
    so purchase history is grouped once up front and looked up per user
    from there instead.
    """
    prior_txn = txn[txn["eval_set"] == "prior"]
    purchase_history = prior_txn.groupby("user_id")["product_id"].apply(set).to_dict()

    candidates = {}
    for user_id in eval_users["user_id"]:
        purchased = purchase_history.get(user_id, set())
        pf_candidates = set(pf_model.recommend(user_id)[:n_each])
        als_candidates = set(als_model.recommend(user_id, n=n_each))
        candidates[user_id] = purchased | pf_candidates | als_candidates

    return candidates
