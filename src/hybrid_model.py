import pandas as pd


class HybridModel:
    """
    Combines the three models into one ranked list, weighted by
    what each one has actually been shown to be good at:

    - Personalized Frequency is the base ranking. At 0.2838 precision@10
      it's roughly 3x ALS and 100x content-based on this dataset, because
      it directly encodes the reorder pattern that dominates most baskets
      (0.59 overall reorder rate from the EDA). There's no reason to
      dilute that signal by blending it proportionally with two much
      weaker models.
    - ALS and content-based contribute as score boosts on top of that
      base ranking, rather than as equal partners in a blended score.
      Their measured overlap with each other is only 0.0246 Jaccard, so
      they're picking up largely different products — each is adding
      coverage the base model and the other one both miss, which is
      exactly what a boost should do.

    A product's final score is its Personalized Frequency rank position
    (inverted so higher is better), plus a fixed boost for every
    additional model that also recommended it. Products the base model
    didn't surface at all but ALS or content-based did are appended
    after the boosted base list, not scored into it, since there's no
    frequency signal to rank them against.
    """

    def __init__(self, top_n: int = 50, als_boost: float = 0.15, cb_boost: float = 0.15):
        self.top_n = top_n
        self.als_boost = als_boost
        self.cb_boost = cb_boost
        self.pf_model = None
        self.als_model = None
        self.cb_model = None

    def fit(self, pf_model, als_model, cb_model) -> "HybridModel":
        # each sub-model is fit independently beforehand
        # the hybrid just holds references and combines
        # their outputs at recommend time
        self.pf_model = pf_model
        self.als_model = als_model
        self.cb_model = cb_model
        return self

    def recommend(self, user_id: int) -> list[int]:
        base = self.pf_model.recommend(user_id)
        als_recs = set(self.als_model.recommend(user_id, n=self.top_n))
        cb_recs = set(self.cb_model.recommend(user_id))

        n_base = len(base)
        scored = {}
        for rank, product_id in enumerate(base):
            score = (n_base - rank) / n_base if n_base else 0
            if product_id in als_recs:
                score += self.als_boost
            if product_id in cb_recs:
                score += self.cb_boost
            scored[product_id] = score

        ranked_base = sorted(scored, key=scored.get, reverse=True)

        # products the base model never saw at all get appended after it,
        # in the order the discovery models found them — no frequency
        # signal exists to rank them against the base list
        extras = [
            pid for pid in list(als_recs) + list(cb_recs)
            if pid not in scored
        ]

        return (ranked_base + extras)[: self.top_n]
