"""Baselines. Each is fit only on the rows it is given (already purged) and predicts from point-in-time columns."""
import numpy as np
from sklearn.linear_model import LogisticRegression


class Persistence:
    """P = 1 if an alert is active at t, else 0. Degenerate in view (b), where Kyiv is clear by definition."""

    name = "persistence"

    def fit(self, frame, y):
        return self

    def predict(self, frame):
        return (~frame["clear"].values).astype(float)


class Climatology:
    """Positive rate per calendar cell (Kyiv local hour, optionally day of week), shrunk towards the global rate."""

    def __init__(self, by=("hour",), prior_strength=20.0):
        self.by = list(by)
        self.m = prior_strength
        self.name = "climatology_" + "_".join(by)

    def fit(self, frame, y):
        y = np.asarray(y, dtype=float)
        self.global_ = y.mean()
        grouped = frame.assign(_y=y).groupby(self.by)["_y"].agg(["sum", "count"])
        self.table_ = ((grouped["sum"] + self.m * self.global_) / (grouped["count"] + self.m)).to_dict()
        return self

    def predict(self, frame):
        keys = list(zip(*[frame[c].values for c in self.by])) if len(self.by) > 1 else frame[self.by[0]].values
        return np.array([self.table_.get(k, self.global_) for k in keys], dtype=float)


class RecentActivity:
    """Logistic regression on how much alert time and how many alerts occurred in the last 24h."""

    name = "recent_activity_24h"

    def fit(self, frame, y):
        self.model_ = LogisticRegression(C=1.0, max_iter=1000).fit(self._x(frame), np.asarray(y, dtype=int))
        return self

    def predict(self, frame):
        return self.model_.predict_proba(self._x(frame))[:, 1]

    @staticmethod
    def _x(frame):
        return np.column_stack([frame["frac_alert_24h"].values, np.log1p(frame["n_starts_24h"].values)])


def all_baselines():
    return [Persistence(), Climatology(("hour",)), Climatology(("hour", "dow")), RecentActivity()]
