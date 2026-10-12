"""Gradient boosting recipe: features + recency weighting + fixed hyperparameters.

Hyperparameters are fixed on purpose (no tuning in the Phase 3 round, see CLAUDE.md).
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from alertforecast import config
from alertforecast.frame import BASE_FEATURES, RATE_FEATURES

GBM_PARAMS = dict(
    learning_rate=0.05, max_iter=200, max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=1.0,
    early_stopping=False,  # sklearn's 'auto' would hold out a RANDOM 10% of rows, which is not time-aware
    random_state=0,
)


def recency_weights(t, ref_time_s, half_life_days):
    """weight = 0.5 ** (age_days / half_life); age counted back from the first evaluation origin. None -> all 1."""
    t = np.asarray(t, dtype="float64")
    if half_life_days is None:
        return np.ones(len(t))
    age_days = (ref_time_s - t) / config.DAY_S
    return 0.5 ** (age_days / half_life_days)


@dataclass(frozen=True)
class Recipe:
    """Everything that defines a model except its training data."""

    name: str
    features: Tuple[str, ...]
    half_life_days: Optional[float] = None

    def fit(self, frame, y, ref_time_s):
        w = recency_weights(frame["t"].values, ref_time_s, self.half_life_days)
        model = HistGradientBoostingClassifier(**GBM_PARAMS)
        model.fit(frame[list(self.features)], np.asarray(y, dtype=int), sample_weight=w)
        return Fitted(self, model)


@dataclass(frozen=True)
class Fitted:
    recipe: Recipe
    model: HistGradientBoostingClassifier

    def predict(self, frame):
        return self.model.predict_proba(frame[list(self.recipe.features)])[:, 1]


def stage_a_recipes():
    """The pre-registered Stage A grid: recency half-life x feature set. The first entry is the base configuration."""
    out = []
    for hl in (None, 180, 60, 30):
        for with_rate in (False, True):
            feats = tuple(BASE_FEATURES + (RATE_FEATURES if with_rate else []))
            name = f"hl{hl or 'none'}_{'rate' if with_rate else 'base'}"
            out.append(Recipe(name, feats, hl))
    return out
