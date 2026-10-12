"""Calibrators for the Stage B candidates. Each is fit on pooled out-of-time predictions only.

All are monotone, so they cannot change PR-AUC (checked in tests). Isotonic is excluded on purpose.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class NoCalibration:
    name = "none"

    def fit(self, p, y, weights=None):
        return self

    def predict(self, p):
        return np.asarray(p, dtype=float)


class InterceptShift:
    """p' = sigmoid(logit(p) + b), slope fixed at 1. b makes the mean prediction equal the (weighted) mean outcome."""

    name = "intercept_shift"

    def fit(self, p, y, weights=None):
        z, y = logit(p), np.asarray(y, dtype=float)
        w = np.ones(len(z)) if weights is None else np.asarray(weights, dtype=float)
        lo, hi = -10.0, 10.0
        for _ in range(80):  # bisection on a monotone score equation
            b = (lo + hi) / 2
            if np.sum(w * (sigmoid(z + b) - y)) > 0:
                hi = b
            else:
                lo = b
        self.b_ = (lo + hi) / 2
        return self

    def predict(self, p):
        return sigmoid(logit(p) + self.b_)


class Platt:
    """p' = sigmoid(a * logit(p) + b), fit by logistic regression on the logit (optionally with sample weights)."""

    def __init__(self, name="platt"):
        self.name = name

    def fit(self, p, y, weights=None):
        self.lr_ = LogisticRegression(C=np.inf, max_iter=1000)  # unpenalised
        self.lr_.fit(logit(p).reshape(-1, 1), np.asarray(y, dtype=int), sample_weight=weights)
        return self

    def predict(self, p):
        return self.lr_.predict_proba(logit(p).reshape(-1, 1))[:, 1]


def stage_b_calibrators():
    """Pre-registered candidates. 'platt_recency' is fit with block weights (1, 0.5, 0.25), nearest block first."""
    return [NoCalibration(), InterceptShift(), Platt("platt"), Platt("platt_recency")]


BLOCK_WEIGHTS = (1.0, 0.5, 0.25)
