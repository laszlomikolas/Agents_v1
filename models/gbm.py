"""Closed-form GBM baseline (Phase B5 sanity check).

A driftless geometric-Brownian-motion reference with no free parameters. It
turns the ``norm_distance`` feature (the driftless ``d2 = signed_log_moneyness /
(sigma*sqrt(h))``) straight into a probability, so the fitted models have an
analytic yardstick:

* terminal: ``P(S_T on the YES side) = Phi(norm_distance)``.
* touch:    reflection principle for driftless BM — the probability the barrier
            is ever hit is ``min(1, 2 * Phi(norm_distance))``. At the money
            (``norm_distance == 0``) that is 1.0 (a barrier at spot is hit almost
            surely over any positive horizon); once spot is past the strike it
            clips to 1.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .base import ProbModel, as_matrix
from .dataset import LABEL_KINDS
from .features import FEATURE_NAMES

_NORM_IDX = FEATURE_NAMES.index("norm_distance")


class GBMModel(ProbModel):
    """Parameter-free lognormal/first-passage baseline keyed by market mechanics."""

    def __init__(self, label_kind: str = "terminal"):
        if label_kind not in LABEL_KINDS:
            raise ValueError(f"label_kind must be one of {LABEL_KINDS}, got {label_kind!r}")
        self.label_kind = label_kind

    def fit(self, X: Any = None, y: Any = None) -> "GBMModel":
        # Parameter-free: nothing to estimate. Present for interface parity.
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        from scipy.stats import norm

        mat = as_matrix(X)
        d = mat[:, _NORM_IDX]
        phi = norm.cdf(d)
        if self.label_kind == "touch":
            proba = np.minimum(1.0, 2.0 * phi)
        else:
            proba = phi
        return np.clip(proba, 1e-6, 1 - 1e-6)
