"""Empirical-frequency benchmark model (Phase B4).

The benchmark every other model must beat: bin the training rows by
(moneyness bucket, horizon bucket) and predict the smoothed hit frequency of
the matching bin. Deliberately dumb but well-calibrated by construction, so if
the logistic model can't match it on held-out data, the logistic isn't ready.

Smoothing pulls sparse bins toward the global base rate:

    p_bin = (hits + alpha * global_rate) / (count + alpha)
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .base import ProbModel, as_matrix
from .features import FEATURE_NAMES

_MONEYNESS_IDX = FEATURE_NAMES.index("signed_log_moneyness")
_HORIZON_IDX = FEATURE_NAMES.index("horizon_years")


def _quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Interior quantile edges for binning; de-duplicated for degenerate cols."""
    if values.size == 0:
        return np.array([0.0])
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    edges = np.unique(np.quantile(values, qs))
    if edges.size == 0:  # constant column
        edges = np.array([values[0]])
    return edges


class EmpiricalModel(ProbModel):
    """Binned empirical-frequency predictor over (moneyness, horizon)."""

    def __init__(self, n_moneyness_bins: int = 10, n_horizon_bins: int = 6, alpha: float = 5.0):
        self.n_moneyness_bins = n_moneyness_bins
        self.n_horizon_bins = n_horizon_bins
        self.alpha = alpha
        self.global_rate_: float = 0.5
        self.moneyness_edges_: np.ndarray | None = None
        self.horizon_edges_: np.ndarray | None = None
        self.bin_prob_: dict[tuple[int, int], float] = {}

    def _bin_indices(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m = np.digitize(X[:, _MONEYNESS_IDX], self.moneyness_edges_)
        h = np.digitize(X[:, _HORIZON_IDX], self.horizon_edges_)
        return m, h

    def fit(self, X: Any, y: Any) -> "EmpiricalModel":
        mat = as_matrix(X)
        y = np.asarray(y, dtype=float)
        if mat.shape[0] != y.shape[0]:
            raise ValueError("X and y have mismatched lengths")
        if y.size == 0:
            raise ValueError("cannot fit on an empty training set")

        self.global_rate_ = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
        self.moneyness_edges_ = _quantile_edges(mat[:, _MONEYNESS_IDX], self.n_moneyness_bins)
        self.horizon_edges_ = _quantile_edges(mat[:, _HORIZON_IDX], self.n_horizon_bins)

        m_idx, h_idx = self._bin_indices(mat)
        df = pd.DataFrame({"m": m_idx, "h": h_idx, "y": y})
        agg = df.groupby(["m", "h"])["y"].agg(["sum", "count"])
        self.bin_prob_ = {}
        for (m, h), row in agg.iterrows():
            p = (row["sum"] + self.alpha * self.global_rate_) / (row["count"] + self.alpha)
            self.bin_prob_[(int(m), int(h))] = float(p)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if self.moneyness_edges_ is None:
            raise RuntimeError("model is not fitted")
        mat = as_matrix(X)
        m_idx, h_idx = self._bin_indices(mat)
        out = np.empty(mat.shape[0], dtype=float)
        for k in range(mat.shape[0]):
            out[k] = self.bin_prob_.get((int(m_idx[k]), int(h_idx[k])), self.global_rate_)
        return np.clip(out, 1e-6, 1 - 1e-6)
