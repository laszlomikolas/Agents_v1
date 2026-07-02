"""Logistic-regression model (Phase B4, primary).

Standardize features, fit ``sklearn.linear_model.LogisticRegression``, predict
``P(YES)``. Persisted with joblib so a trained model can be reused by the
backtest and paper-trader without retraining.

A logistic on ``norm_distance`` alone approximates a calibrated GBM (that column
is the driftless ``d2``); the extra features (vol level, horizon, momentum) let
it correct the GBM's mis-specification — most importantly the fatter,
path-dependent tail of *touch* markets.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import numpy as np

from .base import ProbModel, as_matrix
from .features import FEATURE_NAMES


class LogisticModel(ProbModel):
    """StandardScaler + LogisticRegression behind the ProbModel interface."""

    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        self.C = C
        self.max_iter = max_iter
        self.pipeline_ = None
        self.feature_names_ = list(FEATURE_NAMES)

    def fit(self, X: Any, y: Any) -> "LogisticModel":
        # Imported lazily so the package imports without scikit-learn installed.
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        mat = as_matrix(X)
        y = np.asarray(y, dtype=int)
        if mat.shape[0] != y.shape[0]:
            raise ValueError("X and y have mismatched lengths")
        if np.unique(y).size < 2:
            raise ValueError("training labels must contain both classes (0 and 1)")

        self.pipeline_ = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=self.C, max_iter=self.max_iter)),
        ])
        self.pipeline_.fit(mat, y)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if self.pipeline_ is None:
            raise RuntimeError("model is not fitted")
        mat = as_matrix(X)
        proba = self.pipeline_.predict_proba(mat)[:, 1]
        return np.clip(proba, 1e-6, 1 - 1e-6)

    @property
    def coefficients(self) -> dict[str, float]:
        """Fitted logistic coefficients keyed by feature name (on scaled inputs)."""
        if self.pipeline_ is None:
            raise RuntimeError("model is not fitted")
        coef = self.pipeline_.named_steps["clf"].coef_[0]
        return dict(zip(self.feature_names_, (float(c) for c in coef)))

    def save(self, path: Union[str, Path]) -> Path:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Union[str, Path]) -> "LogisticModel":
        import joblib

        model = joblib.load(path)
        if not isinstance(model, LogisticModel):
            raise TypeError(f"{path} does not contain a LogisticModel")
        return model
