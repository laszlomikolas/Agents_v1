"""Common probability-model interface (Phase B4).

Every model predicts ``P(YES)`` for a market and shares one interface so the
signal/backtest/paper-trade layers stay model-agnostic:

    model.fit(X, y)            # X: DataFrame of FEATURE_NAMES, y: 0/1 labels
    model.predict_proba(X)     # -> np.ndarray of P(YES) in [0, 1]
    model.predict_one(feats)   # feats: a build_features dict -> float
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Union

import numpy as np
import pandas as pd

from .features import FEATURE_NAMES

ArrayLike = Union[pd.DataFrame, np.ndarray, "list[list[float]]"]


def as_matrix(X: ArrayLike) -> np.ndarray:
    """Coerce input to a 2D float matrix with columns ordered as FEATURE_NAMES."""
    if isinstance(X, pd.DataFrame):
        missing = [c for c in FEATURE_NAMES if c not in X.columns]
        if missing:
            raise ValueError(f"input is missing feature columns: {missing}")
        return X.loc[:, FEATURE_NAMES].to_numpy(dtype=float)
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"expected {len(FEATURE_NAMES)} feature columns, got {arr.shape[1]}"
        )
    return arr


class ProbModel(ABC):
    """Abstract binary probability model over the shared feature set."""

    @abstractmethod
    def fit(self, X: ArrayLike, y: Any) -> "ProbModel":
        ...

    @abstractmethod
    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Return P(YES) for each row, as a 1D array in [0, 1]."""

    def predict_one(self, features: dict) -> float:
        """Predict for a single ``build_features`` dict."""
        row = pd.DataFrame([{name: features[name] for name in FEATURE_NAMES}])
        return float(self.predict_proba(row)[0])
