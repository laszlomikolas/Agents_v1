"""Calibration evaluation (Phase B5).

Train/test split, then score the primary logistic model against the empirical
benchmark and the parameter-free GBM baseline on held-out data:

* **Brier score** and **log-loss** — lower is better.
* **Reliability curve** — predicted vs. observed frequency per probability bin.

Gate: the logistic model should be **at least as calibrated as the empirical
benchmark** (Brier no worse) before it is allowed to trade.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from .base import ProbModel
from .dataset import build_training_set
from .empirical import EmpiricalModel
from .features import FEATURE_NAMES
from .gbm import GBMModel
from .logistic import LogisticModel

_EPS = 1e-12


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), _EPS, 1 - _EPS)
    return float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))


def reliability_curve(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Per-bin mean predicted probability vs. observed frequency."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        rows.append({
            "bin": b,
            "mean_predicted": float(y_prob[mask].mean()),
            "observed_freq": float(y_true[mask].mean()),
            "count": int(mask.sum()),
        })
    return pd.DataFrame(rows, columns=["bin", "mean_predicted", "observed_freq", "count"])


def train_test_split_df(
    df: pd.DataFrame,
    test_frac: float = 0.25,
    *,
    by_time: bool = True,
    time_col: str = "anchor_ts",
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataset into train/test.

    ``by_time=True`` (default) puts the latest ``test_frac`` of rows (by
    ``time_col``) in the test set — a mild walk-forward that avoids training on
    the future. Set ``by_time=False`` for a seeded random split.
    """
    if df.empty:
        return df.copy(), df.copy()
    n = len(df)
    n_test = max(1, int(round(n * test_frac)))
    if by_time and time_col in df.columns:
        ordered = df.sort_values(time_col).reset_index(drop=True)
        return ordered.iloc[: n - n_test].copy(), ordered.iloc[n - n_test :].copy()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    test_idx = np.sort(perm[:n_test])
    train_idx = np.sort(perm[n_test:])
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def _score(model: ProbModel, X_test: pd.DataFrame, y_test: np.ndarray) -> dict:
    p = model.predict_proba(X_test)
    return {"brier": brier_score(y_test, p), "log_loss": log_loss(y_test, p)}


def evaluate_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    label_kind: str,
    logistic_C: float = 1.0,
) -> dict:
    """Fit logistic + empirical on ``train_df``, score all three on ``test_df``.

    Returns a report dict with per-model ``brier``/``log_loss``, the reliability
    curve for the logistic model, and a ``gate_passed`` flag (logistic Brier
    <= empirical Brier).
    """
    for frame, name in ((train_df, "train"), (test_df, "test")):
        missing = [c for c in [*FEATURE_NAMES, "label"] if c not in frame.columns]
        if missing:
            raise ValueError(f"{name}_df missing columns: {missing}")
    if train_df.empty or test_df.empty:
        raise ValueError("train_df and test_df must be non-empty")

    X_train = train_df[FEATURE_NAMES]
    y_train = train_df["label"].to_numpy(dtype=int)
    X_test = test_df[FEATURE_NAMES]
    y_test = test_df["label"].to_numpy(dtype=int)

    logistic = LogisticModel(C=logistic_C).fit(X_train, y_train)
    empirical = EmpiricalModel().fit(X_train, y_train)
    gbm = GBMModel(label_kind=label_kind)

    metrics = {
        "logistic": _score(logistic, X_test, y_test),
        "empirical": _score(empirical, X_test, y_test),
        "gbm": _score(gbm, X_test, y_test),
    }
    report = {
        "label_kind": label_kind,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "base_rate_test": float(y_test.mean()),
        "metrics": metrics,
        "gate_passed": metrics["logistic"]["brier"] <= metrics["empirical"]["brier"],
        "logistic_coefficients": logistic.coefficients,
        "reliability_logistic": reliability_curve(y_test, logistic.predict_proba(X_test)),
        "models": {"logistic": logistic, "empirical": empirical, "gbm": gbm},
    }
    return report


def run_evaluation(
    ohlcv: pd.DataFrame,
    *,
    label_kind: str = "terminal",
    n_samples: int = 8000,
    test_frac: float = 0.25,
    seed: int = 0,
    dataset_kwargs: Optional[dict] = None,
) -> dict:
    """End-to-end: build a labeled dataset from OHLCV, split, and evaluate."""
    dataset = build_training_set(
        ohlcv, label_kind=label_kind, n_samples=n_samples, seed=seed,
        **(dataset_kwargs or {}),
    )
    if dataset.empty:
        raise ValueError("training set is empty; check the OHLCV series length")
    train_df, test_df = train_test_split_df(dataset, test_frac=test_frac, seed=seed)
    return evaluate_models(train_df, test_df, label_kind=label_kind)


def format_report(report: dict) -> str:
    """Render an ``evaluate_models`` report as a compact text block."""
    lines = [
        f"Calibration report - {report['label_kind']} "
        f"(train={report['n_train']}, test={report['n_test']}, "
        f"base rate={report['base_rate_test']:.3f})",
        f"{'model':<12}{'brier':>10}{'log_loss':>12}",
    ]
    for name, m in report["metrics"].items():
        lines.append(f"{name:<12}{m['brier']:>10.4f}{m['log_loss']:>12.4f}")
    gate = "PASS" if report["gate_passed"] else "FAIL"
    lines.append(f"gate (logistic Brier <= empirical): {gate}")
    return "\n".join(lines)
