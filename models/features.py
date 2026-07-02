"""As-of feature builder for price-threshold markets (Phase B2).

``build_features`` turns a leak-free OHLCV slice into the numeric features the
models consume. It is a *pure* function reused by the training-set generator,
the backtest replay, and live prediction, so all three see identical features.

Orientation
-----------
Every feature is oriented **toward the YES event** so a single model handles
both directions:

* an ``above`` market pays YES when the price rises to/above the strike;
* a ``below`` market pays YES when the price falls to/below the strike.

So ``signed_log_moneyness`` is ``log(S_t / K)`` for ``above`` and ``log(K / S_t)``
for ``below`` — larger always means "closer to / past the YES side". The
normalized distance ``signed_log_moneyness / (sigma * sqrt(horizon))`` is the
driftless GBM ``d2`` input: ``Phi(norm_distance)`` is a calibrated terminal
probability, and a logistic on it recovers the same shape while letting the
touch/terminal data set its own slope.

No look-ahead
-------------
Features use only rows with ``timestamp <= t``. The slice is defensively
re-filtered here even though callers are expected to pass an as-of slice.
"""
from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import pandas as pd

SECONDS_PER_YEAR = 365.25 * 24 * 3600.0

# The ordered feature vector the models train/predict on. Extra diagnostic keys
# (S_t, sigma_sqrt_h, ...) are returned by build_features but excluded here.
FEATURE_NAMES: list[str] = [
    "signed_log_moneyness",
    "norm_distance",
    "sigma_annual",
    "horizon_years",
    "signed_momentum",
]

# Keep the driftless d2 input in a sane range so a flat-vol edge case can't send
# the logistic's linear term to +/-inf.
_NORM_DISTANCE_CLIP = 8.0
_SIGMA_FLOOR = 1e-9


def _epoch(value: Any) -> float:
    """Timestamp-like -> Unix seconds (UTC), matching datastore.store semantics."""
    ts = pd.Timestamp(value)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.timestamp()


def _infer_period_seconds(epochs: np.ndarray) -> float:
    """Median spacing between candles, in seconds (robust to gaps)."""
    if epochs.size < 2:
        raise ValueError("need at least two candles to infer the candle interval")
    diffs = np.diff(epochs)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        raise ValueError("candle timestamps are not strictly increasing")
    return float(np.median(diffs))


def build_features(
    ohlcv_asof: pd.DataFrame,
    strike: float,
    direction: str,
    t: Any,
    T: Any,
    *,
    vol_window: Optional[int] = None,
    momentum_window: Optional[int] = None,
) -> dict:
    """Build the model feature dict for one (market, decision-time) pair.

    Args:
        ohlcv_asof: Candles as-of ``t`` (columns ``timestamp``, ``close``; the
            standard datastore read shape). Rows after ``t`` are dropped
            defensively. Must be sorted or sortable by ``timestamp``.
        strike: Strike price ``K`` (> 0).
        direction: ``"above"`` or ``"below"`` (orients every feature toward YES).
        t: Decision time. ``S_t`` is the last close at/BEFORE ``t``.
        T: Resolution time (``> t``); sets the horizon.
        vol_window: Trailing number of log-returns for the vol estimate
            (default: all available returns).
        momentum_window: Trailing number of periods for the momentum feature
            (default: ``min(vol_window or all, available)``).

    Returns:
        Dict with every key in ``FEATURE_NAMES`` plus diagnostics ``s_t``,
        ``strike``, ``sigma_sqrt_h`` and ``period_seconds``.

    Raises:
        ValueError: empty/insufficient history, non-positive strike/price, or a
            non-positive horizon.
    """
    if direction not in ("above", "below"):
        raise ValueError(f"direction must be 'above' or 'below', got {direction!r}")
    if strike is None or not math.isfinite(strike) or strike <= 0:
        raise ValueError(f"strike must be a positive finite number, got {strike!r}")
    if ohlcv_asof is None or ohlcv_asof.empty:
        raise ValueError("ohlcv_asof is empty; no price history as-of t")
    if "timestamp" not in ohlcv_asof.columns or "close" not in ohlcv_asof.columns:
        raise ValueError("ohlcv_asof must have 'timestamp' and 'close' columns")

    t_epoch = _epoch(t)
    T_epoch = _epoch(T)
    horizon_seconds = T_epoch - t_epoch
    if horizon_seconds <= 0:
        raise ValueError(f"horizon must be positive: t={t!r} T={T!r}")

    work = ohlcv_asof.loc[:, ["timestamp", "close"]].dropna(subset=["timestamp", "close"])
    epochs_all = work["timestamp"].map(_epoch).to_numpy(dtype=float)
    keep = epochs_all <= t_epoch  # leak-free: only candles closed at/before t
    work = work.loc[keep]
    epochs = epochs_all[keep]
    if work.empty:
        raise ValueError("no candles at/before t after the as-of filter")

    order = np.argsort(epochs, kind="stable")
    epochs = epochs[order]
    closes = work["close"].to_numpy(dtype=float)[order]

    s_t = float(closes[-1])
    if not math.isfinite(s_t) or s_t <= 0:
        raise ValueError(f"non-positive current price S_t={s_t!r}")

    period_seconds = _infer_period_seconds(epochs) if epochs.size >= 2 else horizon_seconds
    periods_per_year = SECONDS_PER_YEAR / period_seconds

    # Trailing log returns for vol + momentum.
    log_close = np.log(closes)
    returns = np.diff(log_close)
    if vol_window is not None and vol_window > 0:
        returns = returns[-vol_window:]
    if returns.size >= 2:
        sigma_period = float(np.std(returns, ddof=1))
    elif returns.size == 1:
        sigma_period = float(abs(returns[0]))
    else:
        sigma_period = 0.0
    sigma_period = max(sigma_period, _SIGMA_FLOOR)
    sigma_annual = sigma_period * math.sqrt(periods_per_year)

    horizon_years = horizon_seconds / SECONDS_PER_YEAR
    sigma_sqrt_h = sigma_annual * math.sqrt(horizon_years)

    # Orientation: +1 for an "above"/up event, -1 for a "below"/down event.
    up = direction == "above"
    sign = 1.0 if up else -1.0

    log_moneyness_up = math.log(s_t / strike)  # >0 when price is above the strike
    signed_log_moneyness = sign * log_moneyness_up

    norm_distance = signed_log_moneyness / max(sigma_sqrt_h, _SIGMA_FLOOR)
    norm_distance = float(np.clip(norm_distance, -_NORM_DISTANCE_CLIP, _NORM_DISTANCE_CLIP))

    # Trailing momentum: return over the momentum window, oriented toward YES.
    if momentum_window is None:
        mwin = returns.size
    else:
        mwin = min(momentum_window, log_close.size - 1)
    if mwin >= 1:
        momentum = float(log_close[-1] - log_close[-1 - mwin])
    else:
        momentum = 0.0
    signed_momentum = sign * momentum

    return {
        "signed_log_moneyness": signed_log_moneyness,
        "norm_distance": norm_distance,
        "sigma_annual": sigma_annual,
        "horizon_years": horizon_years,
        "signed_momentum": signed_momentum,
        # diagnostics (not in FEATURE_NAMES)
        "s_t": s_t,
        "strike": float(strike),
        "sigma_sqrt_h": sigma_sqrt_h,
        "period_seconds": period_seconds,
    }


def feature_vector(features: dict) -> list[float]:
    """Extract ``FEATURE_NAMES`` values from a build_features dict, in order."""
    return [float(features[name]) for name in FEATURE_NAMES]
