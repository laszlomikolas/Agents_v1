"""Training-set generator (Phase B3).

We are **not** bottlenecked on the small number of resolved Polymarket markets:
BTC/ETH OHLCV history alone yields thousands of labeled ``(t, horizon, strike)``
examples. For each sampled anchor candle we build as-of features from the past
and read the realized label from the future window — terminal (close at ``T``)
or touch (barrier crossed anywhere in ``(t, T]``).

Leak-free by construction: features come from ``ohlcv.iloc[lo:i+1]`` (candles
at/before the anchor) and labels from strictly-later candles ``i+1 .. i+n``.

``label_kind``:
    ``"terminal"`` – YES iff the close at ``T`` satisfies the threshold.
    ``"touch"``    – YES iff the window High reaches an ``above`` strike (or the
                     window Low reaches a ``below`` strike) at any point.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from .features import FEATURE_NAMES, build_features

DEFAULT_HORIZONS: tuple[int, ...] = (1, 2, 3, 5, 7, 10, 14, 21, 30)
LABEL_KINDS = ("terminal", "touch")

_OUTPUT_META = [
    "label", "direction", "label_kind",
    "anchor_ts", "resolve_ts", "strike", "s_t", "horizon_candles",
]


def _label(
    direction: str,
    label_kind: str,
    strike: float,
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    i: int,
    n: int,
) -> Optional[int]:
    """Realized YES/NO for one sample, or None if the future window is unusable."""
    future = slice(i + 1, i + n + 1)
    if label_kind == "terminal":
        terminal_close = closes[i + n]
        if not np.isfinite(terminal_close):
            return None
        if direction == "above":
            return int(terminal_close >= strike)
        return int(terminal_close <= strike)

    # touch / barrier: did the path cross the strike at any point in (t, T]?
    if direction == "above":
        window_high = highs[future]
        if window_high.size == 0 or not np.isfinite(window_high).any():
            return None
        return int(np.nanmax(window_high) >= strike)
    window_low = lows[future]
    if window_low.size == 0 or not np.isfinite(window_low).any():
        return None
    return int(np.nanmin(window_low) <= strike)


def build_training_set(
    ohlcv: pd.DataFrame,
    *,
    label_kind: str = "terminal",
    n_samples: int = 5000,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    directions: Sequence[str] = ("above", "below"),
    moneyness_sigma: float = 1.5,
    vol_window: int = 30,
    momentum_window: int = 10,
    seed: int = 0,
) -> pd.DataFrame:
    """Sample labeled training rows from a single OHLCV series.

    Args:
        ohlcv: Columns ``timestamp``, ``high``, ``low``, ``close`` sorted by time
            (the datastore ``read_ohlcv`` shape).
        label_kind: ``"terminal"`` or ``"touch"``.
        n_samples: Number of ``(anchor, horizon, strike, direction)`` draws to
            attempt (rows with an unusable window are skipped, so output may be
            slightly smaller).
        horizons: Candidate horizons **in candles**.
        directions: Which market directions to sample.
        moneyness_sigma: Strike scatter, in units of ``sigma*sqrt(horizon)``
            around the money — controls the spread of realized probabilities.
        vol_window: Trailing candles for the vol estimate (also the min history).
        momentum_window: Trailing candles for the momentum feature.
        seed: RNG seed (reproducible).

    Returns:
        DataFrame with ``FEATURE_NAMES`` columns plus label/metadata columns.
    """
    if label_kind not in LABEL_KINDS:
        raise ValueError(f"label_kind must be one of {LABEL_KINDS}, got {label_kind!r}")
    for d in directions:
        if d not in ("above", "below"):
            raise ValueError(f"direction must be 'above'/'below', got {d!r}")
    if ohlcv is None or ohlcv.empty:
        return pd.DataFrame(columns=[*FEATURE_NAMES, *_OUTPUT_META])

    for col in ("timestamp", "high", "low", "close"):
        if col not in ohlcv.columns:
            raise ValueError(f"ohlcv missing required column {col!r}")

    df = ohlcv.dropna(subset=["timestamp", "close"]).sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    closes = df["close"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    timestamps = pd.to_datetime(df["timestamp"], utc=True)

    max_h = max(horizons)
    hist = max(vol_window, momentum_window) + 1  # candles needed for features
    i_lo = hist
    i_hi = n - 1 - max_h
    if i_hi < i_lo:
        raise ValueError(
            f"series too short: need > {hist + max_h} candles, have {n}"
        )

    rng = np.random.default_rng(seed)
    horizons_arr = np.asarray(horizons)
    dirs = list(directions)

    rows: list[dict] = []
    for _ in range(n_samples):
        i = int(rng.integers(i_lo, i_hi + 1))
        h = int(rng.choice(horizons_arr))
        direction = dirs[int(rng.integers(0, len(dirs)))]

        lo = i - hist + 1
        tail_closes = closes[lo : i + 1]
        if tail_closes.size < 3 or not np.isfinite(tail_closes).all() or (tail_closes <= 0).any():
            continue

        # Per-period vol for strike placement (build_features recomputes it).
        rets = np.diff(np.log(tail_closes))
        sigma_hat = float(np.std(rets, ddof=1)) if rets.size >= 2 else 0.0
        scale = moneyness_sigma * max(sigma_hat, 1e-6) * np.sqrt(h)
        z = float(rng.standard_normal())
        strike = float(closes[i] * np.exp(z * scale))
        if not np.isfinite(strike) or strike <= 0:
            continue

        label = _label(direction, label_kind, strike, closes, highs, lows, i, h)
        if label is None:
            continue

        t = timestamps.iloc[i]
        T = timestamps.iloc[i + h]
        try:
            feats = build_features(
                df.iloc[lo : i + 1],
                strike=strike,
                direction=direction,
                t=t,
                T=T,
                vol_window=vol_window,
                momentum_window=momentum_window,
            )
        except ValueError:
            continue

        row = {name: feats[name] for name in FEATURE_NAMES}
        row.update(
            label=label,
            direction=direction,
            label_kind=label_kind,
            anchor_ts=t,
            resolve_ts=T,
            strike=strike,
            s_t=feats["s_t"],
            horizon_candles=h,
        )
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=[*FEATURE_NAMES, *_OUTPUT_META])
    return pd.DataFrame(rows, columns=[*FEATURE_NAMES, *_OUTPUT_META])


def build_training_set_from_store(
    store: Any,
    symbol: str,
    interval: str,
    **kwargs: Any,
) -> pd.DataFrame:
    """Convenience wrapper: read the full OHLCV series from a store, then sample.

    ``store`` is a ``datastore.store.MarketDataStore`` (or anything with a
    compatible ``read_ohlcv(symbol, interval)``).
    """
    ohlcv = store.read_ohlcv(symbol, interval)
    return build_training_set(ohlcv, **kwargs)
