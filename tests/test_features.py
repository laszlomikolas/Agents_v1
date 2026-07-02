"""Feature-builder tests (B2): correctness, orientation, and no-look-ahead."""
import numpy as np
import pandas as pd
import pytest

from models.features import (
    FEATURE_NAMES,
    SECONDS_PER_YEAR,
    build_features,
    feature_vector,
)


@pytest.fixture
def daily(make_daily_ohlcv):
    """30 rising daily candles; anchor t at the close of the last candle."""
    df = make_daily_ohlcv(periods=30)
    t = df["timestamp"].iloc[-1]
    T = t + pd.Timedelta(days=10)
    return df, t, T


def test_returns_all_feature_names(daily):
    df, t, T = daily
    feats = build_features(df, strike=200.0, direction="above", t=t, T=T)
    for name in FEATURE_NAMES:
        assert name in feats and np.isfinite(feats[name])
    assert feature_vector(feats) == [feats[n] for n in FEATURE_NAMES]


def test_s_t_is_last_close(daily):
    df, t, T = daily
    feats = build_features(df, strike=200.0, direction="above", t=t, T=T)
    assert feats["s_t"] == df["close"].iloc[-1]


def test_horizon_in_years(daily):
    df, t, T = daily
    feats = build_features(df, strike=200.0, direction="above", t=t, T=T)
    assert feats["horizon_years"] == pytest.approx(10 * 86400 / SECONDS_PER_YEAR, rel=1e-9)


def test_orientation_above_vs_below(daily):
    """signed_log_moneyness flips sign between above/below for the same strike."""
    df, t, T = daily
    s_t = df["close"].iloc[-1]
    strike = s_t * 1.1  # strike above spot
    above = build_features(df, strike=strike, direction="above", t=t, T=T)
    below = build_features(df, strike=strike, direction="below", t=t, T=T)
    assert above["signed_log_moneyness"] == pytest.approx(-below["signed_log_moneyness"])
    # Above with strike over spot => negative signed moneyness (OTM, below YES side).
    assert above["signed_log_moneyness"] < 0
    assert below["signed_log_moneyness"] > 0


def test_norm_distance_monotonic_in_strike(daily):
    """Higher 'above' strike => lower (more negative) norm_distance => less likely YES."""
    df, t, T = daily
    s_t = df["close"].iloc[-1]
    nd_near = build_features(df, strike=s_t * 1.01, direction="above", t=t, T=T)["norm_distance"]
    nd_far = build_features(df, strike=s_t * 1.50, direction="above", t=t, T=T)["norm_distance"]
    assert nd_far < nd_near


def test_no_look_ahead(make_daily_ohlcv):
    """Appending post-t candles must not change features (leak-free as-of slice)."""
    df = make_daily_ohlcv(periods=40)
    t = df["timestamp"].iloc[19]  # decide at candle 19's timestamp
    T = t + pd.Timedelta(days=5)
    asof_slice = df.iloc[:20]
    feats_slice = build_features(asof_slice, strike=180.0, direction="above", t=t, T=T)
    feats_full = build_features(df, strike=180.0, direction="above", t=t, T=T)  # extra future rows
    for name in FEATURE_NAMES:
        assert feats_full[name] == pytest.approx(feats_slice[name]), name


def test_deep_itm_otm_extremes(daily):
    df, t, T = daily
    s_t = df["close"].iloc[-1]
    otm = build_features(df, strike=s_t * 100, direction="above", t=t, T=T)
    itm = build_features(df, strike=s_t / 100, direction="above", t=t, T=T)
    assert otm["norm_distance"] == pytest.approx(-8.0)  # clipped
    assert itm["norm_distance"] == pytest.approx(8.0)


@pytest.mark.parametrize("bad_kwargs", [
    {"strike": 0.0}, {"strike": -5.0}, {"strike": float("nan")},
])
def test_bad_strike_raises(daily, bad_kwargs):
    df, t, T = daily
    with pytest.raises(ValueError):
        build_features(df, direction="above", t=t, T=T, **bad_kwargs)


def test_nonpositive_horizon_raises(daily):
    df, t, _ = daily
    with pytest.raises(ValueError):
        build_features(df, strike=200.0, direction="above", t=t, T=t)  # T == t


def test_empty_history_raises(daily):
    _, t, T = daily
    empty = pd.DataFrame(columns=["timestamp", "close"])
    with pytest.raises(ValueError):
        build_features(empty, strike=200.0, direction="above", t=t, T=T)


def test_bad_direction_raises(daily):
    df, t, T = daily
    with pytest.raises(ValueError):
        build_features(df, strike=200.0, direction="sideways", t=t, T=T)
