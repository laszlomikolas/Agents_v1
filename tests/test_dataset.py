"""Training-set generator tests (B3): shape, labels, reproducibility, and the
touch-vs-terminal invariant."""
import numpy as np
import pandas as pd
import pytest

from models.dataset import build_training_set, build_training_set_from_store
from models.features import FEATURE_NAMES


@pytest.fixture
def series(gbm_ohlcv):
    return gbm_ohlcv(n_candles=800, seed=11)


def test_shape_and_columns(series):
    ds = build_training_set(series, label_kind="terminal", n_samples=500, seed=0)
    assert not ds.empty
    for col in [*FEATURE_NAMES, "label", "direction", "strike", "anchor_ts", "resolve_ts"]:
        assert col in ds.columns
    assert set(ds["label"].unique()) <= {0, 1}
    assert set(ds["direction"].unique()) <= {"above", "below"}


def test_both_classes_present(series):
    ds = build_training_set(series, label_kind="terminal", n_samples=800, seed=0)
    assert ds["label"].nunique() == 2  # not degenerate


def test_anchor_precedes_resolution(series):
    ds = build_training_set(series, label_kind="touch", n_samples=400, seed=1)
    assert (ds["anchor_ts"] < ds["resolve_ts"]).all()
    assert (ds["horizon_candles"] > 0).all()
    assert (ds["s_t"] > 0).all()


def test_touch_superset_of_terminal(series):
    """Same samples (same seed): a touch is YES whenever the terminal is, so the
    touch label dominates the terminal label pointwise."""
    common = dict(n_samples=600, seed=42)
    terminal = build_training_set(series, label_kind="terminal", **common)
    touch = build_training_set(series, label_kind="touch", **common)
    # Identical sampling => aligned rows.
    assert len(terminal) == len(touch)
    assert np.allclose(terminal["strike"].to_numpy(), touch["strike"].to_numpy())
    assert (touch["label"].to_numpy() >= terminal["label"].to_numpy()).all()
    assert touch["label"].mean() >= terminal["label"].mean()


def test_reproducible(series):
    a = build_training_set(series, label_kind="terminal", n_samples=300, seed=5)
    b = build_training_set(series, label_kind="terminal", n_samples=300, seed=5)
    pd.testing.assert_frame_equal(a, b)
    c = build_training_set(series, label_kind="terminal", n_samples=300, seed=6)
    assert not a["label"].equals(c["label"])


def test_invalid_label_kind_raises(series):
    with pytest.raises(ValueError):
        build_training_set(series, label_kind="nonsense")


def test_short_series_raises(gbm_ohlcv):
    tiny = gbm_ohlcv(n_candles=20, seed=1)
    with pytest.raises(ValueError):
        build_training_set(tiny, label_kind="terminal", n_samples=100)


def test_empty_returns_empty_frame():
    empty = pd.DataFrame(columns=["timestamp", "high", "low", "close"])
    ds = build_training_set(empty, label_kind="terminal")
    assert ds.empty
    assert list(ds.columns)[: len(FEATURE_NAMES)] == FEATURE_NAMES


def test_from_store_roundtrip(store, series):
    store.upsert_ohlcv("BTCUSD", "1d", series)
    ds = build_training_set_from_store(
        store, "BTCUSD", "1d", label_kind="terminal", n_samples=200, seed=0
    )
    assert not ds.empty
    assert set(ds["label"].unique()) <= {0, 1}
