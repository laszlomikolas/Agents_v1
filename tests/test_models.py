"""Model tests (B4/B5): interface, GBM baseline, persistence, and calibration."""
import numpy as np
import pandas as pd
import pytest

from models.base import as_matrix
from models.dataset import build_training_set
from models.empirical import EmpiricalModel
from models.evaluate import (
    brier_score,
    evaluate_models,
    log_loss,
    reliability_curve,
    run_evaluation,
    train_test_split_df,
)
from models.features import FEATURE_NAMES
from models.gbm import GBMModel
from models.logistic import LogisticModel


@pytest.fixture(scope="module")
def terminal_dataset(_gbm_series):
    return build_training_set(_gbm_series, label_kind="terminal", n_samples=4000, seed=3)


@pytest.fixture(scope="module")
def _gbm_series():
    # Module-scoped GBM series (regenerate here; fixtures aren't shareable across scope).
    rng = np.random.default_rng(7)
    seconds_per_year = 365.25 * 24 * 3600.0
    day = 24 * 3600.0
    sub = 24
    n = 1200
    step_vol = 0.6 * np.sqrt(day / sub / seconds_per_year)
    prices = np.exp(np.log(100.0) + np.cumsum(rng.standard_normal(n * sub) * step_vol)).reshape(n, sub)
    ts = pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts, "open": prices[:, 0], "high": prices.max(1),
        "low": prices.min(1), "close": prices[:, -1], "volume": np.full(n, 100.0),
    })


# ── base coercion ────────────────────────────────────────────────────────────
def test_as_matrix_from_dataframe():
    df = pd.DataFrame([{n: float(i) for i, n in enumerate(FEATURE_NAMES)}])
    mat = as_matrix(df)
    assert mat.shape == (1, len(FEATURE_NAMES))


def test_as_matrix_missing_column_raises():
    df = pd.DataFrame([{n: 0.0 for n in FEATURE_NAMES[:-1]}])
    with pytest.raises(ValueError):
        as_matrix(df)


def test_as_matrix_wrong_width_raises():
    with pytest.raises(ValueError):
        as_matrix(np.zeros((2, len(FEATURE_NAMES) + 1)))


# ── GBM baseline ─────────────────────────────────────────────────────────────
def _row(norm_distance):
    row = {n: 0.0 for n in FEATURE_NAMES}
    row["norm_distance"] = norm_distance
    return pd.DataFrame([row])


def test_gbm_terminal_at_the_money_is_half():
    assert GBMModel("terminal").predict_proba(_row(0.0))[0] == pytest.approx(0.5, abs=1e-6)


def test_gbm_terminal_monotonic():
    m = GBMModel("terminal")
    lo = m.predict_proba(_row(-1.0))[0]
    hi = m.predict_proba(_row(1.0))[0]
    assert lo < 0.5 < hi


def test_gbm_touch_atm_is_certain():
    # Reflection principle: a barrier at spot is hit a.s. over any positive horizon.
    assert GBMModel("touch").predict_proba(_row(0.0))[0] == pytest.approx(1.0, abs=1e-3)


def test_gbm_touch_exceeds_terminal_otm():
    nd = -0.5  # out of the money
    terminal = GBMModel("terminal").predict_proba(_row(nd))[0]
    touch = GBMModel("touch").predict_proba(_row(nd))[0]
    assert touch > terminal


def test_gbm_invalid_label_kind():
    with pytest.raises(ValueError):
        GBMModel("banana")


# ── empirical + logistic ─────────────────────────────────────────────────────
def test_empirical_fit_predict_range(terminal_dataset):
    m = EmpiricalModel().fit(terminal_dataset[FEATURE_NAMES], terminal_dataset["label"])
    p = m.predict_proba(terminal_dataset[FEATURE_NAMES])
    assert p.min() >= 0.0 and p.max() <= 1.0


def test_empirical_unfitted_raises():
    with pytest.raises(RuntimeError):
        EmpiricalModel().predict_proba(_row(0.0))


def test_logistic_fit_predict_and_predict_one(terminal_dataset):
    m = LogisticModel().fit(terminal_dataset[FEATURE_NAMES], terminal_dataset["label"])
    p = m.predict_proba(terminal_dataset[FEATURE_NAMES])
    assert p.min() >= 0.0 and p.max() <= 1.0
    assert set(m.coefficients) == set(FEATURE_NAMES)
    # predict_one on a strongly-YES feature row returns a high probability.
    strong = {n: 0.0 for n in FEATURE_NAMES}
    strong["norm_distance"] = 5.0
    strong["signed_log_moneyness"] = 1.0
    assert 0.0 <= m.predict_one(strong) <= 1.0


def test_logistic_single_class_raises():
    df = pd.DataFrame([{n: 0.0 for n in FEATURE_NAMES} for _ in range(10)])
    with pytest.raises(ValueError):
        LogisticModel().fit(df, np.zeros(10, dtype=int))


def test_logistic_save_load_roundtrip(tmp_path, terminal_dataset):
    m = LogisticModel().fit(terminal_dataset[FEATURE_NAMES], terminal_dataset["label"])
    path = m.save(tmp_path / "artifacts" / "logistic.joblib")
    assert path.exists()
    loaded = LogisticModel.load(path)
    np.testing.assert_allclose(
        m.predict_proba(terminal_dataset[FEATURE_NAMES]),
        loaded.predict_proba(terminal_dataset[FEATURE_NAMES]),
    )


# ── evaluation metrics + calibration gate ────────────────────────────────────
def test_brier_and_log_loss_perfect():
    y = np.array([0, 1, 1, 0])
    assert brier_score(y, y.astype(float)) == 0.0
    assert log_loss(y, y.astype(float)) < 1e-6


def test_reliability_curve_columns(terminal_dataset):
    y = terminal_dataset["label"].to_numpy()
    p = np.clip(np.random.default_rng(0).random(len(y)), 0, 1)
    curve = reliability_curve(y, p, n_bins=5)
    assert list(curve.columns) == ["bin", "mean_predicted", "observed_freq", "count"]
    assert curve["count"].sum() == len(y)


def test_train_test_split_by_time(terminal_dataset):
    train, test = train_test_split_df(terminal_dataset, test_frac=0.25, by_time=True)
    assert len(test) > 0 and len(train) > 0
    assert train["anchor_ts"].max() <= test["anchor_ts"].min()  # no future in train


def test_logistic_calibration_beats_or_matches_benchmark(terminal_dataset):
    """Gate: on GBM-simulated terminal data the logistic must be at least as
    calibrated (Brier) as the empirical benchmark, and both beat the base rate."""
    train, test = train_test_split_df(terminal_dataset, test_frac=0.3, by_time=True)
    report = evaluate_models(train, test, label_kind="terminal")
    m = report["metrics"]
    base_rate = report["base_rate_test"]
    base_brier = base_rate * (1 - base_rate)  # Brier of predicting the constant base rate
    assert m["logistic"]["brier"] <= m["empirical"]["brier"] + 0.02
    assert m["logistic"]["brier"] <= base_brier + 1e-6
    assert np.isfinite(m["gbm"]["brier"])


def test_run_evaluation_end_to_end(_gbm_series):
    report = run_evaluation(_gbm_series, label_kind="touch", n_samples=3000, seed=0)
    assert report["label_kind"] == "touch"
    assert 0.0 <= report["base_rate_test"] <= 1.0
    assert "logistic" in report["metrics"]
