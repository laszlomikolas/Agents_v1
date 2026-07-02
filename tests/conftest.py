"""Shared pytest fixtures for the test suite.

``pythonpath = .`` in pytest.ini makes the project packages importable, so test
modules can ``from datastore.store import ...`` without sys.path manipulation.
"""
from __future__ import annotations

import pandas as pd
import pytest

from datastore.store import MarketDataStore


@pytest.fixture
def store(tmp_path) -> MarketDataStore:
    """A fresh MarketDataStore backed by a per-test temporary SQLite file."""
    return MarketDataStore(tmp_path / "market_data.db")


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Three daily UTC candles starting 2026-01-01 (open != close for ordering)."""
    ts = pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": [100.0, 110.0, 120.0],
            "high": [115.0, 125.0, 135.0],
            "low": [95.0, 105.0, 115.0],
            "close": [110.0, 120.0, 130.0],
            "volume": [10.0, 11.0, 12.0],
        }
    )


@pytest.fixture
def make_daily_ohlcv():
    """Factory: build ``periods`` daily UTC candles for property tests."""

    def _make(start: str = "2026-01-01", periods: int = 30) -> pd.DataFrame:
        ts = pd.date_range(start, periods=periods, freq="D", tz="UTC")
        n = len(ts)
        return pd.DataFrame(
            {
                "timestamp": ts,
                "open": [100.0 + i for i in range(n)],
                "high": [110.0 + i for i in range(n)],
                "low": [90.0 + i for i in range(n)],
                "close": [105.0 + i for i in range(n)],
                "volume": [10.0 + i for i in range(n)],
            }
        )

    return _make


@pytest.fixture
def gbm_ohlcv():
    """Factory: simulate a driftless-GBM daily OHLCV series (no network).

    Each candle is built from ``sub_steps`` intra-day log-return steps so the
    High/Low reflect a realistic intra-candle path — needed for meaningful
    *touch* (barrier) labels. Driftless with a known annual vol, so the
    closed-form GBM baseline is well-specified and the fitted models should be
    well-calibrated on it.
    """
    import numpy as np

    def _make(
        n_candles: int = 1500,
        sub_steps: int = 24,
        sigma_annual: float = 0.6,
        start_price: float = 100.0,
        seed: int = 7,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        seconds_per_year = 365.25 * 24 * 3600.0
        day = 24 * 3600.0
        step_vol = sigma_annual * np.sqrt(day / sub_steps / seconds_per_year)
        steps = rng.standard_normal(n_candles * sub_steps) * step_vol
        log_prices = np.log(start_price) + np.cumsum(steps)
        prices = np.exp(log_prices).reshape(n_candles, sub_steps)
        ts = pd.date_range("2022-01-01", periods=n_candles, freq="D", tz="UTC")
        return pd.DataFrame(
            {
                "timestamp": ts,
                "open": prices[:, 0],
                "high": prices.max(axis=1),
                "low": prices.min(axis=1),
                "close": prices[:, -1],
                "volume": np.full(n_candles, 100.0),
            }
        )

    return _make


@pytest.fixture
def sample_inventory() -> pd.DataFrame:
    """Inventory-shaped rows exercising every select_tradeable_universe branch."""
    return pd.DataFrame(
        [
            {  # selected
                "market": "Will BTC be above $100,000 by Dec 31?",
                "kind": "edge", "symbol": "BTC", "resolution_data_type": "candle_ohlcv",
                "liquidity_usd": 50000.0, "volume_30d_usd": 30000.0,
                "market_id": "m1", "clob_token_ids": ["t_yes", "t_no"],
                "outcomes": ["Yes", "No"],
            },
            {  # dropped: fails liquidity screen
                "market": "Will ETH be below $2,000 this week?",
                "kind": "edge", "symbol": "ETH", "resolution_data_type": "candle_ohlcv",
                "liquidity_usd": 100.0, "volume_30d_usd": 200.0,
                "market_id": "m2", "clob_token_ids": ["e_yes", "e_no"],
                "outcomes": ["Yes", "No"],
            },
            {  # dropped: not a candle market
                "market": "Will BTC dominance be above 60%?",
                "kind": "edge", "symbol": "BTC", "resolution_data_type": "daily_metric",
                "liquidity_usd": 90000.0, "volume_30d_usd": 90000.0,
                "market_id": "m3", "clob_token_ids": ["d_yes", "d_no"],
                "outcomes": ["Yes", "No"],
            },
            {  # dropped: symbol not in BTC/ETH
                "market": "Will SOL be above $300?",
                "kind": "edge", "symbol": "SOL", "resolution_data_type": "candle_ohlcv",
                "liquidity_usd": 90000.0, "volume_30d_usd": 90000.0,
                "market_id": "m4", "clob_token_ids": ["s_yes", "s_no"],
                "outcomes": ["Yes", "No"],
            },
            {  # dropped: no parseable strike
                "market": "Will BTC go up or down today?",
                "kind": "edge", "symbol": "BTC", "resolution_data_type": "candle_ohlcv",
                "liquidity_usd": 90000.0, "volume_30d_usd": 90000.0,
                "market_id": "m5", "clob_token_ids": ["u_yes", "u_no"],
                "outcomes": ["Yes", "No"],
            },
        ]
    )
