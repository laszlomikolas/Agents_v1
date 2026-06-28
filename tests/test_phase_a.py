"""Offline tests for Phase A (data foundation).

No network required. Runnable either with pytest::

    pytest tests/test_phase_a.py

or directly::

    python tests/test_phase_a.py
"""
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from datastore.store import MarketDataStore
from market_inventory.inventory import parse_clob_token_ids, parse_outcome_prices
from market_inventory.polymarket_clients import parse_price_history
from market_inventory.text_utils import parse_threshold
from market_inventory.tradeable_universe import select_tradeable_universe


# ── A2: threshold parsing ────────────────────────────────────────────────────
def test_parse_threshold_above():
    assert parse_threshold("Will BTC be above $100,000 by Dec 31?") == (100000.0, "above")
    assert parse_threshold("Will Bitcoin reach $150k in 2026?") == (150000.0, "above")
    assert parse_threshold("Will ETH hit $5,000?") == (5000.0, "above")


def test_parse_threshold_below():
    assert parse_threshold("Will Ethereum dip below $2k this week?") == (2000.0, "below")
    assert parse_threshold("Will BTC fall under $80,000?") == (80000.0, "below")


def test_parse_threshold_suffixes_and_decimals():
    assert parse_threshold("Will FOO exceed $1.2M?") == (1_200_000.0, "above")
    assert parse_threshold("Will BAR be above $0.50?") == (0.50, "above")


def test_parse_threshold_unparseable():
    # Range markets are not single thresholds.
    assert parse_threshold("Will BTC be between $90k and $100k?") == (None, None)
    # No price level present.
    strike, direction = parse_threshold("Will BTC go up or down today?")
    assert strike is None
    # Bare years/dates must not be mistaken for strikes.
    strike2, _ = parse_threshold("Will BTC do something in 2026?")
    assert strike2 is None


# ── A1: identifier parsing helpers ───────────────────────────────────────────
def test_parse_clob_token_ids():
    assert parse_clob_token_ids('["111", "222"]') == ["111", "222"]
    assert parse_clob_token_ids([111, 222]) == ["111", "222"]
    assert parse_clob_token_ids(None) is None
    assert parse_clob_token_ids("not json") is None


def test_parse_outcome_prices():
    assert parse_outcome_prices('["1", "0"]') == [1.0, 0.0]
    assert parse_outcome_prices(["0.6", 0.4]) == [0.6, 0.4]
    assert parse_outcome_prices(None) is None
    assert parse_outcome_prices('["x"]') == [None]


# ── A3: CLOB price-history parsing ───────────────────────────────────────────
def test_parse_price_history_dict():
    payload = {"history": [{"t": 1700003600, "p": 0.55}, {"t": 1700000000, "p": 0.50}]}
    df = parse_price_history(payload)
    assert list(df.columns) == ["timestamp", "price"]
    assert len(df) == 2
    # Sorted ascending by time.
    assert df["price"].tolist() == [0.50, 0.55]
    assert str(df["timestamp"].dt.tz) == "UTC"


def test_parse_price_history_empty():
    assert list(parse_price_history({}).columns) == ["timestamp", "price"]
    assert parse_price_history({}).empty
    assert parse_price_history(None).empty


# ── A4: SQLite store ─────────────────────────────────────────────────────────
def _sample_ohlcv() -> pd.DataFrame:
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


def test_store_ohlcv_roundtrip_and_idempotent():
    with tempfile.TemporaryDirectory() as tmp:
        store = MarketDataStore(Path(tmp) / "test.db")
        df = _sample_ohlcv()
        store.upsert_ohlcv("BTC", "1d", df)
        # Re-upserting the same rows must not duplicate (idempotent PK).
        store.upsert_ohlcv("BTC", "1d", df)

        out = store.read_ohlcv("BTC", "1d")
        assert len(out) == 3
        assert out["close"].tolist() == [110.0, 120.0, 130.0]
        assert str(out["timestamp"].dt.tz) == "UTC"

        # as-of slice filters by candle CLOSE time, not open time. At the
        # open of day 2 only day 1's candle has fully closed, so the slice
        # contains exactly one candle.
        asof = df["timestamp"].iloc[1]
        sliced = store.read_ohlcv_asof("BTC", "1d", asof)
        assert len(sliced) == 1
        assert sliced["close"].tolist() == [110.0]

        # At the open of day 3, day 1 and day 2 candles have closed.
        sliced2 = store.read_ohlcv_asof("BTC", "1d", df["timestamp"].iloc[2])
        assert len(sliced2) == 2


def test_store_midpoints_and_meta():
    with tempfile.TemporaryDirectory() as tmp:
        store = MarketDataStore(Path(tmp) / "test.db")

        store.record_midpoint("tok1", 0.62, ts=pd.Timestamp("2026-01-01", tz="UTC"))
        store.record_midpoint("tok1", 0.64, ts=pd.Timestamp("2026-01-02", tz="UTC"))
        prices = store.read_market_prices("tok1")
        assert len(prices) == 2
        assert prices["midpoint"].tolist() == [0.62, 0.64]

        store.upsert_market_meta(
            [
                {
                    "market_id": "m1",
                    "market": "Will BTC be above $100k?",
                    "symbol": "BTC",
                    "kind": "edge",
                    "strike": 100000.0,
                    "direction": "above",
                    "yes_token_id": "tok1",
                    "resolution_date": pd.Timestamp("2026-12-31", tz="UTC"),
                    "liquidity_usd": 50000.0,
                    "volume_30d_usd": 25000.0,
                }
            ]
        )
        meta = store.read_meta("m1")
        assert len(meta) == 1
        assert meta.iloc[0]["strike"] == 100000.0
        assert meta.iloc[0]["direction"] == "above"
        assert meta.iloc[0]["yes_token_id"] == "tok1"


# ── Tradeable-universe selector ──────────────────────────────────────────────
def _sample_inventory() -> pd.DataFrame:
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


def test_select_tradeable_universe():
    out = select_tradeable_universe(_sample_inventory())
    assert len(out) == 1
    row = out.iloc[0]
    assert row["market_id"] == "m1"
    assert row["symbol"] == "BTC"
    assert row["strike"] == 100000.0
    assert row["direction"] == "above"
    assert row["yes_token_id"] == "t_yes"


def test_select_tradeable_universe_empty():
    out = select_tradeable_universe(pd.DataFrame())
    assert out.empty


# ── Direct runner (no pytest dependency) ─────────────────────────────────────
def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
