"""MarketDataStore tests (A4): round-trips, idempotency, edge cases, and the
no-look-ahead (as-of) invariant that underpins backtest validity."""
import random

import pandas as pd
import pytest

from datastore.store import _interval_to_seconds


# ── interval helper ──────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "interval, seconds",
    [
        ("1m", 60), ("5m", 300), ("1h", 3600), ("1H", 3600),
        ("1d", 86400), ("1D", 86400), ("1w", 604800),
        ("1M", 2592000), ("3M", 3 * 2592000),  # 'M' month, distinct from 'm'
    ],
)
def test_interval_to_seconds(interval, seconds):
    assert _interval_to_seconds(interval) == seconds


@pytest.mark.parametrize("bad", ["", "1x", "d", "0d", "-1d", "abc"])
def test_interval_to_seconds_invalid(bad):
    with pytest.raises(ValueError):
        _interval_to_seconds(bad)


# ── OHLCV round-trip & idempotency ───────────────────────────────────────────
def test_ohlcv_roundtrip_and_idempotent(store, sample_ohlcv):
    store.upsert_ohlcv("BTC", "1d", sample_ohlcv)
    store.upsert_ohlcv("BTC", "1d", sample_ohlcv)  # re-upsert must not duplicate

    out = store.read_ohlcv("BTC", "1d")
    assert len(out) == 3
    assert out["close"].tolist() == [110.0, 120.0, 130.0]
    assert str(out["timestamp"].dt.tz) == "UTC"


def test_symbol_interval_isolation(store, sample_ohlcv):
    store.upsert_ohlcv("BTC", "1d", sample_ohlcv)
    store.upsert_ohlcv("BTC", "1h", sample_ohlcv)
    store.upsert_ohlcv("ETH", "1d", sample_ohlcv)
    assert len(store.read_ohlcv("BTC", "1d")) == 3
    assert len(store.read_ohlcv("BTC", "1h")) == 3
    assert len(store.read_ohlcv("ETH", "1d")) == 3
    assert store.read_ohlcv("SOL", "1d").empty


def test_range_reads(store, make_daily_ohlcv):
    df = make_daily_ohlcv(periods=10)
    store.upsert_ohlcv("BTC", "1d", df)
    out = store.read_ohlcv("BTC", "1d", start=df["timestamp"].iloc[2], end=df["timestamp"].iloc[5])
    assert out["timestamp"].tolist() == df["timestamp"].iloc[2:6].tolist()


# ── edge cases ───────────────────────────────────────────────────────────────
def test_upsert_empty_returns_zero(store):
    assert store.upsert_ohlcv("BTC", "1d", pd.DataFrame()) == 0
    assert store.upsert_market_prices("tok", pd.DataFrame()) == 0


def test_upsert_missing_timestamp_raises(store):
    bad = pd.DataFrame({"open": [1.0]})
    with pytest.raises(ValueError):
        store.upsert_ohlcv("BTC", "1d", bad)
    with pytest.raises(ValueError):
        store.upsert_market_prices("tok", bad)


def test_nan_and_nat_handling(store):
    ts = pd.to_datetime(["2026-01-01", None, "2026-01-03"], utc=True)  # NaT in middle
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [1.0, 2.0, float("nan")],  # NaN price -> NULL
            "high": [1.0, 2.0, 3.0], "low": [1.0, 2.0, 3.0],
            "close": [1.0, 2.0, 3.0], "volume": [1.0, 2.0, 3.0],
        }
    )
    n = store.upsert_ohlcv("BTC", "1d", df)
    assert n == 2  # NaT timestamp row dropped
    out = store.read_ohlcv("BTC", "1d")
    assert len(out) == 2
    assert out["open"].isna().sum() == 1


# ── midpoints & metadata ─────────────────────────────────────────────────────
def test_midpoints_roundtrip(store):
    store.record_midpoint("tok1", 0.62, ts=pd.Timestamp("2026-01-01", tz="UTC"))
    store.record_midpoint("tok1", 0.64, ts=pd.Timestamp("2026-01-02", tz="UTC"))
    prices = store.read_market_prices("tok1")
    assert prices["midpoint"].tolist() == [0.62, 0.64]


def test_upsert_market_prices_accepts_price_column(store):
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC"),
            "price": [0.4, 0.6],  # 'price' alias for 'midpoint'
        }
    )
    assert store.upsert_market_prices("tok", df) == 2
    assert store.read_market_prices("tok")["midpoint"].tolist() == [0.4, 0.6]


def test_meta_roundtrip(store):
    store.upsert_market_meta(
        [
            {
                "market_id": "m1", "market": "Will BTC be above $100k?",
                "symbol": "BTC", "kind": "edge", "strike": 100000.0,
                "direction": "above", "yes_token_id": "tok1",
                "resolution_date": pd.Timestamp("2026-12-31", tz="UTC"),
                "liquidity_usd": 50000.0, "volume_30d_usd": 25000.0,
            }
        ]
    )
    meta = store.read_meta("m1")
    assert len(meta) == 1
    assert meta.iloc[0]["strike"] == 100000.0
    assert meta.iloc[0]["direction"] == "above"
    assert meta.iloc[0]["yes_token_id"] == "tok1"


def test_meta_idempotent_update(store):
    rec = {"market_id": "m1", "strike": 100000.0, "direction": "above", "yes_token_id": "a"}
    store.upsert_market_meta([rec])
    store.upsert_market_meta([{**rec, "strike": 120000.0}])  # same id -> update, not insert
    meta = store.read_meta("m1")
    assert len(meta) == 1
    assert meta.iloc[0]["strike"] == 120000.0


def test_meta_falls_back_to_slug_when_no_market_id(store):
    store.upsert_market_meta([{"slug": "btc-100k", "strike": 100000.0}])
    meta = store.read_meta("btc-100k")
    assert len(meta) == 1


# ── no-look-ahead (as-of) invariant ──────────────────────────────────────────
def test_asof_boundary_includes_exact_close(store, sample_ohlcv):
    store.upsert_ohlcv("BTC", "1d", sample_ohlcv)
    # Day-1 candle (open 2026-01-01) closes exactly at 2026-01-02 00:00 UTC.
    close_day1 = sample_ohlcv["timestamp"].iloc[0] + pd.Timedelta(days=1)
    out = store.read_ohlcv_asof("BTC", "1d", close_day1)
    assert len(out) == 1                       # included at the exact close instant
    assert out["close"].tolist() == [110.0]


def test_asof_never_returns_unclosed_candle(store, make_daily_ohlcv):
    df = make_daily_ohlcv(periods=30)
    store.upsert_ohlcv("BTC", "1d", df)
    interval_s = 86400
    # Resolution-independent epoch seconds (pandas may use us/ns under the hood).
    closes = df["timestamp"].apply(lambda t: int(t.timestamp())) + interval_s

    start = int(df["timestamp"].iloc[0].timestamp())
    end = int(df["timestamp"].iloc[-1].timestamp()) + 2 * interval_s
    rng = random.Random(20260629)
    for _ in range(300):
        asof_epoch = rng.randint(start - interval_s, end)
        asof = pd.Timestamp(asof_epoch, unit="s", tz="UTC")
        out = store.read_ohlcv_asof("BTC", "1d", asof)

        # Invariant: never a candle whose close is still in the future.
        if not out.empty:
            last_close = out["timestamp"].iloc[-1].timestamp() + interval_s
            assert last_close <= asof_epoch
        # And the slice is exactly the set of already-closed candles.
        assert len(out) == int((closes <= asof_epoch).sum())
