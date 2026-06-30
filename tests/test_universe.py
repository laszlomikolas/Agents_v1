"""Tradeable-universe selector tests, including the financially-critical
YES-token selection (a wrong token id means betting the wrong side)."""
import pandas as pd

from market_inventory.tradeable_universe import select_tradeable_universe


def _row(**overrides) -> dict:
    base = {
        "market": "Will BTC be above $100,000 by Dec 31?",
        "kind": "edge", "symbol": "BTC", "resolution_data_type": "candle_ohlcv",
        "liquidity_usd": 50000.0, "volume_30d_usd": 30000.0,
        "market_id": "m1", "clob_token_ids": ["t_yes", "t_no"],
        "outcomes": ["Yes", "No"],
    }
    base.update(overrides)
    return base


# ── end-to-end selection ─────────────────────────────────────────────────────
def test_select_keeps_only_qualifying_row(sample_inventory):
    out = select_tradeable_universe(sample_inventory)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["market_id"] == "m1"
    assert row["symbol"] == "BTC"
    assert row["strike"] == 100000.0
    assert row["direction"] == "above"
    assert row["yes_token_id"] == "t_yes"


def test_select_empty_input():
    assert select_tradeable_universe(pd.DataFrame()).empty


def test_select_below_direction_kept():
    df = pd.DataFrame([_row(market="Will ETH fall below $2,000?", symbol="ETH",
                            clob_token_ids=["y", "n"])])
    out = select_tradeable_universe(df)
    assert len(out) == 1
    assert out.iloc[0]["direction"] == "below"
    assert out.iloc[0]["strike"] == 2000.0


def test_select_passes_on_volume_only():
    # Below liquidity threshold but above 30d-volume threshold -> still passes.
    df = pd.DataFrame([_row(liquidity_usd=100.0, volume_30d_usd=50000.0)])
    assert len(select_tradeable_universe(df)) == 1


# ── YES-token selection (correctness-critical) ───────────────────────────────
def test_yes_token_picked_by_label_not_index():
    # Outcomes reversed: the YES token is the SECOND id. Must not blindly take [0].
    df = pd.DataFrame([_row(outcomes=["No", "Yes"], clob_token_ids=["NO_TOK", "YES_TOK"])])
    out = select_tradeable_universe(df)
    assert out.iloc[0]["yes_token_id"] == "YES_TOK"


def test_yes_token_fallback_on_length_mismatch():
    # outcomes/token_ids length mismatch -> fall back to the first token id.
    df = pd.DataFrame([_row(outcomes=["Yes", "No", "Maybe"], clob_token_ids=["a", "b"])])
    out = select_tradeable_universe(df)
    assert out.iloc[0]["yes_token_id"] == "a"


def test_row_dropped_when_token_ids_missing():
    df = pd.DataFrame([_row(clob_token_ids=None)])
    assert select_tradeable_universe(df).empty
