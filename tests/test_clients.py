"""Polymarket client tests (A3): price-history parsing + request building.

The request-building tests swap httpx.Client for a stub that records the query
params, so we verify the interval/fidelity defaults without any network call.
"""
import pandas as pd

import market_inventory.polymarket_clients as pmc
from market_inventory.polymarket_clients import ClobClient, parse_price_history


# ── parse_price_history ──────────────────────────────────────────────────────
def test_parse_price_history_dict_sorted_and_utc():
    payload = {"history": [{"t": 1700003600, "p": 0.55}, {"t": 1700000000, "p": 0.50}]}
    df = parse_price_history(payload)
    assert list(df.columns) == ["timestamp", "price"]
    assert df["price"].tolist() == [0.50, 0.55]          # sorted ascending by time
    assert str(df["timestamp"].dt.tz) == "UTC"


def test_parse_price_history_accepts_bare_list_and_price_key():
    # Bare list payload, and the 'price' key as an alternative to 'p'.
    df = parse_price_history([{"t": 1700000000, "price": 0.42}])
    assert len(df) == 1
    assert df["price"].iloc[0] == 0.42


def test_parse_price_history_skips_malformed_points():
    payload = {"history": [{"t": 1700000000}, {"p": 0.5}, "junk", {"t": 1700000060, "p": 0.5}]}
    df = parse_price_history(payload)
    assert len(df) == 1  # only the fully-formed point survives


def test_parse_price_history_empty():
    for payload in ({}, None, {"history": []}):
        df = parse_price_history(payload)
        assert list(df.columns) == ["timestamp", "price"]
        assert df.empty


# ── get_price_history request building (no network) ──────────────────────────
class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, recorder, payload):
        self._recorder = recorder
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, params=None):
        self._recorder["url"] = url
        self._recorder["params"] = params
        return _FakeResponse(self._payload)


def _patch_clob(monkeypatch, payload=None):
    """Patch httpx.Client used by the clients module; return a params recorder."""
    recorder: dict = {}
    payload = {"history": []} if payload is None else payload
    monkeypatch.setattr(pmc.httpx, "Client", lambda *a, **k: _FakeClient(recorder, payload))
    return recorder


def test_get_price_history_defaults_to_max(monkeypatch):
    rec = _patch_clob(monkeypatch)
    result = ClobClient().get_price_history("tok")
    assert rec["params"]["market"] == "tok"
    assert rec["params"]["interval"] == "max"  # no range/interval -> max
    assert "fidelity" not in rec["params"]      # max needs no fidelity
    assert isinstance(result, pd.DataFrame)


def test_get_price_history_bounded_interval_injects_fidelity(monkeypatch):
    rec = _patch_clob(monkeypatch)
    ClobClient().get_price_history("tok", interval="1w")
    assert rec["params"]["interval"] == "1w"
    assert rec["params"]["fidelity"] == 60  # bounded interval requires fidelity


def test_get_price_history_explicit_range_passthrough(monkeypatch):
    rec = _patch_clob(monkeypatch)
    ClobClient().get_price_history("tok", start_ts=100, end_ts=200, fidelity=5)
    assert rec["params"] == {"market": "tok", "startTs": 100, "endTs": 200, "fidelity": 5}
