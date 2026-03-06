"""
Standard exchange / oracle connectors with built-in schema validation.

Each connector returns a pd.DataFrame with a consistent column layout.
All OHLCV connectors share the schema: timestamp (UTC), open, high, low,
close, volume. The Chainlink oracle connector returns: timestamp, round_id,
price.

Quick start
-----------
>>> from connectors import fetch_binance_ohlcv, check_all_connectors
>>> df = fetch_binance_ohlcv(symbol="BTCUSDT", limit=30)
>>> results = check_all_connectors()
>>> for r in results.values(): print(r.summary())
"""
from __future__ import annotations

import logging
from typing import Any

from connectors.binance import fetch_binance_ohlcv
from connectors.bitstamp import fetch_bitstamp_ohlcv
from connectors.bybit import fetch_bybit_ohlcv
from connectors.coinbase import fetch_coinbase_ohlcv
from connectors.coingecko import fetch_coingecko_ohlcv
from connectors.kraken import fetch_kraken_ohlcv
from connectors.okx import fetch_okx_ohlcv
from connectors.schema_validation import (
    ColumnSpec,
    SchemaSpec,
    ValidationResult,
    probe_connector,
    validate_schema,
)

logger = logging.getLogger(__name__)

__all__ = [
    # Connectors
    "fetch_binance_ohlcv",
    "fetch_bitstamp_ohlcv",
    "fetch_bybit_ohlcv",
    "fetch_coinbase_ohlcv",
    "fetch_coingecko_ohlcv",
    "fetch_kraken_ohlcv",
    "fetch_okx_ohlcv",
    # Validation
    "ColumnSpec",
    "SchemaSpec",
    "ValidationResult",
    "SCHEMA_REGISTRY",
    "check_all_connectors",
    "check_connector",
    "probe_connector",
    "validate_schema",
]

# ── Shared column specs ────────────────────────────────────────────────────────

def _ohlcv_cols() -> list[ColumnSpec]:
    """Return a fresh list of OHLCV ColumnSpecs (copied to avoid mutation)."""
    return [
        ColumnSpec("timestamp", "M"),
        ColumnSpec("open",      "f", min_value=0.0),
        ColumnSpec("high",      "f", min_value=0.0),
        ColumnSpec("low",       "f", min_value=0.0),
        ColumnSpec("close",     "f", min_value=0.0),
        ColumnSpec("volume",    "f", min_value=0.0),
    ]


# ── Schema registry ────────────────────────────────────────────────────────────
#
# Maps connector name → (SchemaSpec, connector_fn).
# probe_kwargs use small limits so validation probes are fast (~1–2 s each).

SCHEMA_REGISTRY: dict[str, tuple[SchemaSpec, Any]] = {
    "coingecko": (
        SchemaSpec(
            name="coingecko",
            columns=_ohlcv_cols(),
            check_ohlc_sanity=True,
            probe_kwargs={"coin_id": "bitcoin", "days": 90},
        ),
        fetch_coingecko_ohlcv,
    ),
    "binance": (
        SchemaSpec(
            name="binance",
            columns=_ohlcv_cols(),
            check_ohlc_sanity=True,
            probe_kwargs={"symbol": "BTCUSDT", "interval": "1d", "limit": 30},
        ),
        fetch_binance_ohlcv,
    ),
    "coinbase": (
        SchemaSpec(
            name="coinbase",
            columns=_ohlcv_cols(),
            check_ohlc_sanity=True,
            probe_kwargs={"product_id": "BTC-USD", "granularity": 86400},
        ),
        fetch_coinbase_ohlcv,
    ),
    "kraken": (
        SchemaSpec(
            name="kraken",
            columns=_ohlcv_cols(),
            check_ohlc_sanity=True,
            probe_kwargs={"pair": "XXBTZUSD", "interval": 1440},
        ),
        fetch_kraken_ohlcv,
    ),
    "bitstamp": (
        SchemaSpec(
            name="bitstamp",
            columns=_ohlcv_cols(),
            check_ohlc_sanity=True,
            probe_kwargs={"currency_pair": "btcusd", "step": 86400, "limit": 30},
        ),
        fetch_bitstamp_ohlcv,
    ),
    "okx": (
        SchemaSpec(
            name="okx",
            columns=_ohlcv_cols(),
            check_ohlc_sanity=True,
            probe_kwargs={"inst_id": "BTC-USDT", "bar": "1D", "limit": 30},
        ),
        fetch_okx_ohlcv,
    ),
    "bybit": (
        SchemaSpec(
            name="bybit",
            columns=_ohlcv_cols(),
            check_ohlc_sanity=True,
            probe_kwargs={"symbol": "BTCUSDT", "interval": "D", "limit": 30},
        ),
        fetch_bybit_ohlcv,
    ),
}


# ── Convenience helpers ────────────────────────────────────────────────────────

def check_connector(name: str, **override_kwargs: Any) -> ValidationResult:
    """
    Probe a single registered connector and return its ValidationResult.

    Args:
        name: Key in SCHEMA_REGISTRY, e.g. "binance", "chainlink".
        **override_kwargs: Override any of the spec's probe_kwargs.

    Returns:
        ValidationResult with ok=True iff the connector's output matches its
        expected schema (column names, dtypes, OHLC invariants, etc.).

    Raises:
        KeyError: If *name* is not found in SCHEMA_REGISTRY.
    """
    if name not in SCHEMA_REGISTRY:
        raise KeyError(f"Unknown connector '{name}'. Available: {sorted(SCHEMA_REGISTRY)}")
    spec, fn = SCHEMA_REGISTRY[name]
    return probe_connector(spec, fn, **override_kwargs)


def check_all_connectors(**per_connector_overrides: dict[str, Any]) -> dict[str, ValidationResult]:
    """
    Probe every connector in SCHEMA_REGISTRY and return all ValidationResults.

    Each connector is called with its registered probe_kwargs, optionally
    merged with per-connector overrides.

    Args:
        **per_connector_overrides: Mapping of connector_name → kwargs dict,
            e.g. check_all_connectors(binance={"symbol": "ETHUSDT"}).

    Returns:
        Dict mapping connector name → ValidationResult.

    Example::

        results = check_all_connectors()
        for r in results.values():
            print(r.summary())
    """
    results: dict[str, ValidationResult] = {}
    for name, (spec, fn) in SCHEMA_REGISTRY.items():
        override_kw = per_connector_overrides.get(name, {})
        logger.info("Probing connector: %s", name)
        results[name] = probe_connector(spec, fn, **override_kw)
    return results
