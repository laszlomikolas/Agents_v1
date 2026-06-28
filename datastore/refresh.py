"""Continuously-storable data refresh job.

``refresh_data`` is the "pull the latest data and store it" loop:

    1. Inventory live crypto markets and narrow to the v1 tradeable universe.
    2. Pull the latest underlying OHLCV per symbol (existing connectors) and
       upsert it (schema-validated).
    3. Snapshot the current YES-token midpoint for each market.
    4. Upsert per-market metadata (strike, direction, token id, ...).

It is designed to be run repeatedly (e.g. on a schedule); every write is an
idempotent upsert, so each run just appends the newest points. Network failures
on individual symbols/markets are logged and skipped rather than aborting the run.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

from connectors import SCHEMA_REGISTRY
from connectors.binance import fetch_binance_ohlcv
from connectors.schema_validation import validate_schema
from market_inventory.inventory import inventory_crypto_markets
from market_inventory.polymarket_clients import ClobClient, GammaClient
from market_inventory.tradeable_universe import select_tradeable_universe
from market_inventory.universe import CoinUniverse, ProjectUniverse

from datastore.store import MarketDataStore

logger = logging.getLogger(__name__)

# Map our canonical symbols to the connector used to fetch their OHLCV.
# Binance is the v1 primary (1000 candles/request, no auth).
SYMBOL_TO_BINANCE_PAIR: dict[str, str] = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}


def refresh_data(
    store: Optional[MarketDataStore] = None,
    *,
    gamma: Optional[GammaClient] = None,
    clob: Optional[ClobClient] = None,
    interval: str = "1d",
    ohlcv_limit: int = 1000,
    limit_events: int = 500,
    coins_path: str = "coins_universe.json",
    projects_path: str = "projects_universe.json",
    validate: bool = True,
) -> dict[str, Any]:
    """Pull the latest data for the tradeable universe and persist it.

    Args:
        store: Target store (defaults to a new ``MarketDataStore()``).
        gamma / clob: Polymarket clients (defaults constructed if omitted).
        interval: OHLCV candle interval to fetch/store.
        ohlcv_limit: Number of candles to request per symbol.
        limit_events: Max markets to inventory.
        coins_path / projects_path: Universe JSON files.
        validate: Run schema validation on fetched OHLCV before storing.

    Returns:
        A summary dict: counts of rows stored and any per-item errors.
    """
    store = store or MarketDataStore()
    gamma = gamma or GammaClient()
    clob = clob or ClobClient()

    coin_universe = CoinUniverse.from_json(coins_path)
    project_universe = ProjectUniverse.from_json(projects_path)

    inventory = inventory_crypto_markets(
        gamma=gamma,
        coin_universe=coin_universe,
        project_universe=project_universe,
        limit_events=limit_events,
    )
    universe = select_tradeable_universe(inventory)
    logger.info("Tradeable universe: %d markets", len(universe))

    summary: dict[str, Any] = {
        "markets": int(len(universe)),
        "ohlcv_rows": 0,
        "midpoints": 0,
        "ohlcv_errors": [],
        "midpoint_errors": [],
    }
    if universe.empty:
        return summary

    # 1. OHLCV per symbol.
    binance_spec = SCHEMA_REGISTRY["binance"][0]
    symbols = sorted(s for s in universe["symbol"].dropna().unique())
    for symbol in symbols:
        pair = SYMBOL_TO_BINANCE_PAIR.get(symbol)
        if pair is None:
            logger.warning("No OHLCV connector mapping for %s; skipping", symbol)
            summary["ohlcv_errors"].append(f"{symbol}: no connector mapping")
            continue
        try:
            ohlcv = fetch_binance_ohlcv(symbol=pair, interval=interval, limit=ohlcv_limit)
        except Exception as exc:  # noqa: BLE001 - keep refresh resilient
            logger.warning("OHLCV fetch failed for %s: %s", symbol, exc)
            summary["ohlcv_errors"].append(f"{symbol}: {exc}")
            continue
        if validate:
            result = validate_schema(ohlcv, binance_spec)
            if not result.ok:
                logger.warning("OHLCV schema check failed for %s: %s", symbol, result.errors)
                summary["ohlcv_errors"].append(f"{symbol}: schema validation failed: {result.errors}")
                continue
        summary["ohlcv_rows"] += store.upsert_ohlcv(symbol, interval, ohlcv)

    # 2. Current midpoints + 3. metadata.
    meta_records: list[dict[str, Any]] = []
    for _, row in universe.iterrows():
        token_id = row.get("yes_token_id")
        if token_id:
            try:
                midpoint = clob.get_midpoint(str(token_id))
                store.record_midpoint(str(token_id), midpoint)
                summary["midpoints"] += 1
            except Exception as exc:  # noqa: BLE001 - keep refresh resilient
                logger.warning("Midpoint fetch failed for %s: %s", token_id, exc)
                summary["midpoint_errors"].append(f"{token_id}: {exc}")
        meta_records.append(
            {
                "market_id": row.get("market_id"),
                "market": row.get("market"),
                "symbol": row.get("symbol"),
                "kind": row.get("kind"),
                "strike": row.get("strike"),
                "direction": row.get("direction"),
                "yes_token_id": token_id,
                "condition_id": row.get("condition_id"),
                "slug": row.get("slug"),
                "resolution_date": row.get("resolution_date"),
                "resolution_source": row.get("resolution_source"),
                "liquidity_usd": row.get("liquidity_usd"),
                "volume_30d_usd": row.get("volume_30d_usd"),
            }
        )

    store.upsert_market_meta(meta_records)
    return summary
