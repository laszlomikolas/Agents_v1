"""Bybit public REST API connector – daily OHLCV."""
from __future__ import annotations

import requests
import pandas as pd
from typing import Optional

_BASE = "https://api.bybit.com/v5/market"
_HEADERS = {"User-Agent": "Mozilla/5.0"}
_MAX_LIMIT = 200


def fetch_bybit_ohlcv(
    symbol: str = "BTCUSDT",
    interval: str = "D",
    limit: int = 200,
    category: str = "linear",
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV kline data from the Bybit public REST API v5.

    Endpoint: GET /v5/market/kline
    No authentication required. Returns up to 200 candles per request in
    reverse chronological order (newest first); this function returns them
    sorted ascending.

    Args:
        symbol: Trading pair symbol, e.g. "BTCUSDT", "ETHUSDT".
        interval: Kline interval: 1, 3, 5, 15, 30, 60, 120, 240, 360, 720,
                  D (day), W (week), M (month).
        limit: Number of candles (max 200).
        category: "linear" (USDT-margined), "inverse", or "spot".
        start: Start time as Unix milliseconds (optional).
        end: End time as Unix milliseconds (optional).

    Returns:
        DataFrame with columns: timestamp (UTC), open, high, low, close, volume.
    """
    params: dict = {
        "category": category,
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": min(limit, _MAX_LIMIT),
    }
    if start is not None:
        params["start"] = start
    if end is not None:
        params["end"] = end

    resp = requests.get(
        f"{_BASE}/kline",
        params=params,
        headers=_HEADERS,
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Bybit API error {resp.status_code}: {resp.text[:200]}")

    body = resp.json()
    if body.get("retCode") != 0:
        raise RuntimeError(
            f"Bybit API error: retCode={body.get('retCode')} retMsg={body.get('retMsg')}"
        )

    rows = body.get("result", {}).get("list", [])
    if not rows:
        raise RuntimeError(f"Bybit returned empty kline data for {symbol}")

    # Row layout: [startTime_ms, open, high, low, close, volume, turnover]
    df = pd.DataFrame(
        rows,
        columns=["start_ms", "open", "high", "low", "close", "volume", "turnover"],
    )
    df["timestamp"] = pd.to_datetime(df["start_ms"].astype(int), unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df[["timestamp", "open", "high", "low", "close", "volume"]].sort_values("timestamp").reset_index(drop=True)
