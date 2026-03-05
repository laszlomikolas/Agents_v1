"""Bitstamp public REST API connector – daily OHLCV."""
from __future__ import annotations

import requests
import pandas as pd
from typing import Optional

_BASE = "https://www.bitstamp.net/api/v2"
_HEADERS = {"User-Agent": "Mozilla/5.0"}
_MAX_LIMIT = 1000


def fetch_bitstamp_ohlcv(
    currency_pair: str = "btcusd",
    step: int = 86400,
    limit: int = 1000,
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV data from the Bitstamp public REST API.

    Endpoint: GET /api/v2/ohlc/{currency_pair}/
    No authentication required.

    Args:
        currency_pair: Lowercase pair string, e.g. "btcusd", "ethusd",
                       "xrpusd".
        step: Candle size in seconds. Supported values: 60, 180, 300, 900,
              1800, 3600, 7200, 14400, 21600, 43200, 86400, 259200.
        limit: Number of candles to return (max 1000).
        start: Optional Unix timestamp for window start.
        end: Optional Unix timestamp for window end.

    Returns:
        DataFrame with columns: timestamp (UTC), open, high, low, close, volume.
    """
    params: dict = {"step": step, "limit": min(limit, _MAX_LIMIT)}
    if start is not None:
        params["start"] = start
    if end is not None:
        params["end"] = end

    resp = requests.get(
        f"{_BASE}/ohlc/{currency_pair}/",
        params=params,
        headers=_HEADERS,
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"Bitstamp API error {resp.status_code}: {resp.text[:200]}"
        )

    data = resp.json()
    ohlc_list = data.get("data", {}).get("ohlc", [])
    if not ohlc_list:
        raise RuntimeError(
            f"Bitstamp returned empty OHLC data for {currency_pair}"
        )

    df = pd.DataFrame(ohlc_list)
    # Response fields: timestamp (unix str), open, high, low, close, volume
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df[["timestamp", "open", "high", "low", "close", "volume"]].sort_values("timestamp").reset_index(drop=True)
