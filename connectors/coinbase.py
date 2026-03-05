"""Coinbase Exchange (Advanced Trade) public REST API connector – daily OHLCV."""
from __future__ import annotations

import datetime
import requests
import pandas as pd
from typing import Optional

_BASE = "https://api.exchange.coinbase.com"
_HEADERS = {"User-Agent": "Mozilla/5.0"}
_MAX_CANDLES = 300  # Coinbase Exchange hard limit per request


def fetch_coinbase_ohlcv(
    product_id: str = "BTC-USD",
    granularity: int = 86400,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV candle data from the Coinbase Exchange public REST API.

    Endpoint: GET /products/{product_id}/candles
    No authentication required. Returns up to 300 candles per request.

    For windows larger than 300 * granularity seconds, call the function
    multiple times with paginated start/end values.

    Args:
        product_id: Trading pair, e.g. "BTC-USD", "ETH-USD".
        granularity: Candle size in seconds. Supported: 60, 300, 900, 3600,
                     21600, 86400 (1 day).
        start: ISO 8601 datetime string for window start (UTC). Defaults to
               300 candles before end.
        end: ISO 8601 datetime string for window end (UTC). Defaults to now.

    Returns:
        DataFrame with columns: timestamp (UTC), open, high, low, close, volume.
    """
    end_dt = (
        datetime.datetime.fromisoformat(end)
        if end
        else datetime.datetime.utcnow()
    )
    start_dt = (
        datetime.datetime.fromisoformat(start)
        if start
        else end_dt - datetime.timedelta(seconds=granularity * _MAX_CANDLES)
    )

    params = {
        "granularity": granularity,
        "start": start_dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "end": end_dt.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    resp = requests.get(
        f"{_BASE}/products/{product_id}/candles",
        params=params,
        headers=_HEADERS,
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"Coinbase API error {resp.status_code}: {resp.text[:200]}"
        )

    raw = resp.json()
    if not raw:
        raise RuntimeError(f"Coinbase returned empty candle data for {product_id}")

    # Row layout: [time (unix seconds), low, high, open, close, volume]
    df = pd.DataFrame(raw, columns=["time", "low", "high", "open", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df[["timestamp", "open", "high", "low", "close", "volume"]].sort_values("timestamp").reset_index(drop=True)
