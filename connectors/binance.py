"""Binance public REST API connector – daily OHLCV klines."""
from __future__ import annotations

import requests
import pandas as pd
from typing import Optional

_BASE = "https://api.binance.com/api/v3"
_HEADERS = {"User-Agent": "Mozilla/5.0"}
_MAX_LIMIT = 1000


def fetch_binance_ohlcv(
    symbol: str = "BTCUSDT",
    interval: str = "1d",
    limit: int = 500,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV kline data from the Binance public REST API.

    Endpoint: GET /api/v3/klines
    No authentication required. Returns up to 1000 candles per request.

    Args:
        symbol: Trading pair symbol, e.g. "BTCUSDT", "ETHUSDT".
        interval: Kline interval: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h,
                  8h, 12h, 1d, 3d, 1w, 1M.
        limit: Number of candles (max 1000).
        start_time: Start time as Unix milliseconds (optional).
        end_time: End time as Unix milliseconds (optional).

    Returns:
        DataFrame with columns: timestamp (UTC), open, high, low, close, volume.
    """
    params: dict = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": min(limit, _MAX_LIMIT),
    }
    if start_time is not None:
        params["startTime"] = start_time
    if end_time is not None:
        params["endTime"] = end_time

    resp = requests.get(
        f"{_BASE}/klines",
        params=params,
        headers=_HEADERS,
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Binance API error {resp.status_code}: {resp.text[:200]}")

    raw = resp.json()
    if not raw:
        raise RuntimeError(f"Binance returned empty kline data for {symbol}")

    # Row layout: [open_time, open, high, low, close, volume, close_time,
    #              quote_asset_volume, num_trades, taker_buy_base,
    #              taker_buy_quote, ignore]
    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "n_trades",
        "taker_buy_base", "taker_buy_quote", "_ignore",
    ])
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df[["timestamp", "open", "high", "low", "close", "volume"]].sort_values("timestamp").reset_index(drop=True)
