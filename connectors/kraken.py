"""Kraken public REST API connector – daily OHLCV."""
from __future__ import annotations

import requests
import pandas as pd
from typing import Optional

_BASE = "https://api.kraken.com/0/public"
_HEADERS = {"User-Agent": "Mozilla/5.0"}

# Common Kraken pair aliases → canonical pair name
PAIR_ALIASES: dict[str, str] = {
    "BTCUSD": "XXBTZUSD",
    "ETHUSD": "XETHZUSD",
    "BTCEUR": "XXBTZEUR",
    "ETHEUR": "XETHZEUR",
}


def fetch_kraken_ohlcv(
    pair: str = "XXBTZUSD",
    interval: int = 1440,
    since: Optional[int] = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV data from the Kraken public REST API.

    Endpoint: GET /0/public/OHLC
    No authentication required. Kraken returns up to 720 candles per request
    and always includes the most recent (possibly incomplete) candle.

    Args:
        pair: Kraken pair name, e.g. "XXBTZUSD". Simple aliases like "BTCUSD"
              are also accepted and are translated automatically.
        interval: Candle interval in minutes. Valid values: 1, 5, 15, 30, 60,
                  240, 1440 (1 day), 10080 (1 week), 21600.
        since: Unix timestamp; if provided, fetch candles since that time.

    Returns:
        DataFrame with columns: timestamp (UTC), open, high, low, close, volume.
    """
    canonical = PAIR_ALIASES.get(pair.upper(), pair)
    params: dict = {"pair": canonical, "interval": interval}
    if since is not None:
        params["since"] = since

    resp = requests.get(f"{_BASE}/OHLC", params=params, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if data.get("error"):
        raise RuntimeError(f"Kraken API error: {data['error']}")

    # The result dict has one key with the pair data and a "last" timestamp key
    result_key = next(k for k in data["result"] if k != "last")
    rows = data["result"][result_key]
    if not rows:
        raise RuntimeError(f"Kraken returned empty OHLC data for pair {canonical}")

    # Row layout: [time, open, high, low, close, vwap, volume, count]
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"])
    df["timestamp"] = pd.to_datetime(df["time"].astype(int), unit="s", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df[["timestamp", "open", "high", "low", "close", "volume"]].sort_values("timestamp").reset_index(drop=True)
