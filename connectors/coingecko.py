"""CoinGecko public REST API connector – daily OHLCV."""
from __future__ import annotations

import requests
import pandas as pd
from typing import Optional

_BASE = "https://api.coingecko.com/api/v3"
_HEADERS = {"User-Agent": "Mozilla/5.0"}


def fetch_coingecko_ohlcv(
    coin_id: str = "bitcoin",
    vs_currency: str = "usd",
    days: int = 365,
) -> pd.DataFrame:
    """
    Fetch daily OHLCV data from the CoinGecko public API.

    Uses /coins/{id}/ohlc for OHLC candles and /coins/{id}/market_chart for
    daily volume, then joins on UTC date.

    Note: CoinGecko free tier is rate-limited (~10–30 req/min). For days ≥ 90
    the OHLC endpoint returns daily candles; below 90 it returns 4-hour candles.

    Args:
        coin_id: CoinGecko coin slug, e.g. "bitcoin", "ethereum".
        vs_currency: Quote currency, e.g. "usd".
        days: Number of past days to fetch (use a value ≥ 90 for daily candles).

    Returns:
        DataFrame with columns: timestamp (UTC), open, high, low, close, volume.
    """
    ohlc_resp = requests.get(
        f"{_BASE}/coins/{coin_id}/ohlc",
        params={"vs_currency": vs_currency, "days": days},
        headers=_HEADERS,
        timeout=30,
    )
    if ohlc_resp.status_code != 200:
        raise RuntimeError(
            f"CoinGecko OHLC error {ohlc_resp.status_code}: {ohlc_resp.text[:200]}"
        )
    ohlc_data = ohlc_resp.json()  # [[ts_ms, open, high, low, close], ...]
    if not ohlc_data:
        raise RuntimeError(f"CoinGecko returned empty OHLC data for {coin_id}")

    chart_resp = requests.get(
        f"{_BASE}/coins/{coin_id}/market_chart",
        params={"vs_currency": vs_currency, "days": days, "interval": "daily"},
        headers=_HEADERS,
        timeout=30,
    )
    if chart_resp.status_code != 200:
        raise RuntimeError(
            f"CoinGecko market_chart error {chart_resp.status_code}: {chart_resp.text[:200]}"
        )
    chart_data = chart_resp.json()

    ohlc_df = pd.DataFrame(ohlc_data, columns=["ts_ms", "open", "high", "low", "close"])
    ohlc_df["timestamp"] = pd.to_datetime(ohlc_df["ts_ms"], unit="ms", utc=True)
    ohlc_df["_date"] = ohlc_df["timestamp"].dt.date

    vol_df = pd.DataFrame(chart_data["total_volumes"], columns=["ts_ms", "volume"])
    vol_df["timestamp"] = pd.to_datetime(vol_df["ts_ms"], unit="ms", utc=True)
    vol_df["_date"] = vol_df["timestamp"].dt.date

    merged = ohlc_df.merge(vol_df[["_date", "volume"]], on="_date", how="left")
    result = merged[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    for col in ["open", "high", "low", "close", "volume"]:
        result[col] = result[col].astype(float)
    return result.sort_values("timestamp").reset_index(drop=True)
