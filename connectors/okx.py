"""OKX public REST API connector – daily OHLCV."""
from __future__ import annotations

import requests
import pandas as pd
from typing import Optional

_BASE = "https://www.okx.com/api/v5/market"
_HEADERS = {"User-Agent": "Mozilla/5.0"}
_MAX_LIMIT = 300


def fetch_okx_ohlcv(
    inst_id: str = "BTC-USDT",
    bar: str = "1D",
    limit: int = 300,
    after: Optional[str] = None,
    before: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV candle data from the OKX public REST API.

    Uses /api/v5/market/candles which returns up to 300 candles. For data
    older than ~6.5 months use /api/v5/market/history-candles instead (same
    schema; swap the endpoint URL).

    OKX returns candles in descending order (newest first); this function
    returns them sorted ascending.

    Args:
        inst_id: Instrument ID, e.g. "BTC-USDT", "ETH-USDT".
        bar: Candle granularity: 1m, 3m, 5m, 15m, 30m, 1H, 2H, 4H, 6H, 12H,
             1D, 1W, 1M, 3M.
        limit: Number of candles (max 300).
        after: Pagination cursor – return candles with timestamp < after (Unix ms).
        before: Pagination cursor – return candles with timestamp > before (Unix ms).

    Returns:
        DataFrame with columns: timestamp (UTC), open, high, low, close, volume.
    """
    params: dict = {"instId": inst_id, "bar": bar, "limit": min(limit, _MAX_LIMIT)}
    if after is not None:
        params["after"] = after
    if before is not None:
        params["before"] = before

    resp = requests.get(
        f"{_BASE}/candles",
        params=params,
        headers=_HEADERS,
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"OKX API error {resp.status_code}: {resp.text[:200]}")

    body = resp.json()
    if body.get("code") != "0":
        raise RuntimeError(
            f"OKX API error: code={body.get('code')} msg={body.get('msg')}"
        )

    rows = body.get("data", [])
    if not rows:
        raise RuntimeError(f"OKX returned empty candle data for {inst_id}")

    # Row layout: [ts_ms, open, high, low, close, vol, volCcy, volCcyQuote, confirm]
    df = pd.DataFrame(
        rows,
        columns=["ts_ms", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm"],
    )
    df["timestamp"] = pd.to_datetime(df["ts_ms"].astype(int), unit="ms", utc=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df["volume"] = df["vol"].astype(float)
    return df[["timestamp", "open", "high", "low", "close", "volume"]].sort_values("timestamp").reset_index(drop=True)
