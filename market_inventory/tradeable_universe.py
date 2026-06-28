"""Select the v1 tradeable universe from a normalized inventory DataFrame.

The first strategy trades binary price-threshold markets on majors (BTC/ETH)
that resolve against price candles — these already have working OHLCV
connectors. This module narrows a full inventory down to those markets and
enriches each row with the fields needed to trade and model it:

    strike        – numeric price level the question is about
    direction     – "above" or "below"
    yes_token_id  – CLOB token id for the YES outcome (for prices/midpoints)

Rows that cannot be fully resolved (missing strike, direction, or token id)
are dropped, so the output is directly tradeable.
"""
from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

from .liquidity_screen import apply_liquidity_screen
from .text_utils import parse_threshold

DEFAULT_SYMBOLS: tuple[str, ...] = ("BTC", "ETH")


def _yes_token_id(row: pd.Series) -> Optional[str]:
    """Return the CLOB token id for the YES outcome of a market row.

    Aligns ``outcomes`` with ``clob_token_ids`` by index; falls back to the
    first token id (Polymarket lists YES first by convention).
    """
    token_ids = row.get("clob_token_ids")
    if not isinstance(token_ids, (list, tuple)) or not token_ids:
        return None

    outcomes = row.get("outcomes")
    if isinstance(outcomes, (list, tuple)) and len(outcomes) == len(token_ids):
        for outcome, token_id in zip(outcomes, token_ids):
            if str(outcome).strip().lower() == "yes":
                return str(token_id)

    return str(token_ids[0])


def select_tradeable_universe(
    df: pd.DataFrame,
    symbols: Sequence[str] = DEFAULT_SYMBOLS,
    *,
    min_liquidity_usd: float = 10_000.0,
    min_volume_30d_usd: float = 10_000.0,
) -> pd.DataFrame:
    """Filter an inventory DataFrame down to the v1 tradeable universe.

    Steps:
        1. Keep binary ("edge") price-candle markets on the target symbols.
        2. Apply the liquidity screen and keep only passers.
        3. Parse strike + direction from the question text.
        4. Resolve the YES token id.
        5. Drop rows missing any of strike / direction / yes_token_id.

    Args:
        df: Inventory DataFrame from ``inventory_crypto_markets`` (must include
            the identifier columns added in Phase A: ``clob_token_ids`` etc.).
        symbols: Symbols to include (default BTC/ETH).
        min_liquidity_usd: Liquidity-screen threshold (resting depth).
        min_volume_30d_usd: Liquidity-screen fallback (30-day volume).

    Returns:
        A new DataFrame with added columns ``strike``, ``direction`` and
        ``yes_token_id``, reset index. Empty if nothing qualifies.
    """
    columns = ["strike", "direction", "yes_token_id"]
    if df is None or df.empty:
        return pd.DataFrame(columns=list(getattr(df, "columns", [])) + columns)

    symbol_set = {s.upper() for s in symbols}
    mask = (
        (df["kind"] == "edge")
        & (df["resolution_data_type"] == "candle_ohlcv")
        & (df["symbol"].isin(symbol_set))
    )
    out = df[mask].copy()
    if out.empty:
        return out.assign(strike=pd.NA, direction=pd.NA, yes_token_id=pd.NA)

    out = apply_liquidity_screen(
        out,
        min_liquidity_usd=min_liquidity_usd,
        min_volume_30d_usd=min_volume_30d_usd,
    )
    out = out[out["passes_liquidity_screen"]].copy()
    if out.empty:
        return out.assign(strike=pd.NA, direction=pd.NA, yes_token_id=pd.NA)

    parsed = out["market"].apply(parse_threshold)
    out["strike"] = parsed.apply(lambda pair: pair[0])
    out["direction"] = parsed.apply(lambda pair: pair[1])
    out["yes_token_id"] = out.apply(_yes_token_id, axis=1)

    out = out[
        out["strike"].notna()
        & out["direction"].notna()
        & out["yes_token_id"].notna()
    ].copy()

    return out.reset_index(drop=True)
