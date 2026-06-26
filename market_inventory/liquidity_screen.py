from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Default tradeability thresholds (USD). A market clears the screen if it has
# enough resting liquidity OR enough recent trading activity.
DEFAULT_MIN_LIQUIDITY_USD = 10_000.0
DEFAULT_MIN_VOLUME_30D_USD = 10_000.0


def apply_liquidity_screen(
    df: pd.DataFrame,
    *,
    min_liquidity_usd: float = DEFAULT_MIN_LIQUIDITY_USD,
    min_volume_30d_usd: float = DEFAULT_MIN_VOLUME_30D_USD,
    drop: bool = False,
) -> pd.DataFrame:
    """
    Annotate (or filter) an inventory DataFrame with a tradeability screen.

    A row passes if either signal clears its threshold:

    * ``liquidity_usd >= min_liquidity_usd`` — resting order-book depth, the
      cleaner "can I get filled" signal. Present on most but not all markets.
    * ``volume_30d_usd >= min_volume_30d_usd`` — recent trading activity, used
      as a fallback so active markets that simply lack a ``liquidity`` field are
      not dropped.

    Two columns are added:

    * ``passes_liquidity_screen`` (bool)
    * ``liquidity_screen_reason`` (str) — which signal passed, or why it failed.

    Parameters
    ----------
    df:
        Inventory DataFrame from ``inventory_crypto_markets`` (must contain the
        ``liquidity_usd`` and ``volume_30d_usd`` columns).
    min_liquidity_usd:
        Minimum resting liquidity (USD) to pass on the liquidity signal.
    min_volume_30d_usd:
        Minimum 30-day volume (USD) to pass on the volume fallback.
    drop:
        When True, return only the rows that pass. When False (default),
        return all rows with the two annotation columns added.

    Returns
    -------
    pd.DataFrame
        Annotated (and optionally filtered) DataFrame.
    """
    if df.empty:
        out = df.copy()
        out["passes_liquidity_screen"] = pd.Series(dtype=bool)
        out["liquidity_screen_reason"] = pd.Series(dtype="object")
        return out

    out = df.copy()

    missing = [c for c in ("liquidity_usd", "volume_30d_usd") if c not in out.columns]
    if missing:
        raise KeyError(
            f"apply_liquidity_screen requires columns {missing}; got {list(out.columns)}. "
            "Run inventory_crypto_markets first."
        )

    liquidity = pd.to_numeric(out["liquidity_usd"], errors="coerce")
    volume_30d = pd.to_numeric(out["volume_30d_usd"], errors="coerce")

    if liquidity.isna().all() and volume_30d.isna().all():
        raise ValueError(
            "apply_liquidity_screen: both 'liquidity_usd' and 'volume_30d_usd' are entirely "
            "missing/non-numeric — nothing to screen on. Check the inventory extraction."
        )

    liq_ok = liquidity >= min_liquidity_usd
    vol_ok = volume_30d >= min_volume_30d_usd
    passes = liq_ok | vol_ok

    def _reason(idx: int) -> str:
        if liq_ok.iat[idx]:
            return f"liquidity_usd={liquidity.iat[idx]:.0f}>={min_liquidity_usd:.0f}"
        if vol_ok.iat[idx]:
            return f"volume_30d_usd={volume_30d.iat[idx]:.0f}>={min_volume_30d_usd:.0f}"
        liq_repr = "NA" if pd.isna(liquidity.iat[idx]) else f"{liquidity.iat[idx]:.0f}"
        vol_repr = "NA" if pd.isna(volume_30d.iat[idx]) else f"{volume_30d.iat[idx]:.0f}"
        return f"below_thresholds (liquidity={liq_repr}, volume_30d={vol_repr})"

    out["passes_liquidity_screen"] = passes.to_numpy()
    out["liquidity_screen_reason"] = [_reason(i) for i in range(len(out))]

    n_pass = int(passes.sum())
    logger.info(
        "liquidity screen: %d/%d markets pass "
        "(min_liquidity_usd=%.0f, min_volume_30d_usd=%.0f)",
        n_pass, len(out), min_liquidity_usd, min_volume_30d_usd,
    )

    if drop:
        out = out[passes.to_numpy()].reset_index(drop=True)

    return out
