from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import logging
import time
from pm_agents.historical_data_triage_agent import triage_market_row
from parsing.historical_data_triage_models import HistoricalDataTriage
from market_inventory.liquidity_screen import apply_liquidity_screen


logger = logging.getLogger(__name__)

# Bump when the triage output schema changes in a way that invalidates
# cached parquet rows (new/renamed/removed columns, changed semantics).
_TRIAGE_SCHEMA_VERSION = 1
_TRIAGE_SCHEMA_VERSION_COL = "triage_schema_version"

# ── Columns used to detect whether an inventory row has changed ──────────────
DIFF_COLUMNS: List[str] = [
    "market",
    "kind",
    "symbol",
    "metric",
    "resolution_date",
    "resolution_source",
    "resolution_terms",
    "warning_end_date_mismatch",
    "resolution_data_type",
    "resolution_interval",
    "interval_source",
    "routing_notes",
]

# The "market" column is the natural key (unique question text per row).
_ROW_KEY = "market"

def _row_dict(row: pd.Series) -> Dict[str, Any]:
    # Ensure consistent keys
    d = row.to_dict()

    # Normalize NaN -> None for cleaner JSON to the agent
    for k, v in list(d.items()):
        if isinstance(v, float) and pd.isna(v):
            d[k] = None

    # Make sure we have market_id + market; adapt if your columns differ
    if "market_id" not in d:
        # fallback if your id column is named differently
        for alt in ["id", "condition_id", "marketId", "marketIdNum"]:
            if alt in d:
                d["market_id"] = str(d[alt])
                break

    if "market" not in d and "title" in d:
        d["market"] = d["title"]

    return d


# ---------------------------------------------------------------------------
# Liquidity gate: only triage markets that clear the tradeability screen
# ---------------------------------------------------------------------------

def _gate_liquidity(df: pd.DataFrame, enabled: bool) -> pd.DataFrame:
    """Filter *df* to rows that pass the liquidity screen.

    If the ``passes_liquidity_screen`` column is absent, it is computed via
    :func:`market_inventory.liquidity_screen.apply_liquidity_screen` (which
    raises loudly if the underlying liquidity/volume columns are missing — no
    silent degradation). When *enabled* is False the frame is returned
    unchanged. Returns a frame with a fresh 0-based index.
    """
    if not enabled or df.empty:
        return df

    if "passes_liquidity_screen" not in df.columns:
        df = apply_liquidity_screen(df)

    before = len(df)
    gated = df[df["passes_liquidity_screen"].astype(bool)].reset_index(drop=True)
    logger.info("liquidity gate: %d/%d rows pass the screen", len(gated), before)
    return gated


# ---------------------------------------------------------------------------
# Triage cache: save / load
# ---------------------------------------------------------------------------

DEFAULT_CACHE_PATH = Path("triage_cache.parquet")


def _strip_arrow_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert any pd.ArrowDtype columns to plain numpy/object dtypes.

    Newer pandas versions return ArrowDtype-backed columns from read_parquet.
    When those are concat-ed with numpy-backed data and re-serialised, pyarrow
    raises ArrowKeyError on its internal extension type registry.  This helper
    normalises everything to standard dtypes before any parquet round-trip.
    """
    out = {}
    for col in df.columns:
        s = df[col]
        if isinstance(s.dtype, pd.ArrowDtype):
            try:
                # Preserve numeric/bool precision where possible
                out[col] = s.astype(s.dtype.numpy_dtype)
            except (TypeError, AttributeError):
                out[col] = s.astype(object)
        else:
            out[col] = s
    return pd.DataFrame(out, index=df.index)


def save_triage_cache(
    df: pd.DataFrame,
    path: Union[str, Path] = DEFAULT_CACHE_PATH,
) -> Path:
    """Persist a triaged DataFrame to parquet."""
    path = Path(path)
    if _TRIAGE_SCHEMA_VERSION_COL not in df.columns:
        df = df.copy()
        df[_TRIAGE_SCHEMA_VERSION_COL] = _TRIAGE_SCHEMA_VERSION
    _strip_arrow_dtypes(df).to_parquet(path, index=False)
    logger.info("triage cache saved: %s (%d rows)", path, len(df))
    return path


def load_triage_cache(
    path: Union[str, Path] = DEFAULT_CACHE_PATH,
) -> Optional[pd.DataFrame]:
    """Load a previously-saved triage cache.  Returns None if it doesn't exist."""
    path = Path(path)
    if not path.exists():
        logger.info("no triage cache found at %s", path)
        return None
    df = _strip_arrow_dtypes(pd.read_parquet(path))
    logger.info("triage cache loaded: %s (%d rows)", path, len(df))
    return df


# ---------------------------------------------------------------------------
# Diff: which inventory rows need (re-)triaging?
# ---------------------------------------------------------------------------

def _normalise_for_compare(val: object) -> str:
    """Stringify a cell value so NaN, None, and NaT all become ''."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    if isinstance(val, pd.Timestamp):
        return str(val)
    return str(val)


def diff_inventory_vs_cache(
    inventory_df: pd.DataFrame,
    cache_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Return the subset of *inventory_df* that needs triaging.

    A row needs triaging when:
    1. Its ``market`` value does not appear in the cache at all, **or**
    2. Any of the ``DIFF_COLUMNS`` values differ between the inventory row
       and the cached row for the same ``market``.

    Returns a (possibly empty) DataFrame with the same columns as
    *inventory_df* and a fresh 0-based index.
    """
    if cache_df is None or cache_df.empty:
        logger.info("diff: no cache → all %d rows need triaging", len(inventory_df))
        return inventory_df.reset_index(drop=True)

    if _TRIAGE_SCHEMA_VERSION_COL not in cache_df.columns:
        logger.warning(
            "diff: cache missing %s column → all %d rows need re-triaging",
            _TRIAGE_SCHEMA_VERSION_COL, len(inventory_df),
        )
        return inventory_df.reset_index(drop=True)

    cached_versions = set(cache_df[_TRIAGE_SCHEMA_VERSION_COL].dropna().unique().tolist())
    if cached_versions != {_TRIAGE_SCHEMA_VERSION}:
        logger.warning(
            "diff: cache schema version mismatch (cache=%s, current=%d) → all %d rows need re-triaging",
            sorted(cached_versions), _TRIAGE_SCHEMA_VERSION, len(inventory_df),
        )
        return inventory_df.reset_index(drop=True)

    # Build a lookup: market → dict of diff-column values (stringified)
    cache_lookup: Dict[str, Dict[str, str]] = {}
    for _, row in cache_df.iterrows():
        key = row.get(_ROW_KEY)
        if key is None or (isinstance(key, float) and pd.isna(key)):
            continue
        cache_lookup[str(key)] = {
            col: _normalise_for_compare(row.get(col))
            for col in DIFF_COLUMNS
        }

    needs_triage_mask = []
    for _, row in inventory_df.iterrows():
        key = str(row.get(_ROW_KEY, ""))
        cached = cache_lookup.get(key)
        if cached is None:
            needs_triage_mask.append(True)
            continue

        # Compare each diff column
        changed = False
        for col in DIFF_COLUMNS:
            inv_val = _normalise_for_compare(row.get(col))
            if inv_val != cached.get(col, ""):
                changed = True
                break
        needs_triage_mask.append(changed)

    delta = inventory_df.loc[needs_triage_mask].reset_index(drop=True)
    logger.info(
        "diff: %d/%d inventory rows need triaging (%d cached, %d unchanged)",
        len(delta),
        len(inventory_df),
        len(cache_lookup),
        len(inventory_df) - len(delta),
    )
    return delta


# ---------------------------------------------------------------------------
# Incremental triage: only triage new / changed rows, merge with cache
# ---------------------------------------------------------------------------

async def triage_dataframe_incremental_async(
    inventory_df: pd.DataFrame,
    *,
    cache_path: Union[str, Path] = DEFAULT_CACHE_PATH,
    max_concurrency: int = 20,
    only_source_nan_or_url: bool = True,
    require_liquidity_screen: bool = True,
    max_rows: Optional[int] = None,
    log_every: int = 25,
    row_timeout_s: float = 45.0,
    total_timeout_s: Optional[float] = None,
) -> pd.DataFrame:
    """Incremental wrapper around :func:`triage_dataframe_async`.

    1. Apply the liquidity gate so untradeable markets never enter the cache.
    2. Load the cached triage results (if any).
    3. Diff the gated inventory against the cache to find new / changed rows.
    4. Triage **only** the delta rows.
    5. Merge the fresh triage results back into the full cache.
    6. Save the updated cache and return the merged DataFrame.

    Rows that disappeared from the inventory (i.e. present in cache but not in
    *inventory_df*) are **dropped** from the returned result — they represent
    markets that are no longer active.
    """
    # Gate up front so screened-out markets never enter the diff/cache churn.
    inventory_df = _gate_liquidity(inventory_df, require_liquidity_screen)

    cache_df = load_triage_cache(cache_path)
    delta = diff_inventory_vs_cache(inventory_df, cache_df)

    if delta.empty:
        logger.info("incremental triage: nothing to do — all rows are cached and unchanged")
        # Still restrict to current inventory (drop stale cache rows)
        if cache_df is not None:
            current_markets = set(inventory_df[_ROW_KEY].astype(str))
            result = cache_df[cache_df[_ROW_KEY].astype(str).isin(current_markets)].reset_index(drop=True)
            return result
        return delta

    logger.info(
        "incremental triage: %d new/changed rows to triage (out of %d inventory rows)",
        len(delta), len(inventory_df),
    )

    fresh = await triage_dataframe_async(
        delta,
        max_concurrency=max_concurrency,
        only_source_nan_or_url=only_source_nan_or_url,
        require_liquidity_screen=False,  # already gated above
        max_rows=max_rows,
        log_every=log_every,
        row_timeout_s=row_timeout_s,
        total_timeout_s=total_timeout_s,
    )

    # Merge: start from cache, drop rows that were re-triaged, append fresh
    if cache_df is not None and not cache_df.empty:
        retriaged_markets = set(delta[_ROW_KEY].astype(str))
        kept = cache_df[~cache_df[_ROW_KEY].astype(str).isin(retriaged_markets)]
        merged = pd.concat([kept, fresh], ignore_index=True)
    else:
        merged = fresh

    # Drop rows no longer in inventory
    current_markets = set(inventory_df[_ROW_KEY].astype(str))
    merged = merged[merged[_ROW_KEY].astype(str).isin(current_markets)].reset_index(drop=True)

    save_triage_cache(merged, cache_path)
    logger.info(
        "incremental triage done: %d triaged this run, %d total cached",
        len(fresh), len(merged),
    )
    return merged


def triage_dataframe_incremental(
    inventory_df: pd.DataFrame,
    *,
    cache_path: Union[str, Path] = DEFAULT_CACHE_PATH,
    max_concurrency: int = 20,
    only_source_nan_or_url: bool = True,
    require_liquidity_screen: bool = True,
    max_rows: Optional[int] = None,
    log_every: int = 25,
    row_timeout_s: float = 30.0,
    total_timeout_s: float = 300.0,
) -> pd.DataFrame:
    """Sync wrapper for :func:`triage_dataframe_incremental_async`."""
    return asyncio.run(
        triage_dataframe_incremental_async(
            inventory_df,
            cache_path=cache_path,
            max_concurrency=max_concurrency,
            only_source_nan_or_url=only_source_nan_or_url,
            require_liquidity_screen=require_liquidity_screen,
            max_rows=max_rows,
            log_every=log_every,
            row_timeout_s=row_timeout_s,
            total_timeout_s=total_timeout_s,
        )
    )


async def triage_dataframe_async(
    df: pd.DataFrame,
    *,
    max_concurrency: int = 20,
    only_source_nan_or_url: bool = True,
    require_liquidity_screen: bool = True,
    max_rows: Optional[int] = None,
            log_every: int = 25,
    row_timeout_s: float = 45.0,
    total_timeout_s: Optional[float] = None,
) -> pd.DataFrame:
    """
    Returns a copy of df with triage columns appended.

    require_liquidity_screen: if True, drop rows that fail the liquidity screen
        before any LLM calls, so triage spend only goes to tradeable markets.
    max_rows: if set, only process up to this many rows after filtering.
    log_every: log progress every N completed rows.
    """
    t0 = time.time()
    logger.info(
        "triage_dataframe_async start: rows=%d max_concurrency=%d only_source_nan_or_url=%s "
        "require_liquidity_screen=%s max_rows=%s",
        len(df), max_concurrency, only_source_nan_or_url, require_liquidity_screen, max_rows
    )

    work = df.copy()

    # Liquidity gate: drop untradeable markets before any LLM calls
    work = _gate_liquidity(work, require_liquidity_screen)

    # Filter
    if only_source_nan_or_url and "resolution_source" in work.columns:
        mask = work["resolution_source"].isna() | work["resolution_source"].astype(str).str.startswith("http", na=False)
        work = work.loc[mask].copy()

    logger.info("after filter: rows=%d", len(work))

    # Limit rows (deterministic head)
    if max_rows is not None:
        if max_rows < 0:
            raise ValueError("max_rows must be >= 0 or None")
        work = work.head(max_rows).copy()
        logger.info("after max_rows=%d cap: rows=%d", max_rows, len(work))

    if len(work) == 0:
        logger.warning("No rows to triage after filtering/capping.")
        return work

    sem = asyncio.Semaphore(max_concurrency)
    results: dict[int, HistoricalDataTriage] = {}
    errors: dict[int, str] = {}              # NEW: collect errors
    completed = 0

    async def _one(i: int, row: pd.Series) -> None:
        nonlocal completed
        async with sem:
            try:
                triage = await  triage_market_row(_row_dict(row), timeout_s=row_timeout_s)
                results[i] = triage
            except asyncio.TimeoutError:
                errors[i] = f"Timeout after {row_timeout_s}s"
                logger.warning("triage timeout for index=%s", i)
            except Exception as e:
                # keep going; store error string
                errors[i] = repr(e)
                logger.exception("triage failed for index=%s", i)
            finally:
                completed += 1
                if completed % log_every == 0 or completed == len(work):
                    logger.info("progress: %d/%d completed (%d ok, %d errors)",
                                completed, len(work), len(results), len(errors))

    # Total timeout for the entire batch (optional)
    task_objs = [asyncio.create_task(_one(i, work.loc[i])) for i in work.index]
    all_done = asyncio.gather(*task_objs)

    if total_timeout_s is not None:
        try:
            await asyncio.wait_for(all_done, timeout=total_timeout_s)
        except asyncio.TimeoutError:
            logger.error(
                "TOTAL TIMEOUT after %.1fs; cancelling remaining tasks. completed=%d/%d ok=%d errors=%d",
                total_timeout_s,
                completed,
                len(work),
                len(results),
                len(errors),
            )
            # Cancel unfinished tasks
            for t in task_objs:
                if not t.done():
                    t.cancel()

            # Drain cancellations/exceptions so the event loop is clean
            drained = await asyncio.gather(*task_objs, return_exceptions=True)

            # Mark any still-unrecorded indices as cancelled (best-effort)
            # (We don’t have index mapping from task -> i here without extra wiring,
            #  so we just log how many were cancelled.)
            cancelled_count = sum(isinstance(x, asyncio.CancelledError) for x in drained)
            logger.error("Cancelled tasks: %d", cancelled_count)
    else:
        await all_done
    logger.info("triage completed: ok=%d errors=%d", len(results), len(errors))
    # Flatten outputs into columns
    out = work.copy()
    out["historical_relevance"] = None
    out["relevance_rationale"] = None
    out["data_feasibility"] = None
    out["feasibility_rationale"] = None
    out["paywall_risk"] = None
    out["paywall_rationale"] = None
    out["triage_candidates_json"] = None
    out["triage_plans_json"] = None
    out["recommended_resolution"] = None
    out["triage_routing_notes"] = None
    out["triage_error"] = None
    out[_TRIAGE_SCHEMA_VERSION_COL] = _TRIAGE_SCHEMA_VERSION

    import json
    for i in out.index:
        if i in results:
            triage = results[i]
            out.at[i, "historical_relevance"] = triage.historical_relevance
            out.at[i, "relevance_rationale"] = triage.relevance_rationale
            out.at[i, "data_feasibility"] = triage.data_feasibility
            out.at[i, "feasibility_rationale"] = triage.feasibility_rationale
            out.at[i, "paywall_risk"] = triage.paywall_risk
            out.at[i, "paywall_rationale"] = triage.paywall_rationale
            out.at[i, "triage_candidates_json"] = json.dumps([c.model_dump() for c in triage.candidates], ensure_ascii=False)
            out.at[i, "triage_plans_json"] = json.dumps([p.model_dump() for p in triage.plans], ensure_ascii=False)
            out.at[i, "recommended_resolution"] = triage.recommended_resolution
            out.at[i, "triage_routing_notes"] = triage.routing_notes
        elif i in errors:
            out.at[i, "triage_error"] = errors[i]
        else:
            # This can happen if total timeout cancelled before _one() recorded anything
            out.at[i, "triage_error"] = "Not processed (batch timeout/cancelled)"

    logger.info("triage_dataframe_async done in %.2fs (rows=%d)", time.time() - t0, len(out))
    return out


def triage_dataframe(
    df: pd.DataFrame,
    *,
    max_concurrency: int = 20,
    only_source_nan_or_url: bool = True,
    require_liquidity_screen: bool = True,
    max_rows: Optional[int] = None,
    log_every: int = 25,
    row_timeout_s=30.0,
    total_timeout_s=300.0,
) -> pd.DataFrame:
    return asyncio.run(
        triage_dataframe_async(
            df,
            max_concurrency=max_concurrency,
            only_source_nan_or_url=only_source_nan_or_url,
            require_liquidity_screen=require_liquidity_screen,
            max_rows=max_rows,
            log_every=log_every,
            row_timeout_s=row_timeout_s,
            total_timeout_s=total_timeout_s,
        )
    )

