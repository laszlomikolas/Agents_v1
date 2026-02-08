from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

import pandas as pd
import logging
import time
from pm_agents.historical_data_triage_agent import triage_market_row
from parsing.historical_data_triage_models import HistoricalDataTriage


logger = logging.getLogger(__name__)

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


async def triage_dataframe_async(
    df: pd.DataFrame,
    *,
    max_concurrency: int = 20,
    only_source_nan_or_url: bool = True,
    max_rows: Optional[int] = None,          
            log_every: int = 25,                   
    row_timeout_s: float = 45.0,       
    total_timeout_s: Optional[float] = None,  
) -> pd.DataFrame:
    """
    Returns a copy of df with triage columns appended.

    max_rows: if set, only process up to this many rows after filtering.
    log_every: log progress every N completed rows.
    """
    t0 = time.time()
    logger.info(
        "triage_dataframe_async start: rows=%d max_concurrency=%d only_source_nan_or_url=%s max_rows=%s",
        len(df), max_concurrency, only_source_nan_or_url, max_rows
    )

    work = df.copy()

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
            max_rows=max_rows,
            log_every=log_every,
            row_timeout_s=row_timeout_s,
            total_timeout_s=total_timeout_s,
        )
    )

