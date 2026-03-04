from __future__ import annotations

import asyncio
import json
import logging
import textwrap
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from parsing.connector_models import ConnectorCode
from parsing.historical_data_triage_models import DataSourcePlan
from pm_agents.connector_builder_agent import build_connector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_plans(triaged_df: pd.DataFrame) -> List[DataSourcePlan]:
    """
    Parse triage_plans_json from every row, deduplicate by connector_key,
    and return the ordered list of unique DataSourcePlan objects to build.
    """
    seen: set[str] = set()
    plans: List[DataSourcePlan] = []

    for idx, row in triaged_df.iterrows():
        raw = row.get("triage_plans_json")
        if not raw or (isinstance(raw, float)):
            # None / NaN — row has no plans
            continue

        try:
            plan_dicts = json.loads(raw)
        except Exception:
            logger.warning("Could not parse triage_plans_json for index=%s", idx)
            continue

        for pd_dict in plan_dicts:
            try:
                plan = DataSourcePlan(**pd_dict)
            except Exception as exc:
                logger.warning(
                    "Skipping malformed plan at index=%s: %s", idx, exc
                )
                continue

            if plan.connector_key in seen:
                continue  # already queued
            seen.add(plan.connector_key)
            plans.append(plan)

    return plans


def _should_skip(plan: DataSourcePlan, *, skip_paywall: bool, skip_infeasible: bool) -> bool:
    """Return True if this plan should be skipped based on runner config."""
    if skip_paywall and plan.access == "paywalled":
        return True
    if skip_infeasible and plan.effort == "high" and plan.reliability == "low":
        return True
    return False


# ---------------------------------------------------------------------------
# Async runner
# ---------------------------------------------------------------------------

async def build_connectors_async(
    triaged_df: pd.DataFrame,
    *,
    registry: Optional[Dict[str, ConnectorCode]] = None,
    max_concurrency: int = 10,
    skip_paywall: bool = True,
    skip_infeasible: bool = False,
    max_plans: Optional[int] = None,
    log_every: int = 10,
    plan_timeout_s: float = 120.0,
    total_timeout_s: Optional[float] = None,
) -> Dict[str, ConnectorCode]:
    """
    Given the DataFrame output of ``triage_dataframe_async``, extract every
    unique DataSourcePlan, call the connector-builder agent for each one in
    parallel, and return a mapping ``connector_key -> ConnectorCode``.

    Plans whose ``connector_key`` is already present in ``registry`` are skipped
    entirely — no agent call is made for them.  The returned dict is the full
    merged result: existing registry entries **plus** any newly built connectors.

    Parameters
    ----------
    triaged_df:
        Output of ``pipeline.historical_triage_runner.triage_dataframe_async``.
        Must contain a ``triage_plans_json`` column.
    registry:
        Previously built connectors keyed by ``connector_key``.  Pass the dict
        returned by a previous call (or loaded via :func:`load_registry`) to
        avoid rebuilding connectors that already exist.  The input dict is never
        mutated; a new merged dict is returned.
    max_concurrency:
        Maximum number of connector-builder agent calls running simultaneously.
    skip_paywall:
        When True, skip plans whose ``access == "paywalled"``.
    skip_infeasible:
        When True, skip plans whose ``effort == "high"`` and
        ``reliability == "low"`` (high effort / low reliability combos).
    max_plans:
        If set, cap the total number of *new* plans processed (useful for testing).
    log_every:
        Log progress every N completed plans.
    plan_timeout_s:
        Per-plan agent call timeout in seconds.
    total_timeout_s:
        Optional hard cap on total wall-clock time for the entire batch.

    Returns
    -------
    dict
        ``{connector_key: ConnectorCode}`` containing both pre-existing registry
        entries and any newly built connectors.  Failed plans are logged and
        omitted from the result.
    """
    t0 = time.time()
    existing: Dict[str, ConnectorCode] = dict(registry) if registry else {}

    all_plans = _extract_plans(triaged_df)
    logger.info(
        "build_connectors_async start: unique_plans=%d registry_size=%d "
        "max_concurrency=%d skip_paywall=%s skip_infeasible=%s max_plans=%s",
        len(all_plans), len(existing), max_concurrency, skip_paywall, skip_infeasible, max_plans,
    )

    # Skip plans already in the registry
    already_have = [p for p in all_plans if p.connector_key in existing]
    new_plans = [p for p in all_plans if p.connector_key not in existing]
    if already_have:
        logger.info(
            "Skipping %d plans already in registry: %s",
            len(already_have),
            [p.connector_key for p in already_have],
        )

    # Apply paywall / infeasibility filters
    plans = [
        p for p in new_plans
        if not _should_skip(p, skip_paywall=skip_paywall, skip_infeasible=skip_infeasible)
    ]
    skipped = len(new_plans) - len(plans)
    if skipped:
        logger.info("Skipped %d new plans (paywall/infeasibility filter)", skipped)

    if max_plans is not None:
        plans = plans[:max_plans]
        logger.info("Capped to max_plans=%d; processing %d new plans", max_plans, len(plans))

    if not plans:
        logger.info("No new plans to build; returning registry as-is (%d entries).", len(existing))
        return existing

    sem = asyncio.Semaphore(max_concurrency)
    results: Dict[str, ConnectorCode] = {}
    errors: Dict[str, str] = {}
    completed = 0

    async def _one(plan: DataSourcePlan) -> None:
        nonlocal completed
        async with sem:
            key = plan.connector_key
            try:
                code = await build_connector(plan, timeout_s=plan_timeout_s)
                results[key] = code
            except asyncio.TimeoutError:
                errors[key] = f"Timeout after {plan_timeout_s}s"
                logger.warning("build_connector timeout: key=%s", key)
            except Exception as exc:
                errors[key] = repr(exc)
                logger.exception("build_connector failed: key=%s", key)
            finally:
                completed += 1
                if completed % log_every == 0 or completed == len(plans):
                    logger.info(
                        "progress: %d/%d completed (%d ok, %d errors)",
                        completed, len(plans), len(results), len(errors),
                    )

    task_objs = [asyncio.create_task(_one(p)) for p in plans]
    all_done = asyncio.gather(*task_objs)

    if total_timeout_s is not None:
        try:
            await asyncio.wait_for(all_done, timeout=total_timeout_s)
        except asyncio.TimeoutError:
            logger.error(
                "TOTAL TIMEOUT after %.1fs; cancelling remaining tasks. "
                "completed=%d/%d ok=%d errors=%d",
                total_timeout_s, completed, len(plans), len(results), len(errors),
            )
            for t in task_objs:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*task_objs, return_exceptions=True)
    else:
        await all_done

    logger.info(
        "build_connectors_async done in %.2fs: new_ok=%d errors=%d registry_total=%d",
        time.time() - t0, len(results), len(errors), len(existing) + len(results),
    )
    return {**existing, **results}


# ---------------------------------------------------------------------------
# Synchronous wrapper
# ---------------------------------------------------------------------------

def build_connectors(
    triaged_df: pd.DataFrame,
    *,
    registry: Optional[Dict[str, ConnectorCode]] = None,
    max_concurrency: int = 10,
    skip_paywall: bool = True,
    skip_infeasible: bool = False,
    max_plans: Optional[int] = None,
    log_every: int = 10,
    plan_timeout_s: float = 120.0,
    total_timeout_s: Optional[float] = None,
) -> Dict[str, ConnectorCode]:
    """Synchronous wrapper around :func:`build_connectors_async`."""
    return asyncio.run(
        build_connectors_async(
            triaged_df,
            registry=registry,
            max_concurrency=max_concurrency,
            skip_paywall=skip_paywall,
            skip_infeasible=skip_infeasible,
            max_plans=max_plans,
            log_every=log_every,
            plan_timeout_s=plan_timeout_s,
            total_timeout_s=total_timeout_s,
        )
    )


# ---------------------------------------------------------------------------
# Registry persistence
# ---------------------------------------------------------------------------

def save_registry(
    connectors: Dict[str, ConnectorCode],
    path: str | Path,
) -> None:
    """
    Persist the connector registry to a JSON file.

    The file maps ``connector_key -> ConnectorCode`` fields and can be reloaded
    with :func:`load_registry` to skip already-built connectors on the next run.
    """
    path = Path(path)
    payload = {key: code.model_dump(mode="json") for key, code in connectors.items()}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("save_registry: wrote %d entries to %s", len(connectors), path)


def load_registry(path: str | Path) -> Dict[str, ConnectorCode]:
    """
    Load a connector registry previously saved with :func:`save_registry`.

    Returns an empty dict if the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        logger.info("load_registry: %s not found, starting with empty registry.", path)
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    registry = {key: ConnectorCode(**data) for key, data in payload.items()}
    logger.info("load_registry: loaded %d entries from %s", len(registry), path)
    return registry


# ---------------------------------------------------------------------------
# Optional: write all built connectors to a Python module file
# ---------------------------------------------------------------------------

def write_connectors_module(
    connectors: Dict[str, ConnectorCode],
    output_path: str | Path,
) -> None:
    """
    Write all generated connector functions to a single Python module file.

    The file will have deduplicated imports at the top, followed by each
    connector function separated by blank lines.  A ``__all__`` list is
    included for easy wildcard-import usage.

    Parameters
    ----------
    connectors:
        Mapping returned by :func:`build_connectors` / :func:`build_connectors_async`.
    output_path:
        Destination .py file path.  Parent directories must exist.
    """
    if not connectors:
        logger.warning("write_connectors_module: no connectors to write.")
        return

    output_path = Path(output_path)

    # Deduplicate imports (preserve first-seen order)
    seen_imports: set[str] = set()
    ordered_imports: List[str] = []
    for code in connectors.values():
        for imp in code.imports:
            imp = imp.strip()
            if imp and imp not in seen_imports:
                seen_imports.add(imp)
                ordered_imports.append(imp)

    fn_names = [c.connector_function_name for c in connectors.values()]

    lines: List[str] = [
        "# Auto-generated connector module.",
        "# Do not edit manually — regenerate via pipeline.connector_builder_runner.",
        "",
    ]

    if ordered_imports:
        lines.extend(ordered_imports)
        lines.append("")

    lines.append(f"__all__ = {fn_names!r}")
    lines.append("")

    for code in connectors.values():
        lines.append("")
        # Inline connector_key as a comment header for traceability
        lines.append(f"# connector_key: {code.connector_key}")
        if code.notes:
            for note_line in textwrap.wrap(f"# NOTE: {code.notes}", width=88):
                lines.append(note_line)
        lines.append(code.source_code.rstrip())
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(
        "write_connectors_module: wrote %d connectors to %s",
        len(connectors), output_path,
    )
