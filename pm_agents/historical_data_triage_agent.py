from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, Optional

from agents import Agent, Runner, AgentOutputSchema  # OpenAI Agents SDK
import logging
import time

logger = logging.getLogger(__name__)

from parsing.historical_data_triage_models import HistoricalDataTriage

# ------------------------------------------------------------------
# Instructions: rubric + strict output contract (schema) lives HERE
# ------------------------------------------------------------------

INSTRUCTIONS = r"""
You are a triage agent for forecasting feasibility using historical data.

Goal:
Given ONE market row (title, resolution terms/source/date, kind/metric/symbol),
decide if historical data is relevant for estimating probability, and whether we can obtain it
for free (API, Wayback Machine, or scraping). If likely paywalled, flag it.

Key distinctions:
- Historical data is RELEVANT when the event depends on a measurable time-series or process
  (prices, market cap/FDV, reserves/holdings, official statistics, counts with stable dynamics).
- Historical data is NOT RELEVANT when it is mostly one-off / hazard-rate / adversarial /
  narrative-driven with weak measurable covariates (e.g., "another hack over $100m before 2027"),
  even if one can compile a list of past incidents.

Feasibility:
- "yes": plausible free source exists and acquisition method is clear.
- "maybe": plausible but uncertain (rate limits, unclear endpoint, partial coverage, messy scrape).
- "no": likely not obtainable or not measurable with public data.

Paywall risk:
- "none": clearly free / public / open APIs.
- "possible": unknown restrictions, rate limits, or partial gating.
- "likely": known paywalled providers (Bloomberg, PitchBook, WSJ) or clear login/subscription barrier.

Requirements for output:
- Output MUST be valid JSON matching the schema below.
- If data_feasibility != "no", include:
  - at least 1 candidates[] entry, and
  - at least 1 plans[] entry with a concrete target.
- plans[] should include the most plausible free approach first.
- Do not invent precise endpoints unless you are confident; if unsure, put method="unknown" and explain.
- Keep rationales specific to the market.

========================
OUTPUT JSON SCHEMA (must match):
========================
{
  "market_id": "string",
  "market": "string",
  "kind": "string",
  "metric": "string",

  "historical_relevance": "yes|no|mixed",
  "relevance_rationale": "string",

  "data_feasibility": "yes|maybe|no",
  "feasibility_rationale": "string",

  "paywall_risk": "none|possible|likely",
  "paywall_rationale": "string",

  "candidates": [
    {
      "name": "string",
      "unit": "string|null",
      "frequency": "string|null",
      "proxy_ok": true,
      "proxy_notes": "string|null"
    }
  ],

  "plans": [
    {
      "method": "api|web_scrape|wayback|csv_download|manual|unknown",
      "target": "string",
      "url_or_endpoint_hint": "string|null",

      "access": "free|rate_limited_free|paywalled|unknown",
      "paywall_evidence": "string|null",

      "effort": "low|medium|high",
      "reliability": "low|medium|high",
      "notes": "string|null"
    }
  ],

  "recommended_resolution": "string|null",
  "routing_notes": "string|null"
}
""".strip()

# ------------------------------------------------------------------
# Agent definition
# ------------------------------------------------------------------

historical_data_triage_agent = Agent(
    name="historical_data_triage",
    model="gpt-5.2-pro-2025-12-11",
    instructions=INSTRUCTIONS,
    output_type=AgentOutputSchema(HistoricalDataTriage, strict_json_schema=False),
)

# ------------------------------------------------------------------
# Public API: single-shot triage for one market row
# ------------------------------------------------------------------

def _json_safe(x):
    # keep simple types
    if x is None or isinstance(x, (str, int, float, bool)):
        return x

    # pandas/numpy missing values
    try:
        import pandas as pd
        if pd.isna(x):
            return None
    except Exception:
        pass

    # datetime / pandas Timestamp
    try:
        import datetime as dt
        if isinstance(x, (dt.datetime, dt.date)):
            return x.isoformat()
    except Exception:
        pass

    # numpy scalars
    try:
        import numpy as np
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.bool_,)):
            return bool(x)
    except Exception:
        pass

    # lists/tuples
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]

    # dicts
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}

    # fallback: string representation
    return str(x)


def _row_to_input(row: Dict[str, Any]) -> str:
    """
    Provide the agent a compact, high-signal representation.
    """
    keys = [
        "market_id", "market", "kind", "symbol", "metric",
        "resolution_date", "resolution_source", "resolution_terms",
        "resolution_data_type", "resolution_interval", "interval_source",
        "routing_notes",
    ]
    payload = {k: row.get(k) for k in keys if k in row}
    # Keep as JSON-like text; model will output JSON (schema enforced in instructions)
    import json
    payload = _json_safe(payload)
    return json.dumps(payload, ensure_ascii=False)

async def triage_market_row(
    row: Dict[str, Any],
    timeout_s: float = 45.0,   
) -> HistoricalDataTriage:
    market_id = str(row.get("market_id", ""))
    title = str(row.get("market") or row.get("title") or "")[:120]
    t0 = time.time()

    async def _run_once() -> HistoricalDataTriage:
        result = await Runner.run(
            historical_data_triage_agent,
            input=_row_to_input(row),
        )
        out = result.final_output
        return out if isinstance(out, HistoricalDataTriage) else HistoricalDataTriage.model_validate_json(out)

    logger.debug("triage_market_row start: market_id=%s title=%r timeout_s=%.1f", market_id, title, timeout_s)

    try:
        triage = await asyncio.wait_for(_run_once(), timeout=timeout_s)

        logger.info(
            "triage_market_row ok: market_id=%s relevance=%s feasibility=%s paywall=%s (%.2fs)",
            market_id, triage.historical_relevance, triage.data_feasibility, triage.paywall_risk, time.time() - t0
        )
        return triage

    except asyncio.TimeoutError:
        logger.warning("triage_market_row TIMEOUT: market_id=%s title=%r (%.2fs)", market_id, title, time.time() - t0)
        raise

    except Exception:
        logger.exception("triage_market_row failed: market_id=%s title=%r (%.2fs)", market_id, title, time.time() - t0)
        raise

