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
decide whether historical data is useful for estimating probability, and whether we can obtain it
for free (free API, scraping, Wayback). If likely paywalled, flag it.

=====================================================================
CONNECTOR DISCOVERY GOAL (IMPORTANT)
=====================================================================
We already support structured OHLC/price connectors for these sources:
  chainlink, coinbase, binance, coingecko, kraken, bitstamp, okx, bybit.

Do NOT propose building new connectors for OHLC/price candles from these providers.
You MAY reference them ONLY as secondary proxy series (e.g. to convert holdings BTC→USD).

Your main job is to discover connectors for UNSTRUCTURED or NON-OHLC data, such as:
  - free_api_generic
  - official_stats_api
  - blockchain_explorer_api
  - defi_protocol_api
  - social_platform_api
  - generic_html_table_scrape
  - generic_web_scrape
  - generic_json_endpoint
  - wayback_snapshots
  - pdf_table_extract
  - csv_download / github_raw / google_sheets
  - wikipedia_wikidata
  - official_stats_portal
  - paywalled_provider
  - unknown

For EVERY item in plans[] you MUST set:
  - connector_type: one of the allowed connector types above
  - connector_key: stable identifier primarily based on domain + path
  - required_params: minimal JSON-serializable params needed to fetch the series
  - series_id: snake_case canonical output series name

Examples of connector_key:
  free_api_generic:api.example.com/v1/series
  official_stats_api:api.worldbank.org/v2/indicator
  wayback_snapshots:intel.arkm.com/explorer/entity/el-salvador

=====================================================================
GUARDRAIL: VAGUE RESOLUTION SOURCES (MUST SKIP)
=====================================================================
If resolution_terms explicitly state or imply that the resolution source is vague, discretionary,
or undefined (e.g. "the most liquid price source available", "any reputable source", "best available data",
"at the discretion of the resolver"), then the market is NOT suitable for connector planning.

In this case you MUST:
  - Set historical_data_useful = "no"
  - Set data_feasibility = "no"
  - Set paywall_risk = "none"
  - Set routing_notes = "vague_resolution_source"
  - Explain in relevance_rationale that the rules are too vague to map to a precise historical dataset
  - Set candidates = [] and plans = [] (empty lists)

=====================================================================
URL PRIORITY RULE (MUST FOLLOW)
=====================================================================
If resolution_source is a URL (starts with http/https), treat it as the primary lead.

You MUST include at least TWO plans in this case (unless the market is skipped by the vague-rule guardrail):
  1) FIRST plan: attempt to obtain historical data from that exact URL directly.
     - If the URL looks like an API endpoint (JSON, /api/, obvious query params), choose an API connector_type
       (free_api_generic or a more specific API type).
     - Otherwise choose a scraping connector_type (generic_html_table_scrape / generic_web_scrape /
       generic_json_endpoint).
  2) SECOND plan: wayback_snapshots for that same URL, unless you can justify clearly that Wayback is irrelevant
     (e.g. the URL is a stable API endpoint with a parameterized historical time-series).

Only AFTER those can you propose alternative sources/proxies.

=====================================================================
HOW TO DECIDE IF HISTORICAL DATA IS USEFUL
=====================================================================
Historical data is USEFUL when the event depends on a measurable time-series or process
(holdings/reserves, counts, official statistics, protocol metrics, etc.).

Historical data is NOT useful when it is primarily one-off / hazard-rate / adversarial /
narrative-driven with weak measurable covariates (e.g., "another hack over $100m before 2027"),
even if one can compile a list of past incidents.

Use:
- historical_data_useful = "yes" when the series gives meaningful signal for estimating probability.
- historical_data_useful = "mixed" when you can collect history but predictive value is limited.
- historical_data_useful = "no" when history is not helpful or market is underspecified.

=====================================================================
FEASIBILITY + PAYWALL
=====================================================================
data_feasibility:
- "yes": plausible free source exists and acquisition method is clear.
- "maybe": plausible but uncertain (rate limits, unclear endpoint, partial coverage, messy scrape).
- "no": likely not obtainable or not measurable with public data.

paywall_risk:
- "none": clearly free / public / open APIs.
- "possible": unknown restrictions, rate limits, or partial gating.
- "likely": known paywalled providers (Bloomberg, PitchBook, WSJ) or clear login/subscription barrier.

=====================================================================
OUTPUT REQUIREMENTS
=====================================================================
- Output MUST be valid JSON matching the schema below.
- If data_feasibility != "no", include at least:
  - 1 candidates[] entry, and
  - 1 plans[] entry with a concrete target.
- plans[] should list the most plausible approach FIRST.
- Do not invent precise endpoints unless confident; if unsure, use connector_type="unknown" with method="unknown"
  and explain.

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
      "connector_type": "free_api_generic|official_stats_api|blockchain_explorer_api|defi_protocol_api|social_platform_api|generic_html_table_scrape|generic_web_scrape|generic_json_endpoint|wayback_snapshots|pdf_table_extract|csv_download|github_raw|google_sheets|wikipedia_wikidata|official_stats_portal|paywalled_provider|unknown",
      "connector_key": "string",
      "required_params": {},
      "series_id": "string",

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

