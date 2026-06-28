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
CONNECTOR BUILD SPECIFICATION (CRITICAL FOR DOWNSTREAM AUTOMATION)
=====================================================================
Each plan in plans[] will be consumed by a downstream connector-builder agent that
generates working Python functions. You MUST provide enough detail for that agent to
write a connector WITHOUT visiting the page itself.

For EVERY plan, you MUST fill these additional fields:

1. extraction_target  (REQUIRED — string)
   Describe the EXACT data element to extract from the page or API response.
   Be as specific as possible about what the value represents and where it appears.
   - For dashboards/charts: identify the specific metric, its visual location on the
     page, and any visible text labels near it (e.g. "the headline stat labeled
     'TOTAL SPENDS' showing '$973.3M' in the top-right stats area").
   - For APIs: describe the response field (e.g. "the 'total_volume' field in the
     JSON response array").
   - For tables: describe which row/column contains the data.

2. extraction_method_detail  (REQUIRED — string)
   Step-by-step instructions for how a scraper or fetcher should locate and extract
   the target data. Think about what a developer reading only this field would need:
   - For web pages: suggest CSS selector patterns, text-content patterns to search
     for, or DOM structure hints (e.g. "Look for an element containing text matching
     /TOTAL SPENDS/i; the sibling or child element contains the dollar amount.
     Likely selector: '.stat-value', '.metric-total', or 'h2/h3 near TOTAL SPENDS'.").
   - For JSON APIs: specify the JSON path (e.g. "$.data[*].total_volume").
   - For CSV/tables: specify column names or indices.
   - For Wayback: describe how to reconstruct a time series from snapshots (see
     WAYBACK section below).

3. value_parse_pattern  (string | null)
   How to convert the raw extracted text into a numeric value. Common patterns:
   - Dollar with suffix: "Strip leading '$', parse float, multiply by 1e6 for 'M'
     suffix, 1e9 for 'B' suffix, 1e12 for 'T' suffix."
   - Percentage: "Strip trailing '%', parse float, divide by 100."
   - Comma-separated: "Remove commas, parse as integer."
   - If the value is already numeric (from JSON API), set to null.

4. page_interaction_required  (string | null)
   Any page interactions, dropdown selections, toggle switches, or URL parameters
   needed to put the page into the correct state for extraction.
   - Example: "Select dropdowns: Volume='Cumulative', Scope='All'. These may be
     URL params (?view=cumulative&scope=all) or JS state."
   - If none needed, set to null.

5. rendering_notes  (string | null)
   Whether the page is server-side rendered (SSR) or client-side JS-rendered (CSR).
   This is critical for Wayback and scraping feasibility:
   - SSR: data is in the initial HTML → scraping and Wayback work well.
   - CSR (React/Vue/Angular SPA): data is loaded via XHR/fetch after page load →
     Wayback may only capture the HTML shell. In this case, try to identify the
     underlying API endpoint the frontend calls (look for /api/, GraphQL, or
     data-fetching URLs in common patterns for that site).
   - If unsure, state "likely CSR" and suggest checking for XHR endpoints.

6. output_columns  (list of strings)
   The exact column names the connector function should return as a DataFrame/dict.
   Always include a date/timestamp column first.
   - Example: ["date", "cumulative_crypto_card_volume_usd"]
   - Example: ["timestamp", "btc_holdings", "btc_holdings_usd"]

7. connector_function_name  (REQUIRED — string)
   A descriptive Python function name in snake_case.
   Pattern: fetch_{source}_{metric}
   - Example: "fetch_paymentscan_cumulative_volume"
   - Example: "fetch_arkm_el_salvador_btc_holdings"

=====================================================================
WAYBACK SNAPSHOT PLANS: EXTRACTION DETAIL (IMPORTANT)
=====================================================================
When proposing a wayback_snapshots plan, you must think carefully about how a
connector-builder agent will reconstruct a time series from archived snapshots.

A. Assess rendering mode:
   Many crypto dashboards (Dune, DefiLlama, custom analytics) are JS-rendered SPAs.
   Wayback Machine captures the HTML as-is at crawl time.
   - If SSR: the metric value is in the raw HTML → parse directly.
   - If CSR: the HTML shell may not contain the data. In this case:
     (a) Suggest also looking for an underlying API endpoint that powers the chart.
         The CDX API (web.archive.org/cdx/) can reveal what companion URLs were
         archived alongside the main page.
     (b) If no API is discoverable, note that a headless browser + Wayback replay
         may be needed (higher effort).

B. Identify the extraction target precisely:
   Dashboard pages often have MULTIPLE metrics. You must specify EXACTLY which one:
   - Visible label text near the metric (e.g., "TOTAL SPENDS")
   - Position on page (header stat, sidebar, chart tooltip)
   - Expected format (e.g., "$973.3M", "1,234 BTC")

C. Describe the time-series reconstruction:
   - Each Wayback snapshot yields ONE data point: (snapshot_date, extracted_value).
   - The CDX API at web.archive.org/cdx/search/cdx?url=<target>&output=json returns
     available snapshots with timestamps.
   - Connector should: (1) query CDX for all snapshot timestamps, (2) fetch each
     snapshot, (3) extract the target value, (4) build date→value series.
   - State expected snapshot frequency (daily? weekly? sporadic?) and whether gaps
     are acceptable.

D. Example of a well-specified wayback plan:
   For a market about cumulative crypto card payments using PaymentScan:
   {
     "connector_type": "wayback_snapshots",
     "connector_key": "wayback_snapshots:www.paymentscan.xyz",
     "series_id": "cumulative_crypto_card_volume_usd",
     "method": "wayback",
     "target": "PaymentScan - Cumulative Crypto Card Volumes",
     "url_or_endpoint_hint": "https://www.paymentscan.xyz/",
     "extraction_target": "The 'TOTAL SPENDS' headline metric displayed prominently
       in the page header area, showing the cumulative total as a dollar amount
       (e.g., '$973.3M').",
     "extraction_method_detail": "1) Fetch the Wayback snapshot HTML for each
       archived date. 2) Search for an element whose text content matches
       /TOTAL\\s*SPENDS/i. 3) The dollar amount is in a nearby sibling/child element,
       likely a heading or large-font stat widget. Try selectors: '.stat-value',
       'h2', 'h3', or any element matching /\\$[\\d,.]+[MBT]?/. 4) If the page is
       CSR and the value is not in raw HTML, check CDX for companion API calls
       (e.g., URLs containing /api/ or .json) captured alongside the main page.",
     "value_parse_pattern": "Strip '$', parse float, multiply by 1e6 for 'M',
       1e9 for 'B'. Handle comma separators.",
     "page_interaction_required": "Filters should be set to: Volume type =
       'Cumulative', Card filter = 'All'. Check if these are URL params or
       if the default page state already shows cumulative totals.",
     "rendering_notes": "Likely CSR (JavaScript SPA). Check if Wayback captured
       the rendered state or only the shell. If shell-only, look for XHR/fetch
       endpoints in CDX captures (e.g., an API URL returning JSON chart data).",
     "output_columns": ["date", "cumulative_crypto_card_volume_usd"],
     "connector_function_name": "fetch_paymentscan_cumulative_volume",
     "access": "free",
     "effort": "medium",
     "reliability": "medium",
     "notes": "Snapshot frequency may be sparse; interpolation or gap-filling
       may be needed for daily granularity."
   }

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
- EVERY plan MUST include the connector build spec fields (extraction_target,
  extraction_method_detail, output_columns, connector_function_name). These are
  not optional — a downstream agent depends on them to generate working code.

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
      "notes": "string|null",

      "extraction_target": "string  (REQUIRED — what exact data element to extract)",
      "extraction_method_detail": "string  (REQUIRED — step-by-step extraction instructions)",
      "value_parse_pattern": "string|null  (how to parse raw text to numeric)",
      "page_interaction_required": "string|null  (dropdowns, filters, URL params needed)",
      "rendering_notes": "string|null  (SSR vs CSR, underlying API hints)",
      "output_columns": ["string"],
      "connector_function_name": "string  (REQUIRED — snake_case Python function name)"
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
    model="gpt-5.4",
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

