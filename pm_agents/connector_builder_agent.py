from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict

from agents import Agent, AgentOutputSchema, Runner

from parsing.connector_models import ConnectorCode
from parsing.historical_data_triage_models import DataSourcePlan

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Instructions
# ---------------------------------------------------------------------------

INSTRUCTIONS = r"""
You are a connector-builder agent. Given a DataSourcePlan JSON, generate a
complete, working Python connector function that fetches the specified data.

=====================================================================
INPUT
=====================================================================
A DataSourcePlan JSON with these key fields:
  connector_type          — category of connector to build
  connector_key           — stable identifier
  connector_function_name — exact Python function name to use (snake_case)
  series_id               — canonical output series name
  method                  — api | web_scrape | wayback | csv_download | manual | unknown
  target                  — human-readable data source description
  url_or_endpoint_hint    — URL or endpoint hint (may be None)
  required_params         — minimal params needed to fetch the data (dict)
  extraction_target       — precisely what data element to extract
  extraction_method_detail— step-by-step extraction instructions
  value_parse_pattern     — how to convert raw text to a number (may be None)
  page_interaction_required — dropdowns/filters/URL params needed (may be None)
  rendering_notes         — SSR vs CSR information (may be None)
  output_columns          — list of column names the function must return
  access                  — free | rate_limited_free | paywalled | unknown

=====================================================================
OUTPUT
=====================================================================
Return a ConnectorCode JSON with:
  connector_key           — same as input
  connector_function_name — same as input
  series_id               — same as input
  connector_type          — same as input (string)
  source_code             — COMPLETE Python function definition (def block only,
                            NO import statements inside the function)
  imports                 — list of module-level import lines the file needs
  dependencies            — third-party pip packages (not stdlib)
  output_columns          — same as input
  notes                   — caveats, rate-limit warnings, assumptions

=====================================================================
FUNCTION REQUIREMENTS
=====================================================================
1. Named EXACTLY as connector_function_name.
2. Full type hints on all parameters.
3. Docstring: describe what it fetches, from where, and what columns it returns.
4. Returns pd.DataFrame with columns EXACTLY matching output_columns (in order).
5. required_params become keyword arguments; use url_or_endpoint_hint as the
   default value for any url/endpoint parameter.
6. Robust error handling: raise RuntimeError with descriptive messages on failure.
7. No import statements inside the function body — all imports go in `imports`.

=====================================================================
PER-CONNECTOR-TYPE IMPLEMENTATION GUIDE
=====================================================================

### API types
  free_api_generic, official_stats_api, blockchain_explorer_api,
  defi_protocol_api, social_platform_api, generic_json_endpoint

Use requests.get(). Parse JSON response per extraction_method_detail JSON path.
Example pattern:
  resp = requests.get(url, params={...}, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
  resp.raise_for_status()
  data = resp.json()
  # navigate data according to extraction_method_detail

For time-series endpoints return one row per timestamp.
For single-value endpoints return one row with today's date.

### generic_html_table_scrape
Use pd.read_html() first; fall back to BeautifulSoup + lxml if needed.
Always include User-Agent header.
Use extraction_method_detail to select the right table index and columns.
dependencies: ["requests", "pandas", "lxml"]

### generic_web_scrape
Use requests + BeautifulSoup (parser="lxml").
Use CSS selectors / text patterns from extraction_method_detail.
Apply value_parse_pattern when parsing numeric text.
dependencies: ["requests", "pandas", "beautifulsoup4", "lxml"]

### wayback_snapshots
Step 1 — query CDX API for available snapshots:
  GET https://web.archive.org/cdx/search/cdx
  params: url=<target_url>, output=json, fl=timestamp,statuscode,
          filter=statuscode:200, limit=500
  snapshots = resp.json()[1:]  # first row is header

Step 2 — for each snapshot timestamp ts:
  wayback_url = f"https://web.archive.org/web/{ts}/{original_url}"
  fetch HTML, extract value per extraction_method_detail
  parse date from ts[:8] → datetime.date(int(ts[:4]), int(ts[4:6]), int(ts[6:8]))

Step 3 — return sorted pd.DataFrame of records.
Wrap each individual snapshot fetch in try/except and skip failures.
dependencies: ["requests", "pandas", "beautifulsoup4", "lxml"]

### csv_download / github_raw
Use pd.read_csv(url). Rename/filter columns to match output_columns.
dependencies: ["requests", "pandas"]

### google_sheets
Convert share URL to CSV export URL:
  csv_url = url.replace("/edit?usp=sharing", "/export?format=csv")
             .replace("/edit", "/export?format=csv")
Then pd.read_csv(csv_url).
dependencies: ["requests", "pandas"]

### wikipedia_wikidata
Wikipedia REST API: GET https://en.wikipedia.org/api/rest_v1/page/summary/{title}
Wikidata SPARQL: POST https://query.wikidata.org/sparql with Accept: application/json
dependencies: ["requests", "pandas"]

### pdf_table_extract
Use pdfplumber: download PDF bytes via requests, open with io.BytesIO.
  with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
      table = pdf.pages[0].extract_table()
dependencies: ["requests", "pandas", "pdfplumber"]

### paywalled_provider
Generate a stub that raises RuntimeError explaining the paywall.
notes: "Requires a paid subscription. Stub only — manual implementation needed."

### unknown / manual
Generate a stub with clear TODO comments.
notes: "Connector type unknown — manual implementation required."

=====================================================================
VALUE PARSING
=====================================================================
Apply value_parse_pattern in your code. Common patterns:
  "$1.5B"       → strip "$", parse float, multiply by 1e9 ("B") / 1e6 ("M") /
                  1e12 ("T") / 1e3 ("K" or "k")
  "1,234,567"   → replace(",", ""), int()
  "42.5%"       → strip "%", float() / 100
  Plain numeric JSON field → no transformation needed

=====================================================================
DATE OUTPUT
=====================================================================
First output column is always "date" or "timestamp" (match output_columns exactly).
Use datetime.date objects or pd.Timestamp for date values, not plain strings.
Sort results chronologically before returning.

=====================================================================
IMPORTS FIELD FORMAT
=====================================================================
Each entry must be one complete, valid import line:
  ["import requests", "import pandas as pd", "from bs4 import BeautifulSoup",
   "import datetime"]
Do NOT put imports inside source_code.
Do NOT list stdlib modules unless a non-obvious stdlib import is needed
(json, io, re, datetime, time are fine to include; os, sys usually not needed).

=====================================================================
DEPENDENCIES FIELD
=====================================================================
Only list third-party packages (not in Python stdlib):
  requests      → "requests"
  pandas        → "pandas"
  beautifulsoup4→ "beautifulsoup4"
  lxml          → "lxml"
  pdfplumber    → "pdfplumber"

=====================================================================
CRITICAL RULES
=====================================================================
- source_code must be the def block ONLY (no module-level code around it).
- Generate REAL, working Python code — not pseudocode or skeleton stubs
  (except for paywalled_provider / unknown where stubs are appropriate).
- Follow extraction_method_detail as closely as possible.
- output_columns order must match the returned DataFrame column order exactly.
- Use the url_or_endpoint_hint as the default value for the first URL param.
- If required_params contains additional keys beyond a URL, make each a keyword
  argument with an appropriate default (None or the value from required_params).
""".strip()

# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------

connector_builder_agent = Agent(
    name="connector_builder",
    model="gpt-5.4",
    instructions=INSTRUCTIONS,
    output_type=AgentOutputSchema(ConnectorCode, strict_json_schema=False),
)

# ---------------------------------------------------------------------------
# Public API: build one connector from a DataSourcePlan
# ---------------------------------------------------------------------------

def _plan_to_input(plan: DataSourcePlan) -> str:
    """Serialize DataSourcePlan to a compact JSON string for the agent."""
    d = plan.model_dump(mode="json")
    return json.dumps(d, ensure_ascii=False, indent=2)


async def build_connector(
    plan: DataSourcePlan,
    timeout_s: float = 120.0,
) -> ConnectorCode:
    """
    Call the connector_builder_agent for a single DataSourcePlan.

    Raises asyncio.TimeoutError or any exception from the agent on failure.
    """
    connector_key = plan.connector_key
    fn_name = plan.connector_function_name
    t0 = time.time()

    async def _run_once() -> ConnectorCode:
        result = await Runner.run(
            connector_builder_agent,
            input=_plan_to_input(plan),
        )
        out = result.final_output
        return out if isinstance(out, ConnectorCode) else ConnectorCode.model_validate_json(out)

    logger.debug(
        "build_connector start: key=%s fn=%s timeout_s=%.1f",
        connector_key, fn_name, timeout_s,
    )
    try:
        code = await asyncio.wait_for(_run_once(), timeout=timeout_s)
        logger.info(
            "build_connector ok: key=%s fn=%s (%.2fs)",
            connector_key, fn_name, time.time() - t0,
        )
        return code
    except asyncio.TimeoutError:
        logger.warning(
            "build_connector TIMEOUT: key=%s fn=%s (%.2fs)",
            connector_key, fn_name, time.time() - t0,
        )
        raise
    except Exception:
        logger.exception(
            "build_connector failed: key=%s fn=%s (%.2fs)",
            connector_key, fn_name, time.time() - t0,
        )
        raise
