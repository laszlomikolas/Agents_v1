# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

An agentic scaffold for prediction market research on Polymarket. The pipeline: inventories crypto markets → normalizes metadata → triages historical data feasibility via an LLM agent → builds data connector code via a second LLM agent.

## Commands

```bash
# Run the inventory script (pulls live crypto markets from Polymarket)
python scripts/inventory_crypto_markets.py

# Run a notebook
jupyter notebook notebooks/scratch2.ipynb
```

There is no test suite, linter config, or build system configured in this project yet.

## Architecture

The pipeline has four layers:

### 1. Market inventory (`market_inventory/`)
- `inventory.py` — entry point: `inventory_crypto_markets(gamma, coin_universe, project_universe)` queries the Polymarket Gamma API, filters for `edge`/`range` markets (price-level binary and multi-bucket markets), extracts symbol/metric/routing metadata into a pandas DataFrame.
- `polymarket_clients.py` — minimal `GammaClient` (httpx) and `ClobClient` wrappers. `GammaClient.list_events()` does client-side filtering to exclude already-closed/expired markets.
- `text_utils.py` + `universe.py` — deterministic symbol/project matching driven by `coins_universe.json` and `projects_universe.json`; `parse_underlying_symbol()` and `parse_metric()` extract structured fields from market question text.
- `resolution_routing.py` — `route_resolution_terms()` maps free-text resolution rules to a normalized `ResolutionDataType` (`candle_ohlcv`, `daily_metric`, `ranking_snapshot`, `aggregate_total`, `equity_event`, `other`) and interval.

### 2. LLM triage agent (`pm_agents/historical_data_triage_agent.py`)
Uses the **OpenAI Agents SDK** (`from agents import Agent, Runner, AgentOutputSchema`). The agent model is `gpt-5.4`. Output is schema-enforced via `AgentOutputSchema(HistoricalDataTriage, strict_json_schema=False)`.

`triage_market_row(row, timeout_s)` is the async public API — wraps a single `Runner.run()` call with `asyncio.wait_for`. The instructions embed a detailed rubric: vague-source guardrail, URL-priority rule, connector discovery goal, extraction spec fields.

### 3. LLM connector-builder agent (`pm_agents/connector_builder_agent.py`)
Also uses OpenAI Agents SDK with `gpt-5.4`. Takes a `DataSourcePlan` JSON and returns `ConnectorCode` containing a complete Python `def` block, imports list, and pip dependencies.

### 4. Pipeline runners (`pipeline/`)
- `historical_triage_runner.py` — `triage_dataframe_async()` fans out `triage_market_row` across a DataFrame using `asyncio.Semaphore` for concurrency. `triage_dataframe_incremental()` adds a parquet cache (`triage_cache.parquet`) so only new/changed rows are re-triaged. Cache key is the `market` column (question text); change detection compares `DIFF_COLUMNS`.
- `connector_builder_runner.py` — `build_connectors_async()` parses `triage_plans_json` from the triaged DataFrame, deduplicates by `connector_key`, fans out `build_connector` calls. `save_registry`/`load_registry` persist the connector registry as JSON. `write_connectors_module()` emits a single `.py` file with all generated functions.

### Pydantic models (`parsing/`)
- `HistoricalDataTriage` — triage output schema: relevance, feasibility, paywall, `candidates[]`, `plans[]`.
- `DataSourcePlan` — one acquisition plan: connector type/key, extraction spec fields (`extraction_target`, `extraction_method_detail`, `output_columns`, `connector_function_name`, etc.).
- `ConnectorCode` — LLM-generated connector: `source_code` (def block only), `imports`, `dependencies`.
- `ConnectorType` (`connector_types.py`) — enum of allowed connector categories (e.g. `wayback_snapshots`, `generic_web_scrape`, `defi_protocol_api`).

### Standard exchange connectors (`connectors/`)
Hand-written OHLCV connectors for Binance, Coinbase, Kraken, Bitstamp, OKX, Bybit, CoinGecko, Chainlink. Each returns `pd.DataFrame` with standardized columns. `schema_validation.py` provides `validate_schema()` / `probe_connector()` for schema regression detection.

## Key conventions

- Both pipeline runners expose async (`_async`) and sync wrappers. The sync wrappers use `asyncio.run()`.
- LLM agents use the OpenAI Agents SDK, not the Anthropic SDK. Model is `gpt-5.4`.
- The triage agent filters out rows whose `resolution_source` is a known exchange name (Chainlink, Binance, etc.) — these already have structured OHLC connectors. Only rows with `None` or URL-style sources are passed to the LLM.
- `triage_plans_json` and `triage_candidates_json` columns store JSON strings (not dicts) in the triaged DataFrame to survive parquet round-trips.
- `_strip_arrow_dtypes()` in `historical_triage_runner.py` works around a pandas/pyarrow compatibility issue when reading cached parquet files.
