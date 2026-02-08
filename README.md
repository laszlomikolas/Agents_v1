# Agents_v1

An agentic hedge fund research scaffold for prediction markets, focused on crypto markets on Polymarket. The current codebase builds a **market inventory**, classifies resolution requirements, and runs an **LLM triage agent** that decides whether historical data is relevant and feasible to source for each market.

## What this project is (so far)

The repository is an early-stage pipeline for:

1. **Inventorying crypto markets** from Polymarket (via the public Gamma API).
2. **Normalizing market metadata** (kind, underlying symbol, metric, resolution terms).
3. **Routing resolution data needs** (e.g., OHLC candles vs daily metrics vs rankings).
4. **Triaging historical data feasibility** using an LLM agent that outputs a structured plan for how to fetch non-OHLC data sources.

In other words, it is scaffolding for a larger “agentic hedge fund” system, with the initial focus on building reliable inputs for forecasting models.

## Repository layout

- **`market_inventory/`**: Core inventory pipeline for crypto markets.
  - `inventory.py` pulls Polymarket crypto events, filters for edge/range markets, and extracts symbol/metric/resolution metadata into a pandas DataFrame.【F:market_inventory/inventory.py†L1-L199】
  - `text_utils.py` and `universe.py` provide deterministic symbol/project matching and metric parsing based on `coins_universe.json` and `projects_universe.json`.【F:market_inventory/text_utils.py†L1-L115】【F:market_inventory/universe.py†L1-L117】
  - `resolution_routing.py` maps resolution terms into a normalized data type + interval (e.g., candle OHLC vs daily metric).【F:market_inventory/resolution_routing.py†L1-L158】
  - `polymarket_clients.py` implements minimal Gamma + CLOB HTTP clients for Polymarket APIs.【F:market_inventory/polymarket_clients.py†L1-L69】

- **`pm_agents/`**: LLM agent definitions.
  - `historical_data_triage_agent.py` defines a strict, schema-driven agent that decides if historical data is useful, feasible, and how to fetch it (with explicit connector planning rules).【F:pm_agents/historical_data_triage_agent.py†L1-L308】

- **`parsing/`**: Structured model definitions for the triage output.
  - `connector_types.py` enumerates supported connector categories (APIs, scraping, Wayback, etc.).【F:parsing/connector_types.py†L1-L32】
  - `historical_data_triage_models.py` contains Pydantic models for the triage output JSON schema.【F:parsing/historical_data_triage_models.py†L1-L108】

- **`pipeline/`**: Execution utilities.
  - `historical_triage_runner.py` runs the triage agent over a DataFrame with concurrency + timeout controls, returning a decorated DataFrame with triage fields and errors.【F:pipeline/historical_triage_runner.py†L1-L199】

- **`scripts/`**: Entry points.
  - `inventory_crypto_markets.py` demonstrates pulling a live inventory of crypto markets from Polymarket and printing the first rows.【F:scripts/inventory_crypto_markets.py†L1-L22】

## How the pieces fit together

1. **Market inventory** (`market_inventory.inventory.inventory_crypto_markets`) queries Polymarket’s Gamma API for crypto-tagged events and extracts markets that look like “edge” or “range” questions. It normalizes outcomes, symbols, metrics, and resolution metadata into a DataFrame.【F:market_inventory/inventory.py†L1-L199】
2. **Resolution routing** (`market_inventory.resolution_routing.route_resolution_terms`) inspects the resolution rules to decide the data type needed (OHLC, daily metric, ranking snapshot, etc.) and chooses a default interval if none is specified.【F:market_inventory/resolution_routing.py†L1-L158】
3. **Historical data triage** (`pm_agents.historical_data_triage_agent`) uses a structured LLM prompt + schema to decide whether historical data is relevant and feasible, and it proposes specific connector plans for non-OHLC data sources.【F:pm_agents/historical_data_triage_agent.py†L1-L308】
4. **Pipeline runner** (`pipeline.historical_triage_runner.triage_dataframe`) applies the triage agent across a DataFrame and appends the output fields for downstream use.【F:pipeline/historical_triage_runner.py†L1-L199】

## Quick start

```bash
python scripts/inventory_crypto_markets.py
```

This script pulls up to 500 crypto-tagged markets from Polymarket (Gamma API), extracts structured metadata, and prints a preview of the resulting DataFrame.【F:scripts/inventory_crypto_markets.py†L1-L22】

## Status

This codebase is a **foundation**: it inventories markets and establishes a structured triage pipeline for acquiring historical data, but it does **not yet** include downstream modeling, trading, or portfolio logic.
