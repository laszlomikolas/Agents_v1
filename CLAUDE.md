# CLAUDE.md

## Project overview

Agentic hedge fund research scaffold for crypto prediction markets (Polymarket).
Pipeline: **inventory markets → triage historical data feasibility → build data connectors**.

Uses the **OpenAI Agents SDK** (`agents` package) — not Anthropic/Claude API — for all LLM agent calls.

## Architecture patterns

### Agent module pattern (`pm_agents/`)
Every agent follows this structure:
1. `INSTRUCTIONS` — long-form prompt string with rubric, guardrails, output schema docs
2. `Agent()` — instantiated with `model="gpt-5.2-pro-2025-12-11"`, `output_type=AgentOutputSchema(MyModel, strict_json_schema=False)`
3. Async worker function — wraps `Runner.run()` with `asyncio.wait_for()` for timeout, structured logging on start/ok/timeout/error

### Runner pattern (`pipeline/`)
Every runner follows this structure:
1. Takes a `pd.DataFrame` as input
2. Extracts work items, filters, deduplicates
3. Creates all `asyncio.create_task()` calls upfront in a list comprehension
4. Uses `asyncio.Semaphore(max_concurrency)` for bounded parallelism
5. Uses `asyncio.gather()` with optional `asyncio.wait_for()` total timeout
6. Collects results + errors in separate dicts; never lets one failure kill the batch
7. Provides both `async` and sync wrapper (`asyncio.run(...)`) entry points
8. Logs progress every N completions

### Output models (`parsing/`)
All agent outputs are Pydantic `BaseModel` classes with `Field(...)` descriptions.
Enums go in their own files (e.g., `connector_types.py`).

## Code conventions

- `from __future__ import annotations` at the top of every file
- Type hints on all function signatures
- `logging.getLogger(__name__)` — use logger, not print
- snake_case for everything (files, functions, variables, series IDs)
- Async functions suffixed with `_async`; sync wrappers without suffix
- Keyword-only args (`*`) for all optional parameters in public functions
- Keep agent INSTRUCTIONS as raw strings (`r"""..."""`) in the agent module

## Key dependencies (no requirements.txt yet)

- `agents` (OpenAI Agents SDK) — Agent, Runner, AgentOutputSchema
- `pandas` — DataFrames throughout the pipeline
- `pydantic` — BaseModel, Field for structured outputs
- `httpx` — HTTP client for Polymarket Gamma API

## What NOT to do

- Do not propose OHLC/price connectors for chainlink, coinbase, binance, coingecko, kraken, bitstamp, okx, bybit — these already exist upstream
- Do not use `asyncio.run()` inside an async function — only in sync wrappers
- Do not mutate input DataFrames — always `.copy()` first
- Do not put import statements inside generated connector function bodies — they go in the `imports` field
- Do not swallow exceptions silently in runners — log them and store in the errors dict

## Running things

```bash
# Pull market inventory
python scripts/inventory_crypto_markets.py

# Triage (from Python)
from market_inventory.inventory import inventory_crypto_markets
from pipeline.historical_triage_runner import triage_dataframe
triaged = triage_dataframe(df)

# Build connectors (from Python)
from pipeline.connector_builder_runner import build_connectors, load_registry, save_registry
registry = load_registry("data/connector_registry.json")
registry = build_connectors(triaged, registry=registry)
save_registry(registry, "data/connector_registry.json")
```

## File layout

```
market_inventory/   — Polymarket API clients, inventory extraction, resolution routing
pm_agents/          — Agent definitions (INSTRUCTIONS + Agent + async worker)
parsing/            — Pydantic models, enums, output schemas
pipeline/           — Async parallel runners with semaphore concurrency
scripts/            — CLI entry points
```

## Testing

No test infrastructure yet. When adding tests:
- Agent integration tests should mock `Runner.run()` to avoid real LLM calls
- Runner tests should verify parallel execution, timeout handling, and error collection
- Connector tests should validate returned DataFrame columns match `output_columns`
