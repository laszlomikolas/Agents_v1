# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A research-and-trading scaffold for prediction markets on Polymarket, focused on
**BTC/ETH binary price-threshold markets** that resolve against exchange price candles. The
near-term goal is one end-to-end strategy: data → prediction → allocation → backtest → paper trading.

The repo has two tracks:
- **Trading pipeline (active).** Inventory & normalize markets → select a tradeable universe →
  store price history → predict `P(event)` → allocate / backtest / paper-trade. Built in phases:
  **A** (data foundation) is on `main`; **B** (modeling, `models/`) is in review; **C**
  (signal / allocator / backtest) and **D** (paper trading) are planned.
- **LLM research pipeline (parked for v1).** Two OpenAI Agents SDK agents triage historical-data
  feasibility and generate data connectors. This is bypassed for the price-threshold majors (they
  resolve against existing OHLCV connectors) and returns when the universe expands to long-tail
  symbols / non-price metrics.

## Commands

```bash
# Use the venv interpreter — plain `python` is not on PATH.
#   Windows: ./.venv/Scripts/python.exe

# Run the test suite (pytest; pythonpath is set in pytest.ini)
./.venv/Scripts/python.exe -m pytest -q

# Inventory live crypto markets from Polymarket
./.venv/Scripts/python.exe scripts/inventory_crypto_markets.py

# Refresh the local market-data store (OHLCV + midpoints + meta)
./.venv/Scripts/python.exe scripts/refresh_market_data.py
```

Runtime deps are in `requirements.txt`; test/dev deps (pytest, pytest-cov, matplotlib) in
`requirements-dev.txt`. Tests live in `tests/`. There is no linter config or build system.

## Architecture

### Trading pipeline (active)

**1. Market inventory & universe (`market_inventory/`)**
- `inventory.py` — `inventory_crypto_markets(gamma, coin_universe, project_universe)` queries the
  Polymarket Gamma API, filters `edge`/`range` markets, and extracts symbol/metric/routing metadata
  plus trading identifiers (`clob_token_ids`, `condition_id`, `slug`, `market_id`, `outcomes`,
  `outcome_prices`) into a pandas DataFrame.
- `polymarket_clients.py` — `GammaClient` (httpx) + `ClobClient`. `ClobClient.get_price_history()`
  hits the CLOB `/prices-history` endpoint (real market-price series for backtests);
  `get_midpoint()` for current prices. `GammaClient.list_events()` filters out closed/expired markets.
- `text_utils.py` + `universe.py` — deterministic parsing from question text driven by
  `coins_universe.json` / `projects_universe.json`: `parse_underlying_symbol()`, `parse_metric()`,
  `parse_threshold()` (strike + above/below direction), and the **resolution-mechanics** classifiers
  `parse_threshold_style()` → `"touch"`/`"terminal"` and `resolution_basis()` →
  `"high"`/`"low"`/`"close"`. Touch (barrier) vs terminal (close) is critical: most threshold markets
  resolve intra-window (first-passage on the candle High/Low), not on the terminal close.
- `resolution_routing.py` — `route_resolution_terms()` maps free-text rules to a normalized
  `ResolutionDataType` (`candle_ohlcv`, `daily_metric`, `ranking_snapshot`, ...) and interval.
- `tradeable_universe.py` — `select_tradeable_universe()` narrows an inventory to BTC/ETH
  `candle_ohlcv` markets passing the liquidity screen, enriched with `strike`, `direction`,
  `yes_token_id` (picks the YES token by outcome label, not index).
- `liquidity_screen.py` — `apply_liquidity_screen()` (resting-depth / 30-day-volume thresholds).

**2. Time-series store (`datastore/`)**
- `store.py` — `MarketDataStore`: stdlib-`sqlite3` store (`data/market_data.db`) with tables
  `ohlcv`, `market_prices`, `market_meta`. Idempotent `INSERT OR REPLACE` upserts. `read_ohlcv(...,
  asof=)` is **leak-free**: it filters by candle *close* time (`ts + interval`) so intra-candle reads
  never see an unfinalized candle.
- `refresh.py` — `refresh_data()`: pull OHLCV via the exchange connectors + current midpoints,
  validate with `schema_validation.validate_schema`, and persist. Entry point
  `scripts/refresh_market_data.py`. `data/` and `*.db` are gitignored.

**3. Modeling (`models/`, Phase B — in review)**
- Turns price history into `P(event)` for a market. `features.py` builds leak-free, YES-oriented
  features (signed log-moneyness, normalized distance / driftless `d2`, realized vol, horizon,
  momentum). `dataset.py` samples labeled `(t, horizon, strike)` rows from stored OHLCV — terminal
  (close@T) or touch (window High/Low barrier). `base.py` defines a common `predict_proba`
  interface; `empirical.py` (binned-frequency benchmark), `logistic.py` (scikit-learn primary,
  joblib-persisted), and `gbm.py` (closed-form baseline) implement it. `evaluate.py` reports
  Brier / log-loss / reliability and gates logistic ≥ benchmark.

**Standard exchange connectors (`connectors/`)** — hand-written OHLCV connectors for Binance,
Coinbase, Kraken, Bitstamp, OKX, Bybit, CoinGecko (+ Chainlink round data). Each returns a
standardized `pd.DataFrame`. `schema_validation.py` provides `validate_schema()` / `probe_connector()`
for schema regression detection.

### LLM research pipeline (parked for v1)

Bypassed for the price-threshold majors; kept for when the universe expands to long-tail symbols /
non-price metrics.
- **Triage agent (`pm_agents/historical_data_triage_agent.py`)** — OpenAI Agents SDK
  (`from agents import Agent, Runner, AgentOutputSchema`), model `gpt-5.4`, schema-enforced
  `HistoricalDataTriage`. `triage_market_row(row, timeout_s)` is the async public API.
- **Connector-builder agent (`pm_agents/connector_builder_agent.py`)** — takes a `DataSourcePlan`
  JSON, returns `ConnectorCode` (a complete Python `def` block + imports + pip deps).
- **Runners (`pipeline/`)** — `historical_triage_runner.py` fans out `triage_market_row` with an
  `asyncio.Semaphore` and a parquet cache (`triage_cache.parquet`, keyed on the `market` column);
  `connector_builder_runner.py` dedupes by `connector_key`, persists a connector registry, and emits
  a single connectors module.
- **Pydantic models (`parsing/`)** — `HistoricalDataTriage`, `DataSourcePlan`, `ConnectorCode`,
  `ConnectorType` (`connector_types.py`).

## Key conventions

- **Use the venv Python** (`./.venv/Scripts/python.exe`); run tests with `-m pytest -q`.
- **No look-ahead.** Every feature/decision at time `t` uses only data with `ts <= t` — enforced in
  `store.read_ohlcv(asof=)`, the feature builder, and the dataset generator.
- **Touch vs terminal.** Resolution *terms* are the source of truth; question wording is a fallback
  gated to `candle_ohlcv` rows. Barrier markets key off the window High/Low, terminal markets off the
  close.
- Both LLM pipeline runners expose async (`_async`) and sync wrappers (the sync wrappers use
  `asyncio.run()`). LLM agents use the OpenAI Agents SDK, not the Anthropic SDK; model is `gpt-5.4`.
- The triage agent filters out rows whose `resolution_source` is a known exchange (Chainlink,
  Binance, ...) — these already have structured OHLC connectors; only `None`/URL-style sources go to
  the LLM.
- `triage_plans_json` / `triage_candidates_json` store JSON strings (not dicts) to survive parquet
  round-trips; `_strip_arrow_dtypes()` in `historical_triage_runner.py` works around a pandas/pyarrow
  read issue.
