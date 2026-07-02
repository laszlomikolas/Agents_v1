"""SQLite-backed time-series store for the trading pipeline.

A single SQLite file (``data/market_data.db`` by default) holds the data the
strategy continuously accumulates:

    ohlcv          – underlying price candles, keyed (symbol, interval, ts)
    market_prices  – Polymarket YES-token midpoints over time, keyed (token_id, ts)
    market_meta    – per-market trading metadata (strike, direction, token id, ...)

All writes are idempotent ``INSERT OR REPLACE`` upserts, so re-running the
refresh job simply appends new points and overwrites overlapping ones. All
timestamps are stored as integer Unix seconds (UTC) and returned as tz-aware
``timestamp`` columns.

stdlib ``sqlite3`` only — no new dependency.

TODO: this is a single local file with no backup/durability story and won't
scale well as symbol/price coverage grows. Revisit (DuckDB/Parquet on cloud
storage, or a hosted DB) once the current strategy reaches paper trading —
durability/scalability becomes the first priority after that.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Union

import pandas as pd

DEFAULT_DB_PATH = Path("data") / "market_data.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ohlcv (
    symbol   TEXT    NOT NULL,
    interval TEXT    NOT NULL,
    ts       INTEGER NOT NULL,
    open     REAL,
    high     REAL,
    low      REAL,
    close    REAL,
    volume   REAL,
    PRIMARY KEY (symbol, interval, ts)
);

CREATE TABLE IF NOT EXISTS market_prices (
    token_id TEXT    NOT NULL,
    ts       INTEGER NOT NULL,
    midpoint REAL,
    PRIMARY KEY (token_id, ts)
);

CREATE TABLE IF NOT EXISTS market_meta (
    market_id         TEXT PRIMARY KEY,
    market            TEXT,
    symbol            TEXT,
    kind              TEXT,
    strike            REAL,
    direction         TEXT,
    yes_token_id      TEXT,
    condition_id      TEXT,
    slug              TEXT,
    resolution_date   TEXT,
    resolution_source TEXT,
    liquidity_usd     REAL,
    volume_30d_usd    REAL,
    updated_at        INTEGER
);
"""

_META_COLUMNS = [
    "market_id", "market", "symbol", "kind", "strike", "direction",
    "yes_token_id", "condition_id", "slug", "resolution_date",
    "resolution_source", "liquidity_usd", "volume_30d_usd", "updated_at",
]


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(result):
        return None
    return result


def _epoch_scalar(value: Any) -> int:
    """Convert a timestamp-like value to integer Unix seconds (UTC)."""
    ts = pd.Timestamp(value)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return int(ts.timestamp())


def _epoch_series(values: Iterable[Any]) -> pd.Series:
    """Vectorized conversion of timestamp-like values to Unix seconds (UTC)."""
    ts = pd.to_datetime(pd.Series(list(values)), utc=True)
    epoch0 = pd.Timestamp("1970-01-01", tz="UTC")
    return ((ts - epoch0) // pd.Timedelta(seconds=1)).astype("int64")


def _now_epoch() -> int:
    return int(pd.Timestamp.now(tz="UTC").timestamp())


_INTERVAL_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800, "M": 2592000}


def _interval_to_seconds(interval: str) -> int:
    """Map an interval string like '1m', '1h', '1d', '1w', '1M' to seconds.

    'M' (month ≈ 30 days) is case-sensitive and distinct from 'm' (minute).
    All other unit letters are case-insensitive.
    """
    if not isinstance(interval, str) or not interval:
        raise ValueError(f"interval must be a non-empty string, got {interval!r}")
    stripped = interval.strip()
    unit = stripped[-1:]
    # Preserve 'M' (month) as uppercase; lowercase everything else.
    if unit != "M":
        stripped = stripped.lower()
        unit = stripped[-1:]
    if unit not in _INTERVAL_UNITS:
        raise ValueError(f"unsupported interval unit in {interval!r}")
    try:
        n = int(stripped[:-1])
    except ValueError as exc:
        raise ValueError(f"invalid interval {interval!r}") from exc
    if n <= 0:
        raise ValueError(f"interval must be positive, got {interval!r}")
    return n * _INTERVAL_UNITS[unit]


class MarketDataStore:
    """Thin idempotent wrapper around a SQLite market-data database."""

    def __init__(self, path: Union[str, Path] = DEFAULT_DB_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ── connection / schema ──────────────────────────────────────────────────
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    def _executemany(self, sql: str, records: list[tuple]) -> int:
        if not records:
            return 0
        conn = self._connect()
        try:
            conn.executemany(sql, records)
            conn.commit()
        finally:
            conn.close()
        return len(records)

    def _read(self, sql: str, params: list[Any]) -> list[tuple]:
        conn = self._connect()
        try:
            rows = conn.execute(sql, params).fetchall()
        finally:
            conn.close()
        return [tuple(row) for row in rows]

    # ── writes ───────────────────────────────────────────────────────────────
    def upsert_ohlcv(self, symbol: str, interval: str, df: pd.DataFrame) -> int:
        """Upsert OHLCV candles for one (symbol, interval). Returns row count."""
        if df is None or df.empty:
            return 0
        if "timestamp" not in df.columns:
            raise ValueError("OHLCV DataFrame must have a 'timestamp' column")
        work = df.dropna(subset=["timestamp"]).reset_index(drop=True)
        if work.empty:
            return 0
        epochs = _epoch_series(work["timestamp"]).tolist()
        records = [
            (
                symbol, interval, int(epoch),
                _to_float(row.get("open")), _to_float(row.get("high")),
                _to_float(row.get("low")), _to_float(row.get("close")),
                _to_float(row.get("volume")),
            )
            for epoch, (_, row) in zip(epochs, work.iterrows())
        ]
        return self._executemany(
            "INSERT OR REPLACE INTO ohlcv "
            "(symbol, interval, ts, open, high, low, close, volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            records,
        )

    def record_midpoint(self, token_id: str, midpoint: float, ts: Any = None) -> None:
        """Record a single current midpoint for a token (defaults to now)."""
        epoch = _now_epoch() if ts is None else _epoch_scalar(ts)
        self._executemany(
            "INSERT OR REPLACE INTO market_prices (token_id, ts, midpoint) "
            "VALUES (?, ?, ?)",
            [(str(token_id), int(epoch), _to_float(midpoint))],
        )

    def upsert_market_prices(self, token_id: str, df: pd.DataFrame) -> int:
        """Upsert a midpoint time-series for one token (e.g. CLOB price history).

        Accepts a DataFrame with a ``timestamp`` column and a ``midpoint`` or
        ``price`` column.
        """
        if df is None or df.empty:
            return 0
        if "timestamp" not in df.columns:
            raise ValueError("price DataFrame must have a 'timestamp' column")
        value_col = "midpoint" if "midpoint" in df.columns else "price"
        if value_col not in df.columns:
            raise ValueError("price DataFrame must have a 'midpoint' or 'price' column")
        work = df.dropna(subset=["timestamp"]).reset_index(drop=True)
        if work.empty:
            return 0
        epochs = _epoch_series(work["timestamp"]).tolist()
        records = [
            (str(token_id), int(epoch), _to_float(row.get(value_col)))
            for epoch, (_, row) in zip(epochs, work.iterrows())
        ]
        return self._executemany(
            "INSERT OR REPLACE INTO market_prices (token_id, ts, midpoint) "
            "VALUES (?, ?, ?)",
            records,
        )

    def upsert_market_meta(
        self, records: Union[pd.DataFrame, Iterable[Mapping[str, Any]]]
    ) -> int:
        """Upsert per-market metadata rows keyed by ``market_id``."""
        if isinstance(records, pd.DataFrame):
            rows: Iterable[Mapping[str, Any]] = records.to_dict("records")
        else:
            rows = list(records)

        now = _now_epoch()
        tuples: list[tuple] = []
        for rec in rows:
            market_id = rec.get("market_id") or rec.get("slug") or rec.get("yes_token_id")
            if market_id is None:
                continue
            res_date = rec.get("resolution_date")
            if isinstance(res_date, pd.Timestamp):
                res_date = None if pd.isna(res_date) else res_date.isoformat()
            elif res_date is not None:
                res_date = str(res_date)
            tuples.append((
                str(market_id),
                rec.get("market"),
                rec.get("symbol"),
                rec.get("kind"),
                _to_float(rec.get("strike")),
                rec.get("direction"),
                None if rec.get("yes_token_id") is None else str(rec.get("yes_token_id")),
                None if rec.get("condition_id") is None else str(rec.get("condition_id")),
                rec.get("slug"),
                res_date,
                rec.get("resolution_source"),
                _to_float(rec.get("liquidity_usd")),
                _to_float(rec.get("volume_30d_usd")),
                now,
            ))
        return self._executemany(
            "INSERT OR REPLACE INTO market_meta "
            "(" + ", ".join(_META_COLUMNS) + ") "
            "VALUES (" + ", ".join(["?"] * len(_META_COLUMNS)) + ")",
            tuples,
        )

    # ── reads ────────────────────────────────────────────────────────────────
    def read_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: Any = None,
        end: Any = None,
        asof: Any = None,
    ) -> pd.DataFrame:
        """Read OHLCV candles. ``asof`` restricts to candles whose close time
        (``ts + interval_seconds``) is ``<= asof``, so intra-candle queries
        never see a candle whose OHLC is not yet finalized."""
        sql = "SELECT ts, open, high, low, close, volume FROM ohlcv WHERE symbol=? AND interval=?"
        params: list[Any] = [symbol, interval]
        if asof is not None:
            sql += " AND ts + ? <= ?"
            params.append(_interval_to_seconds(interval))
            params.append(_epoch_scalar(asof))
        if start is not None:
            sql += " AND ts >= ?"
            params.append(_epoch_scalar(start))
        if end is not None:
            sql += " AND ts <= ?"
            params.append(_epoch_scalar(end))
        sql += " ORDER BY ts"
        rows = self._read(sql, params)
        cols = ["open", "high", "low", "close", "volume"]
        if not rows:
            return pd.DataFrame(columns=["timestamp", *cols])
        df = pd.DataFrame(rows, columns=["ts", *cols])
        df["timestamp"] = pd.to_datetime(df["ts"], unit="s", utc=True)
        return df[["timestamp", *cols]]

    def read_ohlcv_asof(self, symbol: str, interval: str, asof: Any) -> pd.DataFrame:
        """Convenience wrapper: OHLCV with ts <= asof (leak-free slice)."""
        return self.read_ohlcv(symbol, interval, asof=asof)

    def read_market_prices(
        self, token_id: str, start: Any = None, end: Any = None, asof: Any = None
    ) -> pd.DataFrame:
        """Read a token's midpoint series as columns ``timestamp``, ``midpoint``."""
        sql = "SELECT ts, midpoint FROM market_prices WHERE token_id=?"
        params: list[Any] = [token_id]
        if asof is not None:
            sql += " AND ts <= ?"
            params.append(_epoch_scalar(asof))
        if start is not None:
            sql += " AND ts >= ?"
            params.append(_epoch_scalar(start))
        if end is not None:
            sql += " AND ts <= ?"
            params.append(_epoch_scalar(end))
        sql += " ORDER BY ts"
        rows = self._read(sql, params)
        if not rows:
            return pd.DataFrame(columns=["timestamp", "midpoint"])
        df = pd.DataFrame(rows, columns=["ts", "midpoint"])
        df["timestamp"] = pd.to_datetime(df["ts"], unit="s", utc=True)
        return df[["timestamp", "midpoint"]]

    def read_meta(self, market_id: Optional[str] = None) -> pd.DataFrame:
        """Read market metadata; all rows, or one row if ``market_id`` given."""
        sql = "SELECT " + ", ".join(_META_COLUMNS) + " FROM market_meta"
        params: list[Any] = []
        if market_id is not None:
            sql += " WHERE market_id=?"
            params.append(str(market_id))
        rows = self._read(sql, params)
        return pd.DataFrame(rows, columns=_META_COLUMNS)
