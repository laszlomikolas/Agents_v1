"""
Schema validation for standard exchange connectors.

Provides:
  - ColumnSpec / SchemaSpec  – declarative schema definitions
  - ValidationResult         – structured validation output
  - validate_schema()        – validate a DataFrame against a SchemaSpec
  - probe_connector()        – call a connector and validate its output

Schema-change detection works by comparing a live connector call against the
expected ColumnSpec list. Any mismatch (missing columns, wrong dtype kind,
all-NaN column, OHLC sanity failure, non-positive prices, etc.) surfaces as
an error in the returned ValidationResult, making it easy to detect when an
upstream API has silently changed its response format.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ── Schema primitives ─────────────────────────────────────────────────────────

@dataclass
class ColumnSpec:
    """Describes one expected column in a connector's output DataFrame.

    Attributes:
        name: Column name.
        dtype_kind: Single-character numpy dtype kind:
            'M' = datetime/timestamp, 'f' = float, 'i' = signed int,
            'u' = unsigned int, 'O' = object/string.
        nullable: If False (default), the column must not be entirely NaN.
        min_value: Optional lower bound on numeric values (inclusive).
        max_value: Optional upper bound on numeric values (inclusive).
    """
    name: str
    dtype_kind: str
    nullable: bool = False
    min_value: Optional[float] = None
    max_value: Optional[float] = None


@dataclass
class SchemaSpec:
    """Full schema specification for one connector.

    Attributes:
        name: Connector name used in log messages and result labels.
        columns: Ordered list of expected output columns.
        min_rows: Minimum acceptable row count.
        probe_kwargs: Default keyword arguments forwarded to the connector
            function when calling probe_connector().
        check_timestamp_monotonic: Assert that the 'timestamp' column is
            strictly non-decreasing.
        check_ohlc_sanity: When True, enforce OHLC invariants:
            high ≥ max(open, close), low ≤ min(open, close), all prices > 0,
            volume ≥ 0.
    """
    name: str
    columns: list[ColumnSpec]
    min_rows: int = 1
    probe_kwargs: dict[str, Any] = field(default_factory=dict)
    check_timestamp_monotonic: bool = True
    check_ohlc_sanity: bool = False


# ── Validation result ──────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """Outcome of validating a connector's output.

    Attributes:
        connector_name: Name of the connector being validated.
        ok: True iff there are no errors.
        errors: List of schema violation messages.
        warnings: List of non-fatal observations (e.g. extra columns).
        actual_columns: Column names present in the DataFrame.
        actual_dtypes: Mapping of column name → dtype string.
        row_count: Number of rows in the DataFrame (0 if connector raised).
    """
    connector_name: str
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    actual_columns: list[str] = field(default_factory=list)
    actual_dtypes: dict[str, str] = field(default_factory=dict)
    row_count: int = 0

    def summary(self) -> str:
        """Return a human-readable one-to-many-line summary."""
        status = "OK  " if self.ok else "FAIL"
        lines = [f"[{status}] {self.connector_name:<12} rows={self.row_count}"]
        for msg in self.errors:
            lines.append(f"       ERROR   {msg}")
        for msg in self.warnings:
            lines.append(f"       WARNING {msg}")
        return "\n".join(lines)


# ── Core validator ─────────────────────────────────────────────────────────────

def validate_schema(df: pd.DataFrame, spec: SchemaSpec) -> ValidationResult:
    """
    Validate connector output *df* against *spec*.

    Checks performed (in order):
    1. Row count ≥ spec.min_rows
    2. Each expected column exists
    3. Each column's dtype kind matches the spec
    4. No expected column is entirely NaN (unless nullable=True)
    5. Numeric min/max bounds where specified
    6. Timestamp monotonicity (if spec.check_timestamp_monotonic)
    7. OHLC sanity invariants (if spec.check_ohlc_sanity)
    8. Unexpected extra columns → warning (schema may have expanded)

    Returns:
        ValidationResult with ok=True iff no errors were found.
    """
    errors: list[str] = []
    warnings: list[str] = []
    actual_cols = list(df.columns)
    actual_dtypes = {c: str(df[c].dtype) for c in actual_cols}

    # 1. Row count
    if len(df) < spec.min_rows:
        errors.append(f"Expected ≥{spec.min_rows} rows, got {len(df)}")

    # 2–5. Per-column checks
    spec_names = {cs.name for cs in spec.columns}
    for cs in spec.columns:
        if cs.name not in df.columns:
            errors.append(f"Missing column '{cs.name}'")
            continue

        col = df[cs.name]

        # 3. Dtype kind
        actual_kind = col.dtype.kind
        if actual_kind != cs.dtype_kind:
            # Allow int stored as float (common in pandas) and uint ↔ int
            compatible = (
                (cs.dtype_kind == "f" and actual_kind in ("i", "u"))
                or (cs.dtype_kind == "i" and actual_kind == "u")
            )
            if not compatible:
                errors.append(
                    f"Column '{cs.name}': expected dtype kind '{cs.dtype_kind}', "
                    f"got '{actual_kind}' ({col.dtype})"
                )

        # 4. All-NaN
        if not cs.nullable and col.isna().all():
            errors.append(f"Column '{cs.name}' is entirely NaN")

        # 5. Numeric bounds
        numeric = col.dropna()
        if len(numeric) > 0 and cs.min_value is not None:
            if numeric.lt(cs.min_value).any():
                errors.append(
                    f"Column '{cs.name}': values below min_value={cs.min_value}"
                )
        if len(numeric) > 0 and cs.max_value is not None:
            if numeric.gt(cs.max_value).any():
                errors.append(
                    f"Column '{cs.name}': values above max_value={cs.max_value}"
                )

    # 6. Timestamp monotonicity
    if spec.check_timestamp_monotonic and "timestamp" in df.columns and len(df) > 1:
        if not df["timestamp"].is_monotonic_increasing:
            errors.append("Column 'timestamp' is not monotonically increasing")

    # 7. OHLC sanity
    if spec.check_ohlc_sanity:
        ohlc = {"open", "high", "low", "close"}
        if ohlc.issubset(df.columns):
            row_max = df[["open", "close"]].max(axis=1)
            row_min = df[["open", "close"]].min(axis=1)
            if (df["high"] < row_max).any():
                errors.append("OHLC sanity: 'high' < max(open, close) in some rows")
            if (df["low"] > row_min).any():
                errors.append("OHLC sanity: 'low' > min(open, close) in some rows")
            if (df[["open", "high", "low", "close"]] <= 0).any(axis=None):
                errors.append("OHLC sanity: non-positive price value detected")
        if "volume" in df.columns:
            if (df["volume"] < 0).any():
                errors.append("OHLC sanity: negative volume value detected")

    # 8. Extra columns (non-fatal)
    extra = [c for c in actual_cols if c not in spec_names]
    if extra:
        warnings.append(
            f"Unexpected extra columns (upstream schema may have expanded): {extra}"
        )

    return ValidationResult(
        connector_name=spec.name,
        ok=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        actual_columns=actual_cols,
        actual_dtypes=actual_dtypes,
        row_count=len(df),
    )


# ── Connector probe ────────────────────────────────────────────────────────────

def probe_connector(
    spec: SchemaSpec,
    connector_fn: Callable[..., pd.DataFrame],
    **override_kwargs: Any,
) -> ValidationResult:
    """
    Call *connector_fn* and validate the result against *spec*.

    Keyword arguments are spec.probe_kwargs merged with override_kwargs
    (overrides win). Any exception raised by the connector is captured and
    returned as an error inside the ValidationResult rather than propagated,
    so callers can probe multiple connectors in a loop without aborting early.

    Args:
        spec: Expected schema.
        connector_fn: The connector function to call.
        **override_kwargs: Per-call overrides to spec.probe_kwargs.

    Returns:
        ValidationResult indicating pass/fail and any schema violations.
    """
    kwargs = {**spec.probe_kwargs, **override_kwargs}
    try:
        df = connector_fn(**kwargs)
    except Exception as exc:
        logger.warning("probe_connector: %s raised %s: %s", spec.name, type(exc).__name__, exc)
        return ValidationResult(
            connector_name=spec.name,
            ok=False,
            errors=[f"Connector raised {type(exc).__name__}: {exc}"],
        )
    return validate_schema(df, spec)
