from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class ConnectorCode(BaseModel):
    connector_key: str = Field(
        ...,
        description="Stable identifier matching DataSourcePlan.connector_key.",
    )
    connector_function_name: str = Field(
        ...,
        description="Python function name in snake_case.",
    )
    series_id: str = Field(
        ...,
        description="Canonical output series name.",
    )
    connector_type: str = Field(
        ...,
        description="ConnectorType enum value as string.",
    )
    source_code: str = Field(
        ...,
        description=(
            "Complete Python function definition (def block only, no module-level imports). "
            "Must return pd.DataFrame with columns exactly matching output_columns."
        ),
    )
    imports: List[str] = Field(
        default_factory=list,
        description=(
            "Module-level import statements required by source_code. "
            "Each entry is one complete import line, e.g. 'import requests'."
        ),
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description=(
            "Third-party pip packages required beyond stdlib. "
            "E.g. ['requests', 'beautifulsoup4', 'lxml']."
        ),
    )
    output_columns: List[str] = Field(
        default_factory=list,
        description="Column names the function returns, matching DataSourcePlan.output_columns.",
    )
    notes: Optional[str] = Field(
        None,
        description="Known caveats, rate-limit warnings, or assumptions made during code generation.",
    )
