from __future__ import annotations

from parsing.connector_types import ConnectorType
from pydantic import BaseModel, Field
from typing import Any, Dict, Literal, Optional, List


class DataCandidate(BaseModel):
    name: str = Field(..., description="Name of the underlying series/variable to fetch.")
    unit: Optional[str] = Field(None, description="Unit, e.g. BTC, USD, count, percent.")
    frequency: Optional[str] = Field(None, description="Expected frequency: daily/weekly/monthly/event-driven.")
    proxy_ok: bool = Field(False, description="Whether a proxy series is acceptable.")
    proxy_notes: Optional[str] = Field(None, description="If proxy_ok, describe the proxy and caveats.")


class DataSourcePlan(BaseModel):
    connector_type: ConnectorType = Field(..., description="Normalized connector category.")
    connector_key: str = Field(
        ...,
        description="Stable connector identifier, e.g. 'free_api_generic:api.example.com/v1/series' or "
                    "'wayback_snapshots:example.com/path'.",
    )
    required_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Minimal params needed to fetch the series. Keep JSON-serializable.",
    )
    series_id: str = Field(
        ...,
        description="Canonical output series name in snake_case, e.g. 'el_salvador_btc_holdings_usd'.",
    )

    method: Literal["api", "web_scrape", "wayback", "csv_download", "manual", "unknown"] = Field(
        ..., description="Acquisition method."
    )
    target: str = Field(..., description="What to fetch (site/API/dataset name).")
    url_or_endpoint_hint: Optional[str] = Field(None, description="URL or endpoint hint (best-effort).")

    access: Literal["free", "rate_limited_free", "paywalled", "unknown"] = Field(
        ..., description="Access type."
    )
    paywall_evidence: Optional[str] = Field(None, description="Explain why it seems paywalled.")

    effort: Literal["low", "medium", "high"] = Field(..., description="Engineering effort estimate.")
    reliability: Literal["low", "medium", "high"] = Field(..., description="Expected data reliability.")
    notes: Optional[str] = Field(None, description="Extra notes / pitfalls.")



class HistoricalDataTriage(BaseModel):
    market_id: str
    market: str
    kind: str
    metric: str

    historical_relevance: Literal["yes", "no", "mixed"] = Field(
        ..., description="Is historical data relevant for estimating probability?"
    )
    relevance_rationale: str = Field(..., description="Why/why not. Be specific to the market.")

    data_feasibility: Literal["yes", "maybe", "no"] = Field(
        ..., description="Can we realistically obtain the needed historical data?"
    )
    feasibility_rationale: str = Field(..., description="Why/why not. Mention method constraints.")

    paywall_risk: Literal["none", "possible", "likely"] = Field(
        ..., description="Risk data is paywalled."
    )
    paywall_rationale: str = Field(..., description="Why you think that.")

    candidates: List[DataCandidate] = Field(default_factory=list, description="Candidate series to fetch.")
    plans: List[DataSourcePlan] = Field(default_factory=list, description="Concrete acquisition plans.")

    recommended_resolution: Optional[str] = Field(
        None, description="How you'd build an estimate from the historical series (high-level)."
    )
    routing_notes: Optional[str] = Field(
        None, description="Short routing tag for downstream pipeline (e.g. 'api_ok', 'wayback', 'paywall')."
    )
