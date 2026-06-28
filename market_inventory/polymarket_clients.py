from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import httpx
import pandas as pd


def parse_price_history(payload: Any) -> pd.DataFrame:
    """Parse a CLOB /prices-history payload into a tidy DataFrame.

    Accepts either ``{"history": [{"t": <unix_s>, "p": <price>}, ...]}`` or a
    bare list of those points. Returns columns ``timestamp`` (UTC) and
    ``price`` (float); empty (with those columns) when there is no data.
    """
    if isinstance(payload, dict):
        history = payload.get("history")
    elif isinstance(payload, list):
        history = payload
    else:
        history = None

    empty = pd.DataFrame(columns=["timestamp", "price"])
    if not history:
        return empty

    rows: list[tuple[int, float]] = []
    for point in history:
        if not isinstance(point, dict):
            continue
        t = point.get("t")
        p = point.get("p") if point.get("p") is not None else point.get("price")
        if t is None or p is None:
            continue
        try:
            rows.append((int(t), float(p)))
        except (TypeError, ValueError):
            continue

    if not rows:
        return empty

    df = pd.DataFrame(rows, columns=["_ts", "price"])
    df["timestamp"] = pd.to_datetime(df["_ts"], unit="s", utc=True)
    return df[["timestamp", "price"]].sort_values("timestamp").reset_index(drop=True)


@dataclass(frozen=True)
class GammaClient:
    base_url: str = "https://gamma-api.polymarket.com"

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.base_url}{path}"
        with httpx.Client(timeout=30.0) as client:
            r = client.get(url, params=params)
            r.raise_for_status()
            return r.json()

    def list_tags(self, limit: int = 200) -> List[Dict[str, Any]]:
        # Gamma returns a list of tag objects
        return self._get("/tags", params={"limit": limit})

    def list_events(
        self,
        tag_id: int,
        active: bool = True,
        closed: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        events = self._get(
            "/events",
            params={
                "tag_id": tag_id,
                "active": str(active).lower(),
                "closed": str(closed).lower(),
                "limit": limit,
                "offset": offset,
            },
        )

        now = datetime.now(timezone.utc)

        def parse_end_date(value: Any) -> datetime | None:
            if not isinstance(value, str) or not value.strip():
                return None
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
                    timezone.utc
                )
            except ValueError:
                return None

        filtered_events: list[dict] = []
        for event in events:
            markets = event.get("markets")
            if not isinstance(markets, list):
                continue

            filtered_markets = []
            for market in markets:
                if not isinstance(market, dict):
                    continue

                market_end = parse_end_date(
                    market.get("endDate") or market.get("endDateIso") or market.get("closeTime")
                )
                market_active = market.get("active")
                market_closed = market.get("closed")

                if market_active is not True:
                    continue
                if market_closed is not False:
                    continue
                if market_end is None or market_end <= now:
                    continue

                filtered_markets.append(market)

            if not filtered_markets:
                continue

            event_copy = dict(event)
            event_copy["markets"] = filtered_markets
            filtered_events.append(event_copy)

        return filtered_events

    def get_tag_by_slug(self, slug: str) -> dict:
        return self._get(f"/tags/slug/{slug}")

@dataclass(frozen=True)
class ClobClient:
    base_url: str = "https://clob.polymarket.com"

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.base_url}{path}"
        with httpx.Client(timeout=30.0) as client:
            r = client.get(url, params=params)
            r.raise_for_status()
            return r.json()

    def get_midpoint(self, token_id: str) -> float:
        # endpoint: /midpoint?token_id=...
        out = self._get("/midpoint", params={"token_id": token_id})
        # response shape can vary; be defensive
        mp = out.get("midpoint") or out.get("mid") or out.get("price")
        if mp is None:
            raise ValueError(f"Unexpected midpoint payload: {out}")
        return float(mp)

    def get_price_history(
        self,
        token_id: str,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        interval: Optional[str] = None,
        fidelity: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch the historical midpoint time-series for a CLOB token.

        Endpoint: GET /prices-history?market=<token_id>

        Args:
            token_id: The CLOB token id (e.g. the YES outcome token).
            start_ts: Start time as Unix seconds (optional).
            end_ts: End time as Unix seconds (optional).
            interval: Coarse range selector, one of "1m", "1h", "6h", "1d",
                "1w", "max". Used when start/end are not given.
            fidelity: Resolution of the returned series in minutes (optional).

        Returns:
            DataFrame with columns ``timestamp`` (UTC) and ``price`` (float),
            sorted ascending. Empty DataFrame with those columns if no data.
        """
        params: Dict[str, Any] = {"market": token_id}
        if start_ts is not None:
            params["startTs"] = int(start_ts)
        if end_ts is not None:
            params["endTs"] = int(end_ts)
        if interval is not None:
            params["interval"] = interval
        # The endpoint requires either a range or an interval; default to "max".
        if start_ts is None and end_ts is None and interval is None:
            params["interval"] = "max"
        # Bounded intervals (e.g. "1w", "1d", "6h", "1h") require a minimum
        # 'fidelity' (in minutes); "max" does not. Default to hourly when the
        # caller requested a bounded interval without specifying fidelity.
        if fidelity is None and params.get("interval") not in (None, "max"):
            fidelity = 60
        if fidelity is not None:
            params["fidelity"] = int(fidelity)

        out = self._get("/prices-history", params=params)
        return parse_price_history(out)
