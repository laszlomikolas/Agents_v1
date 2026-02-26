from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import httpx


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
