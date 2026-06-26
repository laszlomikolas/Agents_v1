import json
import re
from typing import Any, Optional

import pandas as pd

from market_inventory.polymarket_clients import GammaClient

from .resolution_routing import route_resolution_terms
from .text_utils import parse_metric, parse_underlying_symbol
from .universe import CoinUniverse, ProjectUniverse


def get_crypto_tag_id(gamma: Any) -> int:
    tag = gamma.get_tag_by_slug("crypto")
    return int(tag["id"])


def safe_dt(value: Any) -> Optional[pd.Timestamp]:
    if not value:
        return None
    try:
        return pd.to_datetime(value, utc=True)
    except Exception:
        return None


def normalize_outcomes(outcomes: Any) -> Optional[list[str]]:
    if outcomes is None:
        return None
    if isinstance(outcomes, str):
        try:
            outcomes = json.loads(outcomes)
        except Exception:
            return None
    if isinstance(outcomes, list):
        return [str(outcome) for outcome in outcomes]
    return None


def classify_edge_or_range(question: str, outcomes: Optional[list[str]]) -> str:
    q = (question or "").lower()

    if outcomes and set(outcome.strip().lower() for outcome in outcomes) == {"yes", "no"}:
        if re.search(r"\$\s*\d|\babove\b|\bbelow\b|\bover\b|\bunder\b|\bgreater\b|\bless\b", q):
            return "edge"
        return "unknown"

    if outcomes and len(outcomes) >= 3:
        joined = " ".join(outcomes).lower()
        if any(token in joined for token in ["-", "–", "to", "between", "<", ">", "≤", "≥"]):
            return "range"

    if any(token in q for token in [" between ", " range ", " in a range "]):
        return "range"
    if any(token in q for token in [" above ", " below ", " over ", " under ", " greater than ", " less than "]):
        return "edge"

    return "unknown"


def extract_resolution_source_and_terms(
    mkt: dict, ev: Optional[dict] = None
) -> tuple[Optional[str], Optional[str]]:
    source = (
        mkt.get("resolutionSource")
        or mkt.get("resolution_source")
        or mkt.get("oracle")
        or None
    )

    terms = mkt.get("rules") or mkt.get("resolution") or mkt.get("description") or None

    canonical_end = (
        mkt.get("endDate")
        or mkt.get("endDateIso")
        or mkt.get("closeTime")
        or (ev or {}).get("endDate")
        or (ev or {}).get("endDateIso")
        or (ev or {}).get("closeTime")
        or None
    )

    blob = ((source or "") + "\n" + (terms or "")).lower()
    if source is None:
        for candidate in [
            "chainlink",
            "coinbase",
            "binance",
            "coingecko",
            "kraken",
            "bitstamp",
            "okx",
            "bybit",
        ]:
            if candidate in blob:
                source = candidate

    if source is None and terms:
        match = re.search(r"https?://[^\s)>\]]+", terms)
        if match:
            source = match.group(0)

    if terms:
        terms = terms.strip()
        if canonical_end:
            terms = (
                f"{terms}\n\nCanonical market endDate (API metadata): {canonical_end}."
            )

    if source:
        source = str(source).strip()

    return source, terms


def extract_resolution_date(mkt: dict, ev: dict) -> Optional[pd.Timestamp]:
    candidates = [
        mkt.get("endDate"),
        mkt.get("endDateIso"),
        mkt.get("end_date"),
        mkt.get("closeTime"),
        mkt.get("closeTimeIso"),
        mkt.get("resolveTime"),
        mkt.get("resolveTimeIso"),
        ev.get("endDate"),
        ev.get("endDateIso"),
        ev.get("end_date"),
        ev.get("closeTime"),
        ev.get("closeTimeIso"),
    ]
    for candidate in candidates:
        dt = safe_dt(candidate)
        if dt is not None:
            return dt
    return None


def extract_resolution_date_from_terms(terms: Optional[str]) -> Optional[pd.Timestamp]:
    if not terms:
        return None

    for line in terms.splitlines():
        if line.lower().startswith("canonical market enddate") and ":" in line:
            candidate = line.split(":", 1)[1].strip().rstrip(".")
            dt = safe_dt(candidate)
            if dt is not None:
                return dt

    iso_match = re.search(
        r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\b", terms
    )
    if iso_match:
        dt = safe_dt(iso_match.group(0))
        if dt is not None:
            return dt

    return None


def extract_markets_from_event(ev: dict) -> list[dict]:
    mkts = ev.get("markets")
    if isinstance(mkts, list):
        return mkts
    return []


def inventory_crypto_markets(
    gamma: GammaClient,
    coin_universe: CoinUniverse,
    project_universe: ProjectUniverse,
    limit_events: int = 200,
    max_events_pages: int = 10,
    page_size: int = 100,
) -> pd.DataFrame:
    tag_id = get_crypto_tag_id(gamma)

    rows: list[dict[str, Any]] = []
    offset = 0
    pages = 0

    while pages < max_events_pages:
        events = gamma.list_events(
            tag_id=tag_id,
            active=True,
            closed=False,
            limit=page_size,
            offset=offset,
        )
        if not events:
            break

        for ev in events:
            mkts = extract_markets_from_event(ev)
            for mkt in mkts:
                question = (
                    mkt.get("question") or mkt.get("title") or mkt.get("name") or ""
                ).strip()
                if not question:
                    continue

                outcomes = normalize_outcomes(mkt.get("outcomes"))
                kind = classify_edge_or_range(question, outcomes)
                if kind not in {"edge", "range"}:
                    continue

                symbol = parse_underlying_symbol(question, coin_universe, project_universe)
                metric = parse_metric(question)
                res_dt = extract_resolution_date(mkt, ev)
                res_source, res_terms = extract_resolution_source_and_terms(mkt, ev)
                terms_dt = extract_resolution_date_from_terms(res_terms)
                date_mismatch_warning = False
                if terms_dt is not None:
                    date_mismatch_warning = (
                        res_dt is None or pd.Timestamp(res_dt) != pd.Timestamp(terms_dt)
                    )

                routing = route_resolution_terms(res_source, res_terms)

                rows.append(
                    {
                        "market": question,
                        "kind": kind,
                        "symbol": symbol,
                        "metric": metric,
                        "liquidity_usd": mkt.get("liquidityNum"),
                        "volume_24h_usd": mkt.get("volume24hr"),
                        "volume_30d_usd": mkt.get("volume1mo"),
                        "volume_total_usd": mkt.get("volumeNum"),
                        "resolution_date": res_dt,
                        "resolution_source": res_source,
                        "resolution_terms": res_terms,
                        "warning_end_date_mismatch": date_mismatch_warning,
                        "resolution_data_type": routing.data_type,
                        "resolution_interval": routing.interval,
                        "interval_source": routing.interval_source,
                        "routing_notes": routing.notes,
                    }
                )

                if len(rows) >= limit_events:
                    break
            if len(rows) >= limit_events:
                break

        if len(rows) >= limit_events:
            break

        offset += page_size
        pages += 1

    df = pd.DataFrame(rows)

    if not df.empty:
        df["resolution_date"] = pd.to_datetime(
            df["resolution_date"], utc=True, errors="coerce"
        )
        for col in ("liquidity_usd", "volume_24h_usd", "volume_30d_usd", "volume_total_usd"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["resolution_date"])
        df = df.sort_values(["kind", "resolution_date"]).reset_index(drop=True)

    return df
