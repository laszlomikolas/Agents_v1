import re
from typing import Optional

from .universe import CoinUniverse, ProjectUniverse


def normalize_text(text: str) -> str:
    normalized = (text or "").lower()
    normalized = re.sub(r"[^a-z0-9\$\s\-]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def words(text: str) -> list[str]:
    return normalize_text(text).split()


def extract_candidates(question: str) -> list[str]:
    normalized = normalize_text(question)
    candidates: list[str] = []

    match = re.search(r"\bwill\s+([a-z0-9\-]+(?:\s+[a-z0-9\-]+){0,3})", normalized)
    if match:
        candidates.append(match.group(1))

    match = re.search(
        r"\b([a-z0-9\-]+(?:\s+[a-z0-9\-]+){0,3})\s+fdv\b", normalized
    )
    if match:
        candidates.append(match.group(1))

    match = re.search(
        r"\b([a-z0-9\-]+(?:\s+[a-z0-9\-]+){0,3})\s+up\s+or\s+down\b",
        normalized,
    )
    if match:
        candidates.append(match.group(1))

    for keyword in [
        "market cap",
        "price",
        "dominance",
        "tvl",
        "volume",
        "supply",
    ]:
        match = re.search(
            rf"\b([a-z0-9\-]+(?:\s+[a-z0-9\-]+){{0,2}})\s+{keyword}\b",
            normalized,
        )
        if match:
            candidates.append(match.group(1))

    candidates.append(normalized)
    return [candidate.strip() for candidate in candidates if candidate and candidate.strip()]


def match_symbol_from_candidate(candidate: str, universe: CoinUniverse) -> Optional[str]:
    tokens = words(candidate)

    for token in tokens:
        if token in universe.symbols:
            return token.upper()

    for n in range(min(4, len(tokens)), 0, -1):
        for i in range(len(tokens) - n + 1):
            phrase = " ".join(tokens[i : i + n])
            if phrase in universe.name_to_symbol:
                return universe.name_to_symbol[phrase].upper()

    return None


def parse_underlying_symbol(
    question: str, coin_universe: CoinUniverse, project_universe: ProjectUniverse
) -> Optional[str]:
    for candidate in extract_candidates(question):
        symbol = match_symbol_from_candidate(candidate, coin_universe)
        if symbol:
            return symbol
        project = project_universe.match(question)
        if project:
            return project
    return None


def parse_metric(question: str) -> str:
    normalized = normalize_text(question)
    if "fdv" in normalized:
        return "fdv"
    if "market cap" in normalized or "marketcap" in normalized:
        return "market_cap"
    if "dominance" in normalized:
        return "dominance"
    if "tvl" in normalized:
        return "tvl"
    if "up or down" in normalized:
        return "direction"
    if "price" in normalized or "$" in normalized:
        return "price"
    return "unknown"
