from __future__ import annotations
import re
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
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


# ── Threshold parsing (strike + direction) ──────────────────────────────────────

_MULTIPLIERS = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}

# Matches "$100,000", "$100k", "$1.2M", "$0.50" and bare "100k"/"1.2m"
# (bare numbers must carry a k/m/b/t suffix so we never grab years or dates).
# The (?![a-z]) lookahead stops the suffix from swallowing the first letter of
# a following word, e.g. the "b" in "$100,000 by Dec 31".
_MONEY_RE = re.compile(
    r"\$\s?\d[\d,]*(?:\.\d+)?(?:\s?[kmbt](?![a-z]))?"
    r"|\b\d[\d,]*(?:\.\d+)?\s?[kmbt](?![a-z])\b",
    re.IGNORECASE,
)

_DIRECTION_ABOVE = (
    "above", "over", "exceed", "greater", "more than", "higher",
    "reach", "hit", "surpass", "at least", "climb", "rise to", "top",
    ">=", ">", "≥",
)
_DIRECTION_BELOW = (
    "below", "under", "beneath", "less than", "lower", "fewer",
    "dip", "drop", "fall", "<=", "<", "≤",
)


def _parse_money_token(raw: str) -> Optional[float]:
    """Convert a matched money token (e.g. '$1.2M', '100k') to a float."""
    cleaned = raw.strip().lower().replace(",", "").replace(" ", "").lstrip("$")
    if not cleaned:
        return None
    mult = 1.0
    if cleaned[-1] in _MULTIPLIERS:
        mult = _MULTIPLIERS[cleaned[-1]]
        cleaned = cleaned[:-1]
    if not cleaned:
        return None
    try:
        return float(cleaned) * mult
    except ValueError:
        return None


def _first_keyword_index(text: str, keywords: tuple[str, ...]) -> Optional[int]:
    best: Optional[int] = None
    for kw in keywords:
        idx = text.find(kw)
        if idx != -1 and (best is None or idx < best):
            best = idx
    return best


def _detect_direction(text: str) -> Optional[str]:
    above_idx = _first_keyword_index(text, _DIRECTION_ABOVE)
    below_idx = _first_keyword_index(text, _DIRECTION_BELOW)
    if above_idx is None and below_idx is None:
        return None
    if above_idx is None:
        return "below"
    if below_idx is None:
        return "above"
    # Both present — pick whichever keyword appears first in the question.
    return "above" if above_idx <= below_idx else "below"


def parse_threshold(question: str) -> tuple[Optional[float], Optional[str]]:
    """Parse a binary price-threshold question into (strike, direction).

    Returns a ``(strike, direction)`` tuple where ``strike`` is the numeric
    price level (e.g. 100000.0) and ``direction`` is ``"above"`` or
    ``"below"``. Either element is ``None`` when it cannot be determined.

    Range questions ("between $X and $Y") are not single thresholds and return
    ``(None, None)`` — those are handled as ``range`` markets elsewhere.

    Examples:
        "Will BTC be above $100,000 by Dec 31?" -> (100000.0, "above")
        "Will Ethereum dip below $2k this week?" -> (2000.0, "below")
        "Will Bitcoin reach $150k in 2026?"      -> (150000.0, "above")
    """
    if not question:
        return None, None

    text = question.lower()
    if "between" in text:
        return None, None

    direction = _detect_direction(text)

    matches = _MONEY_RE.findall(question)
    # Prefer explicit "$"-prefixed amounts; fall back to suffixed bare numbers.
    ordered = [m for m in matches if "$" in m] + [m for m in matches if "$" not in m]
    strike: Optional[float] = None
    for token in ordered:
        value = _parse_money_token(token)
        if value is not None:
            strike = value
            break

    return strike, direction
