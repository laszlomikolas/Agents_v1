import re
from dataclasses import dataclass
from typing import Literal, Optional

ResolutionDataType = Literal[
    "candle_ohlcv",
    "daily_metric",
    "ranking_snapshot",
    "aggregate_total",
    "equity_event",
    "other",
]

IntervalSource = Literal[
    "explicit",
    "default_1m",
    "default_1d",
    "none",
]


@dataclass(frozen=True)
class ResolutionRouting:
    data_type: ResolutionDataType
    interval: Optional[str]  # "1m", "5m", "1d", etc.
    interval_source: IntervalSource  # explicit vs default
    notes: Optional[str] = None


_INTERVAL_PATTERNS = [
    ("1m", re.compile(r"\b1\s*m(in(ute)?)?\b|\b1m\b|one[-\s]?minute", re.IGNORECASE)),
    ("5m", re.compile(r"\b5\s*m(in(ute)?)?\b|\b5m\b|five[-\s]?minute", re.IGNORECASE)),
    ("15m", re.compile(r"\b15\s*m(in(ute)?)?\b|\b15m\b|fifteen[-\s]?minute", re.IGNORECASE)),
    ("30m", re.compile(r"\b30\s*m(in(ute)?)?\b|\b30m\b|thirty[-\s]?minute", re.IGNORECASE)),
    ("1h", re.compile(r"\b1\s*h(our)?\b|\b1h\b|one[-\s]?hour", re.IGNORECASE)),
    ("4h", re.compile(r"\b4\s*h(our)?\b|\b4h\b|four[-\s]?hour", re.IGNORECASE)),
    ("1d", re.compile(r"\b1\s*d(ay)?\b|\b1d\b|daily\b", re.IGNORECASE)),
]

_CANDLE_SIGNALS = re.compile(
    r"\b(candle|candles|ohlc|open|high|low|close|closing price)\b",
    re.IGNORECASE,
)

_DAILY_METRIC_SIGNALS = re.compile(
    r"\b(daily|historical data|market cap|marketcap|fdv)\b",
    re.IGNORECASE,
)

_RANKING_SIGNALS = re.compile(
    r"\b(top\s*100|rank|ranking)\b",
    re.IGNORECASE,
)

_AGG_TOTAL_SIGNALS = re.compile(
    r"\b(total amount raised|amount raised|total raised|funds raised)\b",
    re.IGNORECASE,
)

_EQUITY_EVENT_SIGNALS = re.compile(
    r"\b(ipo|first trading day|outstanding shares|primary exchange)\b",
    re.IGNORECASE,
)


def _extract_explicit_interval(terms: str) -> Optional[str]:
    if not terms:
        return None
    for code, pattern in _INTERVAL_PATTERNS:
        if pattern.search(terms):
            return code
    return None


def route_resolution_terms(source: Optional[str], terms: Optional[str]) -> ResolutionRouting:
    """
    Deterministic routing:
    1) If OHLC/candle is referenced -> candle_ohlcv, interval explicit or default 1m
    2) Else if equity/ipo -> equity_event (no interval)
    3) Else if aggregate totals -> aggregate_total (no interval)
    4) Else if rankings -> ranking_snapshot (no interval)
    5) Else if daily metric -> daily_metric (interval 1d)
    6) Else other
    """
    blob = ((source or "") + "\n" + (terms or "")).lower()

    explicit_interval = _extract_explicit_interval(blob)

    if _CANDLE_SIGNALS.search(blob):
        if explicit_interval and explicit_interval != "1d":
            return ResolutionRouting(
                data_type="candle_ohlcv",
                interval=explicit_interval,
                interval_source="explicit",
                notes=None,
            )
        return ResolutionRouting(
            data_type="candle_ohlcv",
            interval="1m",
            interval_source="default_1m",
            notes="no explicit interval; defaulted to 1m",
        )

    if _EQUITY_EVENT_SIGNALS.search(blob):
        return ResolutionRouting(
            data_type="equity_event",
            interval=None,
            interval_source="none",
            notes="equity/IPO style resolution",
        )

    if _AGG_TOTAL_SIGNALS.search(blob):
        return ResolutionRouting(
            data_type="aggregate_total",
            interval=None,
            interval_source="none",
            notes="aggregate total (funding/raised) resolution",
        )

    if _RANKING_SIGNALS.search(blob):
        return ResolutionRouting(
            data_type="ranking_snapshot",
            interval=None,
            interval_source="none",
            notes="ranking snapshot resolution",
        )

    if (explicit_interval == "1d") or _DAILY_METRIC_SIGNALS.search(blob):
        return ResolutionRouting(
            data_type="daily_metric",
            interval="1d",
            interval_source="default_1d" if explicit_interval is None else "explicit",
            notes="daily metric / historical data resolution",
        )

    return ResolutionRouting(
        data_type="other",
        interval=None,
        interval_source="none",
        notes="no clear candle/daily/ranking/aggregate/equity signals",
    )
