"""Parser tests: threshold parsing and Gamma identifier helpers (A1/A2)."""
import pytest

from market_inventory.inventory import parse_clob_token_ids, parse_outcome_prices
from market_inventory.text_utils import parse_threshold


# ── A2: parse_threshold ──────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "question, expected",
    [
        ("Will BTC be above $100,000 by Dec 31?", (100000.0, "above")),
        ("Will Bitcoin reach $150k in 2026?", (150000.0, "above")),
        ("Will ETH hit $5,000?", (5000.0, "above")),
        ("Will Ethereum dip below $2k this week?", (2000.0, "below")),
        ("Will BTC fall under $80,000?", (80000.0, "below")),
        ("Will FOO exceed $1.2M?", (1_200_000.0, "above")),
        ("Will BAR be above $0.50?", (0.50, "above")),
        # Range markets are not single thresholds.
        ("Will BTC be between $90k and $100k?", (None, None)),
    ],
)
def test_parse_threshold(question, expected):
    assert parse_threshold(question) == expected


@pytest.mark.parametrize(
    "question",
    [
        "Will BTC go up or down today?",   # no price level
        "Will BTC do something in 2026?",  # bare year must not be a strike
    ],
)
def test_parse_threshold_no_strike(question):
    strike, _direction = parse_threshold(question)
    assert strike is None


# ── A1: parse_clob_token_ids ─────────────────────────────────────────────────
@pytest.mark.parametrize(
    "value, expected",
    [
        ('["111", "222"]', ["111", "222"]),
        ([111, 222], ["111", "222"]),
        (None, None),
        ("not json", None),
    ],
)
def test_parse_clob_token_ids(value, expected):
    assert parse_clob_token_ids(value) == expected


# ── A1: parse_outcome_prices ─────────────────────────────────────────────────
@pytest.mark.parametrize(
    "value, expected",
    [
        ('["1", "0"]', [1.0, 0.0]),
        (["0.6", 0.4], [0.6, 0.4]),
        (None, None),
        ('["x"]', [None]),  # unparseable element -> None placeholder
    ],
)
def test_parse_outcome_prices(value, expected):
    assert parse_outcome_prices(value) == expected
