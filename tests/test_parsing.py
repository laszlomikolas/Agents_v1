"""Parser tests: threshold parsing and Gamma identifier helpers (A1/A2)."""
import pytest

from market_inventory.inventory import parse_clob_token_ids, parse_outcome_prices
from market_inventory.text_utils import (
    parse_threshold,
    parse_threshold_style,
    resolution_basis,
)

# Resolution-terms fixtures paraphrased from real Polymarket rules.
TOUCH_HIGH_TERMS = (
    'This market will immediately resolve to "Yes" if any Binance 1-minute candle '
    'for Bitcoin (BTC/USDT) on the date specified in the title, between 12:00 AM ET '
    'and 11:59 PM ET has a final "High" price equal to or greater than the price '
    'specified in the title. Otherwise, this market will resolve to "No".'
)
TOUCH_LOW_TERMS = TOUCH_HIGH_TERMS.replace("High", "Low").replace("greater", "lower")
TERMINAL_CLOSE_TERMS = (
    'This market will resolve to "Yes" if the Binance 1 minute candle for ETHUSDT '
    'at 12:00 ET on the date has a final "Close" price of 3,500.01 or higher.'
)
FDV_TERMS = (
    'This market will resolve to "Yes" if the Fully Diluted Valuation of the token '
    "is greater than the value specified in the title 1 day after launch."
)
# Touch-up terms with generic UMA-style settlement/fallback boilerplate mixed in.
# "settlement" appears here only as a generic data-source disclaimer, not as the
# "settlement price" phrase that actually denotes a terminal/close-based market.
TOUCH_HIGH_WITH_SETTLEMENT_BOILERPLATE_TERMS = (
    TOUCH_HIGH_TERMS
    + " In the event of a dispute, this market will be settled using the settlement "
    "process and data sources outlined in the UMA Optimistic Oracle documentation."
)


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


# ── threshold_style: touch (barrier) vs terminal (close) ─────────────────────
@pytest.mark.parametrize(
    "question, terms, expected",
    [
        # Resolution terms are the source of truth.
        ("Will Bitcoin reach $68,000 on June 29?", TOUCH_HIGH_TERMS, "touch"),
        ("Will Bitcoin dip to $58,000 on June 29?", TOUCH_LOW_TERMS, "touch"),
        ("ETH above $3,500 on Mar 29?", TERMINAL_CLOSE_TERMS, "terminal"),
        # Question-wording fallback when terms are absent.
        ("Will Bitcoin reach $150k in 2026?", None, "touch"),
        ("Will Ethereum dip to $2,000?", None, "touch"),
        ("Will BTC be above $100,000 on Dec 31?", None, "terminal"),
        # Non-candle metric markets: undetermined.
        ("Opensea FDV above $1B one day after launch?", FDV_TERMS, None),
    ],
)
def test_parse_threshold_style(question, terms, expected):
    assert parse_threshold_style(question, terms) == expected


# ── resolution_basis: which OHLC field resolves the market ───────────────────
@pytest.mark.parametrize(
    "question, terms, expected",
    [
        ("Will Bitcoin reach $68,000 on June 29?", TOUCH_HIGH_TERMS, "high"),
        ("Will Bitcoin dip to $58,000 on June 29?", TOUCH_LOW_TERMS, "low"),
        ("ETH above $3,500 on Mar 29?", TERMINAL_CLOSE_TERMS, "close"),
        # Fallbacks from style + direction when terms are absent.
        ("Will Bitcoin reach $150k in 2026?", None, "high"),   # touch + above
        ("Will Ethereum dip to $2,000?", None, "low"),         # touch + below
        ("Will BTC be above $100,000 on Dec 31?", None, "close"),  # terminal
    ],
)
def test_resolution_basis(question, terms, expected):
    assert resolution_basis(question, terms) == expected


def test_resolution_basis_ignores_generic_settlement_boilerplate():
    """Regression: a touch-up market whose terms also carry generic settlement/
    fallback boilerplate (e.g. UMA dispute-resolution language) must still resolve
    to "high", not be short-circuited to "close" by the bare word "settlement".
    """
    question = "Will Bitcoin reach $68,000 on June 29?"
    assert (
        resolution_basis(question, TOUCH_HIGH_WITH_SETTLEMENT_BOILERPLATE_TERMS)
        == "high"
    )


# ── data_type gate: non-candle markets must not be labeled via question wording ─
def test_fdv_reach_question_not_mislabeled_as_touch_on_daily_metric():
    """Regression: a 'reach'-phrased FDV question must not get
    threshold_style='touch' / resolution_basis='high' via the question-wording
    fallback. _TOUCH_WORDS ("reach", "hit", ...) are generic and not OHLC-specific,
    so the fallback is gated on ``data_type == 'candle_ohlcv'``.
    """
    question = "Will Token FDV reach $1B?"

    # Without the gate (default data_type=None preserves old behavior), the
    # question-wording fallback would fire and mislabel the FDV market.
    assert parse_threshold_style(question, None) == "touch"
    assert resolution_basis(question, None) == "high"

    # With the gate engaged for a daily_metric row, both helpers return None.
    assert parse_threshold_style(question, None, data_type="daily_metric") is None
    assert resolution_basis(question, None, data_type="daily_metric") is None

    # The gate does not interfere with candle markets — a "reach" question on a
    # candle row still resolves to touch/high.
    assert parse_threshold_style(question, None, data_type="candle_ohlcv") == "touch"
    assert resolution_basis(question, None, data_type="candle_ohlcv") == "high"
