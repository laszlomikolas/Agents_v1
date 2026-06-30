"""End-to-end wiring test for inventory_crypto_markets (no network).

Feeds a fake Gamma client one price-ladder event and asserts the new columns
(threshold_style, resolution_basis, event grouping) are populated correctly.
"""
from market_inventory.inventory import inventory_crypto_markets
from market_inventory.universe import CoinUniverse, ProjectUniverse

_TOUCH_HIGH_TERMS = (
    'This market will immediately resolve to "Yes" if any Binance 1-minute candle '
    'for Bitcoin (BTC/USDT) on the date specified in the title has a final "High" '
    'price equal to or greater than the price specified in the title.'
)


class _FakeGamma:
    """Minimal GammaClient stand-in returning one ladder event."""

    def get_tag_by_slug(self, slug):
        return {"id": 21}

    def list_events(self, tag_id, active, closed, limit, offset):
        if offset > 0:
            return []
        return [
            {
                "title": "What price will Bitcoin hit on June 29?",
                "slug": "what-price-will-bitcoin-hit-on-june-29",
                "seriesSlug": "bitcoin-hit-price-daily",
                "endDate": "2026-06-30T04:00:00Z",
                "markets": [
                    {
                        "question": "Will Bitcoin reach $68,000 on June 29?",
                        "outcomes": '["Yes", "No"]',
                        "clobTokenIds": '["t_yes", "t_no"]',
                        "id": "m1",
                        "slug": "will-bitcoin-reach-68000-on-june-29",
                        "conditionId": "0xcond",
                        "endDate": "2026-06-30T04:00:00Z",
                        "rules": _TOUCH_HIGH_TERMS,
                        "liquidityNum": 50000,
                        "volume1mo": 30000,
                    }
                ],
            }
        ]


def test_inventory_populates_threshold_style_and_grouping():
    cu = CoinUniverse(symbols={"btc"}, name_to_symbol={"bitcoin": "btc"})
    pu = ProjectUniverse(key_to_label={})

    df = inventory_crypto_markets(_FakeGamma(), cu, pu, limit_events=10)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["kind"] == "edge"
    assert row["symbol"] == "BTC"
    assert row["threshold_style"] == "touch"     # "any candle High >=" -> barrier
    assert row["resolution_basis"] == "high"
    assert row["event_title"] == "What price will Bitcoin hit on June 29?"
    assert row["event_slug"] == "what-price-will-bitcoin-hit-on-june-29"
    assert row["series_slug"] == "bitcoin-hit-price-daily"
    # A1 identifiers still flow through.
    assert row["clob_token_ids"] == ["t_yes", "t_no"]
