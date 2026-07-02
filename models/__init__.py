"""Phase B modeling package.

Predict ``P(event)`` for a BTC/ETH binary price-threshold market from price
history, so the strategy can compare that probability to the market's YES
price and take positive-edge positions.

Two market mechanics are modelled (see ``market_inventory.text_utils``):
    terminal   – resolves on the closing price at the resolution time.
    touch      – resolves the instant the price crosses the strike (barrier);
                 up-barriers key off the window High, down-barriers off the Low.

All feature computation is *as-of* (leak-free): a decision at time ``t`` uses
only candles whose close is ``<= t``.
"""
from __future__ import annotations

from .features import FEATURE_NAMES, build_features, feature_vector

__all__ = ["FEATURE_NAMES", "build_features", "feature_vector"]
