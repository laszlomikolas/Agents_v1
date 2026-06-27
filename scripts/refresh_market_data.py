"""Entry point: pull the latest market data and store it.

Run repeatedly (manually or on a schedule) to keep the local SQLite store
populated with the latest OHLCV candles and Polymarket midpoints for the v1
tradeable universe.

Usage:
    python scripts/refresh_market_data.py
"""
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from datastore.refresh import refresh_data


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    summary = refresh_data()
    print("Refresh complete:")
    print(f"  markets in universe : {summary['markets']}")
    print(f"  OHLCV rows stored   : {summary['ohlcv_rows']}")
    print(f"  midpoints stored    : {summary['midpoints']}")
    if summary["ohlcv_errors"]:
        print(f"  OHLCV errors        : {summary['ohlcv_errors']}")
    if summary["midpoint_errors"]:
        print(f"  midpoint errors     : {len(summary['midpoint_errors'])} (see logs)")


if __name__ == "__main__":
    main()
