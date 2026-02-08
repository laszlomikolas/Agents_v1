import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from market_inventory.universe import CoinUniverse, ProjectUniverse
from market_inventory.inventory import inventory_crypto_markets
from market_inventory.polymarket_clients import GammaClient

def main() -> None:
    coin_universe = CoinUniverse.from_json()
    project_universe = ProjectUniverse.from_json()
    gamma = GammaClient()

    df = inventory_crypto_markets(
        gamma=gamma,
        coin_universe=coin_universe,
        project_universe=project_universe,
        limit_events=500,
    )
    print(df.head(30))


if __name__ == "__main__":
    main()
