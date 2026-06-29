"""Guard the existing pipeline against the A1 change.

A1 added list-typed columns (clob_token_ids, outcomes, outcome_prices) to the
inventory DataFrame, which the triage runner persists to parquet. This proves
those columns survive a parquet round-trip so the triage cache isn't broken.
"""
import pandas as pd


def test_inventory_list_columns_survive_parquet(tmp_path):
    df = pd.DataFrame(
        [
            {
                "market": "Will BTC be above $100k?",
                "clob_token_ids": ["a", "b"],
                "outcomes": ["Yes", "No"],
                "outcome_prices": [1.0, 0.0],
            },
            {
                "market": "Other market",
                "clob_token_ids": None,
                "outcomes": None,
                "outcome_prices": [0.5, None],
            },
        ]
    )
    path = tmp_path / "inventory.parquet"
    df.to_parquet(path)
    back = pd.read_parquet(path)

    assert list(back.loc[0, "clob_token_ids"]) == ["a", "b"]
    assert list(back.loc[0, "outcomes"]) == ["Yes", "No"]
    assert list(back.loc[0, "outcome_prices"]) == [1.0, 0.0]
    # Null list cell round-trips as a missing value, not a crash.
    assert back.loc[1, "clob_token_ids"] is None
    assert len(back.loc[1, "outcome_prices"]) == 2  # [0.5, None] preserved
