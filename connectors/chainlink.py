"""Chainlink on-chain price feed connector via Ethereum JSON-RPC (no web3 dependency)."""
from __future__ import annotations

import requests
import pandas as pd
from typing import Optional

# Public Ethereum RPC endpoint (no API key required)
_DEFAULT_RPC = "https://eth.llamarpc.com"
_HEADERS = {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}

# ABI function selectors (keccak256 of canonical signature, first 4 bytes)
_SEL_LATEST = "0xfeaf968c"       # latestRoundData()
_SEL_GET_ROUND = "0x9a6fc8f5"   # getRoundData(uint80)

# USD-denominated price feeds use 8 decimal places
_USD_DECIMALS = 8

# Ethereum mainnet Chainlink price feed contract addresses
FEED_ADDRESSES: dict[str, str] = {
    "BTC/USD":  "0xF4030086522a5bEEa4988F8cA5B36dbC97BeE88b",
    "ETH/USD":  "0x5f4eC3Df9cbd43714FE2740f5E3616155c5b8419",
    "LINK/USD": "0x2c1d072e956AFFC0D435Cb7AC38EF18d24d9127c",
    "SOL/USD":  "0x4ffC43a60e009B551865A93d232E33Fce9f01507",
    "BNB/USD":  "0x14e613AC84a31f709eadbEF3bf98bBFad7e5BE6A",
}


def _eth_call(rpc_url: str, contract: str, calldata: str) -> str:
    """Issue a single eth_call JSON-RPC request and return the hex result."""
    payload = {
        "jsonrpc": "2.0",
        "method": "eth_call",
        "params": [{"to": contract, "data": calldata}, "latest"],
        "id": 1,
    }
    resp = requests.post(rpc_url, json=payload, headers=_HEADERS, timeout=15)
    resp.raise_for_status()
    body = resp.json()
    if "error" in body:
        raise RuntimeError(f"JSON-RPC error: {body['error']}")
    return body["result"]


def _decode_round_data(hex_result: str) -> dict:
    """
    Decode the 5-tuple ABI response from latestRoundData() / getRoundData(uint80).

    ABI layout (5 × 32-byte words):
      [0] roundId         uint80
      [1] answer          int256
      [2] startedAt       uint256
      [3] updatedAt       uint256
      [4] answeredInRound uint80
    """
    raw = bytes.fromhex(hex_result.removeprefix("0x"))
    if len(raw) < 160:
        raise ValueError(f"Expected ≥160 bytes in RPC response, got {len(raw)}")
    return {
        "round_id":   int.from_bytes(raw[0:32], "big"),
        "answer":     int.from_bytes(raw[32:64], "big", signed=True),
        "started_at": int.from_bytes(raw[64:96], "big"),
        "updated_at": int.from_bytes(raw[96:128], "big"),
    }


def fetch_chainlink_price_feed(
    feed: str = "BTC/USD",
    contract_address: Optional[str] = None,
    decimals: int = _USD_DECIMALS,
    n_rounds: int = 90,
    rpc_url: str = _DEFAULT_RPC,
) -> pd.DataFrame:
    """
    Fetch recent price history from a Chainlink on-chain price feed.

    Iterates backwards from the latest round ID, calling getRoundData() for
    each, and returns up to n_rounds valid data points. Rounds with missing
    or zero updatedAt are silently skipped (they occur at phase transitions).

    No API key is required. Uses eth.llamarpc.com by default; any public or
    private Ethereum JSON-RPC endpoint can be substituted via rpc_url.

    Args:
        feed: Feed name key in FEED_ADDRESSES, e.g. "BTC/USD".
        contract_address: Override the contract address (required for feeds
                          not in FEED_ADDRESSES).
        decimals: Number of decimal places to divide the raw integer answer by.
                  USD feeds use 8; some feeds use 18.
        n_rounds: Maximum number of historical rounds to fetch.
        rpc_url: Ethereum JSON-RPC endpoint URL.

    Returns:
        DataFrame with columns: timestamp (UTC), round_id, price.
    """
    address = contract_address or FEED_ADDRESSES.get(feed)
    if not address:
        raise ValueError(
            f"Unknown feed '{feed}'. Known feeds: {sorted(FEED_ADDRESSES)}. "
            "Pass contract_address= for custom feeds."
        )

    # 1. Fetch the latest round to get the current round ID
    latest_hex = _eth_call(rpc_url, address, _SEL_LATEST)
    if not latest_hex or latest_hex == "0x":
        raise RuntimeError(f"Empty response from latestRoundData() for {feed}")
    latest = _decode_round_data(latest_hex)
    latest_round_id = latest["round_id"]

    # 2. Iterate backwards through n_rounds historical rounds
    records: list[dict] = []
    for i in range(n_rounds):
        rid = latest_round_id - i
        if rid <= 0:
            break
        try:
            # Encode uint80 roundId as a 32-byte ABI word (left-padded)
            calldata = _SEL_GET_ROUND + rid.to_bytes(32, "big").hex()
            hex_result = _eth_call(rpc_url, address, calldata)
            if not hex_result or hex_result == "0x":
                continue  # round does not exist (phase boundary)
            rd = _decode_round_data(hex_result)
            if rd["updated_at"] == 0:
                continue  # invalid / empty round
            records.append({
                "timestamp": pd.Timestamp(rd["updated_at"], unit="s", tz="UTC"),
                "round_id": rd["round_id"],
                "price": rd["answer"] / (10 ** decimals),
            })
        except Exception:
            continue  # skip malformed or reverted rounds

    if not records:
        raise RuntimeError(
            f"No valid rounds returned for feed '{feed}' at {address}"
        )

    df = pd.DataFrame(records)
    df["round_id"] = df["round_id"].astype("int64")
    df["price"] = df["price"].astype(float)
    return df[["timestamp", "round_id", "price"]].sort_values("timestamp").reset_index(drop=True)
