"""Chainlink on-chain price feed connector via Ethereum JSON-RPC (no web3 dependency)."""
from __future__ import annotations

import logging
import requests
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)

# Ordered list of public Ethereum mainnet RPC endpoints (no API key required).
# The connector tries each in sequence and uses the first one that returns a
# valid non-empty response for latestRoundData().
_PUBLIC_RPCS: list[str] = [
    "https://cloudflare-eth.com",
    "https://eth.llamarpc.com",
    "https://rpc.ankr.com/eth",
    "https://ethereum.publicnode.com",
    "https://1rpc.io/eth",
]

_HEADERS = {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}

# ABI function selectors (keccak256 of canonical signature, first 4 bytes)
_SEL_LATEST    = "0xfeaf968c"   # latestRoundData()
_SEL_GET_ROUND = "0x9a6fc8f5"  # getRoundData(uint80)

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


def _eth_call(rpc_url: str, contract: str, calldata: str) -> Optional[str]:
    """
    Issue a single eth_call JSON-RPC request.

    Returns the hex result string, or None if the endpoint returned an empty /
    null result (which some public nodes do under rate-limiting or when the
    call is unsupported). Raises on network errors or explicit JSON-RPC errors.
    """
    payload = {
        "jsonrpc": "2.0",
        "method":  "eth_call",
        "params":  [{"to": contract, "data": calldata}, "latest"],
        "id":      1,
    }
    resp = requests.post(rpc_url, json=payload, headers=_HEADERS, timeout=15)
    resp.raise_for_status()
    body = resp.json()
    if "error" in body:
        raise RuntimeError(f"JSON-RPC error: {body['error']}")
    result = body.get("result")
    if not result or result == "0x":
        return None
    return result


def _eth_call_with_fallback(
    contract: str,
    calldata: str,
    rpc_url: Optional[str],
) -> tuple[str, str]:
    """
    Try *rpc_url* first (if provided), then each endpoint in _PUBLIC_RPCS,
    and return *(hex_result, working_rpc_url)* for the first endpoint that
    returns a valid non-empty result.

    Raises RuntimeError if every endpoint fails.
    """
    candidates = ([rpc_url] if rpc_url else []) + [
        r for r in _PUBLIC_RPCS if r != rpc_url
    ]
    last_error: str = "no endpoints tried"
    for url in candidates:
        try:
            result = _eth_call(url, contract, calldata)
            if result:
                logger.debug("eth_call succeeded via %s", url)
                return result, url
            logger.debug("eth_call returned empty from %s, trying next", url)
            last_error = f"{url} returned empty result"
        except Exception as exc:
            logger.debug("eth_call failed for %s: %s", url, exc)
            last_error = f"{url}: {exc}"
    raise RuntimeError(
        f"All Ethereum RPC endpoints failed for contract {contract}. "
        f"Last error: {last_error}. "
        "Pass rpc_url= with a working endpoint (e.g. Infura, Alchemy, or your own node)."
    )


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
    rpc_url: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch recent price history from a Chainlink on-chain price feed.

    Iterates backwards from the latest round ID, calling getRoundData() for
    each, and returns up to n_rounds valid data points. Rounds with missing
    or zero updatedAt are silently skipped (they occur at phase transitions).

    No API key is required. Automatically tries several public Ethereum RPC
    endpoints in sequence (_PUBLIC_RPCS) until one responds successfully.
    Supply rpc_url= to use a specific endpoint (e.g. Infura/Alchemy) and skip
    the auto-fallback.

    Args:
        feed: Feed name key in FEED_ADDRESSES, e.g. "BTC/USD".
        contract_address: Override the contract address (required for feeds
                          not in FEED_ADDRESSES).
        decimals: Decimal places to divide the raw integer answer by.
                  USD feeds use 8; some feeds use 18.
        n_rounds: Maximum number of historical rounds to fetch.
        rpc_url: Specific Ethereum JSON-RPC endpoint to use. When None, tries
                 _PUBLIC_RPCS in order and picks the first that works.

    Returns:
        DataFrame with columns: timestamp (UTC), round_id, price.
    """
    address = contract_address or FEED_ADDRESSES.get(feed)
    if not address:
        raise ValueError(
            f"Unknown feed '{feed}'. Known feeds: {sorted(FEED_ADDRESSES)}. "
            "Pass contract_address= for custom feeds."
        )

    # 1. Find a working RPC and fetch the latest round
    latest_hex, working_rpc = _eth_call_with_fallback(address, _SEL_LATEST, rpc_url)
    latest = _decode_round_data(latest_hex)
    latest_round_id = latest["round_id"]
    logger.debug("latestRoundData: round_id=%d via %s", latest_round_id, working_rpc)

    # 2. Iterate backwards through n_rounds historical rounds using the same RPC
    records: list[dict] = []
    for i in range(n_rounds):
        rid = latest_round_id - i
        if rid <= 0:
            break
        try:
            calldata = _SEL_GET_ROUND + rid.to_bytes(32, "big").hex()
            hex_result = _eth_call(working_rpc, address, calldata)
            if not hex_result:
                continue  # round does not exist (phase boundary or rate limit)
            rd = _decode_round_data(hex_result)
            if rd["updated_at"] == 0:
                continue  # invalid / empty round
            records.append({
                "timestamp": pd.Timestamp(rd["updated_at"], unit="s", tz="UTC"),
                "round_id":  rd["round_id"],
                "price":     rd["answer"] / (10 ** decimals),
            })
        except Exception:
            continue  # skip malformed or reverted rounds

    if not records:
        raise RuntimeError(
            f"No valid rounds returned for feed '{feed}' at {address} via {working_rpc}"
        )

    df = pd.DataFrame(records)
    df["round_id"] = df["round_id"].astype("int64")
    df["price"]    = df["price"].astype(float)
    return df[["timestamp", "round_id", "price"]].sort_values("timestamp").reset_index(drop=True)
