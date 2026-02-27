"""
Connector: PaymentScan – Cumulative Crypto Card Volumes
=======================================================
Series: cumulative_crypto_payment_card_volume_usd  (daily)
Source:  https://www.paymentscan.xyz/

Three-tier extraction strategy:
  1. Embedded JSON (SSR / __NEXT_DATA__)  — cheapest, no browser
  2. Playwright network interception       — reliable for CSR
  3. DOM headline fallback                 — single current value only

Requires: httpx, (optionally) playwright
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

URL = "https://www.paymentscan.xyz/"

# ---------------------------------------------------------------------------
# Value parsing: "$973.3M" → 973_300_000.0
# ---------------------------------------------------------------------------

_SUFFIX_MULTIPLIERS = {"T": 1e12, "B": 1e9, "M": 1e6, "K": 1e3}


def _parse_dollar_value(raw: str) -> Optional[float]:
    """Parse strings like '$973.3M', '$1.5B', '1,234,567'."""
    s = raw.strip().lstrip("$").replace(",", "").strip()
    if not s:
        return None
    suffix = s[-1].upper()
    if suffix in _SUFFIX_MULTIPLIERS:
        try:
            return float(s[:-1]) * _SUFFIX_MULTIPLIERS[suffix]
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Tier 1: Embedded JSON extraction (no browser)
# ---------------------------------------------------------------------------

_NEXT_DATA_RE = re.compile(
    r'<script\s+id="__NEXT_DATA__"\s+type="application/json">\s*(\{.*?\})\s*</script>',
    re.DOTALL,
)

# Broader catch: any large inline JSON blob in a <script> tag
_INLINE_JSON_RE = re.compile(
    r"<script[^>]*>\s*(\{.{500,}?\})\s*</script>",
    re.DOTALL,
)


def _search_json_for_series(obj: Any, depth: int = 0) -> Optional[list[dict]]:
    """
    Recursively search a parsed JSON tree for an array that looks like
    chart data: list of dicts/lists with date-like + numeric fields.
    """
    if depth > 12:
        return None

    if isinstance(obj, list) and len(obj) > 10:
        # Check if this looks like a time-series array
        sample = obj[:3]
        if all(isinstance(item, dict) for item in sample):
            keys_lower = {k.lower() for item in sample for k in item}
            has_date = any(
                k in keys_lower
                for k in ("date", "day", "timestamp", "time", "x", "created_at")
            )
            has_value = any(
                k in keys_lower
                for k in ("total", "value", "volume", "y", "amount", "cumulative")
            )
            if has_date and has_value:
                return obj

    if isinstance(obj, dict):
        for v in obj.values():
            result = _search_json_for_series(v, depth + 1)
            if result is not None:
                return result
    elif isinstance(obj, list):
        for item in obj:
            result = _search_json_for_series(item, depth + 1)
            if result is not None:
                return result

    return None


def _series_to_dataframe(series: list[dict]) -> pd.DataFrame:
    """Convert a list-of-dicts chart payload into a clean DataFrame."""
    df = pd.DataFrame(series)

    # Identify the date column
    date_col = None
    for candidate in ("date", "day", "timestamp", "time", "x", "created_at"):
        matches = [c for c in df.columns if c.lower() == candidate]
        if matches:
            date_col = matches[0]
            break
    if date_col is None:
        raise ValueError(f"No date column found in series columns: {list(df.columns)}")

    # Identify the value column
    value_col = None
    for candidate in ("total", "cumulative", "volume", "value", "y", "amount"):
        matches = [c for c in df.columns if c.lower() == candidate]
        if matches:
            value_col = matches[0]
            break
    if value_col is None:
        raise ValueError(f"No value column found in series columns: {list(df.columns)}")

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], utc=True).dt.date
    raw_values = df[value_col]
    if raw_values.dtype == object:
        out["cumulative_crypto_payment_card_volume_usd"] = raw_values.apply(
            lambda v: _parse_dollar_value(str(v)) if pd.notna(v) else None
        )
    else:
        out["cumulative_crypto_payment_card_volume_usd"] = raw_values.astype(float)

    return out.sort_values("date").reset_index(drop=True)


async def _try_embedded_json(html: str) -> Optional[pd.DataFrame]:
    """Tier 1: extract chart data from inline JSON in the HTML."""
    for regex in (_NEXT_DATA_RE, _INLINE_JSON_RE):
        for match in regex.finditer(html):
            try:
                blob = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            series = _search_json_for_series(blob)
            if series is not None:
                logger.info("Tier 1 hit: found %d-point series in embedded JSON", len(series))
                return _series_to_dataframe(series)
    return None


# ---------------------------------------------------------------------------
# Tier 2: Playwright network interception
# ---------------------------------------------------------------------------


async def _try_playwright_intercept() -> Optional[pd.DataFrame]:
    """
    Tier 2: load the page in a headless browser, intercept XHR responses,
    and capture the chart data array from the network.
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        logger.warning("playwright not installed; skipping Tier 2")
        return None

    captured_responses: list[dict] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        async def _on_response(response):
            content_type = response.headers.get("content-type", "")
            if "json" in content_type or response.url.endswith(".json"):
                try:
                    body = await response.json()
                    captured_responses.append({"url": response.url, "body": body})
                except Exception:
                    pass

        page.on("response", _on_response)

        try:
            await page.goto(URL, wait_until="networkidle", timeout=30_000)
        except Exception as e:
            logger.warning("Playwright navigation warning: %s", e)

        # Try to set filters if needed: look for Cumulative / All controls
        for label_pattern in ["Cumulative", "All"]:
            try:
                btn = page.locator(
                    f"button:has-text('{label_pattern}'), "
                    f"[role='option']:has-text('{label_pattern}'), "
                    f"label:has-text('{label_pattern}')"
                ).first
                if await btn.is_visible(timeout=2_000):
                    await btn.click()
                    await page.wait_for_load_state("networkidle", timeout=10_000)
            except Exception:
                pass

        await browser.close()

    # Search captured responses for chart data
    for resp in captured_responses:
        series = _search_json_for_series(resp["body"])
        if series is not None and len(series) > 10:
            logger.info(
                "Tier 2 hit: found %d-point series from %s",
                len(series),
                resp["url"],
            )
            return _series_to_dataframe(series)

    return None


# ---------------------------------------------------------------------------
# Tier 3: DOM headline fallback (single current value)
# ---------------------------------------------------------------------------

_DOLLAR_RE = re.compile(r"\$\s*([\d,.]+)\s*([MBKT])?", re.IGNORECASE)


async def _try_dom_headline(html: str) -> Optional[pd.DataFrame]:
    """
    Tier 3 fallback: extract the headline 'TOTAL SPENDS' value from
    the rendered HTML. Only yields a single data point (today's value).
    """
    # Search near "TOTAL SPENDS" text
    idx = html.upper().find("TOTAL SPENDS")
    if idx == -1:
        idx = html.upper().find("TOTAL SPEND")
    if idx == -1:
        return None

    # Look in a window around the label
    window = html[max(0, idx - 500) : idx + 500]
    match = _DOLLAR_RE.search(window)
    if not match:
        return None

    raw = match.group(0)
    value = _parse_dollar_value(raw)
    if value is None:
        return None

    logger.info("Tier 3 hit: headline value %s → %.0f", raw, value)
    return pd.DataFrame(
        [{"date": datetime.now(timezone.utc).date(), "cumulative_crypto_payment_card_volume_usd": value}]
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def fetch_paymentscan_cumulative_crypto_card_volume_daily(
    timeout_s: float = 60.0,
) -> pd.DataFrame:
    """
    Fetch the daily cumulative crypto payment card volume (USD) from PaymentScan.

    Returns a DataFrame with columns:
        date                                       (datetime.date)
        cumulative_crypto_payment_card_volume_usd   (float)
    """
    # Tier 1: try embedded JSON from a plain HTTP fetch
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout_s) as client:
        resp = await client.get(URL)
        resp.raise_for_status()
        html = resp.text

    result = await _try_embedded_json(html)
    if result is not None and len(result) > 1:
        return result

    # Tier 2: Playwright network interception
    result = await _try_playwright_intercept()
    if result is not None and len(result) > 1:
        return result

    # Tier 3: DOM headline fallback (single value)
    result = await _try_dom_headline(html)
    if result is not None:
        logger.warning(
            "Only Tier 3 (headline scrape) succeeded — returning a single data point, "
            "not a full time series. Consider debugging Tiers 1-2."
        )
        return result

    raise RuntimeError(
        "All extraction tiers failed for PaymentScan. "
        "The site structure may have changed — inspect manually."
    )
