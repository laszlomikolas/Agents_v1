from __future__ import annotations
from enum import Enum


class ConnectorType(str, Enum):
    # ---- Free API / structured (non-OHLC) ----
    FREE_API_GENERIC = "free_api_generic"          # generic REST/JSON, unknown schema
    OFFICIAL_STATS_API = "official_stats_api"      # e.g. World Bank / Eurostat / FRED-like
    BLOCKCHAIN_EXPLORER_API = "blockchain_explorer_api"  # etherscan-like, or chain RPC wrappers
    DEFI_PROTOCOL_API = "defi_protocol_api"        # protocol dashboards / DefiLlama-style endpoints
    SOCIAL_PLATFORM_API = "social_platform_api"    # e.g. YouTube/Twitter-like official APIs (if relevant)

    # ---- Unstructured / semi-structured ----
    GENERIC_HTML_TABLE_SCRAPE = "generic_html_table_scrape"
    GENERIC_WEB_SCRAPE = "generic_web_scrape"
    GENERIC_JSON_ENDPOINT = "generic_json_endpoint"  # JSON embedded / undocumented endpoint
    WAYBACK_SNAPSHOTS = "wayback_snapshots"
    PDF_TABLE_EXTRACT = "pdf_table_extract"
    CSV_DOWNLOAD = "csv_download"
    GITHUB_RAW = "github_raw"
    GOOGLE_SHEETS = "google_sheets"
    WIKIPEDIA_WIKIDATA = "wikipedia_wikidata"
    OFFICIAL_STATS_PORTAL = "official_stats_portal"

    PAYWALLED_PROVIDER = "paywalled_provider"
    UNKNOWN = "unknown"
