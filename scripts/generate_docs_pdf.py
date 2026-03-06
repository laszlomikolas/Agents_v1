"""
generate_docs_pdf.py
Generates project documentation PDF with dependency graph.
Run from /home/user/Agents_v1:  python scripts/generate_docs_pdf.py
"""
from __future__ import annotations

import io
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer,
    Image, Table, TableStyle, HRFlowable, PageBreak, KeepTogether,
)
from reportlab.platypus.flowables import Flowable

W, H = A4

# ─── Colour palette ────────────────────────────────────────────────────────
DARK_BG   = colors.HexColor("#0f1117")
ACCENT    = colors.HexColor("#4f8ef7")
ACCENT2   = colors.HexColor("#a78bfa")
MID_GREY  = colors.HexColor("#1e2130")
LIGHT     = colors.HexColor("#e2e8f0")
MUTED     = colors.HexColor("#94a3b8")
GREEN     = colors.HexColor("#34d399")
AMBER     = colors.HexColor("#fbbf24")
RED_SOFT  = colors.HexColor("#f87171")
WHITE     = colors.white
BLACK     = colors.black

# ─── Styles ────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

def _style(name, parent="Normal", **kw):
    s = ParagraphStyle(name, parent=styles[parent], **kw)
    styles.add(s)
    return s

TITLE_S  = _style("MyTitle",  fontSize=28, textColor=WHITE,    leading=34,
                  fontName="Helvetica-Bold", alignment=TA_CENTER)
SUB_S    = _style("MySub",    fontSize=13, textColor=MUTED,    leading=18,
                  fontName="Helvetica",     alignment=TA_CENTER)
H1_S     = _style("MyH1",    fontSize=16, textColor=ACCENT,   leading=22,
                  fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=4)
H2_S     = _style("MyH2",    fontSize=12, textColor=ACCENT2,  leading=16,
                  fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=3)
BODY_S   = _style("MyBody",  fontSize=9,  textColor=LIGHT,    leading=14,
                  fontName="Helvetica",     alignment=TA_JUSTIFY)
CODE_S   = _style("MyCode",  fontSize=8,  textColor=GREEN,    leading=11,
                  fontName="Courier",       backColor=MID_GREY,
                  leftIndent=8, rightIndent=8, spaceBefore=2, spaceAfter=2)
MONO_S   = _style("MyMono",  fontSize=7.5, textColor=AMBER,   leading=10,
                  fontName="Courier-Bold")
CAPTION_S= _style("MyCaption", fontSize=8, textColor=MUTED,   leading=11,
                  fontName="Helvetica-Oblique", alignment=TA_CENTER, spaceAfter=8)
BULLET_S = _style("MyBullet", fontSize=9, textColor=LIGHT,    leading=13,
                  fontName="Helvetica",     leftIndent=16, spaceBefore=1)


# ─── Dark-page background flowable ─────────────────────────────────────────
class DarkRect(Flowable):
    def __init__(self, w, h, fill=DARK_BG):
        super().__init__()
        self.w, self.h, self.fill = w, h, fill

    def draw(self):
        self.canv.setFillColor(self.fill)
        self.canv.rect(0, 0, self.w, self.h, fill=1, stroke=0)


def _on_page(canvas, doc):
    """Draw dark background and subtle footer on every page."""
    canvas.saveState()
    canvas.setFillColor(DARK_BG)
    canvas.rect(0, 0, W, H, fill=1, stroke=0)
    # Footer rule
    canvas.setStrokeColor(MID_GREY)
    canvas.setLineWidth(0.5)
    canvas.line(2*cm, 1.4*cm, W - 2*cm, 1.4*cm)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(MUTED)
    canvas.drawCentredString(W/2, 1.0*cm, f"Agents_v1 · Project Documentation  |  Page {doc.page}")
    canvas.restoreState()


# ─── Dependency graph (matplotlib) ─────────────────────────────────────────
def _build_graph_image(width_pt: float, height_pt: float) -> Image:
    """Render the module dependency graph as a PNG embedded in the PDF."""

    dpi = 150
    fig_w = width_pt / 72
    fig_h = height_pt / 72
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")
    ax.set_aspect("equal")
    ax.axis("off")

    # Node positions  (x, y) in data coords  0..10 x 0..7
    nodes = {
        # External / entry
        "Polymarket\nGamma API":        (0.7,  6.2),
        "notebooks/\nscratch.ipynb":    (9.3,  6.2),
        "scripts/\ninventory_crypto":   (0.7,  4.7),

        # market_inventory
        "market_inventory/\ninventory": (2.8,  5.4),
        "GammaClient\n(polymarket_clients)": (1.5, 3.9),
        "CoinUniverse\nProjectUniverse\n(universe)": (2.8, 3.1),
        "route_resolution_terms\n(resolution_routing)": (4.5, 4.0),
        "parse_symbol / parse_metric\n(text_utils)": (2.8, 2.1),

        # parsing
        "HistoricalDataTriage\nDataSourcePlan\n(parsing/triage_models)": (5.2, 6.0),
        "ConnectorCode\n(parsing/connector_models)": (8.2, 5.1),
        "ConnectorType\n(parsing/connector_types)": (6.9, 3.7),

        # pipeline
        "triage_dataframe[_incremental]\n(pipeline/triage_runner)": (5.2, 4.7),
        "build_connectors\n(pipeline/connector_builder_runner)": (7.4, 6.1),

        # agents
        "triage_market_row\n(pm_agents/triage_agent)": (5.2, 3.4),
        "build_connector\n(pm_agents/connector_builder)": (7.4, 4.7),

        # connectors
        "connectors/\n__init__ + schema_validation": (8.6, 3.3),
        "fetch_*_ohlcv\n(binance/coinbase/kraken\nokx/bybit/bitstamp\ncoingecko)": (8.6, 2.1),
        "fetch_chainlink\n(connectors/chainlink)": (7.0, 2.1),
    }

    # Colour per layer
    layer_color = {
        "Polymarket\nGamma API":        "#1e3a5f",
        "notebooks/\nscratch.ipynb":    "#2d1f4e",
        "scripts/\ninventory_crypto":   "#1e3a5f",
        "market_inventory/\ninventory": "#14432a",
        "GammaClient\n(polymarket_clients)": "#14432a",
        "CoinUniverse\nProjectUniverse\n(universe)": "#14432a",
        "route_resolution_terms\n(resolution_routing)": "#14432a",
        "parse_symbol / parse_metric\n(text_utils)": "#14432a",
        "HistoricalDataTriage\nDataSourcePlan\n(parsing/triage_models)": "#3b2060",
        "ConnectorCode\n(parsing/connector_models)": "#3b2060",
        "ConnectorType\n(parsing/connector_types)": "#3b2060",
        "triage_dataframe[_incremental]\n(pipeline/triage_runner)": "#1a3a5c",
        "build_connectors\n(pipeline/connector_builder_runner)": "#1a3a5c",
        "triage_market_row\n(pm_agents/triage_agent)": "#4a1a2a",
        "build_connector\n(pm_agents/connector_builder)": "#4a1a2a",
        "connectors/\n__init__ + schema_validation": "#1a3820",
        "fetch_*_ohlcv\n(binance/coinbase/kraken\nokx/bybit/bitstamp\ncoingecko)": "#1a3820",
        "fetch_chainlink\n(connectors/chainlink)": "#1a3820",
    }
    border_color = {
        "Polymarket\nGamma API":        "#4f8ef7",
        "notebooks/\nscratch.ipynb":    "#a78bfa",
        "scripts/\ninventory_crypto":   "#4f8ef7",
        "market_inventory/\ninventory": "#34d399",
        "GammaClient\n(polymarket_clients)": "#34d399",
        "CoinUniverse\nProjectUniverse\n(universe)": "#34d399",
        "route_resolution_terms\n(resolution_routing)": "#34d399",
        "parse_symbol / parse_metric\n(text_utils)": "#34d399",
        "HistoricalDataTriage\nDataSourcePlan\n(parsing/triage_models)": "#a78bfa",
        "ConnectorCode\n(parsing/connector_models)": "#a78bfa",
        "ConnectorType\n(parsing/connector_types)": "#a78bfa",
        "triage_dataframe[_incremental]\n(pipeline/triage_runner)": "#4f8ef7",
        "build_connectors\n(pipeline/connector_builder_runner)": "#4f8ef7",
        "triage_market_row\n(pm_agents/triage_agent)": "#f87171",
        "build_connector\n(pm_agents/connector_builder)": "#f87171",
        "connectors/\n__init__ + schema_validation": "#34d399",
        "fetch_*_ohlcv\n(binance/coinbase/kraken\nokx/bybit/bitstamp\ncoingecko)": "#34d399",
        "fetch_chainlink\n(connectors/chainlink)": "#34d399",
    }

    edges = [
        # inventory stage
        ("Polymarket\nGamma API", "market_inventory/\ninventory"),
        ("scripts/\ninventory_crypto", "market_inventory/\ninventory"),
        ("market_inventory/\ninventory", "GammaClient\n(polymarket_clients)"),
        ("market_inventory/\ninventory", "CoinUniverse\nProjectUniverse\n(universe)"),
        ("market_inventory/\ninventory", "route_resolution_terms\n(resolution_routing)"),
        ("market_inventory/\ninventory", "parse_symbol / parse_metric\n(text_utils)"),
        ("CoinUniverse\nProjectUniverse\n(universe)", "parse_symbol / parse_metric\n(text_utils)"),
        # triage pipeline
        ("market_inventory/\ninventory", "triage_dataframe[_incremental]\n(pipeline/triage_runner)"),
        ("triage_dataframe[_incremental]\n(pipeline/triage_runner)", "triage_market_row\n(pm_agents/triage_agent)"),
        ("triage_dataframe[_incremental]\n(pipeline/triage_runner)", "HistoricalDataTriage\nDataSourcePlan\n(parsing/triage_models)"),
        ("triage_market_row\n(pm_agents/triage_agent)", "HistoricalDataTriage\nDataSourcePlan\n(parsing/triage_models)"),
        ("triage_market_row\n(pm_agents/triage_agent)", "ConnectorType\n(parsing/connector_types)"),
        # connector pipeline
        ("triage_dataframe[_incremental]\n(pipeline/triage_runner)", "build_connectors\n(pipeline/connector_builder_runner)"),
        ("HistoricalDataTriage\nDataSourcePlan\n(parsing/triage_models)", "build_connectors\n(pipeline/connector_builder_runner)"),
        ("build_connectors\n(pipeline/connector_builder_runner)", "build_connector\n(pm_agents/connector_builder)"),
        ("build_connector\n(pm_agents/connector_builder)", "ConnectorCode\n(parsing/connector_models)"),
        ("build_connector\n(pm_agents/connector_builder)", "ConnectorType\n(parsing/connector_types)"),
        ("ConnectorCode\n(parsing/connector_models)", "build_connectors\n(pipeline/connector_builder_runner)"),
        # connectors
        ("connectors/\n__init__ + schema_validation", "fetch_*_ohlcv\n(binance/coinbase/kraken\nokx/bybit/bitstamp\ncoingecko)"),
        ("connectors/\n__init__ + schema_validation", "fetch_chainlink\n(connectors/chainlink)"),
        # notebook
        ("notebooks/\nscratch.ipynb", "triage_dataframe[_incremental]\n(pipeline/triage_runner)"),
        ("notebooks/\nscratch.ipynb", "build_connectors\n(pipeline/connector_builder_runner)"),
        ("notebooks/\nscratch.ipynb", "connectors/\n__init__ + schema_validation"),
        ("notebooks/\nscratch.ipynb", "market_inventory/\ninventory"),
    ]

    xs = {n: p[0] for n, p in nodes.items()}
    ys = {n: p[1] for n, p in nodes.items()}

    # Draw edges first
    for src, dst in edges:
        x0, y0 = xs[src], ys[src]
        x1, y1 = xs[dst], ys[dst]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="#475569",
                                   lw=1.0, mutation_scale=10))

    # Draw nodes
    node_w, node_h = 1.55, 0.52
    for name, (x, y) in nodes.items():
        fc = layer_color.get(name, "#1e2130")
        ec = border_color.get(name, "#4f8ef7")
        rect = mpatches.FancyBboxPatch(
            (x - node_w/2, y - node_h/2), node_w, node_h,
            boxstyle="round,pad=0.04",
            facecolor=fc, edgecolor=ec, linewidth=1.2, zorder=3,
        )
        ax.add_patch(rect)
        lines = name.split("\n")
        fs = 5.2 if len(lines) >= 3 else 5.6
        ax.text(x, y, name, ha="center", va="center",
                fontsize=fs, color="white", zorder=4,
                multialignment="center",
                fontfamily="monospace")

    # Legend
    legend_items = [
        mpatches.Patch(facecolor="#14432a", edgecolor="#34d399", label="market_inventory"),
        mpatches.Patch(facecolor="#1a3a5c", edgecolor="#4f8ef7", label="pipeline"),
        mpatches.Patch(facecolor="#4a1a2a", edgecolor="#f87171", label="pm_agents (LLM)"),
        mpatches.Patch(facecolor="#3b2060", edgecolor="#a78bfa", label="parsing (models)"),
        mpatches.Patch(facecolor="#1a3820", edgecolor="#34d399", label="connectors"),
        mpatches.Patch(facecolor="#1e3a5f", edgecolor="#4f8ef7", label="entry / external"),
    ]
    ax.legend(handles=legend_items, loc="lower left", fontsize=5.5,
              facecolor="#1e2130", edgecolor="#334155", labelcolor="white",
              framealpha=0.9, ncol=3)

    ax.set_xlim(-0.2, 10.2)
    ax.set_ylim(1.3, 7.0)
    ax.set_title("Module Dependency Graph", color="#94a3b8", fontsize=7,
                 pad=4, fontfamily="monospace")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    img = Image(buf, width=width_pt, height=height_pt)
    return img


# ─── Table helpers ──────────────────────────────────────────────────────────
_TS_BASE = TableStyle([
    ("BACKGROUND",   (0, 0), (-1, 0),  MID_GREY),
    ("TEXTCOLOR",    (0, 0), (-1, 0),  ACCENT),
    ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
    ("FONTSIZE",     (0, 0), (-1, 0),  8),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [DARK_BG, colors.HexColor("#141720")]),
    ("TEXTCOLOR",    (0, 1), (-1, -1), LIGHT),
    ("FONTNAME",     (0, 1), (-1, -1), "Helvetica"),
    ("FONTSIZE",     (0, 1), (-1, -1), 7.5),
    ("GRID",         (0, 0), (-1, -1), 0.3, colors.HexColor("#1e2130")),
    ("LEFTPADDING",  (0, 0), (-1, -1), 5),
    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ("TOPPADDING",   (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
    ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
    ("ROWBACKGROUNDS", (0, 0), (0, 0), [MID_GREY]),
])

def _tbl(data, col_widths):
    t = Table(data, colWidths=col_widths)
    t.setStyle(_TS_BASE)
    return t

def _mono(txt): return Paragraph(txt, MONO_S)
def _body(txt): return Paragraph(txt, BODY_S)
def _bullet(txt): return Paragraph(f"• {txt}", BULLET_S)


# ─── Accent rule ────────────────────────────────────────────────────────────
def _rule(): return HRFlowable(width="100%", thickness=0.5,
                                color=ACCENT, spaceAfter=6, spaceBefore=2)


# ─── Build the document ─────────────────────────────────────────────────────
def build_pdf(out_path: str | Path):
    out_path = Path(out_path)
    margin = 2.0 * cm
    doc = BaseDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=margin, rightMargin=margin,
        topMargin=2.2*cm, bottomMargin=2.0*cm,
    )
    content_w = W - 2 * margin

    frame = Frame(margin, 2.0*cm, content_w, H - 4.2*cm, id="body")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=_on_page)])

    story = []

    # ── Cover ────────────────────────────────────────────────────────────────
    story += [
        Spacer(1, 3.5*cm),
        Paragraph("Agents_v1", TITLE_S),
        Spacer(1, 0.4*cm),
        Paragraph("Agentic Data Pipeline for Polymarket Crypto Markets", SUB_S),
        Spacer(1, 0.25*cm),
        Paragraph("Project Documentation  ·  March 2026", _style("CoverDate",
            fontSize=9, textColor=MUTED, fontName="Helvetica", alignment=TA_CENTER)),
        Spacer(1, 1.2*cm),
        _rule(),
        Spacer(1, 0.5*cm),
    ]

    # ── Overview ─────────────────────────────────────────────────────────────
    story.append(Paragraph("Overview", H1_S))
    story.append(_rule())
    story.append(_body(
        "Agents_v1 is a research scaffold that automates the sourcing of historical "
        "price and metric data for active crypto prediction markets on Polymarket. "
        "It follows a three-stage pipeline:"
    ))
    story.append(Spacer(1, 0.2*cm))
    for item in [
        "<b>Stage 1 – Market Inventory:</b> Queries the Polymarket Gamma API, filters "
        "for crypto edge/range markets, and normalises metadata (symbol, metric, "
        "resolution source, resolution terms, data type, interval).",
        "<b>Stage 2 – LLM Triage:</b> An OpenAI agent (gpt-5.4) assesses each market's "
        "historical-data relevance, feasibility, and paywall risk, and proposes concrete "
        "DataSourcePlan objects describing where and how to obtain the data.",
        "<b>Stage 3 – Connector Building:</b> A second LLM agent generates working "
        "Python connector functions from each DataSourcePlan. Results are deduplicated "
        "by connector_key and persisted in a JSON registry.",
    ]:
        story.append(_bullet(item))
        story.append(Spacer(1, 0.1*cm))

    story.append(Spacer(1, 0.3*cm))
    story.append(_body(
        "A triage cache (parquet) records all triage outputs keyed by market question. "
        "Subsequent pipeline runs diff the live inventory against the cache and re-triage "
        "only new or changed rows, making repeated runs cheap."
    ))
    story.append(Spacer(1, 0.5*cm))

    # ── Dependency graph ─────────────────────────────────────────────────────
    story.append(Paragraph("Module Dependency Graph", H1_S))
    story.append(_rule())
    graph_h = 13.5 * cm
    story.append(_build_graph_image(content_w, graph_h))
    story.append(Paragraph(
        "Arrows indicate import / call dependencies.  Colour bands represent logical layers.",
        CAPTION_S,
    ))

    story.append(PageBreak())

    # ── Package: market_inventory ─────────────────────────────────────────────
    story.append(Paragraph("Package: market_inventory", H1_S))
    story.append(_rule())
    story.append(_body(
        "Responsible for discovering and normalising active crypto markets from Polymarket. "
        "Outputs a DataFrame that feeds the triage pipeline."
    ))
    story.append(Spacer(1, 0.25*cm))

    story.append(Paragraph("Key Classes", H2_S))
    cls_data = [
        ["Class", "Module", "Description"],
        [_mono("GammaClient"), _mono("polymarket_clients"),
         _body("HTTP wrapper for the Polymarket Gamma REST API (events, tags).")],
        [_mono("ClobClient"), _mono("polymarket_clients"),
         _body("HTTP wrapper for the Polymarket CLOB API (midpoint prices).")],
        [_mono("CoinUniverse"), _mono("universe"),
         _body("Frozen dataclass: coin symbol + name-to-symbol lookup loaded from coins_universe.json.")],
        [_mono("ProjectUniverse"), _mono("universe"),
         _body("Frozen dataclass: project key/label lookup with fuzzy phrase matching.")],
        [_mono("ResolutionRouting"), _mono("resolution_routing"),
         _body("Frozen dataclass carrying (data_type, interval, interval_source, notes) routing decision.")],
    ]
    story.append(_tbl(cls_data, [3.5*cm, 4.5*cm, content_w - 8.0*cm]))
    story.append(Spacer(1, 0.25*cm))

    story.append(Paragraph("Key Functions", H2_S))
    fn_data = [
        ["Function", "Module", "Description"],
        [_mono("inventory_crypto_markets()"), _mono("inventory"),
         _body("Main entry point. Pages through Gamma API events, filters edge/range markets, extracts symbol, metric, resolution source/terms/date, routing. Returns DataFrame.")],
        [_mono("classify_edge_or_range()"), _mono("inventory"),
         _body("Classifies a market question + outcomes as 'edge', 'range', or 'unknown' using outcome-text heuristics.")],
        [_mono("extract_resolution_source_and_terms()"), _mono("inventory"),
         _body("Extracts resolution URL/provider and full resolution terms from market/event JSON.")],
        [_mono("route_resolution_terms()"), _mono("resolution_routing"),
         _body("Deterministic regex router: maps resolution terms → ResolutionDataType + interval.")],
        [_mono("parse_underlying_symbol()"), _mono("text_utils"),
         _body("Extracts the underlying asset symbol (e.g. BTC, ETH) from the market question.")],
        [_mono("parse_metric()"), _mono("text_utils"),
         _body("Extracts the metric being measured: price, fdv, market_cap, dominance, tvl, etc.")],
    ]
    story.append(_tbl(fn_data, [4.8*cm, 4.0*cm, content_w - 8.8*cm]))
    story.append(Spacer(1, 0.5*cm))

    # ── Package: parsing ─────────────────────────────────────────────────────
    story.append(Paragraph("Package: parsing", H1_S))
    story.append(_rule())
    story.append(_body(
        "Pydantic models and enumerations that define the data contracts between "
        "the triage agent, the connector builder agent, and the pipeline runners."
    ))
    story.append(Spacer(1, 0.25*cm))

    story.append(Paragraph("Key Classes", H2_S))
    pcls_data = [
        ["Class", "Module", "Description"],
        [_mono("ConnectorType"), _mono("connector_types"),
         _body("String enum: FREE_API_GENERIC, WAYBACK_SNAPSHOTS, PAYWALLED_PROVIDER, and 12 others.")],
        [_mono("DataCandidate"), _mono("historical_data_triage_models"),
         _body("One candidate historical series: name, unit, frequency, proxy_ok, proxy_notes.")],
        [_mono("DataSourcePlan"), _mono("historical_data_triage_models"),
         _body("Concrete acquisition plan: connector_type, connector_key, series_id, method, target, url_or_endpoint_hint, access, effort, reliability, extraction_target, extraction_method_detail, output_columns, connector_function_name, …")],
        [_mono("HistoricalDataTriage"), _mono("historical_data_triage_models"),
         _body("Full triage output for one market: historical_relevance, data_feasibility, paywall_risk, candidates, plans, recommended_resolution, routing_notes.")],
        [_mono("ConnectorCode"), _mono("connector_models"),
         _body("Generated connector artefact: source_code, imports, dependencies, output_columns, connector_key, series_id, notes.")],
    ]
    story.append(_tbl(pcls_data, [4.0*cm, 5.0*cm, content_w - 9.0*cm]))
    story.append(Spacer(1, 0.5*cm))

    # ── Package: pm_agents ───────────────────────────────────────────────────
    story.append(Paragraph("Package: pm_agents", H1_S))
    story.append(_rule())
    story.append(_body(
        "Two LLM-backed agents built on the OpenAI Agents SDK, both using model "
        "<b>gpt-5.4</b>. They produce structured Pydantic outputs enforced via strict "
        "JSON schema."
    ))
    story.append(Spacer(1, 0.25*cm))

    story.append(Paragraph("historical_data_triage_agent", H2_S))
    story.append(_body(
        "Given a market row dict (question, symbol, metric, resolution_source, resolution_terms, "
        "resolution_data_type, resolution_interval, …), the agent produces a <i>HistoricalDataTriage</i> "
        "with relevance/feasibility/paywall judgements and a list of DataSourcePlans."
    ))
    story.append(Spacer(1, 0.1*cm))
    ag1_data = [
        ["Symbol", "Type", "Purpose"],
        [_mono("historical_data_triage_agent"), _mono("Agent"),
         _body("Agent instance. Model: gpt-5.4. Output type: HistoricalDataTriage.")],
        [_mono("triage_market_row(row, timeout_s)"), _mono("async fn"),
         _body("Calls the agent for a single row dict. Returns HistoricalDataTriage.")],
    ]
    story.append(_tbl(ag1_data, [5.5*cm, 2.8*cm, content_w - 8.3*cm]))
    story.append(Spacer(1, 0.2*cm))

    story.append(Paragraph("connector_builder_agent", H2_S))
    story.append(_body(
        "Given a DataSourcePlan, the agent generates a self-contained Python connector "
        "function with import statements, source code, and dependency list."
    ))
    story.append(Spacer(1, 0.1*cm))
    ag2_data = [
        ["Symbol", "Type", "Purpose"],
        [_mono("connector_builder_agent"), _mono("Agent"),
         _body("Agent instance. Model: gpt-5.4. Output type: ConnectorCode.")],
        [_mono("build_connector(plan, timeout_s)"), _mono("async fn"),
         _body("Calls the agent for a single DataSourcePlan. Returns ConnectorCode.")],
    ]
    story.append(_tbl(ag2_data, [5.5*cm, 2.8*cm, content_w - 8.3*cm]))
    story.append(Spacer(1, 0.5*cm))

    # ── Package: pipeline ────────────────────────────────────────────────────
    story.append(Paragraph("Package: pipeline", H1_S))
    story.append(_rule())
    story.append(_body(
        "Orchestration runners that fan out agent calls with concurrency control, "
        "timeouts, error handling, and caching."
    ))
    story.append(Spacer(1, 0.25*cm))

    story.append(Paragraph("historical_triage_runner", H2_S))
    tr_data = [
        ["Function / Variable", "Description"],
        [_mono("DIFF_COLUMNS"),
         _body("List of 12 columns used to detect whether an inventory row changed vs cache.")],
        [_mono("DEFAULT_CACHE_PATH"),
         _body("Default parquet path for the triage cache ('triage_cache.parquet').")],
        [_mono("save_triage_cache(df, path)"),
         _body("Persist a triaged DataFrame to parquet.")],
        [_mono("load_triage_cache(path)"),
         _body("Load the triage cache parquet; returns None if absent.")],
        [_mono("diff_inventory_vs_cache(inventory_df, cache_df)"),
         _body("Compare live inventory vs cache. Returns only new or changed rows.")],
        [_mono("triage_dataframe_async(df, ...)"),
         _body("Fan-out async triage: one agent call per row, semaphore-bounded, optional total timeout.")],
        [_mono("triage_dataframe_incremental_async(inventory_df, ...)"),
         _body("Load cache → diff → triage delta → merge → save cache → return merged DataFrame.")],
        [_mono("triage_dataframe[_incremental](...)"),
         _body("Sync wrappers via asyncio.run().")],
    ]
    story.append(_tbl(tr_data, [6.5*cm, content_w - 6.5*cm]))
    story.append(Spacer(1, 0.2*cm))

    story.append(Paragraph("connector_builder_runner", H2_S))
    cb_data = [
        ["Function", "Description"],
        [_mono("build_connectors_async(triaged_df, ...)"),
         _body("Extracts DataSourcePlans from triage_plans_json, deduplicates by connector_key, fans out build_connector calls.")],
        [_mono("build_connectors(...)"),
         _body("Sync wrapper.")],
        [_mono("save_registry(connectors, path)"),
         _body("Persist Dict[connector_key → ConnectorCode] to JSON.")],
        [_mono("load_registry(path)"),
         _body("Load connector registry from JSON.")],
        [_mono("write_connectors_module(connectors, path)"),
         _body("Write all generated connectors to a single .py module with deduplicated imports.")],
    ]
    story.append(_tbl(cb_data, [6.5*cm, content_w - 6.5*cm]))
    story.append(Spacer(1, 0.5*cm))

    # ── Package: connectors ──────────────────────────────────────────────────
    story.append(Paragraph("Package: connectors", H1_S))
    story.append(_rule())
    story.append(_body(
        "Hand-built, schema-validated OHLCV connectors for the seven standard Polymarket "
        "resolution sources, plus an on-chain Chainlink connector. Each connector returns "
        "a DataFrame with columns <b>timestamp, open, high, low, close, volume</b>."
    ))
    story.append(Spacer(1, 0.25*cm))

    story.append(Paragraph("OHLCV Connectors", H2_S))
    oc_data = [
        ["Function", "Source", "Endpoint"],
        [_mono("fetch_coingecko_ohlcv()"), "CoinGecko", "/coins/{id}/ohlc + /market_chart"],
        [_mono("fetch_binance_ohlcv()"), "Binance", "/api/v3/klines"],
        [_mono("fetch_coinbase_ohlcv()"), "Coinbase Exchange", "/products/{id}/candles"],
        [_mono("fetch_kraken_ohlcv()"), "Kraken", "/0/public/OHLC"],
        [_mono("fetch_bitstamp_ohlcv()"), "Bitstamp", "/api/v2/ohlc"],
        [_mono("fetch_okx_ohlcv()"), "OKX", "/api/v5/market/candles"],
        [_mono("fetch_bybit_ohlcv()"), "Bybit v5", "/v5/market/kline"],
    ]
    story.append(_tbl(oc_data, [5.5*cm, 3.5*cm, content_w - 9.0*cm]))
    story.append(Spacer(1, 0.2*cm))

    story.append(Paragraph("Schema Validation", H2_S))
    sv_data = [
        ["Class / Function", "Description"],
        [_mono("ColumnSpec"), _body("Describes one expected column: name, dtype_kind, nullable, min/max value.")],
        [_mono("SchemaSpec"), _body("Full connector schema: expected columns, min_rows, probe_kwargs, OHLC sanity flags.")],
        [_mono("ValidationResult"), _body("Outcome: ok flag, errors list, warnings list, actual columns/dtypes, row_count. summary() → str.")],
        [_mono("validate_schema(df, spec)"), _body("Validates a DataFrame against a SchemaSpec.")],
        [_mono("probe_connector(spec, fn, **kwargs)"), _body("Calls the connector function and validates its output in one step.")],
        [_mono("SCHEMA_REGISTRY"), _body("Dict mapping connector name → (SchemaSpec, fetch_fn). Used by check_all_connectors().")],
        [_mono("check_all_connectors()"), _body("Probes every registered connector and returns Dict[name, ValidationResult].")],
    ]
    story.append(_tbl(sv_data, [5.5*cm, content_w - 5.5*cm]))
    story.append(Spacer(1, 0.5*cm))

    # ── Data flow summary ────────────────────────────────────────────────────
    story.append(Paragraph("End-to-End Data Flow", H1_S))
    story.append(_rule())

    flow_data = [
        ["Step", "Function / Class", "Output"],
        ["1", _mono("inventory_crypto_markets()"), _body("DataFrame: market, kind, symbol, metric, resolution_date, resolution_source, resolution_terms, resolution_data_type, resolution_interval, routing_notes, …")],
        ["2", _mono("diff_inventory_vs_cache()"), _body("Subset of inventory rows that are new or changed vs parquet cache.")],
        ["3", _mono("triage_dataframe_incremental_async()"), _body("DataFrame with triage columns appended: historical_relevance, data_feasibility, paywall_risk, triage_plans_json, triage_candidates_json, triage_routing_notes, triage_error.")],
        ["4", _mono("build_connectors_async()"), _body("Dict[connector_key → ConnectorCode] with generated Python source, imports, and dependencies.")],
        ["5", _mono("write_connectors_module()"), _body("Single .py module containing all generated connector functions, ready to import.")],
    ]
    story.append(_tbl(flow_data, [1.0*cm, 5.5*cm, content_w - 6.5*cm]))
    story.append(Spacer(1, 0.4*cm))
    story.append(_body(
        "The triage cache (parquet) and connector registry (JSON) provide persistence "
        "between runs. On subsequent invocations only the delta is sent to the LLM, "
        "keeping API costs proportional to the rate of market change rather than to the "
        "total number of active markets."
    ))

    doc.build(story)
    print(f"PDF written → {out_path}")


if __name__ == "__main__":
    build_pdf(Path(__file__).parent.parent / "Agents_v1_documentation.pdf")
