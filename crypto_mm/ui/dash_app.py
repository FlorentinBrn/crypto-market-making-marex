"""Dashboard web Dash.

Serveur Flask/Dash standalone qui lit périodiquement les CSV produits par
`CoinbaseMarketDataApp` (`data/` par défaut) et affiche une vue live :

- équity / P&L au cours du temps
- position et exposure
- spread inside + signaux microstructure
- dernier état du carnet
- fills récents
- cartes de risque (loss util, health score, reduce-only)

Conception clé : **aucune dépendance partagée avec le feed**. Le
dashboard est un process séparé qui relit les CSV — on peut le lancer
pendant un run live, après un run, ou sur les outputs d'un replay sans
changement de code.

Usage :
```bash
# Run live dans un terminal
python -m crypto_mm.main

# Dashboard dans un autre terminal
python -m crypto_mm.ui.dash_app --data-dir data --port 8050
```

Par défaut, le dashboard se rafraîchit toutes les 1000 ms (configurable
via --refresh-ms). Pour un run terminé, on peut augmenter à 60000 ms.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from dash import Dash, Input, Output, dcc, html
    import plotly.graph_objects as go
except ImportError as e:
    raise SystemExit(
        "Dash et plotly sont requis pour le dashboard web. "
        "Installe-les avec : pip install dash plotly"
    ) from e


# ---------------------------------------------------------------------
# Lecture CSV — tolérante aux fichiers vides / manquants
# ---------------------------------------------------------------------
def _read_csv_safe(path: Path, tail: int | None = None) -> pd.DataFrame:
    """Lit un CSV s'il existe, retourne un df vide sinon.

    `tail` permet de ne garder que les N dernières lignes (utile quand le
    fichier grossit en live ; évite de re-traiter tout l'historique à
    chaque tick).
    """
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if tail is not None and len(df) > tail:
        df = df.tail(tail).reset_index(drop=True)
    return df


def _parse_ts(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Parse une colonne de timestamp en datetime UTC (format ISO8601)."""
    if df.empty or col not in df.columns:
        return df
    df = df.copy()
    df[col] = pd.to_datetime(df[col], utc=True, errors="coerce", format="ISO8601")
    df = df.dropna(subset=[col])
    return df


# ---------------------------------------------------------------------
# Layout — composants
# ---------------------------------------------------------------------
CARD_STYLE = {
    "background": "rgba(255,255,255,0.04)",
    "border": "1px solid rgba(255,255,255,0.08)",
    "borderRadius": "12px",
    "padding": "16px 18px",
    "minHeight": "80px",
}

PAGE_STYLE = {
    "background": "radial-gradient(circle at top, #12213f 0%, #0b1020 42%, #09111c 100%)",
    "color": "#e6e6ef",
    "minHeight": "100vh",
    "padding": "20px 28px",
    "fontFamily": "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
}


def _metric_card(label: str, value: str, sub: str = "") -> html.Div:
    """Carte de métrique (label + grosse valeur + sous-texte)."""
    return html.Div(
        style=CARD_STYLE,
        children=[
            html.Div(label, style={"fontSize": "0.8rem", "opacity": 0.7}),
            html.Div(
                value,
                style={"fontSize": "1.5rem", "fontWeight": 700, "marginTop": "4px"},
            ),
            html.Div(
                sub, style={"fontSize": "0.75rem", "opacity": 0.55, "marginTop": "2px"}
            ),
        ],
    )


def build_layout(data_dir: Path, refresh_ms: int) -> html.Div:
    """Construit le layout statique. Les composants dynamiques sont remplis
    par les callbacks."""
    return html.Div(
        style=PAGE_STYLE,
        children=[
            # Intervalle qui déclenche les rafraîchissements
            dcc.Interval(id="tick", interval=refresh_ms, n_intervals=0),
            dcc.Store(id="data-dir", data=str(data_dir)),
            # Header
            html.Div(
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "center",
                },
                children=[
                    html.H1(
                        "Crypto MM — Dashboard",
                        style={
                            "margin": 0,
                            "fontSize": "1.6rem",
                            "letterSpacing": "0.3px",
                        },
                    ),
                    html.Div(
                        id="header-status",
                        style={"fontSize": "0.9rem", "opacity": 0.8},
                    ),
                ],
            ),
            html.Div(
                f"Source : {data_dir.resolve()}  ·  Rafraîchissement : {refresh_ms} ms",
                style={
                    "fontSize": "0.8rem",
                    "opacity": 0.5,
                    "marginTop": "4px",
                    "marginBottom": "20px",
                },
            ),
            # Ligne 1 : cartes métriques principales
            html.Div(
                id="metric-cards",
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(6, 1fr)",
                    "gap": "14px",
                    "marginBottom": "20px",
                },
            ),
            # Ligne 2 : P&L et position (deux graphes côte à côte)
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr 1fr",
                    "gap": "14px",
                    "marginBottom": "20px",
                },
                children=[
                    html.Div(dcc.Graph(id="equity-chart"), style=CARD_STYLE),
                    html.Div(dcc.Graph(id="position-chart"), style=CARD_STYLE),
                ],
            ),
            # Ligne 3 : microstructure + spreads
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr 1fr",
                    "gap": "14px",
                    "marginBottom": "20px",
                },
                children=[
                    html.Div(dcc.Graph(id="microstructure-chart"), style=CARD_STYLE),
                    html.Div(dcc.Graph(id="spread-chart"), style=CARD_STYLE),
                ],
            ),
            # Ligne 4 : carnet + fills récents
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr 1fr",
                    "gap": "14px",
                },
                children=[
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.H3("Carnet d'ordres (top 10)", style={"marginTop": 0}),
                            html.Div(id="book-table"),
                        ],
                    ),
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.H3("Fills récents", style={"marginTop": 0}),
                            html.Div(id="fills-table"),
                        ],
                    ),
                ],
            ),
        ],
    )


# ---------------------------------------------------------------------
# Construction des graphes (en charge utile des callbacks)
# ---------------------------------------------------------------------
_PLOTLY_DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=40, r=20, t=40, b=30),
    height=340,
)


def _empty_fig(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=title, **_PLOTLY_DARK_LAYOUT)
    return fig


def build_equity_figure(state_df: pd.DataFrame) -> go.Figure:
    """P&L réalisé et non-réalisé dans le temps (sans l'equity en valeur
    absolue : on regarde la performance, pas le niveau du capital)."""
    if state_df.empty:
        return _empty_fig("P&L — en attente de données")
    df = _parse_ts(state_df, "timestamp")
    if df.empty:
        return _empty_fig("P&L")
    fig = go.Figure()
    if "realized_pnl" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["realized_pnl"],
                name="Realized P&L",
                line=dict(color="#68d391"),
            )
        )
    if "unrealized_pnl" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["unrealized_pnl"],
                name="Unrealized P&L",
                line=dict(color="#f6ad55"),
            )
        )
    fig.update_layout(title="P&L (realized + unrealized)", **_PLOTLY_DARK_LAYOUT)
    fig.update_yaxes(title="USD")
    # Ligne zéro pour référence visuelle.
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)")
    return fig


def build_position_figure(state_df: pd.DataFrame) -> go.Figure:
    """Position BTC + mid price overlay."""
    if state_df.empty:
        return _empty_fig("Position / Mid — en attente de données")
    df = _parse_ts(state_df, "timestamp")
    if df.empty:
        return _empty_fig("Position / Mid")
    fig = go.Figure()
    if "position_btc" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["position_btc"],
                name="Position BTC",
                line=dict(color="#9f7aea"),
            )
        )
    if "mid_price" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["mid_price"],
                name="Mid price",
                yaxis="y2",
                line=dict(color="#63b3ed", dash="dot"),
            )
        )
    fig.update_layout(
        title="Position BTC & Mid price",
        yaxis=dict(title="BTC"),
        yaxis2=dict(title="USD", overlaying="y", side="right", showgrid=False),
        **_PLOTLY_DARK_LAYOUT,
    )
    return fig


def build_microstructure_figure(state_df: pd.DataFrame) -> go.Figure:
    """Micro signal, OFI EWMA, VPIN, fast vol."""
    if state_df.empty:
        return _empty_fig("Microstructure — en attente de données")
    df = _parse_ts(state_df, "timestamp")
    if df.empty:
        return _empty_fig("Microstructure")
    fig = go.Figure()
    colors = {
        "micro_signal_bps": "#f6ad55",
        "ofi_ewma": "#4fd1c5",
        "vpin": "#fc8181",
        "fast_vol_bps": "#b794f4",
    }
    for col, color in colors.items():
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"], y=df[col], name=col, line=dict(color=color)
                )
            )
    fig.update_layout(title="Signaux microstructure", **_PLOTLY_DARK_LAYOUT)
    return fig


def build_spread_figure(state_df: pd.DataFrame) -> go.Figure:
    """Half-spread bps vs inside spread bps."""
    if state_df.empty:
        return _empty_fig("Spread — en attente de données")
    df = _parse_ts(state_df, "timestamp")
    if df.empty:
        return _empty_fig("Spread")
    fig = go.Figure()
    for col, color, name in [
        ("half_spread_bps", "#4fd1c5", "Half-spread (bps)"),
        ("rolling_inside_spread_bps", "#a0aec0", "Inside spread mean (bps)"),
        ("ewma_inside_spread_bps", "#cbd5e0", "Inside spread EWMA (bps)"),
        ("toxicity_widening_bps", "#fc8181", "Toxicity widening"),
        ("fast_vol_widening_bps", "#f6ad55", "Fast vol widening"),
    ]:
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"], y=df[col], name=name, line=dict(color=color)
                )
            )
    fig.update_layout(title="Spread & widening", **_PLOTLY_DARK_LAYOUT)
    fig.update_yaxes(title="bps")
    return fig


def build_book_table(book_df: pd.DataFrame) -> html.Div:
    """Dernier snapshot du carnet : top 10 bids et asks."""
    if book_df.empty or "timestamp" not in book_df.columns:
        return html.Div("En attente de données...", style={"opacity": 0.6})
    # Dernier timestamp présent
    last_ts = book_df["timestamp"].iloc[-1]
    last = book_df[book_df["timestamp"] == last_ts]
    bids = last[last["side"] == "bid"].sort_values("rank").head(10)
    asks = last[last["side"] == "ask"].sort_values("rank").head(10)

    def make_rows(df: pd.DataFrame, color: str) -> list:
        rows = []
        for _, r in df.iterrows():
            rows.append(
                html.Tr(
                    [
                        html.Td(
                            f"{r['price']:,.2f}",
                            style={"color": color, "textAlign": "right"},
                        ),
                        html.Td(
                            f"{r['quantity']:.6f}",
                            style={"color": color, "textAlign": "right"},
                        ),
                    ]
                )
            )
        return rows

    return html.Table(
        style={"width": "100%", "borderCollapse": "collapse"},
        children=[
            html.Thead(
                html.Tr(
                    [
                        html.Th(
                            "BID Px", style={"textAlign": "right", "color": "#68d391"}
                        ),
                        html.Th(
                            "BID Qty", style={"textAlign": "right", "color": "#68d391"}
                        ),
                        html.Th(
                            "ASK Px", style={"textAlign": "right", "color": "#fc8181"}
                        ),
                        html.Th(
                            "ASK Qty", style={"textAlign": "right", "color": "#fc8181"}
                        ),
                    ]
                )
            ),
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td(
                                (
                                    f"{bids.iloc[i]['price']:,.2f}"
                                    if i < len(bids)
                                    else ""
                                ),
                                style={"textAlign": "right", "color": "#68d391"},
                            ),
                            html.Td(
                                (
                                    f"{bids.iloc[i]['quantity']:.6f}"
                                    if i < len(bids)
                                    else ""
                                ),
                                style={"textAlign": "right", "color": "#68d391"},
                            ),
                            html.Td(
                                (
                                    f"{asks.iloc[i]['price']:,.2f}"
                                    if i < len(asks)
                                    else ""
                                ),
                                style={"textAlign": "right", "color": "#fc8181"},
                            ),
                            html.Td(
                                (
                                    f"{asks.iloc[i]['quantity']:.6f}"
                                    if i < len(asks)
                                    else ""
                                ),
                                style={"textAlign": "right", "color": "#fc8181"},
                            ),
                        ]
                    )
                    for i in range(max(len(bids), len(asks)))
                ]
            ),
        ],
    )


def _format_fill_time(raw: object) -> str:
    """Formate une heure de fill en HH:MM:SS.mmm (UTC).

    Accepte soit une chaîne ISO8601 (`2026-04-20T13:42:17.309472+00:00`),
    soit un ``datetime`` (du mode live mémoire où `time` est un dt natif).
    Retourne une chaîne vide si le parsing échoue.
    """
    if raw is None or raw == "":
        return ""
    # Cas datetime natif (mode live mémoire)
    if hasattr(raw, "strftime"):
        try:
            return raw.strftime("%H:%M:%S.") + f"{raw.microsecond // 1000:03d}"
        except Exception:
            return str(raw)
    # Cas string ISO (mode offline CSV)
    ts = pd.to_datetime(raw, utc=True, errors="coerce", format="ISO8601")
    if pd.isna(ts):
        return str(raw)
    return ts.strftime("%H:%M:%S.") + f"{ts.microsecond // 1000:03d}"


def build_fills_table(fills_df: pd.DataFrame, n: int = 15) -> html.Div:
    """Derniers fills."""
    if fills_df.empty:
        return html.Div("Aucun fill", style={"opacity": 0.6})
    df = fills_df.tail(n).iloc[::-1]  # plus récent en haut
    return html.Table(
        style={"width": "100%", "borderCollapse": "collapse"},
        children=[
            html.Thead(
                html.Tr(
                    [
                        html.Th("Heure", style={"textAlign": "left"}),
                        html.Th("Side", style={"textAlign": "center"}),
                        html.Th("Prix", style={"textAlign": "right"}),
                        html.Th("Taille", style={"textAlign": "right"}),
                    ]
                )
            ),
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td(
                                _format_fill_time(r.get("time")),
                                style={
                                    "fontFamily": "monospace",
                                    "fontSize": "0.85rem",
                                },
                            ),
                            html.Td(
                                r.get("side", ""),
                                style={
                                    "textAlign": "center",
                                    "color": (
                                        "#68d391"
                                        if str(r.get("side", ""))
                                        .upper()
                                        .startswith("B")
                                        else "#fc8181"
                                    ),
                                    "fontWeight": 600,
                                },
                            ),
                            html.Td(
                                f"{float(r.get('price', 0)):,.2f}",
                                style={"textAlign": "right"},
                            ),
                            html.Td(
                                f"{float(r.get('size', 0)):.6f}",
                                style={"textAlign": "right"},
                            ),
                        ]
                    )
                    for _, r in df.iterrows()
                ]
            ),
        ],
    )


# ---------------------------------------------------------------------
# Extraction des métriques courantes pour les cartes
# ---------------------------------------------------------------------
def _last_row(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {}
    return df.iloc[-1].to_dict()


def _fmt_money(v: float | None) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"{v:+,.2f} $" if v != 0 else "0 $"


def _fmt_pct(v: float | None) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"{100 * v:.1f}%"


def _fmt_num(v: float | None, digits: int = 4) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"{v:,.{digits}f}"


def build_metric_cards(
    state_df: pd.DataFrame, fills_df: pd.DataFrame
) -> list[html.Div]:
    """6 cartes métriques principales."""
    last = _last_row(state_df)
    equity = last.get("equity")
    realized = last.get("realized_pnl")
    unreal = last.get("unrealized_pnl")
    position = last.get("position_btc")
    mid = last.get("mid_price")
    health = last.get("health_score")
    loss_util = last.get("loss_utilization")
    risk_level = last.get("risk_level", "—")

    cards = [
        _metric_card(
            "Equity",
            _fmt_num(equity, 2),
            f"{_fmt_money((equity or 0) - 1_000_000)} vs init",
        ),
        _metric_card("Realized P&L", _fmt_money(realized)),
        _metric_card("Unrealized P&L", _fmt_money(unreal)),
        _metric_card("Position BTC", _fmt_num(position, 6), f"mid {_fmt_num(mid, 2)}"),
        _metric_card("Health score", _fmt_num(health, 0), f"risk {risk_level}"),
        _metric_card(
            "Loss util.", _fmt_pct(loss_util), f"{len(fills_df)} fills au total"
        ),
    ]
    return cards


def build_header_status(state_df: pd.DataFrame) -> str:
    """Date/heure de la dernière ligne d'état + nombre d'obs."""
    if state_df.empty:
        return "Aucune donnée"
    last = _last_row(state_df)
    ts = last.get("timestamp", "—")
    if isinstance(ts, str):
        ts_short = ts.split("+")[0].replace("T", " ")
    else:
        ts_short = str(ts)
    return f"Dernière mise à jour : {ts_short}  ·  {len(state_df)} points d'état"


# ---------------------------------------------------------------------
# App — mode OFFLINE (lit les CSV)
# ---------------------------------------------------------------------
def create_app(data_dir: Path, refresh_ms: int = 500, state_tail: int = 2000) -> Dash:
    """Crée l'application Dash en mode OFFLINE.

    Les callbacks relisent les CSV à chaque tick — utile pour visualiser
    un run terminé, ou si le dashboard tourne dans un autre process que
    le feed. Latence UI = intervalle du writer async + refresh_ms.
    """
    app = Dash(__name__, title="Crypto MM")
    app.layout = build_layout(data_dir, refresh_ms)

    @app.callback(
        Output("header-status", "children"),
        Output("metric-cards", "children"),
        Output("equity-chart", "figure"),
        Output("position-chart", "figure"),
        Output("microstructure-chart", "figure"),
        Output("spread-chart", "figure"),
        Output("book-table", "children"),
        Output("fills-table", "children"),
        Input("tick", "n_intervals"),
        Input("data-dir", "data"),
    )
    def refresh(_n: int, data_dir_str: str):
        data_dir_p = Path(data_dir_str)
        state_df = _read_csv_safe(
            data_dir_p / "analytics" / "state.csv", tail=state_tail
        )
        fills_df = _read_csv_safe(data_dir_p / "simulation" / "fills.csv", tail=500)
        book_df = _read_csv_safe(data_dir_p / "raw" / "book.csv", tail=200)
        if not book_df.empty:
            book_df = _parse_ts(book_df, "timestamp")

        return (
            build_header_status(state_df),
            build_metric_cards(state_df, fills_df),
            build_equity_figure(state_df),
            build_position_figure(state_df),
            build_microstructure_figure(state_df),
            build_spread_figure(state_df),
            build_book_table(book_df),
            build_fills_table(fills_df),
        )

    return app


# ---------------------------------------------------------------------
# App — mode LIVE (lit la mémoire du feed)
# ---------------------------------------------------------------------
def _snapshot_to_state_df(app, history_cache: list[dict]) -> pd.DataFrame:
    """Consomme `app.get_dashboard_snapshot()` et l'ajoute au cache d'historique.

    Le dashboard a besoin de séries temporelles pour les graphes — on ne
    peut pas tout tirer d'un seul snapshot (qui est un point dans le
    temps). Donc on accumule les snapshots successifs dans un buffer
    persistant côté serveur Dash.

    Cache partagé entre les callbacks via une liste mutable injectée.
    On garde maxlen derniers points pour éviter une fuite mémoire.
    """
    MAX_POINTS = 5000

    snap = app.get_dashboard_snapshot(
        levels=app.settings.top_levels_to_display,
        trades=30,
        spread_points=750,
    )
    portfolio = snap.get("portfolio", {}) or {}
    micro = snap.get("microstructure", {}) or {}
    ctx = snap.get("quote_context", {}) or {}
    risk = snap.get("risk", {}) or {}

    import datetime as _dt

    row = {
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "mid_price": snap.get("mid_price"),
        "best_bid": snap.get("best_bid"),
        "best_ask": snap.get("best_ask"),
        "equity": portfolio.get("equity"),
        "realized_pnl": portfolio.get("realized_pnl"),
        "unrealized_pnl": portfolio.get("unrealized_pnl"),
        "position_btc": portfolio.get("position_btc"),
        "microprice": micro.get("microprice"),
        "microprice_edge_bps": micro.get("microprice_edge_bps"),
        "ofi_ewma": micro.get("ofi_ewma"),
        "trade_flow_signed": micro.get("trade_flow_signed"),
        "vpin": micro.get("vpin"),
        "fast_vol_bps": micro.get("fast_vol_bps"),
        "micro_signal_bps": micro.get("micro_signal_bps"),
        "half_spread_bps": ctx.get("half_spread_bps"),
        "rolling_inside_spread_bps": ctx.get("rolling_inside_spread_bps"),
        "ewma_inside_spread_bps": ctx.get("ewma_inside_spread_bps"),
        "toxicity_widening_bps": ctx.get("toxicity_widening_bps"),
        "fast_vol_widening_bps": ctx.get("fast_vol_widening_bps"),
        "health_score": risk.get("health_score"),
        "loss_utilization": risk.get("loss_utilization"),
        "risk_level": risk.get("risk_level"),
    }
    history_cache.append(row)
    # Trim
    if len(history_cache) > MAX_POINTS:
        del history_cache[: len(history_cache) - MAX_POINTS]
    return pd.DataFrame(history_cache)


def _snapshot_to_book_df(app) -> pd.DataFrame:
    """Convertit le carnet courant en DataFrame attendu par `build_book_table`."""
    snap = app.get_dashboard_snapshot(levels=10, trades=0, spread_points=0)
    rows = []
    from datetime import datetime, timezone

    ts = datetime.now(timezone.utc)
    for rank, item in enumerate(snap.get("bids", []), start=1):
        rows.append(
            {
                "timestamp": ts,
                "side": "bid",
                "rank": rank,
                "price": item["price"],
                "quantity": item["quantity"],
            }
        )
    for rank, item in enumerate(snap.get("asks", []), start=1):
        rows.append(
            {
                "timestamp": ts,
                "side": "ask",
                "rank": rank,
                "price": item["price"],
                "quantity": item["quantity"],
            }
        )
    return pd.DataFrame(rows)


def _snapshot_to_fills_df(app) -> pd.DataFrame:
    """Convertit les fills mémoire en DataFrame."""
    executions = list(app.strategy.executions[-200:])
    if not executions:
        return pd.DataFrame()
    return pd.DataFrame(executions)


def create_live_app(feed_app, refresh_ms: int = 100) -> Dash:
    """Crée l'application Dash en mode LIVE.

    Le dashboard lit directement la mémoire du ``CoinbaseMarketDataApp``
    passé en paramètre. À utiliser quand le feed et le dashboard tournent
    dans le même process (cf. ``main.py --web-dashboard``).

    Intérêts vs mode offline :
    - pas de dépendance aux writers CSV (latence UI ~= refresh_ms) ;
    - granularité 100 ms par défaut (vs 500 ms pour le mode offline) ;
    - reflète instantanément un fill ou un changement de régime.

    Contrainte : dashboard et feed vivent dans le même process. Si le
    feed plante, le dashboard aussi. Pour visualiser un run terminé ou
    isoler les deux, utiliser ``create_app(data_dir)``.
    """
    app = Dash(__name__, title="Crypto MM — Live")
    # Le layout ne dépend pas du data_dir en mode live, on passe un
    # placeholder visible dans l'en-tête.
    app.layout = build_layout(Path("<live memory>"), refresh_ms)

    # Cache persistant côté serveur Dash : conserve l'historique des
    # snapshots successifs pour construire des séries temporelles.
    # Note : closure sur une liste mutable — thread-safe sous Flask dev
    # server (single-request handling par défaut).
    history: list[dict] = []

    @app.callback(
        Output("header-status", "children"),
        Output("metric-cards", "children"),
        Output("equity-chart", "figure"),
        Output("position-chart", "figure"),
        Output("microstructure-chart", "figure"),
        Output("spread-chart", "figure"),
        Output("book-table", "children"),
        Output("fills-table", "children"),
        Input("tick", "n_intervals"),
    )
    def refresh(_n: int):
        state_df = _snapshot_to_state_df(feed_app, history)
        book_df = _snapshot_to_book_df(feed_app)
        fills_df = _snapshot_to_fills_df(feed_app)
        return (
            build_header_status(state_df),
            build_metric_cards(state_df, fills_df),
            build_equity_figure(state_df),
            build_position_figure(state_df),
            build_microstructure_figure(state_df),
            build_spread_figure(state_df),
            build_book_table(book_df),
            build_fills_table(fills_df),
        )

    return app


def run_live_server_threaded(
    feed_app,
    host: str = "127.0.0.1",
    port: int = 8050,
    refresh_ms: int = 100,
) -> "threading.Thread":
    """Lance le dashboard Dash live dans un thread daemon.

    Retourne le thread (qui tourne indéfiniment). Le thread mourra avec
    le process principal puisqu'il est daemon.
    """
    import threading

    dash_app = create_live_app(feed_app, refresh_ms=refresh_ms)

    def _serve() -> None:
        # `use_reloader=False` impératif dans un thread non-principal.
        # `debug=False` pour éviter le hot-reload.
        dash_app.run(host=host, port=port, debug=False, use_reloader=False)

    t = threading.Thread(target=_serve, name="dash-live-server", daemon=True)
    t.start()
    return t


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Dashboard web Dash pour visualiser un run live ou terminé. "
            "Lit les CSV produits par CoinbaseMarketDataApp."
        )
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Répertoire racine des CSV (défaut : data).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port HTTP du dashboard (défaut : 8050).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Interface d'écoute (défaut : 127.0.0.1). Utiliser 0.0.0.0 pour exposer sur le réseau.",
    )
    parser.add_argument(
        "--refresh-ms",
        type=int,
        default=500,
        help=(
            "Intervalle de rafraîchissement en ms (défaut : 500). "
            "Le mode offline dépend aussi du flush des writers CSV "
            "(~ même ordre de grandeur)."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Active le mode debug Dash (hot reload).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(
            f"[dash] Attention : {data_dir} n'existe pas. Le dashboard affichera des graphes vides."
        )
        data_dir.mkdir(parents=True, exist_ok=True)

    app = create_app(data_dir, refresh_ms=args.refresh_ms)
    print(f"[dash] Dashboard démarré sur http://{args.host}:{args.port}")
    print(f"[dash] Source : {data_dir.resolve()}  ·  Refresh : {args.refresh_ms} ms")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
