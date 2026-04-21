"""Dashboard web Dash.

Application Dash qui expose une vue live des métriques de trading :

- P&L réalisé et non-réalisé
- position BTC et prix mid
- spread inside et signaux de microstructure
- état courant du carnet
- fills récents
- cartes de risque (loss utilization, health score, reduce-only)

Deux modes d'exécution :

- ``create_app(data_dir)`` : mode offline, lit périodiquement les CSV
  produits par un run précédent. Utile pour visualiser un run terminé.
- ``create_live_app(feed_app)`` : mode live, branché directement sur la
  mémoire d'un ``CoinbaseMarketDataApp`` en cours d'exécution. Utilise
  le pattern ``extendData`` de Plotly pour des mises à jour incrémentales
  à coût constant par tick.
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
def _local_tz():
    """Retourne le fuseau horaire local du PC.

    Utilise ``datetime.now().astimezone().tzinfo`` qui retourne un
    ``datetime.timezone`` avec offset correct (fonctionne sur tous OS,
    y compris Windows où les noms de fuseau peuvent poser problème à
    pandas).
    """
    from datetime import datetime as _dt

    return _dt.now().astimezone().tzinfo


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
    """Parse une colonne de timestamp et la convertit en **heure locale naïve**.

    Les datetimes tz-aware causent parfois des surprises d'affichage avec
    Plotly (axe X en UTC même si la valeur est tz-aware locale). On retire
    le fuseau après conversion pour que Plotly affiche directement
    l'heure locale sans re-conversion.
    """
    if df.empty or col not in df.columns:
        return df
    df = df.copy()
    parsed = pd.to_datetime(df[col], utc=True, errors="coerce", format="ISO8601")
    try:
        parsed = parsed.dt.tz_convert(_local_tz()).dt.tz_localize(None)
    except Exception:
        parsed = parsed.dt.tz_localize(None)
    df[col] = parsed
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


# Nombre maximum de points envoyés à Plotly par figure. Au-delà, on
# décime (prend 1 point sur N) — l'œil humain ne distingue pas la
# différence sur un graphe de ~1000 px de large, et la charge JSON est
# divisée d'autant.
_MAX_PLOT_POINTS = 300


def _downsample(df: pd.DataFrame, max_points: int = _MAX_PLOT_POINTS) -> pd.DataFrame:
    """Décime un DataFrame pour limiter le nombre de points affichés.

    On garde 1 point sur ``step`` où ``step = ceil(len / max_points)``.
    Préserve toujours le dernier point (utile pour voir la valeur courante).
    """
    n = len(df)
    if n <= max_points:
        return df
    step = (n + max_points - 1) // max_points  # ceil division
    # iloc avec step, puis on s'assure que le dernier point est inclus.
    sub = df.iloc[::step]
    if sub.index[-1] != df.index[-1]:
        sub = pd.concat([sub, df.iloc[[-1]]])
    return sub


def build_equity_figure(state_df: pd.DataFrame, pre_parsed: bool = False) -> go.Figure:
    """P&L réalisé et non-réalisé dans le temps (sans l'equity en valeur
    absolue : on regarde la performance, pas le niveau du capital)."""
    if state_df.empty:
        return _empty_fig("P&L — en attente de données")
    if pre_parsed:
        df = state_df
    else:
        df = _parse_ts(state_df, "timestamp")
        if df.empty:
            return _empty_fig("P&L")
        df = _downsample(df)
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


def build_position_figure(
    state_df: pd.DataFrame, pre_parsed: bool = False
) -> go.Figure:
    """Position BTC + mid price overlay."""
    if state_df.empty:
        return _empty_fig("Position / Mid — en attente de données")
    if pre_parsed:
        df = state_df
    else:
        df = _parse_ts(state_df, "timestamp")
        if df.empty:
            return _empty_fig("Position / Mid")
        df = _downsample(df)
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


def build_microstructure_figure(
    state_df: pd.DataFrame, pre_parsed: bool = False
) -> go.Figure:
    """Micro signal, OFI EWMA, VPIN, fast vol."""
    if state_df.empty:
        return _empty_fig("Microstructure — en attente de données")
    if pre_parsed:
        df = state_df
    else:
        df = _parse_ts(state_df, "timestamp")
        if df.empty:
            return _empty_fig("Microstructure")
        df = _downsample(df)
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


def build_spread_figure(state_df: pd.DataFrame, pre_parsed: bool = False) -> go.Figure:
    """Half-spread bps vs inside spread bps."""
    if state_df.empty:
        return _empty_fig("Spread — en attente de données")
    if pre_parsed:
        df = state_df
    else:
        df = _parse_ts(state_df, "timestamp")
        if df.empty:
            return _empty_fig("Spread")
        df = _downsample(df)
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
    """Formate une heure de fill en HH:MM:SS.mmm (heure locale du PC).

    Accepte soit une chaîne ISO8601 (`2026-04-20T13:42:17.309472+00:00`),
    soit un ``datetime`` (du mode live mémoire où `time` est un dt natif UTC).
    Les deux sont convertis en **heure locale** pour l'affichage (plus naturel
    que UTC pour l'utilisateur qui regarde son dashboard).
    Retourne une chaîne vide si le parsing échoue.
    """
    if raw is None or raw == "":
        return ""
    ts = pd.to_datetime(raw, utc=True, errors="coerce")
    if pd.isna(ts):
        return str(raw)
    try:
        local_ts = ts.tz_convert(_local_tz())
    except Exception:
        local_ts = ts
    return local_ts.strftime("%H:%M:%S.") + f"{local_ts.microsecond // 1000:03d}"


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
    """Heure locale de la dernière mise à jour + nombre d'obs.

    Le timestamp stocké dans state.csv est en UTC (suffixe +00:00 / Z). On le
    convertit en heure locale du PC pour l'affichage.
    """
    if state_df.empty:
        return "Aucune donnée"
    last = _last_row(state_df)
    ts = last.get("timestamp")
    if ts is None:
        return f"{len(state_df)} points d'état"
    ts_parsed = pd.to_datetime(ts, utc=True, errors="coerce", format="ISO8601")
    if pd.isna(ts_parsed):
        return f"{len(state_df)} points d'état"
    try:
        local = ts_parsed.tz_convert(_local_tz())
    except Exception:
        local = ts_parsed
    ts_short = local.strftime("%H:%M:%S")
    return f"Dernière mise à jour : {ts_short} (heure locale)  ·  {len(state_df)} points d'état"


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
# Taille max du buffer d'historique côté serveur Dash.
# - Le buffer grossit jusqu'à cette taille puis l'ancien est évincé (deque avec maxlen).
# Nombre maximum de points conservés par trace côté navigateur.
# Avec le pattern extendData, le serveur ne mémorise pas l'historique :
# il n'envoie que le nouveau point à chaque tick. Cette limite est
# uniquement appliquée côté navigateur par Plotly (éviction des plus
# anciens). À 100 ms de refresh et 1500 points, on a ~2.5 min visibles.
LIVE_HISTORY_MAX_POINTS = 1500


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


def _init_figure_equity() -> go.Figure:
    """Figure initiale vide pour P&L. La structure des traces est fixée une
    fois, les points arrivent ensuite via extendData."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=[], y=[], name="Realized P&L", line=dict(color="#68d391"))
    )
    fig.add_trace(
        go.Scatter(x=[], y=[], name="Unrealized P&L", line=dict(color="#f6ad55"))
    )
    fig.update_layout(title="P&L (realized + unrealized)", **_PLOTLY_DARK_LAYOUT)
    fig.update_yaxes(title="USD")
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)")
    return fig


def _init_figure_position() -> go.Figure:
    """Figure initiale vide pour position / mid."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=[], y=[], name="Position BTC", line=dict(color="#9f7aea"))
    )
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
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


def _init_figure_microstructure() -> go.Figure:
    """Figure initiale vide pour les signaux de microstructure."""
    fig = go.Figure()
    for name, color in [
        ("micro_signal_bps", "#f6ad55"),
        ("ofi_ewma", "#4fd1c5"),
        ("vpin", "#fc8181"),
        ("fast_vol_bps", "#b794f4"),
    ]:
        fig.add_trace(go.Scatter(x=[], y=[], name=name, line=dict(color=color)))
    fig.update_layout(title="Signaux microstructure", **_PLOTLY_DARK_LAYOUT)
    return fig


def _init_figure_spread() -> go.Figure:
    """Figure initiale vide pour spread et widening."""
    fig = go.Figure()
    for col, color, name in [
        ("half_spread_bps", "#4fd1c5", "Half-spread (bps)"),
        ("rolling_inside_spread_bps", "#a0aec0", "Inside spread mean (bps)"),
        ("ewma_inside_spread_bps", "#cbd5e0", "Inside spread EWMA (bps)"),
        ("toxicity_widening_bps", "#fc8181", "Toxicity widening"),
        ("fast_vol_widening_bps", "#f6ad55", "Fast vol widening"),
    ]:
        fig.add_trace(go.Scatter(x=[], y=[], name=name, line=dict(color=color)))
    fig.update_layout(title="Spread & widening", **_PLOTLY_DARK_LAYOUT)
    fig.update_yaxes(title="bps")
    return fig


def _snapshot_row(feed_app) -> dict:
    """Capture un point d'état en mémoire depuis le feed.

    Les valeurs None sont laissées telles quelles : Plotly les affiche
    comme des gaps dans les courbes, ce qui est le comportement attendu
    quand la stratégie n'a pas encore produit le champ.
    """
    from datetime import datetime as _dt, timezone as _tz

    snap = feed_app.get_dashboard_snapshot(
        levels=feed_app.settings.top_levels_to_display,
        trades=30,
        spread_points=50,
    )
    portfolio = snap.get("portfolio", {}) or {}
    micro = snap.get("microstructure", {}) or {}
    ctx = snap.get("quote_context", {}) or {}
    risk = snap.get("risk", {}) or {}

    now_utc = _dt.now(_tz.utc)
    # Pour Plotly : timestamp en datetime local naïve (pas de tz_info)
    # pour un rendu direct de l'heure locale sur l'axe X.
    try:
        ts_local = now_utc.astimezone(_local_tz()).replace(tzinfo=None)
    except Exception:
        ts_local = now_utc.replace(tzinfo=None)

    return {
        "ts_local": ts_local,
        "ts_utc_iso": now_utc.isoformat(),
        "mid_price": snap.get("mid_price"),
        "best_bid": snap.get("best_bid"),
        "best_ask": snap.get("best_ask"),
        "equity": portfolio.get("equity"),
        "realized_pnl": portfolio.get("realized_pnl"),
        "unrealized_pnl": portfolio.get("unrealized_pnl"),
        "position_btc": portfolio.get("position_btc"),
        "microprice": micro.get("microprice"),
        "ofi_ewma": micro.get("ofi_ewma"),
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


def create_live_app(feed_app, refresh_ms: int = 100) -> Dash:
    """Crée l'application Dash en mode live, lisant la mémoire du feed.

    Le dashboard utilise le pattern `extendData` de Plotly : les figures
    sont initialisées une fois avec des traces vides, puis chaque tick
    envoie uniquement les nouveaux points au navigateur, qui les append
    localement. Coût réseau et rendu : constant par tick, indépendant
    du nombre total de points accumulés.

    Trois callbacks distincts :

    - ``update_charts`` : rythme rapide (refresh_ms), envoie un seul
      point par trace pour les 4 graphes (P&L, position, microstructure,
      spread). Utilise la propriété ``extendData`` des graphes.
    - ``update_header_and_cards`` : rythme rapide, met à jour le header
      et les 6 cartes métriques (peu coûteux).
    - ``update_tables`` : rythme plus lent (``refresh_ms * 3``), met à
      jour le carnet et la liste des fills. Les tables sont le poste le
      plus coûteux par mise à jour.

    Contrainte : le dashboard et le feed vivent dans le même process.
    Pour visualiser un run terminé depuis les CSV, utiliser
    ``create_app(data_dir)``.
    """
    app = Dash(__name__, title="Crypto MM — Live")
    app.layout = _build_live_layout(refresh_ms)

    # Nombre maximum de points conservés côté navigateur par trace.
    # Le pattern extendData de Plotly prend ce paramètre en dernier
    # argument ; au-delà, les points les plus anciens sont évincés.
    # Le serveur lui-même ne mémorise rien (voir _snapshot_row).
    max_pts_per_trace = LIVE_HISTORY_MAX_POINTS

    @app.callback(
        Output("equity-chart", "extendData"),
        Output("position-chart", "extendData"),
        Output("microstructure-chart", "extendData"),
        Output("spread-chart", "extendData"),
        Input("tick-fast", "n_intervals"),
        prevent_initial_call=True,
    )
    def update_charts(_n: int):
        row = _snapshot_row(feed_app)
        ts = [row["ts_local"]]

        # Format extendData : (dict, trace_indices, max_points)
        # dict = {"x": [[xs_trace0], [xs_trace1], ...], "y": [[...], ...]}
        equity_data = (
            {"x": [ts, ts], "y": [[row["realized_pnl"]], [row["unrealized_pnl"]]]},
            [0, 1],
            max_pts_per_trace,
        )
        position_data = (
            {"x": [ts, ts], "y": [[row["position_btc"]], [row["mid_price"]]]},
            [0, 1],
            max_pts_per_trace,
        )
        microstructure_data = (
            {
                "x": [ts, ts, ts, ts],
                "y": [
                    [row["micro_signal_bps"]],
                    [row["ofi_ewma"]],
                    [row["vpin"]],
                    [row["fast_vol_bps"]],
                ],
            },
            [0, 1, 2, 3],
            max_pts_per_trace,
        )
        spread_data = (
            {
                "x": [ts, ts, ts, ts, ts],
                "y": [
                    [row["half_spread_bps"]],
                    [row["rolling_inside_spread_bps"]],
                    [row["ewma_inside_spread_bps"]],
                    [row["toxicity_widening_bps"]],
                    [row["fast_vol_widening_bps"]],
                ],
            },
            [0, 1, 2, 3, 4],
            max_pts_per_trace,
        )
        return equity_data, position_data, microstructure_data, spread_data

    @app.callback(
        Output("header-status", "children"),
        Output("metric-cards", "children"),
        Input("tick-fast", "n_intervals"),
    )
    def update_header_and_cards(_n: int):
        row = _snapshot_row(feed_app)
        return (
            _build_header_status_live(row),
            _build_metric_cards_live(row, feed_app),
        )

    @app.callback(
        Output("book-table", "children"),
        Output("fills-table", "children"),
        Input("tick-slow", "n_intervals"),
    )
    def update_tables(_n: int):
        return (
            build_book_table(_snapshot_to_book_df(feed_app)),
            build_fills_table(_snapshot_to_fills_df(feed_app)),
        )

    return app


def _build_live_layout(refresh_ms: int) -> html.Div:
    """Layout du mode live avec deux intervals (tables plus lentes)."""
    slow_ms = max(refresh_ms * 3, 400)
    return html.Div(
        style=PAGE_STYLE,
        children=[
            dcc.Interval(id="tick-fast", interval=refresh_ms, n_intervals=0),
            dcc.Interval(id="tick-slow", interval=slow_ms, n_intervals=0),
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
                        id="header-status", style={"fontSize": "0.9rem", "opacity": 0.8}
                    ),
                ],
            ),
            html.Div(
                f"Mode live · Rafraîchissement graphes : {refresh_ms} ms · tables : {slow_ms} ms",
                style={
                    "fontSize": "0.8rem",
                    "opacity": 0.5,
                    "marginTop": "4px",
                    "marginBottom": "20px",
                },
            ),
            html.Div(
                id="metric-cards",
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(6, 1fr)",
                    "gap": "14px",
                    "marginBottom": "20px",
                },
            ),
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr 1fr",
                    "gap": "14px",
                    "marginBottom": "20px",
                },
                children=[
                    html.Div(
                        dcc.Graph(id="equity-chart", figure=_init_figure_equity()),
                        style=CARD_STYLE,
                    ),
                    html.Div(
                        dcc.Graph(id="position-chart", figure=_init_figure_position()),
                        style=CARD_STYLE,
                    ),
                ],
            ),
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr 1fr",
                    "gap": "14px",
                    "marginBottom": "20px",
                },
                children=[
                    html.Div(
                        dcc.Graph(
                            id="microstructure-chart",
                            figure=_init_figure_microstructure(),
                        ),
                        style=CARD_STYLE,
                    ),
                    html.Div(
                        dcc.Graph(id="spread-chart", figure=_init_figure_spread()),
                        style=CARD_STYLE,
                    ),
                ],
            ),
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


def _build_header_status_live(row: dict) -> str:
    """Heure locale + indicateur vivant basé sur le dernier snapshot."""
    ts_local = row.get("ts_local")
    if ts_local is None:
        return "En attente de données"
    return f"Dernière mise à jour : {ts_local.strftime('%H:%M:%S')} (heure locale)"


def _build_metric_cards_live(row: dict, feed_app) -> list[html.Div]:
    """Cartes métriques construites directement depuis le dernier snapshot."""
    equity = row.get("equity")
    realized = row.get("realized_pnl")
    unreal = row.get("unrealized_pnl")
    position = row.get("position_btc")
    mid = row.get("mid_price")
    health = row.get("health_score")
    loss_util = row.get("loss_utilization")
    risk_level = row.get("risk_level") or "—"
    n_fills = len(feed_app.strategy.executions)

    return [
        _metric_card(
            "Equity",
            _fmt_num(equity, 2),
            f"{_fmt_money((equity or 0) - 1_000_000)} vs init",
        ),
        _metric_card("Realized P&L", _fmt_money(realized)),
        _metric_card("Unrealized P&L", _fmt_money(unreal)),
        _metric_card("Position BTC", _fmt_num(position, 6), f"mid {_fmt_num(mid, 2)}"),
        _metric_card("Health score", _fmt_num(health, 0), f"risk {risk_level}"),
        _metric_card("Loss util.", _fmt_pct(loss_util), f"{n_fills} fills au total"),
    ]


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
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        dev_tools_silence_routes_logging=True,
    )


if __name__ == "__main__":
    main()
