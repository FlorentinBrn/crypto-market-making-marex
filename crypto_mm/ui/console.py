from __future__ import annotations

from typing import Any

from rich.align import Align
from rich.columns import Columns
from rich.console import Group
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..core.utils import format_local_time, local_now, parse_timestamp


# ---------------------------------------------------------------------
# Helpers de formatage
# ---------------------------------------------------------------------
def _fmt_price(value: float | None) -> str:
    return "-" if value is None else f"{value:,.2f}"


def _fmt_qty(value: float | None) -> str:
    return "-" if value is None else f"{value:.6f}"


def _fmt_signed_usd(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:+,.2f}"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * value:.1f}%"


def _fmt_bps(value: float | None, sign: bool = False) -> str:
    if value is None:
        return "-"
    if sign:
        return f"{value:+.2f} bps"
    return f"{value:.2f} bps"


def _color_from_sign(
    value: float | None, *, positive: str = "green", negative: str = "red"
) -> str:
    if value is None:
        return "white"
    if value > 0:
        return positive
    if value < 0:
        return negative
    return "white"


def build_placeholder(message: str) -> Panel:
    return Panel(
        Align.center(f"[bold cyan]{message}[/bold cyan]", vertical="middle"),
        title="Crypto MM",
        border_style="cyan",
    )


# ---------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------
def _build_header(snapshot: dict[str, Any]) -> Panel:
    product = snapshot.get("product_id", "-")
    status = snapshot.get("connection_status", "-")
    status_style = {
        "connected": "green",
        "connecting": "yellow",
        "error": "red",
        "closed": "red",
        "initialized": "cyan",
    }.get(str(status), "white")
    stats = snapshot.get("stats", {})
    last_age = stats.get("last_message_age_ms")
    age_text = "-" if last_age is None else f"{last_age:,.0f} ms"
    fills = stats.get("fills_count", 0)
    trades = stats.get("trades_processed", 0)
    messages = stats.get("messages_received", 0)
    bandit_flag = "ON" if snapshot.get("bandit_enabled", False) else "OFF"

    text = Text()
    text.append(f"{product}  ", style="bold cyan")
    text.append(f"[{local_now().tzname()}]  ", style="white")
    text.append("● ", style=status_style)
    text.append(f"{str(status).upper()}   ", style=f"bold {status_style}")
    text.append(
        f"Messages: {messages:,}   Trades: {trades:,}   Fills: {fills:,}   "
        f"Last msg: {age_text}   Bandit: {bandit_flag}"
    )
    return Panel(text, border_style="cyan")


def _build_orderbook(snapshot: dict[str, Any]) -> Panel:
    # Convention trading : BID à gauche (vert), ASK à droite (rouge).
    # Le meilleur bid est affiché en haut de sa colonne (prix le plus haut),
    # le meilleur ask en haut de la sienne (prix le plus bas).
    table = Table(expand=True, box=None, pad_edge=False)
    table.add_column("BID Qty", justify="right", style="green")
    table.add_column("BID Px", justify="right", style="green")
    table.add_column("ASK Px", justify="right", style="red")
    table.add_column("ASK Qty", justify="right", style="red")

    bids = snapshot.get("bids", [])
    asks = snapshot.get("asks", [])
    rows = max(len(bids), len(asks), 8)
    for idx in range(rows):
        bid = bids[idx] if idx < len(bids) else None
        ask = asks[idx] if idx < len(asks) else None
        bid_qty = _fmt_qty(None if bid is None else bid.get("quantity"))
        bid_px = _fmt_price(None if bid is None else bid.get("price"))
        ask_px = _fmt_price(None if ask is None else ask.get("price"))
        ask_qty = _fmt_qty(None if ask is None else ask.get("quantity"))
        style = "bold" if idx == 0 else ""
        table.add_row(bid_qty, bid_px, ask_px, ask_qty, style=style)

    best_bid = snapshot.get("best_bid")
    best_ask = snapshot.get("best_ask")
    mid = snapshot.get("mid_price")
    spread_abs = None if best_bid is None or best_ask is None else best_ask - best_bid
    spread_bps = None if spread_abs is None or not mid else spread_abs / mid * 10_000
    subtitle = (
        f"Spread: {_fmt_price(spread_abs)} USD | "
        f"{('-' if spread_bps is None else f'{spread_bps:.2f} bps')} | "
        f"Mid: {_fmt_price(mid)}"
    )
    return Panel(table, title="Carnet d'ordres", subtitle=subtitle, border_style="blue")


def _build_spreads(snapshot: dict[str, Any]) -> Panel:
    """Spreads effectifs par taille (0.1, 1, 5, 10 BTC)."""
    table = Table(expand=True, box=None, pad_edge=False)
    table.add_column("Taille", justify="right", style="cyan")
    table.add_column("Avg", justify="right")
    table.add_column("Med", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("Obs", justify="right")

    for size, row in snapshot.get("spread_summaries", {}).items():
        table.add_row(
            f"{float(size):.1f} BTC",
            "-" if row.get("average") is None else f"{row['average']:.2f}",
            "-" if row.get("median") is None else f"{row['median']:.2f}",
            "-" if row.get("minimum") is None else f"{row['minimum']:.2f}",
            "-" if row.get("maximum") is None else f"{row['maximum']:.2f}",
            str(row.get("observations", 0)),
        )
    return Panel(table, title="Spreads effectifs", border_style="magenta")


def _build_portfolio(snapshot: dict[str, Any]) -> Panel:
    portfolio = snapshot.get("portfolio", {})
    risk = snapshot.get("risk", {})
    micro = snapshot.get("microstructure", {})
    quote_ctx = snapshot.get("quote_context", {})
    quotes = snapshot.get("current_quotes", {})

    grid = Table.grid(expand=True)
    grid.add_column(ratio=1)
    grid.add_column(justify="right")

    pos = portfolio.get("position_btc")
    pos_style = _color_from_sign(pos)
    eq = portfolio.get("equity")
    realized = portfolio.get("realized_pnl")
    unreal = portfolio.get("unrealized_pnl")

    rows = [
        (
            "Position BTC",
            f"[{pos_style}]{pos:+.6f}[/{pos_style}]" if pos is not None else "-",
        ),
        ("Prix moyen", _fmt_price(portfolio.get("avg_entry_price"))),
        ("Exposition USD", _fmt_signed_usd(portfolio.get("exposure_usd"))),
        (
            "Realized P&L",
            f"[{_color_from_sign(realized)}]{_fmt_signed_usd(realized)}[/{_color_from_sign(realized)}]",
        ),
        (
            "Unrealized P&L",
            f"[{_color_from_sign(unreal)}]{_fmt_signed_usd(unreal)}[/{_color_from_sign(unreal)}]",
        ),
        (
            "Equity",
            f"[{_color_from_sign((eq or 0) - 1_000_000, positive='green', negative='red')}]"
            f"{_fmt_price(eq)}"
            f"[/{_color_from_sign((eq or 0) - 1_000_000, positive='green', negative='red')}]",
        ),
        ("Risk level", str(risk.get("risk_level", "-"))),
        ("Health score", str(risk.get("health_score", "-"))),
        ("Reduce-only", "YES" if risk.get("reduce_only") else "NO"),
        ("Loss util.", _fmt_pct(risk.get("loss_utilization"))),
        ("Exposure util.", _fmt_pct(risk.get("exposure_utilization"))),
        ("Quote bid", _fmt_price(quotes.get("bid"))),
        ("Quote ask", _fmt_price(quotes.get("ask"))),
        ("Quote size", _fmt_qty(quotes.get("size"))),
        ("Microprice edge", _fmt_bps(micro.get("microprice_edge_bps"), sign=True)),
        (
            "Imbalance L1",
            (
                "-"
                if micro.get("imbalance_l1") is None
                else f"{micro['imbalance_l1']:+.3f}"
            ),
        ),
        (
            "Imbalance L3",
            (
                "-"
                if micro.get("imbalance_l3") is None
                else f"{micro['imbalance_l3']:+.3f}"
            ),
        ),
        (
            "OFI EWMA",
            "-" if micro.get("ofi_ewma") is None else f"{micro['ofi_ewma']:+.2f}",
        ),
        (
            "Trade flow",
            (
                "-"
                if micro.get("trade_flow_signed") is None
                else f"{micro['trade_flow_signed']:+.3f}"
            ),
        ),
        ("VPIN", "-" if micro.get("vpin") is None else f"{micro['vpin']:.3f}"),
        ("Fast vol", _fmt_bps(micro.get("fast_vol_bps"))),
        ("Micro signal", _fmt_bps(micro.get("micro_signal_bps"), sign=True)),
        ("Half spread", _fmt_bps(quote_ctx.get("half_spread_bps"))),
        ("Tox widening", _fmt_bps(quote_ctx.get("toxicity_widening_bps"))),
        ("Vol widening", _fmt_bps(quote_ctx.get("fast_vol_widening_bps"))),
        ("Signal shift", _fmt_bps(quote_ctx.get("signal_shift_bps"), sign=True)),
        ("Inv skew", _fmt_bps(quote_ctx.get("inventory_skew_bps"), sign=True)),
        ("Spread mean", _fmt_bps(quote_ctx.get("rolling_inside_spread_bps"))),
        ("Spread EWMA", _fmt_bps(quote_ctx.get("ewma_inside_spread_bps"))),
    ]
    # Ligne bandit seulement si actif.
    if quote_ctx.get("bandit_arm") is not None:
        rows.extend(
            [
                ("Bandit arm", str(quote_ctx.get("bandit_arm"))),
                ("Bandit state", str(quote_ctx.get("bandit_state", "-"))),
                (
                    "Bandit reward",
                    (
                        "-"
                        if quote_ctx.get("bandit_reward") is None
                        else f"{quote_ctx['bandit_reward']:+.2f}"
                    ),
                ),
            ]
        )

    for label, value in rows:
        grid.add_row(label, str(value))

    extras: list[Any] = [grid]
    alerts = list(risk.get("alerts", []))
    recos = list(risk.get("recommendations", []))
    if alerts:
        alert_text = Text("\n".join(f"• {x}" for x in alerts), style="red")
        extras.append(Panel(alert_text, title="Alertes", border_style="red"))
    if recos:
        reco_text = Text("\n".join(f"• {x}" for x in recos), style="yellow")
        extras.append(Panel(reco_text, title="Recommandations", border_style="yellow"))

    return Panel(Group(*extras), title="Portefeuille / Risque", border_style="green")


def _build_trade_table(
    rows: list[dict[str, Any]], title: str, *, show_reason: bool = False
) -> Panel:
    table = Table(expand=True, box=None, pad_edge=False)
    table.add_column("Heure", style="white", no_wrap=True)
    table.add_column("Side", justify="center")
    table.add_column("Prix", justify="right")
    table.add_column("Taille", justify="right")
    if show_reason:
        table.add_column("Raison", overflow="fold")

    for row in rows[:10]:
        raw_time = row.get("time")
        if isinstance(raw_time, str) and raw_time:
            try:
                time_str = format_local_time(parse_timestamp(raw_time))
            except Exception:
                time_str = (
                    raw_time.split("T")[-1][:12] if "T" in raw_time else raw_time[:12]
                )
        else:
            time_str = str(raw_time or "")[:12]
        side = str(row.get("side", "-"))
        side_style = "green" if side.upper().startswith("B") else "red"
        values = [
            time_str,
            f"[{side_style}]{side}[/{side_style}]",
            _fmt_price(row.get("price")),
            _fmt_qty(row.get("size")),
        ]
        if show_reason:
            values.append(str(row.get("reason", "-")))
        table.add_row(*values)
    return Panel(table, title=title, border_style="cyan")


# ---------------------------------------------------------------------
# Layout complet
# ---------------------------------------------------------------------
def build_dashboard(snapshot: dict[str, Any]) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body", ratio=1),
        Layout(name="footer", size=14),
    )
    layout["body"].split_row(
        Layout(name="left", ratio=5),
        Layout(name="center", ratio=4),
        Layout(name="right", ratio=5),
    )
    layout["left"].split_column(
        Layout(name="orderbook", ratio=3),
        Layout(name="spreads", ratio=2),
    )
    layout["center"].update(_build_portfolio(snapshot))
    layout["right"].split_column(
        Layout(name="trades", ratio=1),
        Layout(name="fills", ratio=1),
    )

    layout["header"].update(_build_header(snapshot))
    layout["orderbook"].update(_build_orderbook(snapshot))
    layout["spreads"].update(_build_spreads(snapshot))
    layout["trades"].update(
        _build_trade_table(snapshot.get("recent_trades", []), "Trades marché")
    )
    layout["fills"].update(
        _build_trade_table(
            snapshot.get("executions", []), "Exécutions simulées", show_reason=True
        )
    )

    footer_cols = Columns(
        [
            Panel(
                f"Best bid\n[bold green]{_fmt_price(snapshot.get('best_bid'))}[/bold green]",
                border_style="green",
            ),
            Panel(
                f"Best ask\n[bold red]{_fmt_price(snapshot.get('best_ask'))}[/bold red]",
                border_style="red",
            ),
            Panel(
                f"Mid\n[bold cyan]{_fmt_price(snapshot.get('mid_price'))}[/bold cyan]",
                border_style="cyan",
            ),
        ],
        expand=True,
    )
    layout["footer"].update(footer_cols)
    return layout
