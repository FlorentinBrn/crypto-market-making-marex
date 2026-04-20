"""Replayer standalone — rejoue un run CSV à vitesse variable.

Architecture :
- Lit `raw/book.csv` et `raw/trades.csv` d'un run précédent,
- Les merge en ordre chronologique,
- Les réinjecte dans `CoinbaseMarketDataApp` comme s'ils venaient du
  websocket, en reconstituant le format JSON Coinbase Advanced Trade.
- La vitesse est contrôlée par `--speed` : 1.0 = temps réel, 10.0 = 10x
  plus rapide, 0.5 = demi-vitesse, `inf` = aussi vite que possible.

Usage :
```bash
python -m crypto_mm.data.replay --data-dir data --speed 10
python -m crypto_mm.data.replay --data-dir data --speed inf --no-console
python -m crypto_mm.data.replay --data-dir data --bench-latency
```

Intérêts :
- Itérer sur la stratégie sans attendre un flux live ;
- Reproduire exactement un moment problématique (ex : pic de vol d'hier)
- Profiler la latence sur un dataset stable (comparable avant/après une
  optimisation).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from ..ui.config import Settings
from .feed import CoinbaseMarketDataApp


# ---------------------------------------------------------------------
# Chargement + merge
# ---------------------------------------------------------------------
def _read_csv(path: Path) -> pd.DataFrame:
    if path.is_file() and path.stat().st_size > 0:
        return pd.read_csv(path)
    return pd.DataFrame()


def load_events(data_dir: Path) -> list[tuple[pd.Timestamp, str, dict]]:
    """Charge book.csv + trades.csv et produit une liste d'événements
    `(ts, kind, payload_coinbase_like)` triée chronologiquement.

    `kind` ∈ {'book', 'trades'}.

    `payload_coinbase_like` reproduit le format JSON reçu du websocket
    Coinbase Advanced Trade (ce qui permet de le passer directement à
    `app.on_message` comme si c'était un vrai message).
    """
    book_df = _read_csv(data_dir / "raw" / "book.csv")
    trades_df = _read_csv(data_dir / "raw" / "trades.csv")
    if book_df.empty and trades_df.empty:
        raise FileNotFoundError(
            f"Aucune donnée dans {data_dir}/raw/ — assure-toi d'avoir un run live précédent."
        )

    events: list[tuple[pd.Timestamp, str, dict]] = []

    # Book : un événement par timestamp unique, avec tous les niveaux en 'snapshot'.
    # Remarque : book.csv stocke les top N niveaux par timestamp — on les
    # renvoie comme snapshot entier.
    if not book_df.empty:
        book_df = book_df.copy()
        book_df["ts"] = pd.to_datetime(
            book_df["timestamp"], utc=True, errors="coerce", format="ISO8601"
        )
        book_df = book_df.dropna(subset=["ts"]).sort_values(["ts", "side", "rank"])
        for ts, grp in book_df.groupby("ts", sort=True):
            updates = [
                {
                    "side": str(row["side"]),
                    "price_level": str(row["price"]),
                    "new_quantity": str(row["quantity"]),
                }
                for _, row in grp.iterrows()
            ]
            payload = {
                "channel": "l2_data",
                "timestamp": ts.isoformat(),
                "events": [{"type": "snapshot", "updates": updates}],
            }
            events.append((ts, "book", payload))

    # Trades : un événement par ligne (Coinbase groupe habituellement
    # plusieurs trades par message mais c'est compatible).
    if not trades_df.empty:
        trades_df = trades_df.copy()
        trades_df["ts"] = pd.to_datetime(
            trades_df["time"], utc=True, errors="coerce", format="ISO8601"
        )
        trades_df = trades_df.dropna(subset=["ts"]).sort_values("ts")
        for _, row in trades_df.iterrows():
            trade_msg = {
                "trade_id": str(row["trade_id"]),
                "product_id": str(row["product_id"]),
                "price": float(row["price"]),
                "size": float(row["size"]),
                "side": str(row["side"]),
                "time": pd.Timestamp(row["ts"]).isoformat(),
            }
            payload = {
                "channel": "market_trades",
                "timestamp": pd.Timestamp(row["ts"]).isoformat(),
                "events": [{"trades": [trade_msg]}],
            }
            events.append((pd.Timestamp(row["ts"]), "trades", payload))

    events.sort(key=lambda e: e[0])
    return events


# ---------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------
def replay(
    app: CoinbaseMarketDataApp,
    events: list[tuple[pd.Timestamp, str, dict]],
    speed: float = 1.0,
    progress_every: int = 5000,
) -> None:
    """Pousse les événements dans `app` en respectant le timing relatif.

    `speed=1.0` → temps réel. `speed=inf` → aussi vite que possible.
    """
    if not events:
        return
    start_sim = events[0][0]
    start_wall = time.perf_counter()
    infinite_speed = not (speed > 0 and speed < float("inf"))

    app.connection_status = "replaying"
    import json as _json

    for i, (ts, _kind, payload) in enumerate(events):
        # Sérialiser comme le ferait le ws, puis passer par on_message
        # pour exercer le chemin complet (parsing JSON inclus) — utile
        # pour le bench de latence.
        msg = _json.dumps(payload)

        if not infinite_speed:
            sim_elapsed = (ts - start_sim).total_seconds()
            target_wall = start_wall + sim_elapsed / speed
            wait = target_wall - time.perf_counter()
            if wait > 0:
                time.sleep(wait)

        app.on_message(None, msg)  # type: ignore[arg-type]  (ws ignoré)

        if (i + 1) % progress_every == 0:
            print(
                f"  replay {i + 1:,}/{len(events):,} events "
                f"(sim time {ts.isoformat()})"
            )

    app.connection_status = "replay_done"


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def _parse_speed(s: str) -> float:
    if s.lower() in ("inf", "infinite", "max"):
        return float("inf")
    # Accepte "10" ou "10x"
    s = s.rstrip("xX")
    return float(s)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rejoue un run CSV en injectant les événements dans le moteur."
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Racine des CSV live à rejouer (contient raw/book.csv et raw/trades.csv).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Répertoire de sortie (par défaut = data-dir/replay).",
    )
    parser.add_argument(
        "--speed",
        type=_parse_speed,
        default=1.0,
        help="Multiplicateur de vitesse (1=real, 10=10x, inf=max). Défaut: 1.",
    )
    parser.add_argument(
        "--no-console",
        action="store_true",
        help="Désactive le dashboard Rich.",
    )
    parser.add_argument(
        "--enable-bandit",
        action="store_true",
        help="Active le reinforcement learning optionnel.",
    )
    parser.add_argument(
        "--bench-latency",
        action="store_true",
        help="Active les mesures de latence (sortie data/bench/latency.csv).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else (data_dir / "replay")

    print(f"[replay] Chargement des événements depuis {data_dir}/raw/ ...")
    events = load_events(data_dir)
    print(f"[replay] {len(events):,} événements chargés.")
    if events:
        print(f"[replay] Plage temporelle : {events[0][0]}  →  {events[-1][0]}")

    settings = Settings(
        output_dir=output_dir,
        render_console=not args.no_console,
        use_contextual_bandit=args.enable_bandit,
        bench_latency=args.bench_latency,
    )
    app = CoinbaseMarketDataApp(settings)

    # On n'utilise pas app.run() parce qu'il lance un websocket ;
    # à la place on initialise le Live manuellement si nécessaire.
    if settings.render_console:
        from rich.live import Live
        from ..ui.console import build_placeholder

        app.live = Live(
            build_placeholder("Replay en cours..."),
            screen=True,
            auto_refresh=False,
            transient=False,
        )
        app.live.start(refresh=True)
    try:
        print(f"[replay] Démarrage du replay à vitesse {args.speed}x ...")
        t0 = time.perf_counter()
        replay(app, events, speed=args.speed)
        elapsed = time.perf_counter() - t0
        print(f"[replay] Terminé en {elapsed:.2f}s.")
    finally:
        app.shutdown()
    print(f"[replay] Outputs dans : {output_dir}")


if __name__ == "__main__":
    main()
