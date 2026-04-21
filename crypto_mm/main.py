from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .ui.config import Settings
    from .data.feed import CoinbaseMarketDataApp
except ImportError:
    from crypto_mm.ui.config import Settings
    from crypto_mm.data.feed import CoinbaseMarketDataApp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Crypto MM Exercise — simulateur de market making BTC-USD sur Coinbase.",
    )
    parser.add_argument(
        "--product-id",
        default="BTC-USD",
        help="Paire suivie (défaut: BTC-USD).",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Répertoire où seront écrits les CSV et graphiques.",
    )
    parser.add_argument(
        "--no-console",
        action="store_true",
        help="Désactive le dashboard Rich (utile si utilisé via un autre front).",
    )
    parser.add_argument(
        "--enable-bandit",
        action="store_true",
        help=(
            "Active le reinforcement learning optionnel (contextual bandit). "
            "Désactivé par défaut."
        ),
    )
    parser.add_argument(
        "--quote-size",
        type=float,
        default=None,
        help="Taille de quote en BTC (écrase la valeur par défaut de la config).",
    )
    parser.add_argument(
        "--base-half-spread-bps",
        type=float,
        default=None,
        help="Half-spread de base en bps.",
    )
    parser.add_argument(
        "--bench-latency",
        action="store_true",
        help=(
            "Active la mesure de latence du chemin critique. "
            "Les mesures sont écrites dans data/bench/latency.csv. "
            "À analyser ensuite avec : python -m crypto_mm.tools.bench --data-dir data"
        ),
    )
    parser.add_argument(
        "--web-dashboard",
        action="store_true",
        help=(
            "Lance le dashboard web Dash en mode live dans un thread "
            "daemon, branché sur la mémoire du feed. Accessible par "
            "défaut sur http://localhost:8050."
        ),
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=8050,
        help="Port du dashboard web (défaut: 8050). Actif si --web-dashboard.",
    )
    parser.add_argument(
        "--web-host",
        default="127.0.0.1",
        help=(
            "Interface d'écoute du dashboard web (défaut: 127.0.0.1). "
            "Utiliser 0.0.0.0 pour exposer sur le réseau local."
        ),
    )
    parser.add_argument(
        "--web-refresh-ms",
        type=int,
        default=250,
        help="Intervalle de refresh du dashboard web en ms (défaut: 250).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    settings = Settings(
        product_id=args.product_id,
        output_dir=Path(args.output_dir),
        render_console=not args.no_console,
        use_contextual_bandit=args.enable_bandit,
        bench_latency=args.bench_latency,
    )
    if args.quote_size is not None:
        settings.quote_size_btc = args.quote_size
    if args.base_half_spread_bps is not None:
        settings.base_half_spread_bps = args.base_half_spread_bps

    app = CoinbaseMarketDataApp(settings)

    # Dashboard web live optionnel : serveur Dash dans un thread daemon
    # branché directement sur la mémoire de ``app``.
    if args.web_dashboard:
        # Import tardif pour ne pas imposer Dash comme dépendance si
        # le mode web n'est pas utilisé.
        from .ui.dash_app import run_live_server_threaded

        print(
            f"[dashboard] Web dashboard live : "
            f"http://{args.web_host}:{args.web_port}  (refresh {args.web_refresh_ms} ms)"
        )
        run_live_server_threaded(
            app,
            host=args.web_host,
            port=args.web_port,
            refresh_ms=args.web_refresh_ms,
        )

    app.run()


if __name__ == "__main__":
    main()
