from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .config import Settings
    from .feed import CoinbaseMarketDataApp
except ImportError:
    from crypto_mm.config import Settings
    from crypto_mm.feed import CoinbaseMarketDataApp


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
            "À analyser ensuite avec : python -m crypto_mm.bench --data-dir data"
        ),
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
    app.run()


if __name__ == "__main__":
    main()
