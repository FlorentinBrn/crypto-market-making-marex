"""Crypto MM — Market making research sandbox for Coinbase BTC-USD.

Structure du package :

- ``crypto_mm.core``    — domaine pur (OrderBook, MarketMaker, RiskManager, ...)
- ``crypto_mm.data``    — I/O : feed WebSocket, persistance CSV, replay, analytics
- ``crypto_mm.tools``   — outils CLI : backtest, bench, stress, analyze, clean, plots
- ``crypto_mm.ui``      — interfaces : config, console Rich, dashboard Dash
- ``crypto_mm.main``    — entry point CLI du run live

Pour les usages courants, les symboles principaux sont réexposés depuis
leurs sous-packages respectifs (cf. ``from crypto_mm.core import ...``).
"""
__version__ = "2.1.0"
