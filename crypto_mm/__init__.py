"""Crypto MM Exercise — simulateur de market making BTC-USD sur Coinbase.

Modules principaux :
- orderbook  : carnet niveau 2, microprice, imbalance, OFI
- analytics  : spread tracker + signaux microstructure (OFI, VPIN, flow)
- strategy   : MarketMaker (quoting, skew, fill sim)
- risk       : RiskManager (kill switch, reduce-only, health score)
- learning   : ContextualBanditQuoter (RL optionnel)
- feed       : application principale (websocket + CSV)
- backtest   : walk-forward offline
- analyze    : reconstruction P&L post-run depuis les fills CSV
"""
__version__ = "2.0.0"
