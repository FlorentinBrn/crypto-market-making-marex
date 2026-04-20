"""Outils d'analyse et de validation post-run.

- `analyze`  : reconstruction P&L depuis fills.csv
- `backtest` : walk-forward avec reconstruction du carnet
- `bench`    : mesure de latence du chemin critique
- `clean`    : nettoyage des artefacts générés
- `plots`    : génération de graphes d'analyse
- `stress`   : scénarios adverses + vérification d'invariants

Chaque module expose un `main()` CLI.
"""
from .bench import LatencyRecorder

__all__ = ["LatencyRecorder"]
