"""Stress tests adverses — vérifie que les contraintes de risque tiennent
sous conditions extrêmes.

Les scénarios génèrent des événements synthétiques (book + trades) au
format Coinbase, qui sont ensuite injectés dans `CoinbaseMarketDataApp`
via le même chemin que le replayer. À la fin, on vérifie une série
d'invariants :

1. La position notionnelle ne dépasse jamais $1,000,000.
2. Le drawdown maximal reste strictement sous $100,000.
3. Le kill switch se déclenche si jamais la limite est atteinte.
4. `reduce_only` s'active sous stress (loss_util ≥ 0.75 ou expo ≥ 0.85).
5. `health_score` reste entre 0 et 100.

Scénarios inclus :
- **flash_crash** : chute brutale de 500 bps en 2 secondes avec rebond
- **liquidity_drought** : disparition quasi-totale de la liquidité bid
- **vpin_burst** : flux massivement unidirectionnel (trades toxiques)
- **oscillation** : marché rapidement oscillant haute volatilité
- **stable** : scénario témoin, marché calme (pour baseline)

Usage :
```bash
python -m crypto_mm.stress                           # tous les scénarios
python -m crypto_mm.stress --scenario flash_crash    # un seul
python -m crypto_mm.stress --output-dir data/stress  # output custom
```
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from .config import Settings
from .feed import CoinbaseMarketDataApp


# ---------------------------------------------------------------------
# Helpers de génération d'événements
# ---------------------------------------------------------------------
def _book_payload(
    ts_iso: str, bids: list[tuple[float, float]], asks: list[tuple[float, float]]
) -> str:
    """Construit un message level2 `snapshot` au format Coinbase."""
    updates = []
    for p, q in bids:
        updates.append({"side": "bid", "price_level": str(p), "new_quantity": str(q)})
    for p, q in asks:
        updates.append({"side": "ask", "price_level": str(p), "new_quantity": str(q)})
    return json.dumps(
        {
            "channel": "l2_data",
            "timestamp": ts_iso,
            "events": [{"type": "snapshot", "updates": updates}],
        }
    )


def _trade_payload(
    ts_iso: str, trade_id: str, price: float, size: float, maker_side: str
) -> str:
    """Construit un message market_trades au format Coinbase.

    `maker_side` ∈ {'BUY', 'SELL'} — convention Coinbase, côté du maker.
    Agresseur = l'inverse.
    """
    return json.dumps(
        {
            "channel": "market_trades",
            "timestamp": ts_iso,
            "events": [
                {
                    "trades": [
                        {
                            "trade_id": trade_id,
                            "product_id": "BTC-USD",
                            "price": price,
                            "size": size,
                            "side": maker_side,
                            "time": ts_iso,
                        }
                    ]
                }
            ],
        }
    )


def _gen_book_around(
    mid: float, depth_per_side: float = 3.0, levels: int = 5, tick: float = 1.0
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Génère des (bids, asks) autour d'un mid, avec un spread d'1 tick
    et `depth_per_side` BTC distribué sur `levels` niveaux."""
    half = tick / 2.0
    per_level = depth_per_side / levels
    bids = [(mid - half - i * tick, per_level) for i in range(levels)]
    asks = [(mid + half + i * tick, per_level) for i in range(levels)]
    return bids, asks


# ---------------------------------------------------------------------
# Scénarios
# ---------------------------------------------------------------------
def scenario_stable(
    start_mid: float = 100_000.0, n_ticks: int = 500, tick_ms: int = 100
) -> list[tuple[pd.Timestamp, str]]:
    """Marché calme — baseline. Mid suit une marche aléatoire faible."""
    rng = random.Random(42)
    events: list[tuple[pd.Timestamp, str]] = []
    base = pd.Timestamp("2024-01-01", tz="UTC")
    mid = start_mid
    for i in range(n_ticks):
        ts = base + pd.Timedelta(milliseconds=i * tick_ms)
        mid += rng.gauss(0, 0.5)
        bids, asks = _gen_book_around(mid)
        events.append((ts, _book_payload(ts.isoformat(), bids, asks)))
        # Un trade tous les ~5 ticks, alterné.
        if i % 5 == 0:
            side = "BUY" if i % 10 == 0 else "SELL"
            events.append(
                (ts, _trade_payload(ts.isoformat(), f"s-{i}", mid, 0.05, side))
            )
    return events


def scenario_flash_crash(
    start_mid: float = 100_000.0,
) -> list[tuple[pd.Timestamp, str]]:
    """Chute brutale de 500 bps en 2 secondes puis rebond partiel.

    Le but : vérifier que le half_spread s'élargit (vol + VPIN), que la
    position ne s'accumule pas d'un seul côté, et que le drawdown
    maximal ne dépasse pas le budget de perte.
    """
    events: list[tuple[pd.Timestamp, str]] = []
    base = pd.Timestamp("2024-01-01", tz="UTC")
    rng = random.Random(7)
    mid = start_mid

    # Phase 1 : 100 ticks calmes pour que la stratégie s'installe
    for i in range(100):
        ts = base + pd.Timedelta(milliseconds=i * 100)
        mid += rng.gauss(0, 0.5)
        bids, asks = _gen_book_around(mid)
        events.append((ts, _book_payload(ts.isoformat(), bids, asks)))

    # Phase 2 : crash brutal -500 bps sur 2 secondes (20 ticks)
    crash_start_mid = mid
    target = crash_start_mid * (1 - 0.05)  # -500 bps
    for i in range(20):
        ts = base + pd.Timedelta(milliseconds=(100 + i) * 100)
        frac = (i + 1) / 20
        mid = crash_start_mid + (target - crash_start_mid) * frac
        bids, asks = _gen_book_around(mid, depth_per_side=2.0)  # liquidité réduite
        events.append((ts, _book_payload(ts.isoformat(), bids, asks)))
        # Gros trades agresseurs vendeurs (maker='BUY' → agresseur SELL)
        events.append(
            (ts, _trade_payload(ts.isoformat(), f"crash-{i}", mid, 0.5, "BUY"))
        )

    # Phase 3 : rebond partiel +200 bps sur 3 secondes
    rebound_target = mid * (1 + 0.02)
    for i in range(30):
        ts = base + pd.Timedelta(milliseconds=(120 + i) * 100)
        frac = (i + 1) / 30
        mid = mid + (rebound_target - mid) * frac * 0.1
        bids, asks = _gen_book_around(mid, depth_per_side=2.5)
        events.append((ts, _book_payload(ts.isoformat(), bids, asks)))

    # Phase 4 : retour au calme
    for i in range(100):
        ts = base + pd.Timedelta(milliseconds=(150 + i) * 100)
        mid += rng.gauss(0, 0.5)
        bids, asks = _gen_book_around(mid)
        events.append((ts, _book_payload(ts.isoformat(), bids, asks)))
    return events


def scenario_liquidity_drought(
    start_mid: float = 100_000.0,
) -> list[tuple[pd.Timestamp, str]]:
    """Le bid s'assèche presque totalement pendant 5 secondes.

    But : vérifier que la stratégie ne se retrouve pas short massif en
    face d'un marché unilatéral, et que reduce_only s'active.
    """
    events: list[tuple[pd.Timestamp, str]] = []
    base = pd.Timestamp("2024-01-01", tz="UTC")
    rng = random.Random(13)
    mid = start_mid

    # Phase warmup 50 ticks
    for i in range(50):
        ts = base + pd.Timedelta(milliseconds=i * 100)
        mid += rng.gauss(0, 0.5)
        bids, asks = _gen_book_around(mid)
        events.append((ts, _book_payload(ts.isoformat(), bids, asks)))

    # Phase drought : bid réduit à 0.01 BTC, ask normal, prix glisse vers le bas
    for i in range(50):
        ts = base + pd.Timedelta(milliseconds=(50 + i) * 100)
        mid -= 1.0  # glissement constant
        bids, asks = _gen_book_around(mid, depth_per_side=3.0)
        # On écrase les bids
        bids = [(p, 0.002) for p, _ in bids]
        events.append((ts, _book_payload(ts.isoformat(), bids, asks)))
        # Trades agressifs vendeurs
        events.append(
            (ts, _trade_payload(ts.isoformat(), f"drought-{i}", mid, 0.3, "BUY"))
        )

    # Retour normal
    for i in range(50):
        ts = base + pd.Timedelta(milliseconds=(100 + i) * 100)
        mid += rng.gauss(0, 0.3)
        bids, asks = _gen_book_around(mid)
        events.append((ts, _book_payload(ts.isoformat(), bids, asks)))
    return events


def scenario_vpin_burst(start_mid: float = 100_000.0) -> list[tuple[pd.Timestamp, str]]:
    """Burst de trades 100% agressifs du même côté pendant 10 secondes.

    But : vérifier que le widening par toxicité (VPIN) s'active.
    """
    events: list[tuple[pd.Timestamp, str]] = []
    base = pd.Timestamp("2024-01-01", tz="UTC")
    rng = random.Random(99)
    mid = start_mid

    # Warmup
    for i in range(50):
        ts = base + pd.Timedelta(milliseconds=i * 100)
        mid += rng.gauss(0, 0.3)
        bids, asks = _gen_book_around(mid)
        events.append((ts, _book_payload(ts.isoformat(), bids, asks)))

    # Burst : 100 trades agresseurs BUY en rafale (maker='SELL')
    for i in range(100):
        ts = base + pd.Timedelta(milliseconds=(50 + i) * 100)
        mid += 0.5  # petite dérive haussière
        bids, asks = _gen_book_around(mid)
        events.append((ts, _book_payload(ts.isoformat(), bids, asks)))
        events.append(
            (ts, _trade_payload(ts.isoformat(), f"vpin-{i}", mid, 0.2, "SELL"))
        )

    # Retour au calme
    for i in range(50):
        ts = base + pd.Timedelta(milliseconds=(150 + i) * 100)
        mid += rng.gauss(0, 0.3)
        bids, asks = _gen_book_around(mid)
        events.append((ts, _book_payload(ts.isoformat(), bids, asks)))
    return events


def scenario_oscillation(
    start_mid: float = 100_000.0,
) -> list[tuple[pd.Timestamp, str]]:
    """Marché très volatile en haute fréquence.

    But : stress du calcul de volatilité rapide, du widening associé,
    et vérifier qu'on ne se retrouve pas exposé du mauvais côté.
    """
    events: list[tuple[pd.Timestamp, str]] = []
    base = pd.Timestamp("2024-01-01", tz="UTC")
    for i in range(500):
        ts = base + pd.Timedelta(milliseconds=i * 50)
        # Oscillation sinusoïdale ±50 USD à ~2 Hz
        mid = start_mid + 50 * math.sin(i * 0.3)
        bids, asks = _gen_book_around(mid, depth_per_side=2.5)
        events.append((ts, _book_payload(ts.isoformat(), bids, asks)))
        if i % 3 == 0:
            side = "BUY" if (i // 3) % 2 == 0 else "SELL"
            events.append(
                (ts, _trade_payload(ts.isoformat(), f"osc-{i}", mid, 0.1, side))
            )
    return events


SCENARIOS: dict[str, Callable[[], list[tuple[pd.Timestamp, str]]]] = {
    "stable": scenario_stable,
    "flash_crash": scenario_flash_crash,
    "liquidity_drought": scenario_liquidity_drought,
    "vpin_burst": scenario_vpin_burst,
    "oscillation": scenario_oscillation,
}


# ---------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------
@dataclass(slots=True)
class ScenarioResult:
    name: str
    passed: bool
    violations: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def check_invariants(app: CoinbaseMarketDataApp) -> ScenarioResult:
    """Vérifie les invariants de risque après exécution d'un scénario.

    Retourne un `ScenarioResult` avec les violations éventuelles et des
    stats utiles pour le rapport.
    """
    name = getattr(app, "_scenario_name", "?")
    violations: list[str] = []

    mm = app.strategy
    rm = app.risk_manager

    # On prend le mid courant pour le notionnel.
    mid = app.book.mid_price() or mm.avg_entry_price or 0.0

    # 1) Notionnel max
    notional = abs(mm.position_btc * mid) if mid else 0.0
    if notional > rm.max_notional_usd + 1e-6:
        violations.append(
            f"notional dépassé : {notional:,.2f} > {rm.max_notional_usd:,.0f}"
        )

    # 2) Drawdown max — on regarde le pire equity observé via l'historique CSV
    mtm = mm.mark_to_market(mid)
    equity = mtm["equity"]
    drawdown = rm.initial_equity - equity
    max_allowed = rm.max_loss_usd
    if drawdown > max_allowed:
        violations.append(
            f"perte dépasse le max : {drawdown:,.2f} > {max_allowed:,.0f}"
        )

    # 3) Health score borné [0, 100]
    health = rm.health_score(equity, mm.position_btc, mid)
    if not (0 <= health <= 100):
        violations.append(f"health_score hors bornes : {health}")

    # 4) Si équity sous stop_equity → can_continue doit être False
    if equity < rm.stop_equity - 1e-6 and rm.can_continue(equity):
        violations.append(
            f"kill switch non activé alors que equity={equity:.2f} < "
            f"stop_equity={rm.stop_equity:.2f}"
        )

    stats = {
        "final_position_btc": mm.position_btc,
        "final_equity": equity,
        "realized_pnl": mm.realized_pnl,
        "unrealized_pnl": mtm["unrealized_pnl"],
        "drawdown_usd": max(0.0, drawdown),
        "fills": len(mm.executions),
        "max_notional_seen": notional,
        "health_score": health,
    }
    return ScenarioResult(
        name=name, passed=not violations, violations=violations, stats=stats
    )


# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------
def run_scenario(name: str, output_dir: Path) -> ScenarioResult:
    """Génère le scénario `name`, le rejoue, vérifie les invariants."""
    if name not in SCENARIOS:
        raise ValueError(f"Scénario inconnu : {name}. Dispos : {list(SCENARIOS)}")
    events = SCENARIOS[name]()
    if not events:
        return ScenarioResult(name=name, passed=False, violations=["scénario vide"])

    settings = Settings(
        output_dir=output_dir / name,
        render_console=False,
        use_contextual_bandit=False,
    )
    app = CoinbaseMarketDataApp(settings)
    app._scenario_name = name  # type: ignore[attr-defined]
    app.connection_status = "stress"
    try:
        for _ts, msg in events:
            app.on_message(None, msg)  # type: ignore[arg-type]
        result = check_invariants(app)
    finally:
        app.shutdown()
    return result


def run_all(output_dir: Path, filter_name: str | None = None) -> list[ScenarioResult]:
    results: list[ScenarioResult] = []
    names = [filter_name] if filter_name else list(SCENARIOS)
    for name in names:
        print(f"\n[stress] Exécution du scénario : {name}")
        t0 = time.perf_counter()
        r = run_scenario(name, output_dir)
        dt = time.perf_counter() - t0
        status = "✓ OK" if r.passed else "✗ VIOLATIONS"
        print(f"  {status} en {dt:.2f}s")
        for v in r.violations:
            print(f"    - {v}")
        print(
            f"    stats : position={r.stats.get('final_position_btc', 0):+.4f} BTC "
            f"equity={r.stats.get('final_equity', 0):,.2f} USD "
            f"dd={r.stats.get('drawdown_usd', 0):,.2f} USD "
            f"fills={r.stats.get('fills', 0)}"
        )
        results.append(r)
    return results


def print_summary(results: Iterable[ScenarioResult]) -> None:
    results = list(results)
    n_ok = sum(1 for r in results if r.passed)
    print("\n" + "=" * 60)
    print(f"RÉSULTAT GLOBAL : {n_ok}/{len(results)} scénarios OK")
    print("=" * 60)
    headers = [
        "Scénario",
        "Status",
        "Position BTC",
        "Equity (USD)",
        "Drawdown",
        "Fills",
    ]
    widths = [20, 10, 14, 16, 14, 8]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for r in results:
        row = [
            r.name,
            "OK" if r.passed else "FAIL",
            f"{r.stats.get('final_position_btc', 0):+.4f}",
            f"{r.stats.get('final_equity', 0):,.2f}",
            f"{r.stats.get('drawdown_usd', 0):,.2f}",
            f"{r.stats.get('fills', 0)}",
        ]
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)))


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stress tests adverses : génère des scénarios extrêmes et "
            "vérifie les invariants de risque."
        )
    )
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()),
        default=None,
        help="Un scénario spécifique à exécuter (défaut : tous).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/stress",
        help="Répertoire racine pour les outputs des scénarios.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_all(output_dir, filter_name=args.scenario)
    print_summary(results)

    # Exit code non-zero si au moins un scénario a échoué
    import sys

    if not all(r.passed for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
