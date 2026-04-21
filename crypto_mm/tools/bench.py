"""Mesure de latence du chemin critique.

Ce module propose deux usages :

1. ``LatencyRecorder`` : instrumente le handler WebSocket pour mesurer
   le temps passé dans chaque étape (parse JSON, apply book updates,
   compute signals, update_quotes, on_trade, render dashboard). Les
   mesures sont stockées dans des deques tournantes et écrites
   périodiquement en CSV. Coût runtime : appel à ``time.perf_counter_ns``
   plus une insertion deque, négligeable.

2. Analyse post-run via ``python -m crypto_mm.tools.bench`` : lit le CSV
   produit, calcule p50 / p95 / p99 / max par étape, et génère un graphe
   de distribution.
"""

from __future__ import annotations

import argparse
import csv
import threading
import time
from collections import deque
from pathlib import Path
from typing import Iterator

# Nombre de mesures conservées en mémoire par étape avant flush.
DEFAULT_BUFFER = 10_000

# Étapes mesurées dans le chemin critique.
STAGE_PARSE = "parse_json"
STAGE_APPLY_BOOK = "apply_book"
STAGE_COMPUTE_SIGNALS = "compute_signals"
STAGE_UPDATE_QUOTES = "update_quotes"
STAGE_ON_TRADE = "on_trade"
STAGE_RENDER = "render_dashboard"
STAGE_TOTAL_L2 = "total_l2_handler"
STAGE_TOTAL_TRADE = "total_trade_handler"


class LatencyRecorder:
    """Enregistre des durées d'étapes en nanosecondes.

    Contrat :
    - ``record(stage, duration_ns)`` : push une mesure, O(1).
    - ``start()`` : retourne `perf_counter_ns()` — sucre syntaxique pour
      mesurer depuis l'appelant (évite une closure).
    - ``flush(path)`` : écrit toutes les mesures accumulées dans un CSV,
      une ligne par mesure (stage, duration_ns, ts_ns), puis vide les
      buffers.

    Thread-safe via un Lock unique (contention négligeable comparée à
    perf_counter_ns).
    """

    def __init__(self, enabled: bool = True, buffer_size: int = DEFAULT_BUFFER) -> None:
        self.enabled = enabled
        self.buffer_size = buffer_size
        self._buffers: dict[str, deque[tuple[int, int]]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def start() -> int:
        """Retourne `perf_counter_ns()` au moment de l'appel."""
        return time.perf_counter_ns()

    def record(self, stage: str, start_ns: int) -> None:
        """Enregistre la durée `now - start_ns` pour `stage`.

        Noop si `enabled=False` — le coût est alors juste le test booléen.
        """
        if not self.enabled:
            return
        now = time.perf_counter_ns()
        duration = now - start_ns
        with self._lock:
            buf = self._buffers.get(stage)
            if buf is None:
                buf = deque(maxlen=self.buffer_size)
                self._buffers[stage] = buf
            buf.append((now, duration))

    def samples(self, stage: str) -> list[int]:
        """Retourne la liste des durées mesurées pour `stage`."""
        with self._lock:
            buf = self._buffers.get(stage)
            return [d for _, d in buf] if buf else []

    def stages(self) -> list[str]:
        with self._lock:
            return list(self._buffers.keys())

    def flush(self, output_path: Path) -> None:
        """Écrit toutes les mesures accumulées dans un CSV puis vide."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            rows: list[tuple[str, int, int]] = []
            for stage, buf in self._buffers.items():
                for ts_ns, dur_ns in buf:
                    rows.append((stage, ts_ns, dur_ns))
                buf.clear()
        if not rows:
            return
        mode = "a" if output_path.exists() else "w"
        with output_path.open(mode, newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if mode == "w":
                w.writerow(["stage", "ts_ns", "duration_ns"])
            for r in rows:
                w.writerow(r)


# ---------------------------------------------------------------------
# Analyse offline
# ---------------------------------------------------------------------
def _percentile(sorted_vals: list[int], p: float) -> float:
    """Percentile linéaire sur une liste triée (p in [0, 100])."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return float(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac)


def summarize_csv(csv_path: Path) -> dict[str, dict[str, float]]:
    """Retourne {stage: {count, mean_us, p50_us, p95_us, p99_us, max_us}}."""
    by_stage: dict[str, list[int]] = {}
    with Path(csv_path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stage = row["stage"]
            dur = int(row["duration_ns"])
            by_stage.setdefault(stage, []).append(dur)

    summary: dict[str, dict[str, float]] = {}
    for stage, vals in by_stage.items():
        vals_sorted = sorted(vals)
        n = len(vals_sorted)
        mean_ns = sum(vals_sorted) / n if n else 0.0
        summary[stage] = {
            "count": float(n),
            "mean_us": mean_ns / 1000.0,
            "p50_us": _percentile(vals_sorted, 50) / 1000.0,
            "p95_us": _percentile(vals_sorted, 95) / 1000.0,
            "p99_us": _percentile(vals_sorted, 99) / 1000.0,
            "max_us": float(vals_sorted[-1]) / 1000.0 if n else 0.0,
        }
    return summary


def print_summary(summary: dict[str, dict[str, float]]) -> None:
    """Imprime un tableau lisible sur stdout."""
    if not summary:
        print("Aucune donnée de latence.")
        return
    # Tri par p95 descendant pour voir les plus coûteux en haut.
    stages = sorted(summary.keys(), key=lambda s: -summary[s]["p95_us"])
    headers = ["stage", "count", "mean µs", "p50 µs", "p95 µs", "p99 µs", "max µs"]
    widths = [max(len(h), 18) for h in headers]
    widths[0] = max(widths[0], max(len(s) for s in stages))
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("-" * len(line))
    for stage in stages:
        s = summary[stage]
        row = [
            stage,
            f"{int(s['count']):,}",
            f"{s['mean_us']:.2f}",
            f"{s['p50_us']:.2f}",
            f"{s['p95_us']:.2f}",
            f"{s['p99_us']:.2f}",
            f"{s['max_us']:.2f}",
        ]
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)))


def plot_distributions(csv_path: Path, output_path: Path) -> None:
    """Trace un boxplot log-scale des latences par étape."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_stage: dict[str, list[int]] = {}
    with Path(csv_path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stage = row["stage"]
            dur = int(row["duration_ns"])
            by_stage.setdefault(stage, []).append(dur / 1000.0)  # en µs

    if not by_stage:
        return
    # Tri par médiane
    stages = sorted(
        by_stage.keys(), key=lambda s: sorted(by_stage[s])[len(by_stage[s]) // 2]
    )
    data = [by_stage[s] for s in stages]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.boxplot(data, labels=stages, vert=True, showfliers=False)
    ax.set_yscale("log")
    ax.set_ylabel("Latence (µs, échelle log)")
    ax.set_title("Distribution des latences par étape du chemin critique")
    ax.grid(True, axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def iter_stages_in_order() -> Iterator[str]:
    """Ordre d'affichage canonique (du plus bas niveau au plus haut)."""
    yield from (
        STAGE_PARSE,
        STAGE_APPLY_BOOK,
        STAGE_COMPUTE_SIGNALS,
        STAGE_UPDATE_QUOTES,
        STAGE_ON_TRADE,
        STAGE_RENDER,
        STAGE_TOTAL_L2,
        STAGE_TOTAL_TRADE,
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse de latence post-run à partir de data/bench/latency.csv."
    )
    parser.add_argument("--data-dir", default="data", help="Racine des CSV live.")
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Ne pas générer le PNG (juste l'affichage texte).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    csv_path = data_dir / "bench" / "latency.csv"
    if not csv_path.exists():
        print(f"Fichier introuvable : {csv_path}")
        print("Relance le live avec --bench-latency pour produire ces données.")
        return

    summary = summarize_csv(csv_path)
    print_summary(summary)

    if not args.no_plot:
        out_png = data_dir / "bench" / "latency_distributions.png"
        try:
            plot_distributions(csv_path, out_png)
            print(f"\nGraphe : {out_png}")
        except Exception as e:
            print(f"\n(Graphe non généré : {e})")


if __name__ == "__main__":
    main()
