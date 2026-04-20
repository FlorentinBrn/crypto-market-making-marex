# Screenshots for README

Place PNG screenshots here with the following names (referenced in the
main README):

| Filename | What to capture |
|---|---|
| `terminal_dashboard.png` | The Rich dashboard in a terminal during a live run (after a few minutes of data accumulation). |
| `dash_dashboard.png` | The Dash web dashboard at `http://localhost:8050` with equity curve, metric cards, microstructure plot, order book table visible. |
| `walkforward_results.png` | Terminal output of `python -m crypto_mm.tools.backtest` showing the metrics table, OR `data/backtests/walkforward_metrics.png`. |
| `stress_summary.png` | Terminal output of `python -m crypto_mm.tools.stress` with the summary table at the end. |
| `latency_distributions.png` | The file generated at `data/bench/latency_distributions.png` after running with `--bench-latency`. |

Any PNG, JPG, or SVG works. Resolution should be readable on a typical
GitHub preview (1200 px wide is fine).
