# Crypto Market Making Research Sandbox

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#installation)
[![Tests](https://img.shields.io/badge/tests-58%20passed-brightgreen.svg)](#testing)
[![Style](https://img.shields.io/badge/focus-market%20microstructure%20%7C%20backtesting%20%7C%20risk-informational.svg)](#project-highlights)

A **crypto market making research sandbox** built around **Coinbase BTC-USD public market data**.

This repository is designed as a minimal but credible quant trading project: it maintains a local level-2 order book, computes microstructure features in real time, runs an inventory-aware quoting engine, simulates fills from public trades, and supports **historical replay, stress testing, and walk-forward evaluation workflows**.

It is positioned between an interview exercise and a research prototype: the codebase aims to be readable, testable, and extensible rather than overly optimized or exchange-specific.

---

## Project Highlights

- **Live public market data ingestion** via Coinbase websocket channels
- **Local L2 order book reconstruction** with best bid / ask tracking
- **Microstructure features** including imbalance, OFI, microprice, VPIN proxy, and fast volatility
- **Inventory-aware market making logic** with adaptive spread control
- **Execution simulation layer** using queue-share heuristics on public trades
- **Risk controls** including kill switch, inventory limits, and reduce-only mode
- **Historical replay engine** for deterministic offline experiments
- **Stress testing framework** for spread shocks, volatility bursts, toxic flow, and inventory pressure
- **Walk-forward backtesting** for realistic offline evaluation on stored data
- **Analytics and diagnostics** exported to CSV and plots
- **Terminal monitoring tools** for observing behavior during runs
- **Strong test coverage** with a lightweight and reproducible local setup

---

## Why this repository is interesting

This project combines:

1. **Market data engineering**: websocket ingestion, state updates, persistence  
2. **Microstructure-driven signals**: order flow and short-term pressure indicators  
3. **Execution logic**: fill simulation and quote placement rules  
4. **Risk management**: inventory and kill-switch safeguards  
5. **Research workflow**: replay, stress tests, backtest, analytics, diagnostics  

---

## Repository Structure

```text
crypto_mm/
├── analytics.py        # performance metrics and reporting helpers
├── analyze.py          # post-run analytics entrypoint
├── backtest.py         # historical replay and walk-forward evaluation
├── bench.py            # latency and local performance utilities
├── clean.py            # remove generated data and cache folders
├── config.py           # model and runtime configuration
├── console.py          # terminal monitoring tools
├── feed.py             # websocket ingestion and market data feed
├── learning.py         # adaptive / learning-based components
├── main.py             # live simulation entrypoint
├── models.py           # core dataclasses / state containers
├── orderbook.py        # local order book reconstruction
├── plots.py            # chart generation utilities
├── replay.py           # historical data replay helpers
├── risk.py             # inventory and risk controls
├── storage.py          # CSV persistence layer
├── strategy.py         # quoting and spread logic
├── stress.py           # scenario stress testing engine
└── utils.py            # shared helpers

tests/
└── ...                 # unit tests for core modules
```

---

## Installation

### 1) Clone the repository

```bash
git clone https://github.com/FlorentinBrn/crypto-market-making-marex.git
cd crypto_mm_exercise
```

### 2) Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

or in editable mode:

```bash
pip install -e .
```

---

## Quick Start

### Live simulation

```bash
python -m crypto_mm.main
```

### Live simulation with adaptive module enabled

```bash
python -m crypto_mm.main --enable-bandit
```

### Live simulation with adaptive module enabled

```bash
python -m crypto_mm.main --bench-latency
```

### Run backtests on stored data

```bash
python -m crypto_mm.backtest --data-dir data
```

### Replay historical sessions

```bash
python -m crypto_mm.replay --data-dir data --speed 10
```

### Run stress scenarios

```bash
python -m crypto_mm.stress
```

### Run latency analysis

```bash
python -m crypto_mm.bench --data-dir data
```

### Run post-trade analytics

```bash
python -m crypto_mm.analyze
```

### Clean generated data and cache folders

```bash
python -m crypto_mm.clean
```

or through the installed console script:

```bash
crypto-mm-clean
```

This command removes generated local folders such as `data/`, `.pytest_cache/`, and nested `__pycache__/` directories.

---

## Testing

The repository ships with a test suite covering the main building blocks.

```bash
pytest -q
```

Current local status on the packaged version: **58 tests passed**.

---

## Research Workflow

A typical workflow with this repository is:

1. **Collect or replay market data**
2. **Run the quoting engine** with a chosen configuration
3. **Store fills, state, spreads, and PnL paths**
4. **Replay sessions deterministically** to inspect behavior
5. **Stress the strategy** under adverse scenarios
6. **Analyze outcomes** through CSV exports and plots
7. **Tune parameters** using walk-forward evaluation

This makes the project suitable for experimenting with:

- spread adaptation rules
- inventory penalties
- fill assumptions
- volatility-aware quoting
- toxicity filters
- stress robustness
- contextual bandits or lightweight RL extensions

---

## Stress Testing Examples

Example scenarios that can be simulated:

- sudden volatility expansion
- one-sided aggressive flow
- widening spreads / thin liquidity
- persistent trend regime
- inventory trapped near limits
- degraded fills or delayed executions

This is useful to evaluate whether a profitable strategy remains robust when market conditions deteriorate.

---

## Replay Engine Benefits

Historical replay allows:

- deterministic debugging
- side-by-side parameter comparison
- reproducible PnL analysis
- quote behavior inspection
- signal validation on past sessions

Replay is often the fastest path to improving execution logic before testing live.

---

## Example Outputs

The project can generate artifacts such as:

- PnL curves from simulated fills
- spread history charts
- walk-forward equity curves
- stress scenario comparisons
- latency distributions
- replay diagnostics
- microstructure state histories

These outputs are useful both for debugging and for presenting research results.

---

## Engineering Notes

The codebase favors:

- explicit module boundaries
- simple dependencies
- testable components
- local reproducibility
- readable implementation over unnecessary abstraction

This is deliberate. In a trading context, clarity and debuggability are often more valuable than cleverness.

---

## Limitations

This repository is a **research and educational project**.

In particular, it does **not** claim to provide:

- real exchange connectivity for order routing
- production-grade latency guarantees
- exact queue position modeling
- slippage and fee realism across all market regimes
- compliance, monitoring, or deployment hardening

---

## Ideas for Future Improvements

- richer queue position modeling
- explicit maker / taker fee schedules
- multi-asset or cross-venue support
- stronger regime detection features
- more realistic event-driven replay engine
- Monte Carlo stress scenario generator
- experiment tracking for parameter sweeps
