.PHONY: install test run dashboard backtest replay stress bench analyze clean

install:
	pip install -e .

test:
	pytest -q

run:
	python -m crypto_mm.main

dashboard:
	python -m crypto_mm.ui.dash_app --data-dir data

backtest:
	python -m crypto_mm.tools.backtest --data-dir data

replay:
	python -m crypto_mm.data.replay --data-dir data --speed 10

stress:
	python -m crypto_mm.tools.stress

bench:
	python -m crypto_mm.tools.bench --data-dir data

analyze:
	python -m crypto_mm.tools.analyze --data-dir data

clean:
	python -m crypto_mm.tools.clean
