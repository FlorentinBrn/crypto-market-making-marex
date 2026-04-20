.PHONY: install test run backtest analyze clean

install:
	pip install -e .

test:
	pytest -q

run:
	python -m crypto_mm.main

backtest:
	python -m crypto_mm.backtest --data-dir data

analyze:
	python -m crypto_mm.analyze

clean:
	python -m crypto_mm.clean
