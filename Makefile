.PHONY: install dev test lint backtest paper clean

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements.txt && pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check . && ruff format --check .

backtest:
	python backtesting/backtest.py --days 30

paper:
	python deploy/live.py --paper

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	rm -rf .pytest_cache htmlcov .coverage dist build
