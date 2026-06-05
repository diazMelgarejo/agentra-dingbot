.PHONY: install dev test coverage lint backtest paper freqtrade-start freqtrade-stop clean

# ── Install ────────────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

dev:
	pip install -r requirements.txt && pip install -e ".[dev]"

# ── Test (TDD) ─────────────────────────────────────────────────────────────────
test:
	python -m pytest tests/

# 80%+ coverage per TDD SKILL.md requirements
coverage:
	python -m pytest tests/ --cov=src --cov-report=term-missing \
	  --cov-fail-under=80

# RED phase helper: run only the new failing tests
red:
	python -m pytest tests/$(FILE) -v --tb=short

# ── Lint ───────────────────────────────────────────────────────────────────────
lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

# ── Run ────────────────────────────────────────────────────────────────────────
run:
	PYTHONPATH=src LLM_PROVIDER=none python src/core/cli.py run --symbol BTC/USDT

paper:
	PYTHONPATH=src LLM_PROVIDER=none python src/deploy/live.py --paper

dashboard:
	PYTHONPATH=src LLM_PROVIDER=none python src/dashboard/app.py

# ── Backtest & MC ──────────────────────────────────────────────────────────────
backtest:
	PYTHONPATH=src python src/backtesting/backtest.py --days 30

monte-carlo:
	PYTHONPATH=src python src/backtesting/monte_carlo.py \
	  --trades data/signals.json --sims 5000 --capital 1000 --p95-dd-gate 0.30

# ── FreqTrade optional sidecar ─────────────────────────────────────────────────
FT_VENV  = $(HOME)/.venvs/freqtrade
FT_DATA  = $(HOME)/freqtrade-data
FT_CFG   = $(FT_DATA)/user_data/config.json

freqtrade-start:
	$(FT_VENV)/bin/freqtrade trade \
	  --config $(FT_CFG) --strategy SuperBotFollower \
	  --userdir $(FT_DATA)/user_data

freqtrade-stop:
	pkill -f "freqtrade trade" || true

# ── Clean ──────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	rm -rf .pytest_cache htmlcov .coverage
