.PHONY: install dev test coverage lint backtest paper dashboard \
        freqtrade-start freqtrade-stop freqtrade-backtest \
        freqtrade-lookahead freqtrade-recursive clean

# ── Install ────────────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

dev:
	pip install -r requirements.txt && pip install -e ".[dev]"

# ── Test (TDD — RED/GREEN/REFACTOR cycle) ─────────────────────────────────────
test:
	python -m pytest tests/

# 80%+ coverage per TDD SKILL.md requirements
coverage:
	python -m pytest tests/ --cov=src --cov-report=term-missing \
	  --cov-fail-under=80

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

# ── Step 8: Snapshot export for GitHub Pages ──────────────────────────────────
snapshot:
	PYTHONPATH=src LLM_PROVIDER=none python -m dashboard.snapshot_export --mode demo

snapshot-live:
	PYTHONPATH=src LLM_PROVIDER=none FREQTRADE_MODE=off \
	  python -m dashboard.snapshot_export --mode cycle --symbol BTC/USDT

# ── Portfolio config (validate the multi-sleeve YAML) ─────────────────────────
portfolio:
	PYTHONPATH=src python -m strategies.portfolio_config

# ── Step 6: Backtest + Monte Carlo ────────────────────────────────────────────
backtest:
	PYTHONPATH=src python src/backtesting/backtest.py --days 30 --sims 1000

backtest-full:
	PYTHONPATH=src python src/backtesting/backtest.py --days 90 --sims 5000 \
	  --save-signals

monte-carlo:
	PYTHONPATH=src python src/backtesting/monte_carlo.py \
	  --trades data/backtest_signals.json --sims 5000 --capital 1000 \
	  --p95-dd-gate 0.30

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

freqtrade-backtest:
	$(FT_VENV)/bin/freqtrade backtesting \
	  --config $(FT_CFG) --strategy SuperBotFollower \
	  --userdir $(FT_DATA)/user_data \
	  --timerange 20240901-20241201 \
	  --export trades

# ── Step 6: FreqTrade anti-bias CI gates (from research — RUN BEFORE DRY-RUN) ─
freqtrade-lookahead:
	@echo "=== Lookahead Analysis (detects future-data leakage) ==="
	$(FT_VENV)/bin/freqtrade lookahead-analysis \
	  --config $(FT_CFG) --strategy SuperBotFollower \
	  --userdir $(FT_DATA)/user_data \
	  --timerange 20240901-20241201

freqtrade-recursive:
	@echo "=== Recursive Analysis (detects indicator instability) ==="
	$(FT_VENV)/bin/freqtrade recursive-analysis \
	  --config $(FT_CFG) --strategy SuperBotFollower \
	  --userdir $(FT_DATA)/user_data \
	  --startup-candle 199 299 399 499

# ── Clean ──────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	rm -rf .pytest_cache htmlcov .coverage
