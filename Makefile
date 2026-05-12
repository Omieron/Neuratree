VENV   = .venv
PYTHON = $(VENV)/bin/python
PIP    = $(VENV)/bin/pip

.DEFAULT_GOAL := help

# ── Setup ──────────────────────────────────────────────────────────────────

.PHONY: install
install: ## Create venv and install the package in editable mode
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip --quiet
	$(PIP) install -e . --quiet
	@# Python 3.14 workaround: generated entry scripts don't load .pth files correctly,
	@# so we replace the script with a shell wrapper that uses the -m flag instead.
	@printf '#!/bin/sh\nexec "$$(dirname "$$0")/python3.14" -m dnt.cli "$$@"\n' \
		> $(VENV)/bin/dnt && chmod +x $(VENV)/bin/dnt
	@echo ""
	@echo "✓  Ready. Activate the environment with:"
	@echo "   source $(VENV)/bin/activate"
	@echo ""
	@echo "Then use:  dnt dashboard | dnt demo | dnt bench | dnt test"

.PHONY: install-anthropic
install-anthropic: ## Also install the optional Anthropic provider
	$(PIP) install -e ".[anthropic]" --quiet
	@echo "✓  anthropic provider installed"

# ── Run ────────────────────────────────────────────────────────────────────

.PHONY: run
run: ## Start the Streamlit dashboard
	$(VENV)/bin/streamlit run dnt/ui/dashboard.py

.PHONY: bench
bench: ## Run the RAG vs DNT token benchmark
	$(PYTHON) benchmarks/token_benchmark.py

.PHONY: demo
demo: ## Quick interactive demo (no API key required)
	$(PYTHON) -m dnt.cli demo

# ── Test ───────────────────────────────────────────────────────────────────

.PHONY: test
test: ## Run the full test suite
	$(VENV)/bin/pytest tests/ -v

.PHONY: test-fast
test-fast: ## Run tests without verbose output
	$(VENV)/bin/pytest tests/ -q

# ── Cleanup ────────────────────────────────────────────────────────────────

.PHONY: clean
clean: ## Remove venv and all cache files
	rm -rf $(VENV) dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	@echo "✓  Cleaned"

# ── Help ───────────────────────────────────────────────────────────────────

.PHONY: help
help: ## Show this help message
	@echo ""
	@echo "  Neuratree — DNT project commands"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36mmake %-18s\033[0m %s\n", $$1, $$2}'
	@echo ""
