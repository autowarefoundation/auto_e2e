.DEFAULT_GOAL := help

PYTEST = python -m pytest

# --- Setup -------------------------------------------------------------------

setup: ## install core dependencies
	pip install torch timm pytest ruff

setup-map: ## extra deps for the map_rendering tests (not installed in CI)
	pip install matplotlib osmnx

setup-local: setup setup-map ## full dev setup

# --- Checks ------------------------------------------------------------------

lint: ## ruff over the whole repo (same as CI)
	ruff check

test: ## unit tests (same selection as CI)
	$(PYTEST) Model/tests -v

# run from Model/ so `data_parsing.*` imports resolve (the package has no __init__.py)
test-map: ## map_rendering tests (deps via setup-map)
	cd Model && $(PYTEST) data_parsing/map_rendering -v

test-local: test test-map ## everything runnable on a dev machine

ci: lint test ## exactly what CI runs

# --- Run ---------------------------------------------------------------------

benchmark: ## speed benchmark
	cd Model/speed_benchmark && python speed_benchmark.py

help: ## list available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

.PHONY: setup setup-map setup-local lint test test-map test-local ci benchmark help
