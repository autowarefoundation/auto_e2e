.DEFAULT_GOAL := help

PYTEST = python -m pytest
# pytorch wheel channel: cpu (default, CI parity) or a CUDA build such as cu118/cu121
TORCH_CHANNEL ?= cpu
# suite for test-local: all | map | integration
SUITE ?= all

# --- Setup -------------------------------------------------------------------

setup: ## pinned deps from requirements.txt (make setup TORCH_CHANNEL=cu118 for CUDA)
	pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/$(TORCH_CHANNEL)

setup-map: ## extra deps for the map_rendering tests (not installed in CI)
	pip install matplotlib osmnx

setup-local: setup setup-map ## full dev setup

# --- Checks ------------------------------------------------------------------

# run setup first if deps are missing (presence check only, versions not verified)
deps:
	@{ python -c "import torch, timm, pytest" && command -v ruff mypy; } >/dev/null 2>&1 || $(MAKE) setup

deps-map:
	@python -c "import matplotlib, osmnx" >/dev/null 2>&1 || $(MAKE) setup-map

lint: deps ## ruff over the whole repo (same as CI)
	ruff check

# Static type-checking over the whole project (see pyproject.toml for the
# lenient/strict policy). Run from Model/ so module paths resolve cleanly.
typecheck: deps ## mypy over the project (same as CI)
	cd Model && mypy .

test: deps ## unit tests (same selection as CI)
	$(PYTEST) Model/tests -v

# map suite runs from Model/ so `data_parsing.*` imports resolve (no __init__.py).
# The integration suite is slow and downloads pretrained backbone weights on first run.
test-local-map: deps deps-map
	cd Model && $(PYTEST) data_parsing/map_rendering -v

test-local-integration: deps
	$(PYTEST) Model/tests -m integration -v

test-local-all: test test-local-map test-local-integration

test-local: test-local-$(SUITE) ## local tests (make test-local SUITE=all|map|integration)

ci: lint typecheck test ## exactly what CI runs

# --- Run ---------------------------------------------------------------------

benchmark: ## speed benchmark
	cd Model/speed_benchmark && python speed_benchmark.py

help: ## list available targets
	@echo "Getting started (activate your virtualenv first — targets use the active python/pip):"
	@echo "  make setup-local                  install everything for local dev (CPU torch wheels)"
	@echo "  make setup TORCH_CHANNEL=cu121    pinned deps with CUDA 12.1 torch wheels instead"
	@echo "  make test-local                   run all local tests (unit + map + integration)"
	@echo "  make test-local SUITE=map         run a single suite (all | map | integration)"
	@echo "  make ci                           run exactly what CI runs (lint + unit tests)"
	@echo ""
	@echo "Targets:"
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Defaults: TORCH_CHANNEL=$(TORCH_CHANNEL) (cpu | cu118 | cu121 | ...), SUITE=$(SUITE) (all | map | integration), PYTEST='$(PYTEST)'"

.PHONY: setup setup-map setup-local deps deps-map lint typecheck test test-local test-local-all test-local-map test-local-integration ci benchmark help
