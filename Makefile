.PHONY: setup test test-map test-local benchmark

setup:
	pip install torch timm pytest

test:
	python -m pytest Model/tests -v

# map_rendering tests need extra deps not used by CI: pip install matplotlib osmnx
# (run from Model/ so `data_parsing.*` imports resolve — the package has no __init__.py)
test-map:
	cd Model && python -m pytest data_parsing/map_rendering -v

# everything runnable on a dev machine: unit tests + map_rendering
test-local:
	cd Model && python -m pytest tests data_parsing/map_rendering -v

benchmark:
	cd Model/speed_benchmark && python speed_benchmark.py
