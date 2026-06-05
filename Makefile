.PHONY: setup test benchmark

setup:
	pip install torch timm pytest

test:
	cd Model/tests && python -m pytest test_auto_e2e.py -v

benchmark:
	cd Model/speed_benchmark && python speed_benchmark.py
