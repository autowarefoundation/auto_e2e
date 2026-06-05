.PHONY: setup test

setup:
	pip install torch timm pytest

test:
	cd Model/tests && python -m pytest test_auto_e2e.py -v
