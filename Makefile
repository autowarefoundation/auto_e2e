.PHONY: setup test test-unit test-integration

setup:
	python -m pip install -r requirements.txt

test:
	cd Model && python -m pytest tests/ -v

test-unit:
	cd Model && python -m pytest tests/ -v -m "not integration"

test-integration:
	cd Model && python -m pytest tests/ -v -m "integration"
