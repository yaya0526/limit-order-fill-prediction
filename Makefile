PYTHON ?= python3

.PHONY: install check test demo ci

install:
	$(PYTHON) -m pip install -r requirements-dev.txt

check:
	$(PYTHON) scripts/check_repository.py

test:
	$(PYTHON) -m pytest -q

demo:
	$(PYTHON) scripts/demo_results.py

ci: check test demo
