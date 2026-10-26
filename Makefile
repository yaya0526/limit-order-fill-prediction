PYTHON ?= python3

.PHONY: install check test ci

install:
	$(PYTHON) -m pip install -r requirements-dev.txt

check:
	$(PYTHON) scripts/check_repository.py

test:
	$(PYTHON) -m pytest -q

ci: check test
