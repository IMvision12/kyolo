.DEFAULT_GOAL := help

PYTEST := python -m pytest

.PHONY: help
help:
	@echo ""
	@echo " kyolo — developer commands"
	@echo " ────────────────────────────────────────────────"
	@echo " make install            Editable install with dev deps"
	@echo " make lint               Ruff lint + format check"
	@echo " make format             Ruff auto-format"
	@echo " make test               Run the test suite (jax backend)"
	@echo " make test-torch         Run tests on the torch backend"
	@echo " make test-tf            Run tests on the tensorflow backend"
	@echo " make build              Build wheel + sdist"
	@echo " make release-check      Print the version that would be released"
	@echo " make clean              Remove build/test artifacts"
	@echo ""

.PHONY: install
install:
	pip install -e ".[tests,viz]"
	pip install -r dev-requirements.txt

.PHONY: lint
lint:
	ruff check .
	ruff format --check .

.PHONY: format
format:
	ruff check --fix .
	ruff format .

.PHONY: test
test:
	KERAS_BACKEND=jax $(PYTEST) tests/ -v --durations=20 -m "not slow and not gpu"

.PHONY: test-torch
test-torch:
	KERAS_BACKEND=torch $(PYTEST) tests/ -v -m "not slow and not gpu"

.PHONY: test-tf
test-tf:
	KERAS_BACKEND=tensorflow $(PYTEST) tests/ -v -m "not slow and not gpu"

.PHONY: build
build:
	python -m pip install --upgrade build
	python -m build

.PHONY: release-check
release-check:
	@python -c "from kyolo.version import __version__; print('kyolo', __version__)"

.PHONY: clean
clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
