.PHONY: install lint format-check type test check build smoke

install:
	python -m pip install -e '.[dev]'

lint:
	python -m ruff check .

format-check:
	python -m ruff format --check .

type:
	python -m mypy src/densetopo_unet

test:
	python -m pytest -W error --cov=densetopo_unet --cov-report=term-missing

check: lint format-check type test

build:
	python -m build

smoke:
	bash scripts/smoke_test.sh artifacts/smoke
