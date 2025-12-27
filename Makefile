# Makefile for lanun

.PHONY: install dev test clean run-all run-case1 run-case2 run-case3 run-case4 help

help:
	@echo "lanun - 2D Lagrangian Transport for Idealized Ocean Basins"
	@echo ""
	@echo "Installation:"
	@echo "  make install     - Install package with pip"
	@echo "  make dev         - Install in development mode with poetry"
	@echo ""
	@echo "Testing:"
	@echo "  make test        - Run all tests"
	@echo "  make test-cov    - Run tests with coverage"
	@echo ""
	@echo "Running simulations:"
	@echo "  make run-all     - Run all test cases"
	@echo "  make run-case1   - Coastal Embayment (Chlorophyll-a)"
	@echo "  make run-case2   - Marginal Sea (DIC)"
	@echo "  make run-case3   - Volcanic Lake (Temperature)"
	@echo "  make run-case4   - Estuary Plume (Sediment)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean       - Remove outputs and cache"

install:
	pip install .

dev:
	poetry install --with dev

test:
	poetry run pytest tests/ -v

test-cov:
	poetry run pytest tests/ -v --cov=lanun --cov-report=term-missing

run-all:
	lanun --all

run-case1:
	lanun case1

run-case2:
	lanun case2

run-case3:
	lanun case3

run-case4:
	lanun case4

clean:
	rm -rf outputs/
	rm -rf logs/
	rm -rf __pycache__/
	rm -rf src/lanun/__pycache__/
	rm -rf src/lanun/core/__pycache__/
	rm -rf src/lanun/io/__pycache__/
	rm -rf src/lanun/utils/__pycache__/
	rm -rf src/lanun/visualization/__pycache__/
	rm -rf .pytest_cache/
	rm -rf *.egg-info/
	rm -rf dist/
	rm -rf build/
	rm -rf __numba_cache__/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".numba_cache" -exec rm -rf {} + 2>/dev/null || true
