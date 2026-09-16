run:
	uv run python -m src 

install:
	uv sync

debug:
		python3 -m pdb main.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .mypy_cache */.mypy_cache

lint:
	uv run python -m flake8 src
	uv run python -m mypy src --warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

.PHONY: run install debug clean lint
