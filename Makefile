run:
	uv run python -m src 

install:
	uv sync

debug:
		python3 -m pdb main.py

clean:
	rm -rf __pycache__ */__pycache__
	rm -rf .mypy_cache */.mypy_cache

lint:
	uv run python -m flake8 .
	uv run python -m mypy .	--warn-return-any \
	 			--warn-unused-ignores \
	 			--ignore-missing-imports \
	 			--disallow-untyped-defs \
	 			--check-untyped-defs

.PHONY: run install debug clean lint
