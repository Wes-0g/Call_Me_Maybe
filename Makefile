run:
	python3 main.py

install:
	python3 -m

debug:
		python3 -m pdb main.py

clean:
	rm -rf __pycache__ */__pycache__
	rm -rf .mypy_cache */.mypy_cache

lint:
	python3 -m flake8 .
	python3.10 -m mypy . --warn-return-any \
	 					 --warn-unused-ignores \
	 					 --ignore-missing-imports \
	 					 --disallow-untyped-defs \
	 					 --check-untyped-defs
lint-strict:
	python3 -m mypy . --strict

.PHONY: run install debug clean lint
