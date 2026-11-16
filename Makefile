lint:
	python -m ruff check .

fmt:
	python -m ruff format .

test:
	pytest
