.PHONY: demo eval test lint typecheck clean

eval:
	uv run invoice-extractor eval --count 120 --seed 42

demo: eval
	uv run invoice-extractor extract data/generated/pdfs/001_standard_grid.pdf

test:
	uv run pytest -q

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

typecheck:
	uv run mypy src tests

clean:
	rm -rf data/generated reports .pytest_cache .mypy_cache .ruff_cache
