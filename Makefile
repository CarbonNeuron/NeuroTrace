.PHONY: setup test test-quick lint trace-demo clean

setup:
	uv sync --all-extras

test:
	uv run pytest tests/ -v

test-quick:
	uv run pytest tests/ -v -m "not model_download"

lint:
	uv run ruff check src/ tests/

trace-demo:
	uv run neurotrace trace --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --prompt "The capital of France is" --db demo.db
	uv run neurotrace list --db demo.db
	uv run neurotrace inspect --db demo.db --trace-id latest

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	rm -f *.db
