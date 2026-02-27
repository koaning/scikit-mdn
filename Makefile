install:
	uv venv
	uv pip install -e ".[dev]"

pypi:
	uv build
	uv publish

test:
	pytest
