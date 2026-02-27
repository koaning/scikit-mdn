install:
	uv venv
	uv pip install -e ".[dev]"

build:
	uvx mobuild export nbs skmdn

pypi:
	uvx mobuild export nbs skmdn
	uv build
	uv publish

test:
	pytest
