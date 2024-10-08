install:
	python -m pip install --upgrade pip
	python -m pip install -e ".[dev]"
	python -m pip install wheel twine

pypi:
	rm -rf dist build
	python setup.py sdist
	python setup.py bdist_wheel --universal
	twine upload dist/*

test:
	pytest