# Directories containing Python code to format and lint.
PYTHON_DIRECTORIES := tvm2onnx tests scripts

format: # Formats the Python code.
	isort --settings-file=pyproject.toml ${PYTHON_DIRECTORIES}
	black --config=pyproject.toml        ${PYTHON_DIRECTORIES}

lint: # Checks the Python code for correct formatting and style.
	isort  --settings-file=pyproject.toml --check-only ${PYTHON_DIRECTORIES}
	black  --config=pyproject.toml        --check      ${PYTHON_DIRECTORIES}
	flake8 --config=.flake8                            ${PYTHON_DIRECTORIES}
	mypy   --config-file=.mypy.ini                     ${PYTHON_DIRECTORIES}

test: # Runs the unit and integration tests.
	pytest --durations 0 tests

test-slow: # Runs the unit and integration tests.
	pytest --runslow --durations 0 tests

.PHONY: format lint test
