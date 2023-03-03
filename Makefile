# Copyright 2023 OctoML
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
