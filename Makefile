.PHONY: format lint test

format:
	isort --settings-file=/usr/tvm2onnx/pyproject.toml tvm2onnx
	black --config=/usr/tvm2onnx/pyproject.toml        tvm2onnx

lint: # Checks for correct formatting and style.
	isort  --settings-file=/usr/tvm2onnx/pyproject.toml --check-only tvm2onnx
	black  --config=/usr/tvm2onnx/pyproject.toml        --check      tvm2onnx
	flake8 --config=/usr/tvm2onnx/.flake8                            tvm2onnx
	mypy   --config-file=/usr/tvm2onnx/.mypy.ini                     tvm2onnx

test:
	pytest --durations 0 /usr/tvm2onnx/tests
