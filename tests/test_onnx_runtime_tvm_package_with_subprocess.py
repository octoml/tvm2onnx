import os
import pickle
import tempfile
import typing

import numpy as np
import pytest

import scripts.utils.testing_utils
import tests.test_onnx_runtime_tvm_package
from tests.test_onnx_runtime_tvm_package import *


@scripts.utils.testing_utils.subprocessable
def wrapper_get_ort_output(output_data_file_name, *args, **kwargs) -> None:
    output = scripts.utils.testing_utils.get_ort_output(*args, **kwargs)
    with open(output_data_file_name, "wb") as output_data_file:
        output_data_file.write(pickle.dumps(output))


def wrapper_inference_function(*args, **kwargs) -> typing.List[np.ndarray]:
    with tempfile.TemporaryDirectory() as output_directory:
        output_data_file_name = os.path.join(output_directory, "input_data")
        scripts.utils.testing_utils.run_func_in_subprocess(
            wrapper_get_ort_output,
            output_data_file_name,
            *args,
            **kwargs,
        )

        with open(output_data_file_name, "rb") as output_data_file:
            output = pickle.loads(output_data_file.read())

    return output


@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown_actions():
    default_packaging_function = tests.test_onnx_runtime_tvm_package.packaging_function
    tests.test_onnx_runtime_tvm_package.inference_function = wrapper_inference_function
    yield
    tests.test_onnx_runtime_tvm_package.packaging_function = default_packaging_function
