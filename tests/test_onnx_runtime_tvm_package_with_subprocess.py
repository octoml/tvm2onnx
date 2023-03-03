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

import os
import pickle
import tempfile
import typing

import numpy as np
import pytest

import scripts.utils.testing_utils
import tests.test_onnx_runtime_tvm_package
from tests.test_onnx_runtime_tvm_package import *  # noqa: F403, F401


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
