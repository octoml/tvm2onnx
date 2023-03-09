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

"""
This script is a utility to run benchmarks of ONNX wrapped TVM models.
Generate these ONNX wrapped TVM models via the onnx_package.py script.
"""

import argparse
import logging
import os
import re
import tarfile
import tempfile

import numpy as np
import onnx
import onnxruntime

logging.basicConfig(level=logging.ERROR)


def find(pattern, path):
    result = []
    regex = re.compile(pattern)
    for root, _, files in os.walk(path):
        for name in files:
            if regex.match(name):
                result.append(os.path.join(root, name))
    return result


def generate_input_data(
    input_shapes,
    input_dtypes,
):
    data = {}
    for name in input_shapes.keys():
        shape = input_shapes[name]
        dtype = input_dtypes[name]
        d = np.random.uniform(size=shape).astype(dtype)
        data[name] = d
    return data


def infer(
    model_path,
):
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(model_path, "r") as tar:
            tar.extractall(tmp)
        files = find(".*\\.onnx$", tmp)
        if len(files) < 1:
            print("No onnx model found")
            exit(-1)
        elif len(files) > 1:
            print("Multiple onnx models found")
            exit(-1)
        onnx_path = files[0]
        custom_op_libs = find("^custom_.*\\.(so|dll|dynlib)$", tmp)

        onnx_model = onnx.load_model(onnx_path)
        input_shapes = {}
        input_dtypes = {}
        for inp in onnx_model.graph.input:
            name = inp.name
            shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            input_shapes[name] = shape
            dtype = inp.type.tensor_type.elem_type
            input_dtypes[name] = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[dtype]

        sess_options = onnxruntime.SessionOptions()
        for custom_op_lib in custom_op_libs:
            sess_options.register_custom_ops_library(custom_op_lib)

        engine = onnxruntime.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"],
            provider_options=[{}],
            sess_options=sess_options,
        )

        input_data = generate_input_data(
            input_shapes=input_shapes, input_dtypes=input_dtypes
        )
        result = engine.run(output_names=None, input_feed=input_data)
        print(result)


def main():  # pragma: no cover
    parser = argparse.ArgumentParser(description="Run a TVM-in-ONNX model in onnxruntime")
    parser.add_argument(
        "--model",
        required=True,
        help="Path TVM-in-ONNX model in tar format.",
    )
    args = parser.parse_args()

    infer(
        model_path=args.model,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
