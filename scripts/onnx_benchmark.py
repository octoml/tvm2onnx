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

logging.basicConfig(level=logging.CRITICAL)


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


def _do_benchmark(
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
        # output_names = [tensor.name for tensor in self.model.graph.output]

        input_data = generate_input_data(
            input_shapes=input_shapes, input_dtypes=input_dtypes
        )
        # output_names = metadata.output_names
        result = engine.run(output_names=None, input_feed=input_data)
        print(result)


def main():  # pragma: no cover
    parser = argparse.ArgumentParser(description="Benchmark an ONNX model")
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the ONNX model tarball to benchmark",
    )
    args = parser.parse_args()

    _do_benchmark(
        model_path=args.model,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
