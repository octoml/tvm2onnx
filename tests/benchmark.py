import argparse
import os
import tarfile
import tempfile
import typing
import time

import numpy as np
import onnx
import onnxruntime
from onnx.external_data_helper import convert_model_to_external_data
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_tensor,
    make_tensor_value_info,
)
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from scripts.utils.relay_model import RelayModel
from scripts.utils.setup_logging import setup_logging


def gen_model(model_dir, input_shape, dtype_str):
    """Returns an ONNX model with external constants."""
    dtype = np.dtype(dtype_str)

    a = make_tensor_value_info("a", NP_TYPE_TO_TENSOR_TYPE[dtype], input_shape)
    b = make_tensor_value_info("b", NP_TYPE_TO_TENSOR_TYPE[dtype], input_shape)
    c = make_tensor_value_info("c", NP_TYPE_TO_TENSOR_TYPE[dtype], input_shape)
    mul = make_node("Mul", ["a", "b"], ["mul"])
    add = make_node("Add", ["mul", "c"], ["result"])

    result = make_tensor_value_info(
        "result", NP_TYPE_TO_TENSOR_TYPE[dtype], input_shape
    )

    graph = make_graph(
        nodes=[mul, add],
        name="abc_model",
        inputs=[a, b, c],
        outputs=[result],
    )

    onnx_proto = make_model(graph)
    onnx.checker.check_model(onnx_proto)

    model_path = os.path.join(model_dir, "test.onnx")
    convert_model_to_external_data(
        onnx_proto,
        all_tensors_to_one_file=False,
        size_threshold=1024,
        convert_attribute=True,
    )
    onnx.save(onnx_proto, model_path)
    return model_path


def benchmark(
    onnx_model_path: str,
    custom_lib: str,
    input_shape: typing.List[int],
    dtype_str: str,
    warmup: int,
    iterations: int,
):
    dtype = np.dtype(dtype_str)

    input_data = {}
    input_data["a"] = np.random.randn(*input_shape).astype(dtype)
    input_data["b"] = np.random.randn(*input_shape).astype(dtype)
    input_data["c"] = np.random.randn(*input_shape).astype(dtype)
    expected = input_data["a"]*input_data["b"]+input_data["c"]

    sess_options = onnxruntime.SessionOptions()
    sess_options.register_custom_ops_library(custom_lib)

    session = onnxruntime.InferenceSession(
        onnx_model_path,
        providers=["CPUExecutionProvider"],
        provider_options=[{}],
        sess_options=sess_options,
    )

    for _ in range(warmup):
        _ = session.run(output_names=None, input_feed=input_data)

    times = []
    for _ in range(iterations):
        start_time = time.time()
        outputs = session.run(output_names=None, input_feed=input_data)
        times.append((time.time() - start_time)*1000)
        assert np.allclose(outputs[0], expected)

    mean = np.mean(times)
    cov = np.cov(times)

    print(f"mean {mean}, cov {cov}")
    print(", ".join(map(str, times)))

def main():  # pragma: no cover
    parser = argparse.ArgumentParser(description="Run benchmark.")
    args = parser.parse_args()
    input_shape = (128, 3, 224, 224)
    dtype_str = "int32"

    setup_logging()

    with tempfile.TemporaryDirectory() as tdir:
        model_path = gen_model(
            model_dir=tdir, input_shape=input_shape, dtype_str=dtype_str
        )

        relay_model = RelayModel.from_onnx(
            onnx.load(model_path), dynamic_axis_substitute=1
        )
        onnx_path = os.path.join(tdir, "test_model.tvm.onnx")
        relay_model.package_to_onnx(
            name=f"test_model_{dtype_str}",
            tvm_target="llvm",
            output_path=onnx_path,
        )
        model_dir = os.path.join(tdir, "model")
        with tarfile.open(onnx_path, "r") as tar:
            tar.extractall(model_dir)

        onnx_model_path = os.path.join(model_dir, f"test_model_{dtype_str}.onnx")
        custom_lib = os.path.join(model_dir, f"custom_test_model_{dtype_str}.so")

        benchmark(
            onnx_model_path=onnx_model_path,
            custom_lib=custom_lib,
            input_shape=input_shape,
            dtype_str=dtype_str,
            warmup=3,
            iterations=10,
        )


if __name__ == "__main__":  # pragma: no cover
    main()
