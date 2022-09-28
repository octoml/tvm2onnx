"""Tests ONNX Packaging."""
import os
import tarfile
import tempfile

import numpy as np
import onnx
import onnxruntime
import pytest
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

_MODEL_PATH = "/usr/tvm2onnx/tests/testdata/abtest.onnx"

_DTYPE_LIST = [
    "float16",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
]


def main():
    dtype_str = "float32"
    with tempfile.TemporaryDirectory() as tdir:
        relay_model = RelayModel.from_onnx(
            onnx.load(_MODEL_PATH), dynamic_axis_substitute=1
        )
        onnx_path = os.path.join(tdir, "test_model.tvm.onnx")
        relay_model.package_to_onnx(
            name="test_model",
            tvm_target="llvm",
            output_path=onnx_path,
        )
        model_dir = os.path.join(tdir, "model")
        with tarfile.open(onnx_path, "r") as tar:
            tar.extractall(model_dir)
        onnx_model_path = os.path.join(model_dir, "test_model.onnx")
        custom_lib = os.path.join(model_dir, "custom_test_model.so")

        input_data = {}
        input_data["a"] = np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=np.float32
        )
        input_data["b"] = np.array(
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], dtype=np.float32
        )

        sess_options = onnxruntime.SessionOptions()
        sess_options.register_custom_ops_library(custom_lib)

        session = onnxruntime.InferenceSession(
            onnx_model_path,
            providers=["TvmExecutionProvider"],
            provider_options=[{}],
            sess_options=sess_options,
        )
        output_data = session.run(output_names=None, input_feed=input_data)

        sum = input_data["a"] + input_data["b"]
        product = input_data["a"] * input_data["b"]
        actual_sum = output_data[0]
        actual_product = output_data[1]
        assert np.allclose(sum, actual_sum)
        assert np.allclose(product, actual_product)

if __name__ == "__main__":  # pragma: no cover
    main()
