"""Tests ONNX Packaging."""
import os
import tarfile
import tempfile
import pytest
import uuid
import gc
import sys

# breakpoint()
# sys.path = ["/usr/tvm2onnx/3rdparty/onnxruntime/build/Linux/RelWithDebInfo/", *sys.path]

import numpy as np
import onnx
import onnxruntime
from onnx import TensorProto
from onnx.external_data_helper import convert_model_to_external_data
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_tensor,
    make_tensor_value_info,
)
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE
from tvm2onnx.onnx_model import ONNXModel
from tvm2onnx.utils import print_path_contents
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "testdata/abtest.onnx")


@pytest.mark.parametrize(
    "dtype_str",
    [
        "float32",
        "int32",
        "int64",
        # "int64",
        # "int64",
        # "int64",
        # "int64",
    ],
)
def test_onnx_package(dtype_str):
    with tempfile.TemporaryDirectory() as tdir:
        source_model = ONNXModel.from_file(_MODEL_PATH)
        source_model.infer_and_update_inputs()
        relay_model = source_model.to_relay()
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
            providers=["CPUExecutionProvider"],
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


def add_constant_onnx_model(model_dir, input_shape, dtype_str, uniform):
    """Returns an ONNX model with external constants."""
    dtype = np.dtype(dtype_str)
    a = make_tensor_value_info(f"a_{dtype_str}", NP_TYPE_TO_TENSOR_TYPE[dtype], input_shape)

    if uniform:
        c1_data = np.full(shape=input_shape, fill_value=3, dtype=dtype)
        c2_data = np.full(shape=input_shape, fill_value=4, dtype=dtype)
    else:
        c1_data = np.random.randn(*input_shape).astype(dtype)
        c2_data = np.random.randn(*input_shape).astype(dtype)

    c1 = make_node(
        "Constant",
        inputs=[],
        outputs=[f"c1_{dtype_str}"],
        name=f"c1_{dtype_str}_const_data",
        value=make_tensor(
            name=f"c1_{dtype_str}_tensor",
            data_type=NP_TYPE_TO_TENSOR_TYPE[dtype],
            dims=c1_data.shape,
            vals=c1_data.flatten().tobytes(),
            raw=True,
        ),
    )

    c2 = make_node(
        "Constant",
        inputs=[],
        outputs=[f"c2_{dtype_str}"],
        name=f"c2_{dtype_str}_const_data",
        value=make_tensor(
            name=f"c2_{dtype_str}_tensor",
            data_type=NP_TYPE_TO_TENSOR_TYPE[dtype],
            dims=c2_data.shape,
            vals=c2_data.flatten().tobytes(),
            raw=True,
        ),
    )

    add = make_node("Add", [f"a_{dtype_str}", f"c1_{dtype_str}"], ["add"])
    mul = make_node("Mul", ["add", f"c2_{dtype_str}"], ["result"])

    result = make_tensor_value_info("result", NP_TYPE_TO_TENSOR_TYPE[dtype], input_shape)

    graph = make_graph(
        nodes=[c1, add, c2, mul], name="ab_model", inputs=[a], outputs=[result]
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
    onnx.save(onnx_proto, f"cmodel_{dtype_str}.onnx")
    return c1_data, c2_data


@pytest.mark.parametrize(
    "dtype_str,input_shape",
    [
        ("int32", [1, 2, 8, 8]),
        ("float32", [1, 2, 7, 7]),
        # "int64",
    ],
)
def test_constant_model(dtype_str, input_shape):
    dtype = np.dtype(dtype_str)
    # input_shape = [1, 2, 8, 8]
    with tempfile.TemporaryDirectory() as tdir:
        model_path = os.path.join(tdir, "test.onnx")
        c1_data, c2_data = add_constant_onnx_model(
            model_dir=tdir, input_shape=input_shape, dtype_str=dtype_str, uniform=True
        )
        onnx_model = ONNXModel.from_file(model_path)
        onnx_model.infer_and_update_inputs()
        relay_model = onnx_model.to_relay()
        onnx_path = os.path.join(tdir, "test_model.tvm.onnx")
        relay_model.package_to_onnx(
            name=f"test_model_{dtype_str}",
            tvm_target="llvm",
            output_path=onnx_path,
        )
        model_dir = os.path.join(tdir, "model")
        with tarfile.open(onnx_path, "r") as tar:
            tar.extractall(model_dir)


        import shutil
        shutil.copy(onnx_path, f"model_{dtype_str}.tvm.onnx")
        print_path_contents(model_dir)


        onnx_model_path = os.path.join(model_dir, f"test_model_{dtype_str}.onnx")
        custom_lib = os.path.join(model_dir, f"custom_test_model_{dtype_str}.so")

        input_data = {}
        input_data[f"a_{dtype_str}"] = np.random.randn(*c1_data.shape).astype(dtype)

        # model_proto = onnx.load_model(onnx_model_path, load_external_data=True)
        # breakpoint()
        # pass

        sess_options = onnxruntime.SessionOptions()
        sess_options.register_custom_ops_library(custom_lib)

        session = onnxruntime.InferenceSession(
            onnx_model_path,
            providers=["CPUExecutionProvider"],
            provider_options=[{}],
            sess_options=sess_options,
        )
        result = session.run(output_names=None, input_feed=input_data)

        expected = (input_data[f"a_{dtype_str}"] + c1_data) * c2_data
        actual = result[0]
        assert np.allclose(expected, actual)


@pytest.mark.parametrize(
    "model_path",
    [
        "/usr/tvm2onnx/tests/testdata/abtest.onnx",
        "/usr/tvm2onnx/tests/testdata/mnist.onnx",
        "/usr/tvm2onnx/cmodel_float32.onnx",
        "/usr/tvm2onnx/cmodel_int32.onnx",
    ],
)
def test_test(model_path):
    with tempfile.TemporaryDirectory() as tdir:
        onnx_model = ONNXModel.from_file(model_path)
        onnx_model.infer_and_update_inputs()
        relay_model = onnx_model.to_relay()
        onnx_path = os.path.join(tdir, "test_model.tvm.onnx")
        relay_model.package_to_onnx(
            name=f"test_model",
            tvm_target="llvm",
            output_path=onnx_path,
        )
        model_dir = os.path.join(tdir, "model")
        with tarfile.open(onnx_path, "r") as tar:
            tar.extractall(model_dir)
        onnx_model_path = os.path.join(model_dir, "test_model.onnx")
        custom_lib = os.path.join(model_dir, "custom_test_model.so")

        sess_options = onnxruntime.SessionOptions()
        sess_options.register_custom_ops_library(custom_lib)
        sess_options.log_verbosity_level = 0
        # sess_options.log_severity_level = 0

        session = onnxruntime.InferenceSession(
            onnx_model_path,
            providers=["CPUExecutionProvider"],
            provider_options=[{}],
            sess_options=sess_options,
        )
        # result = session.run(output_names=None, input_feed=input_data)
