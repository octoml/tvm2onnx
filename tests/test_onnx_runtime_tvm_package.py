"""Tests ONNX Packaging."""
import os
import sys
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

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "testdata/abtest.onnx")

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


def add_constant_onnx_model(model_dir, input_shape, dtype_str, uniform):
    """Returns an ONNX model with external constants."""
    dtype = np.dtype(dtype_str)

    if uniform:
        c1_data = np.full(shape=input_shape, fill_value=3, dtype=dtype)
        c2_data = np.full(shape=input_shape, fill_value=4, dtype=dtype)
    else:
        c1_data = np.random.randn(*input_shape).astype(dtype)
        c2_data = np.random.randn(*input_shape).astype(dtype)

    c1 = make_tensor(
        name="c1",
        data_type=NP_TYPE_TO_TENSOR_TYPE[dtype],
        dims=c1_data.shape,
        vals=c1_data.flatten().tobytes(),
        raw=True,
    )

    c2 = make_tensor(
        name="c2",
        data_type=NP_TYPE_TO_TENSOR_TYPE[dtype],
        dims=c2_data.shape,
        vals=c2_data.flatten().tobytes(),
        raw=True,
    )
    initializers = [c1, c2]

    a = make_tensor_value_info("a", NP_TYPE_TO_TENSOR_TYPE[dtype], input_shape)
    add = make_node("Add", ["a", "c1"], ["add"])
    mul = make_node("Mul", ["add", "c2"], ["result"])

    result = make_tensor_value_info(
        "result", NP_TYPE_TO_TENSOR_TYPE[dtype], input_shape
    )

    graph = make_graph(
        nodes=[add, mul],
        name="ab_model",
        inputs=[a],
        outputs=[result],
        initializer=initializers,
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
    return c1_data, c2_data


def run_with_custom_op(
    custom_op_model_name, custom_op_model_dir, input_data, use_zero_copy=False
):
    import pickle
    import subprocess

    with tempfile.TemporaryDirectory() as temp_directory:
        input_data_file_name = os.path.join(temp_directory, "input_data")
        output_data_file_name = os.path.join(temp_directory, "output_data")

        with open(input_data_file_name, "wb") as input_data_file:
            serialized_input_data = pickle.dumps(input_data)
            input_data_file.write(serialized_input_data)

        inference_cmd = [
            sys.executable,
            "run_inference_in_subprocess.py",
            "--custom_op_model_name",
            custom_op_model_name,
            "--custom_op_model_dir",
            custom_op_model_dir,
            "--input_data_file",
            input_data_file_name,
            "--output_data_file",
            output_data_file_name,
        ]
        if use_zero_copy:
            inference_cmd.append("--use_zero_copy")
        result = subprocess.run(
            inference_cmd, cwd=os.path.join(os.path.dirname(__file__))
        )
        assert result.returncode == 0

        with open(output_data_file_name, "rb") as output_data_file:
            output_data = pickle.loads(output_data_file.read())

    return output_data


def test_onnx_package():
    with tempfile.TemporaryDirectory() as tdir:
        relay_model = RelayModel.from_onnx(
            onnx.load(_MODEL_PATH), dynamic_axis_substitute=1
        )
        custom_op_model_name = "test_model"
        custom_op_tar_path = os.path.join(tdir, f"{custom_op_model_name}.onnx")
        relay_model.package_to_onnx(
            name=custom_op_model_name,
            tvm_target="llvm",
            output_path=custom_op_tar_path,
        )
        model_dir = os.path.join(tdir, "model")
        with tarfile.open(custom_op_tar_path, "r") as tar:
            tar.extractall(model_dir)
        onnx_model_path = os.path.join(model_dir, "test_model.onnx")
        custom_lib = os.path.join(model_dir, "custom_test_model.so")

        input_data = {
            "a": np.array(
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                dtype=np.float32,
            ),
            "b": np.array(
                [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], dtype=np.float32
            ),
        }

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


@pytest.mark.parametrize("use_zero_copy", [False, True])
@pytest.mark.parametrize(
    "dtype_str",
    _DTYPE_LIST,
)
def test_constant_model(dtype_str, use_zero_copy):
    # TODO(agladyshev): investigate this issue
    if dtype_str == "float16":
        pytest.skip("/tmp/tvm_model_XXXXXX.so: undefined symbol: __gnu_h2f_ieee")

    dtype = np.dtype(dtype_str)
    input_shape = [8, 3, 224, 224]
    with tempfile.TemporaryDirectory() as tdir:
        model_path = os.path.join(tdir, "test.onnx")
        c1_data, c2_data = add_constant_onnx_model(
            model_dir=tdir, input_shape=input_shape, dtype_str=dtype_str, uniform=True
        )
        relay_model = RelayModel.from_onnx(
            onnx.load(model_path), dynamic_axis_substitute=1
        )
        custom_op_model_name = f"test_model_{dtype_str}"
        custom_op_tar_path = os.path.join(tdir, f"{custom_op_model_name}.onnx")
        relay_model.package_to_onnx(
            name=custom_op_model_name,
            tvm_target="llvm --system-lib",
            output_path=custom_op_tar_path,
            use_zero_copy=use_zero_copy,
        )
        custom_op_model_dir = os.path.join(tdir, "model")
        with tarfile.open(custom_op_tar_path, "r") as tar:
            tar.extractall(custom_op_model_dir)

        input_data = {
            "a": np.random.randn(*c1_data.shape).astype(dtype),
        }

        result = run_with_custom_op(
            custom_op_model_name, custom_op_model_dir, input_data, use_zero_copy
        )

        expected = (input_data["a"] + c1_data) * c2_data
        actual = result[0]
        assert np.allclose(expected, actual)


def test_debug_build():
    dtype = np.dtype("float32")
    input_shape = [8, 3, 224, 224]
    with tempfile.TemporaryDirectory() as tdir:
        model_path = os.path.join(tdir, "test.onnx")
        c1_data, c2_data = add_constant_onnx_model(
            model_dir=tdir, input_shape=input_shape, dtype_str="float32", uniform=True
        )
        relay_model = RelayModel.from_onnx(
            onnx.load(model_path), dynamic_axis_substitute=1
        )
        custom_op_model_name = "test_model_debug"
        custom_op_tar_path = os.path.join(tdir, f"{custom_op_model_name}.onnx")
        relay_model.package_to_onnx(
            name=custom_op_model_name,
            tvm_target="llvm",
            output_path=custom_op_tar_path,
            debug_build=True,
        )
        custom_op_model_dir = os.path.join(tdir, "model")
        with tarfile.open(custom_op_tar_path, "r") as tar:
            tar.extractall(custom_op_model_dir)

        input_data = {
            "a": np.random.randn(*c1_data.shape).astype(dtype),
        }

        result = run_with_custom_op(
            custom_op_model_name, custom_op_model_dir, input_data
        )

        expected = (input_data["a"] + c1_data) * c2_data
        actual = result[0]
        assert np.allclose(expected, actual)


_DTYPE_FLIST = [
    "float16",
    "float32",
    # "float64",
]


def test_cast_model():
    shape = (1, 2, 3, 4)
    dtype1 = np.dtype("float16")
    dtype2 = np.dtype("float32")
    # print(f"dtype1={dtype_str1} dtype2={dtype_str2}")

    def make_cast_model(model_shape, input_dtype, output_dtype, save_path):
        input_type = NP_TYPE_TO_TENSOR_TYPE[np.dtype(input_dtype)]
        output_type = NP_TYPE_TO_TENSOR_TYPE[np.dtype(output_dtype)]

        cast_node = make_node(
            "Cast", inputs=["input"], outputs=["output"], to=output_type
        )

        graph = make_graph(
            [cast_node],
            "cast_model_test",
            inputs=[
                make_tensor_value_info("input", input_type, model_shape),
            ],
            outputs=[make_tensor_value_info("output", output_type, model_shape)],
        )

        model = make_model(graph, producer_name="cast_model_test")
        onnx.checker.check_model(model)
        onnx.save(model, save_path)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Make source ONNX model
        source_model_path = os.path.join(temp_dir, "cast.onnx")
        make_cast_model(shape, dtype1, dtype2, source_model_path)

        # Package to Custom Op format
        custom_op_model_name = f"cast_{dtype1}_to_{dtype2}"
        custom_op_tar_path = os.path.join(temp_dir, f"{custom_op_model_name}.onnx")
        relay_model = RelayModel.from_onnx(
            onnx.load(source_model_path), dynamic_axis_substitute=1
        )
        relay_model.package_to_onnx(
            name=custom_op_model_name,
            tvm_target="llvm",
            output_path=custom_op_tar_path,
        )

        # Extract Custom Op ONNX file and Custom Op shared library
        custom_op_model_dir = os.path.join(temp_dir, "model")
        with tarfile.open(custom_op_tar_path, "r") as tar:
            tar.extractall(custom_op_model_dir)

        # Run inference
        input_data = {
            "input": np.random.randn(*shape).astype(dtype1),
        }
        output = run_with_custom_op(
            custom_op_model_name, custom_op_model_dir, input_data
        )
        assert output[0].dtype == dtype2
