"""Tests ONNX Packaging."""
import os
import tempfile

import numpy as np
import onnx
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

import scripts.utils.testing_utils as testing

packaging_function = testing.package_model_and_extract_tar
inference_function = testing.get_ort_output

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


def test_onnx_package():
    with tempfile.TemporaryDirectory() as tdir:
        # Package to Custom Op format and extract Custom Op ONNX file and Custom Op shared library
        custom_op_model_name = "test_model"
        custom_op_tar_path = os.path.join(tdir, f"{custom_op_model_name}.onnx")
        custom_op_model_dir = os.path.join(tdir, "model")

        packaging_function(
            custom_op_tar_path,
            custom_op_model_dir,
            onnx_model=onnx.load(_MODEL_PATH),
            dynamic_axis_substitute=1,
            name=custom_op_model_name,
            tvm_target="llvm",
            output_path=custom_op_tar_path,
        )

        # Run inference
        input_data = {
            "a": np.array(
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                dtype=np.float32,
            ),
            "b": np.array(
                [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], dtype=np.float32
            ),
        }

        output_data = inference_function(
            custom_op_model_name, custom_op_model_dir, input_data
        )

        sum = input_data["a"] + input_data["b"]
        product = input_data["a"] * input_data["b"]
        actual_sum = output_data[0]
        actual_product = output_data[1]
        assert np.allclose(sum, actual_sum)
        assert np.allclose(product, actual_product)


@pytest.mark.parametrize("debug_build", [False, True])
@pytest.mark.parametrize("use_zero_copy", [False, True])
@pytest.mark.parametrize(
    "dtype_str",
    _DTYPE_LIST,
)
def test_constant_model(dtype_str, use_zero_copy, debug_build):
    dtype = np.dtype(dtype_str)
    input_shape = [8, 3, 224, 224]
    with tempfile.TemporaryDirectory() as tdir:
        # Make source ONNX model
        model_path = os.path.join(tdir, "test.onnx")
        c1_data, c2_data = add_constant_onnx_model(
            model_dir=tdir, input_shape=input_shape, dtype_str=dtype_str, uniform=True
        )

        # Package to Custom Op format and extract Custom Op ONNX file and Custom Op shared library
        custom_op_model_name = f"test_model_{dtype_str}"
        custom_op_tar_path = os.path.join(tdir, f"{custom_op_model_name}.onnx")
        custom_op_model_dir = os.path.join(tdir, "model")

        packaging_function(
            custom_op_tar_path,
            custom_op_model_dir,
            onnx_model=onnx.load(model_path),
            dynamic_axis_substitute=1,
            name=custom_op_model_name,
            tvm_target="llvm",
            output_path=custom_op_tar_path,
            use_zero_copy=use_zero_copy,
            debug_build=debug_build,
        )

        # Run inference
        input_data = {
            "a": np.random.randn(*c1_data.shape).astype(dtype),
        }

        result = inference_function(
            custom_op_model_name, custom_op_model_dir, input_data, use_zero_copy
        )

        expected = (input_data["a"] + c1_data) * c2_data
        actual = result[0]
        assert np.allclose(expected, actual)


_FLOAT_DTYPE_LIST = [
    "float16",
    "float32",
    "float64",
]

@pytest.mark.parametrize("dtype_str2", _FLOAT_DTYPE_LIST)
@pytest.mark.parametrize("dtype_str1", _FLOAT_DTYPE_LIST)
def test_cast_model(dtype_str1, dtype_str2):
    if dtype_str1 == "float64" and dtype_str2 == "float16":
        # This will fail until this TVM PR is merged
        # https://github.com/apache/tvm/pull/13395
        pytest.xfail("undefined symbol: __truncdfhf2")
    shape = (1, 2, 3, 4)
    dtype1 = np.dtype(dtype_str1)
    dtype2 = np.dtype(dtype_str2)

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

        # Package to Custom Op format and extract Custom Op ONNX file and Custom Op shared library
        custom_op_model_name = f"cast_{dtype1}_to_{dtype2}"
        custom_op_tar_path = os.path.join(temp_dir, f"{custom_op_model_name}.onnx")
        custom_op_model_dir = os.path.join(temp_dir, "model")

        packaging_function(
            custom_op_tar_path,
            custom_op_model_dir,
            onnx_model=onnx.load(source_model_path),
            dynamic_axis_substitute=1,
            name=custom_op_model_name,
            tvm_target="llvm",
            output_path=custom_op_tar_path,
        )

        # Run inference
        input_data = {
            "input": np.random.randn(*shape).astype(dtype1),
        }
        output = inference_function(
            custom_op_model_name, custom_op_model_dir, input_data
        )
        assert output[0].dtype == dtype2
