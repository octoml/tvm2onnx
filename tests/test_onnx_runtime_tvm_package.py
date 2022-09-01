"""Tests ONNX Packaging."""
import os
import pathlib
import tarfile
import tempfile
import shutil

import numpy as np
import onnx
import onnxruntime
from onnx import ModelProto, mapping
from onnx import TensorProto
from onnx.external_data_helper import convert_model_to_external_data
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_tensor,
    make_tensor_value_info,
)
from tvm2onnx.onnx_model import ONNXModel

from typing import AnyStr, Dict, List

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "testdata/abtest.onnx")


def get_input_data_for_model_with_fixed_shapes(onnx_model: ModelProto) -> Dict[AnyStr, np.ndarray]:
    """
    Create input data for model with static shapes
    """

    def get_onnx_input_names(model: ModelProto) -> List[AnyStr]:
        inputs = [node.name for node in model.graph.input]
        initializer = [node.name for node in model.graph.initializer]
        inputs = list(set(inputs) - set(initializer))
        return sorted(inputs)

    def get_onnx_input_types(model: ModelProto) -> List[np.dtype]:
        input_names = get_onnx_input_names(model)
        return [
            mapping.TENSOR_TYPE_TO_NP_TYPE[node.type.tensor_type.elem_type]
            for node in sorted(model.graph.input, key=lambda node: node.name)
            if node.name in input_names
        ]

    def get_onnx_input_shapes(model: ModelProto) -> List[List[int]]:
        input_names = get_onnx_input_names(model)
        return [
            [dv.dim_value for dv in node.type.tensor_type.shape.dim]
            for node in sorted(model.graph.input, key=lambda node: node.name)
            if node.name in input_names
        ]

    input_names = get_onnx_input_names(onnx_model)
    input_shapes = get_onnx_input_shapes(onnx_model)
    input_types = get_onnx_input_types(onnx_model)
    assert len(input_names) == len(input_types) == len(input_shapes)
    random_inputs = [np.random.uniform(size=shape).astype(dtype) for shape, dtype in zip(input_shapes, input_types)]
    return dict(zip(input_names, random_inputs))


def test_mnist():
    with tempfile.TemporaryDirectory() as tdir:
        from tvm2onnx import TVM2ONNX_PACKAGE_ROOT_DIR
        project_root = os.path.split(TVM2ONNX_PACKAGE_ROOT_DIR)[0]
        model_path = os.path.join(project_root, "tests", "testdata", "mnist.onnx")
        # model_path = os.path.join(project_root, "tests", "testdata", "big_model.onnx")

        source_model = ONNXModel.from_file(pathlib.Path(model_path))
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

        sess_options = onnxruntime.SessionOptions()
        sess_options.register_custom_ops_library(custom_lib)

        engine = onnxruntime.InferenceSession(
            onnx_model_path,
            providers=["CPUExecutionProvider"],
            provider_options=[{}],
            sess_options=sess_options,
        )

        output_data = engine.run(output_names=None, input_feed=get_input_data_for_model_with_fixed_shapes(onnx.load(model_path)))
        print(output_data)


def test_onnx_package():
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

        engine = onnxruntime.InferenceSession(
            onnx_model_path,
            providers=["CPUExecutionProvider"],
            provider_options=[{}],
            sess_options=sess_options,
        )
        output_data = engine.run(output_names=None, input_feed=input_data)

        sum = input_data["a"] + input_data["b"]
        product = input_data["a"] * input_data["b"]
        actual_sum = output_data[0]
        actual_product = output_data[1]
        assert np.allclose(sum, actual_sum)
        assert np.allclose(product, actual_product)


def add_constant_onnx_model(model_dir, input_shape, uniform=False):
    """Returns an ONNX model with external constants."""
    a = make_tensor_value_info("a:0", TensorProto.FLOAT, input_shape)

    if uniform:
        c1_data = np.full(shape=input_shape, fill_value=3, dtype=np.dtype("float32"))
        c2_data = np.full(shape=input_shape, fill_value=4, dtype=np.dtype("float32"))
    else:
        c1_data = np.random.randn(*input_shape).astype(np.dtype("float32"))
        c2_data = np.random.randn(*input_shape).astype(np.dtype("float32"))
    c1 = make_node(
        "Constant",
        inputs=[],
        outputs=["c1"],
        name="c1_const_data",
        value=make_tensor(
            name="c1_tensor",
            data_type=TensorProto.FLOAT,
            dims=c1_data.shape,
            vals=c1_data.flatten().tobytes(),
            raw=True,
        ),
    )
    print(f"const array size {c1_data.size * 4}")

    c2 = make_node(
        "Constant",
        inputs=[],
        outputs=["c2"],
        name="c2_const_data",
        value=make_tensor(
            name="c2_tensor",
            data_type=TensorProto.FLOAT,
            dims=c2_data.shape,
            vals=c2_data.flatten().tobytes(),
            raw=True,
        ),
    )

    add = make_node("Add", ["a:0", "c1"], ["add"])
    mul = make_node("Mul", ["add", "c2"], ["result"])

    result = make_tensor_value_info("result", TensorProto.FLOAT, input_shape)

    graph = make_graph(
        nodes=[c1, add, c2, mul], name="ab_model", inputs=[a], outputs=[result]
    )

    onnx_proto = make_model(graph)
    onnx.checker.check_model(onnx_proto)

    onnx_model = ONNXModel(model=onnx_proto)
    onnx_model.infer_and_update_inputs()
    relay_model = onnx_model.to_relay()
    relay_model.to_tvm_file("/usr/constant_add.tvm")

    model_path = os.path.join(model_dir, "test.onnx")
    convert_model_to_external_data(
        onnx_proto,
        all_tensors_to_one_file=False,
        size_threshold=1024,
        convert_attribute=True,
    )
    onnx.save(onnx_proto, model_path)
    return c1_data, c2_data


def test_constant_model():
    input_shape = [8, 3, 224, 224]
    with tempfile.TemporaryDirectory() as tdir:
        model_path = os.path.join(tdir, "test.onnx")
        c1_data, c2_data = add_constant_onnx_model(
            model_dir=tdir, input_shape=input_shape, uniform=True
        )
        onnx_model = ONNXModel.from_file(model_path)
        onnx_model.infer_and_update_inputs()
        relay_model = onnx_model.to_relay()
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
        input_data["a"] = np.random.randn(*c1_data.shape).astype(np.dtype("float32"))

        sess_options = onnxruntime.SessionOptions()
        sess_options.register_custom_ops_library(custom_lib)

        engine = onnxruntime.InferenceSession(
            onnx_model_path,
            providers=["CPUExecutionProvider"],
            provider_options=[{}],
            sess_options=sess_options,
        )
        result = engine.run(output_names=None, input_feed=input_data)

        expected = (input_data["a"] + c1_data) * c2_data
        actual = result[0]
        assert np.allclose(expected, actual)


def main():
    test_mnist()


if __name__ == '__main__':
    main()
