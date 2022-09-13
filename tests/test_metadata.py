import typing

import numpy as np
import onnx
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE


def build_model(metadata: typing.Dict[str, str]):
    dtype = np.dtype("float32")
    input_shape = [2, 3]

    a = make_tensor_value_info("a", NP_TYPE_TO_TENSOR_TYPE[dtype], input_shape)
    b = make_tensor_value_info("b", NP_TYPE_TO_TENSOR_TYPE[dtype], input_shape)
    add = make_node("Add", ["a", "b"], ["result"])

    result = make_tensor_value_info(
        "result", NP_TYPE_TO_TENSOR_TYPE[dtype], input_shape
    )

    graph = make_graph(
        nodes=[add],
        name="add_model",
        inputs=[a, b],
        outputs=[result],
    )

    onnx_proto = make_model(graph)
    for key, value in metadata.items():
        meta = onnx_proto.metadata_props.add()
        meta.key = key
        meta.value = value
    onnx.checker.check_model(onnx_proto)
    return onnx_proto


def test_metadata():
    metadata = {"key1": "value1", "key2": "value2"}
    onnx_proto = build_model(metadata=metadata)
    assert len(onnx_proto.metadata_props) == len(metadata)
    for prop in onnx_proto.metadata_props:
        assert prop.key in metadata.keys()
        assert prop.value == metadata[prop.key]
