import os
import tarfile
import tempfile
import typing

import numpy as np
import onnx
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from tvm2onnx.relay_model import RelayModel


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
    onnx.checker.check_model(onnx_proto)
    return onnx_proto


def test_metadata():
    metadata = {"key1": "value1", "key2": "value2"}
    onnx_proto = build_model(metadata=metadata)
    relay_model = RelayModel.from_onnx(onnx_proto)
    with tempfile.TemporaryDirectory() as tdir:
        saved_path = os.path.join(tdir, "metadata_test.tvm.onnx")
        relay_model.package_to_onnx(
            name="metadata_test",
            tvm_target="llvm",
            output_path=saved_path,
            metadata=metadata,
        )
        with tarfile.open(saved_path, "r") as tar:
            tar.extractall(tdir)
            loaded_proto = onnx.load_model(os.path.join(tdir, "metadata_test.onnx"))

            assert len(loaded_proto.metadata_props) == len(metadata)
            for prop in loaded_proto.metadata_props:
                assert prop.key in metadata.keys()
                assert prop.value == metadata[prop.key]
