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

import os
import tarfile
import tempfile

import numpy as np
import onnx
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from scripts.utils.relay_model import RelayModel


def build_model():
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
    onnx_proto = build_model()
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
