import os
import pathlib
import tarfile
import tempfile

import numpy as np
import onnxruntime
import pytest
import structlog

from tvm2onnx.onnx_model import ONNXModel
from tvm2onnx.utils import get_path_contents

_MODELS_DIR = os.path.join(os.path.dirname(__file__), "../models")
import os

LOG = structlog.get_logger(__name__)


def gather_models():
    if os.path.exists(_MODELS_DIR):
        for model_name in get_path_contents(_MODELS_DIR):
            if pathlib.Path(model_name).suffix == ".onnx":
                yield model_name


@pytest.mark.slow
@pytest.mark.parametrize("model_name", gather_models())
def test_models_in_models_dir(model_name):
    """So far this test is just to see if models fail to either load or run"""
    model_path = os.path.abspath(os.path.join(_MODELS_DIR, model_name))
    print(model_path)
    source_model = ONNXModel.from_file(model_path)
    source_model.infer_and_update_inputs()
    for name, shape in source_model.input_shapes.items():
        if shape[0] == -1:
            shape[0] = 1
            source_model.input_shapes[name] = shape
    print(f"input {name}, shape {shape}")
    relay_model = source_model.to_relay()
    with tempfile.TemporaryDirectory() as tdir:
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

        onnx_model = ONNXModel.from_file(onnx_model_path)
        onnx_model.infer_and_update_inputs()
        input_data = {}
        for name, shape in onnx_model.input_shapes.items():
            dtype = onnx_model.input_dtypes[name]
            input_data[name] = np.random.randn(*shape).astype(np.dtype(dtype))

        sess_options = onnxruntime.SessionOptions()
        sess_options.register_custom_ops_library(custom_lib)

        session = onnxruntime.InferenceSession(
            onnx_model_path,
            providers=["CPUExecutionProvider"],
            provider_options=[{}],
            sess_options=sess_options,
        )

        session.run(output_names=None, input_feed=input_data)
