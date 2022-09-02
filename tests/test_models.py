import pytest
import os
import tempfile
import tarfile
import numpy as np
import onnxruntime

from tvm2onnx.utils import get_path_contents
from tvm2onnx.onnx_model import ONNXModel

_MODELS_DIR = os.path.join(os.path.dirname(__file__), "../models")
import os


@pytest.mark.slow
def test_models_in_models_dir():
    for model_name in get_path_contents(_MODELS_DIR):
        model_path = os.path.join(_MODELS_DIR, model_name)
        print("*********************", model_path)
        source_model = ONNXModel.from_file(model_path)
        source_model.infer_and_update_inputs()
        relay_model = source_model.to_relay()
        for name, shape in relay_model.input_shapes.items:
            print(f"input {name}, shape {shape}")
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
