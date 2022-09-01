import os
import pathlib

TVM2ONNX_PACKAGE_ROOT_DIR = os.path.split(__file__)[0]
EXTERNAL_ROOT = pathlib.Path(TVM2ONNX_PACKAGE_ROOT_DIR).parent / "3rdparty"
TVM_ROOT_DIR = EXTERNAL_ROOT / "tvm"
ONNX_RUNTIME_ROOT_DIR = EXTERNAL_ROOT / "onnxruntime"


def get_templates_dir() -> str:
    return os.path.join(TVM2ONNX_PACKAGE_ROOT_DIR, "templates", "onnx_custom_op")


def get_tvm_root_dir() -> str:
    return str(TVM_ROOT_DIR)


def get_onnx_runtime_root_dir() -> str:
    return str(ONNX_RUNTIME_ROOT_DIR)
