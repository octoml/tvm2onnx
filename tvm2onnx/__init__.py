import os

TVM2ONNX_PACKAGE_ROOT_DIR = os.path.split(__file__)[0]


def get_templates_dir() -> str:
    return os.path.join(TVM2ONNX_PACKAGE_ROOT_DIR, "templates", "onnx_custom_op")
