import os
import pathlib
import tarfile
import tempfile

import numpy as np
import onnx
import onnxruntime
import pytest
import structlog
import contextlib

from tvm2onnx.relay_model import RelayModel
from tvm2onnx.utils import get_path_contents
from tvm import relay

_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
import os

LOG = structlog.get_logger(__name__)

_MACOS_EXTENDED_ATTRIBUTE_FILE_PREFIX = "._"

def _load_model_from_tar_file(model_tar_path: pathlib.Path) -> onnx.ModelProto:
    """Extracts an onnx model from the given bytes if they represent a tarfile.
    :param model_bytes: the bytes to extract a model from.
    :return: the onnx model extracted from the tarfile bytes.
    """
    if not tarfile.is_tarfile(str(model_tar_path)):
        LOG.info("Model bytes do not represent a tarfile. Halting tar extraction.")
        raise Exception("Model is not a tarfile.")

    LOG.info("Extracting ONNX model from given tarfile.")
    with contextlib.ExitStack() as stack:
        model_tar = stack.enter_context(tarfile.open(str(model_tar_path)))
        members = model_tar.getmembers()
        onnx_members = [
            m
            for m in members
            # This picks out any ONNX model files.
            if m.name.endswith(".onnx")
            # This filters MacOS extended attribute files which also end in
            # `.onnx`. For example, if you tar `mnist.onnx` on MacOS, upon
            # programmatic extraction there will be both an `mnist.onnx`
            # and an `._mnist.onnx` within. Only the former is useful to us,
            # and the latter should be ignored.
            # Pathlib is additionally used to make sure this works even for
            # extracted files with a folder prefix, by only looking at the
            # base name.
            and not pathlib.Path(m.name).name.startswith(
                _MACOS_EXTENDED_ATTRIBUTE_FILE_PREFIX
            )
        ]

        if not onnx_members:
            raise Exception(
                "No .onnx files found in given tarfile."
            )
        if len(onnx_members) > 1:
            onnx_file_names = ", ".join(sorted(o.name for o in onnx_members))
            raise Exception(
                f"Multiple .onnx files found in given tarfile - {onnx_file_names}.",
                {"files": onnx_file_names},
            )
        onnx_file = onnx_members[0]

        # It's not safe to use the "extractall" API of TarFile because
        # it allows files to extract themselves into system directories.
        # Instead, we manually extract and save the files to a tempdir.
        tempdir = stack.enter_context(tempfile.TemporaryDirectory())
        for member in members:
            if member.isfile():
                file_bytes = model_tar.extractfile(member).read()  # type: ignore
                basename = os.path.basename(member.name)
                with open(os.path.join(tempdir, basename), "wb") as f:
                    f.write(file_bytes)
        model_path = os.path.join(tempdir, os.path.basename(onnx_file.name))
        onnx.checker.check_model(model_path)
        return onnx.load(model_path)


def load_model(model_path):
    """Loads an ONNXModel from the given path.
    :param model_path: the path to a file containing an ONNXModel.
    :param custom_op_libs: optional list of paths to custom op libraries.
    :return: the ONNXModel loaded from the given file path.
    :raise Exception: if the model could not be loaded.
    """
    LOG.info("Loading an ONNXModel from file.", model_path=model_path)
    if not os.path.exists(model_path):
        raise Exception(f"File '{model_path}' not found")
    try:
        onnx_proto = _load_model_from_tar_file(model_tar_path=model_path)
        LOG.info("ONNXModel successfully loaded from tar file")
    except Exception:
        try:
            onnx_proto = onnx.load_model(str(model_path))
            LOG.info("ONNXModel successfully loaded from file")
        except Exception as e:
            LOG.exception("Failed loading ONNXModel from file")
            raise Exception(
                "Unable to load ONNX model.", {"error": str(e)}
            ) from e
    return onnx_proto


def gather_models():
    if os.path.exists(_MODELS_DIR):
        for model_name in get_path_contents(_MODELS_DIR):
            if pathlib.Path(model_name).suffix == ".onnx":
                yield model_name


def model_runner(model_path):
    print(f"************************** {model_path}")
    onnx_protobuf = load_model(model_path)
    relay_model = RelayModel.from_onnx(onnx_protobuf)
    for name, shape in relay_model.input_shapes.items():
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
        for name, shape in relay_model.input_shapes.items():
            dtype = relay_model.input_dtypes[name]
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
    return 42

@pytest.mark.slow
@pytest.mark.parametrize("model_name", gather_models())
def test_models_in_models_dir(model_name):
    """So far this test is just to see if models fail to either load or run"""
    model_path = os.path.abspath(os.path.join(_MODELS_DIR, model_name))
    import subprocess
    cmd = ["python", "-c", f"from tests.test_models import model_runner; model_runner('{model_path}')"]
    print(" ".join(cmd))
    result = subprocess.run(cmd, capture_output=True)
    assert result.returncode == 0
