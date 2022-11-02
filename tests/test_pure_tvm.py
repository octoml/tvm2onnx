import os
import subprocess
import sys
import tempfile

import pytest


@pytest.mark.parametrize(
    "model_name", ["cast_float16_to_float32", "cast_float32_to_float16", "big_model"]
)
def test_package_and_benchmark(model_name):
    test_root_dir = os.path.dirname(__file__)

    with tempfile.TemporaryDirectory() as export_dir:
        package_cmd = [
            sys.executable,
            os.path.join(test_root_dir, "export_virtual_machine.py"),
            "--model_path",
            os.path.join(test_root_dir, "testdata", f"{model_name}.onnx"),
            "--export_dir",
            export_dir,
        ]
        result = subprocess.run(package_cmd)
        assert result.returncode == 0

        benchmark_cmd = [
            sys.executable,
            os.path.join(test_root_dir, "run_virtual_machine.py"),
            "--model_path",
            os.path.join(test_root_dir, "testdata", f"{model_name}.onnx"),
            "--export_dir",
            export_dir,
        ]
        result = subprocess.run(benchmark_cmd)
        assert result.returncode == 0
