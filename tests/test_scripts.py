import subprocess
import sys
import tempfile


def test_package_and_benchmark():
    with tempfile.NamedTemporaryFile() as tfile:
        model_path = tfile.name
        package_cmd = [
            sys.executable,
            "scripts/onnx_package.py",
            "--input",
            "tests/testdata/abtest.onnx",
            "--output",
            model_path,
        ]
        result = subprocess.run(package_cmd)
        assert result.returncode == 0

        benchmark_cmd = [
            sys.executable,
            "scripts/onnx_benchmark.py",
            "--model",
            model_path,
        ]
        result = subprocess.run(benchmark_cmd)
        assert result.returncode == 0
