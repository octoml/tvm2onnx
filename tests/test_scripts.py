import subprocess
import tempfile

def test_package_and_bencmark():
    with tempfile.NamedTemporaryFile() as tfile:
        model_path = tfile.name
        package_cmd = [
            "python", "scripts/onnx_package.py",
            "--input", "tests/testdata/abtest.onnx",
            "--output", model_path
            ]
        result = subprocess.run(package_cmd, capture_output=True)
        assert result.returncode == 0

        benchmark_cmd = [
            "python", "scripts/onnx_benchmark.py",
            "--model", model_path
            ]
        result = subprocess.run(benchmark_cmd, capture_output=True)
        assert result.returncode == 0
