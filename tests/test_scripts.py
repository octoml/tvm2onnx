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

import subprocess
import sys
import tempfile
import os


def test_package_and_benchmark():
    with tempfile.TemporaryDirectory() as tdir:
        tune_cmd = [
            sys.executable,
            "tutorial/basic_tune_model.py",
            "--model",
            "tests/testdata/abtest.onnx",
            "--output",
            tdir,
        ]
        result = subprocess.run(tune_cmd)
        assert result.returncode == 0

        model_path = os.path.join(tdir, "model.o")
        ro_path = os.path.join(tdir, "vm_exec_code.ro")
        constants = os.path.join(tdir, "constants.pkl")
        metadata = os.path.join(tdir, "metadata.json")
        tvm_runtime = "3rdparty/tvm/build/libtvm_runtime.a"
        output = os.path.join(tdir, "output.onnx")
        package_cmd = [
            sys.executable,
            "scripts/onnx_package.py",
            "--model",
            model_path,
            "--ro",
            ro_path,
            "--constants",
            constants,
            "--metadata",
            metadata,
            "--tvm-runtime",
            tvm_runtime,
            "--output",
            output,
        ]
        result = subprocess.run(package_cmd)
        assert result.returncode == 0
