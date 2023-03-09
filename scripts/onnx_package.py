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

"""
This script is to generate ONNX wrapped TVM models.
These ONNX wrapped TVM models can be run using onnx_benchmark.py.
"""

import argparse
import logging
import pickle
import os
import json
import typing
import tempfile
import shutil

import tvm
from tvm2onnx.onnx_runtime_tvm_package import ONNXRuntimeTVMPackage

logging.basicConfig(level=logging.ERROR)


def package(
    model_path: str,
    ro_path: str,
    constants_path: str,
    metadata_path: str,
    tvm_runtime: str,
    output_path: str,
):
    with open(constants_path, 'rb') as f:
        constants_map = pickle.load(f)

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    input_shapes = {tensor["name"]: tensor["shape"] for tensor in metadata["inputs"]}
    input_dtypes = {tensor["name"]: tensor["dtype"] for tensor in metadata["inputs"]}
    output_shapes = {tensor["name"]: tensor["shape"] for tensor in metadata["outputs"]}
    output_dtypes = {tensor["name"]: tensor["dtype"] for tensor in metadata["outputs"]}
    tvm_target = tvm.target.Target(metadata["target"])

    compiler_flags: typing.List[str] = ["-fPIC"]

    # tvm_dir = pathlib.Path(os.path.dirname(tvm.__file__)).parent.parent
    # compiler_flags.append(f"-I{tvm_dir / '3rdparty/dmlc-core/include'}")
    # compiler_flags.append(f"-I{tvm_dir / '3rdparty/dlpack/include'}")
    # compiler_flags.append(f"-I{tvm_dir / 'include'}")

    if tvm_target.kind.name == "cuda":
        compiler_flags.append("-L/usr/local/cuda/lib64")
        compiler_flags.append("-lcuda")
        compiler_flags.append("-lcudart")
    if "cudnn" in tvm_target.libs:
        compiler_flags.append("-L/usr/lib/x86_64-linux-gnu")
        compiler_flags.append("-lcudnn")
    if "cublas" in tvm_target.libs:
        compiler_flags.append("-L/usr/local/cuda/lib64")
        compiler_flags.append("-lcublas")
    if "cblas" in tvm_target.libs:
        compiler_flags.append("-L/lib/x86_64-linux-gnu")
        compiler_flags.append("-lopenblas")

    # ld segfaults when cross-compiling via tvm2onnx to ARM; use clang + lld instead.
    compiler = "clang++-12"
    compiler_flags.append("-fuse-ld=lld")

    # Required for ONNXRuntime headers
    compiler_flags.append("-fms-extensions")

    # mtriple = (tvm_target.host or tvm_target).attrs.get("mtriple")
    # if mtriple == "aarch64-linux-gnu":
    #     compiler_flags.append("--target=aarch64-linux-gnu")
    #     for include_path in utils.cross_compiler_cpp_include_search_paths(
    #         "aarch64-linux-gnu"
    #     ):
    #         compiler_flags.append(f"-I{include_path}")
    # elif mtriple == "armv8l-linux-gnueabihf":
    #     compiler_flags.append("--target=arm-linux-gnueabihf")
    #     for include_path in utils.cross_compiler_cpp_include_search_paths(
    #         "arm-linux-gnueabihf"
    #     ):
    #         compiler_flags.append(f"-I{include_path}")


    with tempfile.TemporaryDirectory() as build_dir:
        packager = ONNXRuntimeTVMPackage(
            model_name="demo",
            tvm_runtime_lib=tvm_runtime,
            model_object=model_path,
            model_ro=ro_path,
            constants_map=constants_map,
            input_shapes=input_shapes,
            input_dtypes=input_dtypes,
            output_shapes=output_shapes,
            output_dtypes=output_dtypes,
            dl_device_type=metadata["device"],
            compiler=compiler,
            compiler_flags=" ".join(compiler_flags),
        )

        result = packager.build_package(build_dir)
        shutil.copy(result, output_path)

def main():  # pragma: no cover
    parser = argparse.ArgumentParser(description="Package a tuned TVM model to ONNX.")
    parser.add_argument(
        "--model",
        required=True,
        help="A compiled model.o.",
    )
    parser.add_argument(
        "--ro",
        required=True,
        help="The model's .ro file.",
    )
    parser.add_argument(
        "--constants",
        required=True,
        help="The model's constants pickle file.",
    )
    parser.add_argument(
        "--metadata",
        required=True,
        help="The model's metadata.json file.",
    )
    parser.add_argument(
        "--tvm-runtime",
        required=True,
        help="The libtvm_runtime.a file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output file in ONNX tar format.",
    )
    args = parser.parse_args()

    model_path = os.path.abspath(args.model)
    ro_path = os.path.abspath(args.ro)
    constants_path = os.path.abspath(args.constants)
    tvm_runtime_path = os.path.abspath(args.tvm_runtime)
    metadata_path = os.path.abspath(args.metadata)
    output_path = os.path.abspath(args.output)

    if not os.path.exists(model_path):
        print("model file does not exist")
        exit(1)

    if not os.path.exists(ro_path):
        print("ro file does not exist")
        exit(1)

    if not os.path.exists(constants_path):
        print("constants file does not exist")
        exit(1)

    if not os.path.exists(tvm_runtime_path):
        print("tvm runtime file does not exist")
        exit(1)

    if not os.path.exists(metadata_path):
        print("metadata file does not exist")
        exit(1)

    package(
        model_path=model_path,
        ro_path=ro_path,
        constants_path=constants_path,
        metadata_path=metadata_path,
        tvm_runtime=tvm_runtime_path,
        output_path=output_path
    )


if __name__ == "__main__":  # pragma: no cover
    main()
