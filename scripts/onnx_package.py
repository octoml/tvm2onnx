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
import pathlib

import onnx
from utils.relay_model import RelayModel
from utils.setup_logging import setup_logging

logging.basicConfig(level=logging.DEBUG)


def package(
    model_path: str,
    output_path: str,
    debug_build: bool,
):
    relay_model = RelayModel.from_onnx(onnx.load(model_path), dynamic_axis_substitute=1)
    relay_model.package_to_onnx(
        "mnist",
        tvm_target="llvm",
        output_path=pathlib.Path(output_path),
        debug_build=debug_build,
    )


def main():  # pragma: no cover
    parser = argparse.ArgumentParser(description="Package a tuned TVM model to ONNX.")
    parser.add_argument(
        "--input",
        required=True,
        help="Model TVM file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output file in ONNX tar format.",
    )
    parser.add_argument(
        "--debug", action="store_true", default=False, help="Create a debug build"
    )
    args = parser.parse_args()

    setup_logging()
    package(model_path=args.input, output_path=args.output, debug_build=args.debug)


if __name__ == "__main__":  # pragma: no cover
    main()
