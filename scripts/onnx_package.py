"""
This script is to generate ONNX wrapped TVM models.
These ONNX wrapped TVM models can be run using onnx_benchmark.py.
"""

import argparse
import logging
import pathlib

import onnx

from scripts.utils.relay_model import RelayModel
from scripts.utils.setup_logging import setup_logging

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
