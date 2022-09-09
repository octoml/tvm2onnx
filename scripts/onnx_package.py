"""
This script is to generate ONNX wrapped TVM models.
These ONNX wrapped TVM models can be run using onnx_benchmark.py.
"""

import argparse
import logging
import pathlib

import onnx

from scripts.utils import setup_logging
from tvm2onnx.relay_model import RelayModel

logging.basicConfig(level=logging.DEBUG)


def package(
    model_path: str,
    output_path: str,
):
    relay_model = RelayModel.from_onnx(onnx.load(model_path))
    relay_model.package_to_onnx(
        "mnist", tvm_target="llvm", output_path=pathlib.Path(output_path)
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
    args = parser.parse_args()

    setup_logging()
    package(model_path=args.input, output_path=args.output)


if __name__ == "__main__":  # pragma: no cover
    main()
