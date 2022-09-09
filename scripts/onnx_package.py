"""
This script is to generate ONNX wrapped TVM models.
These ONNX wrapped TVM models can be run using onnx_benchmark.py.
"""

import argparse
import logging
import pathlib

import onnx

from scripts.utils.relay_model import RelayModel

logging.basicConfig(level=logging.CRITICAL)


def package(
    model_path: str,
    output_path: str,
):
    relay_model = RelayModel.from_onnx(onnx.load(model_path))
    relay_model.package_to_onnx(
        pathlib.Path(model_path).stem,
        tvm_target="llvm",
        output_path=pathlib.Path(output_path),
    )


def main():  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="Package an ONNX model as an untuned TVM ONNX."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Model ONNX file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output file in ONNX tar format.",
    )
    args = parser.parse_args()

    package(model_path=args.input, output_path=args.output)


if __name__ == "__main__":  # pragma: no cover
    main()
