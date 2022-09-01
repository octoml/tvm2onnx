"""
Performance test of Pure ONNX Runtime and ONNX Runtime with TVM inside custom op
"""

import argparse
from functools import partial
import logging

import tempfile

from test_utils import (
    unpack_onnx_tar,
    generate_input_shapes_dtypes,
    generate_input_data,
    get_ort_inference_session,
    perf_test,
    get_tvm_vm_runner,
)

logging.basicConfig(level=logging.CRITICAL)


def test_benchmark(
    model_path,
):
    benchmark_test = partial(perf_test, iters_number = 1000, model_name = "test model")
    with tempfile.TemporaryDirectory() as tmp:
        onnx_path, custom_op_libs = unpack_onnx_tar(model_path, tmp)

        input_shapes, input_dtypes = generate_input_shapes_dtypes(onnx_path)
        input_data = generate_input_data(input_shapes=input_shapes, input_dtypes=input_dtypes)
        
        # ORT with custom op
        engine = get_ort_inference_session(onnx_path, custom_op_libs)
        ort_runner = partial(engine.run, output_names=None, input_feed=input_data)

        benchmark_test(ort_runner, framework_name = "ONNX Runtime with Custom Op")
        
        # pure ORT
        engine = get_ort_inference_session(onnx_path)
        ort_runner = partial(engine.run, output_names=None, input_feed=input_data)

        benchmark_test(ort_runner, framework_name = "Pure ONNX Runtime")

        # Pure TVM (VirtualMachine)
        tvm_runner = get_tvm_vm_runner(onnx_path, input_shapes, input_data)

        benchmark_test(tvm_runner, framework_name = "TVM")


def main():  # pragma: no cover
    parser = argparse.ArgumentParser(description="Performance test.")
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the Model to tune (in ONNX format).",
    )
    args = parser.parse_args()

    test_benchmark(model_path=args.model)


if __name__ == "__main__":  # pragma: no cover
    main()
