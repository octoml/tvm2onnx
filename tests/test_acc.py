"""
Accuracy test. Results (output tensors) from Pure ONNX Runtime and ONNX Runtime with TVM inside custom op are compared
"""

import argparse
import logging

import tempfile

from test_utils import (
    compare_outputs,
    unpack_onnx_tar,
    generate_input_shapes_dtypes,
    generate_input_data,
    get_ort_inference_session,
    compare_outputs,
)

logging.basicConfig(level=logging.CRITICAL)


def test_benchmark(
    model_path,
):
    with tempfile.TemporaryDirectory() as tmp:
        onnx_path, custom_op_libs = unpack_onnx_tar(model_path, tmp)

        input_shapes, input_dtypes = generate_input_shapes_dtypes(onnx_path)
        input_data = generate_input_data(input_shapes=input_shapes, input_dtypes=input_dtypes)
        
        # ORT with custom op
        engine = get_ort_inference_session(onnx_path, custom_op_libs)
        custom_ort_result = engine.run(output_names=None, input_feed=input_data)
        
        # pure ORT
        engine = get_ort_inference_session(onnx_path)
        pure_ort_result = engine.run(output_names=None, input_feed=input_data)
        
        # Comparison
        for c_arr, p_arr in zip(custom_ort_result, pure_ort_result):
            compare_outputs(c_arr, p_arr)


def main():  # pragma: no cover
    parser = argparse.ArgumentParser(description="Accuracy test.")
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the Model to tune (in ONNX format).",
    )
    args = parser.parse_args()

    test_benchmark(model_path=args.model)


if __name__ == "__main__":  # pragma: no cover
    main()
