"""Running inference in a subprocess to avoid dependency on libtvm.so"""
import argparse
import os
import pickle
import typing

import numpy as np
import onnxruntime


def run_with_custom_op(
    onnx_model_path: str,
    custom_lib: str,
    input_data: typing.Dict[str, np.ndarray],
) -> typing.List[np.ndarray]:
    sess_options = onnxruntime.SessionOptions()
    sess_options.register_custom_ops_library(custom_lib)

    session = onnxruntime.InferenceSession(
        onnx_model_path,
        providers=["CPUExecutionProvider"],
        provider_options=[{}],
        sess_options=sess_options,
    )
    output_data = session.run(output_names=None, input_feed=input_data)

    return output_data


def get_ort_output(
    custom_op_model_name: str,
    custom_op_model_dir: str,
    input_data: typing.Dict[str, np.ndarray],
) -> typing.List[np.ndarray]:
    # Link Custom Op ONNX file and Custom Op shared library
    onnx_model_path = os.path.join(custom_op_model_dir, f"{custom_op_model_name}.onnx")
    custom_lib = os.path.join(custom_op_model_dir, f"custom_{custom_op_model_name}.so")

    # Run inference
    output = run_with_custom_op(onnx_model_path, custom_lib, input_data)

    return output


def main():
    parser = argparse.ArgumentParser(description="Run inference for Custom Op.")
    parser.add_argument(
        "--custom_op_model_name",
        required=True,
        help="Name of the model.",
    )
    parser.add_argument(
        "--custom_op_model_dir",
        required=True,
        help="Directory with ONNX file and Custom Op shared library.",
    )
    parser.add_argument(
        "--input_data_file",
        required=True,
        help="Serialized input dictionary.",
    )
    parser.add_argument(
        "--output_data_file",
        required=True,
        help="Serialized output",
    )
    args = parser.parse_args()

    with open(args.input_data_file, "rb") as input_data_file:
        input_data = pickle.loads(input_data_file.read())

    output_data = get_ort_output(
        args.custom_op_model_name, args.custom_op_model_dir, input_data
    )

    with open(args.output_data_file, "wb") as output_data_file:
        serialized_output_data = pickle.dumps(output_data)
        output_data_file.write(serialized_output_data)


if __name__ == "__main__":
    main()
