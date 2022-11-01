"""Running inference in a subprocess to avoid dependency on libtvm.so"""
import argparse
import os
import pickle
import typing

import numpy as np
import onnxruntime


def get_io_meta(
    node_meta_data: onnxruntime.NodeArg,
) -> typing.Dict[str, typing.Dict[str, typing.Any]]:
    # TODO(agladyshev): maybe it can be defined by onnx.mapping?
    def ort_type_to_numpy_type(ort_type: str) -> str:
        ort_type_to_numpy_type_map = {
            "tensor(float16)": "float16",
            "tensor(float)": "float32",
            "tensor(double)": "float64",
            "tensor(int8)": "int8",
            "tensor(int16)": "int16",
            "tensor(int32)": "int32",
            "tensor(int64)": "int64",
            "tensor(uint8)": "uint8",
            "tensor(uint16)": "uint16",
            "tensor(uint32)": "uint32",
            "tensor(uint64)": "uint64",
        }
        if ort_type not in ort_type_to_numpy_type_map:
            raise ValueError(f"{ort_type} not found in map")
        return ort_type_to_numpy_type_map[ort_type]

    return {
        data.name: {
            "dtype": ort_type_to_numpy_type(data.type),
            "shape": tuple(data.shape),
        }
        for data in node_meta_data
    }


def get_output_meta(
    session: onnxruntime.InferenceSession,
) -> typing.Dict[str, typing.Dict[str, typing.Any]]:
    return get_io_meta(session.get_outputs())


# TODO(agladyshev): We need to find a more correct way to align memory instead of this workaround.
def get_aligned_buffer(source_buffer: np.ndarray, alignment_in_bits: int = 128):
    if (source_buffer.ctypes.data % alignment_in_bits) == 0:
        return source_buffer
    assert alignment_in_bits % source_buffer.itemsize == 0
    extra = alignment_in_bits // source_buffer.itemsize
    temp_buffer = np.empty(source_buffer.size + extra, dtype=source_buffer.dtype)
    offset = (-temp_buffer.ctypes.data % alignment_in_bits) // source_buffer.itemsize
    aligned_buffer = temp_buffer[offset : offset + source_buffer.size].reshape(
        source_buffer.shape
    )
    np.copyto(aligned_buffer, source_buffer)
    assert aligned_buffer.ctypes.data % alignment_in_bits == 0
    return aligned_buffer


def get_numpy_buffers(
    buffers_meta: typing.Dict[str, typing.Dict[str, typing.Any]]
) -> typing.Dict[str, np.ndarray]:
    return {
        name: get_aligned_buffer(np.empty(meta["shape"], meta["dtype"]))
        for name, meta in buffers_meta.items()
    }


def run(
    engine: onnxruntime.InferenceSession, input_dict: typing.Dict[str, np.ndarray]
) -> typing.List[np.ndarray]:
    run_outputs = engine.run(output_names=None, input_feed=input_dict)

    return run_outputs


def run_with_iobinding(
    engine: onnxruntime.InferenceSession,
    input_dict: typing.Dict[str, np.ndarray],
    output_buffers: typing.Dict[str, np.ndarray],
) -> None:
    # TODO(agladyshev): hardcoded values
    device = "cpu"
    device_id = 0

    io_binding = engine.io_binding()

    for input_name, input_value in input_dict.items():
        io_binding.bind_input(
            name=input_name,
            device_type=device,
            device_id=device_id,
            element_type=str(input_value.dtype),
            shape=input_value.shape,
            buffer_ptr=input_value.ctypes.data,
        )

    for output_name, output_buffer in output_buffers.items():
        io_binding.bind_output(
            name=output_name,
            device_type=device,
            device_id=device_id,
            element_type=str(output_buffer.dtype),
            shape=output_buffer.shape,
            buffer_ptr=output_buffer.ctypes.data,
        )

    engine.run_with_iobinding(io_binding)


def run_with_custom_op(
    onnx_model_path: str,
    custom_lib: str,
    input_data: typing.Dict[str, np.ndarray],
    use_io_binding: bool,
) -> typing.List[np.ndarray]:
    sess_options = onnxruntime.SessionOptions()
    sess_options.register_custom_ops_library(custom_lib)

    session = onnxruntime.InferenceSession(
        onnx_model_path,
        providers=["CPUExecutionProvider"],
        provider_options=[{}],
        sess_options=sess_options,
    )
    if use_io_binding:
        for input_name in input_data.keys():
            input_data[input_name] = get_aligned_buffer(input_data[input_name])

        output_buffers = get_numpy_buffers(get_output_meta(session))

        run_with_iobinding(session, input_data, output_buffers)

        output_data = list(output_buffers.values())
    else:
        output_data = run(session, input_data)

    return output_data


def get_ort_output(
    custom_op_model_name: str,
    custom_op_model_dir: str,
    input_data: typing.Dict[str, np.ndarray],
    use_io_binding: bool = False,
) -> typing.List[np.ndarray]:
    # Link Custom Op ONNX file and Custom Op shared library
    onnx_model_path = os.path.join(custom_op_model_dir, f"{custom_op_model_name}.onnx")
    custom_lib = os.path.join(custom_op_model_dir, f"custom_{custom_op_model_name}.so")

    # Run inference
    output = run_with_custom_op(onnx_model_path, custom_lib, input_data, use_io_binding)

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
    parser.add_argument(
        "--use_zero_copy",
        action="store_true",
        help="Use zero_copy methods for TVM and IOBinding mechanism for ORT",
    )
    args = parser.parse_args()

    with open(args.input_data_file, "rb") as input_data_file:
        input_data = pickle.loads(input_data_file.read())

    output_data = get_ort_output(
        args.custom_op_model_name,
        args.custom_op_model_dir,
        input_data,
        args.use_zero_copy,
    )

    with open(args.output_data_file, "wb") as output_data_file:
        serialized_output_data = pickle.dumps(output_data)
        output_data_file.write(serialized_output_data)


if __name__ == "__main__":
    main()
