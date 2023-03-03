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

"""Testing utilities"""
import inspect
import os
import pickle
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import typing

import decorator
import numpy as np
import onnxruntime


@decorator.decorator
def subprocessable(f, *args, **kwargs):
    def is_picklable(obj):
        try:
            pickle.dumps(obj)
        except pickle.PicklingError:
            return False
        return True

    def have_return_value(obj):
        return inspect.signature(obj).return_annotation

    if not is_picklable((args, kwargs)):
        raise TypeError(f"All parameters of {f} must be picklable.")

    if have_return_value(f):
        raise TypeError(f"Function {f} must not have a return value.")

    return f(*args, **kwargs)


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


def get_run_with_custom_op(
    onnx_model_path: str,
    custom_lib: str,
    use_io_binding: bool,
) -> typing.Callable[[typing.Dict[str, np.ndarray]], typing.List[np.ndarray]]:
    sess_options = onnxruntime.SessionOptions()
    sess_options.register_custom_ops_library(custom_lib)

    session = onnxruntime.InferenceSession(
        onnx_model_path,
        providers=["CPUExecutionProvider"],
        provider_options=[{}],
        sess_options=sess_options,
    )
    if use_io_binding:

        def run_with_io_binding(input_data: typing.Dict[str, np.ndarray]):
            for input_name in input_data.keys():
                input_data[input_name] = get_aligned_buffer(input_data[input_name])

            output_buffers = get_numpy_buffers(get_output_meta(session))

            # TODO(agladyshev): hardcoded values
            device = "cpu"
            device_id = 0

            io_binding = session.io_binding()

            for input_name, input_value in input_data.items():
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

            session.run_with_iobinding(io_binding)

            return list(output_buffers.values())

        return run_with_io_binding
    else:
        return lambda input_data: session.run(output_names=None, input_feed=input_data)


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
    output = get_run_with_custom_op(onnx_model_path, custom_lib, use_io_binding)(
        input_data
    )

    return output


def run_func_in_subprocess(func, *args, **kwargs) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
        input_data_file_name = os.path.join(temp_directory, "input_data")
        run_in_subprocess_file_name = os.path.join(
            temp_directory, "run_in_subprocess.py"
        )

        with open(input_data_file_name, "wb") as input_data_file:
            serialized_input_data = pickle.dumps((args, kwargs))
            input_data_file.write(serialized_input_data)

        module = inspect.getmodule(func)
        if not module:
            raise RuntimeError(f"Module for {func} not found.")
        module_name = module.__name__
        func_name = func.__name__
        with open(run_in_subprocess_file_name, "w") as run_in_subprocess_file:
            # TODO(agladyshev): inspect.findsource can be used to find all required imports.
            run_in_subprocess_file.write(
                textwrap.dedent(
                    f"""
                    import pickle
                    import argparse

                    import {module_name}


                    def main():
                        parser = argparse.ArgumentParser()
                        parser.add_argument(
                            "--input_data_file",
                            required=True,
                        )
                        args = parser.parse_args()

                        with open(args.input_data_file, "rb") as input_data_file:
                            args, kwargs = pickle.loads(input_data_file.read())

                        {module_name}.{func_name}(*args, **kwargs)


                    if __name__ == '__main__':
                        main()

                    """
                )
            )

        inference_cmd = [
            sys.executable,
            run_in_subprocess_file_name,
            "--input_data_file",
            input_data_file_name,
        ]
        result = subprocess.run(
            inference_cmd,
        )
        assert result.returncode == 0


@subprocessable
def package_model_and_extract_tar(
    custom_op_tar_path: str, custom_op_model_dir: str, **all_kwargs
) -> None:
    from scripts.utils.relay_model import RelayModel

    def get_function_parameters(func: typing.Callable) -> typing.List[str]:
        return [
            param.name for param in list(inspect.signature(func).parameters.values())
        ]

    def get_func_kwargs(func: typing.Callable) -> typing.Dict:
        parameters = get_function_parameters(func)
        call_dict = {
            key: all_kwargs[key] for key in parameters if key in all_kwargs.keys()
        }
        call_args = inspect.getcallargs(func, **call_dict)
        if "cls" in call_args.keys():
            del call_args["cls"]
        if "self" in call_args.keys():
            del call_args["self"]
        return call_args

    relay_model = RelayModel.from_onnx(**get_func_kwargs(RelayModel.from_onnx))
    relay_model.package_to_onnx(**get_func_kwargs(relay_model.package_to_onnx))

    with tarfile.open(custom_op_tar_path, "r") as tar:
        tar.extractall(custom_op_model_dir)
