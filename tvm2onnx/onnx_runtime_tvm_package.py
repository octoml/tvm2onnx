"""ONNX package job."""
import logging
import os
import pathlib
import re
import shutil
import subprocess
import tarfile
import typing
import uuid
from tempfile import TemporaryDirectory

import cookiecutter.generate
import numpy as np
import onnx
from onnx import numpy_helper
from onnx.external_data_helper import convert_model_to_external_data
from onnx.helper import (
    TensorProto,
    make_graph,
    make_model,
    make_node,
    make_tensor,
    make_tensor_value_info,
)

import tvm2onnx
from tvm2onnx import shapes
from tvm2onnx.error import TVM2ONNXError
from tvm2onnx.utils import get_path_contents

LOG = logging.getLogger(__name__)


class PackagingError(TVM2ONNXError):
    """Indicates an Error occurred with model packaging."""


ONNXTensorElementDataType = [
    "ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED",
    "ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT",
    # maps to c type float
    "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8",
    # maps to c type uint8_t
    "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8",
    # maps to c type int8_t
    "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16",
    # maps to c type uint16_t
    "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16",
    # maps to c type int16_t
    "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32",
    # maps to c type int32_t
    "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64",
    # maps to c type int64_t
    "ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING",
    # maps to c++ type std::string
    "ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL",
    "ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16",
    "ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE",
    # maps to c type double
    "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32",
    # maps to c type uint32_t
    "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64",
    # maps to c type uint64_t
    "ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64",
    # complex with float32 real and imaginary components
    "ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128",
    # complex with float64 real and imaginary components
    "ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16",
    # Non-IEEE floating-point format based on IEEE754 single-precision
]

# numpy dtype to the closest cpp_type possible -- ensuring at least that
# the number of bits between numpy dtype and cpp_type match
NUMPY_TO_CPP_TYPES = {
    "float32": "float",
    "uint8": "uint8_t",
    "int8": "int8_t",
    "uint16": "uint16_t",
    "int16": "int16_t",
    "int32": "int32_t",
    "int64": "int64_t",
    "byte": "char",
    "bool": "bool",
    # NOTE: uint16_t as a cpp primitive translation of float16 is a
    # best-effort representation for random input generation only.
    "float16": "uint16_t",
    "float64": "double",
    "uint32": "uint32_t",
    "uint64": "uint64_t",
    "complex64": None,
    "complex128": None,
}


def sanitize_model_name(model_name: str):
    """Make the supplied model name python wheel friendly."""
    return re.sub(r"[^\w]", "_", model_name)


class ONNXRuntimeTVMPackage:
    """Package Job for Linux Shared Objects."""

    def __init__(
        self,
        model_name: str,
        constants_map: typing.Any,
        model_so: pathlib.Path,
        model_ro: pathlib.Path,
        inputs: shapes.NamedTensorShapes,
        outputs: shapes.NamedTensorShapes,
        dl_device_type: str = "kDLCPU",
    ):
        """Initializes a new package.

        :param model_name: the package name
        :param build_dir: the path to the build directory
        :param host_target: Set the target.host so tvm can export cross compiled shared objects for
            non-cpu based targets.
        """
        self._model_name = sanitize_model_name(model_name)
        self._model_so = model_so
        self._model_ro = model_ro
        self._dl_device_type = dl_device_type
        self._constants_map = constants_map
        self._inputs = inputs
        self._outputs = outputs

    @property
    def template_dir(self):
        """The template dir to copy and modify for this package job."""
        return pathlib.Path(tvm2onnx.__file__).parent / "templates" / "onnx_custom_op"

    def cookiecutter_config(
        self,
        initializer_tensors: typing.List[TensorProto],
        domain: str,
        tvm_constant_names: typing.List[str],
    ) -> typing.Dict[str, typing.Any]:
        """Gets the cookiecutter config for the ONNX package template.

        :param model: The relay model to be packaged.
        :param initializer_tensors: List of initializer (constant) tensors.
        :param domain: Custom op domain.
        :return: config to apply via cookiecutter for the ONNX custom-op template.
        """

        def _emit_element(
            index, name, tensor_shape: shapes.TensorShape
        ) -> typing.Dict[str, typing.Any]:
            onnx_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[
                np.dtype(tensor_shape.dtype)
            ]
            element_count = 1
            for dim in tensor_shape.shape:
                element_count *= dim
            idict: typing.Dict[str, typing.Any] = {
                "name": name,
                "shape": f"{{{', '.join(map(str, tensor_shape.shape))}}}",
                "rank": len(tensor_shape.shape),
                "element_count": element_count,
                "numpy_dtype": tensor_shape.dtype,
                "index": index,
                "cpp_type": NUMPY_TO_CPP_TYPES.get(tensor_shape.dtype),
                "onnx_type": ONNXTensorElementDataType[onnx_type],
            }
            return idict

        def _emit_initializer(
            index: int, initializer: onnx.TensorProto, tvm_constant_name: str
        ) -> typing.Dict[str, typing.Any]:
            tensor_shape = shapes.TensorShape(
                dtype=str(onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[initializer.data_type]),
                shape=initializer.dims,
            )
            idict = _emit_element(
                index,
                initializer.name,
                tensor_shape,
            )
            idict["base_name"] = tvm_constant_name
            return idict

        # Inputs for the custom op are ordered:
        #    Constants
        #    Inputs
        # All constants are first with the inputs following.
        initializers = [
            _emit_initializer(i, initializer, tvm_name)
            for (i, (initializer, tvm_name)) in enumerate(
                zip(initializer_tensors, tvm_constant_names)
            )
        ]

        inputs = [
            _emit_element(
                len(initializers) + i,
                f"input_{i}",
                tensor_shape,
            )
            for i, tensor_shape in enumerate(self._inputs.values())
        ]

        outputs = [
            _emit_element(i, f"output_{i}", tensor_shape)
            for i, tensor_shape in enumerate(self._outputs.values())
        ]

        input_types = []
        input_shapes = []
        for tensor_shape in self._inputs.values():
            onnx_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[
                np.dtype(tensor_shape.dtype)
            ]
            input_types.append(ONNXTensorElementDataType[onnx_type])
            shape_str = f"{{{', '.join(map(str, tensor_shape.shape))}}}"
            input_shapes.append(shape_str)

        for initializer in initializer_tensors:
            input_types.append(ONNXTensorElementDataType[initializer.data_type])
            shape_str = f"{{{', '.join(map(str, initializer.dims))}}}"
            input_shapes.append(shape_str)

        output_dtypes = []
        output_shapes = []
        for tensor_shape in self._outputs.values():
            onnx_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[
                np.dtype(tensor_shape.dtype)
            ]
            output_dtypes.append(ONNXTensorElementDataType[onnx_type])
            shape_str = f"{{{', '.join(map(str, tensor_shape.shape))}}}"
            output_shapes.append(shape_str)

        # Give the custom op a globally unique name
        self.custom_op_name = f"op_{uuid.uuid4().hex}"
        return {
            "op_name": "custom_op_library_source",
            "module_name": self._model_name,
            "custom_op_name": self.custom_op_name,
            "dl_device_type": self._dl_device_type,
            "input_count": str(len(inputs)),
            "output_count": str(len(outputs)),
            "initializer_count": str(len(initializers)),
            "input_types": input_types,
            "output_types": output_dtypes,
            "input_shapes": input_shapes,
            "output_shapes": output_shapes,
            "inputs": inputs,
            "outputs": outputs,
            "initializers": initializers,
            "domain": domain,
        }

    def _sanitize_io_name(self, name: str) -> str:
        """Strip trailing ":<NUMBER>" from names

        :param name: the input/output name to sanitize
        :return: sanitized name
        """
        colon_index = name.rfind(":")
        if colon_index > 0:
            name = name[:colon_index]
        return name

    def build_package(self, build_dir: pathlib.Path) -> pathlib.Path:
        """Exports the compiled relay model as an onnx model where the relay model is
        represented as a single onnx custom operator. Constants are exported as onnx
        protobuf files.

        :param build_dir: path to the directory to put output in.
        """
        input_tensors = []
        output_tensors = []
        output_names = []
        initializers = []
        graph_nodes = []
        custom_op_input_names = []
        tvm_constant_names = []
        domain = "octoml.ai"

        for name, data in self._constants_map.items():
            tvm_constant_names.append(name)
            np_data = data.numpy()
            constant_tensor = make_tensor(
                name=name,
                data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np_data.dtype],
                dims=np_data.shape,
                vals=np_data.flatten().tobytes(),
                raw=True,
            )
            custom_op_input_names.append(name)
            initializers.append(constant_tensor)

        cc_config = self.cookiecutter_config(initializers, domain, tvm_constant_names)
        cookiecutter.generate.generate_files(
            self.template_dir, {"cookiecutter": cc_config}, build_dir
        )

        source_dir = build_dir / cc_config["op_name"]
        shutil.copy(self._model_so, source_dir / "model.so")
        shutil.copy(self._model_ro, source_dir / "vm_exec_code.ro")
        custom_op_name = f"custom_{self._model_name}.so"
        with open(source_dir / "custom_op_library.cc", "r") as f:
            LOG.debug(f"custom op library generated: {f.read()}")
        result = subprocess.run(
            ["make"], capture_output=True, cwd=source_dir, text=True
        )
        if not result.returncode == 0:
            raise PackagingError(
                "Failed to build tvm custom op wrapper:\n" + result.stderr
            )

        for name, tensor_shape in self._inputs.items():
            sanitized_name = self._sanitize_io_name(name)
            tensortype = numpy_helper.mapping.NP_TYPE_TO_TENSOR_TYPE[
                np.dtype(tensor_shape.dtype)
            ]
            tensor = make_tensor_value_info(
                sanitized_name, tensortype, tensor_shape.shape
            )
            input_tensors.append(tensor)
            custom_op_input_names.append(sanitized_name)

        for name, tensor_shape in self._outputs.items():
            sanitized_name = self._sanitize_io_name(name)
            tensortype = numpy_helper.mapping.NP_TYPE_TO_TENSOR_TYPE[
                np.dtype(tensor_shape.dtype)
            ]
            tensor = make_tensor_value_info(
                sanitized_name, tensortype, tensor_shape.shape
            )
            output_tensors.append(tensor)
            output_names.append(sanitized_name)

        custom_op = make_node(
            self.custom_op_name,
            custom_op_input_names,
            output_names,
            domain=domain,
        )
        graph_nodes.append(custom_op)

        graph = make_graph(
            nodes=graph_nodes,
            name=f"{self._model_name}_{uuid.uuid4().hex}",
            inputs=input_tensors,
            outputs=output_tensors,
            initializer=initializers,
        )

        onnx_proto = make_model(graph)
        # TODO: rkimball Can't check because of the custom op.
        # onnx.checker.check_model(onnx_proto)
        # onnx_save_dir is the directory where the .onnx model file along with any
        # external constants files are written. There may be multiple files here
        # with unknown names but they all belong in the output file.
        with TemporaryDirectory() as onnx_save_dir:
            onnx_model_file = pathlib.Path(onnx_save_dir) / f"{self._model_name}.onnx"
            onnx_archive = build_dir / f"{self._model_name}.onnx.tar"
            onnx.save_model(
                proto=onnx_proto,
                f=str(onnx_model_file),
                save_as_external_data=True,
                all_tensors_to_one_file=False,
                size_threshold=1024,
                convert_attribute=True,
            )
            with tarfile.open(onnx_archive, "w") as onnx_tar:
                for file in get_path_contents(onnx_save_dir):
                    onnx_tar.add(os.path.join(onnx_save_dir, file), file)
                onnx_tar.add(source_dir / custom_op_name, custom_op_name)

            return onnx_archive
