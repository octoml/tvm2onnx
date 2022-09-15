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
from tvm2onnx.error import PackagingError
from tvm2onnx.inputs import InputDtypes, InputShapes
from tvm2onnx.utils import get_path_contents

LOG = logging.getLogger(__name__)

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
        tvm_runtime_lib: pathlib.Path,
        tvm_dynamic_libraries: typing.List[str],
        library_search_paths: typing.List[pathlib.Path],
        model_so: pathlib.Path,
        model_ro: pathlib.Path,
        constants_map: typing.Dict[str, np.ndarray],
        input_shapes: InputShapes,
        input_dtypes: InputDtypes,
        output_shapes: InputShapes,
        output_dtypes: InputDtypes,
        dl_device_type: str,
        metadata: typing.Dict[str, str] = {},
        debug_build: bool = False,
    ):
        """Initializes a new package.

        :param model_name: the package name
        :param tvm_runtime_lib: the path to the static TVM runtime lib
        :param tvm_dynamic_libraries: dynamic libraries that the TVM runtime requires
        :param library_search_paths: additional library search paths
        :param model_so: the path to the compiled model.so
        :param model_ro: the path to the compiled model.ro
        :param constants_map: the map of named constants
        :param input_shapes: the input shapes
        :param input_dtypes: the input dtypes
        :param output_shapes: the output shapes
        :param output_dtypes: the output dtypes
        :param dl_device_type: the DLDeviceType
        :param debug_build: whether to generate a debug build
        """
        self._model_name = sanitize_model_name(model_name)
        self._tvm_runtime_lib = tvm_runtime_lib
        self._tvm_dynamic_libraries = tvm_dynamic_libraries
        self._library_search_paths = library_search_paths
        self._model_so = model_so
        self._model_ro = model_ro
        self._constants_map = constants_map
        self._input_shapes = input_shapes
        self._input_dtypes = input_dtypes
        self._output_shapes = output_shapes
        self._output_dtypes = output_dtypes
        self._dl_device_type = dl_device_type
        self._metadata = metadata
        self._debug_build = debug_build

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

        :param initializer_tensors: List of initializer (constant) tensors.
        :param domain: Custom op domain.
        :return: config to apply via cookiecutter for the ONNX custom-op template.
        """

        def _emit_element(index, name, shape, dtype) -> typing.Dict[str, typing.Any]:
            onnx_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)]
            element_count = 1
            # shape can contain funky tvm types so convert everything to int
            shape = list(map(int, shape))
            for dim in shape:
                element_count *= dim
            idict: typing.Dict[str, typing.Any] = {
                "name": name,
                "shape": f"{{{', '.join(map(str, shape))}}}",
                "rank": len(shape),
                "element_count": element_count,
                "numpy_dtype": dtype,
                "index": index,
                "cpp_type": NUMPY_TO_CPP_TYPES.get(dtype),
                "onnx_type": ONNXTensorElementDataType[onnx_type],
            }
            return idict

        inputs = []
        initializers = []

        # Inputs for the custom op are ordered:
        #    Inputs
        #    Constants
        # All constants are first with the inputs following.
        index = 0
        for name in self._input_shapes.keys():
            shape = self._input_shapes[name]
            dtype = self._input_dtypes[name]
            var_name = f"input_{index}"
            idict = _emit_element(index, var_name, shape, dtype)
            inputs.append(idict)
            index += 1

        for initializer, base_name in zip(initializer_tensors, tvm_constant_names):
            var_name = initializer.name
            dtype = str(onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[initializer.data_type])
            idict = _emit_element(index, var_name, initializer.dims, dtype)
            idict["base_name"] = base_name
            initializers.append(idict)
            index += 1

        outputs = []
        for index, name in enumerate(self._output_shapes.keys()):
            shape = self._output_shapes[name]
            dtype = self._output_dtypes[name]
            var_name = f"output_{index}"
            idict = _emit_element(index, var_name, shape, dtype)
            outputs.append(idict)

        input_types = []
        input_shapes = []
        for iname, ishape in self._input_shapes.items():
            dtype = self._input_dtypes[iname]
            onnx_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)]
            input_types.append(ONNXTensorElementDataType[onnx_type])
            shape_str = f"{{{', '.join(map(str, ishape))}}}"
            input_shapes.append(shape_str)

        for index, initializer in enumerate(initializer_tensors):
            input_types.append(ONNXTensorElementDataType[initializer.data_type])
            shape_str = f"{{{', '.join(map(str, initializer.dims))}}}"
            input_shapes.append(shape_str)

        output_dtypes = []
        output_shapes = []
        for name, shape in self._output_shapes.items():
            dtype = self._output_dtypes[name]
            onnx_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)]
            output_dtypes.append(ONNXTensorElementDataType[onnx_type])
            shape_str = f"{{{', '.join(map(str, shape))}}}"
            output_shapes.append(shape_str)

        # Give the custom op a globally unique name
        self.custom_op_name = f"op_{uuid.uuid4().hex}"
        return {
            "op_name": "custom_op_library_source",
            "libtvm_runtime_a": str(self._tvm_runtime_lib),
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
            "dynamic_libraries": self._tvm_dynamic_libraries,
            "library_search_paths": [str(p) for p in self._library_search_paths],
            "debug_build": self._debug_build,
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
        """Exports the relay model as an onnx model where the relay model is
        represented as a single onnx custom operator. Constants are exported as
        onnx protobuf files.

        :param model: the relay model to export.
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

        for name in self._input_dtypes.keys():
            sanitized_name = self._sanitize_io_name(name)
            shape = self._input_shapes[name]
            dtype = self._input_dtypes[name]
            tensortype = numpy_helper.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)]
            tensor = make_tensor_value_info(sanitized_name, tensortype, shape)
            input_tensors.append(tensor)
            custom_op_input_names.append(sanitized_name)

        for name in self._output_dtypes.keys():
            sanitized_name = self._sanitize_io_name(name)
            tensortype = numpy_helper.mapping.NP_TYPE_TO_TENSOR_TYPE[
                np.dtype(self._output_dtypes[name])
            ]
            tensor = make_tensor_value_info(
                sanitized_name, tensortype, self._output_shapes[name]
            )
            output_tensors.append(tensor)
            output_names.append(sanitized_name)

        for name, np_data in self._constants_map.items():
            tvm_constant_names.append(name)
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
        self._create_from_template(cc_config, build_dir)

        source = os.path.join(build_dir, "custom_op_library_source")
        target = os.path.join(build_dir)
        shutil.move(os.path.join(source, "custom_op_library.cc"), target)
        shutil.move(os.path.join(source, "custom_op_library.h"), target)
        shutil.move(os.path.join(source, "Makefile"), target)
        try:
            shutil.copy(self._model_so, os.path.join(target, "model.so"))
        except shutil.SameFileError:
            pass
        try:
            shutil.copy(self._model_ro, os.path.join(target, "vm_exec_code.ro"))
        except shutil.SameFileError:
            pass
        make_dir = build_dir
        custom_op_name = f"custom_{self._model_name}.so"
        with open(os.path.join(build_dir, "custom_op_library.cc"), "r") as f:
            LOG.debug("custom op library generated: " + f.read())
        result = subprocess.run(["make"], capture_output=True, cwd=make_dir, text=True)
        if not result.returncode == 0:
            err = result.stderr
            LOG.error("Error compiling custom op library:\n" + err)
            raise PackagingError("Failed to build tvm custom op wrapper\n" + err)

        custom_op = make_node(
            self.custom_op_name,
            custom_op_input_names,
            output_names,
            domain=domain,
            name=self._model_name,
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
        convert_model_to_external_data(
            onnx_proto,
            all_tensors_to_one_file=False,
            size_threshold=1024,
            convert_attribute=True,
        )
        for key, value in self._metadata.items():
            meta = onnx_proto.metadata_props.add()
            meta.key = key
            meta.value = value
        # onnx_save_dir is the directory where the .onnx model file along with any
        # external constants files are written. There may be multiple files here
        # with unknown names but they all belong in the output file.
        with TemporaryDirectory() as onnx_save_dir:
            onnx_model_file = os.path.join(onnx_save_dir, f"{self._model_name}.onnx")
            onnx_archive = os.path.join(build_dir, f"{self._model_name}.onnx.tar")
            onnx.save(
                proto=onnx_proto,
                f=onnx_model_file,
                save_as_external_data=True,
                all_tensors_to_one_file=False,
                size_threshold=1024,
            )
            with tarfile.open(onnx_archive, "w") as onnx_tar:
                for file in get_path_contents(onnx_save_dir):
                    onnx_tar.add(os.path.join(onnx_save_dir, file), file)
                onnx_tar.add(os.path.join(build_dir, custom_op_name), custom_op_name)

            return pathlib.Path(onnx_archive)

    def _create_from_template(
        self, config: typing.Dict[str, str], build_dir: pathlib.Path
    ):
        """Copies this package job's template dir to the build directory and initializes
        the template. `self._template_dir` must be defined in the subclass of Package.

        :param config: the config values to provide to this template for cookiecutter.
        :param build_dir: path to the build directory.
        """
        # Copy the template dir to the build dir.
        build_template_dir = build_dir / os.path.basename(self.template_dir)
        shutil.copytree(self.template_dir, build_template_dir)

        cookiecutter.generate.generate_files(
            build_template_dir, {"cookiecutter": config}, build_dir
        )
