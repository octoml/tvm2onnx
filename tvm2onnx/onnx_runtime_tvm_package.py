"""ONNX package job."""
import os
import pathlib
import re
import shutil
import subprocess
import tarfile
import typing
import uuid
from tempfile import TemporaryDirectory

import numpy as np
import onnx
import structlog
import tvm
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

from tvm2onnx import package_utils, relay_model
from tvm2onnx.error import PackagingError
from tvm2onnx.utils import get_path_contents

LOG = structlog.get_logger(__name__)

_CPP_TEMPLATE_PATH = "/usr/tvm2onnx/tvm2onnx/templates/onnx_custom_op"

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
        tvm_target: str,
        relay_opt_level: int,
        build_dir: pathlib.Path,
        tvm_host_target: typing.Optional[str] = None,
    ):
        """Initializes a new package.

        :param model_name: the package name
        :param tvm_target: the target platform
        :param relay_opt_level: the optimization level
        :param build_dir: the path to the build directory
        :param host_target: Set the target.host so tvm can export cross compiled shared objects for
            non-cpu based targets.
        """
        self._model_name = sanitize_model_name(model_name)
        self._tvm_target = tvm_target
        self._tvm_host_target = tvm_host_target
        self._relay_opt_level = relay_opt_level
        self._build_dir = pathlib.Path(build_dir)

        self._tvm_dir = pathlib.Path(os.path.dirname(tvm.__file__)).parent.parent
        build_folder = self._get_build_folder(self._tvm_target, self._tvm_host_target)

        self._tvm_build_dir = self._tvm_dir / build_folder

    @staticmethod
    def _get_build_folder(tvm_target, tvm_host_target):
        """Uses the target architecture and target options to determinte the build folder
        for picking up the correct libtvm_runtime.a file
        This is static so it is easily tested.
        """
        cpu_target = tvm_host_target or tvm_target

        if "aarch64-linux-gnu" in cpu_target:
            build_cpu_part = "-aarch64"
        elif "armv8l-linux-gnueabihf" in cpu_target:
            build_cpu_part = "-aarch32"
        else:
            build_cpu_part = "-x86_64"

        backend_part = ""
        if "cuda" in tvm_target:
            backend_part += "-cuda"
        if "cudnn" in tvm_target:
            backend_part += "-cudnn"
        if "cublas" in tvm_target:
            backend_part += "-cublas"
        if "cblas" in tvm_target:
            backend_part += "-openblas"
        if "vulkan" in tvm_target:
            backend_part += "-vulkan"

        build_folder = f"build{build_cpu_part}{backend_part}"
        return build_folder

    @property
    def template_dir(self):
        """The template dir to copy and modify for this package job."""
        return _CPP_TEMPLATE_PATH

    def cookiecutter_config(
        self,
        model: relay_model.RelayModel,
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
        dl_device_type = "kDLCUDA" if "cuda" in str(self._tvm_target) else "kDLCPU"

        def _emit_element(index, name, shape, dtype) -> typing.Dict[str, typing.Any]:
            onnx_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)]
            element_count = 1
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
        #    Constants
        #    Inputs
        # All constants are first with the inputs following.
        index = 0
        for initializer, base_name in zip(initializer_tensors, tvm_constant_names):
            var_name = initializer.name
            print(f"cookiecutter constant {var_name}")
            dtype = str(onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[initializer.data_type])
            idict = _emit_element(index, var_name, initializer.dims, dtype)
            idict["base_name"] = base_name
            initializers.append(idict)
            index += 1

        for name in model.input_shapes.keys():
            shape = model.input_shapes[name]
            dtype = model.input_dtypes[name]
            var_name = f"input_{index}"
            idict = _emit_element(index, var_name, shape, dtype)
            inputs.append(idict)
            index += 1

        outputs = []
        for index, out_info in enumerate(model.get_outputs()):
            name = out_info.name
            shape = out_info.shape
            dtype = out_info.dtype
            var_name = f"output_{index}"
            idict = _emit_element(index, var_name, shape, dtype)
            outputs.append(idict)

        input_types = []
        input_shapes = []
        for iname, ishape in model.input_shapes.items():
            dtype = model.input_dtypes[iname]
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
        for out_info in model.get_outputs():
            dtype = out_info.dtype
            onnx_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)]
            output_dtypes.append(ONNXTensorElementDataType[onnx_type])
            shape_str = f"{{{', '.join(map(str, out_info.shape))}}}"
            output_shapes.append(shape_str)

        # Give the custom op a globally unique name
        self.custom_op_name = f"op_{uuid.uuid4().hex}"
        return {
            "op_name": "custom_op_library_source",
            "module_name": self._model_name,
            "custom_op_name": self.custom_op_name,
            "dl_device_type": dl_device_type,
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

    def build_package(
        self, model: relay_model.RelayModel, build_dir: pathlib.Path
    ) -> pathlib.Path:
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
        domain = "octoml.tvm"

        constants = self._build_vm(model=model, out_dir=build_dir)

        for name, data in constants.items():
            constant_name = name
            print(f"********************* constant_name {constant_name}")
            tvm_constant_names.append(name)
            np_data = data.numpy()
            constant_tensor = make_tensor(
                name=constant_name,
                data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np_data.dtype],
                dims=np_data.shape,
                vals=np_data.flatten().tobytes(),
                raw=True,
            )
            custom_op_input_names.append(constant_name)
            initializers.append(constant_tensor)

        cc_config = self.cookiecutter_config(model, initializers, domain, tvm_constant_names)
        self._create_from_template(cc_config, self._build_dir)

        source = os.path.join(build_dir, "custom_op_library_source")
        target = os.path.join(build_dir)
        shutil.move(os.path.join(source, "custom_op_library.cc"), target)
        shutil.move(os.path.join(source, "custom_op_library.h"), target)
        shutil.move(os.path.join(source, "Makefile"), target)
        make_dir = build_dir
        custom_op_name = f"custom_{self._model_name}.so"
        shutil.copy(
            os.path.join(build_dir, f"{self._model_name}.so"),
            os.path.join(build_dir, "model.so"),
        )
        with open(os.path.join(build_dir, "custom_op_library.cc"), "r") as f:
            LOG.debug("custom op library generated", code=f.read())
        result = subprocess.run(["make"], capture_output=True, cwd=make_dir, text=True)
        if not result.returncode == 0:
            err = result.stderr
            LOG.exception("Error compiling custom op library", output=err)
            raise PackagingError("Failed to build tvm custom op wrapper\n" + err)

        for name in model.input_dtypes.keys():
            sanitized_name = self._sanitize_io_name(name)
            shape = model.input_shapes[name]
            dtype = model.input_dtypes[name]
            tensortype = numpy_helper.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)]
            tensor = make_tensor_value_info(sanitized_name, tensortype, shape)
            input_tensors.append(tensor)
            custom_op_input_names.append(sanitized_name)

        for output in model.get_outputs():
            sanitized_name = self._sanitize_io_name(output.name)
            tensortype = numpy_helper.mapping.NP_TYPE_TO_TENSOR_TYPE[
                np.dtype(output.dtype)
            ]
            tensor = make_tensor_value_info(sanitized_name, tensortype, shape)
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
        convert_model_to_external_data(
            onnx_proto,
            all_tensors_to_one_file=False,
            size_threshold=1024,
            convert_attribute=True,
        )
        print(f"******************************\n{onnx_proto}")
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
            model_proto = onnx.load_model(onnx_model_file, load_external_data=True)
            print(model_proto)
            with tarfile.open(onnx_archive, "w") as onnx_tar:
                for file in get_path_contents(onnx_save_dir):
                    onnx_tar.add(os.path.join(onnx_save_dir, file), file)
                onnx_tar.add(os.path.join(build_dir, custom_op_name), custom_op_name)

            return pathlib.Path(onnx_archive)

    def build(self, model: relay_model.RelayModel) -> pathlib.Path:
        """Packages the given model.

        :param model: the model to package
        :return: the path of the generated package
        """
        package_path = self.build_package(model=model, build_dir=self._build_dir)
        return package_path

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

        package_utils.cookiecut_package(build_template_dir, build_dir, config)

    def _host_cpu_target(self):
        return self._tvm_host_target or self._tvm_target

    def _build_vm(
        self,
        model: relay_model.RelayModel,
        out_dir: pathlib.Path,
    ):
        """Exports the relay model as `{module_name}.so` in dir `out_dir`.
        Saves the Relay VM executable as `vm_exec_code.ro` in dir `out_dir`.

        :param model: the relay model to export.
        :param out_dir: path to the directory to put output in.
        """
        vm_exec, _, _ = model._create_vm_exe(
            self._tvm_target,
            self._tvm_host_target,
            self._relay_opt_level,
        )

        # This call replaces the call to move_late_bound_consts. We aren't saving the
        # constants in TVM's format, we are saving them as part of the onnx model in
        # onnx protobuf format.
        constant_map = vm_exec.get_late_bound_consts(9)

        # Save vm exec code bytes.
        vm_exec_code, mod = vm_exec.save()
        with open(out_dir / "vm_exec_code.ro", "wb") as fo:
            fo.write(vm_exec_code)

        tvm_build_func = relay_model.RelayModel.shared_object_build_func(
            self._host_cpu_target()
        )
        mod_path = str(out_dir / f"{self._model_name}.so")

        # Save module.
        mod.export_library(mod_path, fcompile=tvm_build_func)

        return constant_map
