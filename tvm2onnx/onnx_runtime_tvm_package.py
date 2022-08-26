#!/usr/bin/env python
#  type: ignore
"""ONNX package job."""
import os
import pathlib
import re
import shutil
import subprocess
import tarfile
import typing
import uuid

import numpy as np
import onnx
import structlog
import tvm
from onnx import numpy_helper
from onnx.external_data_helper import convert_model_to_external_data
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info, make_tensor
from onnx import TensorProto

from tvm2onnx import package_utils, relay_model, relay_model_runtime
from tvm2onnx.error import PackagingError
from tvm2onnx.utils import print_path_contents

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
        for picking up the correct libtvm_runtime.so file
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

    def _sanitize_outputs(
        self, tensors: typing.Sequence[relay_model.RelayTensorDetail]
    ) -> typing.Sequence[relay_model.RelayTensorDetail]:
        """Sanitizes the outputs of the model."""
        output_details = []
        for tensor in tensors:
            shape = []
            for dim in tensor.shape:
                # A dim will have dtype int64 if the original ONNX
                # model has a `Shape` operator producing int64 dims.
                # We cast to int32 to avoid producing an invalid model
                # config for Triton where dims are strings like "1i64"
                # rather than just "1".
                new_dim = -1 if isinstance(dim, tvm.tir.expr.Any) else int(dim)
                shape.append(new_dim)
            detail = relay_model.RelayTensorDetail(
                tensor.name,
                shape,
                tensor.dtype,
            )
            output_details.append(detail)
        return output_details

    def cookiecutter_config(
        self, model: relay_model.RelayModel
    ) -> typing.Dict[str, typing.Any]:
        """Gets the cookiecutter config for the ONNX package template.

        :param module_name: The module name.
        :param model: The relay model to be packaged.
        :return: config to apply via cookiecutter for the ONNX custom-op template.
        """
        dl_device_type = "kDLCUDA" if "cuda" in str(self._tvm_target) else "kDLCPU"

        inputs = []
        for index, name in enumerate(model.input_shapes.keys()):
            shape = model.input_shapes[name]
            dtype = model.input_dtypes[name]
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
            inputs.append(idict)

        outputs = []
        for index, out_info in enumerate(model.get_outputs()):
            name = out_info.name
            shape = out_info.shape
            dtype = out_info.dtype
            onnx_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)]
            element_count = 1
            for dim in shape:
                element_count *= dim
            idict: typing.Dict[str, typing.Any] = {
                "name": name,
                "shape": f"{{{', '.join(map(str, out_info.shape))}}}",
                "rank": len(shape),
                "element_count": element_count,
                "numpy_dtype": dtype,
                "index": index,
                "cpp_type": NUMPY_TO_CPP_TYPES.get(dtype),
                "onnx_type": ONNXTensorElementDataType[onnx_type],
            }
            outputs.append(idict)

        input_types = []
        input_shapes = []
        for iname, ishape in model.input_shapes.items():
            dtype = model.input_dtypes[iname]
            onnx_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)]
            input_types.append(ONNXTensorElementDataType[onnx_type])
            shape = f"{{{', '.join(map(str, ishape))}}}"
            input_shapes.append(shape)

        output_dtypes = []
        output_shapes = []
        for out_info in model.get_outputs():
            dtype = out_info.dtype
            onnx_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)]
            output_dtypes.append(ONNXTensorElementDataType[onnx_type])
            shape = f"{{{', '.join(map(str, out_info.shape))}}}"
            output_shapes.append(shape)

        # Give the custom op a globally unique name
        self.custom_op_name = f"op_{uuid.uuid4().hex}"
        return {
            "op_name": "custom_op_library_source",
            "module_name": self._model_name,
            "custom_op_name": self.custom_op_name,
            "consts_name": "consts",
            "dl_device_type": dl_device_type,
            "input_count": str(len(inputs)),
            "output_count": str(len(outputs)),
            "input_types": "{" + (", ".join(input_types)) + "}",
            "output_types": "{" + (", ".join(output_dtypes)) + "}",
            "input_shapes": f"{{{', '.join(input_shapes)}}}",
            "output_shapes": f"{{{', '.join(output_shapes)}}}",
            "inputs": inputs,
            "outputs": outputs,
            "vm_exec_path": "vm_exec_code.ro",
            "tvm_model_lib": f"{self._model_name}.so",
        }

    def config_file(self, source: str, target: str, config: typing.Dict[str, str]):
        with open(source, "r") as f:
            content = f.read()
            for tag, replacement in config.items():
                content = re.sub(
                    pattern=f"\\${tag}\\$",
                    repl=replacement,
                    string=content,
                    flags=re.RegexFlag.MULTILINE,
                )
            with open(target, "w") as out:
                out.write(content)

    def _load_late_bound_constants(self, consts_path):
        """This function is horrific. This is the only way I can get constants from
        tvm. All I can say is that it works."""
        # https://docs.python.org/3/library/struct.html
        import ctypes
        import struct

        kTVMNDArrayListMagic = int(hex(0xF7E58D4F05049CB7), 16)
        kTVMNDArrayMagic = int(hex(0xDD5E40F096B4A13F), 16)

        class DLDataType(ctypes.Structure):
            TYPE_MAP = {
                (1, 1, 1): "bool",
                (0, 32, 1): "int32",
                (0, 64, 1): "int64",
                (1, 32, 1): "uint32",
                (1, 64, 1): "uint64",
                (2, 32, 1): "float32",
                (2, 64, 1): "float64",
            }

        constants = {}
        names = []
        with open(consts_path, "rb") as f:
            magic = struct.unpack("Q", f.read(8))[0]
            if magic != kTVMNDArrayListMagic:
                raise PackagingError("No magic in consts file")
            reserved = struct.unpack("Q", f.read(8))[0]
            # std::vector<std::string> names;
            # ICHECK(strm->Read(&names)) << "Invalid parameters file format";
            name_count = struct.unpack("Q", f.read(8))[0]
            for i in range(name_count):
                name_length = struct.unpack("Q", f.read(8))[0]
                name = f.read(name_length).decode("utf-8")
                print(f"name[{i}] = {name}")
                names.append(name)
            data_count = struct.unpack("Q", f.read(8))[0]
            for i in range(data_count):
                magic = struct.unpack("Q", f.read(8))[0]
                if magic != kTVMNDArrayMagic:
                    raise PackagingError("Data not array")
                print("ndarray")
                reserved = struct.unpack("Q", f.read(8))[0]
                f.read(reserved)  # skip reserved space
                # DLDevice device;
                device_type = struct.unpack("I", f.read(4))[0]
                device_id = struct.unpack("I", f.read(4))[0]
                print(f"device type {device_type}[{device_id}]")
                # int ndim
                ndim = struct.unpack("I", f.read(4))[0]
                # DLDataType dtype;
                dtype_code = struct.unpack("B", f.read(1))[0]
                dtype_bits = struct.unpack("B", f.read(1))[0]
                dtype_lanes = struct.unpack("H", f.read(2))[0]
                shape = []
                for dim in range(ndim):
                    axis = struct.unpack("Q", f.read(8))[0]
                    shape.append(axis)
                data_byte_size = struct.unpack("Q", f.read(8))[0]
                data = f.read(data_byte_size)
                dtype_str = (dtype_code, dtype_bits, dtype_lanes)
                dtype = DLDataType.TYPE_MAP[dtype_str]
                array = np.ndarray(shape=shape, dtype=dtype, buffer=data)
                constants[names[i]] = array
        return constants

    def _export_model(self, model: relay_model.RelayModel, out_dir: pathlib.Path):
        """Exports the relay model as an onnx model where the relay model is
        represented as a single onnx custom operator. Constants are exported as
        onnx protobuf files.

        :param model: the relay model to export.
        :param out_dir: path to the directory to put output in.
        """
        self._build_vm(model=model, out_dir=out_dir)

        constants = self._load_late_bound_constants(os.path.join(out_dir, "consts"))
        print_path_contents(out_dir)
        print(constants)

        input_tensors = []
        input_names = []
        initializers = []
        for name in model.input_dtypes.keys():
            shape = model.input_shapes[name]
            dtype = model.input_dtypes[name]
            tensortype = numpy_helper.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)]
            tensor = make_tensor_value_info(name, tensortype, shape)
            input_tensors.append(tensor)
            input_names.append(name)

        output_tensors = []
        output_names = []
        for output in model.get_outputs():
            tensortype = numpy_helper.mapping.NP_TYPE_TO_TENSOR_TYPE[
                np.dtype(output.dtype)
            ]
            tensor = make_tensor_value_info(output.name, tensortype, None)
            output_tensors.append(tensor)
            output_names.append(output.name)

        # Test to see if we can pass arbitrary data through ort as a constant
        test_string = "Hello World!"
        hello_world_data = bytes(test_string, 'utf-8')

        hello_world = make_node(
            "Constant",
            inputs=[],
            outputs=["c1"],
            name="hello_world_data",
            value=make_tensor(
                name="hello_world",
                data_type=TensorProto.INT8,
                dims=[len(hello_world_data)],
                vals=hello_world_data,
                raw=True,
            ),
        )

        custom_op = make_node(
            self.custom_op_name, input_names, output_names, domain="octoml.customop"
        )

        graph = make_graph(
            nodes=[custom_op, hello_world],
            name="tvm_ort",
            inputs=input_tensors,
            outputs=output_tensors,
            initializer=initializers,
        )

        onnx_model = make_model(graph)
        convert_model_to_external_data(
            onnx_model,
            all_tensors_to_one_file=False,
            size_threshold=1024,
            convert_attribute=True,
        )
        onnx.save(onnx_model, os.path.join(out_dir, f"{self._model_name}.onnx"))
        onnx.save(onnx_model, "hello_world.onnx")

    def build_package(self, build_dir: pathlib.Path) -> pathlib.Path:
        """Builds the ONNX file and returns the path to the package.

        :param module_name: string of the module to build a package around.
        :param build_dir: path to the build directory.
        :return: path of the generated package.
        """
        LOG.info("Begin building ONNX package", build_dir=build_dir)
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
        result = subprocess.run(["make"], capture_output=True, cwd=make_dir)
        if not result.returncode == 0:
            err = result.stderr.decode("utf-8").replace(r"\n", "\n")
            raise PackagingError("Failed to build tvm custom op wrapper\n" + err)

        files = [
            "consts",
            custom_op_name,
        ]

        files.append(f"{self._model_name}.onnx")

        onnx_path = os.path.join(build_dir, "$tempfile$.onnx")
        if os.path.exists(onnx_path):
            os.remove(onnx_path)
        with tarfile.open(onnx_path, "w") as model_tar:
            for file in files:
                model_tar.add(os.path.join(build_dir, file), file)
        return onnx_path

    def build(self, model: relay_model.RelayModel) -> pathlib.Path:
        """Packages the given model.

        :param model: the model to package
        :return: the path of the generated package
        """
        LOG.info(
            "Begin building package",
            model_name=self._model_name,
            tvm_target=self._tvm_target,
            relay_opt_level=self._relay_opt_level,
            build_dir=self._build_dir,
            tvm_dir=self._tvm_dir,
            tvm_build_dir=self._tvm_build_dir,
        )
        cc_config = self.cookiecutter_config(model)
        self._create_from_template(cc_config, self._build_dir)
        self._export_model(model=model, out_dir=self._build_dir)
        package_path = self.build_package(self._build_dir)
        LOG.info("Package complete", package_path=package_path)
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

        # Save consts separately. This is done to support very large models which
        # can't be compiled to a single binary, and there is no harm in doing it
        # for smaller models as well.
        consts_path = str(out_dir / relay_model_runtime.RELAY_VM_CONSTS_NAME)
        vm_exec.move_late_bound_consts(
            consts_path, byte_limit=relay_model_runtime.RELAY_VM_LARGE_CONST_BYTE_LIMIT
        )

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
