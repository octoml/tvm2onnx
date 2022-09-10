"""Defines a representation of Relay models. Requires a full TVM build."""

from __future__ import annotations

import copy
import dataclasses
import logging
import os
import pathlib
import shutil
import tempfile
import typing

import onnx
import tvm
from tvm import relay
from tvm.relay import vm
from tvm.tir.expr import Any

from tvm2onnx import inputs
from tvm2onnx.error import TVM2ONNXError
from tvm2onnx.onnx_runtime_tvm_package import ONNXRuntimeTVMPackage

LOG = logging.getLogger(__name__)


class RelayTensorDetail(typing.NamedTuple):
    """Details of a tensor"""

    name: str
    """Tensor name"""

    shape: typing.List[int]
    """Tensor shape"""

    dtype: str
    """Tensor data type"""


@dataclasses.dataclass
class RelayModel:
    """Represents a Model in Relay format."""

    model: tvm.ir.IRModule
    params: typing.Dict[str, tvm.nd.NDArray]
    input_shapes: inputs.InputShapes
    input_dtypes: inputs.InputDtypes
    output_names: typing.List[str]

    @classmethod
    def from_onnx(
        cls,
        onnx_model: onnx.ModelProto,
        dynamic_axis_substitute: int = -1,
        input_shapes: typing.Optional[inputs.InputShapes] = None,
        input_dtypes: typing.Optional[inputs.InputDtypes] = None,
    ) -> RelayModel:
        """Imports an ONNX model into Relay.

        :param onnx_model: the in-memory ONNX model
        :param dynamic_axis_substitute: the value to use to represent dynamic axes in shapes
        :param input_shapes: optional input shapes to use
        :param input_dtypes: optional input dtypes to use
        :return: a RelayModel
        """

        if not input_shapes or not input_dtypes:
            # Infer from the ONNX model
            input_shapes = {}
            input_dtypes = {}
            initializer_names = [n.name for n in onnx_model.graph.initializer]
            # The inputs contains both the inputs and parameters. We are just interested in the
            # inputs so skip all parameters listed in graph.initializer
            for input_info in onnx_model.graph.input:
                if input_info.name not in initializer_names:
                    _, shape, dtype, _ = relay.frontend.onnx.get_info(input_info)
                    if dtype is None:
                        raise TVM2ONNXError(
                            f"Unknown dtype on input '{input_info.name}' is not supported.",
                            {"inputs": str(input_info.name)},
                        )

                    # Normalize the shape dimensions to integers
                    input_shapes[input_info.name] = [
                        int(s) if not isinstance(s, Any) else 1 for s in shape
                    ]
                    input_dtypes[input_info.name] = dtype

        mod, params = relay.frontend.from_onnx(
            onnx_model,
            shape=input_shapes,
            freeze_params=True,
        )

        return cls(
            mod,
            params,
            input_shapes,
            input_dtypes,
            [tensor.name for tensor in onnx_model.graph.output],
        )

    def package_to_onnx(
        self,
        name: str,
        tvm_target: str,
        output_path: pathlib.Path,
    ):
        """Builds the ONNX file and returns the path to the package.

        :param name: ONNX model name
        :param tvm_target: TVM target for model compile.
        :param output_path: path to the target output file.
        """
        with tempfile.TemporaryDirectory() as tdir:
            tdir_path = pathlib.Path(tdir)

            vm_exec = vm.compile(
                copy.deepcopy(self.model),
                tvm_target,
                params=self.params,
            )

            constants_map = {
                name: data.numpy()
                for name, data in vm_exec.get_late_bound_consts(1024).items()
            }

            # Save vm exec code bytes.
            ro_path = tdir_path / "vm_exec_code.ro"
            vm_exec_code, mod = vm_exec.save()
            with open(ro_path, "wb") as fo:
                fo.write(vm_exec_code)

            so_path = tdir_path / "model.so"

            # Save module.
            mod.export_library(str(so_path))

            libtvm_runtime_a = (
                pathlib.Path(os.environ["TVM_HOME"]) / "build" / "libtvm_runtime.a"
            )
            outputs = self.get_outputs()
            packager = ONNXRuntimeTVMPackage(
                model_name=name,
                libtvm_runtime_a=libtvm_runtime_a,
                model_so=so_path,
                model_ro=ro_path,
                constants_map=constants_map,
                input_shapes=self.input_shapes,
                input_dtypes=self.input_dtypes,
                output_shapes={t.name: t.shape for t in outputs},
                output_dtypes={t.name: t.dtype for t in outputs},
                dl_device_type="kDLCUDA" if "cuda" in tvm_target else "kDLCPU",
            )
            onnx_tar = packager.build_package(tdir_path)
            shutil.move(str(onnx_tar), str(output_path))

    def get_outputs(self) -> typing.List[RelayTensorDetail]:
        """Utility function to infer the IRModule outputs.
        Returns a list of tuples of output shape and output dtype
        """
        mod = relay.transform.InferType()(self.model)
        ret_type = mod["main"].ret_type
        if isinstance(ret_type, tvm.ir.type.TupleType):
            output_list = list(ret_type.fields)
        else:
            output_list = [ret_type]
        result = []
        # Convert tvm types to python types
        if not self.output_names:
            output_names = [f"output{i}" for i in range(len(output_list))]
        else:
            output_names = self.output_names
        for name, tensor in zip(output_names, output_list):
            shape = [dim for dim in tensor.shape]
            shape = list(map(int, shape))
            result.append(RelayTensorDetail(name=name, shape=shape, dtype=tensor.dtype))
        return result
