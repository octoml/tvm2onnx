"""Defines a representation of Relay models. Requires a full TVM build."""

from __future__ import annotations

import copy
import dataclasses
import logging
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
        input_shapes: typing.Optional[inputs.InputShapes] = None,
        input_dtypes: typing.Optional[inputs.InputDtypes] = None,
    ) -> RelayModel:
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
                        int(s) if not isinstance(s, Any) else -1 for s in shape
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
        from tvm2onnx.onnx_runtime_tvm_package import ONNXRuntimeTVMPackage

        with tempfile.TemporaryDirectory() as tdir:
            packager = ONNXRuntimeTVMPackage(
                model_name=name,
                tvm_target=tvm_target,
                build_dir=pathlib.Path(tdir),
            )
            onnx_tar = packager.build(model=self)
            shutil.move(str(onnx_tar), str(output_path))

    def compile(self, tvm_target: str, build_directory: pathlib.Path) -> typing.Any:
        vm_exec = vm.compile(
            copy.deepcopy(self.model),
            tvm_target,
            params=self.params,
        )

        constants_map = vm_exec.get_late_bound_consts(1024)

        # Save vm exec code bytes.
        ro_path = build_directory / "vm_exec_code.ro"
        vm_exec_code, mod = vm_exec.save()
        with open(ro_path, "wb") as fo:
            fo.write(vm_exec_code)

        so_path = build_directory / "model.so"

        # Save module.
        mod.export_library(str(so_path))

        return constants_map

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
            result.append(RelayTensorDetail(name=name, shape=shape, dtype=tensor.dtype))
        return result
