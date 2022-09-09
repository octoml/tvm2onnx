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

from tvm2onnx.error import TVM2ONNXError
from tvm2onnx.shapes import NamedTensorShapes, TensorShape

LOG = logging.getLogger(__name__)


@dataclasses.dataclass
class RelayModel:
    """Represents a Model in Relay format."""

    model: tvm.ir.IRModule
    params: typing.Dict[str, tvm.nd.NDArray]
    inputs: NamedTensorShapes

    @classmethod
    def from_onnx(
        cls,
        onnx_model: onnx.ModelProto,
        inputs: typing.Optional[NamedTensorShapes] = None,
    ) -> RelayModel:
        if not inputs:
            # Infer from the ONNX model
            inputs = {}
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
                    inputs[input_info.name] = TensorShape(
                        dtype, [int(s) if not isinstance(s, Any) else -1 for s in shape]
                    )

        mod, params = relay.frontend.from_onnx(
            onnx_model,
            shape={name: tensor.shape for (name, tensor) in inputs.items()},
            freeze_params=True,
        )

        return cls(mod, params, inputs)

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
        :param relay_opt_level: the Relay optimization level to compile the model with.
        :param host_target: Set the target.host so tvm can export cross compiled shared objects for
            non-cpu based targets.
        """
        from tvm2onnx.onnx_runtime_tvm_package import ONNXRuntimeTVMPackage

        vm_exec = vm.compile(
            copy.deepcopy(self.model),
            tvm_target,
            params=self.params,
        )

        with tempfile.TemporaryDirectory() as tdir:
            tdir_path = pathlib.Path(tdir)
            constants_map = vm_exec.get_late_bound_consts(1024)

            # Save vm exec code bytes.
            ro_path = tdir_path / "vm_exec_code.ro"
            vm_exec_code, mod = vm_exec.save()
            with open(ro_path, "wb") as fo:
                fo.write(vm_exec_code)

            so_path = tdir_path / "model.so"

            # Save module.
            mod.export_library(str(so_path))

            packager = ONNXRuntimeTVMPackage(
                model_name=name,
                model_so=so_path,
                model_ro=ro_path,
                constants_map=constants_map,
                inputs=self.inputs,
                outputs=self.get_outputs(),
                dl_device_type="kDLCPU",
            )
            onnx_tar = packager.build_package(tdir_path)
            shutil.move(str(onnx_tar), output_path)

    def get_outputs(self) -> NamedTensorShapes:
        """Utility function to infer the IRModule outputs.
        Returns a list of tuples of output shape and output dtype
        """
        mod = relay.transform.InferType()(self.model)
        ret_type = mod["main"].ret_type
        if isinstance(ret_type, tvm.ir.type.TupleType):
            output_list = list(ret_type.fields)
        else:
            output_list = [ret_type]

        return {
            f"output_{i}": TensorShape(tensor.dtype, list(tensor.shape))
            for i, tensor in enumerate(output_list)
        }
