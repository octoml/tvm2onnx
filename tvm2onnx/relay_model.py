#!/usr/bin/env python
#  type: ignore
"""Defines a representation of Relay models. Requires a full TVM build."""

from __future__ import annotations

import copy
import dataclasses
import json
import os
import pathlib
import shutil
import tarfile
import tempfile
import time
import typing
from typing import Optional

import structlog
import tvm
from tvm import auto_scheduler, autotvm, relay, target
from tvm.contrib import cc
from tvm.relay import vm
from tvm.runtime import vm as vm_rt
from tvm.tir.expr import Any
from tvm.utils import roofline

from tvm2onnx import relay_serializer
from tvm2onnx.error import TVM2ONNXError
from tvm2onnx.inputs import InputDtypes, InputShapes
from tvm2onnx.model_base import ModelBase
from tvm2onnx.tuning_records import (
    RecordType,
    TuningRecordsType,
    infer_record_type,
    read_tuning_records,
    write_tuning_records,
)


class RelayTensorDetail(typing.NamedTuple):
    """Details of a tensor"""

    name: str
    """Tensor name"""

    shape: typing.List[int]
    """Tensor shape"""

    dtype: str
    """Tensor data type"""

    def sanitized(self) -> RelayTensorDetail:
        """Sanitizes the tensor detail"""
        shape = []
        for dim in self.shape:
            # A dim will have dtype int64 if the original ONNX
            # model has a `Shape` operator producing int64 dims.
            # We cast to int32 to avoid producing an invalid model
            # config for Triton where dims are strings like "1i64"
            # rather than just "1".
            new_dim = -1 if isinstance(dim, Any) else int(dim)
            shape.append(new_dim)
        detail = RelayTensorDetail(
            self.name,
            shape,
            self.dtype,
        )
        return detail


LOG = structlog.get_logger(__name__)

_RELAY_OPT_LEVEL = 3

# N.B. Modify the environment (!) before compiling or tuning to
# ensure that TopHub logs are disabled.
os.environ[
    autotvm.tophub.AUTOTVM_TOPHUB_LOC_VAR
] = autotvm.tophub.AUTOTVM_TOPHUB_NONE_LOC


class RelayTuningRecordTypeError(TVM2ONNXError):
    """Indicates that there is an issue with the record type"""


class ExtractedTask(typing.NamedTuple):
    """A tuning task extracted from the high-level IR via MetaSchedule."""

    task_name: str
    """The name of the task extracted."""

    mod: tvm.ir.IRModule
    """The high-level IR."""

    dispatched: typing.List[tvm.ir.IRModule]
    """A list of low-level IRs that the high-level IR could dispatch to."""


@dataclasses.dataclass
class RelayModel(ModelBase):
    """Represents a Model in Relay format."""

    model: tvm.ir.IRModule
    params: relay_serializer.ModelParams = None
    best_records: TuningRecordsType = dataclasses.field(default_factory=list)
    output_names: typing.Optional[typing.List[str]] = None
    """Since TVM does not have named outputs we can use output_names to set a list of names
    for the output tensors. These names are applied to the outputs when the get_outputs() method
    is called. This is added to allow Triton config files outputs to have appropriate names.
    output_names is intended to be set by framework `to_relay` functions, such as when converting
    an onnx model to relay. This propagates the onnx model output names through relay."""

    @classmethod
    def from_file(
        # Must fuse module and params into one file.
        cls,
        module_and_params: pathlib.Path,
        input_shapes: InputShapes,
        input_dtypes: InputDtypes,
        best_records_path: Optional[pathlib.Path] = None,
    ) -> RelayModel:
        LOG.info(
            "Create RelayModel from file",
            module_and_params=module_and_params,
            input_shapes=input_shapes,
            input_dtypes=input_dtypes,
            best_records_path=best_records_path,
        )
        with open(module_and_params, "rb") as f:
            model_bytes = f.read()
        module, params = relay_serializer.RelaySerializer.deserialize(model_bytes)

        best_records = (
            read_tuning_records(best_records_path) if best_records_path else []
        )

        return cls(  # type: ignore
            model=module,
            input_shapes=input_shapes,
            input_dtypes=input_dtypes,
            params=params,
            best_records=best_records,
        )

    def to_file(
        self,
        module_and_params_filename: str,
        best_records_filename: Optional[pathlib.Path] = None,
    ):
        LOG.info(
            "Write RelayModel to file",
            module_and_params_filename=module_and_params_filename,
            best_records_filename=best_records_filename,
        )
        model_bytes = relay_serializer.RelaySerializer.serialize(
            (self.model, self.params)
        )
        with open(module_and_params_filename, "wb") as f:
            f.write(model_bytes)
        if best_records_filename:
            write_tuning_records(best_records_filename, self.best_records)

    def to_bytes(
        self,
    ) -> bytes:
        return relay_serializer.RelaySerializer.serialize((self.model, self.params))

    @staticmethod
    def create_tvm_filename(path: str) -> str:
        model_name = os.path.split(path).tail()
        ext_list = [
            ".tar.gz",
            ".tgz",
            ".pb",
            "tar",
        ]
        for ext in ext_list:
            if model_name.endswith(ext):
                model_name = model_name[1 : -len(ext)]
        return model_name + ".tvm"

    def to_tvm_file(
        self,
        path: str,
        best_records: typing.IO[bytes] = None,
        full_records: typing.IO[bytes] = None,
    ) -> None:
        with tarfile.open(path, "w") as model_tar:
            model_bytes = relay_serializer.RelaySerializer.serialize(
                (self.model, self.params)
            )
            with tempfile.NamedTemporaryFile("wb") as f:
                f.write(model_bytes)
                f.flush()
                model_tar.add(f.name, "model.bin")
            if best_records:
                model_tar.add(best_records.name, "best_records.log")
            elif self.best_records:
                with tempfile.NamedTemporaryFile("wb") as f:
                    write_tuning_records(f.name, self.best_records)
                    f.flush()
                    model_tar.add(f.name, "best_records.log")
            if full_records:
                model_tar.add(full_records.name, "full_records.log")
            output_shapes = {}
            output_dtypes = {}
            for obj in self.get_outputs():
                obj = obj.sanitized()
                output_shapes[obj.name] = obj.shape
                output_dtypes[obj.name] = obj.dtype
            metadata = {
                "input_shapes": self.input_shapes,
                "input_dtypes": self.input_dtypes,
                "output_shapes": output_shapes,
                "output_dtypes": output_dtypes,
            }
            with tempfile.NamedTemporaryFile("w") as f:
                json.dump(metadata, f)
                f.flush()
                model_tar.add(f.name, "metadata.json")

    @staticmethod
    def from_tvm_file(path: str) -> RelayModel:
        with tarfile.open(path, "r") as model_tar:
            with tempfile.TemporaryDirectory() as tmpdir:
                model_tar.extractall(tmpdir)
                with open(os.path.join(tmpdir, "metadata.json")) as json_file:
                    metadata = json.load(json_file)
                relay_model = RelayModel.from_file(
                    module_and_params=os.path.join(tmpdir, "model.bin"),
                    input_shapes=metadata["input_shapes"],
                    input_dtypes=metadata["input_dtypes"],
                    best_records_path=os.path.join(tmpdir, "best_records.log"),
                )
                relay_model.output_names = metadata["output_shapes"].keys()
                return relay_model

    def package_to_onnx(
        self,
        name: str,
        tvm_target: str,
        output_path: pathlib.Path,
        relay_opt_level=_RELAY_OPT_LEVEL,
        host_target: typing.Optional[str] = None,
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

        with tempfile.TemporaryDirectory() as tdir:
            packager = ONNXRuntimeTVMPackage(
                model_name=name,
                tvm_target=tvm_target,
                relay_opt_level=relay_opt_level,
                build_dir=tdir,
                tvm_host_target=host_target,
            )
            onnx_tar = packager.build(model=self)
            shutil.move(onnx_tar, output_path)

    def infer_inputs(
        self,
    ) -> typing.Tuple[InputShapes, InputDtypes]:
        """
        Returns input shapes and input dtype specified as model attributes.
        Relay models must have input shapes and input dtypes specified when constructed.

        :return: input shapes and input dtypes of this Relay Model.
        """
        return self.input_shapes, self.input_dtypes

    def infer_and_update_inputs(self):
        """
        No need to overwrites existing input info on Relay Model,
        Relay models must have input shapes and input dtypes specified when constructed.
        """
        pass

    @staticmethod
    def static_inputs(mod: tvm.ir.IRModule) -> bool:
        """Check to see if the arguments to an IRModule are statically shaped

        :param mod: The IRModule to check
        :return: True or False, the inputs to the module are static
        """
        # Check the model inputs to see if inputs are dynamic
        mod = relay.transform.InferType()(mod)
        for var in mod["main"].params:
            if "?" in str(var.checked_type.shape):
                return False
        return True

    @staticmethod
    def shared_object_build_func(host_target: str):
        """Gets a TVM build function for creating a shared object

        :param host_target: the TVM target host
        :return: a function to be passed to `export_library`
        """
        if "aarch64-linux-gnu" in host_target:
            return cc.cross_compiler("aarch64-linux-gnu-gcc")
        elif "armv8l-linux-gnueabihf" in host_target:
            return cc.cross_compiler("arm-linux-gnueabihf-gcc")
        else:
            # The above are the only cross-compile targets we currently support.
            return cc.create_shared

    def _create_vm_exe(
        self,
        tvm_target: typing.Union[str, target.Target],
        tvm_target_host: typing.Optional[str],
        opt_level: int,
    ) -> typing.Tuple[
        vm_rt.Executable, typing.Dict[tvm.ir.GlobalVar, tvm.tir.PrimFunc], float
    ]:
        """Compiles a model into a Relay VM Executable

        :param tvm_target: the TVM target for which to compile the model
        :param tvm_target_host: the TVM target host for which to compile the model
        :param opt_level: the Relay optimization level
        :return: a tuple containing a RelayVM Executable, the lowered TIR
                 functions, and the compile duration in seconds
        """
        start_compile = time.perf_counter()

        best_records = [] if self.best_records is None else self.best_records

        tvm_target_, tvm_host_target_ = target.Target.canon_target_and_host(
            tvm_target, tvm_target_host
        )

        saved_tir = roofline.SaveLoweredTIR()

        record_type = infer_record_type(best_records)
        if record_type == RecordType.AUTOTVM:
            with autotvm.apply_history_best(best_records):
                with tvm.transform.PassContext(
                    opt_level=opt_level,
                    config={"relay.FuseOps.max_depth": 30},
                    instruments=[saved_tir],
                ):
                    exe = vm.compile(
                        copy.deepcopy(self.model),
                        tvm_target_,
                        params=self.params,
                        target_host=tvm_host_target_,
                    )
        elif record_type == RecordType.AUTOSCHEDULE:
            with auto_scheduler.ApplyHistoryBest(best_records):
                with tvm.transform.PassContext(
                    opt_level=opt_level,
                    config={
                        "relay.backend.use_auto_scheduler": True,
                        "relay.FuseOps.max_depth": 30,
                    },
                    instruments=[saved_tir],
                ):
                    exe = vm.compile(
                        copy.deepcopy(self.model),
                        tvm_target_,
                        params=self.params,
                        target_host=tvm_host_target_,
                    )
        else:
            raise RelayTuningRecordTypeError(f"Unknown RecordType: {record_type}")

        end_compile = time.perf_counter()
        compile_s = end_compile - start_compile
        return exe, saved_tir.functions, compile_s

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

    @staticmethod
    def apply_relay_passes(
        mod: tvm.IRModule,
    ):
        """Applies relay passes to the input IRModule.

        :param mod: The input IRModule
        :return: The IRModule after all the relays passes have been applied
        """
        # N.B. Defer the import so as not to unconditionally require other runtimes.
        from tvm import relay, transform

        from tvm2onnx import relay_model

        passes = []

        # If the inputs are static, run DynamicToStatic to remove
        # any residual dynamism in the model.
        # If the inputs are dynamic, this pass is much more expensive
        # and will not remove dynamism from the model, so we skip it.
        if relay_model.RelayModel.static_inputs(mod):
            passes.append(relay.transform.DynamicToStatic())

        # Infer types prior to the quantization pass below as some
        # transforms might need them.
        passes.append(relay.transform.InferType())

        # Transform fake quantized sub-graphs to actual integer ops.
        # Should have no effect on graphs without the relevant patterns.
        passes.append(relay.transform.FakeQuantizationToInteger())

        # Fold constants after FQ2I becuase some weights are stored in FP32.
        passes.append(relay.transform.FoldConstant())

        # Use sequential to solve for dependent passes
        seq = transform.Sequential(passes)
        return seq(mod)
