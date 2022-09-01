#!/usr/bin/env python
#  type: ignore
"""Defines a representation of ONNX models."""

from __future__ import annotations

import contextlib
import dataclasses
import os
import pathlib
import sys
import tarfile
import tempfile
import typing

import onnx
import structlog
from onnx.external_data_helper import convert_model_to_external_data

from tvm2onnx.error import RelayConvertError, RelayOpNotImplementedError, TVM2ONNXError
from tvm2onnx.inputs import InputDtypes, InputShapes, generate_static_shapes
from tvm2onnx.model_base import ModelBase

# Ensure we don't unconditionally require other runtimes.
if typing.TYPE_CHECKING:
    from tvm2onnx import relay_model

LOG = structlog.get_logger(__name__)

_CUDA_DEVICE_STR = "CUDA:0"
"""The CUDA device string passed to ONNX TRT for inference."""

_ONNX_SUFFIX = ".onnx"
"""The suffix associated with raw ONNX model files."""

_MACOS_EXTENDED_ATTRIBUTE_FILE_PREFIX = "._"
"""The prefix of files created inside tars to store MacOS extra file info."""


class ONNXIngestError(TVM2ONNXError):
    """Indicates an error occurred with model ingest, maybe it's not quite delicious enough."""


class ONNXNotTarFileError(TVM2ONNXError):
    """Indicates that the provided resource does not represent a tarfile."""


class ONNXIngestNoModelInTarFileError(TVM2ONNXError):
    """Indicates that the extracted tarfile does not contain a model."""


class ONNXIngestMultipleAssetsInTarFileError(TVM2ONNXError):
    """Indicates that the extracted tarfile contains more than one asset."""


class ONNXInferInputsNoneFoundError(TVM2ONNXError):
    """Indicates that that no inputs were found for the model."""


class ONNXInferInputsUnknownDataTypeError(TVM2ONNXError):
    """Indicates that we found inputs with unknown data types."""


@dataclasses.dataclass
class ONNXModel(ModelBase):
    """Represents a Model in ONNX format."""

    custom_op_libs: typing.List[str] = dataclasses.field(default_factory=list)

    @classmethod
    def from_file(
        cls,
        model_path: pathlib.Path,
        custom_op_libs: typing.Optional[typing.List[str]] = [],
    ) -> ONNXModel:
        """Loads an ONNXModel from the given path.

        :param model_path: the path to a file containing an ONNXModel.
        :param custom_op_libs: optional list of paths to custom op libraries.
        :return: the ONNXModel loaded from the given file path.
        :raise ONNXIngestError: if the model could not be loaded.
        """
        LOG.info("Loading an ONNXModel from file.", model_path=model_path)
        if not os.path.exists(model_path):
            raise ONNXIngestError(f"File '{model_path}' not found")
        try:
            onnx_proto = ONNXModel._load_model_from_tar_file(model_tar=model_path)
            LOG.info("ONNXModel successfully loaded from tar file")
        except ONNXNotTarFileError:
            try:
                onnx_proto = onnx.load_model(str(model_path))
                LOG.info("ONNXModel successfully loaded from file")
            except Exception as e:
                LOG.exception("Failed loading ONNXModel from file")
                raise ONNXIngestError(
                    "Unable to load ONNX model.", {"error": str(e)}
                ) from e
        return cls(
            model=onnx_proto,
            custom_op_libs=custom_op_libs,
        )

    @staticmethod
    def _load_model_from_tar_file(model_tar: pathlib.Path) -> onnx.ModelProto:
        """Extracts an onnx model from the given bytes if they represent a tarfile.

        :param model_bytes: the bytes to extract a model from.
        :return: the onnx model extracted from the tarfile bytes.
        :raise ONNXNotTarFileError: if the bytes do not represent a tarfile.
        :raise ONNXIngestNoModelInTarFileError: if there is no onnx model in the tarfile.
        :raise ONNXIngestMultipleAssetsInTarFileError: if there is more than one file ending in
            .onnx within the tarfile, apart from MacOS metadata files.
        """
        if not tarfile.is_tarfile(str(model_tar)):
            LOG.info("Model bytes do not represent a tarfile. Halting tar extraction.")
            raise ONNXNotTarFileError("Model bytes don't represent a tarfile.")

        LOG.info("Extracting ONNX model from given tarfile byte.")
        with contextlib.ExitStack() as stack:
            model_tar = stack.enter_context(tarfile.open(str(model_tar)))
            members = model_tar.getmembers()
            onnx_members = [
                m
                for m in members
                # This picks out any ONNX model files.
                if m.name.endswith(_ONNX_SUFFIX)
                # This filters MacOS extended attribute files which also end in
                # `.onnx`. For example, if you tar `mnist.onnx` on MacOS, upon
                # programmatic extraction there will be both an `mnist.onnx`
                # and an `._mnist.onnx` within. Only the former is useful to us,
                # and the latter should be ignored.
                # Pathlib is additionally used to make sure this works even for
                # extracted files with a folder prefix, by only looking at the
                # base name.
                and not pathlib.Path(m.name).name.startswith(
                    _MACOS_EXTENDED_ATTRIBUTE_FILE_PREFIX
                )
            ]

            if not onnx_members:
                raise ONNXIngestNoModelInTarFileError(
                    "No .onnx files found in given tarfile."
                )
            if len(onnx_members) > 1:
                onnx_file_names = ", ".join(sorted(o.name for o in onnx_members))
                raise ONNXIngestMultipleAssetsInTarFileError(
                    f"Multiple .onnx files found in given tarfile - {onnx_file_names}.",
                    {"files": onnx_file_names},
                )
            onnx_file = onnx_members[0]

            # It's not safe to use the "extractall" API of TarFile because
            # it allows files to extract themselves into system directories.
            # Instead, we manually extract and save the files to a tempdir.
            tempdir = stack.enter_context(tempfile.TemporaryDirectory())
            for member in members:
                if member.isfile():
                    file_bytes = model_tar.extractfile(member).read()  # type: ignore
                    basename = os.path.basename(member.name)
                    with open(os.path.join(tempdir, basename), "wb") as f:
                        f.write(file_bytes)
            return onnx.load(os.path.join(tempdir, os.path.basename(onnx_file.name)))

    def infer_inputs(
        self,
    ) -> typing.Tuple[InputShapes, InputDtypes]:
        """
        Infers this model's input shapes and input dtypes.

        :return: input shapes and input dtypes of this ONNX Model.
        :raises: ONNXInferInputsUnknownDataType when dtype is unknown.
        """
        # N.B. Defer the import so as not to unconditionally require other runtimes.
        from tvm import relay
        from tvm.tir import Any as Any

        input_shapes: InputShapes = {}
        input_dtypes: InputDtypes = {}
        initializer_names = [n.name for n in self.model.graph.initializer]
        # The inputs contains both the inputs and parameters. We are just interested in the
        # inputs so skip all parameters listed in graph.initializer
        for input_info in self.model.graph.input:
            if input_info.name not in initializer_names:
                _, shape, dtype, _ = relay.frontend.onnx.get_info(input_info)
                if dtype is None:
                    raise ONNXInferInputsUnknownDataTypeError(
                        f"Unknown dtype on input '{input_info.name}' is not supported.",
                        {"inputs": str(input_info.name)},
                    )

                # Normalize the shape dimensions to integers
                input_shapes.update(
                    {
                        input_info.name: [
                            int(s) if not isinstance(s, Any) else -1 for s in shape
                        ]
                    }
                )
                input_dtypes.update({input_info.name: dtype})

        return input_shapes, input_dtypes

    def infer_and_update_inputs(self):
        """Modifies the input_shapes, input_dtypes on this model with inferred info.
        Overwrites existing input info on this model.
        :raises: ONNXInferInputsError when dtype is unknown.
        """
        input_shapes, input_dtypes = self.infer_inputs()
        if not input_shapes or not input_dtypes:
            raise ONNXInferInputsNoneFoundError("No inputs found in the model.")

        self.input_shapes = input_shapes
        self.input_dtypes = input_dtypes

    def to_file(self, model_path: pathlib.Path, save_as_external_data=False) -> None:
        """Saves the model to a file.

        :param model_path: location to save the file to
        :param save_as_external_data: Sets all tensors with raw data as external data.
        """
        LOG.info(f"Write model to file '{model_path}'")

        model = self.model

        if save_as_external_data:
            # Copy the model because converting to external data changes
            # the model structure when converting to relay
            model = onnx.ModelProto()
            model.CopyFrom(self.model)
            convert_model_to_external_data(
                model,
                all_tensors_to_one_file=True,
                location="external_data",
                size_threshold=1024,
                convert_attribute=False,
            )
        onnx.save(model, str(model_path))

    def to_relay(self) -> relay_model.RelayModel:
        """Converts this model to a RelayModel.

        :return: this model converted to a RelayModel.
        :raise RelayConvertError: if the conversion fails.
        """
        # N.B. Defer the import so as not to unconditionally require other runtimes.
        from tvm import relay
        from tvm.error import OpNotImplemented as TVMOpNotImplemented

        from tvm2onnx import relay_model

        LOG.info("Convert model to_relay")
        try:
            self._infer_and_update_missing_inputs()
            try:
                mod, params = relay.frontend.from_onnx(
                    self.model, shape=self.input_shapes, freeze_params=True
                )
                output_names = [tensor.name for tensor in self.model.graph.output]
            except TVMOpNotImplemented as e:
                not_implemented_operator_msg = str(e)
                LOG.exception(
                    "Exception during relay conversion - relay.frontend.from_onnx"
                )
                not_implemented_operator_name = not_implemented_operator_msg.split(":")[
                    -1
                ].strip()
                raise RelayOpNotImplementedError(
                    not_implemented_operator_msg,
                    {"operator": not_implemented_operator_name},
                )
            except Exception:
                LOG.warning(
                    "Ignoring exception during relay conversion - relay.frontend.from_onnx "
                    + "attempting conversion with default statically supplied inputs"
                )

                try:
                    # Import with static vs dynamic shapes are on different TVM code
                    # paths, so where dynamic import fails static import may succeed.
                    static_input_shapes = generate_static_shapes(self.input_shapes)
                    mod, params = relay.frontend.from_onnx(
                        self.model, shape=static_input_shapes, freeze_params=True
                    )
                    output_names = [tensor.name for tensor in self.model.graph.output]
                except Exception as e:
                    LOG.exception(
                        "Exception during static relay conversion - relay.frontend.from_onnx"
                    )
                    raise e

            mod = relay_model.RelayModel.apply_relay_passes(mod)

            return relay_model.RelayModel(
                model=mod,
                params=params,
                input_shapes=self.input_shapes,
                input_dtypes=self.input_dtypes,
                best_records=None,
                output_names=output_names,
            )

        except Exception as e:
            LOG.exception("Exception converting model to_relay")
            if isinstance(e, RelayConvertError):
                raise
            raise RelayConvertError("Unable to convert model to relay").with_traceback(
                sys.exc_info()[2]
            )
