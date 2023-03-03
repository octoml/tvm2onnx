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

from typing import Dict, Optional


class TVM2ONNXError(Exception):
    """The base error type for all tvm2onnx exceptions."""

    def __init__(self, message: str, kwargs: Optional[Dict[str, str]] = None):
        self.message = message
        self.kwargs = kwargs
        super().__init__(message, kwargs)


class IngestError(TVM2ONNXError):
    """Indicates an error occurred with model ingest, maybe it's not quite delicious enough."""


class IngestNoModelInTarFileError(IngestError):
    """Indicates that the extracted tarfile does not contain a model."""


class IngestMultipleAssetsInTarFileError(IngestError):
    """Indicates that the extracted tarfile contains more than one model or asset.
    Provides a "files" kwarg with a comma-delimited string of file names.
    """


class InferInputsError(TVM2ONNXError):
    """Indicates an error occurred with a model's inputs."""


class InferInputsNoneFoundError(InferInputsError):
    """Indicates that that no inputs were found for the model."""


class InferInputsUnknownDataTypeError(InferInputsError):
    """
    Indicates that the model has inputs with unknown data types.
    Provides a "inputs" kwarg with a comma-delimited string of input names.
    """


class InferSymbolicShapesError(TVM2ONNXError):
    """Indicates an error occurred while running symbolic shape inference."""


class InputUpdateError(TVM2ONNXError):
    """Indicates an error occurred with a model's inputs during updating."""


class InputUpdateUnknownName(InputUpdateError):
    """Indicates that the provided input name does not match the inputs set on the model."""


class InputUpdateUnexpectedShape(InputUpdateError):
    """Indicates that the shape for the given input name does not match what is set on the model."""


class InputUnexpectedDynamicShapeError(TVM2ONNXError):
    """Indicates that the model unexpectedly found inputs with dynamic shapes.
    Provides a "inputs" kwarg with a comma-delimited string of input names.
    """


class RelayConvertError(TVM2ONNXError):
    """Indicates an error occurred while converting to Relay."""


class RelayOpNotImplementedError(RelayConvertError):
    """Indicates an error occurred due to an operator not implemented on the tvm level"""


class ModelError(TVM2ONNXError):
    """Indicates an error occurred while interacting with an model."""


class InvalidRecordFormatError(TVM2ONNXError):
    """Indicates that the format of tuning records was invalid."""


class TuningRecordTypeError(TVM2ONNXError):
    """Indicates that there is an issue with the record type of a previous tune"""


class PackagingError(TVM2ONNXError):
    """Indicates an Error occurred with model packaging."""
