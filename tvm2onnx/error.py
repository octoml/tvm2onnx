from typing import Dict, Optional


class TVM2ONNXError(Exception):
    """The base error type for all tvm2onnx exceptions."""

    def __init__(self, message: str, kwargs: Optional[Dict[str, str]] = None):
        self.message = message
        self.kwargs = kwargs
        super().__init__(message, kwargs)


# ----
# General errors are defined below, and more granular versions are defined in the appropriate files.
# Example: In tf_model.py, there exists TFIngestError


class IngestError(TVM2ONNXError):
    """Indicates an error occurred with model ingest, maybe it's not quite delicious enough."""


class IngestNotTarFileError(IngestError):
    """Indicates that the provided resource does not represent a tarfile."""


class IngestNoModelInTarFileError(IngestError):
    """Indicates that the extracted tarfile does not contain a model."""


class IngestMultipleAssetsInTarFileError(IngestError):
    """Indicates that the extracted tarfile contains more than one model or asset.
    Provides a "files" kwarg with a comma-delimited string of file names.
    """


class IngestDeviceError(IngestError):
    """Indicates that the model has parameters or tensors pinned to a device that
    is different from where the model is run."""


class InferInputsError(TVM2ONNXError):
    """Indicates an error occurred with a model's inputs."""


class InferInputsNoneFoundError(InferInputsError):
    """Indicates that that no inputs were found for the model."""


class InferInputsMissingShapeError(InferInputsError):
    """Indicates that the model has inputs without any shapes.
    Provides a "inputs" kwarg with a comma-delimited string of input names.
    """


class InferInputsDynamicShapeError(InferInputsError):
    """Indicates that the model has inputs with dynamic shapes.
    Provides a "inputs" kwarg with a comma-delimited string of input names.
    """


class InferInputsUnknownDataTypeError(InferInputsError):
    """
    Indicates that the model has inputs with unknown data types.
    Provides a "inputs" kwarg with a comma-delimited string of input names.
    """


class InputConfigurationUnexpectedCountError(TVM2ONNXError):
    """Indicates that the number of inputs provided does not match the number
    of inputs indicated by the model.
    """


class InputConfigurationMismatchedInputsError(TVM2ONNXError):
    """Indicates that the provided inputs shapes and dtypes do not match."""


class InputConfigurationInvalidError(TVM2ONNXError):
    """Indicates that inference failed for the provided input configuration."""


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


class ONNXConvertError(TVM2ONNXError):
    """Indicates an error occurred while converting to ONNX, Aww nix!
    Provides a "opset" kwarg with the value of the opset used.
    Provides an optional "error" kwarg with the output of the conversion failure.
    """


class RelayConvertError(TVM2ONNXError):
    """Indicates an error occurred while converting to Relay."""


class RelayOpNotImplementedError(RelayConvertError):
    """Indicates an error occurred due to an operator not implemented on the tvm level"""


class BenchmarkError(TVM2ONNXError):
    """Indicates an error occurred while benchmarking a model."""


class ModelError(TVM2ONNXError):
    """Indicates an error occurred while interacting with an model."""


class InvalidRecordFormatError(TVM2ONNXError):
    """Indicates that the format of tuning records was invalid."""


class TuningRecordTypeError(TVM2ONNXError):
    """Indicates that there is an issue with the record type of a previous tune"""


class PackagingError(TVM2ONNXError):
    """Indicates an Error occurred with model packaging."""
