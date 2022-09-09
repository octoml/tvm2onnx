from typing import Dict, Optional


class TVM2ONNXError(Exception):
    """The base error type for all tvm2onnx exceptions."""

    def __init__(self, message: str, kwargs: Optional[Dict[str, str]] = None):
        self.message = message
        self.kwargs = kwargs
        super().__init__(message, kwargs)
