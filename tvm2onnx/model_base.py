"""Defines the abstract base class for all models."""

import abc
import typing

import numpy as np

from tvm2onnx.error import (
    InputUpdateError,
    InputUpdateUnexpectedShape,
    InputUpdateUnknownName,
)
from tvm2onnx.inputs import (
    Dtype,
    InputDtypes,
    InputShapes,
    Shape,
    generate_input_data,
    get_dynamic_inputs,
)


class ModelArgument(typing.NamedTuple):
    """Stores the shape and dtype of an argument"""

    dtype: Dtype
    """The data type of the argument."""

    shape: Shape
    """The shape of the argument."""


class ModelBase(abc.ABC):
    """A wrapper class providing a common interface to various model types."""

    def __init__(
        self,
        model: typing.Any,
        input_shapes: InputShapes = None,
        input_dtypes: InputDtypes = None,
    ):
        """Initializes a new ModelBase.

        :param model: The underlying model wrapped by this Model, e.g.
            a onnx.ModelProto.
        :param input_shapes: The shapes of this Model's input tensors.
        :param input_dtypes: The dtypes of this Model's input tensors.
        """
        self.model = model
        self.input_shapes = input_shapes or {}
        self.input_dtypes = input_dtypes or {}

    @abc.abstractmethod
    def infer_inputs(
        self,
    ) -> typing.Tuple[InputShapes, InputDtypes]:
        """Infers the input shapes and input dtypes of this model.

        :return: This model's inferred input shapes and input dtypes.
        """

    @abc.abstractmethod
    def infer_and_update_inputs(self):
        """Modifies the input_shapes, input_dtypes on this model with inferred info.
        Overwrites existing input info on this model.
        """

    def _infer_and_update_missing_inputs(self):
        """Infers and updates input_shapes and input_dtypes
        on this model if neither is already set.
        """
        # TODO: When do we allow creation of valid model objects with missing inputs?
        if not self.input_shapes and not self.input_dtypes:
            self.infer_and_update_inputs()

    def generate_input_data(self) -> typing.Dict[str, np.ndarray]:
        """Generates an input dict for benchmarking.

        :return: a dict of name to value for all input names in the provided
            shapes dict.
        """
        self._infer_and_update_missing_inputs()
        return generate_input_data(self.input_shapes, self.input_dtypes)

    def inputs_are_static(self) -> bool:
        """Checks if inputs on this model are fully static.

        :return: True if inputs on this model are fully static. False otherwise.
        """
        return len(get_dynamic_inputs(self.input_shapes)) == 0

    def matches_inputs(
        self,
        inferred_shapes: InputShapes,
        inferred_dtypes: InputDtypes,
        input_shapes: InputShapes,
        input_dtypes: InputDtypes,
    ) -> bool:
        """Returns True if the input info provided in the arguments is an exact
        match for the inferred inputs on this model. For example, if self.input_shapes
        is [?, 2], a successful match would be [1, 2], but [?, 2] would also be
        successful. Returns False otherwise. -1 and ? and None are acceptable alternatives
        for indication of an unknown dimension.

        :param inferred_shapes: inferred input shape info
        :param inferred_dtypes: inferred input dtype info
        :param input_shapes: input shape info
        :param input_dtypes: input dtype info
        :return: True if the given input info matches the inferred inputs on this model,
            False otherwise.
        """
        # Check that all the ambiguous input dimensions are filled by the provided
        # input_shapes.
        for iname, ishape in inferred_shapes.items():
            for i, dim in enumerate(ishape):
                if (
                    # Check that iname exists in the provided input_shapes
                    (iname not in input_shapes)
                    # Check the input shape lengths match
                    or (len(input_shapes[iname]) != len(ishape))
                    # Check that the known dimensions match
                    or (
                        isinstance(dim, int)
                        and dim > 0
                        and input_shapes[iname][i] != dim
                    )
                    # check that the user input is positive
                    or (
                        isinstance(input_shapes[iname][i], int)
                        and input_shapes[iname][i] <= 0
                    )
                ):
                    return False
        # Check that the dtypes match and that no extra input shapes were provided
        return inferred_dtypes == input_dtypes and len(inferred_shapes) == len(
            input_shapes
        )

    def update_inputs(
        self,
        input_shapes: InputShapes,
        input_dtypes: InputDtypes,
    ):
        """Updates the input_shapes, input_dtypes on this model with given info.

        :param input_shapes: shapes of the model's input tensors
        :param input_dtypes: dtypes of the model's input tensors
        """
        if input_shapes:
            if self.input_shapes is None:
                self.input_shapes = input_shapes
            else:
                # Check to see if the model's inputs are dynamic
                if self.inputs_are_static():
                    raise InputUpdateError(
                        f"No dynamic inputs found in {self.input_shapes}."
                    )

                for iname, ishape in input_shapes.items():
                    # Check that static inputs are provided
                    for dim in ishape:
                        if not isinstance(dim, int) or dim < 0:
                            raise InputUpdateError(
                                f"Must use static inputs in {ishape}."
                            )
                        if dim == 0:
                            raise InputUpdateError(
                                f"Input shapes must be greater than zero {ishape}."
                            )

                    # Check that iname exists in the model's input_shapes
                    if iname not in self.input_shapes:
                        raise InputUpdateUnknownName(
                            f"Unknown input name found in {iname}.",
                            {"input_name": iname},
                        )

                    # Check the input shape lengths match
                    if len(self.input_shapes[iname]) != len(ishape):
                        raise InputUpdateUnexpectedShape(
                            f"Unexpected input shape {ishape} found {iname}.",
                            {"input_name": iname, "input_shape": str(ishape)},
                        )
                    self.input_shapes.update({iname: ishape})

        if input_dtypes:
            if self.input_dtypes is None:
                self.input_dtypes = input_dtypes
            else:
                for iname, idtype in input_dtypes.items():
                    # Check that iname exists in the model's input_dtypes
                    if iname not in self.input_dtypes:
                        raise InputUpdateUnknownName(
                            f"Unknown input name found in {iname}.",
                            {"input_name": iname},
                        )
                self.input_dtypes.update({iname: idtype})
