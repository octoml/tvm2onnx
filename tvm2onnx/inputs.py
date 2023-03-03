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

import typing

import numpy as np

from tvm2onnx.error import InputUnexpectedDynamicShapeError

Shape = typing.List[int]
Dtype = str
InputShapes = typing.Dict[str, Shape]
InputDtypes = typing.Dict[str, Dtype]


def is_dynamic_dim(dim: typing.Any) -> bool:
    """Returns true if the provided dimension is considered to be dynamic.

    :param dim: an integer dimension.
    """
    return True if not isinstance(dim, int) or dim < 0 else False


def is_dynamic_shape(shape: Shape) -> bool:
    """Returns true if any dimension in the Shape was considered to be dynamic.

    :param shape: a list of dimensions.
    """
    return any(is_dynamic_dim(dim) for dim in shape)


def get_dynamic_inputs(input_shapes: InputShapes) -> InputShapes:
    """Returns a new dictionary containing any shapes which were detected to be dynamic.

    :param input_shapes: shapes of the model's input tensors.
    """
    return {
        iname: ishape
        for iname, ishape in input_shapes.items()
        if is_dynamic_shape(ishape)
    }


def verify_inputs_are_static(input_shapes: InputShapes):
    """Raises exceptions if any of input shapes are not static.

    :param input_shapes: shapes of the model's input tensors.
    """
    dynamic_inputs = get_dynamic_inputs(input_shapes)
    if len(dynamic_inputs):
        dynamic_input_names = ",".join(dynamic_inputs.keys())
        raise InputUnexpectedDynamicShapeError(
            f"Dynamic shape on input(s) `{dynamic_input_names}`.",
            {"inputs": dynamic_input_names},
        )


def generate_static_shapes(
    input_shapes: InputShapes, dynamic_shape_replacement: int = 1
) -> InputShapes:
    """Generates static input shapes by replacing any dynamic dimension
    in the given `input_shapes` with the given `dynamic_shape_replacement`.

    :param input_shapes: shapes template to generate static shapes for
    :param dynamic_shape_replacement: dynamic dimensions will be replaced
        with this given value

    :return: static InputShapes with all originally dynamic dimensions
        replaced by the given `dynamic_shape_replacement`.
    """
    static_input_shapes = {}
    for iname in input_shapes.keys():
        shape = input_shapes[iname]
        static_shape = []
        for i in range(len(shape)):
            if is_dynamic_dim(shape[i]):
                static_shape.append(dynamic_shape_replacement)
            else:
                static_shape.append(shape[i])
        static_input_shapes[iname] = static_shape
    return static_input_shapes


def generate_input_data(
    input_shapes: InputShapes,
    input_dtypes: InputDtypes,
    dynamic_shape_replacement=None,
) -> typing.Dict[str, np.ndarray]:
    """Generates an input dict for benchmarking or initial configuration.
    It will swap dynamic shapes with the provided replacement, if any.

    :param input_shapes: shapes template to generate input data for
    :param dynamic_shape_replacement: dynamic dimensions will be replaced
        with this given value if not None

    :return: a dict of name to value for all input names in the provided
        shapes dict.
    """
    if dynamic_shape_replacement:
        input_shapes = generate_static_shapes(
            input_shapes, dynamic_shape_replacement=dynamic_shape_replacement
        )
    verify_inputs_are_static(input_shapes)

    data = {}
    for iname, ishape in input_shapes.items():
        shape = list(ishape)
        dtype = input_dtypes[iname]
        d = np.random.uniform(size=shape).astype(dtype)
        data[iname] = d
    return data
