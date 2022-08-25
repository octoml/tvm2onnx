"""Common utilities for ONNX runtime."""
import typing

import numpy as np
import onnx


def get_vinfo_map(
    onnx_model: onnx.ModelProto,
) -> typing.Dict[str, onnx.ValueInfoProto]:
    """Returns value info map of a given model.
    :param onnx_model: The model to analyze.
    :return: A dictionary of value infos of the given model.
    """
    value_info_map: typing.Dict[str, onnx.ValueInfoProto] = {}
    for input in onnx_model.graph.input:
        value_info_map[input.name] = input
    for output in onnx_model.graph.output:
        value_info_map[output.name] = output
    for vinfo in onnx_model.graph.value_info:
        value_info_map[vinfo.name] = vinfo
    return value_info_map


def validate_tensor_type(value_info: onnx.ValueInfoProto):
    """Checks to make sure the value info type is a tensor.
    :param value_info: A value info proto.
    """
    v_type = value_info.type
    assert v_type.HasField("tensor_type")
    assert not v_type.HasField("sequence_type")


def rewrite(
    onnx_model: onnx.ModelProto, tracing_tensors: typing.Dict[str, np.ndarray]
) -> onnx.ModelProto:
    """Realize input shapes for the given input onnx model based on data in tracing_tensors.
    :param onnx_model: The model to rewrite.
    :param tracing_tensors: A map of edge names to tensors we have type information of.
    :return: A new onnx model with shape information realized.
    """
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model, data_prop=True)

    old_vinfo_map = get_vinfo_map(onnx_model)

    # Erase type annotations for intermediate nodes
    old_outputs = []
    while len(onnx_model.graph.output) > 0:
        old_outputs.append(onnx_model.graph.output.pop().name)
    while len(onnx_model.graph.value_info) > 0:
        onnx_model.graph.value_info.pop()

    new_vinfos_inputs = {}

    # Construct new vinfos for each input based on tracing tensor
    for old_vinfo in list(onnx_model.graph.input):
        name = old_vinfo.name
        input_value = tracing_tensors[name]

        # TODO: handle other types besides tensors, e.g. sequence
        validate_tensor_type(old_vinfo_map[name])
        new_type_proto = onnx.helper.make_tensor_type_proto(
            elem_type=old_vinfo_map[name].type.tensor_type.elem_type,
            shape=input_value.shape,
        )
        vinfo = onnx.helper.make_value_info(name, new_type_proto)
        new_vinfos_inputs[name] = vinfo

    for input_vinfo in onnx_model.graph.input:
        input_vinfo.CopyFrom(new_vinfos_inputs[input_vinfo.name])

    # With input tensor shapes realized, this should solve much more!
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model, data_prop=True)

    # re-add outputs by moving from value_infos --> outputs
    new_vinfo_map = get_vinfo_map(onnx_model)
    for output_name in old_outputs:
        onnx_model.graph.output.append(new_vinfo_map[output_name])
        onnx_model.graph.value_info.remove(new_vinfo_map[output_name])

    return onnx_model
