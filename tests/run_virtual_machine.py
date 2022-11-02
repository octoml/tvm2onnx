import argparse
import os
import typing

import numpy as np
import onnx
import tvm
from tvm.runtime import vm as runtime_vm


def load_virtual_machine(
    exported_module_path: str,
    exported_consts_path: str,
    serialized_vm_exec_path: str,
    ctx: tvm.runtime.Device,
) -> runtime_vm.VirtualMachine:
    mod = tvm.runtime.load_module(str(exported_module_path))

    with open(serialized_vm_exec_path, "rb") as f:
        vm_bytes = f.read()

    vm_exec = runtime_vm.Executable.load_exec(vm_bytes, mod)
    vm_exec.mod["load_late_bound_consts"](str(exported_consts_path))

    return runtime_vm.VirtualMachine(vm_exec, ctx)


def get_input_data_for_model_with_fixed_shapes(
    onnx_model: onnx.ModelProto,
) -> typing.Dict[str, typing.Any]:
    """
    Create input data for model with static shapes
    """

    def get_onnx_input_names(model: onnx.ModelProto) -> typing.List[str]:
        inputs = [node.name for node in model.graph.input]
        initializer = [node.name for node in model.graph.initializer]
        inputs = list(set(inputs) - set(initializer))
        return sorted(inputs)

    def get_onnx_input_types(model: onnx.ModelProto) -> typing.List[str]:
        input_names = get_onnx_input_names(model)
        return [
            str(onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[node.type.tensor_type.elem_type])
            for node in sorted(model.graph.input, key=lambda node: node.name)
            if node.name in input_names
        ]

    def get_onnx_input_shapes(model: onnx.ModelProto) -> typing.List[typing.List[int]]:
        input_names = get_onnx_input_names(model)
        return [
            [dv.dim_value for dv in node.type.tensor_type.shape.dim]
            for node in sorted(model.graph.input, key=lambda node: node.name)
            if node.name in input_names
        ]

    input_names = get_onnx_input_names(onnx_model)
    input_shapes = get_onnx_input_shapes(onnx_model)
    input_types = get_onnx_input_types(onnx_model)
    assert len(input_names) == len(input_types) == len(input_shapes)
    return {
        name: {"dtype": dtype, "shape": shape}
        for name, dtype, shape in zip(input_names, input_types, input_shapes)
    }


def get_input_dict(
    model_path: str, ctx: tvm.runtime.Device
) -> typing.Dict[str, tvm.runtime.NDArray]:
    input_dict = get_input_data_for_model_with_fixed_shapes(onnx.load(model_path))
    inputs = {
        input_name: tvm.nd.array(
            np.random.uniform(size=tuple(input_dict[input_name]["shape"])).astype(
                input_dict[input_name]["dtype"]
            ),
            device=ctx,
        )
        for input_name in input_dict.keys()
    }
    return inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        required=True,
    )
    parser.add_argument(
        "--export_dir",
        required=True,
    )
    args = parser.parse_args()

    ctx = tvm.device("llvm")

    engine = load_virtual_machine(
        os.path.join(args.export_dir, "model.so"),
        os.path.join(args.export_dir, "consts"),
        os.path.join(args.export_dir, "vm_exec_code.ro"),
        ctx,
    )

    print(engine.run(**get_input_dict(args.model_path, ctx)))


if __name__ == "__main__":
    main()
