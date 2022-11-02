import argparse
import copy
import os

import onnx
import tvm.runtime.vm
import tvm.testing
from tvm import relay
from tvm.relay import vm


def compile_virtual_machine(
    onnx_model_path: str,
    target: str = "llvm",
) -> tvm.runtime.vm.Executable:
    model = onnx.load(onnx_model_path)
    mod, params = relay.frontend.from_onnx(model, freeze_params=True)

    vm_exec = vm.compile(
        copy.deepcopy(mod),
        target,
        params=params,
    )

    return vm_exec


def serialize_virtual_machine(
    vm_exec: tvm.runtime.vm.Executable, directory: str
) -> None:
    path_consts = os.path.join(directory, "consts")
    vm_exec.move_late_bound_consts(path_consts, byte_limit=256)
    lib_path = os.path.join(directory, "model.so")
    code_path = os.path.join(directory, "vm_exec_code.ro")
    code, lib = vm_exec.save()
    lib.export_library(lib_path)
    with open(code_path, "wb") as fo:
        fo.write(code)


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

    vm_exec = compile_virtual_machine(args.model_path)
    serialize_virtual_machine(vm_exec, args.export_dir)


if __name__ == "__main__":
    main()
