import argparse
import copy
import json
import os
import pathlib
import pickle
import re
import tempfile
import typing

import onnx
import tvm
from tvm import autotvm, relay
from tvm.autotvm.tuner import XGBTuner
from tvm.contrib import cc
from tvm.relay import vm


def partial_link_build_func(tvm_target: tvm.target.Target):
    """Gets a TVM build function for creating a partially-linked object file

    :param tvm_target: the TVM target
    :return: a function to be passed to `export_library`
    """
    # -r performs partial (or incremental) linking which links a set of object
    # files into a single object file
    return cc.cross_compiler(
        lambda *args, **kwargs: cc.create_executable(*args, **kwargs, cc="g++"),
        options=["-r"],
    )


def get_io_info(onnx_model, axis_map):
    initializer_names = [n.name for n in onnx_model.graph.initializer]
    inputs = []
    outputs = []

    for i in onnx_model.graph.input:
        if i.name not in initializer_names:
            input_name, _, dtype, axis_names = relay.frontend.onnx.get_info(i)
            shape = []
            for val in axis_names:
                if val in axis_map.keys():
                    val = axis_map[val]
                elif val % 1 != 0:  # is integer
                    print(
                        f"Model has dynamic input axis {axis_names} which must be defined. "
                        f"Use '--axis-size {val}=N' where N is an integer."
                    )
                    exit(1)
                shape.append(val)
            inputs.append({"name": input_name, "shape": shape, "dtype": dtype})
    for o in onnx_model.graph.output:
        output_name, _, dtype, axis_names = relay.frontend.onnx.get_info(o)
        shape = []
        for val in axis_names:
            if val in axis_map.keys():
                val = axis_map[val]
            elif val % 1 != 0:  # is integer
                print(
                    f"Model has dynamic input axis {axis_names} which must be defined. "
                    f"Use '--axis-size {val}=N' where N is an integer."
                )
                exit(1)
            shape.append(val)
        outputs.append({"name": output_name, "shape": shape, "dtype": dtype})

    return {"inputs": inputs, "outputs": outputs}


def tune(
    model_path: str,
    tvm_target: tvm.target.Target,
    output_path: str,
    axis_map: typing.Dict[str, int] = {},
) -> None:
    """
    Tune an onnx model using TVM's autotune. The tuning results and metadata required by tvm2onnx
        are saved in output_path.
    :param model_path: Path to the source model in .onnx format.
    :param tvm_target: The tvm target information used for tuning.
    :param output_path: Directory where the tuning results are stored.
    :param axis_map: TVM requires static input shapes. If a model has named dynamic axes specify
        the axis name and its static value to used for tuning, like {"batch_size": 4}
    :return: None
    """
    # Generate some metadata required by tvm2onnx
    tvm_target = tvm.target.Target(tvm_target)
    onnx_model = onnx.load(model_path)
    metadata = get_io_info(onnx_model, axis_map)
    metadata["target"] = str(tvm_target)
    input_shapes = {tensor["name"]: tensor["shape"] for tensor in metadata["inputs"]}

    # Convert the onnx model to Relay. This Relay graph is used as input to the TVM tuning process.
    mod, params = relay.frontend.from_onnx(
        onnx_model, shape=input_shapes, freeze_params=True
    )

    # Extract tuning tasks from the Relay graph. A task is equivalent to a kernel or function
    # containing multiple Relay operators.
    tasks = autotvm.task.extract_from_program(
        mod["main"], target=tvm_target, params=params, ops=(relay.op.get("nn.conv2d"),)
    )

    # Measurement options are used duing to tuning process to measure the performance of each task.
    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(
            number=1, repeat=10, min_repeat_ms=0, enable_cpu_cache_flush=True
        ),
    )

    # Start the tuning process. Temporary files are used to store the tuning records generated
    # by the tuning process and another for the best solution for each task.
    with tempfile.NamedTemporaryFile() as full_records_tmp:
        with tempfile.NamedTemporaryFile() as best_records_tmp:
            full_records_file = full_records_tmp.name
            best_records_file = best_records_tmp.name
            # Loop through all tuning tasks and display progress.
            for i, task in enumerate(tasks):
                prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

                tuner_obj = XGBTuner(task, loss_type="rank")

                # do tuning
                n_trial = 16
                tuner_obj.tune(
                    n_trial=n_trial,
                    early_stopping=None,
                    measure_option=measure_option,
                    callbacks=[
                        autotvm.callback.progress_bar(n_trial, prefix=prefix),
                        autotvm.callback.log_to_file(full_records_file),
                    ],
                )

            # From the full results of tuning select the best tuning solution for each task.
            # Store the results in best_records_file.
            autotvm.record.pick_best(full_records_file, best_records_file)
            best_records = []
            with open(best_records_file, "r") as f:
                tuning_records = f.readlines()
                best_records = list(map(autotvm.record.decode, tuning_records))

    # Using the best_records from the tuning process compile the model to use the TVM VM.
    # This produces the compile model vm_exec.
    with autotvm.apply_history_best(best_records):
        with tvm.transform.PassContext(
            opt_level=3,
            config={"relay.FuseOps.max_depth": 30},
        ):
            vm_exec = vm.compile(
                mod=copy.deepcopy(mod),
                target=tvm_target,
                params=params,
            )

    # Generate files required to launch tvm2onnx. This includes
    #     * vm_exec_code.ro - a sequence of bitcode which specifies the order that the tuned
    #           kernels are called.
    #     * model.o - a compiled object ready for linking to the tvm2onnx-created custom op
    #           shared library.
    #     * constants.pkl - the constant map created here must be passed to tvm2onnx so for this
    #           demo it is stored in a pickle file. This is just an easy method to get this
    #           information to the script which builds the file .onnx model.
    #     * metadata.json - stores model input and output information and which target the model
    #           is compiled for.
    tdir_path = pathlib.Path(output_path)
    constants_map = {
        name: data.numpy() for name, data in vm_exec.get_late_bound_consts(0).items()
    }
    ro_path = tdir_path / "vm_exec_code.ro"
    model_object = tdir_path / "model.o"
    constants_path = tdir_path / "constants.pkl"
    metadata_path = tdir_path / "metadata.json"

    vm_exec_code, mod = vm_exec.save()
    with open(ro_path, "wb") as fo:
        fo.write(vm_exec_code)

    mod.export_library(
        model_object,
        fcompile=partial_link_build_func(tvm_target),
    )

    with open(constants_path, "wb") as f:
        pickle.dump(constants_map, f)

    with open(metadata_path, "w") as f:
        json.dump(metadata, f)


def main():  # pragma: no cover
    parser = argparse.ArgumentParser(description="Tune an onnx model with TVM.")
    parser.add_argument(
        "--model",
        required=True,
        help="Source model in .onnx format",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Output path where model.o and vm_exec_code.ro files are saved",
    )
    parser.add_argument(
        "--axis-size",
        required=False,
        type=str,
        help="Define a static value for a named dynamic axis",
    )
    args = parser.parse_args()

    axis_map = {}
    if args.axis_size:
        m = re.match("(.+)=(\\d+)", args.axis_size)
        axis_map[m[1]] = int(m[2])
    print(f"Static axis replacement: {axis_map}")

    os.makedirs(args.output_path, exist_ok=True)
    if not os.path.exists(args.output_path):
        print("The --output-path must exist")
        exit(1)

    tune(
        model_path=args.model,
        output_path=args.output_path,
        axis_map=axis_map,
        tvm_target=tvm.target.Target("llvm"),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
