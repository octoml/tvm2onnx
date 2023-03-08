import argparse
import tempfile
import typing
import re
import copy
import pathlib
import pickle

import tvm
import tvm.meta_schedule.measure_callback as measure_callback
from tvm import meta_schedule, nd, relay
from tvm.contrib import cc
from tvm.meta_schedule import database as ms_database
from tvm.relay import vm
from tvm.runtime import profiler_vm
from tvm.runtime import vm as vm_rt

import onnx


import os
import numpy as np

import tvm
from tvm import relay, autotvm
from tvm.relay import testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
import tvm.contrib.graph_executor as runtime


tuning_option = {
    "log_filename": "tuning_records.log",
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(
            number=1, repeat=10, min_repeat_ms=0, enable_cpu_cache_flush=True
        ),
    ),
}


# Use graph tuner to achieve graph level optimal schedules
# Set use_DP=False if it takes too long to finish.
def tune_graph(graph, dshape, records, opt_sch_file, use_DP=True):
    target_op = [
        relay.op.get("nn.conv2d"),
    ]
    Tuner = DPTuner if use_DP else PBQPTuner
    executor = Tuner(graph, {input_name: dshape}, records, target_op, target)
    executor.benchmark_layout_transform(min_exec_num=2000)
    executor.run()
    executor.write_opt_sch2record_file(opt_sch_file)

def evaluate_performance(lib, data_shape):
    # upload parameters to device
    dev = tvm.cpu()
    data_tvm = tvm.nd.array((np.random.uniform(size=data_shape)).astype(dtype))
    module = runtime.GraphModule(lib["default"](dev))
    module.set_input(input_name, data_tvm)

    # evaluate
    print("Evaluate inference time cost...")
    print(module.benchmark(dev, number=100, repeat=3))


def compile_tvm_model(relay_model, params, tvm_target, best_records):
    with autotvm.apply_history_best(best_records):
        with tvm.transform.PassContext(
            opt_level=3,
            config={"relay.FuseOps.max_depth": 30},
        ):
            return vm.compile(
                relay_model,
                tvm_target,
                params=params,
            )


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


def tune(model_path, tvm_target, output_path, axis_map={}):
    # extract workloads from relay program
    print("Extract tasks...")
    onnx_model = onnx.load(model_path)
    initializer_names = [n.name for n in onnx_model.graph.initializer]
    input_shapes = {}
    for i in onnx_model.graph.input:
        if i.name not in initializer_names:
            input_name, _, dtype, axis_names = relay.frontend.onnx.get_info(i)
            shape = []
            for val in axis_names:
                if val in axis_map.keys():
                    val = axis_map[val]
                shape.append(val)
            input_shapes[input_name] = shape

    print(input_shapes)

    mod, params = relay.frontend.from_onnx(
        onnx_model, shape=input_shapes, freeze_params=True
    )

    tasks = autotvm.task.extract_from_program(
        mod["main"], target=tvm_target, params=params, ops=(relay.op.get("nn.conv2d"),)
    )

    measure_option=autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(
            number=1,
            repeat=10,
            min_repeat_ms=0,
            enable_cpu_cache_flush=True
        ),
    )

    # run tuning tasks
    log_filename="tuning_records.log"
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
                autotvm.callback.log_to_file(log_filename),
            ],
        )

    best_records_file = "best_records.log"

    autotvm.record.pick_best(log_filename, best_records_file)
    best_records = []
    with open(best_records_file, "r") as f:
        tuning_records = f.readlines()
        best_records = list(map(autotvm.record.decode, tuning_records))

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

    tdir_path = pathlib.Path(output_path)
    constants_map = {
        name: data.numpy()
        for name, data in vm_exec.get_late_bound_consts(0).items()
    }
    ro_path = tdir_path / "vm_exec_code.ro"
    model_object = tdir_path / "model.o"
    constants_path = tdir_path / "constants.pkl"

    vm_exec_code, mod = vm_exec.save()
    with open(ro_path, "wb") as fo:
        fo.write(vm_exec_code)

    mod.export_library(
        model_object,
        fcompile=partial_link_build_func(tvm_target),
    )

    with open(constants_path, 'wb') as f:
        pickle.dump(constants_map, f)

    # with open('saved_dictionary.pkl', 'rb') as f:
    #     loaded_dict = pickle.load(f)

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
        m = re.match("(.+)=(\d+)", args.axis_size)
        axis_map[m[1]] = int(m[2])
    print(axis_map)

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
