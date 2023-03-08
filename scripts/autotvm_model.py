import argparse
import tempfile
import typing
import re

import tvm
import tvm.meta_schedule.measure_callback as measure_callback
from tvm import meta_schedule, nd, relay
from tvm.contrib import graph_executor
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


def tune(model_path, target):
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
                # if val in axis_map.keys():
                #     val = axis_map[val]
                shape.append(val)
            input_shapes[input_name] = shape

    print(input_shapes)

    mod, params = relay.frontend.from_onnx(
        onnx_model, shape=input_shapes, freeze_params=True
    )

    tasks = autotvm.task.extract_from_program(
        mod["main"], target=target, params=params, ops=(relay.op.get("nn.conv2d"),)
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

    log_file = "best_records.txt"

    # run tuning tasks
    log_filename="tuning_records.log"
    for i, task in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        tuner_obj = XGBTuner(task, loss_type="rank")

        # do tuning
        n_trial = len(task.config_space)
        tuner_obj.tune(
            n_trial=n_trial,
            early_stopping=None,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(n_trial, prefix=prefix),
                autotvm.callback.log_to_file(log_filename),
            ],
        )




    # tune_graph(mod["main"], data_shape, log_file, graph_opt_sch_file)

    # # compile kernels in default mode
    # print("Evaluation of the network compiled in 'default' mode without auto tune:")
    # with tvm.transform.PassContext(opt_level=3):
    #     print("Compile...")
    #     lib = relay.build(mod, target=target, params=params)
    #     evaluate_performance(lib, data_shape)

    # # compile kernels in kernel tuned only mode
    # print("\nEvaluation of the network been tuned on kernel level:")
    # with autotvm.apply_history_best(log_file):
    #     print("Compile...")
    #     with tvm.transform.PassContext(opt_level=3):
    #         lib = relay.build(mod, target=target, params=params)
    #     evaluate_performance(lib, data_shape)

    # # compile kernels with graph-level best records
    # print("\nEvaluation of the network been tuned on graph level:")
    # with autotvm.apply_graph_best(graph_opt_sch_file):
    #     print("Compile...")
    #     with tvm.transform.PassContext(opt_level=3):
    #         lib = relay.build_module.build(mod, target=target, params=params)
    #     evaluate_performance(lib, data_shape)


# We do not run the tuning in our webpage server since it takes too long.
# Uncomment the following line to run it by yourself.

# tune_and_evaluate(tuning_option)




























# def tune(
#     model,
#     target: tvm.target.Target,
#     axis_map: typing.Dict[str, int],
#     max_trials_global=128,
# ):
#     onnx_model = onnx.load(model)
#     initializer_names = [n.name for n in onnx_model.graph.initializer]
#     input_shapes = {}
#     for i in onnx_model.graph.input:
#         if i.name not in initializer_names:
#             input_name, _, dtype, axis_names = relay.frontend.onnx.get_info(i)
#             shape = []
#             for val in axis_names:
#                 if val in axis_map.keys():
#                     val = axis_map[val]
#                 shape.append(val)
#             input_shapes[input_name] = shape

#     mod, params = relay.frontend.from_onnx(
#         onnx_model, shape=input_shapes, freeze_params=True
#     )

#     with tempfile.TemporaryDirectory() as work_dir:
#         with target:
#             database = meta_schedule.relay_integration.tune_relay(
#                 mod,
#                 params,
#                 target,
#                 work_dir=work_dir,
#                 max_trials_global=max_trials_global,
#             )

#     print("Tuning Time:")
#     return database


def main():  # pragma: no cover
    parser = argparse.ArgumentParser(description="Tune an onnx model with TVM.")
    parser.add_argument(
        "--model",
        required=True,
        help="Source model in .onnx format",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output file in tvm in onnx format.",
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

    tune(
        model_path=args.model,
        # axis_map=axis_map,
        target=tvm.target.Target("llvm"),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
