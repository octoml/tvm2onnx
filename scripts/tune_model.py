import argparse
import tempfile
import typing
import re

import tvm
import tvm.meta_schedule.measure_callback as measure_callback
from tvm import meta_schedule, nd, relay
from tvm.meta_schedule import database as ms_database
from tvm.relay import vm
from tvm.runtime import vm as vm_rt

import onnx


def tune(
    model,
    target: tvm.target.Target,
    axis_map: typing.Dict[str, int]
):
    onnx_model = onnx.load(model)
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

    mod, params = relay.frontend.from_onnx(
        onnx_model, shape=input_shapes, freeze_params=True
    )

    with tempfile.TemporaryDirectory() as work_dir:
        with target:
            database = meta_schedule.relay_integration.tune_relay(
                mod,
                params,
                target,
                work_dir=work_dir,
                max_trials_global=1000,
                num_trials_per_iter=64,
                max_trials_per_task=64
            )

    print("Tuning Complete")
    return database


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

    database = tune(
        model=args.model,
        axis_map=axis_map,
        target=tvm.target.Target("llvm -num-cores 8"),
    )

    records = []
    for workload in self._mod_hashes_to_workloads.values():
        top_k = self.get_top_k(workload, 1)
        if top_k:
            records.append(top_k[0])
    return records



if __name__ == "__main__":  # pragma: no cover
    main()