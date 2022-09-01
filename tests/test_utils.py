"""
Utilities for tests
"""

import os
import re
import numpy as np
import time

import tarfile
import onnx
import onnxruntime
import tvm
from tvm import relay, auto_scheduler
from tvm.relay import vm
from tvm import autotvm


def find(pattern, path):
    result = []
    regex = re.compile(pattern)
    for root, _, files in os.walk(path):
        for name in files:
            if regex.match(name):
                result.append(os.path.join(root, name))
    return result

def unpack_onnx_tar(model_path, tmp_dir):
    with tarfile.open(model_path, "r") as tar:
      tar.extractall(tmp_dir)
    files = find(".*\\.onnx$", tmp_dir)
    if len(files) < 1:
        print("No onnx model found")
        exit(-1)
    elif len(files) > 1:
        print("Multiple onnx models found")
        exit(-1)
    onnx_path = files[0]
    custom_op_libs_paths = find("^custom_.*\\.(so|dll|dynlib)$", tmp_dir)
    
    return onnx_path, custom_op_libs_paths

def generate_input_shapes_dtypes(onnx_model_path):
    onnx_model = onnx.load_model(onnx_model_path)
    input_shapes = {}
    input_dtypes = {}
    for inp in onnx_model.graph.input:
        name = inp.name
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        input_shapes[name] = shape
        dtype = inp.type.tensor_type.elem_type
        input_dtypes[name] = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[dtype]

    return input_shapes, input_dtypes

def generate_input_data(
    input_shapes,
    input_dtypes,
):
    data = {}
    for name in input_shapes.keys():
        shape = input_shapes[name]
        dtype = input_dtypes[name]
        high_val = 1.0
        if (dtype == "int32" or dtype == "int64"):
            high_val = 1000.0
        d = np.random.uniform(size=shape, high=high_val).astype(dtype)
        data[name] = d
    return data

def get_ort_inference_session(onnx_path, custom_op_libs = None):
    sess_options = onnxruntime.SessionOptions()
    if custom_op_libs:
        for custom_op_lib in custom_op_libs:
            sess_options.register_custom_ops_library(custom_op_lib)

    engine = onnxruntime.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
        provider_options=[{}],
        sess_options=sess_options,
    )
    return engine

def perf_test(run, iters_number = 1000, model_name = "ResNet50-v1", framework_name = "ort+tvm", pre_iters_num = 5):
    assert iters_number > 0

    # warmup run
    for i in range(pre_iters_num):
        run()

    tic = time.perf_counter()
    for i in range(iters_number):
        run()
    toc = time.perf_counter()
    dur_ms = 1000*(toc - tic) / iters_number
    print(f"Averaged time: {dur_ms:0.4f} milliseconds for {iters_number} iterations of inference of {model_name} model by {framework_name}")

def compare_outputs(actual, desired, rtol=5e-5, atol=5e-5):
    actual = np.asanyarray(actual)
    desired = np.asanyarray(desired)
    assert actual.dtype == desired.dtype
    np.testing.assert_allclose(actual.shape, desired.shape)
    np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol, verbose=True)

def get_tvm_vm_runner(onnx_path,
                      input_shapes,
                      input_data,
                      opt_level=3,
                      target="llvm",
                      target_host="llvm",):
    mod, params = relay.frontend.from_onnx(onnx_path, input_shapes, freeze_params=True)
    mod = relay.transform.DynamicToStatic()(mod)

    dev = tvm.device(str(target), 0)
    vm_exec = get_tvm_virtual_machine(mod,
                                      opt_level,
                                      target,
                                      target_host,
                                      params,
                                      dev,)

    tvm_inputs = {input_name: tvm.nd.array(input) for (input_name, input) in input_data}
    vm_exec.set_input(func_name="main", **tvm_inputs)
    return vm_exec.run

ANSOR_TYPE = "Ansor"
AUTO_TVM_TYPE = "AutoTVM"
def get_tvm_virtual_machine(mod,
        opt_level,
        target,
        target_host,
        params,
        dev,
        nhwc = False,
        tuning_logfile = "",
        tuning_type = ANSOR_TYPE):
    def get_tvm_vm_lib(irmod, target, target_host, params):
        return vm.compile(
            irmod,
            target,
            params=params,
            target_host=target_host,
        )

    if tuning_logfile == "":
        tuning_logfile = os.getenv("AUTOTVM_TUNING_LOG")
    lib = None
    if tuning_logfile:
        print("Use tuning file from ", tuning_logfile, ": ", tuning_logfile)
        if tuning_type == ANSOR_TYPE:
            desired_layouts = {
                "nn.conv2d": ["NHWC", "default"],
                "nn.conv2d_transpose": ["NHWC", "default"],
                "nn.upsampling": ["NHWC", "default"],
                "vision.roi_align": ["NHWC", "default"],
            }
            with auto_scheduler.ApplyHistoryBest(tuning_logfile):
                with tvm.transform.PassContext(
                    opt_level=opt_level,
                    config={
                        "relay.backend.use_auto_scheduler": True,
                        "relay.FuseOps.max_depth": 30,
                        }
                    ):
                    if nhwc:
                        mod = relay.transform.InferType()(mod)
                        model_nhwc = relay.transform.ConvertLayout(desired_layouts)(mod)
                        model_nhwc = relay.transform.EliminateCommonSubexpr()(model_nhwc)
                        mod = relay.transform.FoldConstant()(model_nhwc)
                    lib = get_tvm_vm_lib(mod, target, target_host, params)
        elif tuning_type == AUTO_TVM_TYPE:
            with relay.build_config(opt_level=opt_level):
                with autotvm.apply_history_best(tuning_logfile):
                    lib = get_tvm_vm_lib(mod, target, target_host, params)
        else:
            print("ERROR: Tuning log type {} is unsupported. ".format(tuning_type),
                "Only {} and {} types are supported".format(ANSOR_TYPE, AUTO_TVM_TYPE))
            return None
    else:
        with tvm.transform.PassContext(opt_level=opt_level):
            lib = get_tvm_vm_lib(mod, target, target_host, params)

    if lib is None:
        return None

    return tvm.runtime.vm.VirtualMachine(lib, dev)
