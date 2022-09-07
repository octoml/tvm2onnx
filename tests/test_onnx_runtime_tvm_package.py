"""Tests ONNX Packaging."""
import os
import pathlib
import tarfile
import tempfile
import shutil
import time
from functools import partial

import numpy as np
import onnx
import onnxruntime
from onnx import ModelProto, mapping
from onnx import TensorProto
from onnx.external_data_helper import convert_model_to_external_data
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_tensor,
    make_tensor_value_info,
)
from tvm2onnx.onnx_model import ONNXModel
import tvm
from tvm import relay, auto_scheduler
from tvm.relay import vm
from tvm import autotvm

from typing import AnyStr, Dict, List

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "testdata/abtest.onnx")


def get_input_data_for_model_with_fixed_shapes(onnx_model: ModelProto) -> Dict[AnyStr, np.ndarray]:
    """
    Create input data for model with static shapes
    """

    def get_onnx_input_names(model: ModelProto) -> List[AnyStr]:
        inputs = [node.name for node in model.graph.input]
        initializer = [node.name for node in model.graph.initializer]
        inputs = list(set(inputs) - set(initializer))
        return sorted(inputs)

    def get_onnx_input_types(model: ModelProto) -> List[np.dtype]:
        input_names = get_onnx_input_names(model)
        return [
            mapping.TENSOR_TYPE_TO_NP_TYPE[node.type.tensor_type.elem_type]
            for node in sorted(model.graph.input, key=lambda node: node.name)
            if node.name in input_names
        ]

    def get_onnx_input_shapes(model: ModelProto) -> List[List[int]]:
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
    random_inputs = [np.random.uniform(size=shape).astype(dtype) for shape, dtype in zip(input_shapes, input_types)]
    return dict(zip(input_names, random_inputs))


def generate_input_shapes_dtypes(onnx_model_path):
    def get_onnx_input_names(model: ModelProto) -> List[AnyStr]:
        inputs = [node.name for node in model.graph.input]
        initializer = [node.name for node in model.graph.initializer]
        inputs = list(set(inputs) - set(initializer))
        return sorted(inputs)

    onnx_model = onnx.load_model(onnx_model_path)
    input_shapes = {}
    input_dtypes = {}
    valid_names = get_onnx_input_names(onnx_model)
    for inp in onnx_model.graph.input:
        name = inp.name
        if name in valid_names:
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


def get_ort_inference_session(onnx_path, custom_lib = None):
    sess_options = onnxruntime.SessionOptions()
    if custom_lib:
        sess_options.register_custom_ops_library(custom_lib)

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

    tvm_inputs = {input_name: tvm.nd.array(input) for (input_name, input) in input_data.items()}
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


def test_mnist():
    with tempfile.TemporaryDirectory() as tdir:
        from tvm2onnx import TVM2ONNX_PACKAGE_ROOT_DIR
        project_root = os.path.split(TVM2ONNX_PACKAGE_ROOT_DIR)[0]
        # model_path = os.path.join(project_root, "tests", "testdata", "mnist.onnx")
        model_path = os.path.join(project_root, "tests", "testdata", "big_model.onnx")

        source_model = ONNXModel.from_file(pathlib.Path(model_path))
        source_model.infer_and_update_inputs()
        relay_model = source_model.to_relay()
        onnx_path = os.path.join(tdir, "test_model.tvm.onnx")
        relay_model.package_to_onnx(
            name="test_model",
            tvm_target="llvm",
            output_path=onnx_path,
        )
        model_dir = os.path.join(tdir, "model")
        with tarfile.open(onnx_path, "r") as tar:
            tar.extractall(model_dir)
        onnx_model_path = os.path.join(model_dir, "test_model.onnx")
        custom_lib = os.path.join(model_dir, "custom_test_model.so")

        benchmark_test = partial(perf_test, iters_number = 20, model_name = "test model")

        input_shapes, input_dtypes = generate_input_shapes_dtypes(model_path)
        input_data = generate_input_data(input_shapes=input_shapes, input_dtypes=input_dtypes)

        # ORT with custom op
        engine = get_ort_inference_session(onnx_model_path, custom_lib)
        ort_runner = partial(engine.run, output_names=None, input_feed=input_data)

        benchmark_test(ort_runner, framework_name = "ONNX Runtime with Custom Op")

        # pure ORT
        engine = get_ort_inference_session(model_path)
        ort_runner = partial(engine.run, output_names=None, input_feed=input_data)

        benchmark_test(ort_runner, framework_name = "Pure ONNX Runtime")

        # Pure TVM (VirtualMachine)
        tvm_runner = get_tvm_vm_runner(onnx.load(model_path), input_shapes, input_data)

        benchmark_test(tvm_runner, framework_name = "TVM")


def test_onnx_package():
    with tempfile.TemporaryDirectory() as tdir:
        source_model = ONNXModel.from_file(_MODEL_PATH)
        source_model.infer_and_update_inputs()
        relay_model = source_model.to_relay()
        onnx_path = os.path.join(tdir, "test_model.tvm.onnx")
        relay_model.package_to_onnx(
            name="test_model",
            tvm_target="llvm",
            output_path=onnx_path,
        )
        model_dir = os.path.join(tdir, "model")
        with tarfile.open(onnx_path, "r") as tar:
            tar.extractall(model_dir)
        onnx_model_path = os.path.join(model_dir, "test_model.onnx")
        custom_lib = os.path.join(model_dir, "custom_test_model.so")

        input_data = {}
        input_data["a"] = np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=np.float32
        )
        input_data["b"] = np.array(
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], dtype=np.float32
        )

        sess_options = onnxruntime.SessionOptions()
        sess_options.register_custom_ops_library(custom_lib)

        engine = onnxruntime.InferenceSession(
            onnx_model_path,
            providers=["CPUExecutionProvider"],
            provider_options=[{}],
            sess_options=sess_options,
        )
        output_data = engine.run(output_names=None, input_feed=input_data)

        sum = input_data["a"] + input_data["b"]
        product = input_data["a"] * input_data["b"]
        actual_sum = output_data[0]
        actual_product = output_data[1]
        assert np.allclose(sum, actual_sum)
        assert np.allclose(product, actual_product)


def add_constant_onnx_model(model_dir, input_shape, uniform=False):
    """Returns an ONNX model with external constants."""
    a = make_tensor_value_info("a:0", TensorProto.FLOAT, input_shape)

    if uniform:
        c1_data = np.full(shape=input_shape, fill_value=3, dtype=np.dtype("float32"))
        c2_data = np.full(shape=input_shape, fill_value=4, dtype=np.dtype("float32"))
    else:
        c1_data = np.random.randn(*input_shape).astype(np.dtype("float32"))
        c2_data = np.random.randn(*input_shape).astype(np.dtype("float32"))
    c1 = make_node(
        "Constant",
        inputs=[],
        outputs=["c1"],
        name="c1_const_data",
        value=make_tensor(
            name="c1_tensor",
            data_type=TensorProto.FLOAT,
            dims=c1_data.shape,
            vals=c1_data.flatten().tobytes(),
            raw=True,
        ),
    )
    print(f"const array size {c1_data.size * 4}")

    c2 = make_node(
        "Constant",
        inputs=[],
        outputs=["c2"],
        name="c2_const_data",
        value=make_tensor(
            name="c2_tensor",
            data_type=TensorProto.FLOAT,
            dims=c2_data.shape,
            vals=c2_data.flatten().tobytes(),
            raw=True,
        ),
    )

    add = make_node("Add", ["a:0", "c1"], ["add"])
    mul = make_node("Mul", ["add", "c2"], ["result"])

    result = make_tensor_value_info("result", TensorProto.FLOAT, input_shape)

    graph = make_graph(
        nodes=[c1, add, c2, mul], name="ab_model", inputs=[a], outputs=[result]
    )

    onnx_proto = make_model(graph)
    onnx.checker.check_model(onnx_proto)

    onnx_model = ONNXModel(model=onnx_proto)
    onnx_model.infer_and_update_inputs()
    relay_model = onnx_model.to_relay()
    relay_model.to_tvm_file("/usr/constant_add.tvm")

    model_path = os.path.join(model_dir, "test.onnx")
    convert_model_to_external_data(
        onnx_proto,
        all_tensors_to_one_file=False,
        size_threshold=1024,
        convert_attribute=True,
    )
    onnx.save(onnx_proto, model_path)
    return c1_data, c2_data


def test_constant_model():
    input_shape = [8, 3, 224, 224]
    with tempfile.TemporaryDirectory() as tdir:
        model_path = os.path.join(tdir, "test.onnx")
        c1_data, c2_data = add_constant_onnx_model(
            model_dir=tdir, input_shape=input_shape, uniform=True
        )
        onnx_model = ONNXModel.from_file(model_path)
        onnx_model.infer_and_update_inputs()
        relay_model = onnx_model.to_relay()
        onnx_path = os.path.join(tdir, "test_model.tvm.onnx")
        relay_model.package_to_onnx(
            name="test_model",
            tvm_target="llvm",
            output_path=onnx_path,
        )
        model_dir = os.path.join(tdir, "model")
        with tarfile.open(onnx_path, "r") as tar:
            tar.extractall(model_dir)

        onnx_model_path = os.path.join(model_dir, "test_model.onnx")
        custom_lib = os.path.join(model_dir, "custom_test_model.so")

        input_data = {}
        input_data["a"] = np.random.randn(*c1_data.shape).astype(np.dtype("float32"))

        sess_options = onnxruntime.SessionOptions()
        sess_options.register_custom_ops_library(custom_lib)

        engine = onnxruntime.InferenceSession(
            onnx_model_path,
            providers=["CPUExecutionProvider"],
            provider_options=[{}],
            sess_options=sess_options,
        )
        result = engine.run(output_names=None, input_feed=input_data)

        expected = (input_data["a"] + c1_data) * c2_data
        actual = result[0]
        assert np.allclose(expected, actual)


def main():
    test_mnist()


if __name__ == '__main__':
    main()
