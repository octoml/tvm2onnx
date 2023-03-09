# Tutorial

## Introduction

In this tutorial we will walk through the end-to-end process of
* Converting an ONNX model to TVM
* Optimizing the TVM model
* Packaging the TVM model in ONNX
* Running the new ONNX model in onnxruntime

tvm2onnx is designed to run on Linux and is not tested on other platforms. For reproducibility this tutorial was tested on an AWS *m6i.24xlarge* instance.

## Building TVM

TVM models depend on a tvm_runtime library for execution. The version of TVM used to generate
the TVM model and the version of the tvm_runtime must match exactly. In order to ensure these
versions match, tvm2onnx statically links tvm_runtime to the packaged model.

By default when building TVM, tvm_runtime is built with dynamic linkage. For this tutorial we will
build TVM and specify that tvm_runtime is built for static linkage. Because tvm_runtime is statically linked to the model any version of TVM can be used. tvm2onnx is not tied to a particular version of TVM.

NOTE: This tutorial is geared to build TVM for CPU only. To use a GPU you will need to follow instructions on the TVM site.

The following command should be run from a directory which will be referred to as PROJECT_ROOT in this document. For this tutorial we will use PROJECT_ROOT=~/tvm2onnx_tutorial

```bash
mkdir ~/tvm2onnx_tutorial
cd ~/tvm2onnx_tutorial
git clone --recursive https://github.com/apache/tvm.git
cd tvm
mkdir build
cp cmake/config.cmake .
echo "set(BUILD_STATIC_RUNTIME ON)" >> config.cmake
echo "set(USE_FALLBACK_STL_MAP ON)" >> config.cmake
cd build
cmake ..
make -j 8
```

When done, the build will produce both *libtvm.so* and *libtvm_runtime.a* in the build directory.

## Tuning Your TVM Model

```bash
python scripts/autotvm_model.py --model tutorial/super-resolution.onnx --output opt_model --axis-size batch_size=1
```

## Convert Your TVM Model to ONNX

```bash
python scripts/onnx_package.py --model opt_model/model.o --ro opt_model/vm_exec_code.ro --constants opt_model/constants.pkl --metadata opt_model/metadata.json --tvm-runtime 3rdparty/tvm/build/libtvm_runtime.a --output demo.onnx.tar
```

## Run Your New ONNX Model with onnxruntime

After converting the model to .onnx format you should have a demo.onnx.tar file. This tar file contains the .onnx model file along with external constants and the tvm models custom op shared library. To run the model in onnx first untar the file to a directory
```bash
mkdir demo_model
tar -xf demo.onnx.tar -C demo_model
```
The demo_model directory will contain the following files. The const files may be named differently.
* const_10
* const_14
* const_20
* const_4
* custom_demo.so
* demo.onnx

|![](demo_model.png "Converted model in netron.app")|
|:--:|
|*A view of demo.onnx in netron.app*|

The entire TVM-compiled model is contained in the node *demo* which is implemented as an onnxruntime custom operator.

To run the model use the following python code. The function *register_custom_ops_library* registers the model's custom op library with onnxruntime.
```python
sess_options = onnxruntime.SessionOptions()
sess_options.register_custom_ops_library(<path to custom op>)
engine = onnxruntime.InferenceSession(
    onnx_path,
    providers=["CPUExecutionProvider"],
    provider_options=[{}],
    sess_options=sess_options,
)
```