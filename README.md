# tvm2onnx
Convert pre-optimized-and-compiled TVM models to .onnx format for execution in ONNX Runtime.

## Getting Started
This project is designed to run in a docker container and integrate with Visual Studio Code. For
Visual Studio Code add the Remove Development (ms-vscode-remote.vscode-remote-extensionpack)
extension. With this installed you can click the green status display in the lower left
of VSCode and select "Open in Remove Container". This should build the docker container and
launch an interactive session. Testing and developer tools lint and format are designed to work
in this container.
## Developing
Code development should be done in the docker container as it maintains all project dependencies.
While it may be possible to develop without using the docker container that is not supported.
## Running lint and test
From the root development directory in the docker container you can run</br>
`make format` to format the code. Proper formatting is required for merge requests.</br>
`make lint` to lint the code. Passing lint is required for merge requests.</br>
`make test` to run the test suite.
## Scripts
Scripts are located in the tvm2onnx/scripts directory.
`onnx_package.py`  can package a TVM model to a .tvm.onnx
file which is simply an onnx model in tar format. The packaged model contains code to implement
custom ops which is also located in the .tvm.onnx. This custom op files is registered with the
ONNX Runtime. The python code below demonstrates using the custom op library.
```
        sess_options = onnxruntime.SessionOptions()
        sess_options.register_custom_ops_library(<path_to_custom_op_library>)
        engine = onnxruntime.InferenceSession(
            onnx_model_path,
            sess_options=sess_options,
        )
        output_data = engine.run(output_names=None, input_feed=input_data)
```
`onnx_benchmark.py` runs an ONNX model which is in the .tvm.onnx file format.