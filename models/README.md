Any model in this directory is packaged and checked. Since the models are not optimized
no benchmarks are run. 

This directory is excluded from the docker context so it's contents are not copied
to the container. Instead this directory is mounted to /usr/tvm2onnx/models and is used for
the test_models slow unit test.
