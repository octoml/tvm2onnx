cd /usr/tvm2onnx/3rdparty/onnxruntime && \
    ./build.sh \
        --update \
        --build \
        --use_tvm \
        --config Release \
        --skip_tests \
        --build_wheel \
        --parallel $(nproc)
python3 -m pip uninstall -y onnxruntime_tvm-1.13.0-cp38-cp38-linux_x86_64.whl
python3 -m pip install /usr/tvm2onnx/3rdparty/onnxruntime/build/Linux/Release/dist/*.whl
