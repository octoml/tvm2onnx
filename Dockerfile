# Copyright 2023 OctoML
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM nvcr.io/nvidia/tensorrt:22.07-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update --fix-missing && \
    apt-get install -y software-properties-common && \
    apt-get update && \
    apt-get install -y \
        build-essential \
        clang-12 \
        lld-12 \
        git \
        libopenblas-dev \
        gcc-aarch64-linux-gnu

# Install a more modern cmake version
WORKDIR /usr
RUN wget --no-check-certificate -q https://github.com/Kitware/CMake/releases/download/v3.25.2/cmake-3.25.2-linux-x86_64.sh && \
    chmod +x cmake-3.25.2-linux-x86_64.sh && \
    ./cmake-3.25.2-linux-x86_64.sh --skip-license --prefix=/usr/local && \
    rm -rf cmake-3.25.2-linux-x86_64.sh


ENV TVM2ONNX_HOME="/usr/tvm2onnx"
ENV THIRDPARTY_HOME="${TVM2ONNX_HOME}/3rdparty"
ENV TVM_HOME="${THIRDPARTY_HOME}/tvm"
ENV ORT_HOME="${THIRDPARTY_HOME}/onnxruntime"
ENV PATH="/root/.poetry/bin:${TVM_HOME}/build:$PATH"
ENV PYTHONPATH=${TVM2ONNX_HOME}:${TVM_HOME}/python:${PYTHONPATH}

# Build TVM before we copy all the project source files
# This is so we don't have to rebuild TVM every time we modify project source
WORKDIR ${THIRDPARTY_HOME}
# For TVM I can't checkout a hash directly, I need to clone then checkout the hash
RUN git clone \
    --recursive \
    https://github.com/apache/tvm.git && \
    cd tvm && \
    git checkout 082c4432fb48ea7dc58f553a7272e7a2585bb550 && \
    git submodule update

WORKDIR ${THIRDPARTY_HOME}
RUN git clone \
    -b v1.14.1 \
    --depth 1 \
    https://github.com/microsoft/onnxruntime.git

WORKDIR ${TVM2ONNX_HOME}
COPY pyproject.toml poetry.lock ./

RUN pip install --upgrade pip && \
    pip install poetry==1.1.15 && \
    poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --no-root -v

WORKDIR ${TVM_HOME}
RUN mkdir -p build && \
    cd       build && \
    cp ../cmake/config.cmake . && \
    echo "set(USE_LLVM llvm-config-12)" >> config.cmake && \
    echo "set(USE_LIBBACKTRACE OFF)" >> config.cmake && \
    echo "set(USE_SORT ON)" >> config.cmake && \
    echo "set(USE_RPC ON)" >> config.cmake && \
    # TODO: rkimball build cuda/non-cuda builds
    # echo "set(USE_CUDA ON)" >> config.cmake && \
    # echo "set(USE_CUDNN ON)" >> config.cmake && \
    # echo "set(USE_CUBLAS ON)" >> config.cmake && \
    # echo "set(USE_VULKAN OFF)" >> config.cmake && \
    # echo "set(USE_PROFILER ON)" >> config.cmake && \
    echo "set(BUILD_STATIC_RUNTIME ON)" >> config.cmake && \
    echo "set(USE_FALLBACK_STL_MAP ON)" >> config.cmake && \
    cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo && \
    make -j $(nproc) && \
    strip libtvm.so

# Environment variables for CUDA.
ENV PATH=/usr/local/cuda/bin:/usr/local/nvidia/bin:${PATH}
# Note that /usr/local/cuda/compat/lib.real provides a "libcuda.so of last resort", but must
# not be used for actual cloud GPU scenarios. In those cases, libcuda.so should be mounted
# from the host VM instead.
#
# GCP: /usr/local/nvidia/lib64 (already present in LD_LIBRARY_PATH)
# AWS: /usr/lib/x86_64-linux-gnu (must be added earlier than /usr/local/cuda/compat/lib.real)
#
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/compat/lib.real:/opt/intel/openvino_2021/inference_engine/lib/intel64/:/opt/intel/openvino_2021.4.689/deployment_tools/ngraph/lib/:/opt/intel/openvino_2021.4.689/deployment_tools/inference_engine/external/tbb/lib/
ENV C_INCLUDE_PATH=${THIRDPARTY_HOME}/onnxruntime/include
ENV CPLUS_INCLUDE_PATH=${THIRDPARTY_HOME}/onnxruntime/include:/usr/tvm2onnx/3rdparty/tvm/3rdparty/dmlc-core/include:/usr/tvm2onnx/3rdparty/tvm/3rdparty/dlpack/include:/usr/tvm2onnx/3rdparty/tvm/include

COPY . /usr/tvm2onnx
WORKDIR /usr/tvm2onnx

CMD ["bash"]
