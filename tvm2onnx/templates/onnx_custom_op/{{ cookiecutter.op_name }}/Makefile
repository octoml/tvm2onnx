# Makefile to compile an onnxruntime custom op library

TVM_ROOT=/usr/tvm2onnx/3rdparty/tvm
TVM_RUNTIME_PATH=$(TVM_ROOT)/build
FLAGS=-DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\> -DTVM_USE_LIBBACKTRACE=0 -DUSE_FALLBACK_STL_MAP=1

custom_{{cookiecutter.module_name}}.so: custom_op_library.cc
	g++ \
		-shared \
		-Iinclude \
		-I. \
		-I$(TVM_ROOT)/include \
		-I$(TVM_ROOT)/3rdparty/dlpack/include \
		-I${TVM_ROOT}/3rdparty/dmlc-core/include \
		-I/usr/tvm2onnx/3rdparty/onnxruntime/include \
		-fPIC \
		-o custom_{{cookiecutter.module_name}}.so \
		custom_op_library.cc \
		-Wl,--exclude-libs,ALL \
		-Wl,--whole-archive {{cookiecutter.libtvm_runtime_a}} -Wl,--no-whole-archive \
		-L. \
		-std=c++17 \
		-Wl,--format=binary -Wl,vm_exec_code.ro -Wl,--format=default \
		-Wl,--format=binary -Wl,model.so -Wl,--format=default \
		$(FLAGS)
	strip custom_{{cookiecutter.module_name}}.so
