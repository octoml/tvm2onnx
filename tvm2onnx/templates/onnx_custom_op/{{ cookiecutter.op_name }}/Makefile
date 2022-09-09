# Makefile to compile an onnxruntime custom op library

TVM_RUNTIME_PATH=$(TVM_HOME)/build
FLAGS=-DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\> -DTVM_USE_LIBBACKTRACE=0 -DUSE_FALLBACK_STL_MAP=1

custom_{{cookiecutter.module_name}}.so: custom_op_library.cc
	g++ \
		-shared \
		-Iinclude \
		-I. \
		-I$(TVM_HOME)/include \
		-I$(TVM_HOME)/3rdparty/dlpack/include \
		-I${TVM_HOME}/3rdparty/dmlc-core/include \
		-I${ORT_HOME}/include \
		-fPIC \
		-o custom_{{cookiecutter.module_name}}.so \
		custom_op_library.cc \
		-Wl,--exclude-libs,ALL \
		-Wl,--whole-archive $(TVM_RUNTIME_PATH)/libtvm_runtime.a -Wl,--no-whole-archive \
		-L. \
		-std=c++17 \
		-Wl,--format=binary -Wl,{{cookiecutter.model_ro}} -Wl,--format=default \
		-Wl,--format=binary -Wl,{{cookiecutter.model_so}} -Wl,--format=default \
		$(FLAGS)
	strip custom_{{cookiecutter.module_name}}.so
