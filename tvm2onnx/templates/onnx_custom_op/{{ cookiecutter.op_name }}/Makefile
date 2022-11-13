# Makefile to compile an onnxruntime custom op library

.DEFAULT_GOAL := custom_{{cookiecutter.module_name}}.so

FLAGS=-DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\> -DTVM_USE_LIBBACKTRACE=0 -DUSE_FALLBACK_STL_MAP=1

model.h : model.so
	xxd -i -C model.so > model_so.h

vm_exec_code.h : vm_exec_code.ro
	xxd -i -C vm_exec_code.ro > vm_exec_code_ro.h

custom_{{cookiecutter.module_name}}.so: custom_op_library.cc model.h vm_exec_code.h
	{{cookiecutter.compiler}} \
		-shared \
		-Iinclude \
		-I. \
		-O2 \
		-o custom_{{cookiecutter.module_name}}.so \
		custom_op_library.cc \
		-Wl,--whole-archive {{cookiecutter.libtvm_runtime_a}} -Wl,--no-whole-archive \
		-L. \
		-std=c++17 \
		-stdlib=libc++ \
		-lc++abi \
		$(FLAGS) {{cookiecutter.compiler_flags}} \
