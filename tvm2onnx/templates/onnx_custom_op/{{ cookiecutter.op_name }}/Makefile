# Makefile to compile an onnxruntime custom op library

TVM_ROOT={{cookiecutter.tvm_root}}
TVM_RUNTIME_PATH={{cookiecutter.tvm_runtime_path}}
ONNX_RUNTIME_ROOT={{cookiecutter.onnx_runtime_root}}
FLAGS=-DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\> -DTVM_USE_LIBBACKTRACE=0

custom_{{cookiecutter.module_name}}.so: custom_op_library.cc
	g++ \
		-shared \
		-Iinclude \
		-I. \
		-I$(TVM_ROOT)/include \
		-I$(TVM_ROOT)/3rdparty/dlpack/include \
		-I${TVM_ROOT}/3rdparty/dmlc-core/include \
		-I$(ONNX_RUNTIME_ROOT)/include \
		-fPIC \
		{% if cookiecutter.debug_build -%}
		-g \
		-O0 \
		{% else -%}
		-O2 \
		{% endif -%}
		-o custom_{{cookiecutter.module_name}}.so \
		custom_op_library.cc \
		-Wl,--exclude-libs,ALL \
		-Wl,--whole-archive {{cookiecutter.libtvm_runtime_a}} -Wl,--no-whole-archive \
		-L. \
		{% for path in cookiecutter.library_search_paths -%}
		-L{{path}} \
		{% endfor -%}
		-std=c++17 \
		-Wl,--format=binary -Wl,vm_exec_code.ro -Wl,--format=default \
		-Wl,--format=binary -Wl,model.so -Wl,--format=default \
		{% for lib in cookiecutter.dynamic_libraries -%}
		-l{{lib}} \
		{% endfor -%}
		$(FLAGS)
	{%- if not cookiecutter.debug_build %}
	strip custom_{{cookiecutter.module_name}}.so
	{%- endif %}
