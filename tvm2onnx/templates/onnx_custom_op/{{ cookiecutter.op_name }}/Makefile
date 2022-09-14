# Makefile to compile an onnxruntime custom op library

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
