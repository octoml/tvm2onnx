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

# Makefile to compile an onnxruntime custom op library

.DEFAULT_GOAL := custom_{{cookiecutter.module_name}}

vm_exec_code.h : vm_exec_code.ro
	xxd -i -C vm_exec_code.ro > vm_exec_code_ro.h

custom_{{cookiecutter.module_name}}: custom_op_library.cc vm_exec_code.h
	{{cookiecutter.compiler}} \
		-shared \
		-Iinclude \
		-I. \
		-O2 \
		-o custom_{{cookiecutter.module_name}} \
		custom_op_library.cc \
		-L. \
		-std=c++17 \
		{{cookiecutter.compiler_flags}} \
		-DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\> \
		-DTVM_USE_LIBBACKTRACE=0 \
		-DUSE_FALLBACK_STL_MAP=1
