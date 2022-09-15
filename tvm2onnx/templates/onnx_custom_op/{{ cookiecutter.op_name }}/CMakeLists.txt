cmake_minimum_required(VERSION 3.13)
set(LIB_NAME custom_{{cookiecutter.module_name}})

project(custom_op LANGUAGES C CXX)

add_library(
    ${LIB_NAME} MODULE
    custom_op_library.cc
)

set(CMAKE_SHARED_LIBRARY_PREFIX "blah_")
target_include_directories(
    ${LIB_NAME}
    PRIVATE
        include
        $ENV{TVM_HOME}/include
        $ENV{TVM_HOME}/3rdparty/dlpack/include
        $ENV{TVM_HOME}/3rdparty/dmlc-core/include
        $ENV{ORT_HOME}/include
)

target_compile_definitions(
    ${LIB_NAME}
    PRIVATE
        DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>
        TVM_USE_LIBBACKTRACE=0
        USE_FALLBACK_STL_MAP=1
)

target_compile_features(${LIB_NAME} PRIVATE cxx_std_17)

target_link_libraries(
    ${LIB_NAME}
    PRIVATE
        "-Wl,--exclude-libs,ALL"
        "-Wl,--whole-archive"
        {{cookiecutter.libtvm_runtime_a}}
        "-Wl,--no-whole-archive"
        "-Wl,--format=binary -Wl,vm_exec_code.ro -Wl,--format=default"
        "-Wl,--format=binary -Wl,model.so -Wl,--format=default"

)




# cmake_minimum_required(VERSION 3.13)

# project(custom_op LANGUAGES C CXX)

# add_library(
#     ${LIB_NAME} SHARED
#     custom_op_library.cc
# )

# target_compile_features(${LIB_NAME} PRIVATE cxx_std_17)

# target_include_directories(
#     ${LIB_NAME}
#     PRIVATE
#         include
#         $ENV{TVM_HOME}/include
#         $ENV{TVM_HOME}/3rdparty/dlpack/include
#         $ENV{TVM_HOME}/3rdparty/dmlc-core/include
#         $ENV{ORT_HOME}/include
# )

# target_compile_definitions(
#     ${LIB_NAME}
#     PRIVATE
#         DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>
#         TVM_USE_LIBBACKTRACE=0
#         USE_FALLBACK_STL_MAP=1
# )

# # target_link_directories(
# #     ${LIB_NAME}
# #     PRIVATE
# #         ${repo-tvm_BINARY_DIR}
# # )

# target_link_libraries(
#     ${LIB_NAME}
#     PRIVATE
# )

# target_link_options(
#     ${LIB_NAME}
#         PRIVATE_HEADER  "LINKER:-exclude-libs,ALL"
# )
