#include "custom_op_library.h"

#define ORT_API_MANUAL_INIT
#include "onnxruntime/core/session/onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include "onnxruntime/core/common/common.h"

#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <mutex>
#include <iostream>
#include <fstream>
#include <regex>
#include <filesystem>
#include <memory>

#include <dlpack/dlpack.h>
#include <tvm/runtime/container/adt.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/vm/vm.h>

extern uint8_t vm_exec_code_ro_start[] asm("_binary_vm_exec_code_ro_start");
extern uint8_t vm_exec_code_ro_end[] asm("_binary_vm_exec_code_ro_end");
extern const char model_so_start[] asm("_binary_model_so_start");
extern const char model_so_end[] asm("_binary_model_so_end");

namespace {
static const char* c_OpDomain = "{{ cookiecutter.domain }}";

static void AddOrtCustomOpDomainToContainer(Ort::CustomOpDomain&& domain) {
  static std::vector<Ort::CustomOpDomain> ort_custom_op_domain_container;
  static std::mutex ort_custom_op_domain_mutex;
  std::lock_guard<std::mutex> lock(ort_custom_op_domain_mutex);
  ort_custom_op_domain_container.push_back(std::move(domain));
}

class TempFile {
  public:
  TempFile() {
    std::string tmp_path = std::filesystem::temp_directory_path();
    tmp_path = tmp_path + "/tvm_model_XXXXXX.so";
    // +1 for the null terminator
    auto tmp_buffer = std::unique_ptr<char[]>(new char[tmp_path.size()+1]);
    strcpy(tmp_buffer.get(), tmp_path.c_str());
    mkstemps(tmp_buffer.get(), 3);
    filename = std::string(tmp_buffer.get());
  }

  ~TempFile() {
    auto rc = std::remove(filename.c_str());
  }
  std::string filename;
};

struct TVMRuntime {
  TVMRuntime() {
    // Binary data is linked into this shared library
    // These symbols are defined by adding lines like this to the compile string
    // -Wl,--format=binary -Wl,vm_exec_code.ro -Wl,--format=default
    size_t vm_exec_code_ro_size = vm_exec_code_ro_end - vm_exec_code_ro_start;
    size_t model_so_size = model_so_end - model_so_start;

    DLDeviceType dl_device_type = {{ cookiecutter.dl_device_type }};

    // TVM's model shared library needs to be a standalone shared lib
    std::ofstream model_so_f(model_so_file.filename, std::ios::binary);
    model_so_f.write(model_so_start, model_so_size);
    model_so_f.close();
    tvm::runtime::Module lib = tvm::runtime::Module::LoadFromFile(model_so_file.filename);

    // Copy vm_exec_code to a string for TVM consumption.
    std::stringstream ss;
    ss.write((const char*)&vm_exec_code_ro_start, vm_exec_code_ro_size);

    exec_mod = tvm::runtime::vm::Executable::Load(ss.str(), lib);
    const tvm::runtime::vm::Executable* tmp =
        exec_mod.as<tvm::runtime::vm::Executable>();
    exec = tvm::runtime::GetObjectPtr<tvm::runtime::vm::Executable>(
        const_cast<tvm::runtime::vm::Executable*>(tmp));
  }

  ~TVMRuntime() {
  }

  void LateBoundConstants(const Ort::KernelContext& ctx) {
    DLDevice dl_device_type = {DLDeviceType::{{ cookiecutter.dl_device_type }}, 0};
    tvm::runtime::Map<::tvm::runtime::String, ::tvm::runtime::NDArray> const_map;

    {% for details in cookiecutter.initializers -%}
    auto _{{details.name}} = ctx.GetInput({{details.index}});
    const {{details.cpp_type}}* _{{details.name}}_ptr = _{{details.name}}.GetTensorData<{{details.cpp_type}}>();
    DLDataType _{{details.name}}_dtype = tvm::runtime::String2DLDataType("{{details.numpy_dtype}}");
    tvm::runtime::NDArray _{{details.name}}_ndarray = tvm::runtime::NDArray::Empty({{details.shape}}, _{{details.name}}_dtype, dl_device_type);
    _{{details.name}}_ndarray.CopyFromBytes(_{{details.name}}_ptr, {{details.element_count}}*sizeof({{details.cpp_type}}));
    const_map.Set("{{details.base_name}}", _{{details.name}}_ndarray);
    {% endfor %}

    // We can't LoadExecutable util we have added late-bound constants so the actual creation
    // of loading takes place here.
    // void Executable::LoadLateBoundConstantsFromMap(Map<String, NDArray> map)
    // tvm::runtime::Map<tvm::runtime::String, tvm::runtime::NDArray> runtime_map(const_map);
    exec->LoadLateBoundConstantsFromMap(const_map);
    vm.LoadExecutable(exec);

    // Initialize the VM for the specified device. If the device is not a CPU,
    // We'll need to add a CPU context to drive it.
    int arity;
    if (dl_device_type.device_type == kDLCPU) {
      arity = 3;
    } else {
      arity = 6;
    }
    // Specify how to allocate memory for the target devices.
    uint64_t alloc_type = uint64_t(tvm::runtime::vm::AllocatorType::kPooled);
    // TODO: rkimball use proper device
    uint64_t device_id = 0;
    // Create a variable length input to the packed function.
    std::vector<TVMValue> init_vals(arity);
    std::vector<int> codes(arity);
    tvm::runtime::TVMArgsSetter setter(init_vals.data(), codes.data());
    // Set up the main device context.
    setter(0, (uint64_t(dl_device_type.device_type)));
    setter(1, device_id);
    setter(2, alloc_type);
    // Also initialize a CPU device context.
    if (dl_device_type.device_type != kDLCPU) {
      setter(3, (uint64_t(kDLCPU)));
      setter(4, device_id);
      setter(5, alloc_type);
    }
    tvm::runtime::TVMRetValue rv;
    // Call the packed func with the init arguments.
    vm.GetFunction("init", nullptr)
        .CallPacked(
            tvm::runtime::TVMArgs(init_vals.data(), codes.data(), arity), &rv);

    set_input_func = vm.GetFunction("set_input", nullptr);
    set_outputs_func = vm.GetFunction("set_outputs", nullptr);
    get_output_func = vm.GetFunction("get_output", nullptr);
    run_func = vm.GetFunction("invoke", nullptr);
  }

  static DLDataType GetDataType(const std::string& type) {
    // TODO(vvchernov): support float16 -> {kDLFloat, 16, 1}
    static std::unordered_map<std::string, DLDataType> cppTypeToDLType = {
      {"float", DLDataType{kDLFloat, 32, 1}},
      {"double", DLDataType{kDLFloat, 64, 1}},
      {"int8_t", {kDLInt, 8, 1}},
      {"int", {kDLInt, 32, 1}},
      {"int64_t", {kDLInt, 64, 1}},
      {"bool", {kDLUInt, 1, 1}}
    };
    if(!cppTypeToDLType.count(type)) {
      // TODO(vvchernov): implement with ORT or TVM check API
      throw std::logic_error("Unsupported data type");
    }
    return cppTypeToDLType[type];
  }

  void Compute(OrtKernelContext* context) {
    Ort::KernelContext ctx(context);
    // Get data points for the input data
    DLDevice dl_device_type = {DLDeviceType::{{ cookiecutter.dl_device_type }}, 0};

    if (!constants_bound) {
      // During the first iteration we need to bind the late-bound constants to TVM and
      // create the VM. This is our first opportunity to access the onnx external constants.
      LateBoundConstants(ctx);
      constants_bound = true;
    }

    std::vector<DLTensor> ort_dl_inputs;
    {% for details in cookiecutter.inputs -%}
    auto input_tensor{{details.index}} = ctx.GetInput({{details.index}});
    std::vector<int64_t> input{{details.index}}_shape = {{details.shape}};

    DLTensor dl_input{{details.index}};
    // TODO(vvchernov): device?
    // auto ort_device_type = input_tensor{{details.index}}.GetTensorMemoryInfo().GetDeviceType();
    dl_input{{details.index}}.device = dl_device_type;
    dl_input{{details.index}}.dtype = tvm::runtime::String2DLDataType("{{details.numpy_dtype}}");
    dl_input{{details.index}}.strides = nullptr;
    dl_input{{details.index}}.byte_offset = 0;
    dl_input{{details.index}}.data = input_tensor{{details.index}}.GetTensorMutableRawData();
    dl_input{{details.index}}.ndim = {{details.rank}};
    dl_input{{details.index}}.shape = input{{details.index}}_shape.data();
    ort_dl_inputs.emplace_back(dl_input{{details.index}});
    {% endfor %}

    size_t num_total_iargs = ort_dl_inputs.size() + 1;
    std::vector<TVMValue> tvm_in_values(num_total_iargs);
    std::vector<int> tvm_in_type_codes(num_total_iargs);
    tvm::runtime::TVMArgsSetter isetter(tvm_in_values.data(), tvm_in_type_codes.data());
    const std::string func_name = "main";
    isetter(0, func_name.c_str());
    for (size_t k = 0; k < num_total_iargs - 1; ++k) {
      isetter(k+1, &ort_dl_inputs[k]);
    }

    tvm::runtime::TVMRetValue irv;
    set_input_func.CallPacked(
        tvm::runtime::TVMArgs(tvm_in_values.data(), tvm_in_type_codes.data(), int(num_total_iargs)), &irv);

    std::vector<DLTensor> ort_dl_outputs;
    {% for details in cookiecutter.outputs -%}
    std::vector<int64_t> output{{details.index}}_shape = {{details.shape}};
    auto output{{details.index}} = ctx.GetOutput({{details.index}}, output{{details.index}}_shape);
    // TODO(vvchernov): check output{{details.index}}->IsTensor()
    DLTensor dl_output{{details.index}};
    dl_output{{details.index}}.device = dl_device_type;
    dl_output{{details.index}}.dtype = GetDataType("{{details.cpp_type}}");
    dl_output{{details.index}}.data = output{{details.index}}.GetTensorMutableRawData();
    dl_output{{details.index}}.ndim = {{details.rank}};
    dl_output{{details.index}}.shape = output{{details.index}}_shape.data();
    ort_dl_outputs.emplace_back(dl_output{{details.index}});
    {% endfor %}

    size_t num_total_args = ort_dl_outputs.size() + 1;
    std::vector<TVMValue> tvm_values(num_total_args);
    std::vector<int> tvm_type_codes(num_total_args);
    tvm::runtime::TVMArgsSetter setter(tvm_values.data(), tvm_type_codes.data());
    const std::string func_name = "main";
    setter(0, func_name.c_str());
    for (size_t k = 0; k < num_total_args - 1; ++k) {
      setter(k+1, &ort_dl_outputs[k]);
    }

    tvm::runtime::TVMRetValue rv;
    set_outputs_func.CallPacked(tvm::runtime::TVMArgs(tvm_values.data(), tvm_type_codes.data(), num_total_args), &rv);

    // Inference
    run_func("main");
  }

 private:
  tvm::runtime::vm::VirtualMachine vm;
  tvm::runtime::PackedFunc set_input_func;
  tvm::runtime::PackedFunc set_outputs_func;
  tvm::runtime::PackedFunc get_output_func;
  tvm::runtime::PackedFunc run_func;
  tvm::runtime::Module exec_mod;
  tvm::runtime::ObjectPtr<tvm::runtime::vm::Executable> exec;
  TempFile model_so_file;
  // TODO(vvchernov): define device type for specific case. define device id
  DLDevice dl_device_type = {DLDeviceType::{{ cookiecutter.dl_device_type }}, 0};
  bool constants_bound = false;
};

struct TVMModelOp : Ort::CustomOpBase<TVMModelOp, TVMRuntime> {
  void* CreateKernel(const OrtApi& /* api */, const OrtKernelInfo* /* info */) const {
    return new TVMRuntime();
  };

  const char* GetName() const {
    auto name = "{{cookiecutter.custom_op_name}}";
    return name;
  };

  size_t GetInputTypeCount() const { return {{cookiecutter.input_count}} + {{cookiecutter.initializer_count}}; };
  ONNXTensorElementDataType GetInputType(size_t index) const {
    static ONNXTensorElementDataType input_types[] = {
      {% for input_type in cookiecutter.input_types -%}
      {{input_type}},
      {% endfor %}
    };

    return input_types[index];
  };

  size_t GetOutputTypeCount() const { return {{cookiecutter.output_count}}; };
  ONNXTensorElementDataType GetOutputType(size_t index) const {
    static ONNXTensorElementDataType output_types[] = {
      {% for output_type in cookiecutter.output_types -%}
      {{output_type}},
      {% endfor %}
    };

    return output_types[index];
  };

};
} // End anonymous namespace

OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) {
  Ort::Global<void>::api_ = api->GetApi(ORT_API_VERSION);

  static const TVMModelOp c_TVMModelOp;

  OrtStatus* result = nullptr;

  ORT_TRY {
    Ort::CustomOpDomain domain{c_OpDomain};
    domain.Add(&c_TVMModelOp);

    Ort::UnownedSessionOptions session_options(options);
    session_options.Add(domain);
    AddOrtCustomOpDomainToContainer(std::move(domain));
  } ORT_CATCH (const std::exception& e) {
    ORT_HANDLE_EXCEPTION([&]() {
      Ort::Status status{e};
      result = status.release();
    });
  }
  return result;
}
