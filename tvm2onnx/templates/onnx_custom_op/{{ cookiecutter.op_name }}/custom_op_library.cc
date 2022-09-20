#include "custom_op_library.h"

#define ORT_API_MANUAL_INIT
#include "onnxruntime/core/session/onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <vector>
#include <cmath>
#include <mutex>
#include <iostream>
#include <fstream>
#include <regex>
#include <filesystem>
#include <memory>
#include <initializer_list>

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

struct OrtCustomOpDomainDeleter {
  explicit OrtCustomOpDomainDeleter(const OrtApi* ort_api) {
    ort_api_ = ort_api;
  }
  void operator()(OrtCustomOpDomain* domain) const {
    ort_api_->ReleaseCustomOpDomain(domain);
  }

  const OrtApi* ort_api_;
};

using OrtCustomOpDomainUniquePtr = std::unique_ptr<OrtCustomOpDomain, OrtCustomOpDomainDeleter>;
static std::vector<OrtCustomOpDomainUniquePtr> ort_custom_op_domain_container;
static std::mutex ort_custom_op_domain_mutex;

static void AddOrtCustomOpDomainToContainer(OrtCustomOpDomain* domain, const OrtApi* ort_api) {
  std::lock_guard<std::mutex> lock(ort_custom_op_domain_mutex);
  auto ptr = std::unique_ptr<OrtCustomOpDomain, OrtCustomOpDomainDeleter>(domain, OrtCustomOpDomainDeleter(ort_api));
  ort_custom_op_domain_container.push_back(std::move(ptr));
}

struct OrtTensorDimensions : std::vector<int64_t> {
  OrtTensorDimensions(Ort::CustomOpApi ort, const OrtValue* value) {
    OrtTensorTypeAndShapeInfo* info = ort.GetTensorTypeAndShape(value);
    std::vector<int64_t>::operator=(ort.GetTensorShape(info));
    ort.ReleaseTensorTypeAndShapeInfo(info);
  }
};

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
  TVMRuntime(const OrtApi& api)
      : ort_(api) {
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

  void LateBoundConstants(OrtKernelContext* context) {
    DLDevice dl_device_type = {DLDeviceType::{{ cookiecutter.dl_device_type }}, 0};
    ::tvm::runtime::Map<::tvm::runtime::String, ::tvm::runtime::NDArray> const_map;

{% for details in cookiecutter.initializers -%}
    const OrtValue* _{{details.name}} = ort_.KernelContext_GetInput(context, {{details.index}});
    const {{details.cpp_type}}* _{{details.name}}_ptr = ort_.GetTensorData<{{details.cpp_type}}>(_{{details.name}});
    DLDataType _{{details.name}}_dtype = ::tvm::runtime::String2DLDataType("{{details.numpy_dtype}}");
    ::tvm::runtime::NDArray _{{details.name}}_ndarray = ::tvm::runtime::NDArray::Empty({{details.shape}}, _{{details.name}}_dtype, dl_device_type);
    _{{details.name}}_ndarray.CopyFromBytes(_{{details.name}}_ptr, {{details.element_count}}*sizeof({{details.cpp_type}}));
    const_map.Set("{{details.base_name}}", _{{details.name}}_ndarray);

{% endfor %}
    // We can't LoadExecutable util we have added late-bound constants so the actual creation
    // of loading takes place here.
    // void Executable::LoadLateBoundConstantsFromMap(Map<String, NDArray> map)
    // ::tvm::runtime::Map<::tvm::runtime::String, ::tvm::runtime::NDArray> runtime_map(const_map);
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
    get_output_func = vm.GetFunction("get_output", nullptr);
    run_func = vm.GetFunction("invoke", nullptr);
  }

  void Compute(OrtKernelContext* context) {
    // Get data points for the input data
    DLDevice dl_device_type = {DLDeviceType::{{ cookiecutter.dl_device_type }}, 0};

    if (!constants_bound) {
      // During the first iteration we need to bind the late-bound constants to TVM and
      // create the VM. This is our first opportunity to access the onnx external constants.
      LateBoundConstants(context);
      constants_bound = true;
    }

    std::vector<tvm::runtime::NDArray> input_vec;
    {% for details in cookiecutter.inputs -%}
    const OrtValue* input{{details.index}} = ort_.KernelContext_GetInput(context, {{details.index}});
    const {{details.cpp_type}}* input{{details.index}}_ptr = ort_.GetTensorData<{{details.cpp_type}}>(input{{details.index}});
    static DLDataType input{{details.index}}_dtype = ::tvm::runtime::String2DLDataType("{{details.numpy_dtype}}");
    ::tvm::runtime::NDArray input{{details.index}}_ndarray = ::tvm::runtime::NDArray::Empty({{details.shape}}, input{{details.index}}_dtype, dl_device_type);
    input{{details.index}}_ndarray.CopyFromBytes(input{{details.index}}_ptr, {{details.element_count}}*sizeof({{details.cpp_type}}));
    input_vec.push_back(input{{details.index}}_ndarray);

    {% endfor %}
    {% for details in cookiecutter.outputs -%}
    int64_t output{{details.index}}_shape[] = {{details.shape}};
    OrtValue* output{{details.index}} = ort_.KernelContext_GetOutput(context, {{details.index}}, output{{details.index}}_shape, {{details.rank}});
    {{details.cpp_type}}* output{{details.index}}_ptr = ort_.GetTensorMutableData<{{details.cpp_type}}>(output{{details.index}});

    {% endfor %}
    std::vector<tvm::runtime::NDArray> outputs = TVMRun(input_vec);

    // Copy result data to ort output tensors
    {% for details in cookiecutter.outputs -%}
    outputs[{{details.index}}].CopyToBytes(output{{details.index}}_ptr, {{details.element_count}}*sizeof({{details.cpp_type}}));
    {% endfor %}
  }

  std::vector<tvm::runtime::NDArray>
  TVMRun(const std::vector<tvm::runtime::NDArray>& input_vec)
  {
    // arity is num of inputs + 1, because first argument to the set_input_func
    // is the name of the function that should take those inputs.
    size_t arity = input_vec.size() + 1;
    std::vector<TVMValue> values(arity);
    std::vector<int> codes(arity);
    tvm::runtime::TVMArgsSetter setter(values.data(), codes.data());
    setter(0, "main");
    for (size_t i = 1; i < arity; i++) {
      setter(i, input_vec.at(i - 1));
    }

    tvm::runtime::TVMRetValue rv;
    set_input_func.CallPacked(
        tvm::runtime::TVMArgs(values.data(), codes.data(), arity), &rv);

    tvm::runtime::ObjectRef out = run_func("main");
    std::vector<tvm::runtime::NDArray> outputs;
    if (out.as<tvm::runtime::ADTObj>()) {
      auto adt = tvm::Downcast<tvm::runtime::ADT>(out);
      for (size_t i = 0; i < adt.size(); ++i) {
        tvm::runtime::NDArray arr = tvm::Downcast<tvm::runtime::NDArray>(adt[i]);
        outputs.push_back(arr);
      }
    } else {
      tvm::runtime::NDArray arr = tvm::Downcast<tvm::runtime::NDArray>(out);
      outputs.push_back(arr);
    }
    return outputs;
  }

 private:
  Ort::CustomOpApi ort_;
  tvm::runtime::vm::VirtualMachine vm;
  tvm::runtime::PackedFunc set_input_func;
  tvm::runtime::PackedFunc get_output_func;
  tvm::runtime::PackedFunc run_func;
  tvm::runtime::Module exec_mod;
  tvm::runtime::ObjectPtr<tvm::runtime::vm::Executable> exec;
  TempFile model_so_file;
  bool constants_bound = false;

  DLTensor create_dltensor(
    void* data, std::vector<int64_t>& shape, const DLDevice& dev, const DLDataType& dtype
  ) {
    DLTensor tensor;
    tensor.data = data;
    tensor.device = dev;
    tensor.ndim = shape.size();
    tensor.dtype = dtype;
    tensor.shape = shape.data();
    tensor.strides = nullptr;
    tensor.byte_offset = 0;
    return tensor;
  }

  tvm::runtime::NDArray create_ndarray(
    void* data, std::vector<int64_t>& shape, const DLDevice& dev, const DLDataType& dtype
  ) {
    DLTensor tensor = create_dltensor(data, shape, dev, dtype);
    tvm::runtime::NDArray ndarray = ::tvm::runtime::NDArray::NewFromDLTensor(&tensor, dev);
    return ndarray;
  }
};

struct TVMModelOp : Ort::CustomOpBase<TVMModelOp, TVMRuntime> {
  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
    return new TVMRuntime(api);
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

} c_TVMModelOp;
} // End anonymous namespace

OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) {
  OrtCustomOpDomain* domain = nullptr;
  const OrtApi* ortApi = api->GetApi(ORT_API_VERSION);

  if (auto status = ortApi->CreateCustomOpDomain(c_OpDomain, &domain)) {
    return status;
  }

  AddOrtCustomOpDomainToContainer(domain, ortApi);

  if (auto status = ortApi->CustomOpDomain_Add(domain, &c_TVMModelOp)) {
    return status;
  }

  return ortApi->AddCustomOpDomain(options, domain);
}