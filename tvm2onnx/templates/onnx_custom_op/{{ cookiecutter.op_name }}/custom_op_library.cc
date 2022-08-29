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

#include <dlpack/dlpack.h>
#include <tvm/runtime/container/adt.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/vm/vm.h>

#ifdef USE_CUDA
#include <cuda_runtime.h>
template <typename T1, typename T2, typename T3>
void cuda_add(int64_t, T3*, const T1*, const T2*, cudaStream_t compute_stream);
#endif

static const char* c_OpDomain = "octoml.customop";

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
    std::remove(filename.c_str());
  }
  std::string filename;
};

struct TVMRuntime {
  TVMRuntime(const OrtApi& api)
      : ort_(api) {
    // Binary data is linked into this shared library
    // These symbols are defined by adding lines like this to the compile string
    // -Wl,--format=binary -Wl,vm_exec_code.ro -Wl,--format=default
    extern uint8_t vm_exec_code_ro_start[] asm("_binary_vm_exec_code_ro_start");
    extern uint8_t vm_exec_code_ro_end[] asm("_binary_vm_exec_code_ro_end");
    size_t vm_exec_code_ro_size = vm_exec_code_ro_end - vm_exec_code_ro_start;

    extern const char model_so_start[] asm("_binary_model_so_start");
    extern const char model_so_end[] asm("_binary_model_so_end");
    size_t model_so_size = model_so_end - model_so_start;

    std::string consts_path = "{{ cookiecutter.consts_name }}";
    DLDeviceType dl_device_type = {{ cookiecutter.dl_device_type }};

    std::string path = get_this_directory();

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
    exec->LoadLateBoundConstantsFromFile(path+consts_path);
    vm.LoadExecutable(exec);

    // Initialize the VM for the specified device. If the device is not a CPU,
    // We'll need to add a CPU context to drive it.
    int arity;
    if (dl_device_type == kDLCPU) {
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
    setter(0, (uint64_t(dl_device_type)));
    setter(1, device_id);
    setter(2, alloc_type);
    // Also initialize a CPU device context.
    if (dl_device_type != kDLCPU) {
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

  ~TVMRuntime() {
  }

  std::string get_this_directory() {
    // In order to know the paths to the above files we need to first know the path
    // of the currently running shared library. All of the other files are located
    // in that same directory.
    // This only works on linux.
    std::string custom_op_lib = "custom_{{ cookiecutter.module_name }}.so";
    std::ifstream f("/proc/self/maps");
    std::string line;
    std::smatch m;
    std::string path;
    std::regex reg("([\\S]+)"+custom_op_lib);
    for (std::string line; std::getline(f, line); ) {
      if (std::regex_search(line, m, reg)) {
        path = m[1];
        break;
      }
    }
    return path;
  }

  void LateBoundConstants(OrtKernelContext* context) {
    DLDevice dl_device_type = {DLDeviceType::{{ cookiecutter.dl_device_type }}, 0};
    std::vector<tvm::runtime::NDArray> initializers;

    {% for details in cookiecutter.initializers -%}
    const OrtValue* initializer{{details.index}} = ort_.KernelContext_GetInput(context, {{details.index}});
    const {{details.cpp_type}}* initializer{{details.index}}_ptr = ort_.GetTensorData<{{details.cpp_type}}>(initializer{{details.index}});
    std::cout << "{{details.index}}[0] = " << initializer{{details.index}}_ptr[0] << std::endl;
    int64_t initializer{{details.index}}_shape[] = {{details.shape}};
    DLDataType initializer{{details.index}}_dtype = ::tvm::runtime::String2DLDataType("{{details.numpy_dtype}}");
    ::tvm::runtime::NDArray initializer{{details.index}}_ndarray = ::tvm::runtime::NDArray::Empty({{details.shape}}, initializer{{details.index}}_dtype, dl_device_type);
    initializer{{details.index}}_ndarray.CopyFromBytes(initializer{{details.index}}_ptr, {{details.element_count}}*sizeof({{details.cpp_type}}));
    initializers.push_back(initializer{{details.index}}_ndarray);
    {% endfor %}
  }

  void Compute(OrtKernelContext* context) {
    // Get data points for the input data
    DLDevice dl_device_type = {DLDeviceType::{{ cookiecutter.dl_device_type }}, 0};

    if (!constants_bound) {
      LateBoundConstants(context);
      constants_bound = true;
    }

    std::vector<tvm::runtime::NDArray> input_vec;
    {% for details in cookiecutter.inputs -%}
    const OrtValue* input{{details.index}} = ort_.KernelContext_GetInput(context, {{details.index}});
    const {{details.cpp_type}}* input{{details.index}}_ptr = ort_.GetTensorData<{{details.cpp_type}}>(input{{details.index}});
    int64_t input{{details.index}}_shape[] = {{details.shape}};
    DLDataType input{{details.index}}_dtype = ::tvm::runtime::String2DLDataType("{{details.numpy_dtype}}");
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
};

struct TVMModelOp : Ort::CustomOpBase<TVMModelOp, TVMRuntime> {
  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
    return new TVMRuntime(api);
  };

  const char* GetName() const {
    return "{{cookiecutter.custom_op_name}}"; };

#ifdef USE_CUDA
  const char* GetExecutionProviderType() const { return "CUDAExecutionProvider"; };
#endif

  size_t GetInputTypeCount() const { return {{cookiecutter.input_count}} + {{cookiecutter.initializer_count}}; };
  ONNXTensorElementDataType GetInputType(size_t index) const {
    static std::vector<ONNXTensorElementDataType> input_types = {{cookiecutter.input_types}};
    std::cout << "get input type " << index << std::endl;
    return input_types[index];
  };

  size_t GetOutputTypeCount() const { return {{cookiecutter.output_count}}; };
  ONNXTensorElementDataType GetOutputType(size_t index) const {
    static std::vector<ONNXTensorElementDataType> output_types = {{cookiecutter.output_types}};
    return output_types[index];
  };

} c_TVMModelOp;

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