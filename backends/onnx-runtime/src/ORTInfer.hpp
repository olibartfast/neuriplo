#pragma once
#include "InferenceInterface.hpp"

#include <glog/logging.h>
#include <onnxruntime_c_api.h>   // for CUDA execution provider (if using CUDA)
#include <onnxruntime_cxx_api.h> // for ONNX Runtime C++ API
#include <string>
#include <vector>

// Verifies that the ONNX Runtime actually loaded implements the API version
// these headers were compiled against, and throws if it does not.
//
// This has to be a base class rather than a check at the top of ORTInfer's
// constructor body. ONNX Runtime's RAII wrappers release a handle with
// `GetApi().ReleaseX(p)`, and `GetApi()` dereferences the API table without
// checking it -- for a null handle just the same. Once a mismatched runtime has
// left that table null, destroying even a default-constructed Ort::Env or
// Ort::Session faults. A throw from the constructor body would do precisely
// that as it unwound past those members, turning a reportable version mismatch
// back into the crash this exists to prevent.
//
// Declared first, it runs before any Ort:: member is constructed, so the throw
// unwinds with nothing left to release.
struct ORTRuntimeApiGuard {
    ORTRuntimeApiGuard();
};

// Adapter: exposes the ONNX Runtime engine through the common InferenceInterface contract.
class ORTInfer : private ORTRuntimeApiGuard, public InferenceInterface {
  public:
    std::string print_shape(const std::vector<std::int64_t>& v);
    ORTInfer(const std::string& model_path, bool use_gpu = false, size_t batch_size = 1,
             const std::vector<std::vector<int64_t>>& input_sizes = std::vector<std::vector<int64_t>>());
    size_t getSizeByDim(const std::vector<int64_t>& dims);
    static std::vector<std::string> parseExecutionProviderList(const std::string& provider_list);
    static std::string providerAliasToOrtName(const std::string& provider_alias);
    static bool isProviderBuildEnabled(const std::string& provider_alias);
    std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
    get_infer_results(const std::vector<std::vector<uint8_t>>& input_tensors) override;
    std::vector<RawOutputTensor> get_infer_results_raw(const std::vector<std::vector<uint8_t>>& input_tensors) override;

  private:
    // Ort::Env's constructor defaults every parameter, so a plain `Ort::Env
    // env_;` would call CreateEnv here in the member-init list. Start null and
    // assign in the constructor body instead, so construction of the real
    // environment happens after ORTRuntimeApiGuard has run.
    Ort::Env env_{nullptr};
    Ort::Session session_{nullptr};
    std::vector<Ort::Value> run_session(const std::vector<std::vector<uint8_t>>& input_tensors);
    static std::string getDataTypeString(ONNXTensorElementDataType type);
    // Map an ONNX Runtime element type to the neuriplo TensorDataType carried in
    // InferenceMetadata so non-FP32 tensors survive the serving metadata
    // boundary. Both throw (rather than silently reporting FP32) for element
    // types the corresponding data path does not support.
    //
    // Inputs accept every type run_session() can build a tensor for; outputs are
    // limited to the element kinds get_infer_results_raw() can emit.
    static TensorDataType inputTensorDataType(ONNXTensorElementDataType type);
    static TensorDataType outputTensorDataType(ONNXTensorElementDataType type);

    template <typename T>
    void processTensorData(std::vector<TensorElement>& tensor_data, const T* data, size_t num_elements) {
        for (size_t i = 0; i < num_elements; ++i) {
            tensor_data.emplace_back(data[i]);
        }
    }
};
