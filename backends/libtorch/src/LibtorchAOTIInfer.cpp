#include "LibtorchAOTIInfer.hpp"

#include "LibtorchTensorUtils.hpp"

#include <string>

// The .pt2 package loader shipped with LibTorch 2.6. Older installs still build
// the TorchScript path; only loading a .pt2 then fails, with a clear message.
#if __has_include(<torch/csrc/inductor/aoti_package/model_package_loader.h>)
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#define NEURIPLO_LIBTORCH_HAS_AOTI_PACKAGE 1
#else
#define NEURIPLO_LIBTORCH_HAS_AOTI_PACKAGE 0
namespace torch::inductor {
class AOTIModelPackageLoader {};
} // namespace torch::inductor
#endif

namespace {

std::string shape_to_string(const std::vector<int64_t>& shape) {
    std::string s = "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        s += std::to_string(shape[i]);
        if (i + 1 < shape.size()) {
            s += ", ";
        }
    }
    return s + ")";
}

} // namespace

LibtorchAOTIInfer::~LibtorchAOTIInfer() = default;

LibtorchAOTIInfer::LibtorchAOTIInfer(const std::string& model_path, bool use_gpu, size_t batch_size,
                                     const std::vector<std::vector<int64_t>>& input_sizes)
    : InferenceInterface{model_path, use_gpu, batch_size, input_sizes} {
#if !NEURIPLO_LIBTORCH_HAS_AOTI_PACKAGE
    state_ = BackendState::Failed;
    throw ModelLoadException("AOTInductor package '" + model_path +
                             "' needs LibTorch >= 2.6 (torch::inductor::AOTIModelPackageLoader); this build of "
                             "neuriplo was compiled against an older LibTorch. Use a TorchScript (.pt) model or "
                             "rebuild against the LibTorch pinned in versions.env.");
#else
    if (input_sizes.empty()) {
        state_ = BackendState::Failed;
        throw ModelLoadException("AOTInductor packages do not expose input shapes; pass input_sizes (full shapes, "
                                 "batch included) for '" +
                                 model_path + "'");
    }

    try {
        loader_ = std::make_unique<torch::inductor::AOTIModelPackageLoader>(model_path);
    } catch (const std::exception& e) {
        LOG(ERROR) << "Failed to load the AOTInductor package: " << e.what();
        state_ = BackendState::Failed;
        throw ModelLoadException(std::string("LibTorch AOTInductor package load failed: ") + e.what());
    }

    // The package was compiled for one device type. Running it elsewhere is
    // impossible, and quietly ignoring use_gpu would hide the mismatch.
    const auto metadata = loader_->get_metadata();
    const auto device_it = metadata.find("AOTI_DEVICE_KEY");
    const std::string package_device = device_it != metadata.end() ? device_it->second : "cpu";
    if (package_device == "cuda") {
        if (!use_gpu) {
            state_ = BackendState::Failed;
            throw ModelLoadException("AOTInductor package '" + model_path +
                                     "' was compiled for CUDA but use_gpu is false; re-export it for CPU");
        }
        if (!torch::cuda::is_available()) {
            state_ = BackendState::Failed;
            throw ModelLoadException("AOTInductor package '" + model_path +
                                     "' was compiled for CUDA but CUDA is not available to LibTorch");
        }
        device_ = torch::Device(torch::kCUDA);
    } else if (package_device == "cpu") {
        if (use_gpu) {
            state_ = BackendState::Failed;
            throw ModelLoadException("AOTInductor package '" + model_path +
                                     "' was compiled for CPU but use_gpu is true; re-export it for CUDA");
        }
        device_ = torch::Device(torch::kCPU);
    } else {
        state_ = BackendState::Failed;
        throw ModelLoadException("AOTInductor package '" + model_path + "' targets unsupported device '" +
                                 package_device + "'");
    }
    LOG(INFO) << "Using " << package_device << " (AOTInductor package device)";

    LOG(INFO) << "Input Node Name/Shape:";
    std::vector<at::Tensor> probe_inputs;
    for (size_t i = 0; i < input_sizes.size(); ++i) {
        const std::string name = "input" + std::to_string(i);
        LOG(INFO) << "\t" << name << " : " << shape_to_string(input_sizes[i]);
        inference_metadata_.addInput(name, input_sizes[i], batch_size);
        probe_inputs.push_back(
            torch::zeros(input_sizes[i], torch::TensorOptions().dtype(torch::kFloat32).device(device_)));
    }

    // Output shapes are only known by running the compiled graph once. This
    // also rejects input_sizes the package was not compiled to accept.
    std::vector<at::Tensor> probe_outputs;
    try {
        probe_outputs = loader_->run(probe_inputs);
    } catch (const std::exception& e) {
        state_ = BackendState::Failed;
        throw ModelLoadException(std::string("AOTInductor probe run failed (check input_sizes): ") + e.what());
    }

    LOG(INFO) << "Output Node Name/Shape:";
    for (size_t i = 0; i < probe_outputs.size(); ++i) {
        const std::string name = "output" + std::to_string(i);
        const auto shape = probe_outputs[i].sizes().vec();
        LOG(INFO) << "\t" << name << " : " << shape_to_string(shape);
        inference_metadata_.addOutput(name, shape, batch_size);
    }

    state_ = BackendState::Ready;
#endif
}

std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
LibtorchAOTIInfer::get_infer_results(const std::vector<std::vector<uint8_t>>& input_tensors) {
#if !NEURIPLO_LIBTORCH_HAS_AOTI_PACKAGE
    (void)input_tensors;
    throw InferenceExecutionException("AOTInductor support is not compiled in");
#else
    const auto& inputs_meta = inference_metadata_.getInputs();
    if (input_tensors.size() != inputs_meta.size()) {
        throw std::runtime_error("Input tensor count mismatch. Expected " + std::to_string(inputs_meta.size()) +
                                 ", got " + std::to_string(input_tensors.size()));
    }

    std::vector<at::Tensor> torch_inputs;
    for (size_t i = 0; i < inputs_meta.size(); ++i) {
        const auto& shape = inputs_meta[i].shape;
        int64_t numel = 1;
        for (auto dim : shape) {
            numel *= dim;
        }
        const size_t expected_bytes = static_cast<size_t>(numel) * sizeof(float);
        if (input_tensors[i].size() != expected_bytes) {
            throw std::runtime_error("Input " + std::to_string(i) + " has " + std::to_string(input_tensors[i].size()) +
                                     " bytes, expected " + std::to_string(expected_bytes) + " for float32 shape " +
                                     shape_to_string(shape));
        }

        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor input = torch::from_blob(const_cast<uint8_t*>(input_tensors[i].data()), shape, options);
        torch_inputs.push_back(input.to(device_));
    }

    std::vector<at::Tensor> outputs;
    try {
        outputs = loader_->run(torch_inputs);
    } catch (const std::exception& e) {
        state_ = BackendState::Failed;
        throw InferenceExecutionException(std::string("AOTInductor run failed: ") + e.what());
    }

    std::vector<std::vector<TensorElement>> output_vectors;
    std::vector<std::vector<int64_t>> shape_vectors;
    for (const auto& output : outputs) {
        torch::Tensor tensor = output.to(torch::kCPU).contiguous();
        try {
            output_vectors.push_back(libtorch_tensor_to_elements(tensor));
        } catch (const InferenceExecutionException&) {
            state_ = BackendState::Failed;
            throw;
        }
        shape_vectors.push_back(tensor.sizes().vec());
    }

    return std::make_tuple(output_vectors, shape_vectors);
#endif
}
