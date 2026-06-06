#pragma once
#include "BackendDecorator.hpp"
#include "InferenceInterface.hpp"

#include <cstdint>
#include <memory>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

// ==========================================================================
// QuantizedBackend  --  OPT-IN ONLY.  REQUIRES HUMAN REVIEWER SIGN-OFF.
// --------------------------------------------------------------------------
// This decorator can rewrite the *numeric* inference output (dequantizing
// integer tensor elements into floats), which places it RIGHT ON THE EDGE of
// the `inference-logic-change` class that REPO_META.yaml forbids.
//
// For that reason it is built to be SAFE BY DEFAULT:
//   * QuantizationParams::enabled defaults to FALSE.
//   * When disabled (the default and ONLY safe path) get_infer_results() is an
//     EXACT PASSTHROUGH: the wrapped backend's output is returned byte-for-byte
//     unchanged. No numeric transformation occurs.
//   * The dequantization path is for demonstration / explicit opt-in only.
//
// DO NOT wrap any backend with this decorator in a default decorator chain,
// and DO NOT enable quantization on any default/production path, without
// explicit human reviewer sign-off. Enabling it changes inference numerics.
// ==========================================================================
// Configuration for the (opt-in) dequantization transform.
// enabled == false means exact passthrough (the safe default).
struct QuantizationParams {
    bool enabled = false;
    float scale = 1.0f;
    int32_t zero_point = 0;
};

class QuantizedBackend : public BackendDecorator {

  public:
    explicit QuantizedBackend(std::unique_ptr<InferenceInterface> inner, QuantizationParams params = {})
        : BackendDecorator(std::move(inner)), params_(params) {}

    std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
    get_infer_results(const std::vector<std::vector<uint8_t>>& input_tensors) override {
        // Default and only safe path: exact passthrough, no numeric change.
        if (!params_.enabled) {
            return BackendDecorator::get_infer_results(input_tensors);
        }

        // Opt-in path (requires reviewer sign-off): dequantize integer-typed
        // elements to float using real = scale * (q - zero_point). Float
        // elements are left untouched and shapes are preserved.
        auto result = BackendDecorator::get_infer_results(input_tensors);
        auto& tensors = std::get<0>(result);
        for (auto& tensor : tensors) {
            for (auto& element : tensor) {
                element = dequantize(element);
            }
        }
        return result;
    }

    void set_quantization(QuantizationParams p) { params_ = p; }
    QuantizationParams quantization() const { return params_; }

  private:
    // Converts a single quantized integer element to a dequantized float.
    // Float elements pass through unchanged.
    TensorElement dequantize(const TensorElement& element) const {
        return std::visit(
            [this](auto value) -> TensorElement {
                using T = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<T, float>) {
                    return value;
                } else {
                    const auto q = static_cast<float>(value);
                    return params_.scale * (q - static_cast<float>(params_.zero_point));
                }
            },
            element);
    }

    QuantizationParams params_;
};
