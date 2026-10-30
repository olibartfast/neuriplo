#include "LibtorchRuntimeFactory.hpp"

#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
template <typename T> std::vector<uint8_t> bytes(const std::vector<T>& values) {
    std::vector<uint8_t> result(values.size() * sizeof(T));
    std::memcpy(result.data(), values.data(), result.size());
    return result;
}
} // namespace

// Component-test driver only: the expected type is an independent fixture
// oracle, not a datatype override accepted by any production backend.
int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            throw std::runtime_error("Usage: LibtorchInputContractProbe model INT32|INT64 [bad-shape]");
        }
        const auto expected = std::string(argv[2]) == "INT64" ? TensorDataType::Int64 : TensorDataType::Int32;
        std::vector<std::vector<int64_t>> shapes{{1, 3, 2, 2}, {1, 2}};
        if (argc > 3) {
            shapes[0] = {1, 3, 1, 1};
        }
        LibtorchRuntimeFactory factory;
        std::unique_ptr<InferenceInterface> engine;
        try {
            engine = factory.create_backend(argv[1], false, 1, shapes);
        } catch (const ModelLoadException&) {
            if (argc > 3) {
                return 0;
            }
            throw;
        }
        if (argc > 3) {
            throw std::runtime_error("AOTInductor accepted a shape that differs from the compiled contract");
        }
        const auto inputs = engine->get_inference_metadata().getInputs();
        if (inputs.size() != 2 || inputs[0].datatype != TensorDataType::Float32 || inputs[1].datatype != expected) {
            throw std::runtime_error("Serialized runtime input metadata differs from the exporter fixture");
        }
        auto image = bytes<float>(std::vector<float>(12, 0.25F));
        const auto sizes = expected == TensorDataType::Int64 ? bytes<int64_t>({3, 5}) : bytes<int32_t>({3, 5});
        auto [outputs, output_shapes] = engine->get_infer_results({image, sizes});
        if (outputs.size() != 1 || output_shapes[0] != shapes[0] || outputs[0].size() != 12) {
            throw std::runtime_error("Unexpected synthetic forward output shape");
        }
        for (const auto& value : outputs[0]) {
            if (std::get<float>(value) != 8.25F) {
                throw std::runtime_error("Synthetic mixed-input forward value is incorrect");
            }
        }
        image.push_back(0);
        try {
            engine->get_infer_results({image, sizes});
        } catch (const std::runtime_error&) {
            return 0;
        }
        throw std::runtime_error("Oversized input was accepted");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
