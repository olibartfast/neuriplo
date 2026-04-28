#include "CactusInfer.hpp"

#include <cstdio>
#include <stdexcept>

CactusInfer::CactusInfer(const std::string& model_path, bool use_gpu, size_t batch_size,
                         const std::vector<std::vector<int64_t>>& input_sizes)
    : InferenceInterface{model_path, use_gpu, batch_size, input_sizes}, model_(nullptr), model_loaded_(false),
      vocab_size_(0) {
    LOG(INFO) << "Running using Cactus runtime: " << model_path;

    // cactus_init expects the folder that contains the model weights and an
    // optional RAG text path.  Pass an empty string for the RAG path so the
    // engine performs straightforward completion only.
    model_ = cactus_init(model_path.c_str(), "", static_cast<int>(use_gpu));
    if (!model_) {
        throw std::runtime_error("Failed to initialise Cactus model from: " + model_path);
    }

    model_loaded_ = true;

    // Register metadata so callers can inspect inputs/outputs.
    // Input: variable-length byte sequence (encoded prompt).
    // Output: completion tokens encoded as float values (one element per byte).
    for (size_t i = 0; i < std::max(input_sizes.size(), static_cast<size_t>(1)); ++i) {
        std::vector<int64_t> shape = input_sizes.empty() ? std::vector<int64_t>{-1} : input_sizes[i];
        inference_metadata_.addInput("prompt" + std::to_string(i + 1), shape, batch_size);
    }

    inference_metadata_.addOutput("completion", {-1}, batch_size);
}

CactusInfer::~CactusInfer() {
    if (model_) {
        cactus_free(model_);
        model_ = nullptr;
    }
}

static constexpr size_t kDefaultResponseBufferSize = 65536; // 64 KiB

// ---------------------------------------------------------------------------
// get_infer_results
//
// Input tensors: each element is a UTF-8 encoded prompt (uint8_t bytes).
// Output tensors: the completion text re-encoded as float values where each
//   element is the float-cast value of the corresponding output byte.  This
//   preserves compatibility with InferenceInterface while conveying the full
//   text response to the caller.
// ---------------------------------------------------------------------------
std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
CactusInfer::get_infer_results(const std::vector<std::vector<uint8_t>>& input_tensors) {
    if (!model_loaded_) {
        throw std::runtime_error("Cactus model not loaded");
    }
    if (input_tensors.empty()) {
        throw std::runtime_error("No input tensors provided");
    }

    start_timer();

    std::vector<std::vector<TensorElement>> all_outputs;
    std::vector<std::vector<int64_t>> all_shapes;

    try {
        for (const auto& raw_input : input_tensors) {
            const std::string prompt = bytes_to_prompt(raw_input);

            // Build a minimal single-turn chat message compatible with the
            // Cactus Engine API.
            std::string messages = R"([{"role":"user","content":")" + prompt + R"("}])";

            std::vector<char> response_buf(kDefaultResponseBufferSize, '\0');
            const int rc = cactus_complete(model_, messages.c_str(), response_buf.data(),
                                           static_cast<int>(response_buf.size()), "{}", nullptr, nullptr, nullptr);
            if (rc != 0) {
                throw InferenceExecutionException("cactus_complete returned error code " + std::to_string(rc));
            }

            std::vector<TensorElement> output = response_to_tensor(std::string(response_buf.data()));
            std::vector<int64_t> shape{static_cast<int64_t>(output.size())};

            all_outputs.push_back(std::move(output));
            all_shapes.push_back(std::move(shape));
        }
    } catch (const InferenceException&) {
        end_timer();
        throw;
    } catch (const std::exception& e) {
        end_timer();
        throw InferenceExecutionException(e.what());
    }

    end_timer();
    return std::make_tuple(all_outputs, all_shapes);
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

std::string CactusInfer::bytes_to_prompt(const std::vector<uint8_t>& data) const {
    // Interpret the raw bytes as a UTF-8 string and produce a JSON-safe
    // representation suitable for embedding inside a JSON string value.
    std::string result;
    result.reserve(data.size() * 2);
    for (const uint8_t byte : data) {
        switch (byte) {
        case '"':
            result += "\\\"";
            break;
        case '\\':
            result += "\\\\";
            break;
        case '\n':
            result += "\\n";
            break;
        case '\r':
            result += "\\r";
            break;
        case '\t':
            result += "\\t";
            break;
        case '\b':
            result += "\\b";
            break;
        case '\f':
            result += "\\f";
            break;
        default:
            if (byte < 0x20) {
                // Escape other control characters as \uXXXX
                char buf[7];
                std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(byte));
                result += buf;
            } else {
                result += static_cast<char>(byte);
            }
            break;
        }
    }
    return result;
}

std::vector<TensorElement> CactusInfer::response_to_tensor(const std::string& response) const {
    // Encode the response as a float vector: each element is the float-cast
    // value of the corresponding byte.  This is a simple lossless encoding
    // that fits the TensorElement interface.
    std::vector<TensorElement> tensor;
    tensor.reserve(response.size());
    for (const char c : response) {
        tensor.push_back(static_cast<float>(static_cast<unsigned char>(c)));
    }
    return tensor;
}
