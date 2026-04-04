#include "LlamaCppInfer.hpp"

#include <cstdio>
#include <stdexcept>

LlamaCppInfer::LlamaCppInfer(const std::string& model_path, bool use_gpu, size_t batch_size,
                             const std::vector<std::vector<int64_t>>& input_sizes)
    : InferenceInterface{model_path, use_gpu, batch_size, input_sizes}, model_(nullptr), ctx_llama_(nullptr),
      vocab_(nullptr), model_loaded_(false) {
    LOG(INFO) << "Running using llama.cpp runtime: " << model_path;

    // Initialise llama.cpp backend (once per process)
    llama_backend_init();

    // Load model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = use_gpu ? 99 : 0;
    model_ = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model_) {
        llama_backend_free();
        throw std::runtime_error("Failed to load llama.cpp model from: " + model_path);
    }

    vocab_ = llama_model_get_vocab(model_);

    // Create context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    ctx_params.n_batch = 512;
    ctx_llama_ = llama_init_from_model(model_, ctx_params);
    if (!ctx_llama_) {
        llama_model_free(model_);
        llama_backend_free();
        throw std::runtime_error("Failed to create llama.cpp context");
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

LlamaCppInfer::~LlamaCppInfer() {
    if (ctx_llama_) {
        llama_free(ctx_llama_);
        ctx_llama_ = nullptr;
    }
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
    }
    llama_backend_free();
}

static constexpr int kMaxTokens = 512;

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
LlamaCppInfer::get_infer_results(const std::vector<std::vector<uint8_t>>& input_tensors) {
    if (!model_loaded_) {
        throw std::runtime_error("llama.cpp model not loaded");
    }
    if (input_tensors.empty()) {
        throw std::runtime_error("No input tensors provided");
    }

    start_timer();

    std::vector<std::vector<TensorElement>> all_outputs;
    std::vector<std::vector<int64_t>> all_shapes;

    try {
        for (const auto& raw_input : input_tensors) {
            const std::string prompt(raw_input.begin(), raw_input.end());

            // Tokenize the prompt
            const int n_prompt_max = static_cast<int>(prompt.size()) + 16;
            std::vector<llama_token> tokens(n_prompt_max);
            const int n_tokens = llama_tokenize(vocab_, prompt.c_str(), static_cast<int>(prompt.size()), tokens.data(),
                                                n_prompt_max, true, true);
            if (n_tokens < 0) {
                throw InferenceExecutionException("Failed to tokenize prompt");
            }
            tokens.resize(n_tokens);

            // Clear KV cache for a fresh generation
            llama_kv_cache_clear(ctx_llama_);

            // Create a sampler chain with greedy sampling
            auto sparams = llama_sampler_chain_default_params();
            llama_sampler* smpl = llama_sampler_chain_init(sparams);
            llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

            // Create a batch and evaluate prompt tokens
            llama_batch batch = llama_batch_init(n_tokens + kMaxTokens, 0, 1);

            for (int i = 0; i < n_tokens; ++i) {
                batch.token[batch.n_tokens] = tokens[i];
                batch.pos[batch.n_tokens] = i;
                batch.n_seq_id[batch.n_tokens] = 1;
                // batch.seq_id[i] is pre-allocated by llama_batch_init with n_seq_max=1
                batch.seq_id[batch.n_tokens][0] = 0;
                batch.logits[batch.n_tokens] = (i == n_tokens - 1);
                batch.n_tokens++;
            }

            if (llama_decode(ctx_llama_, batch) != 0) {
                llama_sampler_free(smpl);
                llama_batch_free(batch);
                throw InferenceExecutionException("llama_decode failed during prompt evaluation");
            }

            // Auto-regressive generation loop
            std::string response;
            int n_cur = n_tokens;

            for (int i = 0; i < kMaxTokens; ++i) {
                // Sample the next token
                const llama_token new_token_id = llama_sampler_sample(smpl, ctx_llama_, -1);
                llama_sampler_accept(smpl, new_token_id);

                // Check for end of generation
                if (llama_vocab_is_eog(vocab_, new_token_id)) {
                    break;
                }

                // Convert token to text
                char buf[256];
                const int n = llama_token_to_piece(vocab_, new_token_id, buf, sizeof(buf), 0, true);
                if (n > 0 && n <= static_cast<int>(sizeof(buf))) {
                    response.append(buf, n);
                }

                // Prepare next batch
                batch.n_tokens = 0;
                batch.token[batch.n_tokens] = new_token_id;
                batch.pos[batch.n_tokens] = n_cur;
                batch.n_seq_id[batch.n_tokens] = 1;
                batch.seq_id[batch.n_tokens][0] = 0;
                batch.logits[batch.n_tokens] = true;
                batch.n_tokens++;
                n_cur++;

                if (llama_decode(ctx_llama_, batch) != 0) {
                    llama_sampler_free(smpl);
                    llama_batch_free(batch);
                    throw InferenceExecutionException("llama_decode failed during generation");
                }
            }

            llama_sampler_free(smpl);
            llama_batch_free(batch);

            std::vector<TensorElement> output = response_to_tensor(response);
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

std::string LlamaCppInfer::bytes_to_prompt(const std::vector<uint8_t>& data) const {
    return std::string(data.begin(), data.end());
}

std::vector<TensorElement> LlamaCppInfer::response_to_tensor(const std::string& response) const {
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
