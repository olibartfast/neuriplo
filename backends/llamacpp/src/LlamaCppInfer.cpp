#include "LlamaCppInfer.hpp"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <stdexcept>

static std::atomic<int> g_backend_refcount{0};

// ---------------------------------------------------------------------------
// Constructor / destructor
// ---------------------------------------------------------------------------

LlamaCppInfer::LlamaCppInfer(const std::string& model_path, bool use_gpu, size_t batch_size,
                             const std::vector<std::vector<int64_t>>& input_sizes)
    : InferenceInterface{model_path, use_gpu, batch_size, input_sizes}, model_(nullptr), ctx_llama_(nullptr),
      vocab_(nullptr), ctx_mtmd_(nullptr), model_loaded_(false) {

    // Split "model.gguf|mmproj=/path/to/mmproj.gguf" if present
    std::string actual_model_path = model_path;
    std::string mmproj_path;
    const std::string sep = "|mmproj=";
    const auto sep_pos = model_path.find(sep);
    if (sep_pos != std::string::npos) {
        actual_model_path = model_path.substr(0, sep_pos);
        mmproj_path = model_path.substr(sep_pos + sep.size());
    }

    LOG(INFO) << "Running using llama.cpp runtime: " << actual_model_path;
    if (!mmproj_path.empty()) {
        LOG(INFO) << "Multimodal projector: " << mmproj_path;
    }

    if (g_backend_refcount.fetch_add(1) == 0) {
        llama_backend_init();
    }

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = use_gpu ? 99 : 0;
    model_ = llama_model_load_from_file(actual_model_path.c_str(), model_params);
    if (!model_) {
        if (g_backend_refcount.fetch_sub(1) == 1)
            llama_backend_free();
        throw std::runtime_error("Failed to load llama.cpp model from: " + actual_model_path);
    }

    vocab_ = llama_model_get_vocab(model_);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 8192;
    ctx_params.n_batch = 512;
    ctx_llama_ = llama_init_from_model(model_, ctx_params);
    if (!ctx_llama_) {
        llama_model_free(model_);
        if (g_backend_refcount.fetch_sub(1) == 1)
            llama_backend_free();
        throw std::runtime_error("Failed to create llama.cpp context");
    }

    // Initialise multimodal projector if path was provided
    if (!mmproj_path.empty()) {
        mtmd_context_params mparams = mtmd_context_params_default();
        mparams.use_gpu = use_gpu;
        mparams.print_timings = false;
        mparams.n_threads = 4;
        mparams.warmup = false;
        ctx_mtmd_ = mtmd_init_from_file(mmproj_path.c_str(), model_, mparams);
        if (!ctx_mtmd_) {
            llama_free(ctx_llama_);
            llama_model_free(model_);
            if (g_backend_refcount.fetch_sub(1) == 1)
                llama_backend_free();
            throw std::runtime_error("Failed to load multimodal projector from: " + mmproj_path);
        }
    }

    model_loaded_ = true;

    for (size_t i = 0; i < std::max(input_sizes.size(), static_cast<size_t>(1)); ++i) {
        std::vector<int64_t> shape = input_sizes.empty() ? std::vector<int64_t>{-1} : input_sizes[i];
        inference_metadata_.addInput("prompt" + std::to_string(i + 1), shape, batch_size);
    }
    inference_metadata_.addOutput("completion", {-1}, batch_size);

    state_ = BackendState::Ready;
}

LlamaCppInfer::~LlamaCppInfer() {
    if (ctx_mtmd_) {
        mtmd_free(ctx_mtmd_);
        ctx_mtmd_ = nullptr;
    }
    if (ctx_llama_) {
        llama_free(ctx_llama_);
        ctx_llama_ = nullptr;
    }
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
    }
    if (g_backend_refcount.fetch_sub(1) == 1) {
        llama_backend_free();
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
LlamaCppInfer::get_infer_results(const std::vector<std::vector<uint8_t>>& input_tensors) {
    if (!model_loaded_)
        throw std::runtime_error("llama.cpp model not loaded");
    if (input_tensors.empty())
        throw std::runtime_error("No input tensors provided");

    start_timer();

    try {
        const std::string raw_prompt(input_tensors[0].begin(), input_tensors[0].end());

        // Multimodal: second tensor present and mmproj loaded
        if (input_tensors.size() >= 2 && !input_tensors[1].empty() && ctx_mtmd_) {
            auto result = infer_multimodal(raw_prompt, input_tensors[1]);
            end_timer();
            return result;
        }

        auto result = infer_text_only(raw_prompt);
        end_timer();
        return result;

    } catch (const InferenceException&) {
        end_timer();
        throw;
    } catch (const std::exception& e) {
        end_timer();
        throw InferenceExecutionException(e.what());
    }
}

// ---------------------------------------------------------------------------
// Text-only inference
// ---------------------------------------------------------------------------

static constexpr int kMaxTokens = 512;

std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
LlamaCppInfer::infer_text_only(const std::string& raw_prompt) {
    const std::string prompt = apply_chat_template(raw_prompt);

    const int n_prompt_max = static_cast<int>(prompt.size()) * 4 + 64;
    std::vector<llama_token> tokens(n_prompt_max);
    const int n_tokens = llama_tokenize(vocab_, prompt.c_str(), static_cast<int>(prompt.size()), tokens.data(),
                                        n_prompt_max, true, true);
    if (n_tokens < 0)
        throw InferenceExecutionException("Failed to tokenize prompt");
    tokens.resize(n_tokens);

    llama_memory_clear(llama_get_memory(ctx_llama_), true);

    llama_batch batch = llama_batch_init(n_tokens + kMaxTokens, 0, 1);
    for (int i = 0; i < n_tokens; ++i) {
        batch.token[batch.n_tokens] = tokens[i];
        batch.pos[batch.n_tokens] = i;
        batch.n_seq_id[batch.n_tokens] = 1;
        batch.seq_id[batch.n_tokens] = &(batch.seq_id[batch.n_tokens][0]);
        batch.seq_id[batch.n_tokens][0] = 0;
        batch.logits[batch.n_tokens] = (i == n_tokens - 1);
        batch.n_tokens++;
    }

    if (llama_decode(ctx_llama_, batch) != 0) {
        llama_batch_free(batch);
        throw InferenceExecutionException("llama_decode failed during prompt evaluation");
    }

    std::string response = autoregressiveGenerate(static_cast<llama_pos>(n_tokens));

    llama_batch_free(batch);

    auto output = response_to_tensor(response);
    std::vector<int64_t> shape{static_cast<int64_t>(output.size())};
    return std::make_tuple(std::vector<std::vector<TensorElement>>{std::move(output)},
                           std::vector<std::vector<int64_t>>{std::move(shape)});
}

// ---------------------------------------------------------------------------
// Multimodal inference (text + image via mtmd)
// ---------------------------------------------------------------------------

std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
LlamaCppInfer::infer_multimodal(const std::string& raw_prompt, const std::vector<uint8_t>& image_bytes) {
    if (image_bytes.size() < 8)
        throw InferenceExecutionException("Image tensor too small");

    uint32_t nx, ny;
    std::memcpy(&nx, image_bytes.data() + 0, 4);
    std::memcpy(&ny, image_bytes.data() + 4, 4);
    const size_t expected = 8 + static_cast<size_t>(nx) * ny * 3;
    if (image_bytes.size() != expected)
        throw InferenceExecutionException("Image tensor size mismatch");

    const unsigned char* rgb_data = image_bytes.data() + 8;

    // Build the prompt using the model's own chat template if it produces the
    // correct assistant-turn prefix; otherwise fall back to the Gemma4 native
    // token strings that llama.cpp's basic template does not emit correctly.
    //
    // Strategy: apply_chat_template on the text-only question, then splice the
    // image marker in immediately before the user question text.  This keeps
    // the conversation structure intact while letting mtmd find the marker.
    const std::string marker = mtmd_default_marker();

    // Use Gemma4's native special-token strings so that llama_tokenize with
    // parse_special=true maps them to the correct token IDs.  This avoids the
    // llama_chat_apply_template Gemma3/4 mismatch for the multimodal path.
    const std::string formatted_prompt = "<bos><|turn>user\n" + marker + "\n" + raw_prompt + "<turn|>\n<|turn>model\n";

    // Build mtmd bitmap
    mtmd_bitmap* raw_bmp = mtmd_bitmap_init(nx, ny, rgb_data);
    if (!raw_bmp)
        throw InferenceExecutionException("Failed to create mtmd bitmap");
    mtmd::bitmap bmp(raw_bmp);

    // Tokenize prompt + image
    mtmd_input_chunks* raw_chunks = mtmd_input_chunks_init();
    mtmd_input_text input_text{formatted_prompt.c_str(), /*add_special=*/true, /*parse_special=*/true};
    const mtmd_bitmap* bitmaps_arr[] = {bmp.ptr.get()};
    const int32_t tok_ret = mtmd_tokenize(ctx_mtmd_, raw_chunks, &input_text, bitmaps_arr, 1);
    mtmd::input_chunks chunks(raw_chunks);
    if (tok_ret != 0)
        throw InferenceExecutionException("mtmd_tokenize failed: " + std::to_string(tok_ret));

    // Clear KV cache and evaluate all chunks
    llama_memory_clear(llama_get_memory(ctx_llama_), true);

    llama_pos n_past = 0;
    const int32_t eval_ret =
        mtmd_helper_eval_chunks(ctx_mtmd_, ctx_llama_, chunks.ptr.get(),
                                /*n_past=*/0, /*seq_id=*/0, /*n_batch=*/512, /*logits_last=*/true, &n_past);
    if (eval_ret != 0)
        throw InferenceExecutionException("mtmd_helper_eval_chunks failed");

    std::string response = autoregressiveGenerate(n_past);

    auto output = response_to_tensor(response);
    std::vector<int64_t> shape{static_cast<int64_t>(output.size())};
    return std::make_tuple(std::vector<std::vector<TensorElement>>{std::move(output)},
                           std::vector<std::vector<int64_t>>{std::move(shape)});
}

// ---------------------------------------------------------------------------
// Autoregressive generation — shared by both paths
// ---------------------------------------------------------------------------

std::string LlamaCppInfer::autoregressiveGenerate(llama_pos n_past) {
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler* smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    llama_batch gen_batch = llama_batch_init(1, 0, 1);
    std::string response;

    for (int i = 0; i < kMaxTokens; ++i) {
        const llama_token tok = llama_sampler_sample(smpl, ctx_llama_, -1);
        llama_sampler_accept(smpl, tok);

        if (llama_vocab_is_eog(vocab_, tok))
            break;

        char buf[256];
        const int n = llama_token_to_piece(vocab_, tok, buf, sizeof(buf), 0, true);
        if (n > 0)
            response.append(buf, n);

        gen_batch.n_tokens = 0;
        gen_batch.token[0] = tok;
        gen_batch.pos[0] = n_past;
        gen_batch.n_seq_id[0] = 1;
        gen_batch.seq_id[0][0] = 0;
        gen_batch.logits[0] = true;
        gen_batch.n_tokens = 1;
        n_past++;

        if (llama_decode(ctx_llama_, gen_batch) != 0) {
            llama_sampler_free(smpl);
            llama_batch_free(gen_batch);
            throw InferenceExecutionException("llama_decode failed during generation");
        }
    }

    llama_sampler_free(smpl);
    llama_batch_free(gen_batch);
    return response;
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

std::string LlamaCppInfer::bytes_to_prompt(const std::vector<uint8_t>& data) const {
    return std::string(data.begin(), data.end());
}

std::vector<TensorElement> LlamaCppInfer::response_to_tensor(const std::string& response) const {
    std::vector<TensorElement> tensor;
    tensor.reserve(response.size());
    for (const char c : response) {
        tensor.push_back(static_cast<float>(static_cast<unsigned char>(c)));
    }
    return tensor;
}

std::string LlamaCppInfer::apply_chat_template(const std::string& user_prompt) const {
    std::string tmpl_str;
    {
        int32_t len = llama_model_meta_val_str(model_, "tokenizer.chat_template", nullptr, 0);
        if (len < 0)
            len = -len;
        if (len > 0) {
            tmpl_str.resize(static_cast<size_t>(len));
            const int32_t ret =
                llama_model_meta_val_str(model_, "tokenizer.chat_template", tmpl_str.data(), tmpl_str.size() + 1);
            if (ret >= 0)
                tmpl_str.resize(static_cast<size_t>(ret));
            else
                tmpl_str.clear();
        }
    }
    if (tmpl_str.empty())
        return user_prompt;

    const char* tmpl = tmpl_str.c_str();
    const llama_chat_message message = {"user", user_prompt.c_str()};

    const int32_t tmpl_size = llama_chat_apply_template(tmpl, &message, 1, true, nullptr, 0);
    if (tmpl_size <= 0)
        return user_prompt;

    std::string formatted(tmpl_size, '\0');
    const int32_t result = llama_chat_apply_template(tmpl, &message, 1, true, formatted.data(), tmpl_size);
    if (result < 0)
        return user_prompt;

    formatted.resize(result);
    return formatted;
}
