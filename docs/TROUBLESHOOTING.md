# Troubleshooting Notes

Hard-won debugging lessons from CI and integration issues.

## Illegal instruction (SIGILL) in Docker CI

**Symptom:** `Illegal instruction (core dumped)` when running GGML backend tests in CI.

**Root cause:** `-march=native` bakes build-machine CPU features into binaries. When Docker
layer caching (`type=gha`) reuses layers across GitHub runners with different CPU generations
(e.g. AVX-512 on Ice Lake vs AVX2 on Broadwell), the cached binary crashes on the older CPU.

**Two independent sources:**
1. `cmake/SetCompilerFlags.cmake` — `set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")`
   applied to all neuriplo C++ code.
2. ggml library (`GGML_NATIVE=ON` by default) — adds `-march=native` to the ggml build in
   Dockerfile.ggml and Dockerfile.llamacpp.

**Fix:**
- Replace `-march=native` with `-march=x86-64-v3` (AVX2/FMA/BMI2 — safe on all GitHub runners).
- Guard by `CMAKE_SYSTEM_PROCESSOR`: `x86-64-v3` for x86_64, `armv8.2-a` for aarch64, nothing for other arches.
- Add `-DGGML_NATIVE=OFF` to ggml/llamacpp cmake in Dockerfiles.

**Key insight:** **Never use `-march=native` in CI/distribution builds.** Always pin to a
specific architecture level. Docker layer caching makes this especially dangerous because
the build host and runtime host may differ across CI runs.

---

## llama.cpp API break on version bump

**Symptom:** `error: cannot convert 'const llama_vocab* const' to 'const char*'` when building
llamacpp backend after bumping `LLAMACPP_VERSION`.

**Root cause:** The `llama_chat_apply_template` API changed between versions. The first argument
changed from `const llama_vocab*` (vocabulary object) to `const char*` (template string directly).

**Fix:**
1. Use `llama_model_chat_template(model, key)` to get the template string.
2. Pass the template string to `llama_chat_apply_template(tmpl, &message, 1, true, buf, len)`.
3. Remove the `nullptr` template-override argument (no longer exists).

**Key insight:** Always check the API changelog when bumping llama.cpp versions. The chat
template API was restructured significantly around b9xxx.

---

## llama_backend_init/free lifecycle

**Symptom:** Second (or later) `LlamaCppInfer` instance produces empty output, while the
first instance works fine. Occurs in test fixtures where `SetUp` creates a new instance
for each test.

**Root cause:** `llama_backend_init()` was called in every constructor and
`llama_backend_free()` in every destructor. The first test's destructor tears down the
backend (thread pools, graph allocators, etc.), and the second constructor's
re-initialization partially restores it — resulting in a broken backend that silently
produces no output.

**Fix:** Use a file-scope atomic refcount:
```cpp
static std::atomic<int> g_backend_refcount{0};
// In constructor: if (g_backend_refcount.fetch_add(1) == 0) llama_backend_init();
// In destructor:  if (g_backend_refcount.fetch_sub(1) == 1) llama_backend_free();
```

**Key insight:** **llama_backend_init/free is process-scope, not object-scope.**
Call it once per process lifetime. If you have multiple `LlamaCppInfer` objects
(e.g. in test fixtures), use a refcount.

---

## llama_model_chat_template returns NULL

**Symptom:** `llama_model_chat_template(model_, "tokenizer.chat_template")` returns NULL
even though the model metadata dump clearly shows `kv 47: tokenizer.chat_template str = ...`.

**Root cause:** Unknown — the function exists and the key exists, but it returns NULL.
May be a version-specific bug in b9049/b9085, or the `name` parameter semantics differ
from expectations (prefix vs full key).

**Fix:** Bypass `llama_model_chat_template` entirely and use `llama_model_meta_val_str`
directly with a two-pass approach (first call with nullptr to get size, second call to
fill a dynamically-sized buffer):
```cpp
int32_t len = llama_model_meta_val_str(model_, "tokenizer.chat_template", nullptr, 0);
if (len < 0) len = -len;
if (len > 0) {
    tmpl_str.resize(len);
    int32_t ret = llama_model_meta_val_str(model_, key, tmpl_str.data(), tmpl_str.size() + 1);
    ...
}
```

**Key insight:** **Don't trust convenience wrappers for critical lookups.**
When a metadata key is known to exist but the convenience function returns NULL,
fall back to the lower-level API. Also: **always use two-pass sizing** for
metadata values — templates can be 16KB+ and hardcoded buffers will truncate them.

---

## Complex Jinja2 templates exceed llama.cpp's template engine

**Symptom:** `llama_chat_apply_template` returns ≤ 0 (failure) for the Gemma-4-E2B-it
chat template, causing `apply_chat_template` to fall back to the raw user prompt.
The raw prompt then fails to elicit a response from the instruct model.

**Root cause:** The Gemma-4 chat template is 16,804 bytes of Jinja2 with macros and
conditionals. llama.cpp's built-in minimal Jinja2 implementation (`llama_chat_apply_template`)
cannot handle templates this complex. It silently returns 0 (no output size needed),
and the code falls through to raw prompt mode.

**Fix:** Silent fallback to raw prompt is correct behavior (better than crashing).
For tests, use prompts the model responds to in raw mode (e.g. "Hello", factual questions).
For production use, consider applying the template via Python/HuggingFace tokenizer
before passing to llama.cpp, or use a simpler model with a template the C engine can handle.

**Key insight:** **llama.cpp's template engine is minimal — test with your actual model.**
Complex instruct models (Gemma-4, etc.) have large Jinja2 templates that won't work with
the C-level `llama_chat_apply_template`. The silent fallback to raw prompt is correct:
it won't crash, but the model may not respond without proper formatting.

---

## Gemma4 multimodal inference returns empty response

**Symptom:** `infer_multimodal()` runs without error (image encodes in ~14s, decodes in ~18s)
but the response string is empty or `"(no response)"`. Text-only inference on the same model
works correctly.

**Root cause:** `llama_chat_apply_template` applies Gemma3's turn tokens (`<start_of_turn>`,
`<end_of_turn>`) rather than Gemma4's native token strings (`<|turn>` id=105, `<turn|>` id=106).
The malformed conversation structure causes the model to emit the EOG token immediately,
producing zero generated tokens. Since token 106 (`<turn|>`) is also Gemma4's EOG token,
the autoregressive loop exits at the very first sample.

**Fix:** Bypass `llama_chat_apply_template` entirely for the multimodal path. Construct the
formatted prompt using Gemma4's native special-token strings, which `llama_tokenize` with
`parse_special=true` correctly maps to their token IDs:

```cpp
const std::string marker = mtmd_default_marker();  // e.g. "<__media__>"
const std::string formatted_prompt =
    "<bos><|turn>user\n" + marker + "\n" + raw_prompt + "<turn|>\n<|turn>model\n";

mtmd_input_text input_text{
    formatted_prompt.c_str(),
    /*add_special=*/true,
    /*parse_special=*/true   // ← required: maps <bos>, <|turn>, <turn|> to IDs 2, 105, 106
};
```

Token mapping for Gemma4:
- `<bos>` → 2
- `<|turn>` → 105
- `<turn|>` → 106 (also the EOG token — model stops here naturally)

**Also check `n_ctx`:** Gemma4 produces ~266 image tokens. With prompt and generation
budget, `n_ctx=2048` is too small and causes silent truncation. Use at least 8192:

```cpp
ctx_params.n_ctx = 8192;
```

**Key insight:** **Each model family uses its own turn/BOS token vocabulary.** When adding
a new VLM, inspect `llama_vocab_is_eog()` results and the model's vocabulary to confirm
which token strings map to the correct IDs. Never assume `llama_chat_apply_template`
handles multimodal turn structure correctly for complex instruct models.

---

## Docker Buildx bootstrap timeout (Docker Hub flake)

**Symptom:** A CPU/lint/sanitizer Docker job fails at `Set up Docker Buildx` with:

```
ERROR: Error response from daemon: Get "https://registry-1.docker.io/v2/":
net/http: request canceled while waiting for connection
(Client.Timeout exceeded while awaiting headers)
```

The log shows Buildx bootstrapping and pulling `moby/buildkit:buildx-stable-1`.

**Root cause:** Transient Docker Hub / network flake while bootstrapping the
`docker-container` Buildx driver. Not a Dockerfile or backend bug — other jobs in the
same run (e.g. `Sanitizers / ONNX_RUNTIME`) may succeed while one matrix cell fails.

**Side effect:** `build-and-test-cpu` uses `fail-fast: true`, so one flaky Buildx failure
cancels the remaining backend matrix jobs.

**Fix:**
- CI uses `.github/actions/setup-buildx-retry`: retries pull + bootstrap up to 3 times
  with 30s backoff (same pattern as `nick-fields/retry` on `docker buildx build`).
- Re-run failed jobs once the workflow run has **finished** (`gh run rerun … --failed`).
  Rerun is rejected while the run is still `in_progress`.

**Key insight:** **Retry Buildx setup, not just the image build.** The build step already
had retry; the bootstrap pull did not.
