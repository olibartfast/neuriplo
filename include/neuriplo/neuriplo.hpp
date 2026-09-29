#pragma once
// Header-only C++ wrapper over the neuriplo consumer C ABI (neuriplo_c.h).
//
// Everything here compiles into the application: the only thing that crosses
// the library boundary is the C ABI, so an application built with a different
// compiler, standard library, or MSVC runtime than libneuriplo can use it.
// Errors are thrown as neuriplo::Error on the application's side of the
// boundary; libneuriplo itself never throws across it.
//
// Rule ([R-8], [V-10]): this header includes no neuriplo header other than
// neuriplo_c.h, and only standard C++ headers otherwise.
//
// The public declarations below are the contract the acceptance suite
// (backends/src/test/CApiWrapperTest.cpp) compiles against; changing one
// needs the specifier (specs/2026-09-25-consumer-c-abi).

#include "neuriplo_c.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace neuriplo {

// A failed C API call, thrown by every wrapper function that can fail.
// status() is the C status; what() is neuriplo_last_error() captured on the
// failing thread immediately after the failing call (never empty: when the
// library gave no message, neuriplo_status_string(status) is used).
class Error : public std::runtime_error {
  public:
    Error(neuriplo_status_t status, const std::string& message) : std::runtime_error(message), status_(status) {}

    neuriplo_status_t status() const noexcept { return status_; }

  private:
    neuriplo_status_t status_;
};

// Throws Error(status, <last error>) when status is not NEURIPLO_STATUS_OK.
inline void check(neuriplo_status_t status) {
    if (status == NEURIPLO_STATUS_OK) {
        return;
    }
    const char* message = neuriplo_last_error();
    if (message == nullptr || message[0] == '\0') {
        message = neuriplo_status_string(status);
    }
    throw Error(status, message);
}

// neuriplo_api_version() of the loaded library.
inline uint32_t api_version() noexcept { return neuriplo_api_version(); }

namespace detail {

// Releases a neuriplo_backend_list_t on every path out of backends().
class BackendListHolder {
  public:
    BackendListHolder() = default;
    BackendListHolder(const BackendListHolder&) = delete;
    BackendListHolder& operator=(const BackendListHolder&) = delete;
    ~BackendListHolder() { neuriplo_backend_list_release(list_); }

    neuriplo_backend_list_t* list_ = nullptr;
};

template <typename T> struct dependent_false {
    static constexpr bool value = false;
};

template <typename T> struct dtype_of {
    static_assert(dependent_false<T>::value, "data_as<T>: T must be float, int32_t, int64_t, or uint8_t");
};
template <> struct dtype_of<float> {
    static constexpr neuriplo_tensor_dtype_t value = NEURIPLO_TENSOR_DTYPE_FLOAT32;
};
template <> struct dtype_of<int32_t> {
    static constexpr neuriplo_tensor_dtype_t value = NEURIPLO_TENSOR_DTYPE_INT32;
};
template <> struct dtype_of<int64_t> {
    static constexpr neuriplo_tensor_dtype_t value = NEURIPLO_TENSOR_DTYPE_INT64;
};
template <> struct dtype_of<uint8_t> {
    static constexpr neuriplo_tensor_dtype_t value = NEURIPLO_TENSOR_DTYPE_UINT8;
};

} // namespace detail

// Backend ids available in this process (neuriplo_available_backends), in the
// library's order. plugin_dir "" scans nothing extra.
inline std::vector<std::string> backends(const std::string& plugin_dir = std::string()) {
    detail::BackendListHolder holder;
    check(neuriplo_available_backends(plugin_dir.c_str(), &holder.list_));
    size_t count = 0;
    check(neuriplo_backend_list_count(holder.list_, &count));
    std::vector<std::string> ids;
    ids.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        const char* id = nullptr;
        check(neuriplo_backend_list_get(holder.list_, i, &id));
        ids.emplace_back(id);
    }
    return ids;
}

// Owning mirror of neuriplo_engine_config_t. Empty strings mean "default"
// exactly as NULL/"" do in C; batch_size 0 means 1.
struct EngineConfig {
    std::string backend_id;
    std::string model_path;
    bool use_gpu = false;
    size_t batch_size = 1;
    std::vector<std::vector<int64_t>> input_sizes;
    std::string plugin_dir;
};

// Owning copy of one neuriplo_tensor_info_t.
struct TensorInfo {
    std::string name;
    neuriplo_tensor_dtype_t dtype = NEURIPLO_TENSOR_DTYPE_FLOAT32;
    std::vector<int64_t> shape;
    size_t batch_size = 0;
};

// Non-owning view of one input tensor's raw bytes; the bytes must outlive the
// Engine::infer call it is passed to. Implicitly constructible from a
// std::vector of any element type, so `engine.infer({values})` works.
class InputView {
  public:
    InputView(const void* data, size_t size_bytes) noexcept : data_(data), size_bytes_(size_bytes) {}

    template <typename T>
    InputView(const std::vector<T>& values) noexcept // NOLINT(google-explicit-constructor): intended
        : data_(values.data()), size_bytes_(values.size() * sizeof(T)) {}

    const void* data() const noexcept { return data_; }
    size_t size_bytes() const noexcept { return size_bytes_; }

  private:
    const void* data_;
    size_t size_bytes_;
};

// Non-owning view of one output tensor, valid as long as the Result it came
// from (it wraps a neuriplo_tensor_view_t owned by that result).
class TensorView {
  public:
    explicit TensorView(const neuriplo_tensor_view_t* view) noexcept : view_(view) {}

    neuriplo_tensor_dtype_t dtype() const noexcept { return view_->dtype; }
    const void* data() const noexcept { return view_->data; }
    size_t size_bytes() const noexcept { return view_->size_bytes; }
    size_t element_count() const noexcept { return view_->element_count; }
    std::vector<int64_t> shape() const {
        if (view_->ndim == 0) {
            return {};
        }
        return std::vector<int64_t>(view_->shape, view_->shape + view_->ndim);
    }

    // Typed element pointer. T must match dtype(): float <-> FLOAT32,
    // int32_t <-> INT32, int64_t <-> INT64, uint8_t <-> UINT8; any other
    // combination throws Error(NEURIPLO_STATUS_INVALID_ARGUMENT). May return
    // nullptr for a zero-element tensor.
    template <typename T> const T* data_as() const {
        if (dtype() != detail::dtype_of<T>::value) {
            throw Error(NEURIPLO_STATUS_INVALID_ARGUMENT, "neuriplo::TensorView::data_as: T does not match dtype()");
        }
        return static_cast<const T*>(data());
    }

  private:
    const neuriplo_tensor_view_t* view_;
};

// Move-only owner of one neuriplo_result_t; releases it exactly once.
class Result {
  public:
    // Takes ownership of `owned` (may be nullptr).
    explicit Result(neuriplo_result_t* owned) noexcept : handle_(owned) {}
    Result(Result&& other) noexcept : handle_(other.handle_) { other.handle_ = nullptr; }
    Result& operator=(Result&& other) noexcept {
        if (this != &other) {
            neuriplo_result_release(handle_);
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }
    Result(const Result&) = delete;
    Result& operator=(const Result&) = delete;
    ~Result() { neuriplo_result_release(handle_); }

    // Number of outputs. Throws Error (INVALID_ARGUMENT on a moved-from Result).
    size_t size() const {
        size_t count = 0;
        check(neuriplo_result_output_count(handle_, &count));
        return count;
    }

    // Output `index`. Throws Error(INVALID_ARGUMENT) when index >= size() or
    // on a moved-from Result.
    TensorView output(size_t index) const {
        const neuriplo_tensor_view_t* view = nullptr;
        check(neuriplo_result_output(handle_, index, &view));
        return TensorView(view);
    }

    // The owned handle; nullptr after a move.
    neuriplo_result_t* handle() const noexcept { return handle_; }

  private:
    neuriplo_result_t* handle_ = nullptr;
};

// Move-only owner of one neuriplo_engine_t; destroys it exactly once.
// Thread-safety is the C API's: infer() on one Engine from several threads is
// safe (serialised by the library); moving or destroying it is not.
class Engine {
  public:
    // neuriplo_engine_create. Throws Error on any failure.
    explicit Engine(const EngineConfig& config) {
        std::vector<neuriplo_dims_t> dims;
        dims.reserve(config.input_sizes.size());
        for (const std::vector<int64_t>& shape : config.input_sizes) {
            dims.push_back(neuriplo_dims_t{shape.empty() ? nullptr : shape.data(), shape.size()});
        }
        neuriplo_engine_config_t c_config;
        std::memset(&c_config, 0, sizeof(c_config));
        c_config.struct_size = static_cast<uint32_t>(sizeof(c_config));
        c_config.use_gpu = config.use_gpu ? 1 : 0;
        c_config.backend_id = config.backend_id.c_str();
        c_config.model_path = config.model_path.c_str();
        c_config.batch_size = config.batch_size;
        c_config.input_sizes = dims.empty() ? nullptr : dims.data();
        c_config.n_input_sizes = dims.size();
        c_config.plugin_dir = config.plugin_dir.c_str();
        check(neuriplo_engine_create(&c_config, &handle_));
    }
    Engine(Engine&& other) noexcept : handle_(other.handle_) { other.handle_ = nullptr; }
    Engine& operator=(Engine&& other) noexcept {
        if (this != &other) {
            neuriplo_engine_destroy(handle_);
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    ~Engine() { neuriplo_engine_destroy(handle_); }

    // Resolved backend id. Throws Error (INVALID_ARGUMENT on a moved-from Engine).
    std::string backend_id() const {
        const char* id = nullptr;
        check(neuriplo_engine_backend_id(handle_, &id));
        return id;
    }

    // Model inputs / outputs, in model order. Throw Error.
    std::vector<TensorInfo> inputs() const { return infos(neuriplo_engine_input_count, neuriplo_engine_input); }
    std::vector<TensorInfo> outputs() const { return infos(neuriplo_engine_output_count, neuriplo_engine_output); }

    // neuriplo_infer. Throws Error (INVALID_ARGUMENT on a moved-from Engine).
    Result infer(const std::vector<InputView>& inputs) {
        std::vector<neuriplo_input_view_t> views;
        views.reserve(inputs.size());
        for (const InputView& input : inputs) {
            views.push_back(neuriplo_input_view_t{input.data(), input.size_bytes()});
        }
        neuriplo_result_t* owned = nullptr;
        check(neuriplo_infer(handle_, views.empty() ? nullptr : views.data(), views.size(), &owned));
        return Result(owned);
    }

    // The owned handle; nullptr after a move.
    neuriplo_engine_t* handle() const noexcept { return handle_; }

  private:
    using CountFn = neuriplo_status_t(NEURIPLO_CALL*)(const neuriplo_engine_t*, size_t*) NEURIPLO_NOEXCEPT;
    using InfoFn = neuriplo_status_t(NEURIPLO_CALL*)(const neuriplo_engine_t*, size_t,
                                                     const neuriplo_tensor_info_t**) NEURIPLO_NOEXCEPT;

    std::vector<TensorInfo> infos(CountFn count_fn, InfoFn info_fn) const {
        size_t count = 0;
        check(count_fn(handle_, &count));
        std::vector<TensorInfo> result;
        result.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            const neuriplo_tensor_info_t* info = nullptr;
            check(info_fn(handle_, i, &info));
            TensorInfo copy;
            copy.name = info->name;
            copy.dtype = info->dtype;
            if (info->ndim > 0) {
                copy.shape.assign(info->shape, info->shape + info->ndim);
            }
            copy.batch_size = info->batch_size;
            result.push_back(std::move(copy));
        }
        return result;
    }

    neuriplo_engine_t* handle_ = nullptr;
};

} // namespace neuriplo
