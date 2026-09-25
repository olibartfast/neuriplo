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
// STATUS: Group 0 skeleton (specs/2026-09-25-consumer-c-abi). The declarations
// below are the exact public API Group 4 implements -- names, signatures,
// const/noexcept qualifiers, and documented behaviour are the contract the
// acceptance suite (backends/src/test/CApiWrapperTest.cpp) compiles against.
// Group 4 replaces the bodies (and may add private helpers and members) but
// must not change a public declaration without the specifier.

#include "neuriplo_c.h"

#include <cstddef>
#include <cstdint>
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

namespace detail {
// Group 0 placeholder; Group 4 removes it.
[[noreturn]] inline void unimplemented() {
    throw Error(NEURIPLO_STATUS_UNIMPLEMENTED, "neuriplo.hpp: not implemented yet (Group 4)");
}
} // namespace detail

// Throws Error(status, <last error>) when status is not NEURIPLO_STATUS_OK.
inline void check(neuriplo_status_t status) {
    (void)status;
    detail::unimplemented();
}

// neuriplo_api_version() of the loaded library.
inline uint32_t api_version() noexcept { return 0; }

// Backend ids available in this process (neuriplo_available_backends), in the
// library's order. plugin_dir "" scans nothing extra.
inline std::vector<std::string> backends(const std::string& plugin_dir = std::string()) {
    (void)plugin_dir;
    detail::unimplemented();
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

    neuriplo_tensor_dtype_t dtype() const noexcept {
        (void)view_;
        return NEURIPLO_TENSOR_DTYPE_FLOAT32;
    }
    const void* data() const noexcept { return nullptr; }
    size_t size_bytes() const noexcept { return 0; }
    size_t element_count() const noexcept { return 0; }
    std::vector<int64_t> shape() const { detail::unimplemented(); }

    // Typed element pointer. T must match dtype(): float <-> FLOAT32,
    // int32_t <-> INT32, int64_t <-> INT64, uint8_t <-> UINT8; any other
    // combination throws Error(NEURIPLO_STATUS_INVALID_ARGUMENT). May return
    // nullptr for a zero-element tensor.
    template <typename T> const T* data_as() const { detail::unimplemented(); }

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
        std::swap(handle_, other.handle_);
        return *this;
    }
    Result(const Result&) = delete;
    Result& operator=(const Result&) = delete;
    ~Result() {}

    // Number of outputs. Throws Error (INVALID_ARGUMENT on a moved-from Result).
    size_t size() const { detail::unimplemented(); }

    // Output `index`. Throws Error(INVALID_ARGUMENT) when index >= size() or
    // on a moved-from Result.
    TensorView output(size_t index) const {
        (void)index;
        detail::unimplemented();
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
        (void)config;
        detail::unimplemented();
    }
    Engine(Engine&& other) noexcept : handle_(other.handle_) { other.handle_ = nullptr; }
    Engine& operator=(Engine&& other) noexcept {
        std::swap(handle_, other.handle_);
        return *this;
    }
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    ~Engine() {}

    // Resolved backend id. Throws Error (INVALID_ARGUMENT on a moved-from Engine).
    std::string backend_id() const { detail::unimplemented(); }

    // Model inputs / outputs, in model order. Throw Error.
    std::vector<TensorInfo> inputs() const { detail::unimplemented(); }
    std::vector<TensorInfo> outputs() const { detail::unimplemented(); }

    // neuriplo_infer. Throws Error (INVALID_ARGUMENT on a moved-from Engine).
    Result infer(const std::vector<InputView>& inputs) {
        (void)inputs;
        detail::unimplemented();
    }

    // The owned handle; nullptr after a move.
    neuriplo_engine_t* handle() const noexcept { return handle_; }

  private:
    neuriplo_engine_t* handle_ = nullptr;
};

} // namespace neuriplo
