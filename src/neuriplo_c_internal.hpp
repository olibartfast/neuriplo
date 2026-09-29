#pragma once

// Internal helpers shared by the neuriplo C ABI implementation
// (src/neuriplo_c*.cpp). Never installed; not part of the public API and not
// on the library's public include path.

#include "neuriplo/neuriplo_c.h"

#include <exception>
#include <new>
#include <string>
#include <utility>

namespace neuriplo_capi {

// Sets the calling thread's last-error message. Must not throw: if storing
// the message fails, neuriplo_last_error() still returns a non-empty static
// string for that thread.
void set_last_error(const char* message) noexcept;

// Clears the calling thread's last-error message (sets it to "").
void clear_last_error() noexcept;

// Sets the last-error message to `message` and returns `status`.
neuriplo_status_t fail(neuriplo_status_t status, const char* message) noexcept;

// Builds "<function>: <detail>" (both null-safe: NULL is treated as an empty
// placeholder) and sets it as the last-error message, then returns `status`.
// The message is built inside a try block; if building or storing it fails
// (e.g. std::bad_alloc), the existing "message unavailable" fallback is used
// instead and `status` is still returned. Never throws.
neuriplo_status_t fail_prefixed(neuriplo_status_t status, const char* function, const char* detail) noexcept;

// Sets the last-error message to "<function>: <detail>" and returns
// NEURIPLO_STATUS_INVALID_ARGUMENT.
neuriplo_status_t invalid_argument(const char* function, const char* detail) noexcept;

// Selects which status an uncaught std::exception (other than std::bad_alloc)
// maps to inside guarded().
enum class GuardContext { Create, Infer, Other };

// Runs body() and maps any exception it throws to a neuriplo_status_t,
// setting the last-error message accordingly:
//   std::bad_alloc         -> OUT_OF_MEMORY
//   other std::exception   -> MODEL_LOAD (Create) / INFERENCE (Infer) / INTERNAL (Other)
//   anything else          -> INTERNAL
// The message is "<function>: <what()>", or "<function>: unknown exception"
// for a non-std::exception. body() must return neuriplo_status_t.
template <typename Body> neuriplo_status_t guarded(GuardContext context, const char* function, Body&& body) noexcept {
    try {
        return std::forward<Body>(body)();
    } catch (const std::bad_alloc& e) {
        return fail_prefixed(NEURIPLO_STATUS_OUT_OF_MEMORY, function, e.what());
    } catch (const std::exception& e) {
        neuriplo_status_t status = NEURIPLO_STATUS_INTERNAL;
        switch (context) {
        case GuardContext::Create:
            status = NEURIPLO_STATUS_MODEL_LOAD;
            break;
        case GuardContext::Infer:
            status = NEURIPLO_STATUS_INFERENCE;
            break;
        case GuardContext::Other:
            status = NEURIPLO_STATUS_INTERNAL;
            break;
        }
        return fail_prefixed(status, function, e.what());
    } catch (...) {
        return fail_prefixed(NEURIPLO_STATUS_INTERNAL, function, "unknown exception");
    }
}

} // namespace neuriplo_capi
