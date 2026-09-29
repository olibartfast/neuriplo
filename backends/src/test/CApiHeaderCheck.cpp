// Build-time check ([V-1], [V-10]): neuriplo_c.h and the C++ wrapper
// neuriplo.hpp compile as strict C++17 with warnings as errors, alongside
// plugin_abi.h (no name collides). That they include no other neuriplo header
// is the include audit's job (CheckHeaderIncludes.cmake); that they compile
// from an installed prefix alone is test/consumer's ([V-12]). Owned by the
// specifier; read-only to every implementer.

#include "neuriplo/neuriplo.hpp"
#include "neuriplo/neuriplo_c.h"
#include "neuriplo/plugin_abi.h"

#include <stdexcept>
#include <type_traits>

static_assert(NEURIPLO_C_API_VERSION == 1u, "consumer C API version is 1");
static_assert(sizeof(neuriplo_status_t) == 4, "enums are 32 bits wide");
static_assert(std::is_standard_layout<neuriplo_engine_config_t>::value, "config is a C struct");
static_assert(std::is_trivial<neuriplo_tensor_view_t>::value, "views are C structs");

// The wrapper's ownership types are move-only.
static_assert(!std::is_copy_constructible<neuriplo::Engine>::value, "Engine is move-only");
static_assert(!std::is_copy_assignable<neuriplo::Engine>::value, "Engine is move-only");
static_assert(std::is_nothrow_move_constructible<neuriplo::Engine>::value, "Engine moves without throwing");
static_assert(std::is_nothrow_move_assignable<neuriplo::Engine>::value, "Engine moves without throwing");
static_assert(!std::is_copy_constructible<neuriplo::Result>::value, "Result is move-only");
static_assert(!std::is_copy_assignable<neuriplo::Result>::value, "Result is move-only");
static_assert(std::is_nothrow_move_constructible<neuriplo::Result>::value, "Result moves without throwing");
static_assert(std::is_nothrow_move_assignable<neuriplo::Result>::value, "Result moves without throwing");
static_assert(std::is_base_of<std::runtime_error, neuriplo::Error>::value, "Error is a std::runtime_error");

// Every C function is noexcept as seen from C++.
static_assert(noexcept(neuriplo_api_version()), "C API functions are noexcept in C++");
static_assert(noexcept(neuriplo_engine_create(nullptr, nullptr)), "C API functions are noexcept in C++");
static_assert(noexcept(neuriplo_infer(nullptr, nullptr, 0, nullptr)), "C API functions are noexcept in C++");

// Anchor so the object is never empty.
int neuriplo_capi_header_check_cxx();
int neuriplo_capi_header_check_cxx() { return static_cast<int>(NEURIPLO_STATUS_OK); }
