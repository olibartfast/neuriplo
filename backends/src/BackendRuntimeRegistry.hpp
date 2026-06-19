#pragma once

#include "IBackendRuntimeFactory.hpp"

#include <memory>
#include <string_view>
#include <vector>

struct BackendRuntimeRegistration {
    const char* id;
    const char* display_name;
    std::unique_ptr<IBackendRuntimeFactory> (*create_factory)();
    bool force_gpu;
};

// All backends compiled into this build (one entry per enabled USE_* definition).
const std::vector<BackendRuntimeRegistration>& get_registered_backends() noexcept;

// Registration matching `id`, or nullptr when that backend is not compiled in.
const BackendRuntimeRegistration* find_backend_registration(std::string_view id) noexcept;

// Default registration: the NEURIPLO_DEFAULT_BACKEND entry when that macro is
// defined and present, otherwise the first registered backend.
const BackendRuntimeRegistration* get_compiled_backend_registration() noexcept;
const char* compiled_backend_id() noexcept;
std::unique_ptr<IBackendRuntimeFactory> create_compiled_backend_factory();
