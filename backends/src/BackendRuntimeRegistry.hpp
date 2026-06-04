#pragma once

#include "IBackendRuntimeFactory.hpp"

#include <memory>

struct BackendRuntimeRegistration {
    const char* id;
    const char* display_name;
    std::unique_ptr<IBackendRuntimeFactory> (*create_factory)();
    bool force_gpu;
};

const BackendRuntimeRegistration* get_compiled_backend_registration() noexcept;
const char* compiled_backend_id() noexcept;
std::unique_ptr<IBackendRuntimeFactory> create_compiled_backend_factory();
