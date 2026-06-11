#pragma once

// Host-side plugin machinery: discovers libneuriplo_backend_*.so files,
// dlopen()s them with RTLD_NOW | RTLD_LOCAL (each plugin's framework
// dependencies stay private to it), and exposes their backends alongside the
// compiled-in registrations.

#include "InferenceInterface.hpp"
#include "neuriplo/plugin_abi.h"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

struct PluginBackendDescriptor {
    std::string id;
    std::string display_name;
    bool force_gpu = false;
    std::string library_path;
    const neuriplo_plugin_api_v1* api = nullptr;
};

// Loads every libneuriplo_backend_*.so under `directory`. Idempotent per
// library path; incompatible or broken plugins are skipped with a logged
// reason, never a failure. Returns the number of newly loaded plugins.
size_t load_backend_plugins(const std::string& directory);

// Loads a single plugin library by path. Returns true when the plugin is
// available afterwards (already loaded counts as success).
bool load_backend_plugin(const std::string& library_path);

const std::vector<PluginBackendDescriptor>& get_plugin_backends() noexcept;

const PluginBackendDescriptor* find_plugin_backend(std::string_view id) noexcept;

// Creates an eagerly-loaded backend instance from a plugin descriptor.
// Returns nullptr on failure (the plugin's error message is logged).
std::unique_ptr<InferenceInterface> create_plugin_backend(const PluginBackendDescriptor& descriptor,
                                                          const std::string& model_path, bool use_gpu,
                                                          size_t batch_size,
                                                          const std::vector<std::vector<int64_t>>& input_sizes);
