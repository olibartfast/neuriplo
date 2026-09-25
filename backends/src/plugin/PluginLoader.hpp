#pragma once

// Host-side plugin machinery: discovers libneuriplo_backend_* modules (.so on
// POSIX, .dll on Windows), loads each one so that its framework dependencies
// stay private to it -- dlopen() with RTLD_NOW | RTLD_LOCAL, LoadLibraryEx()
// with LOAD_WITH_ALTERED_SEARCH_PATH -- and exposes their backends alongside
// the compiled-in registrations.

#include "InferenceInterface.hpp"
#include "neuriplo/plugin_abi.h"

#include <cstddef>
#include <iterator>
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

// A read-only, point-in-time view over the loaded plugin descriptors, taken
// under the loader's mutex. Descriptor storage itself never moves or is
// destroyed once a descriptor is added (see PluginLoader.cpp), so the
// pointers held here stay valid for the process lifetime; this snapshot only
// bounds which descriptors were visible at the moment it was taken, the same
// way a copied std::vector would. It mimics the small slice of
// std::vector<PluginBackendDescriptor>'s interface existing callers use:
// empty(), size(), front(), and a range-for over `const PluginBackendDescriptor&`.
class PluginBackendSnapshot {
  public:
    class iterator {
      public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = PluginBackendDescriptor;
        using difference_type = std::ptrdiff_t;
        using pointer = const PluginBackendDescriptor*;
        using reference = const PluginBackendDescriptor&;

        explicit iterator(std::vector<const PluginBackendDescriptor*>::const_iterator it) : it_(it) {}

        reference operator*() const { return **it_; }
        pointer operator->() const { return *it_; }
        iterator& operator++() {
            ++it_;
            return *this;
        }
        iterator operator++(int) {
            iterator tmp(*this);
            ++(*this);
            return tmp;
        }
        bool operator==(const iterator& other) const { return it_ == other.it_; }
        bool operator!=(const iterator& other) const { return it_ != other.it_; }

      private:
        std::vector<const PluginBackendDescriptor*>::const_iterator it_;
    };

    PluginBackendSnapshot() = default;
    explicit PluginBackendSnapshot(std::vector<const PluginBackendDescriptor*> descriptors)
        : descriptors_(std::move(descriptors)) {}

    bool empty() const noexcept { return descriptors_.empty(); }
    size_t size() const noexcept { return descriptors_.size(); }
    const PluginBackendDescriptor& front() const { return *descriptors_.front(); }
    iterator begin() const { return iterator(descriptors_.begin()); }
    iterator end() const { return iterator(descriptors_.end()); }

  private:
    std::vector<const PluginBackendDescriptor*> descriptors_;
};

// Loads every libneuriplo_backend_* module under `directory`. Idempotent per
// library path; incompatible or broken plugins are skipped with a logged
// reason, never a failure. Returns the number of newly loaded plugins.
size_t load_backend_plugins(const std::string& directory);

// Loads a single plugin library by path. Returns true when the plugin is
// available afterwards (already loaded counts as success).
bool load_backend_plugin(const std::string& library_path);

PluginBackendSnapshot get_plugin_backends() noexcept;

const PluginBackendDescriptor* find_plugin_backend(std::string_view id) noexcept;

// Creates an eagerly-loaded backend instance from a plugin descriptor.
// Returns nullptr on failure (the plugin's error message is logged).
std::unique_ptr<InferenceInterface> create_plugin_backend(const PluginBackendDescriptor& descriptor,
                                                          const std::string& model_path, bool use_gpu,
                                                          size_t batch_size,
                                                          const std::vector<std::vector<int64_t>>& input_sizes);
