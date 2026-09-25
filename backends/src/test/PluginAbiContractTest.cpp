// Plugin ABI contract suite: the acceptance tests for
// specs/2026-09-22-plugin-abi-loader-hardening. Every case runs against the
// first-party fixtures in plugin_fixtures/, so it builds and runs in the default
// configuration with no vendor SDK installed.
//
// Owned by the specifier and read-only to every implementer (see that packet's
// orchestration.md): a worker hardening the loader does not edit the tests
// that score it.
//
// Suites map to the validation checks so `ctest -R` can select them:
//   PluginAbiLoad         load-time rejections             [V-9]
//   PluginAbiMetadata     metadata validation              [V-3]
//   PluginAbiOutput       output validation + ownership    [V-4], [V-5]
//   PluginAbiDtype        dtype validation                 [V-6]
//   PluginAbiConcurrency  descriptor access                [V-7]
//   PluginAbiIsolation    broken plugins beside good ones  [V-8]
//
// gtest_discover_tests runs each case in its own process: the loader's plugin
// table is process-global and never unloads, so no case sees another's loads.

#include "BackendRuntimeRegistry.hpp"
#include "InferenceBackendSetup.hpp"
#include "plugin/PluginLoader.hpp"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace fs = std::filesystem;

constexpr const char* kGoodId = "FIXTURE_GOOD";
constexpr const char* kScriptedId = "FIXTURE_SCRIPTED";

std::string fixture_path(const std::string& name, const char* directory = NEURIPLO_FIXTURE_PLUGIN_DIR) {
    return (fs::path(directory) / ("libneuriplo_backend_fixture_" + name + NEURIPLO_FIXTURE_PLUGIN_SUFFIX)).string();
}

// The loader reports canonical paths; compare against the same spelling.
std::string canonical(const std::string& path) { return fs::weakly_canonical(path).string(); }

std::vector<std::string> loaded_ids() {
    std::vector<std::string> ids;
    // Copy first: works whether get_plugin_backends() returns a reference or a
    // snapshot, which is the implementer's choice under [R-5].
    const auto descriptors = get_plugin_backends();
    for (const PluginBackendDescriptor& descriptor : descriptors) {
        ids.push_back(descriptor.id);
    }
    return ids;
}

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

// glog writes to stderr when it has not been initialised (and with
// logtostderr set), so capturing stderr captures the loader's diagnostics
// without depending on a particular glog version's LogSink interface.
std::string capture_log(const std::function<void()>& body) {
    FLAGS_logtostderr = true;
    FLAGS_minloglevel = 0;
    testing::internal::CaptureStderr();
    body();
    return testing::internal::GetCapturedStderr();
}

struct FixtureCounters {
    size_t handed = 0;
    size_t released = 0;
    size_t live_instances = 0;
};

// Reads the scripted fixture's own bookkeeping. The module must already be
// loaded; this takes a second reference to it rather than loading a copy.
FixtureCounters counters(const std::string& path) {
    using CountersFn = void (*)(size_t*, size_t*, size_t*);
    FixtureCounters result;
#ifdef _WIN32
    HMODULE module = GetModuleHandleA(canonical(path).c_str());
    EXPECT_NE(module, nullptr) << "fixture not loaded: " << path;
    if (module == nullptr) {
        return result;
    }
    auto fn = reinterpret_cast<CountersFn>(GetProcAddress(module, "neuriplo_fixture_counters"));
#else
    void* module = dlopen(canonical(path).c_str(), RTLD_NOW | RTLD_NOLOAD);
    EXPECT_NE(module, nullptr) << "fixture not loaded: " << path;
    if (module == nullptr) {
        return result;
    }
    auto fn = reinterpret_cast<CountersFn>(dlsym(module, "neuriplo_fixture_counters"));
#endif
    EXPECT_NE(fn, nullptr);
    if (fn != nullptr) {
        fn(&result.handed, &result.released, &result.live_instances);
    }
#ifndef _WIN32
    dlclose(module);
#endif
    return result;
}

std::vector<uint8_t> fp32_bytes(const std::vector<float>& values) {
    std::vector<uint8_t> bytes(values.size() * sizeof(float));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

std::vector<std::vector<uint8_t>> fixture_input() { return {fp32_bytes({1.0F, 2.0F, 3.0F, 4.0F})}; }

std::vector<float> as_floats(const RawOutputTensor& tensor) {
    std::vector<float> values(tensor.bytes.size() / sizeof(float));
    std::memcpy(values.data(), tensor.bytes.data(), values.size() * sizeof(float));
    return values;
}

// A scripted-fixture instance whose call-time behaviour is `mode` (the
// fixture reads it from model_path).
std::unique_ptr<InferenceInterface> create_scripted(const std::string& mode) {
    const PluginBackendDescriptor* descriptor = find_plugin_backend(kScriptedId);
    EXPECT_NE(descriptor, nullptr);
    if (descriptor == nullptr) {
        return nullptr;
    }
    return create_plugin_backend(*descriptor, mode, false, 1, {});
}

class PluginAbiTest : public testing::Test {
  protected:
    void SetUp() override {
        ASSERT_TRUE(fs::exists(fixture_path("good"))) << "fixtures not built: " << fixture_path("good");
    }

    void load_scripted() { ASSERT_TRUE(load_backend_plugin(fixture_path("scripted"))); }
};

// ---------------------------------------------------------------------------
// [V-9] Load-time rejections. Implemented before this phase ([D-3]); these
// tests prove them and pin the diagnostics.

class PluginAbiLoad : public PluginAbiTest {
  protected:
    // Loads `path`, expects rejection, and returns what the loader logged.
    std::string expect_rejected(const std::string& path) {
        const std::vector<std::string> before = loaded_ids();
        bool loaded = true;
        const std::string log = capture_log([&] { loaded = load_backend_plugin(path); });
        EXPECT_FALSE(loaded) << path;
        EXPECT_EQ(loaded_ids(), before) << "a rejected plugin must leave the descriptor list unchanged";
        EXPECT_TRUE(contains(log, canonical(path))) << "diagnostic must name the plugin path\n" << log;
        return log;
    }
};

TEST_F(PluginAbiLoad, AcceptsConformingPluginIdempotently) {
    EXPECT_TRUE(load_backend_plugin(fixture_path("good")));
    const PluginBackendDescriptor* descriptor = find_plugin_backend(kGoodId);
    ASSERT_NE(descriptor, nullptr);
    EXPECT_EQ(descriptor->library_path, canonical(fixture_path("good")));
    EXPECT_EQ(descriptor->api->abi_version, NEURIPLO_PLUGIN_ABI_VERSION);

    EXPECT_TRUE(load_backend_plugin(fixture_path("good")));
    EXPECT_EQ(loaded_ids(), std::vector<std::string>{kGoodId});
}

TEST_F(PluginAbiLoad, RejectsUnopenableLibrary) {
    const fs::path directory =
        fs::temp_directory_path() /
        ("neuriplo_abi_garbage_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(directory);
    const fs::path garbage = directory / (std::string("libneuriplo_backend_garbage") + NEURIPLO_FIXTURE_PLUGIN_SUFFIX);
    {
        std::ofstream out(garbage);
        out << "this is not a shared library\n";
    }
    const std::string log = expect_rejected(garbage.string());
    EXPECT_TRUE(contains(log, "skipping plugin")) << log;
    EXPECT_EQ(load_backend_plugins(directory.string()), 0U);
    fs::remove_all(directory);
}

TEST_F(PluginAbiLoad, RejectsMissingEntrySymbol) {
    const std::string log = expect_rejected(fixture_path("missing_entry"));
    EXPECT_TRUE(contains(log, std::string("missing ") + NEURIPLO_PLUGIN_ENTRY_SYMBOL)) << log;
}

TEST_F(PluginAbiLoad, RejectsEntryReturningNull) {
    const std::string log = expect_rejected(fixture_path("null_entry"));
    EXPECT_TRUE(contains(log, "entry point returned null")) << log;
}

TEST_F(PluginAbiLoad, RejectsAbiVersionMismatch) {
    const std::string log = expect_rejected(fixture_path("wrong_abi"));
    EXPECT_TRUE(contains(log, "ABI version")) << log;
    EXPECT_TRUE(contains(log, std::to_string(NEURIPLO_PLUGIN_ABI_VERSION + 1U))) << log;
}

TEST_F(PluginAbiLoad, RejectsIncompleteApiTable) {
    const std::string log = expect_rejected(fixture_path("incomplete"));
    EXPECT_TRUE(contains(log, "incomplete api table")) << log;
}

TEST_F(PluginAbiLoad, RejectsDuplicateBackendId) {
    ASSERT_TRUE(load_backend_plugin(fixture_path("good")));
    const std::string log = expect_rejected(fixture_path("duplicate", NEURIPLO_FIXTURE_DUPLICATE_DIR));
    EXPECT_TRUE(contains(log, "already provided by")) << log;
    EXPECT_TRUE(contains(log, canonical(fixture_path("good")))) << "must name the plugin that owns the id\n" << log;
    EXPECT_EQ(find_plugin_backend(kGoodId)->library_path, canonical(fixture_path("good")));
}

// ---------------------------------------------------------------------------
// [V-3] Metadata validation ([R-1]). A malformed layer rejects the backend:
// create_plugin_backend returns nullptr, the logged diagnostic names the
// plugin, the layer, and the field, and the plugin instance is destroyed.

class PluginAbiMetadata : public PluginAbiTest {
  protected:
    void expect_rejected(const std::string& mode, const std::vector<std::string>& tokens) {
        load_scripted();
        std::unique_ptr<InferenceInterface> backend;
        const std::string log = capture_log([&] { backend = create_scripted(mode); });
        EXPECT_EQ(backend, nullptr) << mode << " must be rejected";
        EXPECT_TRUE(contains(log, canonical(fixture_path("scripted"))))
            << mode << ": diagnostic must name the plugin path\n"
            << log;
        for (const std::string& token : tokens) {
            EXPECT_TRUE(contains(log, token)) << mode << ": diagnostic must contain '" << token << "'\n" << log;
        }
        EXPECT_EQ(counters(fixture_path("scripted")).live_instances, 0U) << "rejected instance must be destroyed";
    }
};

TEST_F(PluginAbiMetadata, ConformingMetadataIsExposed) {
    load_scripted();
    std::unique_ptr<InferenceInterface> backend = create_scripted("ok");
    ASSERT_NE(backend, nullptr);
    const InferenceMetadata metadata = backend->get_inference_metadata();
    ASSERT_EQ(metadata.getInputs().size(), 1U);
    ASSERT_EQ(metadata.getOutputs().size(), 1U);
    EXPECT_EQ(metadata.getInputs()[0].name, "input");
    EXPECT_EQ(metadata.getInputs()[0].shape, (std::vector<int64_t>{1, 4}));
    EXPECT_EQ(metadata.getInputs()[0].datatype, TensorDataType::Float32);
    EXPECT_EQ(metadata.getOutputs()[0].name, "output");
}

TEST_F(PluginAbiMetadata, QueryFailureIsRejected) { expect_rejected("metadata_fail", {"metadata query failed"}); }

TEST_F(PluginAbiMetadata, CreateFailureIsReported) {
    load_scripted();
    std::unique_ptr<InferenceInterface> backend;
    const std::string log = capture_log([&] { backend = create_scripted("create_fail"); });
    EXPECT_EQ(backend, nullptr);
    EXPECT_TRUE(contains(log, "fixture create failure")) << log;
}

TEST_F(PluginAbiMetadata, NullLayerArrayIsRejected) { expect_rejected("meta_null_inputs", {"inputs"}); }

TEST_F(PluginAbiMetadata, NullLayerNameIsRejected) { expect_rejected("meta_null_name", {"input layer 0", "name"}); }

TEST_F(PluginAbiMetadata, NullLayerShapeIsRejected) { expect_rejected("meta_null_shape", {"input layer 0", "shape"}); }

TEST_F(PluginAbiMetadata, OutOfBoundNdimIsRejected) { expect_rejected("meta_huge_ndim", {"input layer 0", "ndim"}); }

TEST_F(PluginAbiMetadata, OutputLayersAreValidatedToo) {
    expect_rejected("meta_bad_output_shape", {"output layer 0", "shape"});
}

// ---------------------------------------------------------------------------
// [V-4], [V-5] Output validation and release ownership ([R-2], [R-3]).
// [Q-3]: size_bytes must equal dtype size x product of shape, exactly.

class PluginAbiOutput : public PluginAbiTest {
  protected:
    void expect_rejected(const std::string& mode, const std::vector<std::string>& tokens) {
        load_scripted();
        std::unique_ptr<InferenceInterface> backend = create_scripted(mode);
        ASSERT_NE(backend, nullptr) << mode << ": metadata is conforming, the backend must load";
        try {
            (void)backend->get_infer_results_raw(fixture_input());
            ADD_FAILURE() << mode << ": malformed outputs must be rejected";
        } catch (const InferenceExecutionException& e) {
            const std::string message = e.what();
            EXPECT_TRUE(contains(message, kScriptedId)) << message;
            for (const std::string& token : tokens) {
                EXPECT_TRUE(contains(message, token)) << mode << ": message must contain '" << token << "'\n"
                                                      << message;
            }
        }
        const FixtureCounters after = counters(fixture_path("scripted"));
        EXPECT_EQ(after.released, after.handed)
            << mode << ": every output array infer handed over must be released exactly once";
    }
};

TEST_F(PluginAbiOutput, ConformingOutputsAreCopiedAndReleasedOnce) {
    load_scripted();
    std::unique_ptr<InferenceInterface> backend = create_scripted("ok");
    ASSERT_NE(backend, nullptr);
    for (int call = 1; call <= 3; ++call) {
        const std::vector<RawOutputTensor> outputs = backend->get_infer_results_raw(fixture_input());
        ASSERT_EQ(outputs.size(), 1U);
        EXPECT_EQ(outputs[0].dtype, TensorDtype::FP32);
        EXPECT_EQ(outputs[0].shape, (std::vector<int64_t>{1, 4}));
        EXPECT_EQ(as_floats(outputs[0]), (std::vector<float>{2.0F, 4.0F, 6.0F, 8.0F}));
        const FixtureCounters now = counters(fixture_path("scripted"));
        EXPECT_EQ(now.handed, static_cast<size_t>(call));
        EXPECT_EQ(now.released, static_cast<size_t>(call));
    }
}

TEST_F(PluginAbiOutput, InferFailureIsReported) {
    load_scripted();
    std::unique_ptr<InferenceInterface> backend = create_scripted("infer_fail");
    ASSERT_NE(backend, nullptr);
    try {
        (void)backend->get_infer_results_raw(fixture_input());
        ADD_FAILURE() << "infer failure must throw";
    } catch (const InferenceExecutionException& e) {
        EXPECT_TRUE(contains(e.what(), "fixture infer failure")) << e.what();
    }
    EXPECT_EQ(counters(fixture_path("scripted")).released, 0U) << "nothing was handed over, nothing to release";
}

TEST_F(PluginAbiOutput, NullTensorArrayIsRejected) { expect_rejected("out_null_tensors", {"tensors"}); }

TEST_F(PluginAbiOutput, NullDataIsRejected) { expect_rejected("out_null_data", {"output 0", "data"}); }

TEST_F(PluginAbiOutput, NullShapeIsRejected) { expect_rejected("out_null_shape", {"output 0", "shape"}); }

TEST_F(PluginAbiOutput, OutOfBoundNdimIsRejected) { expect_rejected("out_huge_ndim", {"output 0", "ndim"}); }

TEST_F(PluginAbiOutput, ShortBufferIsRejected) { expect_rejected("out_size_short", {"output 0", "size_bytes"}); }

TEST_F(PluginAbiOutput, OversizedBufferIsRejected) { expect_rejected("out_size_long", {"output 0", "size_bytes"}); }

TEST_F(PluginAbiOutput, NegativeDimensionIsRejected) { expect_rejected("out_negative_dim", {"output 0", "shape"}); }

// ---------------------------------------------------------------------------
// [V-6] Dtype validation ([R-4]). [Q-2]: an unknown dtype rejects the whole
// call; metadata and outputs agree on rejecting rather than each guessing.

class PluginAbiDtype : public PluginAbiTest {};

TEST_F(PluginAbiDtype, UnknownOutputDtypeRejectsCall) {
    load_scripted();
    std::unique_ptr<InferenceInterface> backend = create_scripted("out_unknown_dtype");
    ASSERT_NE(backend, nullptr);
    try {
        (void)backend->get_infer_results_raw(fixture_input());
        ADD_FAILURE() << "an unknown output dtype must reject the call";
    } catch (const InferenceExecutionException& e) {
        EXPECT_TRUE(contains(e.what(), kScriptedId)) << e.what();
        EXPECT_TRUE(contains(e.what(), "output 0")) << e.what();
        EXPECT_TRUE(contains(e.what(), "dtype")) << e.what();
    }
    const FixtureCounters after = counters(fixture_path("scripted"));
    EXPECT_EQ(after.released, after.handed);
}

TEST_F(PluginAbiDtype, LegacyViewRejectsUnknownDtype) {
    load_scripted();
    std::unique_ptr<InferenceInterface> backend = create_scripted("out_unknown_dtype");
    ASSERT_NE(backend, nullptr);
    EXPECT_THROW((void)backend->get_infer_results(fixture_input()), InferenceExecutionException)
        << "the variant view must not turn an unknown dtype into an empty element vector";
    const FixtureCounters after = counters(fixture_path("scripted"));
    EXPECT_EQ(after.released, after.handed);
}

TEST_F(PluginAbiDtype, UnknownMetadataDtypeIsRejected) {
    load_scripted();
    std::unique_ptr<InferenceInterface> backend;
    const std::string log = capture_log([&] { backend = create_scripted("meta_unknown_dtype"); });
    EXPECT_EQ(backend, nullptr) << "no silent Float32 substitution for an unknown metadata dtype";
    EXPECT_TRUE(contains(log, canonical(fixture_path("scripted")))) << log;
    EXPECT_TRUE(contains(log, "input layer 0")) << log;
    EXPECT_TRUE(contains(log, "element_type")) << log;
}

// ---------------------------------------------------------------------------
// [V-7] Descriptor access ([R-5]). Run under TSan (or ASan) for the race; the
// pointer-stability case fails deterministically without a sanitizer.

class PluginAbiConcurrency : public PluginAbiTest {};

TEST_F(PluginAbiConcurrency, DescriptorPointersSurviveLaterLoads) {
    ASSERT_TRUE(load_backend_plugin(fixture_path("good")));
    const PluginBackendDescriptor* good = find_plugin_backend(kGoodId);
    ASSERT_NE(good, nullptr);

    // Further loads grow the table; a pointer handed out earlier (callers such
    // as setup_inference_engine hold one across calls) must stay valid.
    ASSERT_TRUE(load_backend_plugin(fixture_path("scripted")));
    EXPECT_EQ(load_backend_plugins(NEURIPLO_FIXTURE_PLUGIN_DIR), 0U);

    EXPECT_EQ(find_plugin_backend(kGoodId), good) << "descriptor storage must not move once handed out";
    EXPECT_EQ(good->id, kGoodId);
    EXPECT_EQ(good->library_path, canonical(fixture_path("good")));
}

TEST_F(PluginAbiConcurrency, LoadWhileLookingUp) {
    std::atomic<bool> done{false};
    std::atomic<size_t> lookups{0};

    std::vector<std::thread> readers;
    for (int i = 0; i < 4; ++i) {
        readers.emplace_back([&] {
            while (!done.load()) {
                const PluginBackendDescriptor* found = find_plugin_backend(kScriptedId);
                if (found != nullptr) {
                    EXPECT_EQ(found->id, kScriptedId);
                }
                for (const std::string& id : loaded_ids()) {
                    EXPECT_FALSE(id.empty()) << "torn descriptor read";
                }
                lookups.fetch_add(1);
            }
        });
    }

    // Wait until the readers are demonstrably running before loading.
    while (lookups.load() < 16) {
        std::this_thread::yield();
    }
    const size_t loaded = load_backend_plugins(NEURIPLO_FIXTURE_PLUGIN_DIR);
    done.store(true);
    for (std::thread& reader : readers) {
        reader.join();
    }

    EXPECT_EQ(loaded, 2U);
    EXPECT_NE(find_plugin_backend(kGoodId), nullptr);
    EXPECT_NE(find_plugin_backend(kScriptedId), nullptr);
}

// ---------------------------------------------------------------------------
// [V-8] Isolation ([R-7]). The fixture directory holds one conforming plugin,
// one that is valid at load time but misbehaves per instance, and four that
// fail load-time checks.

class PluginAbiIsolation : public PluginAbiTest {};

TEST_F(PluginAbiIsolation, GoodPluginServesBesideBrokenOnes) {
    std::string log;
    size_t loaded = 0;
    log = capture_log([&] { loaded = load_backend_plugins(NEURIPLO_FIXTURE_PLUGIN_DIR); });
    EXPECT_EQ(loaded, 2U) << "only accepted plugins are counted\n" << log;

    std::vector<std::string> ids = loaded_ids();
    std::sort(ids.begin(), ids.end());
    EXPECT_EQ(ids, (std::vector<std::string>{kGoodId, kScriptedId}));

    // A broken instance of another plugin does not disturb the good one.
    EXPECT_EQ(create_scripted("metadata_fail"), nullptr);

    EngineOptions options;
    options.model_path = "ok";
    options.backend_id = kGoodId;
    options.plugin_dir = NEURIPLO_FIXTURE_PLUGIN_DIR;
    std::unique_ptr<InferenceInterface> good = setup_inference_engine(options);
    ASSERT_NE(good, nullptr);
    const std::vector<RawOutputTensor> outputs = good->get_infer_results_raw(fixture_input());
    ASSERT_EQ(outputs.size(), 1U);
    EXPECT_EQ(as_floats(outputs[0]), (std::vector<float>{2.0F, 4.0F, 6.0F, 8.0F}));

    const auto [elements, shapes] = good->get_infer_results(fixture_input());
    ASSERT_EQ(elements.size(), 1U);
    ASSERT_EQ(elements[0].size(), 4U);
    EXPECT_EQ(std::get<float>(elements[0][3]), 8.0F);
    EXPECT_EQ(shapes[0], (std::vector<int64_t>{1, 4}));
}

TEST_F(PluginAbiIsolation, CompiledInBackendUnaffected) {
    (void)load_backend_plugins(NEURIPLO_FIXTURE_PLUGIN_DIR);

    const BackendRuntimeRegistration* compiled = get_compiled_backend_registration();
    ASSERT_NE(compiled, nullptr);
    EXPECT_STREQ(compiled->id, NEURIPLO_DEFAULT_BACKEND);

    const std::vector<std::string> ids = available_backend_ids(NEURIPLO_FIXTURE_PLUGIN_DIR);
    EXPECT_NE(std::find(ids.begin(), ids.end(), NEURIPLO_DEFAULT_BACKEND), ids.end());
    EXPECT_NE(std::find(ids.begin(), ids.end(), kGoodId), ids.end());
    for (const char* rejected : {"FIXTURE_WRONG_ABI", "FIXTURE_NULL_ENTRY", "FIXTURE_INCOMPLETE"}) {
        EXPECT_EQ(std::find(ids.begin(), ids.end(), rejected), ids.end()) << rejected;
    }
}

} // namespace
