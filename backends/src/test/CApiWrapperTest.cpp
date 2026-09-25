// C++ wrapper suite ([V-10]) for specs/2026-09-25-consumer-c-abi: exercises
// include/neuriplo/neuriplo.hpp exactly as a third-party C++ application
// would. The only neuriplo header it includes is neuriplo.hpp (which itself
// may include only neuriplo_c.h); everything runs against the first-party
// fixture plugins.
//
// Owned by the specifier and read-only to every implementer (see that
// packet's orchestration.md).
//
// gtest_discover_tests runs each case in its own process (ctest names
// `CApiWrapper.<Case>`): the plugin table is process-global.

#include "neuriplo/neuriplo.hpp"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <gtest/gtest.h>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

constexpr const char* kGoodId = "FIXTURE_GOOD";
constexpr const char* kScriptedId = "FIXTURE_SCRIPTED";

neuriplo::EngineConfig fixture_config(const std::string& backend_id, const std::string& model_path) {
    neuriplo::EngineConfig config;
    config.backend_id = backend_id;
    config.model_path = model_path;
    config.plugin_dir = NEURIPLO_FIXTURE_PLUGIN_DIR;
    return config;
}

bool contains(const std::vector<std::string>& ids, const std::string& id) {
    return std::find(ids.begin(), ids.end(), id) != ids.end();
}

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

struct FixtureCounts {
    size_t handed = 0;
    size_t released = 0;
    size_t live = 0;
};

// The fixture module's own bookkeeping (test-only export); the module must
// already be loaded by the library.
FixtureCounts fixture_counts(const std::string& module) {
    using CountersFn = void (*)(size_t*, size_t*, size_t*);
    FixtureCounts counts;
    const std::string path =
        std::filesystem::weakly_canonical(std::filesystem::path(NEURIPLO_FIXTURE_PLUGIN_DIR) /
                                          ("libneuriplo_backend_fixture_" + module + NEURIPLO_FIXTURE_PLUGIN_SUFFIX))
            .string();
    CountersFn fn = nullptr;
#ifdef _WIN32
    HMODULE handle = GetModuleHandleA(path.c_str());
    EXPECT_NE(handle, nullptr) << "fixture not loaded: " << path;
    if (handle == nullptr) {
        return counts;
    }
    fn = reinterpret_cast<CountersFn>(GetProcAddress(handle, "neuriplo_fixture_counters"));
#else
    void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_NOLOAD);
    EXPECT_NE(handle, nullptr) << "fixture not loaded: " << path;
    if (handle == nullptr) {
        return counts;
    }
    fn = reinterpret_cast<CountersFn>(dlsym(handle, "neuriplo_fixture_counters"));
    dlclose(handle);
#endif
    EXPECT_NE(fn, nullptr);
    if (fn != nullptr) {
        fn(&counts.handed, &counts.released, &counts.live);
    }
    return counts;
}

// Runs `body`, expecting a neuriplo::Error with `status`; returns what().
template <typename Body> std::string expect_error(neuriplo_status_t status, Body&& body) {
    try {
        body();
    } catch (const neuriplo::Error& error) {
        EXPECT_EQ(error.status(), status) << error.what();
        EXPECT_NE(std::string(error.what()), "");
        return error.what();
    } catch (const std::exception& other) {
        ADD_FAILURE() << "expected neuriplo::Error, got another exception: " << other.what();
        return {};
    }
    ADD_FAILURE() << "expected neuriplo::Error with status " << static_cast<int>(status) << ", nothing was thrown";
    return {};
}

void expect_doubled(const neuriplo::Result& result, const std::vector<float>& in) {
    ASSERT_EQ(result.size(), 1u);
    const neuriplo::TensorView view = result.output(0);
    EXPECT_EQ(view.dtype(), NEURIPLO_TENSOR_DTYPE_FLOAT32);
    EXPECT_EQ(view.shape(), (std::vector<int64_t>{1, static_cast<int64_t>(in.size())}));
    ASSERT_EQ(view.element_count(), in.size());
    EXPECT_EQ(view.size_bytes(), in.size() * sizeof(float));
    const float* out = view.data_as<float>();
    ASSERT_NE(out, nullptr);
    for (size_t i = 0; i < in.size(); ++i) {
        EXPECT_EQ(out[i], 2.0f * in[i]) << "element " << i;
    }
}

static_assert(!std::is_copy_constructible<neuriplo::Engine>::value, "Engine is move-only");
static_assert(!std::is_copy_constructible<neuriplo::Result>::value, "Result is move-only");
static_assert(std::is_nothrow_move_constructible<neuriplo::Engine>::value, "Engine moves without throwing");
static_assert(std::is_nothrow_move_constructible<neuriplo::Result>::value, "Result moves without throwing");

} // namespace

TEST(CApiWrapper, ApiVersionMatchesHeader) { EXPECT_EQ(neuriplo::api_version(), NEURIPLO_C_API_VERSION); }

TEST(CApiWrapper, BackendsListsFixtures) {
    const std::vector<std::string> ids = neuriplo::backends(NEURIPLO_FIXTURE_PLUGIN_DIR);
    ASSERT_FALSE(ids.empty());
    EXPECT_EQ(ids.front(), NEURIPLO_DEFAULT_BACKEND);
    EXPECT_TRUE(contains(ids, kGoodId));
    EXPECT_TRUE(contains(ids, kScriptedId));
    EXPECT_FALSE(contains(ids, "FIXTURE_WRONG_ABI"));
    EXPECT_FALSE(contains(ids, "FIXTURE_INCOMPLETE"));
}

TEST(CApiWrapper, EngineMetadata) {
    const neuriplo::Engine engine(fixture_config(kGoodId, "ok"));
    EXPECT_EQ(engine.backend_id(), kGoodId);

    const std::vector<neuriplo::TensorInfo> inputs = engine.inputs();
    ASSERT_EQ(inputs.size(), 1u);
    EXPECT_EQ(inputs[0].name, "input");
    EXPECT_EQ(inputs[0].dtype, NEURIPLO_TENSOR_DTYPE_FLOAT32);
    EXPECT_EQ(inputs[0].shape, (std::vector<int64_t>{1, 4}));
    EXPECT_EQ(inputs[0].batch_size, 1u);

    const std::vector<neuriplo::TensorInfo> outputs = engine.outputs();
    ASSERT_EQ(outputs.size(), 1u);
    EXPECT_EQ(outputs[0].name, "output");
    EXPECT_EQ(outputs[0].dtype, NEURIPLO_TENSOR_DTYPE_FLOAT32);
    EXPECT_EQ(outputs[0].shape, (std::vector<int64_t>{1, 4}));
}

TEST(CApiWrapper, InferDoublesInput) {
    neuriplo::Engine engine(fixture_config(kGoodId, "ok"));
    const std::vector<float> in{1.0f, 2.0f, 3.0f, 4.0f};

    // Implicit InputView from a std::vector.
    const neuriplo::Result from_vector = engine.infer({in});
    expect_doubled(from_vector, in);

    // Explicit pointer + byte size.
    const neuriplo::Result from_pointer = engine.infer({neuriplo::InputView(in.data(), in.size() * sizeof(float))});
    expect_doubled(from_pointer, in);
}

TEST(CApiWrapper, EmptyOutputIsConforming) {
    neuriplo::Engine engine(fixture_config(kScriptedId, "out_empty"));
    const std::vector<float> in{1.0f, 2.0f, 3.0f, 4.0f};
    const neuriplo::Result result = engine.infer({in});
    ASSERT_EQ(result.size(), 1u);
    const neuriplo::TensorView view = result.output(0);
    EXPECT_EQ(view.element_count(), 0u);
    EXPECT_EQ(view.size_bytes(), 0u);
    EXPECT_EQ(view.shape(), (std::vector<int64_t>{0, 4}));
    EXPECT_NO_THROW((void)view.data_as<float>());
}

TEST(CApiWrapper, TypeMismatchThrows) {
    neuriplo::Engine engine(fixture_config(kGoodId, "ok"));
    const std::vector<float> in{1.0f, 2.0f, 3.0f, 4.0f};
    const neuriplo::Result result = engine.infer({in});
    ASSERT_EQ(result.size(), 1u);
    const neuriplo::TensorView view = result.output(0);
    expect_error(NEURIPLO_STATUS_INVALID_ARGUMENT, [&] { (void)view.data_as<int32_t>(); });
    expect_error(NEURIPLO_STATUS_INVALID_ARGUMENT, [&] { (void)view.data_as<int64_t>(); });
    expect_error(NEURIPLO_STATUS_INVALID_ARGUMENT, [&] { (void)view.data_as<uint8_t>(); });
    expect_error(NEURIPLO_STATUS_INVALID_ARGUMENT, [&] { (void)result.output(1); });
}

TEST(CApiWrapper, ErrorCarriesStatusAndMessage) {
    std::string message = expect_error(NEURIPLO_STATUS_BACKEND_NOT_FOUND,
                                       [] { neuriplo::Engine engine(fixture_config("NO_SUCH_BACKEND", "ok")); });
    EXPECT_TRUE(contains(message, "NO_SUCH_BACKEND")) << message;

    message = expect_error(NEURIPLO_STATUS_MODEL_LOAD,
                           [] { neuriplo::Engine engine(fixture_config(kScriptedId, "create_fail")); });
    EXPECT_TRUE(contains(message, "create_fail")) << message;

    neuriplo::Engine failing(fixture_config(kScriptedId, "infer_fail"));
    const std::vector<float> in{1.0f, 2.0f, 3.0f, 4.0f};
    message = expect_error(NEURIPLO_STATUS_INFERENCE, [&] { (void)failing.infer({in}); });
    EXPECT_TRUE(contains(message, "fixture infer failure")) << message;

    // check() is the same mapping, usable on raw C calls.
    EXPECT_NO_THROW(neuriplo::check(NEURIPLO_STATUS_OK));
    expect_error(NEURIPLO_STATUS_INVALID_ARGUMENT, [] { neuriplo::check(neuriplo_engine_create(nullptr, nullptr)); });
}

TEST(CApiWrapper, MoveSemantics) {
    const std::vector<float> in{1.0f, 2.0f, 3.0f, 4.0f};
    {
        neuriplo::Engine first(fixture_config(kGoodId, "ok"));
        ASSERT_NE(first.handle(), nullptr);
        neuriplo_engine_t* const raw = first.handle();

        neuriplo::Engine second(std::move(first));
        EXPECT_EQ(first.handle(), nullptr); // NOLINT(bugprone-use-after-move): testing the moved-from state
        EXPECT_EQ(second.handle(), raw);
        expect_error(NEURIPLO_STATUS_INVALID_ARGUMENT, [&] { (void)first.infer({in}); });

        neuriplo::Result result_a = second.infer({in});
        neuriplo::Result result_b(std::move(result_a));
        EXPECT_EQ(result_a.handle(), nullptr); // NOLINT(bugprone-use-after-move)
        expect_error(NEURIPLO_STATUS_INVALID_ARGUMENT, [&] { (void)result_a.size(); });
        expect_doubled(result_b, in);

        // Move-assignment releases what the target held, exactly once.
        neuriplo::Result result_c = second.infer({in});
        result_c = std::move(result_b);
        expect_doubled(result_c, in);
        EXPECT_EQ(fixture_counts("good").live, 1u);

        neuriplo::Engine third(fixture_config(kGoodId, "ok"));
        EXPECT_EQ(fixture_counts("good").live, 2u);
        third = std::move(second);
        EXPECT_EQ(fixture_counts("good").live, 1u);
        EXPECT_EQ(third.handle(), raw);
        expect_doubled(third.infer({in}), in);
    }
    const FixtureCounts counts = fixture_counts("good");
    EXPECT_EQ(counts.live, 0u);
    EXPECT_EQ(counts.handed, 3u);
    EXPECT_EQ(counts.released, counts.handed);
}

TEST(CApiWrapper, ResultOutlivesEngine) {
    const std::vector<float> in{5.0f, 6.0f, 7.0f, 8.0f};
    neuriplo::Result result(nullptr);
    {
        neuriplo::Engine engine(fixture_config(kGoodId, "ok"));
        result = engine.infer({in});
    }
    EXPECT_EQ(fixture_counts("good").live, 0u);
    expect_doubled(result, in);
}
