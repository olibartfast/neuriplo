// Backend-agnostic unit tests for the design-pattern scaffolding introduced by
// the refactor: the State machine, the ModelRunner Bridge, the BackendDecorator
// base, and the four concrete decorators. These tests use a hand-rolled
// FakeBackend (no real model, no vendor SDK) so they build and run under any
// DEFAULT_BACKEND. They depend only on gtest (no gmock), matching the project's
// no-new-dependency constraint.

#include "BackendDecorator.hpp"
#include "BackendState.hpp"
#include "HostTensorConverter.hpp"
#include "IAllocator.hpp"
#include "ITensorConverter.hpp"
#include "InferenceInterface.hpp"
#include "ModelRunner.hpp"
#include "decorators/CachingBackend.hpp"
#include "decorators/LoggingBackend.hpp"
#include "decorators/ProfilingBackend.hpp"
#include "decorators/QuantizedBackend.hpp"

#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

// Deterministic, dependency-free backend for exercising the patterns.
class FakeBackend : public InferenceInterface {
  public:
    FakeBackend() : InferenceInterface("fake_model", false, 1, {}) {
        // Canned output: one int32 element (7) and one float element (2.0).
        output_.push_back(static_cast<int32_t>(7));
        output_.push_back(2.0f);
    }

    std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
    get_infer_results(const std::vector<std::vector<uint8_t>>& input_tensors) override {
        (void)input_tensors;
        ++call_count_;
        if (throw_on_infer_) {
            throw InferenceExecutionException("fake failure");
        }
        std::vector<std::vector<TensorElement>> outputs;
        outputs.push_back(output_);
        std::vector<std::vector<int64_t>> shapes;
        shapes.push_back(std::vector<int64_t>{static_cast<int64_t>(output_.size())});
        return std::make_tuple(outputs, shapes);
    }

    void load() override {
        ++load_count_;
        state_ = load_should_fail_ ? BackendState::Failed : BackendState::Ready;
    }

    int call_count_ = 0;
    int load_count_ = 0;
    bool throw_on_infer_ = false;
    bool load_should_fail_ = false;
    std::vector<TensorElement> output_;
};

std::vector<std::vector<uint8_t>> make_input(uint8_t seed = 1) {
    std::vector<uint8_t> bytes;
    bytes.push_back(seed);
    bytes.push_back(static_cast<uint8_t>(seed + 1));
    bytes.push_back(static_cast<uint8_t>(seed + 2));
    std::vector<std::vector<uint8_t>> tensors;
    tensors.push_back(std::move(bytes));
    return tensors;
}

// ---------------------------------------------------------------------------
// State machine
// ---------------------------------------------------------------------------

TEST(BackendStateTest, AllowsOnlyDefinedTransitions) {
    EXPECT_TRUE(is_valid_transition(BackendState::Uninitialized, BackendState::Loading));
    EXPECT_FALSE(is_valid_transition(BackendState::Uninitialized, BackendState::Ready));
    EXPECT_FALSE(is_valid_transition(BackendState::Uninitialized, BackendState::Failed));

    EXPECT_TRUE(is_valid_transition(BackendState::Loading, BackendState::Ready));
    EXPECT_TRUE(is_valid_transition(BackendState::Loading, BackendState::Failed));
    EXPECT_FALSE(is_valid_transition(BackendState::Loading, BackendState::Uninitialized));

    EXPECT_TRUE(is_valid_transition(BackendState::Ready, BackendState::Failed));
    EXPECT_FALSE(is_valid_transition(BackendState::Ready, BackendState::Loading));

    EXPECT_TRUE(is_valid_transition(BackendState::Failed, BackendState::Loading));
    EXPECT_FALSE(is_valid_transition(BackendState::Failed, BackendState::Ready));
}

TEST(BackendStateTest, ToStringCoversEveryState) {
    EXPECT_STREQ(to_string(BackendState::Uninitialized), "Uninitialized");
    EXPECT_STREQ(to_string(BackendState::Loading), "Loading");
    EXPECT_STREQ(to_string(BackendState::Ready), "Ready");
    EXPECT_STREQ(to_string(BackendState::Failed), "Failed");
}

// ---------------------------------------------------------------------------
// Bridge: ModelRunner
// ---------------------------------------------------------------------------

TEST(ModelRunnerTest, RejectsNullBackend) { EXPECT_THROW(ModelRunner{nullptr}, InferenceException); }

TEST(ModelRunnerTest, LoadTransitionsToReadyAndIsIdempotent) {
    auto fake = std::make_unique<FakeBackend>();
    FakeBackend* raw = fake.get();
    ModelRunner runner(std::move(fake));

    EXPECT_EQ(runner.state(), BackendState::Uninitialized);
    runner.load();
    EXPECT_EQ(runner.state(), BackendState::Ready);
    EXPECT_EQ(raw->load_count_, 1);

    // Second load() is a no-op because the backend is already Ready.
    runner.load();
    EXPECT_EQ(raw->load_count_, 1);
}

TEST(ModelRunnerTest, RunAutoLoadsAndDelegatesOutput) {
    auto fake = std::make_unique<FakeBackend>();
    FakeBackend* raw = fake.get();
    ModelRunner runner(std::move(fake));

    auto [outputs, shapes] = runner.run(make_input());
    EXPECT_EQ(runner.state(), BackendState::Ready);
    EXPECT_EQ(raw->load_count_, 1);
    EXPECT_EQ(raw->call_count_, 1);
    ASSERT_EQ(outputs.size(), 1u);
    ASSERT_EQ(outputs[0].size(), 2u);
    EXPECT_EQ(std::get<int32_t>(outputs[0][0]), 7);
    EXPECT_FLOAT_EQ(std::get<float>(outputs[0][1]), 2.0f);
}

TEST(ModelRunnerTest, LoadThrowsWhenBackendNeverBecomesReady) {
    auto fake = std::make_unique<FakeBackend>();
    fake->load_should_fail_ = true;
    ModelRunner runner(std::move(fake));
    EXPECT_THROW(runner.load(), ModelLoadException);
}

TEST(ModelRunnerTest, LoadThrowsWhenBackendAlreadyFailed) {
    auto fake = std::make_unique<FakeBackend>();
    fake->load_should_fail_ = true;
    ModelRunner runner(std::move(fake));
    EXPECT_THROW(runner.load(), ModelLoadException);
    EXPECT_EQ(runner.state(), BackendState::Failed);
    EXPECT_THROW(runner.load(), ModelLoadException);
}

TEST(ModelRunnerTest, RunThrowsWhenBackendFailed) {
    auto fake = std::make_unique<FakeBackend>();
    fake->load_should_fail_ = true;
    ModelRunner runner(std::move(fake));
    EXPECT_THROW(runner.load(), ModelLoadException);
    EXPECT_EQ(runner.state(), BackendState::Failed);
    EXPECT_THROW(runner.run(make_input()), InferenceExecutionException);
}

// ---------------------------------------------------------------------------
// Decorator base
// ---------------------------------------------------------------------------

TEST(BackendDecoratorTest, RejectsNullInner) { EXPECT_THROW(BackendDecorator{nullptr}, InferenceException); }

TEST(BackendDecoratorTest, ForwardsModelMetadata) {
    auto fake = std::make_unique<FakeBackend>();
    ProfilingBackend deco(std::move(fake));
    EXPECT_EQ(deco.get_model_path(), "fake_model");
    EXPECT_FALSE(deco.is_gpu_available());
    EXPECT_EQ(deco.get_batch_size(), 1u);
}

// ---------------------------------------------------------------------------
// ProfilingBackend
// ---------------------------------------------------------------------------

TEST(ProfilingBackendTest, CountsInferencesAndForwardsOutput) {
    auto fake = std::make_unique<FakeBackend>();
    ProfilingBackend deco(std::move(fake));

    EXPECT_EQ(deco.get_total_inferences(), 0u);
    auto [outputs, shapes] = deco.get_infer_results(make_input());
    EXPECT_EQ(deco.get_total_inferences(), 1u);
    EXPECT_GE(deco.get_last_inference_time_ms(), 0.0);
    EXPECT_EQ(std::get<int32_t>(outputs[0][0]), 7);
}

TEST(ProfilingBackendTest, DoesNotCountFailedInference) {
    auto fake = std::make_unique<FakeBackend>();
    fake->throw_on_infer_ = true;
    ProfilingBackend deco(std::move(fake));

    EXPECT_THROW(deco.get_infer_results(make_input()), InferenceExecutionException);
    EXPECT_EQ(deco.get_total_inferences(), 0u);
}

// ---------------------------------------------------------------------------
// CachingBackend
// ---------------------------------------------------------------------------

TEST(CachingBackendTest, HitsAvoidInnerCallAndClearResets) {
    auto fake = std::make_unique<FakeBackend>();
    FakeBackend* raw = fake.get();
    CachingBackend deco(std::move(fake), 8);

    deco.get_infer_results(make_input(1));
    EXPECT_EQ(raw->call_count_, 1);

    // Same input -> cache hit, inner not invoked again.
    deco.get_infer_results(make_input(1));
    EXPECT_EQ(raw->call_count_, 1);

    // Different input -> miss.
    deco.get_infer_results(make_input(50));
    EXPECT_EQ(raw->call_count_, 2);

    // After clearing, the first input misses again.
    deco.clear_cache();
    deco.get_infer_results(make_input(1));
    EXPECT_EQ(raw->call_count_, 3);
}

TEST(CachingBackendTest, EvictsLeastRecentlyUsed) {
    auto fake = std::make_unique<FakeBackend>();
    FakeBackend* raw = fake.get();
    CachingBackend deco(std::move(fake), 1);

    deco.get_infer_results(make_input(1)); // miss -> count 1, caches A
    deco.get_infer_results(make_input(1)); // hit  -> count 1
    deco.get_infer_results(make_input(2)); // miss -> count 2, evicts A
    deco.get_infer_results(make_input(1)); // miss -> count 3 (A was evicted)
    EXPECT_EQ(raw->call_count_, 3);
}

// ---------------------------------------------------------------------------
// LoggingBackend
// ---------------------------------------------------------------------------

TEST(LoggingBackendTest, ForwardsOutputUnchanged) {
    auto fake = std::make_unique<FakeBackend>();
    LoggingBackend deco(std::move(fake));

    deco.load();
    EXPECT_EQ(deco.state(), BackendState::Ready);

    auto [outputs, shapes] = deco.get_infer_results(make_input());
    ASSERT_EQ(outputs.size(), 1u);
    EXPECT_EQ(std::get<int32_t>(outputs[0][0]), 7);
    EXPECT_FLOAT_EQ(std::get<float>(outputs[0][1]), 2.0f);
}

// ---------------------------------------------------------------------------
// QuantizedBackend
// ---------------------------------------------------------------------------

TEST(QuantizedBackendTest, DisabledIsExactPassthrough) {
    auto fake = std::make_unique<FakeBackend>();
    QuantizedBackend deco(std::move(fake)); // params.enabled defaults to false

    auto [outputs, shapes] = deco.get_infer_results(make_input());
    EXPECT_EQ(std::get<int32_t>(outputs[0][0]), 7);
    EXPECT_FLOAT_EQ(std::get<float>(outputs[0][1]), 2.0f);
}

TEST(QuantizedBackendTest, EnabledDequantizesIntegersOnly) {
    QuantizationParams params;
    params.enabled = true;
    params.scale = 2.0f;
    params.zero_point = 1;

    auto fake = std::make_unique<FakeBackend>();
    QuantizedBackend deco(std::move(fake), params);

    auto [outputs, shapes] = deco.get_infer_results(make_input());
    // int32 7 -> 2 * (7 - 1) = 12.0; float 2.0 left untouched.
    EXPECT_FLOAT_EQ(std::get<float>(outputs[0][0]), 12.0f);
    EXPECT_FLOAT_EQ(std::get<float>(outputs[0][1]), 2.0f);
}

// ---------------------------------------------------------------------------
// Abstract Factory products: HostAllocator
// ---------------------------------------------------------------------------

TEST(HostAllocatorTest, AllocatesUsableMemory) {
    HostAllocator alloc;
    EXPECT_STREQ(alloc.name(), "HostAllocator");

    void* p = alloc.allocate(64);
    ASSERT_NE(p, nullptr);
    std::memset(p, 0, 64); // must be writable
    alloc.deallocate(p);
}

// ---------------------------------------------------------------------------
// Abstract Factory products: HostTensorConverter
// ---------------------------------------------------------------------------

TEST(HostTensorConverterTest, ToTypedDecodesFloatBytes) {
    HostTensorConverter conv;
    EXPECT_STREQ(conv.name(), "HostTensorConverter");

    const float values[2] = {1.5f, -2.25f};
    std::vector<uint8_t> raw(sizeof(values));
    std::memcpy(raw.data(), values, sizeof(values));

    auto out = conv.to_typed(raw, TensorDataType::Float32);
    ASSERT_EQ(out.size(), 2u);
    EXPECT_FLOAT_EQ(std::get<float>(out[0]), 1.5f);
    EXPECT_FLOAT_EQ(std::get<float>(out[1]), -2.25f);
}

TEST(HostTensorConverterTest, FromBackendDecodesInt64) {
    HostTensorConverter conv;
    const int64_t values[3] = {7, -1, 1000};
    auto out = conv.from_backend(values, 3, TensorDataType::Int64);
    ASSERT_EQ(out.size(), 3u);
    EXPECT_EQ(std::get<int64_t>(out[0]), 7);
    EXPECT_EQ(std::get<int64_t>(out[1]), -1);
    EXPECT_EQ(std::get<int64_t>(out[2]), 1000);
}

TEST(HostTensorConverterTest, WidensInt8ToInt32) {
    HostTensorConverter conv;
    const int8_t values[2] = {-5, 5};
    auto out = conv.from_backend(values, 2, TensorDataType::Int8);
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(std::get<int32_t>(out[0]), -5);
    EXPECT_EQ(std::get<int32_t>(out[1]), 5);
}

TEST(HostTensorConverterTest, HandlesEmptyAndNull) {
    HostTensorConverter conv;
    EXPECT_TRUE(conv.to_typed({}, TensorDataType::Float32).empty());
    EXPECT_TRUE(conv.from_backend(nullptr, 4, TensorDataType::Int32).empty());
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
